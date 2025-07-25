import inspect
from typing import Optional

from diffusers import ModelMixin
from pytorch3d.datasets import BlenderCamera
from pytorch3d.renderer import PointsRasterizer, PointsRasterizationSettings
from pytorch3d.structures import Pointclouds
import torch
from torch import nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from torch import Tensor
from tqdm import tqdm
import math

from .TrDPNet import TrDPNet
from .feature_model import FeatureModel
from .model_utils import get_custom_betas, set_requires_grad


class MultiViewPointCloudDiffusionModel(ModelMixin):

    def __init__(
            self,
            beta_start: float,
            beta_end: float,
            beta_schedule: str,
            n_views: 3,
            image_size,
            scale_factor,
            n_points=16384,
            image_feature_model=None,
            w=2.0,
            dim: int = 384,
            depth: int = 12,
            mlp_dim: int = 1024,
            drop_rate: float = 0.1
    ):
        super().__init__()
        self.n_points = n_points
        self.w = w
        self.raster_point_radius: float = 0.0075  # point size
        self.raster_points_per_pixel: int = 1  # a single point per pixel, for now
        self.bin_size: int = 0
        self.image_size = image_size
        self.scale_factor: float = scale_factor
        self.n_views = n_views
        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {}
        if beta_schedule == 'custom':
            scheduler_kwargs.update(dict(trained_betas=get_custom_betas(beta_start=beta_start, beta_end=beta_end)))
        else:
            scheduler_kwargs.update(dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule))
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False),
            'pndm': PNDMScheduler(**scheduler_kwargs),
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference

        # Create point cloud model for processing point cloud at each diffusion step
        self.point_cloud_model = TrDPNet(dim=dim, depth=depth, mlp_dim=mlp_dim, drop_rate=drop_rate)

        self.feature_model = FeatureModel(model_name=image_feature_model)
        set_requires_grad(self.feature_model, False)

        self.raster_settings = PointsRasterizationSettings(
            image_size=(self.image_size, self.image_size),
            radius=self.raster_point_radius,
            points_per_pixel=self.raster_points_per_pixel,
            bin_size=self.bin_size,
        )

    def get_multi_view_feature(self, images):
        """
            return: [V, B, C, W, H]
        """
        images = torch.transpose(images, 0, 1)
        features = []
        for i, img in enumerate(images):  # [B, C, W, H]
            features.append(self.feature_model(img))
        return torch.cat(features, -1)

    def forward_train(
            self,
            x_0,
            images: Optional[Tensor],
            cameras,
    ):
        x_0 = x_0 * self.scale_factor
        B, N, D = x_0.shape

        # Sample random noise
        noise = torch.randn_like(x_0)

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,),
                                 device=self.device, dtype=torch.long)
        # noise_start = time.time()
        # Add noise to points
        x_t = self.scheduler.add_noise(x_0, noise, timestep)
        # noise_time = time.time() - noise_start
        img_feature = self.get_multi_view_feature(images)
        random_indices = torch.randperm(B)[:math.ceil(0.1 * B)]
        img_feature[random_indices, :, :] = 0
        # train_start = time.time()
        noise_pred = self.point_cloud_model(x_t, img_feature, timestep)

        # train_time = time.time() - train_start
        # Check
        if not noise_pred.shape == noise.shape:
            raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

        # Loss
        loss = F.mse_loss(noise_pred, noise)
        # return loss, noise_time, feature_time, projecting_time, train_time
        return loss

    def image_encode(self, x_t, cameras, img):
        return self.surface_projection(x_t, cameras[0], img)

    @torch.no_grad()
    def forward_sample(
            self,
            num_points,
            images,
            cameras,
            # Optional overrides
            scheduler: Optional[str] = 'ddpm',
            return_sample_every_n_steps: int = -1,
            # Inference parameters
            num_inference_steps: Optional[int] = 1000,
            eta: Optional[float] = 0.3,  # for DDIM
            # Whether to return all the intermediate steps in generation
            # Whether to disable tqdm
            disable_tqdm: bool = False,
    ):
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Get the size of the noise
        N = num_points
        B = 1 if images is None else images.shape[0]
        D = 3

        # Sample noise
        x_t = torch.randn(B, N, D, device=self.device)

        # Set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # Loop over timesteps
        scheduler.timesteps = scheduler.timesteps.to(self.device)
        progress_bar = tqdm(scheduler.timesteps.to(self.device))

        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)

        w = self.w
        img_feature = self.get_multi_view_feature(images)
        img_feature_z = torch.zeros_like(img_feature)
        c = torch.cat((img_feature, img_feature_z), 0)
        for i, t in enumerate(progress_bar):
            # Forward
            noise_pred = self.point_cloud_model(x_t.repeat(2, 1, 1), c, t.reshape(1).expand(B * 2))
            noise_pred_c, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred + w * (noise_pred_c - noise_pred)
            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample
            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                all_outputs.append(x_t)
        if return_all_outputs:
            all_outputs = torch.stack(all_outputs, dim=1)  # (B, sample_steps, N, D)
            all_outputs = [self.tensor_to_point_cloud(o, denormalize=True, unscale=True) for o in all_outputs]
        output = self.tensor_to_point_cloud(x_t, unscale=True)
        return (output, all_outputs) if return_all_outputs else output

    def build_cameras(self, data):
        Rs = torch.transpose(data['Rs'], 0, 1)
        Ts = torch.transpose(data['Ts'], 0, 1)
        K = torch.transpose(data['K'], 0, 1)
        V = data['images'].shape[1]
        cameras = []
        for i in range(V):
            cameras.append(BlenderCamera(Rs[i], Ts[i], K[i]).to(self.device))
        return cameras

    @torch.autocast('cuda', dtype=torch.float32)
    def surface_projection(
            self, points: Tensor, camera, images: Tensor,
    ):

        B, C, H, W = *images.shape,
        R = self.raster_settings.points_per_pixel
        N = points.shape[1]

        # Scale camera by scaling T. ASSUMES CAMERA IS LOOKING AT ORIGIN!
        camera = camera.clone()
        camera.T = camera.T * self.scale_factor

        # Create rasterizer
        rasterizer = PointsRasterizer(cameras=camera, raster_settings=self.raster_settings)

        # Associate points with features via rasterization
        fragments = rasterizer(Pointclouds(points))  # (B, H, W, R)
        fragments_idx: Tensor = fragments.idx.long()
        visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
        points_to_visible_pixels = fragments_idx[visible_pixels]

        # Reshape local features to (B, H, W, R, C)
        images = images.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R,
                                                                 -1)  # (B, H, W, R, C)

        # Get local features corresponding to visible points
        local_features_proj = torch.zeros(B * N, C, device=self.device)
        local_features_proj[points_to_visible_pixels] = images[visible_pixels]
        local_features_proj = local_features_proj.reshape(B, N, C)

        # import pdb; pdb.set_trace()
        return local_features_proj

    def tensor_to_point_cloud(self, x: Tensor, /, denormalize: bool = False, unscale: bool = False):
        points = x[:, :, :3] / (self.scale_factor if unscale else 1)
        assert x.shape[2] == 3
        return Pointclouds(points=points)

    def forward(self, data, mode: str = 'train', scheduler='ddpm', num_inference_steps=1000,
                return_sample_every_n_steps: int = -1):
        cameras = self.build_cameras(data)
        if mode == 'train':
            return self.forward_train(data['label'].to(self.device), data['images'].to(self.device), cameras)
        elif mode == 'sample':
            return self.forward_sample(self.n_points, data['images'].to(self.device), cameras, scheduler=scheduler,
                                       num_inference_steps=1000,
                                       return_sample_every_n_steps=return_sample_every_n_steps)
        else:
            raise NotImplementedError()
