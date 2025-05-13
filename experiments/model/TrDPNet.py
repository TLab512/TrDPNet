import torch
from .pvcnn.pvcnn_utils import create_mlp_components, create_pointnet2_sa_components, create_pointnet2_fp_modules
from torch import nn
import numpy as np


if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops

        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Not that this should work just as well for
    continuous values as for discrete values.
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embed_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
    return emb


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).float()
        if ATTENTION_MODE == 'flash':
            # qkv = einops.rearrange(qkv, 'B N (K H D) -> K B H N D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H N D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            # x = einops.rearrange(x, 'B H N D -> B N (H D)')
            x = x.transpose(1, 2).flatten(-2)
        elif ATTENTION_MODE == 'xformers':
            qkv = qkv.transpose(2, 3)
            # qkv = einops.rearrange(qkv, 'B N (K H D) -> K B N H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B N H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            # x = einops.rearrange(x, 'B N H D -> B N (H D)', H=self.num_heads)
            x = x.flatten(-2)
        elif ATTENTION_MODE == 'math':
            # qkv = einops.rearrange(qkv, 'B N (K H D) -> K B H N D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H N D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, dim=768, mlp_dim=3072, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attention = Attention(dim=dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x, c):
        x = x + (self.attn(self.norm1(torch.cat((c, x), dim=1)))[:, c.shape[1]:, :])
        x = x + (self.cross_attention(self.norm2(torch.cat((x, c), dim=1)))[:, :x.shape[1], :])
        x = x + self.mlp(self.norm3(x))
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, dim=768, mlp_dim=3072, drop_rate=0.0, depth=4):
        super().__init__()
        self.depth = depth
        blocks = [TransformerBlock(dim, mlp_dim, drop_rate) for _ in range(depth)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, c):
        all_feat = []
        for block in self.blocks:
            x = block(x, c)
            all_feat.append(x)
        return x, all_feat


class TrDPNet(nn.Module):
    def __init__(self, dim=768, depth=12, mlp_dim=3072, drop_rate=0.1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.drop_rate = drop_rate
        self.pointnet_sa = PVCNN2SA(
            extra_feature_channels=0,
            dropout=0.1, width_multiplier=1,
            voxel_resolution_multiplier=1,
            dim=dim
        )
        self.pointnet_fp = PVCNN2FP(
            num_classes=3,
            sa_in_channels=self.pointnet_sa.sa_in_channels,
            channels_sa_features=self.pointnet_sa.channels_sa_features,
            extra_feature_channels=0,
            dropout=0.1, width_multiplier=1,
            voxel_resolution_multiplier=1
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.dim))
        self.encoder = TransformerEncoder(dim=self.dim, depth=self.depth, mlp_dim=self.mlp_dim, drop_rate=self.drop_rate)

    def forward(self, pts, image_feature, time_step):
        in_features_list, coords_list, group_input_tokens, center = self.pointnet_sa(pts)
        # divide the point clo  ud in the same form. This is important
        group_input_tokens = group_input_tokens.transpose(-1, -2)
        center = center.transpose(-1, -2)
        # time_step_encode
        time_embedding = get_timestep_embedding(self.dim, time_step, time_step.device)
        time_token = time_embedding.unsqueeze(1)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos_point = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos_point), dim=1)
        # condition
        condition = torch.cat((time_token, image_feature), dim=1)
        # transformer
        x, _ = self.encoder(x + pos, condition)
        return self.pointnet_fp(in_features_list, coords_list,
                                x[:, x.shape[1] - group_input_tokens.size(1):, :].transpose(-1, -2),
                                center.transpose(-1, -2))


class PVCNN2SA(nn.Module):

    def __init__(self, use_att=True, dropout=0.1, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1, dim=384):
        super().__init__()
        assert extra_feature_channels >= 0
        self.sa_blocks = [
            ((32, 2, 32), (512, 0.1, 32, (128, dim))),
        ]
        self.dropout = dropout
        self.width_multiplier = width_multiplier

        self.in_channels = extra_feature_channels + 3

        # Create PointNet-2 model
        sa_layers, self.sa_in_channels, self.channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks,
            extra_feature_channels=extra_feature_channels,
            with_se=True,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

    def forward(self, inputs):
        inputs = inputs.transpose(-1, -2)
        # Separate input coordinates and features
        coords = inputs[:, :3, :].contiguous()  # (B, 3, N)
        features = inputs  # (B, 3 + S, N)

        # Downscaling layers
        coords_list = []
        in_features_list = []
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))
        # Replace the input features
        in_features_list[0] = inputs[:, 3:, :].contiguous()
        return in_features_list, coords_list, features, coords


class PVCNN2FP(nn.Module):
    fp_blocks = [
        ((128,), (32, 2, 32)),
    ]

    def __init__(self, num_classes, sa_in_channels, channels_sa_features, use_att=True, dropout=0.1,
                 extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1, ):
        super().__init__()
        assert extra_feature_channels >= 0
        self.dropout = dropout
        self.width_multiplier = width_multiplier

        # Only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        # Create MLP layers
        self.channels_fp_features = channels_fp_features
        layers, _ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, num_classes],  # was 0.5
            classifier=True,
            dim=2,
            width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)

    def forward(self, in_features_list, coords_list, features, coords):
        # Upscaling layers
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks(
                (  # this is a tuple because of nn.Sequential
                    coords_list[-1 - fp_idx],  # reverse coords list from above
                    coords,  # original point coordinates
                    features,  # keep concatenating upsampled features
                    in_features_list[-1 - fp_idx],  # reverse features list from above
                )
            )
        # Output MLP layers
        output = self.classifier(features)
        return output.transpose(1, 2)

