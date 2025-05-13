from config.structured import MultiViewConfig

from .model_gen import MultiViewPointCloudDiffusionModel


def get_model(cfg: MultiViewConfig):
    multi_view_model = MultiViewPointCloudDiffusionModel(**cfg.model)
    return multi_view_model

