# Copyright (c) 2023-2024 DeepSeek. Distributed under LICENSE-CODE (MIT).
from typing import Tuple

from transformers import PreTrainedConfig

from .configuration_deepseek import DeepseekV2Config


PretrainedConfig = PreTrainedConfig


class VisionEncoderConfig(PretrainedConfig):
    model_type: str = "vision"

    model_name: str = "siglip_large_patch16_384"
    image_size: int = 384
    patch_size: int = 16
    width: int = 1024
    layers: int = 24
    heads: int = 16
    mlp_ratio: int = 4
    global_pool: str = "map"
    ignore_head: bool = True
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False
    weight_init: str = "skip"
    deterministic: bool = False
    num_recomputing_layers: int = 0

    def __init__(
        self,
        model_name: str = "siglip_large_patch16_384",
        image_size: int = 384,
        patch_size: int = 16,
        width: int = 1024,
        layers: int = 24,
        heads: int = 16,
        mlp_ratio: int = 4,
        global_pool: str = "map",
        ignore_head: bool = True,
        class_token: bool = False,
        num_classes: int = 0,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.global_pool = global_pool
        self.ignore_head = ignore_head
        self.class_token = class_token
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        super().__init__(**kwargs)


class MlpProjectorConfig(PretrainedConfig):
    model_type = "mlp_projector"
    projector_type: str = "downsample_mlp_gelu"
    input_dim: int = 1152
    n_embed: int = 2048
    depth: int = 2
    mlp_ratio: int = 1
    downsample_ratio: int = 2
    token_pooling: bool = False

    def __init__(
        self,
        projector_type: str = "downsample_mlp_gelu",
        input_dim: int = 1152,
        n_embed: int = 2048,
        depth: int = 2,
        mlp_ratio: int = 1,
        downsample_ratio: int = 2,
        **kwargs,
    ):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.downsample_ratio = downsample_ratio

        super().__init__(**kwargs)


class DeepseekVLV2Config(PretrainedConfig):
    model_type = "deepseek_vl_v2"
    sub_configs = {
        "vision_config": VisionEncoderConfig,
        "projector_config": MlpProjectorConfig,
        "language_config": DeepseekV2Config,
    }
    vision_config: VisionEncoderConfig
    projector_config: MlpProjectorConfig
    language_config: DeepseekV2Config

    tile_tag: str = "2D"
    global_view_pos: str = "head"
    candidate_resolutions: Tuple[Tuple[int, int]] = ((384, 384),)

    def get_text_config(self, decoder=False, encoder=False):
        return self.language_config

    def __init__(
        self,
        tile_tag: str = "2D",
        global_view_pos: str = "head",
        candidate_resolutions: Tuple[Tuple[int, int]] = ((384, 384),),
        **kwargs,
    ):
        vision_config = kwargs.pop("vision_config", {})
        self.vision_config = (
            vision_config if isinstance(vision_config, VisionEncoderConfig) else VisionEncoderConfig(**vision_config)
        )

        projector_config = kwargs.pop("projector_config", {})
        self.projector_config = (
            projector_config
            if isinstance(projector_config, MlpProjectorConfig)
            else MlpProjectorConfig(**projector_config)
        )

        language_config = kwargs.pop("language_config", {})
        if isinstance(language_config, DeepseekV2Config):
            self.language_config = language_config
        else:
            self.language_config = DeepseekV2Config(**language_config)

        super().__init__(**kwargs)
        self.use_cache = self.language_config.use_cache
        self.vocab_size = self.language_config.vocab_size
        self.tie_word_embeddings = self.language_config.tie_word_embeddings
        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos
        self.candidate_resolutions = candidate_resolutions
