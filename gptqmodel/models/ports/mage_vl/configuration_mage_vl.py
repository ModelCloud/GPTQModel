# SPDX-FileCopyrightText: 2026 Microsoft
# SPDX-License-Identifier: Apache-2.0
# Adapted from microsoft/Mage-VL@d88b153285f1633a61b2f693c59c8576693af185.
from huggingface_hub.dataclasses import strict
from transformers import CONFIG_MAPPING, AutoConfig
from transformers.configuration_utils import PreTrainedConfig, classproperty


@strict
class MageVLVisionConfig(PreTrainedConfig):
    model_type = "mage_vl_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 448
    patch_size: int = 14
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    layer_norm_type: str = "layer_norm"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    rope_theta: float = 10000.0
    use_head: bool = False
    out_hidden_size: int = 1024
    spatial_merge_size: int = 2
    tokens_per_second: int = 1
    frame_windows_size: int = 4
    use_patch_position_encoding: bool = False
    patch_position_encoding_type: str = "absolute"
    max_position_embeddings: int = 8192


@strict
class MageVLConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MageVLModel`]. It is used to instantiate a
    MageVLModel model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a Mage-VL configuration.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3Config`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `MageVLVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The token index to denote start of vision input.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The token index to denote end of vision input.
    """

    model_type = "mage_vl"
    # `text_config` is resolved dynamically based on its `model_type` (defaults to `qwen3`),
    # so we use `AutoConfig` here as a placeholder; `__post_init__` swaps it for the
    # concrete config class via `CONFIG_MAPPING`.
    @classproperty
    def sub_configs(cls):
        return {"vision_config": MageVLVisionConfig, "text_config": AutoConfig}

    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False
    # Generation-related token ids are mirrored from `text_config` in `__post_init__`
    # so downstream tools (e.g. `generate`, vLLM) that read them at the top level keep working.
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        # Resolve vision_config
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        # Resolve text_config dynamically via CONFIG_MAPPING (defaults to qwen3)
        if isinstance(self.text_config, dict):
            text_model_type = self.text_config.get("model_type", "qwen3")
            text_config_cls = CONFIG_MAPPING[text_model_type]
            self.text_config = text_config_cls(**self.text_config)
        elif self.text_config is None:
            text_config_cls = CONFIG_MAPPING["qwen3"]
            self.text_config = text_config_cls()

        # Mirror generation-related token ids from text_config to the top level so
        # downstream tools (e.g. `generate`, chat templates, vLLM) that read them
        # from the top-level config keep working.
        for tok_key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            text_val = getattr(self.text_config, tok_key, None)
            if text_val is not None and getattr(self, tok_key, None) is None:
                setattr(self, tok_key, text_val)

        super().__post_init__(**kwargs)


__all__ = ["MageVLConfig", "MageVLVisionConfig"]
