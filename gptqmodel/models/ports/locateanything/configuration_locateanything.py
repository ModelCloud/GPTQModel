# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/locateanything/SPEC.md; see NOTICE.
"""Configuration classes for the package-owned LocateAnything port."""

from __future__ import annotations

from transformers.configuration_utils import PreTrainedConfig

from ..native.qwen2.configuration_qwen2 import Qwen2Config
from ..native.qwen3.configuration_qwen3 import Qwen3Config


_DECODER_CONFIGS = {
    "Qwen2ForCausalLM": Qwen2Config,
    "Qwen3ForCausalLM": Qwen3Config,
}
_ALLOWED_ATTENTION = (None, "eager", "sdpa")


def normalize_attn_implementation(value):
    """Map the checkpoint's ``magi`` backend to SDPA and reject anything but eager/SDPA."""
    if value == "magi":
        return "sdpa"
    if value not in _ALLOWED_ATTENTION:
        raise ValueError(f"LocateAnything supports eager or sdpa attention, got {value!r}")
    return value


def _as_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, PreTrainedConfig):
        return value.to_dict()
    return dict(value)


class MoonViTConfig(PreTrainedConfig):
    model_type = "moonvit"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size=(2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.merge_kernel_size = list(merge_kernel_size)


class LocateAnythingConfig(PreTrainedConfig):
    model_type = "locateanything"
    sub_configs = {"vision_config": MoonViTConfig, "text_config": Qwen2Config}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        use_backbone_lora: int = 0,
        use_llm_lora: int = 0,
        downsample_ratio: float = 0.5,
        template=None,
        loss_version: str = "v1",
        mlp_checkpoint: bool = False,
        image_token_index: int = 151667,
        box_start_token_id: int = 151668,
        box_end_token_id: int = 151669,
        coord_start_token_id: int = 151677,
        coord_end_token_id: int = 152677,
        ref_start_token_id: int = 151672,
        ref_end_token_id: int = 151673,
        none_token_id: int = 4064,
        **kwargs,
    ):
        requested = kwargs.pop("attn_implementation", None)
        fallback = kwargs.pop("_attn_implementation", None)
        if requested is None:
            requested = fallback
        attn_implementation = normalize_attn_implementation(requested)

        super().__init__(**kwargs)
        self._attn_implementation = attn_implementation or "sdpa"

        vision = _as_dict(vision_config)
        if vision.get("model_type", "moonvit") != "moonvit":
            raise ValueError("LocateAnything requires MoonViT vision_config")
        vision.pop("model_type", None)
        vision.pop("_attn_implementation", None)
        self.vision_config = MoonViTConfig(**vision)

        text = _as_dict(text_config)
        architectures = text.get("architectures") or ["Qwen2ForCausalLM"]
        architecture = architectures[0]
        text_config_class = _DECODER_CONFIGS.get(architecture)
        if text_config_class is None:
            raise ValueError(f"Unsupported LocateAnything decoder: {architecture}")
        text["architectures"] = [architecture]
        text.pop("_attn_implementation", None)
        text.pop("model_type", None)
        self.text_config = text_config_class(**text)

        self.vision_config._attn_implementation = self._attn_implementation
        self.text_config._attn_implementation = self._attn_implementation

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.loss_version = loss_version
        self.mlp_checkpoint = mlp_checkpoint
        self.image_token_index = image_token_index
        self.box_start_token_id = box_start_token_id
        self.box_end_token_id = box_end_token_id
        self.coord_start_token_id = coord_start_token_id
        self.coord_end_token_id = coord_end_token_id
        self.ref_start_token_id = ref_start_token_id
        self.ref_end_token_id = ref_end_token_id
        self.none_token_id = none_token_id
        self.tie_word_embeddings = self.text_config.tie_word_embeddings


__all__ = ["LocateAnythingConfig", "MoonViTConfig", "normalize_attn_implementation"]
