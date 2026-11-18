# Copyright (c) 2024 OpenGVLab
# SPDX-License-Identifier: MIT
"""Legacy chat config with explicit, package-owned decoder selection."""

import math
from copy import deepcopy

from transformers.configuration_utils import PretrainedConfig

from ..native.llama.configuration_llama import LlamaConfig
from ..native.phi3.configuration_phi3 import Phi3Config
from ..native.qwen2.configuration_qwen2 import Qwen2Config
from ..native.qwen3.configuration_qwen3 import Qwen3Config
from ..native.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from .configuration_intern_vit import InternVisionConfig


def decoder_config_class(architecture):
    classes = {
        "LlamaForCausalLM": LlamaConfig,
        "Qwen2ForCausalLM": Qwen2Config,
        "Qwen3ForCausalLM": Qwen3Config,
        "Qwen3MoeForCausalLM": Qwen3MoeConfig,
        "Phi3ForCausalLM": Phi3Config,
    }
    if architecture == "InternLM2ForCausalLM":
        from ..internlm2.configuration_internlm2 import InternLM2Config

        return InternLM2Config
    if architecture not in classes:
        raise ValueError(f"Unsupported InternVL chat decoder: {architecture!r}")
    return classes[architecture]


class InternVLChatConfig(PretrainedConfig):
    model_type = "internvl_chat"
    is_composition = True
    sub_configs = {"vision_config": InternVisionConfig, "llm_config": PretrainedConfig}

    def __init__(self, vision_config=None, llm_config=None, use_backbone_lora=0,
                 use_llm_lora=0, select_layer=-1, force_image_size=None,
                 downsample_ratio=0.5, template="internvl2_5", dynamic_image_size=False,
                 use_thumbnail=False, ps_version="v1", min_dynamic_patch=1,
                 max_dynamic_patch=6, **kwargs):
        vision_config = vision_config or {}
        self.vision_config = (InternVisionConfig(**deepcopy(vision_config))
                              if isinstance(vision_config, dict) else vision_config)
        llm_config = deepcopy(llm_config) if llm_config is not None else {
            "architectures": ["Qwen2ForCausalLM"]}
        raw = llm_config if isinstance(llm_config, dict) else llm_config.to_dict()
        architectures = raw.get("architectures")
        if not isinstance(architectures, list) or len(architectures) != 1:
            raise ValueError("InternVL llm_config must declare exactly one architecture")
        config_class = decoder_config_class(architectures[0])
        if raw.get("model_type", config_class.model_type) != config_class.model_type:
            raise ValueError("InternVL decoder architecture/model_type mismatch")
        # Preserve both OpenGVLab Phi3 multipliers using native LongRoPE's
        # explicit attention_factor. Released configs carry the checkpoint field
        # `rope_scaling`; it is converted to `rope_parameters` here, at parse time.
        legacy_rope = raw.get("rope_scaling")
        if architectures[0] == "Phi3ForCausalLM" and legacy_rope:
            kind = legacy_rope.get("type", legacy_rope.get("rope_type"))
            if kind in ("su", "yarn"):
                scale = raw.get("max_position_embeddings", 4096) / raw.get("original_max_position_embeddings", 4096)
                factor = 1.0 if scale <= 1 else (
                    0.1 * math.log(scale) + 1 if kind == "yarn" else
                    math.sqrt(1 + math.log(scale) / math.log(raw.get("original_max_position_embeddings", 4096))))
                raw.pop("rope_scaling")
                raw["rope_parameters"] = {**legacy_rope, "rope_type": "longrope", "attention_factor": factor}
                raw["rope_parameters"].pop("type", None)
        self.llm_config = config_class(**raw)
        if ps_version not in ("v1", "v2"):
            raise ValueError("ps_version must be v1 or v2")
        if downsample_ratio <= 0 or downsample_ratio > 1 or not (1 / downsample_ratio).is_integer():
            raise ValueError("downsample_ratio must be the reciprocal of a positive integer")
        image_size = force_image_size or self.vision_config.image_size
        factor = int(1 / downsample_ratio)
        if image_size % (self.vision_config.patch_size * factor):
            raise ValueError("Image patch grid must be divisible by pixel shuffle factor")
        if not -self.vision_config.num_hidden_layers - 1 <= select_layer <= self.vision_config.num_hidden_layers:
            raise ValueError("select_layer is outside the vision hidden states")
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=self.tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder=False):
        return self.llm_config
