# SPDX-FileCopyrightText: 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: MIT
"""OpenVLA configuration, adapted from github.com/openvla/openvla@c8f03f48af692657d3060c19588038c7220e9af9."""

from typing import Dict, List, Optional

from transformers import PreTrainedConfig

from ..native.llama.configuration_llama import LlamaConfig
from ..native.mistral.configuration_mistral import MistralConfig
from ..native.phi.configuration_phi import PhiConfig


# === Utilities for Mapping Prismatic names to HF names ===
# fmt: off
VISION_BACKBONE_TO_RESOLUTION: Dict[str, List[int]] = {
    "clip-vit-l": [224], "siglip-vit-so400m": [224], "dinov2-vit-l": [224], "in1k-vit-l": [224],

    "clip-vit-l-336px": [336],
    "siglip-vit-so400m-384px": [384],

    "dinoclip-vit-l-336px": [336, 336],
    "dinosiglip-vit-so-224px": [224, 224],
    "dinosiglip-vit-so-384px": [384, 384],
}
VISION_BACKBONE_TO_TIMM_ID: Dict[str, List[str]] = {
    "clip-vit-l": ["vit_large_patch14_clip_224.openai"],
    "clip-vit-l-336px": ["vit_large_patch14_clip_336.openai"],

    "dinov2-vit-l": ["vit_large_patch14_reg4_dinov2.lvd142m"],
    "in1k-vit-l": ["vit_large_patch16_224.augreg_in21k_ft_in1k"],

    "siglip-vit-so400m": ["vit_so400m_patch14_siglip_224"],
    "siglip-vit-so400m-384px": ["vit_so400m_patch14_siglip_384"],

    "dinoclip-vit-l-336px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_large_patch14_clip_336.openai"],
    "dinosiglip-vit-so-224px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
    "dinosiglip-vit-so-384px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_384"],
}
TIMM_OVERRIDE_ACT_LAYER: Dict[str, List[Optional[str]]] = {
    "clip-vit-l": ["quick_gelu"], "clip-vit-l-336px": ["quick_gelu"],
    "dinov2-vit-l": [None], "in1k-vit-l": [None],
    "siglip-vit-so400m": [None], "siglip-vit-so400m-384px": [None],
    "dinoclip-vit-l-336px": [None, "quick_gelu"],
    "dinosiglip-vit-so-224px": [None, None], "dinosiglip-vit-so-384px": [None, None]
}

LLM_BACKBONE_TO_HF_PATH = {
    "llama2-7b-pure": "meta-llama/Llama-2-7b-hf", "llama2-13b-pure": "meta-llama/Llama-2-13b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf", "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",

    "vicuna-v15-7b": "lmsys/vicuna-7b-v1.5", "vicuna-v15-13b": "lmsys/vicuna-13b-v1.5",

    "mistral-v0.1-7b-pure": "mistralai/Mistral-7B-v0.1",
    "mistral-v0.1-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",

    "phi-2-3b": "microsoft/phi-2",
}
LLM_BACKBONE_TO_HF_METACLASS = {
    "llama2-7b-pure": "llama", "llama2-13b-pure": "llama", "llama2-7b-chat": "llama", "llama2-13b-chat": "llama",
    "vicuna-v15-7b": "llama", "vicuna-v15-13b": "llama",

    "mistral-v0.1-7b-pure": "mistral", "mistral-v0.1-7b-instruct": "mistral",

    "phi-2-3b": "phi",
}

VALID_VISION_BACKBONES = set(VISION_BACKBONE_TO_RESOLUTION.keys())
VALID_LLM_BACKBONES = set(LLM_BACKBONE_TO_HF_PATH)
# fmt: on



_TEXT_CONFIGS = {"llama": LlamaConfig, "mistral": MistralConfig, "phi": PhiConfig}


class PrismaticConfig(PreTrainedConfig):
    model_type = "prismatic"
    sub_configs = {"text_config": PreTrainedConfig}

    def __init__(
        self,
        vision_backbone_id="siglip-vit-so400m",
        llm_backbone_id="vicuna-v15-7b",
        arch_specifier="no-align+gelu-mlp",
        use_fused_vision_backbone=None,
        image_resize_strategy="letterbox",
        text_config=None,
        llm_max_length=2048,
        pad_token_id=32000,
        pad_to_multiple_of=64,
        output_projector_states=False,
        image_sizes=None,
        vision_backbone_kwargs=None,
        **kwargs,
    ):
        if vision_backbone_id not in VALID_VISION_BACKBONES:
            raise ValueError(f"Unsupported OpenVLA vision backbone: {vision_backbone_id}")
        if llm_backbone_id not in VALID_LLM_BACKBONES:
            raise ValueError(f"Unsupported OpenVLA language backbone: {llm_backbone_id}")
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states
        fused = vision_backbone_id.startswith(("dinoclip", "dinosiglip"))
        if use_fused_vision_backbone is not None and bool(use_fused_vision_backbone) != fused:
            raise ValueError("OpenVLA fused-vision flag disagrees with the selected backbone")
        self.use_fused_vision_backbone = fused
        self.timm_model_ids = list(VISION_BACKBONE_TO_TIMM_ID[vision_backbone_id])
        self.timm_override_act_layers = list(TIMM_OVERRIDE_ACT_LAYER[vision_backbone_id])
        self.image_sizes = list(image_sizes or VISION_BACKBONE_TO_RESOLUTION[vision_backbone_id])
        if len(self.image_sizes) != len(self.timm_model_ids) or any(s <= 0 for s in self.image_sizes):
            raise ValueError("OpenVLA requires one positive image size per vision backbone")
        if fused and len(set(self.image_sizes)) != 1:
            raise ValueError("Channel-stacked fused OpenVLA images require equal resolutions")
        self.vision_backbone_kwargs = vision_backbone_kwargs or [{} for _ in self.timm_model_ids]
        if len(self.vision_backbone_kwargs) != len(self.timm_model_ids):
            raise ValueError("OpenVLA requires one vision geometry per backbone")
        for geometry in self.vision_backbone_kwargs:
            if set(geometry) - {"embed_dim", "depth", "num_heads", "mlp_ratio"}:
                raise ValueError("Only explicit OpenVLA vision geometry overrides are supported")
            if geometry.get("depth", 2) < 2:
                raise ValueError("OpenVLA extracts the second-to-last vision block and requires depth >= 2")
        self.image_resize_strategy = image_resize_strategy
        self.hf_llm_id = LLM_BACKBONE_TO_HF_PATH[llm_backbone_id]
        self.llm_max_length = llm_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        expected_type = LLM_BACKBONE_TO_HF_METACLASS[llm_backbone_id]
        if isinstance(text_config, PreTrainedConfig):
            text_config = text_config.to_dict()
        values = dict(text_config or {})
        if values.pop("model_type", expected_type) != expected_type:
            raise ValueError("OpenVLA text_config does not match its language backbone")
        # Fixed native classes, never checkpoint AutoConfig/auto_map resolution.
        self.text_config = _TEXT_CONFIGS[expected_type](**values)
        for key in ("timm_model_ids", "timm_override_act_layers", "hf_llm_id"):
            supplied = kwargs.pop(key, getattr(self, key))
            if supplied != getattr(self, key):
                raise ValueError(f"OpenVLA {key} disagrees with its selected backbone")
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class OpenVLAConfig(PrismaticConfig):
    model_type = "openvla"

    def __init__(self, norm_stats=None, n_action_bins=256, **kwargs):
        if not isinstance(n_action_bins, int) or n_action_bins < 2:
            raise ValueError("OpenVLA requires at least two action-bin edges")
        self.norm_stats = norm_stats
        self.n_action_bins = n_action_bins
        super().__init__(**kwargs)
