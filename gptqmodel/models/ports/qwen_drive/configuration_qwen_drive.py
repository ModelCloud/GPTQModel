# Copyright 2026 Alibaba Group Holding Limited
# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Released Qwen-Drive metadata with a fixed, native Qwen3.5 VLM binding.

Adapted from QwenLM/Qwen-Drive-1.0 at
28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0. Planner geometry is metadata only.
"""

from copy import deepcopy

from transformers.configuration_utils import PreTrainedConfig, SubConfigSpec

from ..native.qwen3_5.configuration_qwen3_5 import Qwen3_5Config


class QwenDriveConfig(PreTrainedConfig):
    model_type = "qwen_drive"
    sub_configs_defaults = {"vlm_config": SubConfigSpec(Qwen3_5Config)}

    def __init__(
        self,
        vlm_config=None,
        expert_config=None,
        num_future_points=50,
        num_history_points=16,
        trajectory_point_dim=3,
        trajectory_hz=10.0,
        trajectory_scale=(165.0, 25.0, 1.5703125),
        num_inference_steps=10,
        noise_init_std=1.0,
        noise_seed=42,
        min_one_minus_t=0.1,
        history_image_pixels=174080,
        current_image_pixels=921600,
        image_patch_size=16,
        image_temporal_patch_size=2,
        image_spatial_merge_size=2,
        max_reasoning_tokens=256,
        min_reasoning_tokens=10,
        **kwargs,
    ):
        if isinstance(vlm_config, dict):
            vlm_config = deepcopy(vlm_config)
            if vlm_config.pop("model_type", "qwen3_5") != "qwen3_5":
                raise ValueError("Qwen-Drive supports only the released qwen3_5 VLM layout")
            for name, expected in (("text_config", "qwen3_5_text"), ("vision_config", "qwen3_5_vision")):
                nested = vlm_config.get(name)
                if isinstance(nested, dict) and nested.get("model_type", expected) != expected:
                    raise ValueError(f"Qwen-Drive {name} must use {expected}")
            vlm_config = Qwen3_5Config(**vlm_config)
        elif vlm_config is None:
            vlm_config = Qwen3_5Config()
        if type(vlm_config) is not Qwen3_5Config:
            raise ValueError("Qwen-Drive requires the native Qwen3_5Config")
        self.vlm_config = vlm_config
        self.expert_config = deepcopy(expert_config)
        self.num_future_points = num_future_points
        self.num_history_points = num_history_points
        self.trajectory_point_dim = trajectory_point_dim
        self.trajectory_hz = trajectory_hz
        self.trajectory_scale = list(trajectory_scale)
        self.num_inference_steps = num_inference_steps
        self.noise_init_std = noise_init_std
        self.noise_seed = noise_seed
        self.min_one_minus_t = min_one_minus_t
        self.history_image_pixels = history_image_pixels
        self.current_image_pixels = current_image_pixels
        self.image_patch_size = image_patch_size
        self.image_temporal_patch_size = image_temporal_patch_size
        self.image_spatial_merge_size = image_spatial_merge_size
        self.max_reasoning_tokens = max_reasoning_tokens
        self.min_reasoning_tokens = min_reasoning_tokens
        super().__init__(**kwargs)
        self.tie_word_embeddings = vlm_config.tie_word_embeddings

    def get_text_config(self, decoder=None, encoder=None):
        return self.vlm_config.get_text_config(decoder=decoder, encoder=encoder)

    @property
    def full_attention_layers(self):
        return [i for i, kind in enumerate(self.get_text_config().layer_types) if kind == "full_attention"]

    @property
    def history_query_points(self):
        return self.num_history_points - 1
