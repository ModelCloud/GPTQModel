# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""ZDTaichu5 configuration adapted from the pinned TaichuAI/ZDTaichu5.0-9B release."""

from copy import deepcopy

from transformers import PreTrainedConfig, Qwen3_5TextConfig

from .cradio_config import RADIOConfig


class ZDTaichu5_0_Config(PreTrainedConfig):
    model_type = "zdtaichu5_0"
    sub_configs = {"llm_config": Qwen3_5TextConfig, "vision_config": RADIOConfig}

    def __init__(
        self, vision_config=None, llm_config=None, force_image_size=512,
        downsample_ratio=0.5, template=None, ps_version="v2", image_tag_type="internvl",
        projector_hidden_size=20480, vit_hidden_size=1280,
        img_context_token_id=248056, video_context_token_id=248057,
        attn_implementation=None, **kwargs,
    ):
        if llm_config is None:
            llm_config = {
                "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 32,
                "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
                "vocab_size": 248320, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                                     "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                                     "mrope_section": [11, 11, 10]},
            }
        if isinstance(llm_config, dict):
            llm_config = deepcopy(llm_config)
            if llm_config.pop("model_type", "qwen3_5_text") != "qwen3_5_text":
                raise ValueError("ZDTaichu5 requires a native qwen3_5_text decoder")
            llm_config = Qwen3_5TextConfig(**llm_config)
        if type(llm_config) is not Qwen3_5TextConfig:
            raise TypeError("ZDTaichu5 requires Qwen3_5TextConfig")
        if isinstance(vision_config, dict):
            vision_config = deepcopy(vision_config)
            if vision_config.pop("model_type", "radio") != "radio":
                raise ValueError("ZDTaichu5 requires the released C-RADIO vision layout")
            vision_config = RADIOConfig(**vision_config)
        self.vision_config = vision_config or RADIOConfig()
        if type(self.vision_config) is not RADIOConfig:
            raise TypeError("ZDTaichu5 requires package-owned RADIOConfig")
        self.llm_config = llm_config
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            kwargs.setdefault(name, getattr(llm_config, name, None))
        kwargs.setdefault("tie_word_embeddings", llm_config.tie_word_embeddings)
        super().__init__(**kwargs)
        self.force_image_size = force_image_size or 512
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.ps_version = ps_version
        self.image_tag_type = image_tag_type
        self.projector_hidden_size = projector_hidden_size
        self.vit_hidden_size = vit_hidden_size
        self.img_context_token_id = img_context_token_id
        self.video_context_token_id = video_context_token_id
        if downsample_ratio <= 0:
            raise ValueError("downsample_ratio must be the reciprocal of a positive integer")
        factor = round(1 / downsample_ratio)
        if factor < 1 or factor * downsample_ratio != 1:
            raise ValueError("downsample_ratio must be the reciprocal of a positive integer")
        if ps_version not in ("v1", "v2"):
            raise ValueError("ps_version must be v1 or v2")
        if self.force_image_size % (self.vision_config.patch_size * factor):
            raise ValueError("image size must be divisible by patch_size / downsample_ratio")
        if attn_implementation is not None:
            self._attn_implementation = attn_implementation

    def get_text_config(self, decoder=False, **kwargs):
        return self.llm_config

    @property
    def text_config(self):
        return self.llm_config
