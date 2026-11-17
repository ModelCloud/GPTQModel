# SPDX-FileCopyrightText: 2024 OpenBMB
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Flattened checkpoint-compatible MiniCPM-V config with explicit decoder policy."""

from copy import deepcopy

from transformers import LlamaConfig, PretrainedConfig, Qwen2Config, Qwen3Config, SiglipVisionConfig


class MiniCPMVSliceConfig(PretrainedConfig):
    model_type = "minicpmv"

    def __init__(self, patch_size=14, max_slice_nums=9, scale_resolution=448, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution


class MiniCPMVConfig(PretrainedConfig):
    model_type = "minicpmv"
    has_no_defaults_at_init = True
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"vision_config": SiglipVisionConfig, "slice_config": MiniCPMVSliceConfig}

    def __init__(
        self, version=None, query_num=64, image_size=448, vision_config=None,
        slice_config=None, drop_vision_last_layer=True, batch_vision_input=True,
        vision_batch_size=16, batch_3d_resampler=True, use_image_id=None,
        slice_mode=None, vision_encoder=None, **kwargs,
    ):
        # Original and v2 checkpoints predate the public `version` field.
        if version is None and vision_encoder is not None and "scale_depth" in kwargs:
            version = 2.0 if slice_mode else 1.0
        if version is None:
            raise ValueError("MiniCPM-V requires a supported version or original MiniCPM/vision_encoder metadata")
        version = float(version)
        if version not in (1.0, 2.0, 2.5, 2.6, 4.0, 4.5):
            raise ValueError(f"Unqualified MiniCPM-V version {version}; 4.6 uses its separate native definition")
        self.version = version
        self.query_num = query_num
        self.image_size = image_size
        self.drop_vision_last_layer = drop_vision_last_layer
        self.batch_vision_input = batch_vision_input
        self.vision_batch_size = vision_batch_size
        self.batch_3d_resampler = batch_3d_resampler
        self.use_image_id = version >= 2.6 if use_image_id is None else use_image_id
        self.slice_mode = version >= 2.0 if slice_mode is None else slice_mode
        self.vision_encoder = vision_encoder or "vit_so400m_patch14_siglip_384.webli"
        self.slice_config = (slice_config if isinstance(slice_config, MiniCPMVSliceConfig)
                             else MiniCPMVSliceConfig(**(slice_config or {"max_slice_nums": 1})))
        defaults = {"hidden_size": 1152, "image_size": 980, "intermediate_size": 4304,
                        "num_attention_heads": 16, "num_hidden_layers": 27, "patch_size": 14}
        self.vision_config = (vision_config if isinstance(vision_config, SiglipVisionConfig)
                              else SiglipVisionConfig(**(vision_config or defaults)))
        self.patch_size = self.vision_config.patch_size if version >= 2.5 else kwargs.pop("patch_size", 14)
        if query_num <= 0 or vision_batch_size <= 0:
            raise ValueError("query_num and vision_batch_size must be positive")
        if version < 2.5:
            kwargs.setdefault("scale_emb", 1.0)
            kwargs.setdefault("scale_depth", 1.0)
            kwargs.setdefault("dim_model_base", 1)
        # Expose native decoder fields at the original flattened checkpoint paths.
        native_class = Qwen3Config if version == 4.5 else Qwen2Config if version == 2.6 else LlamaConfig
        native = native_class(**deepcopy(kwargs))
        super().__init__(**native.to_dict())
        self.model_type = "minicpmv"
        self._attn_implementation = native._attn_implementation

    def decoder_config(self):
        payload = self.to_dict()
        for name in ("vision_config", "slice_config", "model_type", "auto_map", "architectures",
                     "quantization_config", "transformers_version"):
            payload.pop(name, None)
        cls = Qwen3Config if self.version == 4.5 else Qwen2Config if self.version == 2.6 else LlamaConfig
        result = cls(**payload)
        result._attn_implementation = self._attn_implementation
        return result
