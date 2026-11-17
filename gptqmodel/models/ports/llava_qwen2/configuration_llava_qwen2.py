# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Flat Qwen2 configuration with the original LLaVA multimedia metadata."""

from transformers import Qwen2Config


_DEFAULT_VISION_TOWER = object()


class LlavaConfig(Qwen2Config):
    model_type = "llava_qwen2"

    def __init__(self, mm_vision_tower=_DEFAULT_VISION_TOWER, mm_hidden_size=3072,
                 mm_projector_type="mlp2x_gelu", mm_patch_merge_type="flat",
                 mm_vision_select_layer=-2, mm_vision_select_feature="patch",
                 vision_config=None, tokenizer_padding_side="right", **kwargs):
        alias = kwargs.pop("vision_tower", "mobileclip_l_1024")
        if mm_vision_tower is _DEFAULT_VISION_TOWER:
            mm_vision_tower = alias
        super().__init__(**kwargs)
        self.mm_vision_tower = mm_vision_tower
        self.mm_hidden_size = mm_hidden_size
        self.mm_projector_type = mm_projector_type
        self.mm_patch_merge_type = mm_patch_merge_type
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.vision_config = vision_config
        self.tokenizer_padding_side = tokenizer_padding_side
        if tokenizer_padding_side not in ("left", "right"):
            raise ValueError("tokenizer_padding_side must be left or right")
        if mm_patch_merge_type not in ("flat", "spatial", "spatial_unpad"):
            raise ValueError(f"Unsupported mm_patch_merge_type: {mm_patch_merge_type}")
