# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0
"""Native computation with temporal-pooling ownership and native-format saving."""

from typing import Any

import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.kimi_k25.configuration_kimi_k25 import Kimi_K25Config
from transformers.models.kimi_k25.modeling_kimi_k25 import (
    Kimi_K25ForConditionalGeneration as NativeKimiForConditionalGeneration,
)
from transformers.models.kimi_k25.modeling_kimi_k25 import Kimi_K25Model as NativeKimiModel
from transformers.models.kimi_k25.modeling_kimi_k25 import (
    Kimi_K25MultimodalProjection,
    Kimi_K25PreTrainedModel,
)
from transformers.models.kimi_k25.modeling_kimi_k25 import Kimi_K25VisionModel as NativeKimiVisionModel
from transformers.utils.generic import can_return_tuple
from transformers.utils.output_capturing import capture_outputs
from transformers.vision_utils import get_vision_attention_seqlens, get_vision_position_ids

from .processing_kimi_k25 import Kimi_K25Processor


class Kimi_K25VisionModel(NativeKimiVisionModel):
    @capture_outputs
    def forward(self, pixel_values, grid_thw, **kwargs: Any) -> Any:
        hidden_states = self.patch_embed(pixel_values, grid_thw=grid_thw, **kwargs)
        position_ids = get_vision_position_ids(grid_thw, spatial_merge_size=1, kwargs=kwargs)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(
            grid_thw, self.config, merge_temporal=True, kwargs=kwargs,
        )
        for block in self.layers:
            hidden_states = block(
                hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                position_embeddings=position_embeddings, **kwargs,
            )
        hidden_states = self.final_layernorm(hidden_states)
        # Native ragged pooling supports a partial final video chunk; rectangular gather does not.
        pooled_hidden_states = self.temporal_patch_merger(hidden_states, grid_thw)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=pooled_hidden_states)


class Kimi_K25Model(NativeKimiModel):
    def __init__(self, config):
        Kimi_K25PreTrainedModel.__init__(self, config)
        self.vision_tower = Kimi_K25VisionModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config, trust_remote_code=False)
        self.mm_projector = Kimi_K25MultimodalProjection(config)
        self.post_init()

    @classmethod
    def is_custom_code(cls):
        # Native checkpoint transforms still own every parameter and layout.
        return False

    @can_return_tuple
    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        if image_grid_thw is None:
            raise ValueError("Kimi-K2.5 requires image_grid_thw for vision feature splitting.")
        vision_outputs = self.vision_tower(pixel_values, grid_thw=image_grid_thw, **kwargs)
        image_embeds = self.mm_projector(vision_outputs.pooler_output).squeeze(1)
        merge_area = self.vision_tower.merge_kernel_size[0] * self.vision_tower.merge_kernel_size[1]
        # The vision tower pools time already: each clip owns only its spatial grid.
        split_sizes = (image_grid_thw[:, 1:].prod(-1) // merge_area).tolist()
        vision_outputs.pooler_output = torch.split(image_embeds, split_sizes)
        return vision_outputs


class Kimi_K25ForConditionalGeneration(NativeKimiForConditionalGeneration):
    def __init__(self, config):
        Kimi_K25PreTrainedModel.__init__(self, config)
        self.model = Kimi_K25Model(config)
        text_config = self.model.language_model.config
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def is_custom_code(cls):
        return False

    def save_pretrained(self, *args, save_original_format=False, **kwargs):
        # Native reverse QKV conversion also matches dense text q_proj in 5.23.
        return super().save_pretrained(*args, save_original_format=save_original_format, **kwargs)


__all__ = ["Kimi_K25Config", "Kimi_K25ForConditionalGeneration", "Kimi_K25Processor"]
