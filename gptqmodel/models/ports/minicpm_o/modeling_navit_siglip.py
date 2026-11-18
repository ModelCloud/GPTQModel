# coding=utf-8
# Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Siglip model."""
# Copied from  HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit and add tgt_sizes
# Modified by ModelCloud.ai, 2026: the config, encoder, attention and MLP are the
# native Transformers 5 SigLIP modules (identical parameter names); only the
# NaViT variable-resolution position bucketing remains local.

from typing import Optional

import torch
from torch import nn
from transformers import SiglipVisionConfig
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.siglip.modeling_siglip import SiglipEncoder, SiglipPreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings as NativeSiglipVisionEmbeddings
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs


__all__ = ["SiglipVisionConfig", "SiglipVisionEmbeddings", "SiglipVisionTransformer"]


class SiglipVisionEmbeddings(NativeSiglipVisionEmbeddings):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        self.num_patches_per_side = self.image_size // self.patch_size

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        patch_attention_mask: torch.BoolTensor,
        tgt_sizes: Optional[torch.IntTensor] = None,
    ) -> torch.Tensor:
        batch_size = pixel_values.size(0)

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = (
            max_im_h // self.patch_size,
            max_im_w // self.patch_size,
        )
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class SiglipVisionTransformer(SiglipPreTrainedModel):
    config: SiglipVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.patch_embedding

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        tgt_sizes: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_attention_mask = torch.ones(
                size=(
                    batch_size,
                    pixel_values.size(2) // self.config.patch_size,
                    pixel_values.size(3) // self.config.patch_size,
                ),
                dtype=torch.bool,
                device=pixel_values.device,
            )

        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            tgt_sizes=tgt_sizes,
        )

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask.view(batch_size, -1),
        )
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=attention_mask, **kwargs)
        return BaseModelOutputWithPooling(last_hidden_state=self.post_layernorm(encoder_outputs.last_hidden_state))
