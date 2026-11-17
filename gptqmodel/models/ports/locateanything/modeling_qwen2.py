# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
# Adapted by ModelCloud for Transformers 5. The pinned source repository's
# accompanying NVIDIA LICENSE also applies; see NOTICE and README.md.
"""LocateAnything's block-diffusion decoder on Transformers 5 Qwen2 layers.

Native layers preserve the Q/K/V bias, RoPE, GQA, RMSNorm and SwiGLU equations.
The package owns visibility, image injection and cache lifecycle: native causal
Qwen2 alone does not implement this architecture.
"""

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as NativeQwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as NativeQwen2Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as NativeQwen3ForCausalLM

from .mask_sdpa_utils import (
    create_block_diff_mask_by_pe_4d,
    find_prefix_seq_length_by_pe,
    update_causal_mask_for_one_gen_window_2d,
    update_causal_mask_with_pad_non_visible_2d,
)


def inject_images(model, input_ids, visual_features, image_token_index, inputs_embeds=None):
    if inputs_embeds is None:
        if input_ids is None:
            raise ValueError("input_ids or inputs_embeds is required")
        inputs_embeds = model.get_input_embeddings()(input_ids)
    if visual_features is not None:
        if input_ids is None:
            raise ValueError("Visual injection requires input_ids")
        selected = input_ids == image_token_index
        features = visual_features.reshape(-1, inputs_embeds.shape[-1])
        if int(selected.sum()) != features.shape[0]:
            raise ValueError("Image placeholder count does not match visual features")
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[selected] = features.to(inputs_embeds)
    return inputs_embeds


class Qwen2Model(NativeQwen2Model):
    def block_mask(self, input_ids, inputs_embeds, position_ids, attention_mask, past_length, use_cache):
        batch, length = inputs_embeds.shape[:2]
        key_length = past_length + length
        if attention_mask is not None and attention_mask.ndim == 4:
            return attention_mask.to(inputs_embeds)
        if self.training:
            prefix = find_prefix_seq_length_by_pe(position_ids).to(inputs_embeds.device)
            return create_block_diff_mask_by_pe_4d(
                getattr(self.config, "block_size", 6), prefix, position_ids,
                getattr(self.config, "causal_attn", False),
            )[0].to(inputs_embeds)
        rows = torch.arange(length, device=inputs_embeds.device)[:, None] + past_length
        cols = torch.arange(key_length, device=inputs_embeds.device)[None, :]
        allowed = cols <= rows
        window = getattr(self.config, "sliding_window", None)
        if window is not None:
            allowed = allowed & (cols > rows - window)
        mask = torch.zeros((batch, 1, length, key_length), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        mask.masked_fill_(~allowed, float("-inf"))
        if attention_mask is not None:
            if attention_mask.shape != (batch, key_length):
                raise ValueError("attention_mask must include the cached prefix")
            mask.masked_fill_(~attention_mask[:, None, None, :].bool(), float("-inf"))
        # Upstream selects AR/MTP for the batch from its first row's final ID.
        mask_id = getattr(self.config, "text_mask_token_id", 151676)
        if length > 1 and input_ids is not None and input_ids[0, -1].item() == mask_id:
            for b in range(batch):
                if use_cache:
                    update_causal_mask_for_one_gen_window_2d(
                        input_ids[b], mask[b, 0], getattr(self.config, "block_size", 6),
                        use_cache=True, causal_attn=getattr(self.config, "causal_attn", False),
                    )
                else:
                    update_causal_mask_with_pad_non_visible_2d(
                        input_ids[b], mask[b, 0], mask_id, getattr(self.config, "block_size", 6),
                        causal_attn=getattr(self.config, "causal_attn", False),
                    )
        return mask

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, visual_features=None, image_token_index=None, use_cache=None,
                output_hidden_states=None, output_attentions=None, return_dict=True, **kwargs):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise ValueError("LocateAnything supports Transformers DynamicCache only")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()  # full prefix; never silently evict MTP context
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify input_ids or inputs_embeds, not both")
        hidden_states = inject_images(self, input_ids, visual_features, image_token_index, inputs_embeds)
        length = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(past_length, past_length + length, device=hidden_states.device)[None, :]
        position_ids = position_ids.expand(hidden_states.shape[0], -1)
        mask = self.block_mask(input_ids, hidden_states, position_ids, attention_mask, past_length, use_cache)
        positions = self.rotary_emb(hidden_states, position_ids)
        all_hidden = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # Explicit collection also works with ordinary layer capture/replay hooks.
        for layer in self.layers:
            if output_hidden_states:
                all_hidden += (hidden_states,)
            if output_attentions:
                residual = hidden_states
                values, weights = layer.self_attn(
                    hidden_states=layer.input_layernorm(hidden_states), attention_mask=mask,
                    position_ids=position_ids, position_embeddings=positions,
                    past_key_values=past_key_values, use_cache=use_cache, **kwargs,
                )
                hidden_states = residual + values
                hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
                all_attentions += (weights,)
            else:
                hidden_states = layer(hidden_states, attention_mask=mask, position_ids=position_ids,
                                      position_embeddings=positions, past_key_values=past_key_values,
                                      use_cache=use_cache, **kwargs)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden += (hidden_states,)
        result = BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                        past_key_values=past_key_values if use_cache else None,
                                        hidden_states=all_hidden, attentions=all_attentions)
        return result if return_dict else result.to_tuple()


class Qwen2ForCausalLM(NativeQwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.post_init()


    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        if hasattr(result, "logits"):
            result.logits = result.logits.float()
        return result

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      inputs_embeds=None, **kwargs):
        # Also used directly by the localization sampler; independent of the
        # GenerationMixin iteration bookkeeping introduced in Transformers 5.
        if past_key_values is not None:
            seen = past_key_values.get_seq_length()
            if input_ids.shape[1] > seen:
                input_ids = input_ids[:, seen:]
        positions = kwargs.pop("position_ids", None)
        if positions is None and attention_mask is not None:
            positions = attention_mask.long().cumsum(-1) - 1
            positions.masked_fill_(attention_mask == 0, 1)
        if positions is not None:
            positions = positions[:, -input_ids.shape[1]:]
        return {"input_ids": input_ids, "past_key_values": past_key_values,
                    "attention_mask": attention_mask, "position_ids": positions,
                    "use_cache": kwargs.get("use_cache", True)}


class Qwen3ForCausalLM(NativeQwen3ForCausalLM):
    """Upstream's native Qwen3 variant with the same image injection entry point."""

    prepare_inputs_for_generation = Qwen2ForCausalLM.prepare_inputs_for_generation

    def forward(self, input_ids=None, visual_features=None, image_token_index=None, inputs_embeds=None, **kwargs):
        if visual_features is not None:
            inputs_embeds = inject_images(self, input_ids, visual_features, image_token_index, inputs_embeds)
            input_ids = None
        return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
