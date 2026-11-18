# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
# Modified by ModelCloud.ai for GPT-QModel: Transformers 5 APIs, masks and meta-load buffer initialization.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT and Qwen implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT and Qwen used by the Meta AI and Qwen team that trained the model.
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
"""PyTorch Dream model (bidirectional Qwen2-style diffusion decoder)."""

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, MaskedLMOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple, logging
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ..native.qwen2.modeling_qwen2 import (
    Qwen2MLP as DreamMLP,
)
from ..native.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm as DreamRMSNorm,
)
from ..native.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from .configuration_dream import DreamConfig
from .generation_utils import DreamGenerationConfig, DreamGenerationMixin


logger = logging.get_logger(__name__)


class DreamRotaryEmbedding(Qwen2RotaryEmbedding):
    """Native Qwen2 RoPE; the name keeps TF5 rotary buffer initialization."""


def dream_eager_attention_forward(module, query, key, value, attention_mask, dropout=0.0, **kwargs):
    """Pinned Dream eager attention: bool masks are keep masks and fully masked rows become zero."""
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(module.head_dim)
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        if attention_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
    return attn_output, attn_weights


class DreamAttention(nn.Module):
    """Bidirectional (non-causal) multi-headed attention with Qwen2 projections."""

    def __init__(self, config: DreamConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        # Upstream fallback for callers that replay a layer with position_ids only.
        self.rotary_emb = DreamRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            position_embeddings = self.rotary_emb(value_states, position_ids)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, dream_eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=False,
            **kwargs,
        )
        attn_output = self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())
        return attn_output, attn_weights


class DreamDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DreamConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DreamAttention(config, layer_idx)
        self.mlp = DreamMLP(config)
        self.input_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DreamPreTrainedModel(PreTrainedModel):
    config_class = DreamConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DreamDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    generation_config_class = DreamGenerationConfig
    _supports_sdpa = True
    _can_record_outputs = {"hidden_states": DreamDecoderLayer, "attentions": DreamAttention}
    # Weight and RoPE buffer init is the native PreTrainedModel._init_weights.


class DreamBaseModel(DreamPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DreamDecoderLayer`]

    Args:
        config: DreamConfig
    """

    def __init__(self, config: DreamConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DreamDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DreamRotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[Union[torch.Tensor, str]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError("Dream past_key_values must be a Transformers Cache")
        # Dream attention is bidirectional, so the causal masking_utils builders do not apply.
        if isinstance(attention_mask, str):
            if attention_mask != "full":
                raise ValueError("Dream only accepts the 'full' string attention mask")
            attention_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                # Public HF padding masks are keep masks, never additive masks.
                attention_mask = attention_mask[:, None, None, :].bool()
            elif attention_mask.ndim != 4:
                raise ValueError("Dream attention_mask must have rank 2 or 4")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=past_key_values if use_cache else None,
        )


class DreamModel(DreamGenerationMixin, DreamPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = DreamBaseModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[Union[torch.Tensor, str]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MaskedLMOutput:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.lm_head(slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep))

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
