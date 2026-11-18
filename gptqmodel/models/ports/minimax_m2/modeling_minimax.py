# Copyright 2025 MiniMaxAI and HuggingFace Inc. teams. All rights reserved.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Source-matched partial RoPE on the native Transformers 5 Text-01 hybrid."""

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.minimax.modeling_minimax import (
    MiniMaxAttention,
    MiniMaxCache,
    MiniMaxDecoderLayer,
    MiniMaxForCausalLM,
    MiniMaxLightningAttention,
    MiniMaxModel,
    MiniMaxPreTrainedModel,
    MiniMaxRMSNorm,
    MiniMaxRotaryEmbedding,
    eager_attention_forward,
)

from .configuration_minimax import MiniMaxText01Config


class MiniMaxText01Cache(MiniMaxCache):
    """MiniMax cache whose sequence length comes from the softmax-attention KV layers.

    Lightning-attention layers keep only a recurrent state, and layer zero is usually one of them, so the
    length is read from the first initialized KV layer instead of layer zero. A cache without any
    softmax-attention layer tracks no positions (none are consumed).
    """

    def get_seq_length(self, layer_idx=0):
        for layer in self.layers:
            if layer.is_initialized:
                return layer.get_seq_length()
        return 0

    def get_mask_sizes(self, query_length, layer_idx):
        return self.get_seq_length() + query_length, 0

    def get_linear_cache(self, layer_idx):
        return self.linear_cache[layer_idx] if layer_idx < len(self.linear_cache) else None

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        self.linear_cache = [
            value.index_select(0, beam_idx.to(value.device)) if isinstance(value, torch.Tensor) else value
            for value in self.linear_cache
        ]

    def batch_select_indices(self, indices):
        self.reorder_cache(cast(torch.LongTensor, indices))

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            getattr(layer, "batch_repeat_interleave")(repeats)
        self.linear_cache = [
            value.repeat_interleave(repeats, dim=0) if isinstance(value, torch.Tensor) else value
            for value in self.linear_cache
        ]

    def reset(self):
        super().reset()
        self.linear_cache = []


class MiniMaxText01RotaryEmbedding(MiniMaxRotaryEmbedding):
    @staticmethod
    def compute_default_rope_parameters(config, **kwargs):
        dim = config.rotary_dim
        theta = (config.rope_parameters or {}).get("rope_theta")
        if theta is None:
            raise ValueError("MiniMax requires rope_theta")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        # The released implementation applies FP32 trigonometry before casting Q/K.
        return super().forward(x.float(), position_ids)


class MiniMaxText01LightningAttention(MiniMaxLightningAttention):
    def __init__(self, config, layer_idx):
        self.config = config
        super().__init__(config, layer_idx)
        self._non_persistent_buffers_set.update(("slope_rate", "query_decay", "key_decay", "diagonal_decay"))


def _partial_rotate(states, cos, sin):
    dim = cos.shape[-1]
    rotary, passthrough = states[..., :dim], states[..., dim:]
    half = dim // 2
    rotated = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
    result = rotary * cos[:, None] + rotated * sin[:, None]
    return torch.cat((result, passthrough), dim=-1).to(states.dtype)


class MiniMaxText01Attention(MiniMaxAttention):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = _partial_rotate(query, cos, sin), _partial_rotate(key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        attention = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation or "eager", eager_attention_forward
        )
        output, weights = attention(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.config.sliding_window,
            **kwargs,
        )
        return self.o_proj(output.reshape(*input_shape, -1).contiguous()), weights


class MiniMaxText01SharedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MiniMaxText01DecoderLayer(MiniMaxDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.postnorm = getattr(config, "postnorm", True)
        self.shared_moe = getattr(config, "shared_intermediate_size", 0) > 0
        if self.shared_moe:
            self.shared_mlp = MiniMaxText01SharedMLP(config)
            self.coefficient = nn.Linear(config.hidden_size, 1, bias=False)
        attention_cls = (
            MiniMaxText01Attention if self.block_type == "full_attention" else MiniMaxText01LightningAttention
        )
        self.self_attn = attention_cls(config, layer_idx)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        **kwargs: Any,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.postnorm:
            residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual * self.attn_alpha_factor + hidden_states * self.attn_beta_factor
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.postnorm:
            residual = hidden_states
        routed = self.mlp(hidden_states)
        if self.shared_moe:
            coefficient = (
                F.linear(hidden_states.float(), self.coefficient.weight.float()).sigmoid().to(hidden_states.dtype)
            )
            routed = routed * (1 - coefficient) + self.shared_mlp(hidden_states) * coefficient
        return residual * self.mlp_alpha_factor + routed * self.mlp_beta_factor


class MiniMaxText01Model(MiniMaxModel):
    def __init__(self, config):
        MiniMaxPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniMaxText01DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = MiniMaxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MiniMaxText01RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs: Any,
    ):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(past_key_values, MiniMaxText01Cache):
            raise ValueError("Original MiniMax requires MiniMaxText01Cache, not a foreign or M2 cache")
        if use_cache and past_key_values is None:
            past_key_values = MiniMaxText01Cache()
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds")
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, MiniMaxText01RotaryEmbedding):
            with torch.device(module.inv_freq.device):
                inv_freq, _ = module.compute_default_rope_parameters(module.config)
            init.copy_(module.inv_freq, inv_freq)
            init.copy_(module.original_inv_freq, inv_freq)
        else:
            super()._init_weights(module)


class MiniMaxText01ForCausalLM(MiniMaxForCausalLM):
    config_class = MiniMaxText01Config

    @classmethod
    def is_custom_code(cls):
        # Opt this exact-layout native adapter into Transformers' MiniMax weight conversions.
        return False

    def __init__(self, config):
        if not isinstance(config, MiniMaxText01Config):
            implementation = config._attn_implementation
            config = MiniMaxText01Config(**config.to_dict())
            config._attn_implementation = implementation
        MiniMaxPreTrainedModel.__init__(self, config)
        self.model = MiniMaxText01Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()
