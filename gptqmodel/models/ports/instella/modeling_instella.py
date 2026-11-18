# Modifications copyright(c) 2025 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: LicenseRef-Instella-Research-Only-RAIL-MS
# This code was adapted from OLMo2 (Transformers v4.47.1) by AMD.
# Modified for Transformers 5 by ModelCloud.ai. See LICENSE and NOTICES.
"""Instella's pre-norm decoder and full-projection Q/K RMSNorm, on Transformers 5."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ..native.llama.modeling_llama import LlamaRMSNorm as InstellaRMSNorm
from ..native.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from ..native.qwen2.modeling_qwen2 import Qwen2MLP as InstellaMLP
from .configuration_instella import InstellaConfig


class InstellaRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        rope = config.rope_parameters
        self.base = rope["rope_theta"]
        self.max_position_embeddings = config.max_position_embeddings
        self.scaling_type = rope["rope_type"]
        self.scaling_factor = rope.get("factor", 1.0)
        self.register_buffer("inv_freq", self.initial_frequencies(), persistent=False)

    def initial_frequencies(self, device=None):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Retain AMD's linear and dynamic NTK equations and per-layer buffer.
        inv_freq = self.inv_freq
        if self.scaling_type == "linear":
            position_ids = position_ids.float() / self.scaling_factor
        elif self.scaling_type == "dynamic":
            seq_len = torch.max(position_ids) + 1
            if seq_len > self.max_position_embeddings:
                base = self.base * (
                    self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)
                ) ** (self.dim / (self.dim - 2))
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim)
                )
                # Publish a new tensor, then use this call's snapshot. A second
                # request updating the module must not replace our frequencies.
                self.inv_freq = inv_freq
        inv_freq = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ position_ids[:, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """AMD's eager equations (scores divided by sqrt(head_dim))."""
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    weights = query @ key.transpose(2, 3) / math.sqrt(module.head_dim)
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, : key.shape[-2]]
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    return (weights @ value).transpose(1, 2).contiguous(), weights


class InstellaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.q_norm = InstellaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.k_norm = InstellaRMSNorm(self.num_key_value_heads * self.head_dim, config.rms_norm_eps)
        self.rotary_emb = InstellaRotaryEmbedding(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        bsz, q_len, hidden_size = hidden_states.shape
        # Q/K normalization is over the complete projected width BEFORE splitting heads.
        query = self.q_norm(self.q_proj(hidden_states))
        key = self.k_norm(self.k_proj(hidden_states))
        value = self.v_proj(hidden_states)
        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(value, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        output, weights = attention_interface(
            self, query, key, value, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0, scaling=self.scaling, **kwargs,
        )
        output = output.contiguous().reshape(bsz, q_len, hidden_size)
        return self.o_proj(output), weights


class InstellaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = InstellaAttention(config, layer_idx)
        self.mlp = InstellaMLP(config)
        self.pre_attention_layernorm = InstellaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = InstellaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        attn, _ = self.self_attn(self.pre_attention_layernorm(hidden_states), **kwargs)
        hidden_states = residual + attn
        return hidden_states + self.mlp(self.pre_feedforward_layernorm(hidden_states))


class InstellaPreTrainedModel(PreTrainedModel):
    config_class = InstellaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InstellaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": InstellaDecoderLayer, "attentions": InstellaAttention}

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, InstellaRotaryEmbedding):
            # HF materializes non-persistent meta buffers with empty storage on reload.
            # Rebuild the frequencies, rather than accepting uninitialized RoPE.
            module.inv_freq.copy_(module.initial_frequencies(module.inv_freq.device))


class InstellaModel(InstellaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([InstellaDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = InstellaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, use_cache=None, **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError("Instella requires a Transformers Cache")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen).unsqueeze(0)
        mask = create_causal_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=mask, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=use_cache, **kwargs,
            )
        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states), past_key_values=past_key_values if use_cache else None,
        )


class InstellaForCausalLM(InstellaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = InstellaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
        )
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
