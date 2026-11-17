# Copyright 2023 Baichuan Inc. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0; see the repository LICENSE.
"""Baichuan fused-MHA decoder adapted for Transformers 5 Cache/GenerationMixin.

The four upstream variants and numerical differences are recorded in
PROVENANCE.md. No checkpoint Python, optional CUDA imports, global registration,
mutable position caches, or first-forward weight normalization are used.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_baichuan import BaichuanConfig


def _get_interleave(n):
    """Upstream ALiBi slope order, including non-power-of-two head counts."""
    if math.log2(n).is_integer():
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]
    closest = 2 ** math.floor(math.log2(n))
    return _get_interleave(closest) + _get_interleave(2 * closest)[0::2][:n - closest]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.fp32_rope = config.normalize_lm_head
        self.max_position_embeddings = config.max_position_embeddings
        # Only Baichuan 1 owns this checkpoint tensor. Baichuan 2's frequencies
        # were unregistered runtime state; derive them on the input device.
        if not self.fp32_rope:
            self.register_buffer("inv_freq", self.frequencies())

    def frequencies(self, device=None):
        return 1.0 / (10000 ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))

    def forward(self, query, key, position_ids, seq_len):
        # Baichuan 1 builds its initial table in FP32 before a model.half()
        # cast. Its extension path instead uses inv_freq's current dtype.
        # Reproduce both without publishing a growing, shared tensor cache.
        initial_table = self.fp32_rope or seq_len <= self.max_position_embeddings
        inv_freq = self.frequencies(query.device) if initial_table else self.inv_freq.to(query.device)
        angles = position_ids.to(inv_freq.dtype).unsqueeze(-1) * inv_freq
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        cos, sin = angles.cos(), angles.sin()
        dtype = query.dtype
        if self.fp32_rope:
            query, key = query.float(), key.float()
        else:
            cos, sin = cos.to(dtype), sin.to(dtype)

        def rotate(x):
            left, right = x.chunk(2, dim=-1)
            return torch.cat((-right, left), dim=-1)

        return ((query * cos + rotate(query) * sin).to(dtype),
                (key * cos + rotate(key) * sin).to(dtype))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.W_pack = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        if config.position_embedding_type == "rope":
            self.rotary_emb = RotaryEmbedding(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        batch, length, width = hidden_states.shape
        # Checkpoint rows are contiguous Q, K, V, each head-major (not QKV
        # interleaved per head, and not a Llama split-projection checkpoint).
        q, k, v = [x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
                   for x in self.W_pack(hidden_states).chunk(3, dim=-1)]
        if hasattr(self, "rotary_emb"):
            past_length = past_key_value.get_seq_length(self.layer_idx) if past_key_value is not None else 0
            q, k = self.rotary_emb(q, k, position_ids, seq_len=past_length + length)
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = (scores + attention_mask).clamp_min(torch.finfo(scores.dtype).min)
        # Original 7B computes softmax in FP32; 13B retains activation dtype.
        probs = F.softmax(scores, dim=-1, dtype=(torch.float32 if hasattr(self, "rotary_emb") else None)).to(q.dtype)
        output = torch.matmul(probs, v).transpose(1, 2).reshape(batch, length, width)
        return self.o_proj(output), probs if output_attentions else None


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        residual = hidden_states
        attention, weights = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache,
        )
        hidden_states = residual + attention
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, weights


class BaichuanPreTrainedModel(PreTrainedModel):
    config_class = BaichuanConfig
    base_model_prefix = "model"
    _no_split_modules = ["DecoderLayer"]
    _supports_sdpa = False
    _supports_flash_attn = False

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, RotaryEmbedding) and hasattr(module, "inv_freq"):
            with torch.no_grad():
                module.inv_freq.copy_(module.frequencies(module.inv_freq.device))


class BaichuanModel(BaichuanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError("Baichuan Transformers 5 expects a Cache, not legacy tuple past_key_values")
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise TypeError("Baichuan currently supports DynamicCache only")
        if not use_cache and past_key_values is not None:
            raise ValueError("past_key_values requires use_cache=True")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length, _ = hidden_states.shape
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        total = past + length
        positions = torch.arange(past, total, device=hidden_states.device)
        if position_ids is None:
            position_ids = positions.unsqueeze(0)
        if position_ids.shape[-1] != length:
            raise ValueError("position_ids must describe the current query tokens")
        # Validate context before publishing any cache entries: malformed
        # attention input can be corrected and retried with the same Cache.
        allowed = torch.arange(total, device=hidden_states.device)[None, :] <= positions[:, None]
        allowed = allowed[None, None, :, :].expand(batch, 1, length, total)
        if attention_mask is not None:
            if attention_mask.ndim == 2 and attention_mask.shape == (batch, total):
                allowed = allowed & attention_mask[:, None, None, :].bool()
                if self.config.position_embedding_type == "alibi":
                    # 13B supports packed document IDs as well as 0/1 masks.
                    query_mask = attention_mask[:, past:]
                    allowed = allowed & (query_mask[:, None, :, None] > 0)
                    allowed = allowed & (query_mask[:, None, :, None] == attention_mask[:, None, None, :])
            elif attention_mask.ndim == 3 and attention_mask.shape == (batch, length, total):
                allowed = allowed & attention_mask[:, None, :, :].bool()
            else:
                raise ValueError("Baichuan attention_mask must be a 2D padding or 3D keep mask covering all keys")
        mask = torch.zeros((batch, 1, length, total), device=hidden_states.device, dtype=hidden_states.dtype)
        if self.config.position_embedding_type == "alibi":
            slopes = hidden_states.new_tensor(_get_interleave(self.config.num_attention_heads))
            # Preserve upstream inference's positive absolute key offsets and
            # training's relative offsets (also their low-precision rounding).
            offsets = torch.arange(total, device=hidden_states.device)[None, :]
            if self.training:
                offsets = offsets - positions[:, None]
            bias = slopes[:, None, None] * offsets
            mask = mask + bias.to(hidden_states.dtype)
        mask = mask.masked_fill(~allowed, torch.finfo(hidden_states.dtype).min)
        cache = (past_key_values if past_key_values is not None else DynamicCache(config=self.config)) if use_cache else None
        all_hidden, all_attn = (() if output_hidden_states else None), (() if output_attentions else None)
        for layer in self.layers:
            if output_hidden_states:
                all_hidden += (hidden_states,)
            hidden_states, attn = layer(hidden_states, attention_mask=mask, position_ids=position_ids,
                                        past_key_value=cache, use_cache=use_cache,
                                        output_attentions=output_attentions)
            if output_attentions:
                all_attn += (attn,)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden += (hidden_states,)
        output = BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=cache,
                                        hidden_states=all_hidden, attentions=all_attn)
        return output if return_dict else output.to_tuple()


class NormHead(nn.Linear):
    """Baichuan 2 normalized head without mutating checkpoint weights on use."""

    def forward(self, hidden_states):
        return F.linear(hidden_states, F.normalize(self.weight, dim=1))


class BaichuanForCausalLM(BaichuanPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = BaichuanModel(config)
        head = NormHead if config.normalize_lm_head else nn.Linear
        self.lm_head = head(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, value):
        self.model = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                logits_to_keep=0, **kwargs):
        return_dict = self.config.return_dict if return_dict is None else return_dict
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache,
                             output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                             return_dict=True, **kwargs)
        hidden = outputs.last_hidden_state
        if labels is None:
            hidden = hidden[:, -logits_to_keep:, :] if isinstance(logits_to_keep, int) else hidden[:, logits_to_keep, :]
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shifted = logits[:, :-1].contiguous().view(-1, self.config.vocab_size)
            loss = F.cross_entropy(shifted, labels[:, 1:].contiguous().view(-1).to(logits.device))
            loss = loss + self.config.z_loss_weight * shifted.max(-1).values.square().mean()
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                       hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        return output if return_dict else output.to_tuple()


BaiChuanForCausalLM = BaichuanForCausalLM
