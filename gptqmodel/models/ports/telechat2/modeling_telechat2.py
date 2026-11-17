# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
# Copyright (c) 2020, 2023 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 EleutherAI.
# SPDX-License-Identifier: Apache-2.0
"""Source-matched TeleChat attention, residual blocks and dynamic NTK RoPE."""

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_telechat2 import Telechat2Config


class MixedFusedRMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        values = hidden_states.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * values.to(hidden_states.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.dim = dim
        self.config = config

    def forward(self, x, offset):
        length = max(offset + x.shape[-2], self.config.training_seqlen)
        alpha = max(2 ** math.ceil(math.log2(length / self.config.base_seqlen) + 1) - 1, 1)
        base = self.config.rope_theta * alpha ** (self.dim / (self.dim - 2))
        inv = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32) / self.dim))
        positions = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=torch.float32)
        angles = torch.outer(positions, inv)
        angles = torch.cat((angles, angles), dim=-1)
        scale = max(1.0, 1 + 0.1 * math.log(length / self.config.training_seqlen))
        # Source rounds the trigonometric table to half BEFORE amplitude scaling.
        cos, sin = scale * angles.cos().half(), scale * angles.sin().half()
        if self.config.telechat_rope_precision == "bfloat16":
            cos, sin = cos.bfloat16(), sin.bfloat16()
        return cos, sin


def _rotate(x, cos, sin):
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), dim=-1) * sin


class TelechatAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.n_head
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.n_head
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key_value = nn.Linear(config.hidden_size, self.head_dim * self.num_key_value_heads * 2, bias=False)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config)

    def forward(self, hidden_states, residual, attention_mask, past_key_values=None, output_attentions=False):
        batch, length, _ = hidden_states.shape
        query = self.query(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.key_value(hidden_states).view(batch, length, self.num_key_value_heads, 2 * self.head_dim)
        key, value = (part.transpose(1, 2) for part in kv.chunk(2, dim=-1))
        offset = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        cos, sin = self.rotary_emb(value, offset)
        query, key = _rotate(query, cos, sin).to(query.dtype), _rotate(key, cos, sin).to(key.dtype)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        groups = self.num_heads // self.num_key_value_heads
        key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
        dropout = self.config.attention_dropout if self.training else 0.0
        probabilities = None
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            context = F.scaled_dot_product_attention(query, key, value, attn_mask=~attention_mask, dropout_p=dropout)
        else:
            scores = (query @ key.transpose(-1, -2)) / math.sqrt(self.head_dim)
            dtype = scores.dtype
            if dtype == torch.float16:
                scores = scores.float()
            scores = scores.masked_fill(attention_mask, torch.finfo(scores.dtype).min)
            probabilities = scores.softmax(-1).to(dtype)
            context = F.dropout(probabilities, p=dropout, training=self.training) @ value
        context = context.transpose(1, 2).reshape(batch, length, -1)
        output = F.dropout(self.dense(context), p=self.config.hidden_dropout, training=self.training) + residual
        return output, probabilities


class TelechatMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=True)

    def forward(self, hidden_states, residual):
        result = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return F.dropout(result, p=self.config.hidden_dropout, training=self.training) + residual


class TelechatBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = MixedFusedRMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.self_attention = TelechatAttention(config, layer_idx)
        self.post_attention_layernorm = MixedFusedRMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.mlp = TelechatMLP(config)
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

    def forward(self, hidden_states, attention_mask, past_key_values=None, output_attentions=False, **kwargs):
        normalized = self.input_layernorm(hidden_states)
        residual = normalized if self.apply_residual_connection_post_layernorm else hidden_states
        hidden_states, weights = self.self_attention(
            normalized, residual, attention_mask, past_key_values, output_attentions,
        )
        normalized = self.post_attention_layernorm(hidden_states)
        residual = normalized if self.apply_residual_connection_post_layernorm else hidden_states
        return self.mlp(normalized, residual), weights


class TelechatPreTrainedModel(PreTrainedModel):
    config_class = Telechat2Config
    base_model_prefix = "transformer"
    _no_split_modules = ["TelechatBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, MixedFusedRMSNorm):
            nn.init.ones_(module.weight)


class TelechatModel(TelechatPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.embed_layernorm:
            self.word_embeddings_layernorm = MixedFusedRMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.h = nn.ModuleList([TelechatBlock(config, index) for index in range(config.n_layer)])
        self.ln_f = MixedFusedRMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, inputs_embeds=None,
                use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise TypeError("TeleChat supports DynamicCache, not foreign or legacy tuple caches")
        if past_key_values is not None and not use_cache:
            raise ValueError("A past_key_values cache requires use_cache=True")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        hidden_states = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        if self.config.embed_layernorm:
            hidden_states = self.word_embeddings_layernorm(hidden_states)
        batch, length, _ = hidden_states.shape
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        key_length = past_length + length
        future = torch.arange(key_length, device=hidden_states.device) > (
            past_length + torch.arange(length, device=hidden_states.device)[:, None]
        )
        mask = future[None, None, :, :].expand(batch, 1, length, key_length)
        if attention_mask is not None:
            if attention_mask.shape != (batch, key_length):
                raise ValueError("attention_mask must cover the full cached and current sequence")
            mask = mask | ~attention_mask.to(device=hidden_states.device, dtype=torch.bool)[:, None, None, :]
        states, attentions = (), ()
        for block in self.h:
            if output_hidden_states:
                states += (hidden_states,)
            hidden_states, weights = block(hidden_states, mask, past_key_values, output_attentions)
            if output_attentions:
                attentions += (weights,)
        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            states += (hidden_states,)
        output = BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values,
                                        hidden_states=states if output_hidden_states else None,
                                        attentions=attentions if output_attentions else None)
        return output if return_dict else output.to_tuple()


class Telechat2ForCausalLM(TelechatPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.word_embeddings.weight"}

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> Any:
        output_loading_info = kwargs.pop("output_loading_info", False)
        result: Any = super().from_pretrained(*args, output_loading_info=True, **kwargs)
        model, info = result
        if info.get("missing_keys"):
            raise RuntimeError(f"TeleChat causal checkpoint is missing required weights: {info['missing_keys']}")
        return (model, info) if output_loading_info else model

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TelechatModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.word_embeddings

    def set_input_embeddings(self, value):
        self.transformer.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, inputs_embeds=None,
                labels=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, logits_to_keep=0, **kwargs):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
                                  inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states, return_dict=True)
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(output.last_hidden_state[:, indices, :])
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size).float(),
                                   labels[:, 1:].reshape(-1).to(logits.device))
        result = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=output.past_key_values,
                                       hidden_states=output.hidden_states, attentions=output.attentions)
        return result if (self.config.return_dict if return_dict is None else return_dict) else result.to_tuple()
