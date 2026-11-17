# Copyright 2023, 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Based on EleutherAI GPT-NeoX and Hugging Face OPT; see README.md for provenance.
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
"""Portable original Ling / Ling 2.0 / 2.6 with Transformers 5 cache/generation APIs."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_bailing import BailingConfig, is_native_hybrid


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6, groups=1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = eps
        self.groups = groups

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = x.reshape(*shape[:-1], self.groups, shape[-1] // self.groups).float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype).reshape(shape)


class RotaryEmbedding(nn.Module):
    def __init__(self, config, mla=False):
        super().__init__()
        self.config = config
        self.dim = config.qk_rope_head_dim if mla else int(config.head_dim * config.partial_rotary_factor)
        if self.dim % 2:
            raise ValueError("Rotary dimension must be even")
        scaling = config.rope_scaling or {}
        self.rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if self.rope_type not in ("default", "linear", "dynamic", "yarn"):
            raise ValueError(f"Unqualified Ling RoPE type: {self.rope_type}")
        inv_freq, self.attention_scaling = self._parameters_for_length(config.max_position_embeddings)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _parameters_for_length(self, length):
        config = self.config
        scaling = config.rope_scaling or {}
        factor = scaling.get("factor", 1.0)
        base, dim = config.rope_theta, self.dim
        attention_scaling = 1.0
        if self.rope_type == "dynamic" and length > config.max_position_embeddings:
            base *= (factor * length / config.max_position_embeddings - (factor - 1)) ** (dim / (dim - 2))
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        if self.rope_type == "linear":
            inv_freq /= factor
        elif self.rope_type == "yarn":
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            # Modern YaRN reads rope_parameters, not the legacy rope_scaling dict.
            rope_config = BailingConfig.from_dict(config.to_dict())
            rope_config.head_dim = dim
            rope_config.rope_parameters = {"rope_theta": base, **scaling}
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS["yarn"](rope_config)
        return inv_freq, attention_scaling

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = self.inv_freq
        if self.rope_type == "dynamic":
            inv_freq, _ = self._parameters_for_length(int(position_ids.max()) + 1)
        freqs = position_ids[..., None].float() * inv_freq.to(x.device).float()
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)


def apply_rotary(q, k, position_embeddings, interleave=False):
    cos, sin = (v.unsqueeze(1) for v in position_embeddings)
    dim = cos.shape[-1]

    def rotate(x):
        rot, rest = x[..., :dim], x[..., dim:]
        if interleave:
            rot = rot.reshape(*rot.shape[:-1], dim // 2, 2).transpose(-1, -2).reshape_as(rot)
        first, second = rot.chunk(2, dim=-1)
        return torch.cat((rot * cos + torch.cat((-second, first), dim=-1) * sin, rest), dim=-1)

    return rotate(q), rotate(k)


class MLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        if not config.legacy_gqa:
            self.register_buffer("expert_bias", torch.zeros(config.num_experts))

    def forward(self, x):
        config = self.config
        if config.legacy_gqa:
            logits = F.linear(x.reshape(-1, x.shape[-1]), self.weight)
            scores = F.softmax(logits, dim=-1, dtype=torch.float32)
            weights, indices = scores.topk(config.num_experts_per_tok, dim=-1, sorted=False)
            if config.num_experts_per_tok > 1 and config.norm_topk_prob:
                weights = weights / weights.sum(-1, keepdim=True)
            return indices, weights, logits
        logits = F.linear(x.reshape(-1, x.shape[-1]).float(), self.weight.float())
        scores = logits.sigmoid()
        routing_scores = scores + self.expert_bias
        groups = routing_scores.view(-1, config.n_group, config.num_experts // config.n_group)
        group_scores = groups.topk(2, dim=-1).values.sum(-1)
        selected_groups = group_scores.topk(config.topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores).scatter_(1, selected_groups, 1).bool()
        mask = group_mask.unsqueeze(-1).expand_as(groups).reshape_as(scores)
        indices = routing_scores.masked_fill(~mask, -torch.inf).topk(config.num_experts_per_tok, dim=-1).indices
        weights = scores.gather(1, indices)
        if config.num_experts_per_tok > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return indices, weights * config.routed_scaling_factor, logits


class SparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([MLP(config, config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.gate = Gate(config)
        if config.num_shared_experts is not None:
            self.shared_experts = MLP(config, config.moe_intermediate_size * config.num_shared_experts)

    def forward(self, x):
        indices, weights, logits = self.gate(x)
        flat = x.reshape(-1, x.shape[-1])
        # Upstream inference accumulates in the router's FP32 before casting.
        out = torch.zeros(flat.shape, device=x.device, dtype=weights.dtype)
        for idx, expert in enumerate(self.experts):
            tokens, slots = torch.where(indices == idx)
            if tokens.numel():
                values = expert(flat[tokens]).to(weights.dtype) * weights[tokens, slots, None]
                out.index_add_(0, tokens, values)
        out = out.to(x.dtype).reshape_as(x)
        if hasattr(self, "shared_experts"):
            out = out + self.shared_experts(x)
        return out, (logits.reshape(*x.shape[:-1], -1), indices.reshape(*x.shape[:-1], -1))


def softmax_attention(q, k, v, mask, scale, dropout, training):
    repeat = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
    scores = (q @ k.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    weights = F.dropout(weights, p=dropout, training=training)
    return (weights @ v).transpose(1, 2).contiguous(), weights


class DenseAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.num_heads, self.head_dim = config.num_attention_heads, config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.query_key_value = nn.Linear(
            config.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )
        if config.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias)

    def project(self, x, positions):
        qkv = self.query_key_value(x).view(*x.shape[:-1], -1, self.head_dim)
        q, k, v = qkv.split((self.num_heads, self.num_key_value_heads, self.num_key_value_heads), dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.config.use_qk_norm:
            q, k = self.query_layernorm(q), self.key_layernorm(k)
        q, k = apply_rotary(q, k, positions)
        return q, k, v

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        q, k, v = self.project(hidden_states, position_embeddings)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        out, weights = softmax_attention(
            q, k, v, attention_mask, self.head_dim ** -0.5, self.config.attention_dropout, self.training,
        )
        return self.dense(out.reshape(*hidden_states.shape[:-1], -1)), weights


class MLAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        if config.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.use_qkv_bias)
            self.q_a_layernorm = RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, config.num_attention_heads * config.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=config.use_qkv_bias,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank, config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim), bias=False,
        )
        self.dense = nn.Linear(config.num_attention_heads * config.v_head_dim, config.hidden_size, bias=config.use_qkv_bias)
        self.scaling = config.qk_head_dim ** -0.5
        scaling = config.rope_scaling or {}
        if scaling.get("mscale_all_dim", 0) and scaling["factor"] > 1:
            self.scaling *= (0.1 * scaling["mscale_all_dim"] * math.log(scaling["factor"]) + 1.0) ** 2

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        c = self.config
        batch, length = hidden_states.shape[:2]
        q = self.q_proj(hidden_states) if c.q_lora_rank is None else self.q_b_proj(
            self.q_a_layernorm(self.q_a_proj(hidden_states))
        )
        q = q.view(batch, length, c.num_attention_heads, c.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = q.split((c.qk_nope_head_dim, c.qk_rope_head_dim), dim=-1)
        compressed = self.kv_a_proj_with_mqa(hidden_states)
        latent, k_rot = compressed.split((c.kv_lora_rank, c.qk_rope_head_dim), dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(latent)).view(
            batch, length, c.num_attention_heads, c.qk_nope_head_dim + c.v_head_dim,
        ).transpose(1, 2)
        k_pass, v = kv.split((c.qk_nope_head_dim, c.v_head_dim), dim=-1)
        k_rot = k_rot.view(batch, 1, length, c.qk_rope_head_dim)
        q_rot, k_rot = apply_rotary(q_rot, k_rot, position_embeddings, interleave=c.rope_interleave)
        q = torch.cat((q_pass, q_rot), dim=-1)
        k = torch.cat((k_pass, k_rot.expand(*k_pass.shape[:-1], -1)), dim=-1)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        out, weights = softmax_attention(q, k, v, attention_mask, self.scaling, c.attention_dropout, self.training)
        return self.dense(out.reshape(batch, length, -1)), weights


def lightning_slopes(heads):
    # OpenNLPLab/lightning-attention d15c38529bbd5c2c82b44ddda3cac885825aa873.
    def power_of_two(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start ** (i + 1) for i in range(n)]

    if math.log2(heads).is_integer():
        return power_of_two(heads)
    closest = 2 ** math.floor(math.log2(heads))
    return power_of_two(closest) + lightning_slopes(2 * closest)[0::2][:heads - closest]


class LinearAttention(DenseAttention):
    def __init__(self, config, layer_idx):
        # Ling's linear branch uses all heads for K/V, regardless of dense GQA.
        super().__init__(config, layer_idx)
        self.num_key_value_heads = config.num_attention_heads
        self.query_key_value = nn.Linear(config.hidden_size, self.num_heads * 3 * self.head_dim, bias=config.use_qkv_bias)
        self.g_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.g_norm = RMSNorm(self.num_heads * self.head_dim, config.rms_norm_eps, groups=config.group_norm_size)
        # Kept for checkpoint/module compatibility; frequencies are shared by the model.
        self.rotary_emb = RotaryEmbedding(config)
        slope = -torch.tensor(lightning_slopes(self.num_heads)) * (
            1 - (layer_idx - 1) / max(config.num_hidden_layers - 1, 1) + 1e-5
        )
        self.register_buffer("slope", slope, persistent=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        q, k, v = self.project(hidden_states, position_embeddings)
        state = None if past_key_values is None else past_key_values.recurrent_states.get(self.layer_idx)
        if state is None:
            state = torch.zeros(
                q.shape[0], self.num_heads, self.head_dim, self.head_dim, device=q.device, dtype=torch.float32,
            )
        decay = self.slope.to(q.device).float().exp()[None, :, None, None]
        outputs = []
        for token in range(q.shape[2]):
            value = v[:, :, token].float()
            if attention_mask is not None:
                value = value * attention_mask[:, -q.shape[2] + token, None, None]
            state = state * decay + k[:, :, token].float().unsqueeze(-1) * value.unsqueeze(-2)
            # FLA's default scale is 1 / sqrt(key head dimension).
            outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, :, token].float(), state) * self.head_dim ** -0.5)
        out = torch.stack(outputs, dim=1).to(hidden_states.dtype).reshape(*hidden_states.shape[:-1], -1)
        out = self.g_norm(out) * self.g_proj(hidden_states).sigmoid()
        if past_key_values is not None:
            past_key_values.recurrent_states[self.layer_idx] = state
        return self.dense(out), None


class BailingCache(DynamicCache):
    """KV tensors and recurrent state have separate ownership and sequence length."""

    def __init__(self):
        super().__init__()
        self.recurrent_states = {}
        self._ling_seen_tokens = 0
        self.incomplete = False

    def get_seq_length(self, layer_idx=0):
        return self._ling_seen_tokens

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        self.recurrent_states = {
            key: value.index_select(0, beam_idx.to(value.device)) for key, value in self.recurrent_states.items()
        }

    def batch_repeat_interleave(self, repeats):
        super().batch_repeat_interleave(repeats)
        self.recurrent_states = {key: value.repeat_interleave(repeats, dim=0) for key, value in self.recurrent_states.items()}

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self.recurrent_states = {
            key: value[indices.to(value.device)] for key, value in self.recurrent_states.items()
        }

    def crop(self, tokens_to_remove):
        if self.recurrent_states and tokens_to_remove != 0:
            raise NotImplementedError("Ling recurrent cache cannot be cropped; replay the prefix into a fresh cache")
        super().crop(tokens_to_remove)
        self._ling_seen_tokens = max(0, self._ling_seen_tokens + tokens_to_remove)

    def reset(self):
        self.layers.clear()
        self.recurrent_states.clear()
        self._ling_seen_tokens = 0
        self.incomplete = False


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, mtp=False):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_mtp = mtp
        hybrid = config.model_type == "bailing_hybrid"
        linear = hybrid and not mtp and (layer_idx + 1) % config.layer_group_size != 0 and (
            layer_idx < config.num_hidden_layers // config.layer_group_size * config.layer_group_size
        )
        attention_cls = LinearAttention if linear else MLAttention if hybrid or config.q_lora_rank is not None else DenseAttention
        self.attention = attention_cls(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SparseMoeBlock(config) if config.num_experts is not None and (
            mtp or layer_idx >= config.first_k_dense_replace
        ) else MLP(config, config.intermediate_size)
        if mtp:
            self.enorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.hnorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
            self.final_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, input_embeds=None, **kwargs):
        if self.is_mtp:
            hidden_states = self.eh_proj(torch.cat((self.enorm(input_embeds), self.hnorm(hidden_states)), dim=-1))
        residual = hidden_states
        out, weights = self.attention(
            self.input_layernorm(hidden_states), position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values,
        )
        hidden_states = residual + out
        out = self.mlp(self.post_attention_layernorm(hidden_states))
        router_logits = None
        if isinstance(out, tuple):
            out, router_logits = out
        hidden_states = hidden_states + out
        if self.is_mtp:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, weights, router_logits


@dataclass
class BailingModelOutput(MoeModelOutputWithPast):
    mtp_hidden_states: tuple | None = None


@dataclass
class BailingCausalLMOutput(MoeCausalLMOutputWithPast):
    mtp_logits: tuple | None = None
    mtp_loss: tuple | None = None


class BailingPreTrainedModel(PreTrainedModel):
    config_class = BailingConfig
    base_model_prefix = "model"
    _no_split_modules = ["DecoderLayer"]
    _supports_sdpa = False
    _is_stateful = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Gate)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
            if isinstance(module, Gate) and hasattr(module, "expert_bias"):
                nn.init.zeros_(module.expert_bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, RotaryEmbedding):
            frequency, _ = module._parameters_for_length(module.config.max_position_embeddings)
            module.inv_freq.copy_(frequency.to(module.inv_freq.device))
        elif isinstance(module, LinearAttention):
            slope = -torch.tensor(lightning_slopes(module.num_heads), device=module.slope.device) * (
                1 - (module.layer_idx - 1) / max(module.config.num_hidden_layers - 1, 1) + 1e-5
            )
            module.slope.copy_(slope)


class BailingModel(BailingPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([
            DecoderLayer(config, idx, mtp=idx >= config.num_hidden_layers)
            for idx in range(config.num_hidden_layers + config.num_nextn_predict_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        if any(isinstance(layer.attention, MLAttention) for layer in self.layers):
            self.rotary_emb_mla = RotaryEmbedding(config, mla=True)
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
        output_router_logits=None, return_dict=True, **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, BailingCache):
            raise TypeError("Ling requires BailingCache, not a KV-only or static cache")
        if past_key_values is not None and past_key_values.incomplete:
            raise ValueError("Ling cache contains a failed partial forward; reset it and replay the prefix")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        output_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits
        hidden = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length = hidden.shape[:2]
        seen = 0 if past_key_values is None else past_key_values.get_seq_length()
        if attention_mask is not None and (attention_mask.ndim != 2 or attention_mask.shape != (batch, seen + length)):
            raise ValueError("Ling attention_mask must be [batch, past_length + input_length]")
        if position_ids is None:
            position_ids = torch.arange(seen, seen + length, device=hidden.device).unsqueeze(0)
        positions = self.rotary_emb(hidden, position_ids)
        mla_positions = self.rotary_emb_mla(hidden, position_ids) if hasattr(self, "rotary_emb_mla") else positions
        causal = torch.arange(seen + length, device=hidden.device)[None, :] <= torch.arange(
            seen, seen + length, device=hidden.device,
        )[:, None]
        causal = causal[None, None].expand(batch, 1, length, seen + length)
        if attention_mask is not None:
            causal = causal & attention_mask[:, None, None, :].bool()
        mask = torch.zeros(causal.shape, dtype=hidden.dtype, device=hidden.device).masked_fill(
            ~causal, torch.finfo(hidden.dtype).min,
        )
        cache = past_key_values if use_cache else None
        if use_cache and cache is None:
            cache = BailingCache()
        if cache is not None:
            cache.incomplete = True
        all_hidden, all_attn, all_router, mtp_hidden = (), (), (), ()
        main_hidden = None
        mtp_ids = input_ids
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden += (hidden,)
            embeds = None
            if layer.is_mtp:
                if mtp_ids is None:
                    raise ValueError("Ling MTP requires input_ids for shifted token embeddings")
                mtp_ids = torch.roll(mtp_ids, -1, -1).clone()
                mtp_ids[:, -1] = 0
                embeds = self.word_embeddings(mtp_ids)
            linear = isinstance(layer.attention, LinearAttention)
            hidden, weights, router = layer(
                hidden, position_embeddings=mla_positions if isinstance(layer.attention, MLAttention) else positions,
                attention_mask=attention_mask if linear else mask, past_key_values=cache, input_embeds=embeds,
            )
            if idx == self.config.num_hidden_layers - 1:
                hidden = self.norm(hidden)
                main_hidden = hidden
            if layer.is_mtp:
                mtp_hidden += (hidden,)
            if output_attentions:
                all_attn += (weights,)
            if output_router_logits and router is not None:
                all_router += (router,)
        if cache is not None:
            cache._ling_seen_tokens = seen + length
            cache.incomplete = False
        if output_hidden_states:
            all_hidden += (hidden,)
        output = BailingModelOutput(
            last_hidden_state=main_hidden, past_key_values=cache,
            hidden_states=all_hidden if output_hidden_states else None,
            attentions=all_attn if output_attentions else None,
            router_logits=all_router if output_router_logits else None, mtp_hidden_states=mtp_hidden or None,
        )
        return output if return_dict else output.to_tuple()


class BailingForCausalLM(BailingPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.word_embeddings.weight"}

    @classmethod
    def _from_config(cls, config, **kwargs):
        if not isinstance(config, BailingConfig):
            from transformers.models.bailing_hybrid.configuration_bailing_hybrid import BailingHybridConfig
            from transformers.models.bailing_hybrid.modeling_bailing_hybrid import BailingHybridForCausalLM

            if not isinstance(config, BailingHybridConfig):
                raise TypeError("Bailing loader requires a Bailing config, not another architecture")
            if not is_native_hybrid(config.to_dict()):
                return super()._from_config(BailingConfig.from_dict(config.to_dict()), **kwargs)
            return BailingHybridForCausalLM._from_config(config, **kwargs)
        return super()._from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, config=None, **kwargs):
        if config is None:
            config = BailingConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(config, BailingConfig):
            from transformers.models.bailing_hybrid.configuration_bailing_hybrid import BailingHybridConfig
            from transformers.models.bailing_hybrid.modeling_bailing_hybrid import BailingHybridForCausalLM

            if not isinstance(config, BailingHybridConfig):
                raise TypeError("Bailing loader requires a Bailing config, not another architecture")
            if not is_native_hybrid(config.to_dict()):
                config = BailingConfig.from_dict(config.to_dict())
                return super().from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)
            return BailingHybridForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)

    def __init__(self, config):
        super().__init__(config)
        self.model = BailingModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        if generation_config.cache_implementation is not None:
            raise ValueError("Ling supports only its dynamic hybrid cache")
        if generation_config.use_cache and model_kwargs.get("past_key_values") is None:
            model_kwargs["past_key_values"] = BailingCache()

    def forward(self, input_ids=None, attention_mask=None, labels=None, logits_to_keep=0, return_dict=True, **kwargs):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs)
        cache = output.past_key_values
        if cache is not None:
            cache.incomplete = True
        selection = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        def compute_logits(hidden):
            if self.config.norm_head:
                weight = self.lm_head.weight.float()
                weight = (weight / weight.norm(p=2, dim=0, keepdim=True).clamp(min=1e-7)).to(hidden.dtype)
                return F.linear(hidden, weight).float()
            return self.lm_head(hidden).float()

        logits = compute_logits(output.last_hidden_state[:, selection])
        loss = None if labels is None else self.loss_function(logits, labels, self.config.vocab_size)
        mtp_logits = tuple(compute_logits(hidden) for hidden in output.mtp_hidden_states or ())
        mtp_loss = []
        if labels is not None:
            shifted = labels
            for prediction in mtp_logits:
                shifted = torch.roll(shifted, -1, -1).clone()
                shifted[:, -1] = -100
                mtp_loss.append(self.loss_function(prediction, shifted, self.config.vocab_size))
                loss = loss + self.config.mtp_loss_scaling_factor * mtp_loss[-1]
        result = BailingCausalLMOutput(
            loss=loss, logits=logits, past_key_values=output.past_key_values,
            hidden_states=output.hidden_states, attentions=output.attentions, router_logits=output.router_logits,
            mtp_logits=mtp_logits or None, mtp_loss=tuple(mtp_loss) or None,
        )
        result = result if return_dict else result.to_tuple()
        if cache is not None:
            cache.incomplete = False
        return result
