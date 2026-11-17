# Copyright 2024 HuggingFace Inc. team.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright 2025 Manifest AI
# SPDX-License-Identifier: Apache-2.0
# Modified by ModelCloud.ai, 2026: package-local Transformers 5 implementation,
# portable retention arithmetic, transactional recurrent cache, and padding.
"""Brumby with the publisher's projections and degree-2 power retention.

This is a portable Torch implementation, not a claim to qualify retention's
optional Triton/CUDA kernels. See PROVENANCE.md for sources and limitations.
"""

import math

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel

from .configuration_brumby import BrumbyConfig


class _PowerCacheLayer(CacheLayerMixin):
    """Short KV/gate tail plus degree-2 state; length counts compressed tokens."""

    def __init__(self):
        super().__init__()
        self.gates = self.valid = self.state = self.normalizer = None
        self.cumulative_length = 0

    def lazy_initialization(self, key_states, value_states):
        self.device, self.dtype = key_states.device, key_states.dtype
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        raise TypeError("Brumby requires keys, values AND retention gates")

    def get_seq_length(self):
        return self.cumulative_length

    def get_mask_sizes(self, query_length):
        return self.cumulative_length + query_length, 0

    def get_max_length(self):
        return -1

    def reset(self):
        self.keys = self.values = self.gates = self.valid = self.state = self.normalizer = None
        self.cumulative_length = 0
        self.is_initialized = False

    def reorder_cache(self, beam_idx):
        for name in ("keys", "values", "gates", "valid", "state", "normalizer"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.index_select(0, beam_idx.to(tensor.device)))

    def batch_repeat_interleave(self, repeats):
        for name in ("keys", "values", "gates", "valid", "state", "normalizer"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)


class PowerAttentionDynamicCache(Cache):
    """Request-owned recurrent cache. Do not share a mutable cache across requests.

    No legacy/static/offloaded cache conversion is offered: a KV-only cache
    cannot represent the retention gates, state, normalizer and token count.
    """

    def __init__(self, config, batch_size=None, dtype=None, device=None):
        super().__init__(layers=[_PowerCacheLayer() for _ in range(config.num_hidden_layers)])
        self.config = config

    def crop(self, max_length):
        raise ValueError("Brumby recurrent state cannot be cropped; use a fresh cache")

    def offload(self, layer_idx, only_non_sliding=True):
        raise ValueError("Brumby cache offloading is not supported")


class BrumbyRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(
            hidden_states.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * hidden_states.to(dtype)


class BrumbyMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x, repeats):
    return x.repeat_interleave(repeats, dim=1)


def _feature_map(x, key=False):
    """Symmetric square: <phi_q(q), phi_k(k)> == (q @ k)**2.

    Upper-triangular storage avoids retaining the duplicate off-diagonals.
    The publisher's blocked expansion is an equivalent redundant basis.
    """
    rows, cols = torch.triu_indices(x.shape[-1], x.shape[-1], device=x.device)
    features = x.float()[..., rows] * x.float()[..., cols]
    if key:
        features = features * torch.where(rows == cols, 1.0, 2.0)
    return features


def _compress(layer):
    log_g = layer.gates.cumsum(-1)
    discounted_keys = layer.keys.float() * ((log_g[..., -1:] - log_g) / 2).exp().unsqueeze(-1)
    features = _feature_map(discounted_keys, key=True) * layer.valid[:, None, :, None]
    state = features.transpose(-1, -2) @ layer.values.float()
    normalizer = features.sum(-2)
    if layer.state is not None:
        decay = log_g[..., -1].exp()
        state = state + layer.state * decay[..., None, None]
        normalizer = normalizer + layer.normalizer * decay[..., None]
    layer.state, layer.normalizer = state, normalizer
    layer.keys = layer.values = layer.gates = layer.valid = None


def power_retention(query, key, value, log_gate, valid, layer, chunk_size, scaling):
    """Portable publisher attention formula plus equivalent recurrent prefix.

    All accumulation is FP32 (including recurrent state). Query blocks end at
    cache compression boundaries, so continuation may contain multiple tokens.
    Mutations replace tensors: the model can restore a pre-call cache on error.
    """
    outputs = []
    repeats = query.shape[1] // key.shape[1]
    start = 0
    while start < query.shape[2]:
        pending = 0 if layer.keys is None else layer.keys.shape[2]
        count = query.shape[2] - start if chunk_size is None else min(chunk_size - pending, query.shape[2] - start)
        end = start + count
        q, k, v, g, mask = (
            query[:, :, start:end],
            key[:, :, start:end],
            value[:, :, start:end],
            log_gate[:, :, start:end],
            valid[:, start:end],
        )
        if layer.keys is not None:
            k = torch.cat((layer.keys, k), dim=2)
            v = torch.cat((layer.values, v), dim=2)
            g = torch.cat((layer.gates, g), dim=2)
            mask = torch.cat((layer.valid, mask), dim=1)
        cumulative = g.float().cumsum(-1)
        log_weights = (
            2 * ((q.float() @ repeat_kv(k.float(), repeats).transpose(-1, -2)) * scaling).abs().add(1e-7).log()
        )
        log_weights = log_weights + repeat_kv(cumulative[:, :, -count:, None] - cumulative[:, :, None, :], repeats)
        causal = (
            torch.arange(k.shape[2], device=q.device)[None, :]
            <= (pending + torch.arange(count, device=q.device))[:, None]
        )
        allowed = causal[None, None] & mask[:, None, None, :]
        log_weights = log_weights.masked_fill(~allowed, -torch.inf)
        rowmax = log_weights.amax(-1, keepdim=True)
        rowmax = torch.where(torch.isfinite(rowmax), rowmax, 0.0)
        weights = (log_weights - rowmax).exp()
        numerator = weights @ repeat_kv(v.float(), repeats)
        denominator = weights.sum(-1, keepdim=True) + 1e-6
        if layer.state is not None:
            # Query previous state at the tail's current gated coordinate.
            features = _feature_map(q)
            state_num = features @ repeat_kv(layer.state, repeats)
            state_den = (features * repeat_kv(layer.normalizer, repeats).unsqueeze(2)).sum(-1, keepdim=True)
            log_discount = repeat_kv(cumulative[:, :, -count:], repeats).unsqueeze(-1)
            # Stable merge of unnormalized prefix and normalized local scores.
            log_state_scale = log_discount + 2 * math.log(scaling)
            alpha = torch.maximum(rowmax, log_state_scale)
            local_factor = (rowmax - alpha).exp()
            state_factor = (log_state_scale - alpha).exp()
            numerator = numerator * local_factor + state_num * state_factor
            denominator = denominator * local_factor + state_den * state_factor
        outputs.append((numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)).to(query.dtype))
        layer.keys, layer.values, layer.gates, layer.valid = k, v, g, mask
        if chunk_size is not None and k.shape[2] >= chunk_size:
            _compress(layer)
        start = end
    layer.cumulative_length += query.shape[2]
    layer.is_initialized = True
    return torch.cat(outputs, dim=2)


class BrumbyAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.g_proj = nn.Linear(config.hidden_size, config.num_key_value_heads, bias=config.attention_bias)
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = BrumbyRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = BrumbyRMSNorm(self.head_dim, config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        gate = nn.functional.logsigmoid(self.g_proj(hidden_states).float()).transpose(1, 2)
        gate = gate * attention_mask[:, None, :]
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        layer = past_key_values.layers[self.layer_idx] if past_key_values is not None else _PowerCacheLayer()
        if self.config.use_exp:
            pending = layer.cumulative_length
            key = torch.cat((layer.keys, key), 2) if layer.keys is not None else key
            value = torch.cat((layer.values, value), 2) if layer.values is not None else value
            valid = torch.cat((layer.valid, attention_mask), 1) if layer.valid is not None else attention_mask
            rows = pending + torch.arange(query.shape[2], device=query.device)
            cols = torch.arange(key.shape[2], device=query.device)
            allowed = cols[None, :] <= rows[:, None]
            if self.sliding_window is not None:
                allowed = allowed & (cols[None, :] > rows[:, None] - self.sliding_window)
            scores = query.float() @ repeat_kv(key.float(), self.num_key_value_groups).transpose(-1, -2) * self.scaling
            scores = scores.masked_fill(~(allowed[None, None] & valid[:, None, None, :]), -torch.inf)
            weights = scores.softmax(-1).nan_to_num()
            weights = nn.functional.dropout(weights, self.config.attention_dropout, self.training)
            output = (weights @ repeat_kv(value.float(), self.num_key_value_groups)).to(query.dtype)
            layer.keys, layer.values, layer.valid = key, value, valid
            layer.cumulative_length += query.shape[2]
            layer.is_initialized = True
        else:
            output = power_retention(
                query, key, value, gate, attention_mask, layer, self.config.chunk_size, self.scaling
            )
        return self.o_proj(output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()), None


class BrumbyDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = BrumbyAttention(config, layer_idx)
        self.mlp = BrumbyMLP(config)
        self.input_layernorm = BrumbyRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = BrumbyRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, past_key_values=None, **kwargs):
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, attention_mask, past_key_values
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class BrumbyRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]
        self.max_seq_len_cached = self.original_max_seq_len = config.max_position_embeddings
        init_fn = (
            self.compute_default_rope_parameters
            if self.rope_type == "default"
            else ROPE_INIT_FUNCTIONS[self.rope_type]
        )
        self.rope_init_fn = init_fn
        inv_freq, self.attention_scaling = init_fn(config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config, device=None, **kwargs):
        dim = config.head_dim
        return 1.0 / config.rope_parameters["rope_theta"] ** (
            torch.arange(0, dim, 2, device=device).float() / dim
        ), 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        with torch.autocast(device_type=x.device.type if x.device.type != "mps" else "cpu", enabled=False):
            freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float32)
            emb = torch.cat((freqs, freqs), dim=-1)
            return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)


class BrumbyPreTrainedModel(PreTrainedModel):
    config_class = BrumbyConfig
    base_model_prefix = "model"
    _no_split_modules = ["BrumbyDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_attention_backend = False
    _can_compile_fullgraph = False

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                module.bias.data.zero_()
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, BrumbyRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, BrumbyRotaryEmbedding):
            # from_pretrained materializes nonpersistent meta buffers with
            # empty storage. Reconstruct frequencies rather than serializing
            # a runtime-only tensor or relying on checkpoint Python to do it.
            init_fn = (
                module.compute_default_rope_parameters
                if module.rope_type == "default"
                else ROPE_INIT_FUNCTIONS[module.rope_type]
            )
            inv_freq, module.attention_scaling = init_fn(module.config, device=module.inv_freq.device)
            module.inv_freq.copy_(inv_freq)
            module.original_inv_freq.copy_(inv_freq)


class BrumbyModel(BrumbyPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([BrumbyDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = BrumbyRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = BrumbyRotaryEmbedding(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if past_key_values is not None and not isinstance(past_key_values, PowerAttentionDynamicCache):
            raise TypeError("Brumby requires PowerAttentionDynamicCache, not a KV-only or legacy cache")
        if past_key_values is not None and not use_cache:
            raise ValueError("A Brumby cache requires use_cache=True; omit it for an independent forward")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length = hidden_states.shape[:2]
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        if attention_mask is None:
            valid = torch.ones((batch, length), device=hidden_states.device, dtype=torch.bool)
        else:
            if attention_mask.ndim != 2 or attention_mask.shape != (batch, past + length):
                raise ValueError("Brumby attention_mask must be [batch, past_length + input_length]")
            valid = attention_mask[:, -length:].to(device=hidden_states.device, dtype=torch.bool)
        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)[:, -length:]
            else:
                position_ids = torch.arange(past, past + length, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids.to(hidden_states.device))
        cache = past_key_values
        if use_cache and cache is None:
            cache = PowerAttentionDynamicCache(self.config)
        snapshots = [vars(layer).copy() for layer in cache.layers] if cache is not None else None
        all_hidden = () if output_hidden_states else None
        try:
            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden += (hidden_states,)
                hidden_states = decoder_layer(
                    hidden_states, attention_mask=valid, position_embeddings=position_embeddings, past_key_values=cache
                )
            hidden_states = self.norm(hidden_states)
        except Exception:
            if snapshots is not None:
                for layer, snapshot in zip(cache.layers, snapshots):
                    layer.__dict__.clear()
                    layer.__dict__.update(snapshot)
            raise
        if output_hidden_states:
            all_hidden += (hidden_states,)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=cache, hidden_states=all_hidden
        )
        return output if return_dict else output.to_tuple()


class BrumbyForCausalLM(BrumbyPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = BrumbyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        # The publisher's kernel prefill_chunk_size predates HF's generation
        # option of the same name. Do not silently enable HF chunking from it.
        self.generation_config.prefill_chunk_size = None

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        if generation_config.cache_implementation not in (None, "dynamic"):
            raise ValueError("Brumby supports only its dynamic power-retention cache")
        if generation_config.is_assistant:
            raise ValueError("Brumby recurrent cache cannot support speculative rollback")
        cache = model_kwargs.get("past_key_values")
        if cache is not None and not isinstance(cache, PowerAttentionDynamicCache):
            raise TypeError("Brumby requires PowerAttentionDynamicCache")
        if generation_config.use_cache and cache is None:
            model_kwargs["past_key_values"] = PowerAttentionDynamicCache(self.config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        return_dict=None,
        **kwargs,
    ):
        snapshots = (
            [vars(layer).copy() for layer in past_key_values.layers]
            if isinstance(past_key_values, PowerAttentionDynamicCache)
            else None
        )
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=True,
                **kwargs,
            )
            indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(outputs.last_hidden_state[:, indices, :])
            loss = (
                self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
                if labels is not None
                else None
            )
        except Exception:
            # Decoder-level rollback alone does not cover LM-head/loss errors.
            # All cache changes replace tensors, so a shallow snapshot restores
            # the entire request's original storage without cloning large state.
            if snapshots is not None:
                for layer, snapshot in zip(past_key_values.layers, snapshots):
                    layer.__dict__.clear()
                    layer.__dict__.update(snapshot)
            raise
        output = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states
        )
        return output if (self.config.return_dict if return_dict is None else return_dict) else output.to_tuple()
