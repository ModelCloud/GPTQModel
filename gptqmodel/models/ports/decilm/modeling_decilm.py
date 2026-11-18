# Copyright 2024 NVIDIA Corporation, Google Inc, HuggingFace Inc, EleutherAI. All rights reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See NOTICE for pinned Deci/NVIDIA/Hugging Face source provenance.
"""Transformers 5 DeciLM: heterogeneous GQA and pruned NAS decoder blocks."""

import copy

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, CacheLayerMixin, DummyLayer, DynamicLayer
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, slice_logits_to_keep
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_decilm import DeciLMConfig


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def project(linear, x, tp, output_split=False):
    """Preserve legacy pretraining_tp's bias-free partial projections."""
    if tp == 1:
        return linear(x)
    axis = 1 if output_split else 0
    size = linear.weight.shape[axis]
    if size % tp:
        raise ValueError("pretraining_tp must divide every projection slice width")
    weights = linear.weight.split(size // tp, dim=axis)
    if output_split:
        return sum(F.linear(part, weight) for part, weight in zip(x.split(size // tp, dim=-1), weights))
    return torch.cat([F.linear(x, weight) for weight in weights], dim=-1)


def rope(config, x, positions, seq_len):
    """Stateless RoPE: no process/module-wide mutable frequency cache."""
    kind = config.rope_parameters["rope_type"]
    if kind == "default":
        inv = 1.0 / (config.rope_parameters["rope_theta"] ** (
            torch.arange(0, config.head_dim, 2, device=x.device).float() / config.head_dim
        ))
        scale = 1.0
    else:
        # The official HF functions implement linear/dynamic/Llama-3 scaling.
        # No checkpoint module or global model registration is involved.
        inv, scale = ROPE_INIT_FUNCTIONS[kind](config, device=x.device, seq_len=seq_len)
    with torch.autocast(device_type=x.device.type, enabled=False):
        angles = positions.float()[..., None] * inv.float()
        angles = torch.cat((angles, angles), dim=-1)
        return (angles.cos() * scale).to(x.dtype), (angles.sin() * scale).to(x.dtype)


class DeciLMSinkLayer(DynamicLayer):
    """Shifted sink-window KV layer (upstream SinkCache semantics).

    Retains sink+window keys and re-rotates retained keys on eviction, so the
    logical length is tracked in `cumulative_length`, like 5.23 sliding layers.
    """

    is_croppable = False

    def __init__(self, window_length, num_sink_tokens, **kwargs):
        super().__init__(**kwargs)
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cumulative_length = 0
        self.sink_cos = self.sink_sin = None

    def update(self, key_states, value_states, cache_kwargs=None, *args, **kwargs):
        window, sinks = self.window_length, self.num_sink_tokens
        cos, sin = cache_kwargs["cos"][0], cache_kwargs["sin"][0]
        if self.sink_cos is None:
            self.sink_cos, self.sink_sin = cos, sin
        elif self.sink_cos.shape[0] < window:
            self.sink_cos = torch.cat((self.sink_cos, cos), dim=0)
            self.sink_sin = torch.cat((self.sink_sin, sin), dim=0)
        count = key_states.shape[-2]
        old_count = super().get_seq_length()
        if not old_count or old_count + count < window:
            self.cumulative_length += count
            return super().update(key_states, value_states)
        if count >= window - sinks:
            raise ValueError("Shifted-sink continuation chunks must be smaller than window_length - num_sink_tokens")
        # Keep fixed sink keys and the most recent window-sinks-count keys.
        keep = window - sinks - count
        keys = self.keys[..., -keep:, :] if keep else self.keys[..., :0, :]
        values = self.values[..., -keep:, :] if keep else self.values[..., :0, :]
        # Pinned SinkCache uses the first window's RoPE table to shift retained
        # keys back by the incoming chunk width. Preserve this even when the
        # caller supplies absolute positions after the first eviction.
        old_cos = self.sink_cos[sinks + count:window].float()[None]
        old_sin = self.sink_sin[sinks + count:window].float()[None]
        new_cos = self.sink_cos[sinks:window - count].float()[None]
        new_sin = self.sink_sin[sinks:window - count].float()[None]
        cos = old_cos * new_cos + old_sin * new_sin
        sin = -old_sin * new_cos + old_cos * new_sin
        keys = keys * cos[:, None].to(keys.dtype) + rotate_half(keys) * sin[:, None].to(keys.dtype)
        self.keys = torch.cat((self.keys[..., :sinks, :], keys, key_states), dim=-2)
        self.values = torch.cat((self.values[..., :sinks, :], values, value_states), dim=-2)
        self.cumulative_length += count
        return self.keys, self.values

    def get_seq_length(self):
        return self.cumulative_length

    def get_mask_sizes(self, query_length):
        return super().get_seq_length() + query_length, 0

    def reset(self):
        super().reset()
        self.sink_cos = self.sink_sin = None

    def crop(self, tokens_to_remove):
        raise ValueError("Shifted-sink cache cannot restore discarded history")


def _cache_layer(block):
    attn = block["attention"]
    if attn.get("no_op") or attn.get("replace_with_linear"):
        return DummyLayer()
    window, sinks = attn.get("window_length"), attn.get("num_sink_tokens")
    if window is not None and sinks is not None and not attn.get("unshifted_sink"):
        return DeciLMSinkLayer(window, sinks)
    # Global/sliding/unshifted attention keeps full history and masks at consumption.
    return DynamicLayer()


class DeciLMCache(Cache):
    """Caller-owned heterogeneous cache, including stacks whose first block is pruned.

    Pruned or linear attention blocks own a `DummyLayer`; shifted sink blocks a
    `DeciLMSinkLayer`; everything else a `DynamicLayer`. Updates are
    transactional at model-forward scope (see `fork`) so a failed later layer
    leaves a reusable cache.
    """

    def __init__(self, config):
        super().__init__(layers=[_cache_layer(config.layer_config(i)) for i in range(config.num_hidden_layers)])

    def get_seq_length(self, layer_idx=0):
        if not any(isinstance(layer, CacheLayerMixin) for layer in self.layers):
            # Attention-free stacks hold no KV state; every token is independent.
            return 0
        return super().get_seq_length(layer_idx)

    def fork(self):
        other = copy.copy(self)
        other.layers = [copy.copy(layer) for layer in self.layers]
        return other

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    weights = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights


class DeciLMRMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(x.dtype)


class DeciLMMLP(nn.Module):
    def __init__(self, config, part):
        super().__init__()
        self.config = config
        width = part.get("intermediate_size")
        if width is None:
            # NVIDIA's _ffn_mult_to_intermediate_size / _find_multiple.
            raw = int(2 * round(part["ffn_mult"], 6) * config.hidden_size / 3)
            width = ((raw + 255) // 256) * 256
        self.gate_proj = nn.Linear(config.hidden_size, width, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, width, bias=config.mlp_bias)
        self.down_proj = nn.Linear(width, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        tp = self.config.pretraining_tp
        hidden = self.act_fn(project(self.gate_proj, x, tp)) * project(self.up_proj, x, tp)
        return project(self.down_proj, hidden, tp, output_split=True)


class DeciLMLinear(nn.Module):
    def __init__(self, config, name):
        super().__init__()
        self.name = name
        self.add_module(name, nn.Linear(config.hidden_size, config.hidden_size, bias=False))

    def forward(self, x):
        return self.get_submodule(self.name)(x)


class DeciLMAttention(nn.Module):
    def __init__(self, config, part, layer_idx):
        super().__init__()
        self.config, self.part, self.layer_idx = config, part, layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = part["n_heads_in_group"]
        self.num_key_value_heads = self.num_heads // self.num_key_value_groups
        self.scaling = self.head_dim ** -0.5
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        batch, length, _ = hidden_states.shape
        tp = self.config.pretraining_tp
        q = project(self.q_proj, hidden_states, tp).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = project(self.k_proj, hidden_states, tp).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = project(self.v_proj, hidden_states, tp).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        seen = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        cos, sin = rope(self.config, hidden_states, position_ids, max(seen + length, int(position_ids.max()) + 1))
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx, {"cos": cos, "sin": sin})
        key_length = k.shape[-2]
        # The heterogeneous window/sink masks (including shifted-sink re-indexing
        # and unmasked unshifted sinks) are block-specific, so this block builds
        # its own additive mask instead of using masking_utils. Construct it from physical sequence positions,
        # independently of RoPE's per-example positions (left padding).
        query_pos = torch.arange(seen, seen + length, device=q.device)
        key_pos = torch.arange(key_length, device=q.device)
        shifted = self.part.get("num_sink_tokens") is not None and not self.part.get("unshifted_sink")
        if shifted and past_key_values is not None and seen + length > key_length:
            query_pos = torch.arange(key_length - length, key_length, device=q.device)
        allowed = key_pos[None] <= query_pos[:, None]
        window = self.part.get("window_length")
        unshifted = self.part.get("unshifted_sink", False)
        sinks = self.part.get("num_sink_tokens")
        if window is not None and (sinks is None or unshifted
                                   or self.part.get("use_prefill_window_in_sink_attention")):
            allowed = allowed & (key_pos[None] > query_pos[:, None] - window)
        if unshifted and sinks is not None:
            # Upstream _unshifted_sink_mask explicitly unmasks all sink
            # columns, including padding/future sinks; preserve that behavior.
            allowed = allowed | (key_pos[None] < sinks)
        mask = torch.zeros((batch, 1, length, key_length), dtype=q.dtype, device=q.device)
        mask = mask.masked_fill(~allowed[None, None], torch.finfo(q.dtype).min)
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                mask = mask + attention_mask[..., :length, :key_length]
            elif attention_mask.ndim == 2:
                keep = attention_mask
                if shifted and keep.shape[-1] > key_length:
                    keep = torch.cat((keep[:, :sinks], keep[:, -(key_length - sinks):]), dim=-1)
                mask = mask.masked_fill(~keep[:, None, None, :key_length].bool(), torch.finfo(q.dtype).min)
            else:
                raise ValueError("attention_mask must be 2D keep mask or 4D additive mask")
            if unshifted and sinks is not None:
                mask[..., :sinks] = 0
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,
                                                                    eager_attention_forward)
        output, weights = attention_interface(
            self, q, k, v, mask, dropout=self.config.attention_dropout if self.training else 0.0,
            scaling=self.scaling, **kwargs,
        )
        output = output.reshape(batch, length, -1).contiguous()
        return project(self.o_proj, output, tp, output_split=True), weights


class DeciLMDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.block_config = config.layer_config(layer_idx)
        attn, ffn = self.block_config["attention"], self.block_config["ffn"]
        if not attn.get("no_op"):
            self.input_layernorm = DeciLMRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.self_attn = (DeciLMLinear(config, "linear_attn") if attn.get("replace_with_linear")
                              else DeciLMAttention(config, attn, layer_idx))
        if not ffn.get("no_op"):
            self.post_attention_layernorm = DeciLMRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.mlp = (DeciLMLinear(config, "linear_mlp") if ffn.get("replace_with_linear")
                        else DeciLMMLP(config, ffn))

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        attn, ffn = self.block_config["attention"], self.block_config["ffn"]
        if self.training and any(p.get("sparsify") is not None for p in (attn, ffn)):
            raise NotImplementedError("Sparse NAS checkpoints support inference, not training")
        if hasattr(self, "self_attn"):
            value = self.input_layernorm(hidden_states)
            if attn.get("replace_with_linear"):
                value = self.self_attn(value)
            else:
                value, _ = self.self_attn(
                    value, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_values=past_key_values, **kwargs,
                )
            hidden_states = hidden_states + value
        if hasattr(self, "mlp"):
            hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class DeciLMPreTrainedModel(PreTrainedModel):
    config_class = DeciLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeciLMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flash_attn = False
    _supports_attention_backend = False
    _can_record_outputs = {"hidden_states": DeciLMDecoderLayer, "attentions": DeciLMAttention}
    # TF5's default _init_weights covers Linear, padded Embedding and DeciLMRMSNorm.


class DeciLMModel(DeciLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([DeciLMDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = DeciLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, DeciLMCache):
            raise TypeError("DeciLM requires its package-owned DeciLMCache")
        cache = past_key_values
        if cache is None and use_cache:
            cache = DeciLMCache(self.config)
        # Publish no KV updates until every layer and the final norm complete.
        working = cache.fork() if cache is not None else None
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if position_ids is None:
            seen = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(seen, seen + x.shape[1], device=x.device)[None]
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, position_ids=position_ids, past_key_values=working, **kwargs)
        x = self.norm(x)
        if use_cache:
            cache.layers = working.layers
        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=cache if use_cache else None)


class DeciLMForCausalLM(DeciLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = DeciLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        if generation_config.cache_implementation not in (None, "dynamic"):
            raise ValueError("DeciLM supports its dynamic heterogeneous cache only")
        if generation_config.use_cache and model_kwargs.get("past_key_values") is None:
            model_kwargs["past_key_values"] = DeciLMCache(self.config)

    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        # The decoder's transaction ends before LM projection/loss. Stage its
        # publication too, so a failure anywhere in this forward leaves the
        # caller's populated cache (including sinks) intact.
        caller_cache = past_key_values if isinstance(past_key_values, DeciLMCache) else None
        working_cache = caller_cache.fork() if caller_cache is not None else past_key_values
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=working_cache, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
        )
        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        logits = project(self.lm_head, hidden_states, self.config.pretraining_tp).float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        publish = caller_cache is not None and outputs.past_key_values is not None
        if publish:
            caller_cache.layers = outputs.past_key_values.layers
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=caller_cache if publish else outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
