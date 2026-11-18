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
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

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


class DeciLMCache(Cache):
    """Caller-owned heterogeneous cache, including stacks whose first block is pruned.

    Global/sliding/unshifted attention retains full KV history and masks it at
    consumption. Shifted sink attention retains sink+window KV and re-rotates
    retained keys exactly as the upstream SinkCache. Updates are transactional
    at model-forward scope so a failed later layer leaves a reusable cache.
    """

    def __init__(self, config):
        super().__init__(layers=[DynamicLayer() for _ in range(config.num_hidden_layers)])
        self.blocks = [copy.deepcopy(config.layer_config(i)) for i in range(config.num_hidden_layers)]
        self.seen_tokens = 0

    def get_seq_length(self, layer_idx=0):
        return self.seen_tokens

    @property
    def seen_tokens(self):
        return self._seen_tokens

    @seen_tokens.setter
    def seen_tokens(self, value):
        self._seen_tokens = value

    def fork(self):
        other = copy.copy(self)
        other.layers = [copy.copy(layer) for layer in self.layers]
        return other

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        layer = self.layers[layer_idx]
        attn = self.blocks[layer_idx]["attention"]
        window, sinks = attn.get("window_length"), attn.get("num_sink_tokens")
        if window is None or sinks is None or attn.get("unshifted_sink"):
            return layer.update(key_states, value_states)
        cos, sin = cache_kwargs["cos"][0], cache_kwargs["sin"][0]
        if not hasattr(layer, "sink_cos"):
            layer.sink_cos, layer.sink_sin = cos, sin
        elif layer.sink_cos.shape[0] < window:
            layer.sink_cos = torch.cat((layer.sink_cos, cos), dim=0)
            layer.sink_sin = torch.cat((layer.sink_sin, sin), dim=0)
        count = key_states.shape[-2]
        old_count = layer.get_seq_length()
        if not old_count or old_count + count < window:
            return layer.update(key_states, value_states)
        if count >= window - sinks:
            raise ValueError("Shifted-sink continuation chunks must be smaller than window_length - num_sink_tokens")
        # Keep fixed sink keys and the most recent window-sinks-count keys.
        keep = window - sinks - count
        keys = layer.keys[..., -keep:, :] if keep else layer.keys[..., :0, :]
        values = layer.values[..., -keep:, :] if keep else layer.values[..., :0, :]
        # Pinned SinkCache uses the first window's RoPE table to shift retained
        # keys back by the incoming chunk width. Preserve this even when the
        # caller supplies absolute positions after the first eviction.
        old_cos = layer.sink_cos[sinks + count:window].float()[None]
        old_sin = layer.sink_sin[sinks + count:window].float()[None]
        new_cos = layer.sink_cos[sinks:window - count].float()[None]
        new_sin = layer.sink_sin[sinks:window - count].float()[None]
        cos = old_cos.float() * new_cos.float() + old_sin.float() * new_sin.float()
        sin = -old_sin.float() * new_cos.float() + old_cos.float() * new_sin.float()
        keys = keys * cos[:, None].to(keys.dtype) + rotate_half(keys) * sin[:, None].to(keys.dtype)
        layer.keys = torch.cat((layer.keys[..., :sinks, :], keys, key_states), dim=-2)
        layer.values = torch.cat((layer.values[..., :sinks, :], values, value_states), dim=-2)
        return layer.keys, layer.values

    def reset(self):
        self.layers = [DynamicLayer() for _ in self.layers]
        self.seen_tokens = 0

    def crop(self, max_length):
        # Full-history branches can be cropped; shifted sinks discard history.
        if any(b["attention"].get("num_sink_tokens") is not None
               and not b["attention"].get("unshifted_sink") for b in self.blocks):
            raise ValueError("Shifted-sink cache cannot restore discarded history")
        if max_length < 0:
            max_length = self.seen_tokens + max_length
        for layer in self.layers:
            layer.crop(max_length)
        self.seen_tokens = min(self.seen_tokens, max_length)


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
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

    def forward(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions=False, **kwargs):
        batch, length, _ = hidden_states.shape
        tp = self.config.pretraining_tp
        q = project(self.q_proj, hidden_states, tp).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = project(self.k_proj, hidden_states, tp).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = project(self.v_proj, hidden_states, tp).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        seen = past_key_value.seen_tokens if past_key_value is not None else 0
        cos, sin = rope(self.config, hidden_states, position_ids, max(seen + length, int(position_ids.max()) + 1))
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx, {"cos": cos, "sin": sin})
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        key_length = k.shape[-2]
        # Construct the native causal mask from physical sequence positions,
        # independently of RoPE's per-example positions (left padding).
        query_pos = torch.arange(seen, seen + length, device=q.device)
        key_pos = torch.arange(key_length, device=q.device)
        shifted = self.part.get("num_sink_tokens") is not None and not self.part.get("unshifted_sink")
        if shifted and past_key_value is not None and seen + length > key_length:
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
        weights = None
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.config.attention_dropout if self.training else 0.0,
            )
        else:
            weights = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) + mask,
                                dim=-1, dtype=torch.float32).to(q.dtype)
            weights = F.dropout(weights, p=self.config.attention_dropout, training=self.training)
            output = weights @ v
        output = output.transpose(1, 2).contiguous().view(batch, length, -1)
        return project(self.o_proj, output, tp, output_split=True), weights if output_attentions else None


class DeciLMDecoderLayer(nn.Module):
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

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, **kwargs):
        attn, ffn = self.block_config["attention"], self.block_config["ffn"]
        if self.training and any(p.get("sparsify") is not None for p in (attn, ffn)):
            raise NotImplementedError("Sparse NAS checkpoints support inference, not training")
        weights = None
        if hasattr(self, "self_attn"):
            value = self.input_layernorm(hidden_states)
            if attn.get("replace_with_linear"):
                value = self.self_attn(value)
            else:
                value, weights = self.self_attn(
                    value, attention_mask, position_ids, past_key_value, output_attentions,
                )
            hidden_states = hidden_states + value
        if hasattr(self, "mlp"):
            hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, weights


class DeciLMPreTrainedModel(PreTrainedModel):
    config_class = DeciLMConfig
    base_model_prefix = "model"
    _no_split_modules = ["DeciLMDecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = False
    _supports_attention_backend = False
    # TF5's default _init_weights covers Linear, padded Embedding and DeciLMRMSNorm.


class DeciLMModel(DeciLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([DeciLMDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = DeciLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=False, output_hidden_states=False,
                return_dict=True, cache_position=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(past_key_values, DeciLMCache):
            raise TypeError("DeciLM requires its package-owned DeciLMCache")
        cache = past_key_values
        if cache is None and use_cache:
            cache = DeciLMCache(self.config)
        working = cache.fork() if cache is not None else None
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        seen = cache.seen_tokens if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(seen, seen + x.shape[1], device=x.device)[None]
        states, attentions = (() if output_hidden_states else None), (() if output_attentions else None)
        # Publish no KV updates until layers, final norm and output encoding complete.
        for layer in self.layers:
            if output_hidden_states:
                states += (x,)
            x, weights = layer(
                x, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=working, output_attentions=output_attentions,
            )
            if output_attentions:
                attentions += (weights,)
        x = self.norm(x)
        if output_hidden_states:
            states += (x,)
        result = BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=cache if use_cache else None,
            hidden_states=states, attentions=attentions,
        )
        encoded = result if return_dict else result.to_tuple()
        if use_cache:
            cache.layers = working.layers
            cache.seen_tokens = seen + x.shape[1]
        return encoded


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

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=False,
                output_hidden_states=False, return_dict=True, logits_to_keep=0, **kwargs):
        # The decoder's transaction ends before LM projection/loss/encoding.
        # Stage its publication too, so a failure anywhere in this public
        # forward leaves the caller's populated cache (including sinks) intact.
        caller_cache = past_key_values if isinstance(past_key_values, DeciLMCache) else None
        working_cache = caller_cache.fork() if caller_cache is not None else past_key_values
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=working_cache, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            **kwargs,
        )
        index = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = project(self.lm_head, outputs.last_hidden_state[:, index], self.config.pretraining_tp).float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        publish = caller_cache is not None and outputs.past_key_values is not None
        result = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=caller_cache if publish else outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
        encoded = result if return_dict else result.to_tuple()
        if publish:
            caller_cache.layers = outputs.past_key_values.layers
            caller_cache.seen_tokens = outputs.past_key_values.seen_tokens
        return encoded
