# Copyright 2021 The LG AI Research EXAONE Lab.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 The LG AI Research and HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Based on the GPT-NeoX and OPT implementations; see NOTICE.
"""EXAONE 3.0/3.5 decoder with package-owned modeling and Transformers 5 caches."""

import math
from functools import wraps

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, StaticLayer
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from .configuration_exaone import ExaoneConfig


def _rollback_cache_on_error(forward):
    @wraps(forward)
    def wrapped(self, *args, **kwargs):
        cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
        inputs = kwargs.get("input_ids", args[0] if args else None)
        if inputs is None:
            inputs = kwargs.get("inputs_embeds", args[4] if len(args) > 4 else None)
        if not isinstance(cache, Cache) or inputs is None:
            return forward(self, *args, **kwargs)
        layers = list(cache.layers)
        if getattr(cache, "offloading", False) or any(type(layer) not in (DynamicLayer, StaticLayer) for layer in layers):
            raise ValueError("EXAONE supports non-offloaded DynamicCache and StaticCache layers only")
        snapshots = []
        for layer in layers:
            window = counter = None
            if isinstance(layer, StaticLayer):
                counter = layer.cumulative_length.clone()
                if layer.is_initialized:
                    assert layer.keys is not None and layer.values is not None
                    start = int(layer.get_seq_length())
                    end = start + inputs.shape[1]
                    # Copy only this call's static write window, never the full KV history.
                    window = (start, end, layer.keys[:, :, start:end].clone(), layer.values[:, :, start:end].clone())
            snapshots.append((layer, vars(layer).copy(), window, counter))
        try:
            return forward(self, *args, **kwargs)
        except BaseException:
            cache.layers[:] = layers
            for layer, state, window, counter in snapshots:
                if window is not None:
                    start, end, keys, values = window
                    state["keys"][:, :, start:end].copy_(keys)
                    state["values"][:, :, start:end].copy_(values)
                if counter is not None:
                    state["cumulative_length"].copy_(counter)
                vars(layer).clear()
                vars(layer).update(state)
            raise
    return wrapped


class ExaoneRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.eps)).to(dtype)


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(hidden_states, repeats):
    batch, heads, length, dim = hidden_states.shape
    return hidden_states[:, :, None].expand(batch, heads, repeats, length, dim).reshape(
        batch, heads * repeats, length, dim,
    )


class ExaoneRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]
        self.register_buffer("inv_freq", self.frequencies()[0], persistent=False)

    def frequencies(self, device=None, seq_len=None):
        config = self.config
        params = config.rope_parameters
        dim = config.head_dim
        base = params["rope_theta"]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        if self.rope_type == "default":
            return inv_freq, 1.0
        if self.rope_type == "yarn":
            # EXAONE 3.0's YaRN uses max_position_embeddings, not the optional
            # original_max_position_embeddings used by newer generic helpers.
            factor = params["factor"]
            attention = params.get("attention_factor", 0.1 * math.log(factor) + 1.0)
            def correction(rotations):
                return dim * math.log(config.max_position_embeddings / (rotations * 2 * math.pi)) / (
                    2 * math.log(base)
                )
            low = max(math.floor(correction(params.get("beta_fast") or 32)), 0)
            high = min(math.ceil(correction(params.get("beta_slow") or 1)), dim - 1)
            if low == high:
                high += 0.001
            ramp = ((torch.arange(dim // 2, device=device).float() - low) / (high - low)).clamp(0, 1)
            return inv_freq / factor * ramp + inv_freq * (1 - ramp), attention
        return ROPE_INIT_FUNCTIONS[self.rope_type](config, device=device, seq_len=seq_len)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Sequence-derived frequencies stay local: an unrelated long request
        # must not change subsequent short requests or another thread's RoPE.
        seq_len = position_ids.max() + 1 if self.rope_type in ("dynamic", "longrope") else None
        inv_freq, scaling = self.frequencies(device=x.device, seq_len=seq_len)
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            angles = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
            angles = torch.cat((angles, angles), dim=-1)
            cos, sin = angles.cos() * scaling, angles.sin() * scaling
        return cos.to(x.dtype), sin.to(x.dtype)


class ExaoneSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None,
                past_key_value=None, output_attentions=False, **kwargs):
        if position_embeddings is None:
            raise ValueError("EXAONE attention requires rotary position_embeddings")
        batch, length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)
        k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
        dropout = self.config.attention_dropout if self.training else 0.0
        weights = None
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=dropout)
        else:
            weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                weights = weights + attention_mask[:, :, :, :k.shape[-2]]
            weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
            weights = F.dropout(weights, p=dropout, training=self.training)
            output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.out_proj(output), weights if output_attentions else None


class ExaoneAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.attention = ExaoneSelfAttention(config, layer_id)

    def forward(self, hidden_states, **kwargs):
        return self.attention(hidden_states, **kwargs)


class ExaoneGatedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc_0 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.c_fc_1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.activation_function]

    def forward(self, hidden_states):
        return self.c_proj(self.act(self.c_fc_0(hidden_states)) * self.c_fc_1(hidden_states))


class ExaoneBlock(GradientCheckpointingLayer):
    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = ExaoneRMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.attn = ExaoneAttention(config, layer_id)
        self.ln_2 = ExaoneRMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.mlp = ExaoneGatedMLP(config)

    def forward(self, hidden_states, **kwargs):
        attention, weights = self.attn(self.ln_1(hidden_states), **kwargs)
        hidden_states = hidden_states + attention
        return hidden_states + self.mlp(self.ln_2(hidden_states)), weights


class ExaonePreTrainedModel(PreTrainedModel):
    config_class = ExaoneConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ExaoneBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, ExaoneRMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, ExaoneRotaryEmbedding):
            module.inv_freq = module.frequencies(device=module.inv_freq.device)[0]


class ExaoneModel(ExaonePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.drop = nn.Dropout(float(config.embed_dropout))
        self.h = nn.ModuleList([ExaoneBlock(config, i) for i in range(config.num_layers)])
        self.ln_f = ExaoneRMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.rotary = ExaoneRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    @_rollback_cache_on_error
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, cache_position=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError("EXAONE requires a Transformers 5 Cache, not legacy tuple past_key_values")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if self.gradient_checkpointing and self.training:
            use_cache, past_key_values = False, None
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        if attention_mask is not None:
            batch, length = inputs_embeds.shape[:2]
            key_length = seen + length
            if attention_mask.ndim == 2:
                valid = attention_mask.shape == (batch, key_length)
            elif attention_mask.ndim == 4:
                key_length = past_key_values.get_mask_sizes(length, 0)[0] if past_key_values is not None else key_length
                valid = (attention_mask.shape[0] in (1, batch)
                         and attention_mask.shape[1] in (1, self.config.num_attention_heads)
                         and attention_mask.shape[-2:] == (length, key_length))
            else:
                valid = False
            if not valid:
                raise ValueError("EXAONE attention_mask must cover the complete key sequence and current queries")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            if cache_position is None:
                cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + seen
            position_ids = cache_position.unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids,
            allow_is_causal_skip=False,
        )
        hidden_states = self.drop(inputs_embeds)
        position_embeddings = self.rotary(hidden_states, position_ids)
        all_hidden = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for block in self.h:
            if all_hidden is not None:
                all_hidden += (hidden_states,)
            hidden_states, weights = block(
                hidden_states, attention_mask=causal_mask, position_embeddings=position_embeddings,
                past_key_value=past_key_values, output_attentions=output_attentions,
            )
            if all_attentions is not None:
                all_attentions += (weights,)
        hidden_states = self.ln_f(hidden_states)
        if all_hidden is not None:
            all_hidden += (hidden_states,)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden, attentions=all_attentions,
        )
        return output if return_dict else output.to_tuple()


class ExaoneForCausalLM(ExaonePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ExaoneModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @_rollback_cache_on_error
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, cache_position=None,
                logits_to_keep=0, **kwargs):
        return_dict = self.config.return_dict if return_dict is None else return_dict
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            cache_position=cache_position, return_dict=True,
        )
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, indices]).float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            logits = logits.to(outputs.last_hidden_state.dtype)
            loss = loss.to(outputs.last_hidden_state.dtype)
        output = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
        return output if return_dict else output.to_tuple()
