# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""InternLM 1 on Transformers 5, preserving its unrotated-key NTK cache.

Source and adaptation notes are in README.md. InternLM 1 re-rotates ALL cached
keys when the dynamic NTK base changes. A Llama cache of already rotated keys
does not implement that computation.
"""

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import can_return_tuple

from .configuration_internlm import InternLMConfig


class InternLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.float().square().mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(self.weight.dtype)


class InternLMRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.head_dim
        self.base = config.rotary["base"]
        self.dynamic = config.rotary["type"] == "dynamic"
        self.scaling_factor = config.rotary.get("scaling_factor", 1.0)
        self.max_position_embeddings = config.max_position_embeddings
        # The pinned original checkpoint owns this tensor, although the newer
        # upstream Python marks its corresponding buffer non-persistent.
        self.register_buffer("inv_freq", self.initial_frequencies())

    def initial_frequencies(self, device=None):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))

    def forward(self, x, position_ids, seq_len):
        base = self.base
        if self.dynamic and seq_len > self.max_position_embeddings:
            base *= (self.scaling_factor * seq_len / self.max_position_embeddings
                     - (self.scaling_factor - 1)) ** (self.dim / (self.dim - 2))
        # Call-local tables: independent requests cannot change one another's
        # NTK base, and meta construction leaves no uninitialized rope buffers.
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        if self.dynamic and seq_len > self.max_position_embeddings:
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
        freqs = position_ids.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
        angles = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)


def _rotate(x, cos, sin):
    left, right = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-right, left), dim=-1) * sin


class InternLMCache(DynamicCache):
    """Raw keys plus their positions; both follow beam/crop/reset operations."""

    def __init__(self, config):
        super().__init__(config=config)
        self.positions = {}

    def append(self, keys, values, positions, layer_idx, update):
        positions = positions.expand(keys.shape[0], -1)
        if self.get_seq_length(layer_idx):
            positions = torch.cat((self.positions[layer_idx], positions), dim=-1)
        if update:
            keys, values = super().update(keys, values, layer_idx)
            self.positions[layer_idx] = positions
        elif self.get_seq_length(layer_idx):
            keys = torch.cat((self.layers[layer_idx].keys, keys), dim=-2)
            values = torch.cat((self.layers[layer_idx].values, values), dim=-2)
        return keys, values, positions

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        self.positions = {i: p.index_select(0, beam_idx.to(p.device)) for i, p in self.positions.items()}

    def batch_repeat_interleave(self, repeats):
        super().batch_repeat_interleave(repeats)
        self.positions = {i: p.repeat_interleave(repeats, dim=0) for i, p in self.positions.items()}

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self.positions = {i: p.index_select(0, indices.to(p.device)) for i, p in self.positions.items()}

    def crop(self, tokens_to_remove):
        super().crop(tokens_to_remove)
        self.positions = {i: p[:, :self.get_seq_length(i)] for i, p in self.positions.items()}

    def reset(self):
        super().reset()
        self.positions.clear()


class InternLMAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias))
        self.rotary_emb = InternLMRotaryEmbedding(config)

    def forward(self, hidden_states, attention_mask, position_ids, past_key_values=None,
                use_cache=False, output_attentions=False):
        batch, length, width = hidden_states.shape
        def project(layer):
            return layer(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        query, key, value = project(self.q_proj), project(self.k_proj), project(self.v_proj)
        key_positions = position_ids
        if past_key_values is not None:
            key, value, key_positions = past_key_values.append(
                key, value, position_ids, self.layer_idx, update=use_cache,
            )
        q_cos, q_sin = self.rotary_emb(query, position_ids, key.shape[-2])
        k_cos, k_sin = self.rotary_emb(key, key_positions, key.shape[-2])
        query, key = _rotate(query, q_cos, q_sin), _rotate(key, k_cos, k_sin)
        weights = None
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            result = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        else:
            weights = query @ key.transpose(-1, -2) / self.head_dim ** 0.5
            weights = (weights + attention_mask).clamp_min(torch.finfo(weights.dtype).min)
            weights = weights.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
            result = weights @ value
        result = self.o_proj(result.transpose(1, 2).reshape(batch, length, width))
        return result, weights if output_attentions else None


class InternLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class InternLMDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = InternLMAttention(config, layer_idx)
        self.mlp = InternLMMLP(config)
        self.input_layernorm = InternLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = InternLMRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, position_ids, past_key_values=None,
                use_cache=False, output_attentions=False):
        residual = hidden_states
        hidden_states, weights = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, position_ids,
            past_key_values, use_cache, output_attentions,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, weights


class InternLMPreTrainedModel(PreTrainedModel):
    config_class = InternLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["InternLMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        if isinstance(module, InternLMRotaryEmbedding):
            with torch.no_grad():
                module.inv_freq.copy_(module.initial_frequencies(device=module.inv_freq.device))
        else:
            super()._init_weights(module)


class InternLMModel(InternLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([InternLMDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = InternLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if self.gradient_checkpointing and self.training:
            use_cache = False
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if past_key_values is not None and not isinstance(past_key_values, InternLMCache):
            if type(past_key_values) is DynamicCache and past_key_values.get_seq_length() == 0:
                # generate() allocates an empty standard cache; no keys are lost.
                past_key_values = None
            else:
                raise ValueError("InternLM requires its raw-key InternLMCache; static/quantized/foreign caches are unsupported")
        if use_cache and past_key_values is None:
            past_key_values = InternLMCache(self.config)
        batch, length = inputs_embeds.shape[:2]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        total_length = past_length + length
        if position_ids is None:
            position_ids = torch.arange(past_length, total_length, device=inputs_embeds.device).unsqueeze(0)
        if position_ids.ndim != 2 or position_ids.shape[-1] != length or position_ids.shape[0] not in (1, batch):
            raise ValueError("InternLM position_ids must have shape (1 or batch_size, query_length)")
        query_indices = torch.arange(past_length, total_length, device=inputs_embeds.device)
        key_indices = torch.arange(total_length, device=inputs_embeds.device)
        allowed = key_indices[None, :] <= query_indices[:, None]
        mask = torch.zeros((batch, 1, length, total_length), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        mask.masked_fill_(~allowed, torch.finfo(mask.dtype).min)
        if attention_mask is not None:
            if attention_mask.ndim == 2 and attention_mask.shape == (batch, total_length):
                mask.masked_fill_(~attention_mask[:, None, None, :].to(device=mask.device, dtype=torch.bool),
                                  torch.finfo(mask.dtype).min)
            elif attention_mask.ndim == 4 and attention_mask.shape == mask.shape:
                mask = (mask + attention_mask.to(mask)).clamp_min(torch.finfo(mask.dtype).min)
            else:
                raise ValueError("InternLM attention_mask must cover the complete key sequence")
        hidden_states = inputs_embeds
        all_hidden = () if output_hidden_states else None
        all_attn = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden += (hidden_states,)
            hidden_states, weights = layer(hidden_states, attention_mask=mask, position_ids=position_ids,
                                           past_key_values=past_key_values,
                                           use_cache=use_cache, output_attentions=output_attentions)
            if output_attentions:
                all_attn += (weights,)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden += (hidden_states,)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=past_key_values if use_cache else None,
                                       hidden_states=all_hidden, attentions=all_attn)


class InternLMForCausalLM(InternLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, output_attentions=None, output_hidden_states=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                             output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                             return_dict=True)
        selection = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, selection, :]).float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                     hidden_states=outputs.hidden_states, attentions=outputs.attentions)
