# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Modifications Copyright (c) 2026 ModelCloud.ai.
# Licensed under the FAIR Noncommercial Research License; see LICENSE and NOTICE.
#
# Derived from facebookresearch/MobileLLM utils/modeling_llama.py.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# The upstream implementation also includes Apache-2.0 licensed Transformers
# code; see the repository root LICENSE (Apache-2.0) and NOTICE.

from contextlib import contextmanager
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, slice_logits_to_keep
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_mobilellm import MobileLLMConfig


@contextmanager
def _restore_forward_state_on_error(cache, decoder_layers):
    # DynamicLayer.update concatenates into replacement tensors; retain references,
    # not copies of the full KV storage. Newly appended slots must disappear too.
    layers = list(cache.layers) if isinstance(cache, DynamicCache) else []
    cache_states = [vars(layer).copy() for layer in layers]
    rotary_states = [(layer.self_attn.rotary_emb, layer.self_attn.rotary_emb._base,
                      layer.self_attn.rotary_emb.inv_freq) for layer in decoder_layers]
    try:
        yield
    except Exception:
        if isinstance(cache, DynamicCache):
            cache.layers[:] = layers
            for layer, state in zip(layers, cache_states):
                vars(layer).clear()
                vars(layer).update(state)
        for rotary, base, frequencies in rotary_states:
            rotary._base = base
            rotary.inv_freq = frequencies
        raise


class MobileLLMRMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = eps

    def forward(self, x):
        wide = x.float()
        return self.weight * (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True)
                                               + self.variance_epsilon)).to(x.dtype)


class MobileLLMRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._base = config.rope_theta
        inv_freq = 1 / config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                                           / config.head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        config = self.config
        self.inv_freq = 1 / self._base ** (
            torch.arange(0, config.head_dim, 2, device=self.inv_freq.device, dtype=torch.float32) / config.head_dim)
        return self

    @torch.no_grad()
    def forward(self, x, position_ids):
        config = self.config
        scaling = config.rope_parameters
        kind, factor = scaling.get("rope_type"), scaling.get("factor", 1.0)
        inv_freq = self.inv_freq.to(x.device).float()
        if kind == "linear":
            position_ids = position_ids.float() / factor
        elif kind == "dynamic" and position_ids.max() + 1 > config.max_position_embeddings:
            length = position_ids.max() + 1
            base = config.rope_theta * (factor * length / config.max_position_embeddings - (factor - 1)) ** (
                config.head_dim / (config.head_dim - 2))
            inv_freq = 1 / base ** (torch.arange(0, config.head_dim, 2, device=x.device, dtype=torch.float32)
                                   / config.head_dim)
            self._base = base.item()
            self.inv_freq = inv_freq
        device_type = x.device.type if x.device.type not in {"mps", "meta"} else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            angles = position_ids.float().unsqueeze(-1) * inv_freq
            angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
            return angles.cos().to(x.dtype), angles.sin().to(x.dtype)


def _rotate(x, cos, sin):
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), dim=-1) * sin


def _column_parallel(x, module, shards):
    return torch.cat([F.linear(x, weight) for weight in module.weight.chunk(shards, dim=0)], dim=-1)


def _row_parallel(x, module, shards):
    return sum(F.linear(value, weight) for value, weight in
               zip(x.chunk(shards, dim=-1), module.weight.chunk(shards, dim=1)))


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    weights = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights


class MobileLLMAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_causal = True
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        kv_width = config.num_key_value_heads * config.head_dim
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.rotary_emb = MobileLLMRotaryEmbedding(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                cache_idx=None, **kwargs):
        config = self.config
        batch, length, _ = hidden_states.shape
        # Source Llama slices projections for pretraining_tp everywhere except the SDPA attention class.
        split = config.pretraining_tp > 1 and config._attn_implementation != "sdpa"
        project = (lambda module: _column_parallel(hidden_states, module, config.pretraining_tp)) if split else (
            lambda module: module(hidden_states))
        q = project(self.q_proj).view(batch, length, config.num_attention_heads, config.head_dim).transpose(1, 2)
        k = project(self.k_proj).view(batch, length, config.num_key_value_heads, config.head_dim).transpose(1, 2)
        v = project(self.v_proj).view(batch, length, config.num_key_value_heads, config.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx if cache_idx is None else cache_idx)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation,
                                                                    eager_attention_forward)
        output, weights = attention_interface(
            self, q, k, v, attention_mask,
            dropout=config.attention_dropout if self.training else 0.0, scaling=self.scaling, **kwargs,
        )
        output = output.reshape(batch, length, config.hidden_size)
        output = _row_parallel(output, self.o_proj, config.pretraining_tp) if split else self.o_proj(output)
        return output, weights


class MobileLLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            shards = self.config.pretraining_tp
            gate = _column_parallel(x, self.gate_proj, shards)
            up = _column_parallel(x, self.up_proj, shards)
            return _row_parallel(self.act_fn(gate) * up, self.down_proj, shards)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MobileLLMDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = MobileLLMAttention(config, layer_idx)
        self.mlp = MobileLLMMLP(config)
        self.input_layernorm = MobileLLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MobileLLMRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, **kwargs):
        attention, _ = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        hidden_states = hidden_states + attention
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class MobileLLMPreTrainedModel(PreTrainedModel):
    config_class = MobileLLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MobileLLMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": MobileLLMDecoderLayer, "attentions": MobileLLMAttention}
    _keep_in_fp32_modules_strict = ["inv_freq"]

    @torch.no_grad()
    def _init_weights(self, module):
        # transformers.initialization skips tensors already loaded from a
        # checkpoint (`_is_hf_initialized`); raw `.data` writes would not.
        super()._init_weights(module)
        if isinstance(module, MobileLLMRotaryEmbedding):
            config = module.config
            init.copy_(module.inv_freq, 1 / config.rope_theta ** (
                torch.arange(0, config.head_dim, 2, device=module.inv_freq.device, dtype=torch.float32) / config.head_dim))


class MobileLLMModel(MobileLLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens: nn.Module = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([MobileLLMDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = MobileLLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        self.embed_tokens = value

    @merge_with_config_defaults
    @capture_outputs
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise ValueError("MobileLLM supports DynamicCache only")
        repetitions = 2 if self.config.layer_sharing else 1
        if (past_key_values is not None and past_key_values.layers
                and getattr(past_key_values, "layer_class_to_replicate", None) is None
                and len(past_key_values.layers) != self.config.num_hidden_layers * repetitions):
            raise ValueError("MobileLLM cache must have one slot for each decoder use")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        length = hidden_states.shape[1]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if attention_mask is not None and (attention_mask.ndim != 2 or
                                          attention_mask.shape != (hidden_states.shape[0], past_length + length)):
            raise ValueError("MobileLLM attention_mask must be [batch, past + current tokens]")
        if use_cache and past_key_values is None:
            # Layer sharing needs one cache slot per decoder use, so the cache grows lazily.
            past_key_values = DynamicCache()
        if position_ids is None:
            position_ids = torch.arange(past_length, past_length + length, device=hidden_states.device).unsqueeze(0)
        mask = create_causal_mask(config=self.config, inputs_embeds=hidden_states, attention_mask=attention_mask,
                                  past_key_values=past_key_values, position_ids=position_ids)
        with _restore_forward_state_on_error(past_key_values, self.layers):
            for index, layer in enumerate(self.layers):
                for repeat in range(repetitions):
                    hidden_states = layer(
                        hidden_states, attention_mask=mask, position_ids=position_ids,
                        past_key_values=past_key_values, cache_idx=index * repetitions + repeat, **kwargs,
                    )
            return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states),
                                           past_key_values=past_key_values if use_cache else None)


class MobileLLMForCausalLM(MobileLLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    @classmethod
    def _supports_default_dynamic_cache(cls):
        # Generation's physical-layer config cannot describe both shared uses.
        return False

    def __init__(self, config):
        super().__init__(config)
        self.model = MobileLLMModel(config)
        if config.share_embedding:
            self._tied_weights_keys = {}
        if not config.share_embedding:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.model.embed_tokens if self.config.share_embedding else self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if self.config.share_embedding:
            self.set_input_embeddings(new_embeddings)
        else:
            self.lm_head = new_embeddings

    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        with _restore_forward_state_on_error(past_key_values, self.model.layers):
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                                **kwargs)
            states = slice_logits_to_keep(output.last_hidden_state, logits_to_keep)
            if self.config.pretraining_tp > 1:
                logits = _column_parallel(states, self.get_output_embeddings(), self.config.pretraining_tp).float()
            else:
                logits = (F.linear(states, cast(torch.Tensor, self.model.embed_tokens.weight)) if self.config.share_embedding
                          else self.lm_head(states)).float()
            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=output.past_key_values,
                                          hidden_states=output.hidden_states, attentions=output.attentions)
