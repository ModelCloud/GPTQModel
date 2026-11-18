# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright 2025 ByteDance and the Ouro authors.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Transformers 5 Ouro port: Qwen2 blocks with sandwich norms, a shared recurrent stack and early exit."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ..native.qwen2.modeling_qwen2 import (
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_ouro import OuroConfig


@dataclass
class OuroModelOutputWithPast(BaseModelOutputWithPast):
    """Base output plus the normalized hidden state and early-exit gate logits of every recurrent step."""

    step_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    gate_logits: tuple[torch.FloatTensor, ...] | None = None


class UniversalTransformerCache(DynamicCache):
    """One full-history KV slot per recurrent step and physical layer."""

    def __init__(self, max_cache_size=None):
        super().__init__()
        self.max_cache_size = max_cache_size

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if layer_idx < 0:
            raise ValueError("layer_idx must be non-negative")
        if self.max_cache_size is not None and layer_idx >= self.max_cache_size:
            raise IndexError("Cache index exceeds Ouro recurrent depth")
        if key_states.shape != value_states.shape:
            raise ValueError("Ouro cached keys and values must have the same shape")
        if layer_idx < len(self.layers) and self.layers[layer_idx].is_initialized:
            old = self.layers[layer_idx].keys
            if old.shape[:2] != key_states.shape[:2] or old.shape[-1] != key_states.shape[-1]:
                raise ValueError("Cached batch, heads and head_dim must match incoming keys")
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    def validate_depth(self, depth):
        if getattr(self, "max_cache_size", depth) not in (None, depth):
            raise ValueError("Cannot reuse Ouro cache across different recurrent depths")
        if len(self.layers) > depth:
            raise ValueError("Ouro cache contains unexpected recurrent slots")
        lengths = [self.get_seq_length(i) for i in range(depth)]
        if len(set(lengths)) > 1:
            raise ValueError("Incomplete Ouro cache; clear it or retry with a fresh cache")

    def clear(self):
        self.layers = []


class OuroAttention(nn.Module):
    """Qwen2 attention without q/k/v bias; KV slot = recurrent step * depth + layer."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        current_ut=0,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, current_ut * self.config.num_hidden_layers + self.layer_idx)
        # Masks are built for config._attn_implementation, so the kernel must match it.
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        result, weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        return self.o_proj(result.reshape(*input_shape, -1).contiguous()), weights


class OuroDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = OuroAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        for name in ("input_layernorm", "input_layernorm_2", "post_attention_layernorm", "post_attention_layernorm_2"):
            setattr(self, name, Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps))
        self.attention_type = config.layer_types[layer_idx]

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        residual = hidden_states
        attention, _ = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, attention_mask=attention_mask, **kwargs
        )
        hidden_states = residual + self.input_layernorm_2(attention)
        hidden_states = hidden_states + self.post_attention_layernorm_2(
            self.mlp(self.post_attention_layernorm(hidden_states))
        )
        return hidden_states


class OuroPreTrainedModel(PreTrainedModel):
    config_class = OuroConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OuroDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _can_record_outputs = {"hidden_states": OuroDecoderLayer, "attentions": OuroAttention}


class OuroModel(OuroPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([OuroDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config)
        self.total_ut_steps = config.total_ut_steps
        self.early_exit_gate = nn.Linear(config.hidden_size, 1)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if past_key_values is not None and (
            not isinstance(past_key_values, DynamicCache)
            or past_key_values.offloading
            or any(type(layer) is not DynamicLayer for layer in past_key_values.layers)
        ):
            raise ValueError("Ouro requires a full-history dynamic recurrent cache")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if use_cache:
            depth = len(self.layers) * self.total_ut_steps
            if past_key_values is None:
                past_key_values = UniversalTransformerCache(depth)
            if past_key_values.layer_class_to_replicate is None and len(past_key_values.layers) < depth:
                if past_key_values.get_seq_length():
                    raise ValueError("The supplied cache cannot store all Ouro recurrent slots")
                past_key_values = UniversalTransformerCache(depth)
            UniversalTransformerCache.validate_depth(past_key_values, depth)
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)
        if isinstance(attention_mask, dict):
            masks = attention_mask
        else:
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "allow_is_causal_skip": False,
            }
            masks = {"full_attention": create_causal_mask(**mask_kwargs)}
            if "sliding_attention" in self.config.layer_types:
                masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        step_hidden, gates = [], []
        for current_ut in range(self.total_ut_steps):
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=masks[layer.attention_type],
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    current_ut=current_ut,
                    **kwargs,
                )
            hidden_states = self.norm(hidden_states)
            step_hidden.append(hidden_states)
            gates.append(self.early_exit_gate(hidden_states))
        return OuroModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            step_hidden_states=tuple(step_hidden),
            gate_logits=tuple(gates),
        )


class OuroForCausalLM(OuroPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = OuroModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.early_exit_step = getattr(config, "early_exit_step", None)
        self.early_exit_threshold = config.early_exit_threshold
        self.post_init()

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    @can_return_tuple
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
        use_weighted_exit=False,
        exit_at_step=None,
        exit_threshold=None,
        **kwargs,
    ):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        hidden, gates = output.step_hidden_states, output.gate_logits
        exit_at_step = self.early_exit_step if exit_at_step is None else exit_at_step
        exit_threshold = self.early_exit_threshold if exit_threshold is None else exit_threshold
        remaining = torch.ones_like(gates[0].squeeze(-1))
        probabilities = []
        for i, gate in enumerate(gates):
            rate = gate.squeeze(-1).sigmoid()
            probabilities.append(rate * remaining if i < len(gates) - 1 else remaining)
            remaining = remaining * (1.0 - rate)
        pdf = torch.stack(probabilities, dim=2)
        valid_exit_step = exit_at_step is not None and 0 <= exit_at_step < len(hidden)
        if labels is not None or (not valid_exit_step and exit_threshold is None and use_weighted_exit):
            logits = None
            for i, h in enumerate(hidden):
                step_logits = self.lm_head(slice_logits_to_keep(h, logits_to_keep))
                step_pdf = slice_logits_to_keep(pdf[..., i, None], logits_to_keep)
                weighted = step_logits * step_pdf.to(step_logits.dtype)
                logits = weighted if logits is None else logits + weighted
        else:
            selected = hidden[-1]
            if valid_exit_step:
                selected = hidden[exit_at_step]
            elif exit_threshold is not None:
                passed = pdf.cumsum(2) >= torch.as_tensor(exit_threshold, device=pdf.device)
                step = passed.float().argmax(2)
                step = torch.where(passed.any(2), step, len(hidden) - 1)
                states = torch.stack(hidden, dim=2)
                selected = states.gather(2, step[..., None, None].expand(-1, -1, 1, states.shape[-1])).squeeze(2)
            logits = self.lm_head(slice_logits_to_keep(selected, logits_to_keep))
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


__all__ = [
    "OuroPreTrainedModel",
    "OuroModel",
    "OuroModelOutputWithPast",
    "OuroForCausalLM",
    "UniversalTransformerCache",
]
