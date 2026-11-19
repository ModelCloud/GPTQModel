# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/instella/SPEC.md; see NOTICE.
"""Instella decoder-only causal LM on the Transformers 5.23 ABI.

Standard pieces (RMSNorm, rotate-half RoPE application, ``repeat_kv`` and the
SwiGLU MLP) come from the vendored Apache-2.0 Llama/Qwen2 families. The
Instella-specific parts are the pre-norm layer, full-width Q/K RMSNorm before
the head split, one stateful rotary module per attention layer (default,
linear and dynamic-NTK), and an eager kernel that divides scores by sqrt(D).
"""

import math
from collections.abc import Callable

import torch
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ..native.llama.modeling_llama import LlamaRMSNorm, apply_rotary_pos_emb, repeat_kv
from ..native.qwen2.modeling_qwen2 import Qwen2MLP
from .configuration_instella import InstellaConfig


class InstellaRotaryEmbedding(nn.Module):
    """Per-layer rotary module with default, linear and stateful dynamic-NTK scaling."""

    def __init__(self, config: InstellaConfig):
        super().__init__()
        parameters = config.rope_parameters
        self.rope_type = parameters["rope_type"]
        self.base = parameters["rope_theta"]
        self.scaling_factor = parameters.get("factor", 1.0)
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.register_buffer("inv_freq", self.default_inv_freq(), persistent=False)

    def default_inv_freq(self, device=None) -> torch.Tensor:
        exponent = torch.arange(0, self.dim, 2, device=device).float() / self.dim
        return 1.0 / (self.base ** exponent)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq
        positions = position_ids.float()
        if self.rope_type == "linear":
            positions = positions / self.scaling_factor
        elif self.rope_type == "dynamic":
            length = torch.max(position_ids) + 1
            if length > self.max_position_embeddings:
                factor = self.scaling_factor
                base = self.base * (
                    (factor * length) / self.max_position_embeddings - (factor - 1)
                ) ** (self.dim / (self.dim - 2))
                exponent = torch.arange(0, self.dim, 2, device=x.device).float() / self.dim
                new_inv_freq = 1.0 / (base ** exponent)
                self.inv_freq = new_inv_freq
                inv_freq = new_inv_freq

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        positions_expanded = positions[:, None, :].to(x.device)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ positions_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # Instella divides by sqrt(D) rather than multiplying by `scaling`; the two differ in the last bit.
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class InstellaAttention(nn.Module):
    def __init__(self, config: InstellaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        kv_width = config.num_key_value_heads * self.head_dim
        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=bias)
        # Q/K norms span the full projected width, applied before the head split.
        self.q_norm = LlamaRMSNorm(config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(kv_width, eps=config.rms_norm_eps)
        self.rotary_emb = InstellaRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_ids is None:
            past_seen = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
            position_ids = (torch.arange(input_shape[-1], device=hidden_states.device) + past_seen).unsqueeze(0)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class InstellaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: InstellaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pre_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = InstellaAttention(config=config, layer_idx=layer_idx)
        self.pre_feedforward_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            hidden_states=self.pre_attention_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return hidden_states


class InstellaPreTrainedModel(PreTrainedModel):
    config: InstellaConfig
    config_class = InstellaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InstellaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": InstellaDecoderLayer,
        "attentions": InstellaAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, InstellaRotaryEmbedding):
            # Non-persistent buffers are materialized uninitialized; always restore the cold frequencies.
            init.copy_(module.inv_freq, module.default_inv_freq(device=module.inv_freq.device))


class InstellaModel(InstellaPreTrainedModel):
    def __init__(self, config: InstellaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [InstellaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError("Instella past_key_values must be a Transformers Cache instance")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class InstellaForCausalLM(InstellaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: InstellaConfig):
        super().__init__(config)
        self.model = InstellaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["InstellaForCausalLM", "InstellaModel", "InstellaPreTrainedModel"]
