# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

import math
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs

from .configuration_k2_horizon import K2HorizonConfig


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def split_to_interleaved(x):
    # Split halves:  x0 x1 x2 x3 ... y0 y1 y2 y3 ...
    # Interleaved:   x0 y0 x1 y1 x2 y2 x3 y3 ...
    return x.reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def interleaved_to_split(x):
    # Interleaved:   x0 y0 x1 y1 x2 y2 x3 y3 ...
    # Split halves:  x0 x1 x2 x3 ... y0 y1 y2 y3 ...
    return x.reshape(*x.shape[:-1], -1, 2).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def calc_router_weights(
    router_logits: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    score_func: str,
    top_k: int,
    scaling_factor: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native-XLLM-compatible routing weights and selected experts.

    XLLM applies router bias only to the values used for top-k selection. The
    selected routes are still weighted by the original router probabilities,
    then optionally normalized and scaled.
    """
    if score_func == "softmax":
        routing_scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    elif score_func == "sigmoid":
        routing_scores = torch.sigmoid(router_logits.to(torch.float32))
    else:
        raise ValueError(f"Unsupported router score function: {score_func}")

    selection_scores = routing_scores
    if router_bias is not None:
        selection_scores = selection_scores + router_bias.to(selection_scores)

    selected_indices = torch.topk(selection_scores, top_k, dim=-1).indices
    routing_weights = torch.gather(routing_scores, dim=-1, index=selected_indices)
    if top_k > 1:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    if scaling_factor is not None:
        routing_weights = routing_weights * scaling_factor
    return routing_weights, selected_indices


def combine_routed_experts(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_indices: torch.Tensor,
    experts: nn.ModuleList,
    activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    num_tokens, hidden_dim = hidden_states.shape
    final_hidden_states = torch.zeros(
        (num_tokens, experts[0].out_features),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    expert_mask = torch.nn.functional.one_hot(
        selected_indices, num_classes=len(experts)
    ).permute(2, 1, 0)

    for expert_idx in torch.nonzero(expert_mask.sum(dim=(-1, -2)), as_tuple=False).flatten():
        topk_positions, token_positions = torch.where(expert_mask[int(expert_idx)])
        expert_states = experts[int(expert_idx)](hidden_states[token_positions])
        if activation is not None:
            expert_states = activation(expert_states)
        expert_states = expert_states * routing_weights[token_positions, topk_positions, None].to(expert_states.dtype)
        final_hidden_states.index_add_(0, token_positions, expert_states.to(hidden_states.dtype))

    return final_hidden_states


class K2HorizonAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: K2HorizonConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.rope_head_dim = self.head_dim if config.rope_head_dim is None else config.rope_head_dim

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.gate_func = config.attention_gate_func
        if self.gate_func is not None:
            self.gate_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=False)

        if config.query_key_norm:
            self.q_norm = K2HorizonRMSNorm(
                hidden_size=config.num_attention_heads * self.head_dim,
                n_groups=config.num_attention_heads,
                eps=config.rms_norm_eps)
            self.k_norm = K2HorizonRMSNorm(
                hidden_size=config.num_key_value_heads * self.head_dim,
                n_groups=config.num_key_value_heads,
                eps=config.rms_norm_eps)

        self.sliding_window = getattr(config, "sliding_window", None)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.config.query_key_norm:
            query_states = self.q_norm(self.q_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.rope_head_dim == self.head_dim:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, query_states_ = torch.split(
                split_to_interleaved(query_states),
                split_size_or_sections=[self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1)

            key_states, key_states_ = torch.split(
                split_to_interleaved(key_states),
                split_size_or_sections=[self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1)

            query_states, key_states = apply_rotary_pos_emb(
                interleaved_to_split(query_states),
                interleaved_to_split(key_states),
                cos,
                sin)

            query_states = interleaved_to_split(torch.cat(
                [split_to_interleaved(query_states), query_states_], dim=-1))
            key_states = interleaved_to_split(torch.cat(
                [split_to_interleaved(key_states), key_states_], dim=-1))

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        if self.gate_func is not None:
            gate = self.gate_proj(hidden_states).view(
                input_shape + (-1, self.head_dim))
            if self.gate_func == 'silu':
                gate = F.silu(gate)
            else:
                assert self.gate_func == 'softplus'
                gate = F.softplus(gate, beta=math.log(2))

            attn_output = attn_output * gate

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def apply_rotary_pos_emb_xllm(q, k, freqs_cis):
    if q.shape[-1] % 2 != 0 or k.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE dimensions must be even, got q={q.shape[-1]} and k={k.shape[-1]}")
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    if freqs_cis.ndim == 2:
        freqs_cis = freqs_cis.unsqueeze(1)
    elif freqs_cis.ndim == 3:
        freqs_cis = freqs_cis.unsqueeze(2)
    else:
        raise ValueError(f"Unsupported freqs_cis shape: {tuple(freqs_cis.shape)}")
    if freqs_cis.shape[-1] != q_.shape[-1] or freqs_cis.shape[-1] != k_.shape[-1]:
        raise ValueError(
            "RoPE frequency dimension mismatch: "
            f"q_rope_dim={q.shape[-1]}, k_rope_dim={k.shape[-1]}, "
            f"q_complex_dim={q_.shape[-1]}, k_complex_dim={k_.shape[-1]}, "
            f"freqs_complex_dim={freqs_cis.shape[-1]}, freqs_shape={tuple(freqs_cis.shape)}"
        )
    q_embed = torch.view_as_real(q_ * freqs_cis).flatten(-2).to(q.dtype)
    k_embed = torch.view_as_real(k_ * freqs_cis).flatten(-2).to(k.dtype)
    return q_embed, k_embed


class K2HorizonMoVAAttention(nn.Module):
    """MoVA attention with routed value experts and optional post-attention gate."""

    def __init__(self, config: K2HorizonConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.num_experts_per_tok = config.mova_num_experts_per_tok
        self.router_score_func = config.router_score_func
        self.router_scaling_factor = config.router_scaling_factor
        self.gate_func = config.attention_gate_func

        self.rope_head_dim = self.head_dim if config.rope_head_dim is None else config.rope_head_dim

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.v_router = nn.Linear(
            config.hidden_size,
            config.mova_num_experts,
            bias=config.moe_gate_bias)
        self.v_experts = nn.ModuleList([
            nn.Linear(
                config.hidden_size,
                config.num_key_value_heads * self.head_dim,
                bias=False
            ) for _ in range(config.mova_num_experts)
        ])

        if self.gate_func is not None:
            self.gate_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=False)

        if config.query_key_norm:
            self.q_norm = K2HorizonRMSNorm(
                hidden_size=config.num_attention_heads * self.head_dim,
                n_groups=config.num_attention_heads,
                eps=config.rms_norm_eps,
            )
            self.k_norm = K2HorizonRMSNorm(
                hidden_size=config.num_key_value_heads * self.head_dim,
                n_groups=config.num_key_value_heads,
                eps=config.rms_norm_eps,
            )

        self.sliding_window = getattr(config, "sliding_window", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        # Match native MOVAttention router semantics exactly: compute logits with
        # the weight-only linear and apply router bias only to selection scores.
        router_logits = F.linear(flat_hidden_states, self.v_router.weight)
        routing_weights, selected_values = calc_router_weights(
            router_logits=router_logits,
            router_bias=self.v_router.bias,
            score_func=self.router_score_func,
            top_k=self.num_experts_per_tok,
            scaling_factor=self.router_scaling_factor,
        )

        mixed_value_states = combine_routed_experts(
            hidden_states=flat_hidden_states,
            routing_weights=routing_weights,
            selected_indices=selected_values,
            experts=self.v_experts,
            activation=F.silu)

        value_states = mixed_value_states.view(hidden_shape).transpose(1, 2)

        if self.config.query_key_norm:
            query_states = self.q_norm(self.q_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.rope_head_dim == self.head_dim:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, query_states_ = torch.split(
                split_to_interleaved(query_states),
                split_size_or_sections=[self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1)

            key_states, key_states_ = torch.split(
                split_to_interleaved(key_states),
                split_size_or_sections=[self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1)

            query_states, key_states = apply_rotary_pos_emb(
                interleaved_to_split(query_states),
                interleaved_to_split(key_states),
                cos,
                sin)

            query_states = interleaved_to_split(torch.cat(
                [split_to_interleaved(query_states), query_states_], dim=-1))
            key_states = interleaved_to_split(torch.cat(
                [split_to_interleaved(key_states), key_states_], dim=-1))

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        if self.gate_func is not None:
            gate = self.gate_proj(hidden_states).view(input_shape + (-1, self.head_dim))
            if self.gate_func == 'silu':
                gate = F.silu(gate)
            else:
                assert self.gate_func == 'softplus'
                gate = F.softplus(gate, beta=math.log(2))

            attn_output = attn_output * gate

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class K2HorizonMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class K2HorizonSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.num_shared_experts = config.num_shared_experts
        self.router_score_func = config.router_score_func
        self.router_scaling_factor = config.router_scaling_factor

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=config.moe_gate_bias)
        self.experts = nn.ModuleList(
            [K2HorizonMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        if config.num_shared_experts > 0:
            self.shared_experts = K2HorizonMLP(
                config=config,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        residuals = hidden_states

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        # router_logits = self.gate(hidden_states)
        router_logits = F.linear(hidden_states, self.gate.weight)

        if self.router_score_func == "softmax":
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        else:
            assert self.router_score_func == "sigmoid"
            routing_weights = F.sigmoid(router_logits.to(torch.float32))

        routing_weights_for_choice = routing_weights
        if self.gate.bias is not None:
            routing_weights_for_choice = routing_weights + self.gate.bias.to(routing_weights.dtype)

        _, selected_experts = torch.topk(routing_weights_for_choice, self.top_k, dim=-1)
        routing_weights = torch.gather(routing_weights, dim=-1, index=selected_experts)

        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights * self.router_scaling_factor
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if self.num_shared_experts > 0:
            final_hidden_states = final_hidden_states + self.shared_experts(residuals)

        return final_hidden_states, router_logits


# @use_kernel_forward_from_hub("RMSNorm")
class K2HorizonRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, n_groups: int, eps=1e-6):
        """
        K2HorizonRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.n_groups = n_groups
        self.hidden_size = hidden_size
        assert hidden_size % n_groups == 0
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        hidden_states = hidden_states.reshape(*hidden_states.shape[:-1], self.n_groups, -1)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.reshape(*hidden_states.shape[:-2], -1)
        hidden_states = self.weight * hidden_states

        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class K2HorizonDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: K2HorizonConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        is_sparse_layer = (layer_idx not in config.mlp_only_layers) and (
                config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0)

        if is_sparse_layer and config.mova_num_experts > 0:
            self.self_attn = K2HorizonMoVAAttention(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = K2HorizonAttention(config, layer_idx)

        if is_sparse_layer:
            self.mlp = K2HorizonSparseMoeBlock(config)
        else:
            self.mlp = K2HorizonMLP(config, intermediate_size=config.intermediate_size)

        assert config.hidden_size % config.layernorm_num_groups == 0
        self.input_layernorm = K2HorizonRMSNorm(
            hidden_size=config.hidden_size,
            n_groups=config.layernorm_num_groups,
            eps=config.rms_norm_eps)
        self.post_attention_layernorm = K2HorizonRMSNorm(
            hidden_size=config.hidden_size,
            n_groups=config.layernorm_num_groups,
            eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states

        return hidden_states


class K2HorizonRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: K2HorizonConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: K2HorizonConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        # dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = (
            config.rope_head_dim
            if config.rope_head_dim is not None
            else getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class K2HorizonPreTrainedModel(PreTrainedModel):
    config_class = K2HorizonConfig
    config: K2HorizonConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["K2HorizonDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(K2HorizonSparseMoeBlock, index=1),
        "hidden_states": K2HorizonDecoderLayer,
        "attentions": [K2HorizonAttention, K2HorizonMoVAAttention],
    }

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, K2HorizonRotaryEmbedding):
            rope_init = module.compute_default_rope_parameters
            if module.rope_type != "default":
                rope_init = ROPE_INIT_FUNCTIONS[module.rope_type]
            inv_freq, module.attention_scaling = rope_init(module.config, module.inv_freq.device)
            init.copy_(module.inv_freq, inv_freq)
            init.copy_(module.original_inv_freq, inv_freq)
        else:
            super()._init_weights(module)


class K2HorizonModel(K2HorizonPreTrainedModel):
    def __init__(self, config: K2HorizonConfig):
        super().__init__(config)
        # self.padding_idx = config.pad_token_id
        self.padding_idx = getattr(config, "padding_idx", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [K2HorizonDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        assert config.hidden_size % config.layernorm_num_groups == 0
        self.norm = K2HorizonRMSNorm(
            hidden_size=config.hidden_size,
            n_groups=config.layernorm_num_groups,
            eps=config.rms_norm_eps)

        self.rotary_emb = K2HorizonRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def load_balancing_loss_func(
        gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
        num_experts: Optional[int] = None,
        top_k=2,
        attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class K2HorizonForCausalLM(K2HorizonPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    # _tp_plan = {"lm_head": "colwise_rep"}
    # _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = K2HorizonModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from gptqmodel.models.ports.k2_horizon import K2HorizonForCausalLM

        >>> model = K2HorizonForCausalLM.from_pretrained("IFM/K2-Horizon-MoVA-36B-A4B")
        >>> tokenizer = AutoTokenizer.from_pretrained("IFM/K2-Horizon-MoVA-36B-A4B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            ) if outputs.router_logits else logits.new_zeros(())
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class K2HorizonForSequenceClassification(GenericForSequenceClassification, K2HorizonPreTrainedModel):
    pass


class K2HorizonForTokenClassification(GenericForTokenClassification, K2HorizonPreTrainedModel):
    pass


class K2HorizonForQuestionAnswering(GenericForQuestionAnswering, K2HorizonPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "K2HorizonForCausalLM",
    "K2HorizonForQuestionAnswering",
    "K2HorizonModel",
    "K2HorizonPreTrainedModel",
    "K2HorizonForSequenceClassification",
    "K2HorizonForTokenClassification",
]
