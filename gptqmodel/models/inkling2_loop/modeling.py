# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Inkling 2 Mini language model.

Architecture notes (all of them are visible in the checkpoint's tensor shapes):

* the decoder is 24 layers; every sixth layer attends over the full context and
  the others are limited to a 1024-token sliding window;
* layers 6..17 form a *loop core*: the same weights are visited three times and
  every pass after the first receives four learned messages (q/k/v/mlp) that are
  retrieved from the block states of the previous pass;
* before each attention and each MLP the layer retrieves its input from a bank of
  block-boundary states with one learned query (three for the full-attention
  layers, which build separate q/k/v inputs);
* keys and values pass through a short causal convolution (static taps in the
  sliding layers, hidden-state-conditioned taps in the full layers), attention
  has one learned sink logit per head and a hidden-state-conditioned relative
  position bias, and the output projection is gated by the layer input.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
    find_first_full_attention_index,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    ModelOutput,
    MoeModelOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel, slice_logits_to_keep

from .configuration import Inkling2LoopConfig, Inkling2LoopTextConfig
from .experts import Inkling2LoopExperts, build_experts
from .vision import Inkling2LoopVisionModel


# --------------------------------------------------------------------------- #
# normalization, clipping and scaling primitives
# --------------------------------------------------------------------------- #


class Inkling2LoopUnweightedRMSNorm(nn.Module):
    """RMS normalization without a learned gain."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        states = hidden_states.float()
        states = states * torch.rsqrt(states.square().mean(-1, keepdim=True) + self.eps)
        return states.to(dtype)


class Inkling2LoopRMSNorm(nn.Module):
    """RMS normalization with a learned per-channel gain."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        states = hidden_states.float()
        states = states * torch.rsqrt(states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight * states).to(dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Inkling2LoopL2Clip(nn.Module):
    """Rescale every head vector whose L2 norm exceeds ``threshold``."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        states = hidden_states.float()
        norm = torch.linalg.vector_norm(states, dim=-1, keepdim=True)
        return (states * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(dtype)


class Inkling2LoopScaledL2Clip(nn.Module):
    """Per-channel gain followed by :class:`Inkling2LoopL2Clip`."""

    def __init__(self, num_heads: int, head_dim: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, hidden_states: torch.Tensor, position_ids=None) -> torch.Tensor:
        dtype = hidden_states.dtype
        states = hidden_states.float() * self.scale.float()
        norm = torch.linalg.vector_norm(states, dim=-1, keepdim=True)
        return (states * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(dtype)


class Inkling2LoopLogScale(nn.Module):
    """Grow the head vectors logarithmically once the position passes a floor."""

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__()
        self.alpha = config.log_scaling_alpha
        self.floor = config.log_scaling_n_floor

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        scale = 1.0 + self.alpha * torch.log(
            ((position_ids + 1).float() / self.floor).clamp(min=1.0)
        )
        return (hidden_states.float() * scale[..., None, None]).to(dtype)


class Inkling2LoopLogScaledL2Clip(nn.Module):
    """:class:`Inkling2LoopScaledL2Clip` followed by :class:`Inkling2LoopLogScale`."""

    def __init__(self, config: Inkling2LoopTextConfig, num_heads: int, head_dim: int, threshold: float):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))
        self.l2_clip = Inkling2LoopL2Clip(threshold)
        self.log_scale = Inkling2LoopLogScale(config)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        states = self.l2_clip(hidden_states.float() * self.scale.float()).to(dtype)
        return self.log_scale(states, position_ids)


class Inkling2LoopSoftcap(nn.Module):
    def __init__(self, softcap: float):
        super().__init__()
        self.softcap = softcap

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(hidden_states / self.softcap) * self.softcap


# --------------------------------------------------------------------------- #
# attention helpers
# --------------------------------------------------------------------------- #


def _repeat_key_value(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, heads, length, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None].expand(batch, heads, repeats, length, head_dim)
    return expanded.reshape(batch, heads * repeats, length, head_dim)


def _attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    scaling: float,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    num_key_value_groups: int = 1,
    dropout: float = 0.0,
) -> torch.Tensor:
    key = _repeat_key_value(key, num_key_value_groups)
    weights = torch.matmul(query, key.transpose(2, 3))
    del key
    weights = weights * scaling
    if position_bias is not None:
        weights = weights + position_bias
    if mask is not None:
        weights = weights + mask
    if sinks is not None:
        # one extra logit per query with a zero value: it only absorbs probability mass
        weights = torch.cat([weights, sinks], dim=-1)
    # Release each score buffer at its last use. Prefill score buffers can
    # be several GiB before the FP32 softmax and narrowing cast.
    weights = weights.float()
    weights = F.softmax(weights, dim=-1)
    if sinks is not None:
        weights = weights[..., :-1]
    weights = weights.to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=dropout > 0.0)
    value = _repeat_key_value(value, num_key_value_groups)
    return torch.matmul(weights, value).transpose(1, 2).contiguous()


class Inkling2LoopRelativePositionBias(nn.Module):
    """Hidden-state-conditioned bias over the last ``rel_extent`` backward distances."""

    def __init__(self, config: Inkling2LoopTextConfig, num_heads: int, rel_extent: int, log_scaling: bool):
        super().__init__()
        self.num_heads = num_heads
        self.rel_extent = rel_extent
        self.linear = nn.Linear(config.hidden_size, num_heads * config.d_rel, bias=False)
        if log_scaling:
            self.norm = Inkling2LoopLogScaledL2Clip(
                config, num_heads, config.d_rel, config.rel_l2_clip_threshold
            )
        else:
            self.norm = Inkling2LoopScaledL2Clip(
                num_heads, config.d_rel, config.rel_l2_clip_threshold
            )
        self.proj = nn.Parameter(torch.empty(config.d_rel, rel_extent))

    def forward(
        self,
        hidden_states: torch.Tensor,
        relative_distance: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        states = self.linear(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, -1)
        states = self.norm(states, position_ids)
        logits = (states @ self.proj).transpose(1, 2)
        gather_index = relative_distance.clamp(0, self.rel_extent - 1).expand(
            *logits.shape[:2], -1, -1
        )
        bias = logits.gather(-1, gather_index)
        out_of_range = (relative_distance < 0) | (relative_distance >= self.rel_extent)
        return bias.masked_fill(out_of_range, 0.0)


def _mask_padding(hidden_states: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return hidden_states
    return (hidden_states * mask[:, :, None]).to(hidden_states.dtype)


class Inkling2LoopShortConvolution(nn.Module):
    """Depthwise causal convolution mixed back into its input with a fixed 1/sqrt(k+1) scale."""

    def __init__(self, hidden_size: int, kernel_size: int, layer_idx: int, conv_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_idx = conv_idx
        self.conv_kernel_size = kernel_size
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=False,
        )

    def _conv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            hidden_states,
            weight=self.conv1d.weight,
            bias=self.conv1d.bias,
            padding=self.conv_kernel_size - 1,
            groups=hidden_states.shape[1],
        )

    def _update(self, hidden_states: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        state_len = conv_state.shape[-1]
        seq_len = hidden_states.shape[-1]
        merged = torch.cat([conv_state, hidden_states], dim=-1).to(self.conv1d.weight.dtype)
        conv_state.copy_(merged[:, :, -state_len:])
        out = F.conv1d(
            merged,
            weight=self.conv1d.weight,
            bias=self.conv1d.bias,
            padding=0,
            groups=merged.shape[1],
        )
        return out[:, :, -seq_len:]

    def forward(
        self,
        hidden_states: torch.Tensor,
        dynamic_input: Optional[torch.Tensor] = None,
        cache: Optional[Cache] = None,
        conv_mask: Optional[torch.Tensor] = None,
        cache_offset: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        layer_idx = self.layer_idx + cache_offset
        dtype = hidden_states.dtype
        states = hidden_states.float()
        residual = states
        states = _mask_padding(states, conv_mask)
        seq_len = states.shape[1]
        states = states.transpose(1, 2)

        conv_state = None if cache is None else cache.get_conv_state(layer_idx, state_idx=self.conv_idx)
        if conv_state is not None and seq_len == 1 and not cache.is_recording_past(layer_idx):
            states = self._update(states, conv_state)
        else:
            if cache is not None:
                states = cache.update_conv_state(
                    states,
                    layer_idx,
                    state_idx=self.conv_idx,
                    conv_kernel_size=self.conv_kernel_size,
                )
            states = self._conv(states)[:, :, -seq_len:]

        states = states.transpose(1, 2)
        mixed = (states + residual) / math.sqrt(self.conv_kernel_size + 1)
        return mixed.to(dtype)


class Inkling2LoopDynamicShortConvolution(nn.Module):
    """Short convolution whose taps are static weights plus a projection of the layer input."""

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        num_heads: int,
        kernel_size: int,
        layer_idx: int,
        conv_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_idx = conv_idx
        self.conv_kernel_size = kernel_size
        self.num_heads = num_heads
        self.static_weight = nn.Parameter(torch.zeros(kernel_size, hidden_size))
        self.weight_proj = nn.Linear(input_size, num_heads * kernel_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        dynamic_input: torch.Tensor,
        cache: Optional[Cache] = None,
        conv_mask: Optional[torch.Tensor] = None,
        cache_offset: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        batch, seq_len, hidden_size = hidden_states.shape
        kernel = self.conv_kernel_size

        dynamic = self.weight_proj(dynamic_input).float()
        dynamic = dynamic.view(batch, seq_len, self.num_heads, 1, kernel)
        taps = self.static_weight.float().T.view(self.num_heads, -1, kernel) + dynamic

        states = hidden_states.float()
        residual = states
        context = _mask_padding(states, conv_mask).transpose(1, 2)
        if cache is not None:
            context = cache.update_conv_state(
                context,
                self.layer_idx + cache_offset,
                state_idx=self.conv_idx,
                conv_kernel_size=kernel,
            )
        context = F.pad(context, (kernel - 1, 0))[..., -(seq_len + kernel - 1):]
        # windows[..., lag] is the token `lag` steps back
        windows = context.unfold(-1, kernel, 1).flip(-1).transpose(1, 2)
        windows = windows.reshape(batch, seq_len, self.num_heads, -1, kernel)

        seq_idx = kwargs.get("seq_idx")
        if seq_idx is not None:
            seq_idx = seq_idx.to(windows.device)
            context_seq_idx = torch.cat(
                [seq_idx[:, :1].expand(-1, kernel - 1), seq_idx], dim=1
            )
            window_seq_idx = context_seq_idx.unfold(-1, kernel, 1).flip(-1)
            windows = windows * (window_seq_idx == seq_idx[..., None])[:, :, None, None, :]

        mixed = (windows * taps).sum(-1).reshape(batch, seq_len, hidden_size)
        mixed = (mixed + residual) / math.sqrt(2 * kernel + 1)
        return mixed.to(dtype)


class Inkling2LoopQueryProjection(nn.Module):
    """Projection that produces the per-head queries, with the clip (and optional log scale)."""

    def __init__(self, config: Inkling2LoopTextConfig, num_heads: int, head_dim: int, log_scaling: bool):
        super().__init__()
        self.head_dim = head_dim
        self.linear = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
        if log_scaling:
            self.norm = Inkling2LoopLogScaledL2Clip(
                config, num_heads, head_dim, config.qk_l2_clip_threshold
            )
        else:
            self.norm = Inkling2LoopScaledL2Clip(num_heads, head_dim, config.qk_l2_clip_threshold)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        states = self.linear(hidden_states)
        states = states.view(*states.shape[:-1], -1, self.head_dim)
        return self.norm(states, position_ids).transpose(1, 2)


class Inkling2LoopKeyValueProjection(nn.Module):
    """Projection that produces the per-head keys or values behind the short convolution."""

    def __init__(
        self,
        config: Inkling2LoopTextConfig,
        num_heads: int,
        head_dim: int,
        layer_idx: int,
        conv_idx: int,
        is_key: bool,
        dynamic: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.linear = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
        if dynamic:
            self.sconv = Inkling2LoopDynamicShortConvolution(
                num_heads * head_dim,
                config.hidden_size,
                num_heads,
                config.conv_kernel_size,
                layer_idx,
                conv_idx=conv_idx,
            )
        else:
            self.sconv = Inkling2LoopShortConvolution(
                num_heads * head_dim, config.conv_kernel_size, layer_idx, conv_idx=conv_idx
            ).to(dtype=torch.float32)
        if is_key:
            self.norm = Inkling2LoopL2Clip(config.qk_l2_clip_threshold)
        else:
            self.norm = Inkling2LoopUnweightedRMSNorm(eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[Cache] = None,
        conv_mask: Optional[torch.Tensor] = None,
        cache_offset: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        states = self.linear(hidden_states)
        states = self.sconv(
            states,
            hidden_states,
            cache=cache,
            conv_mask=conv_mask,
            cache_offset=cache_offset,
            **kwargs,
        )
        states = states.view(*states.shape[:-1], -1, self.head_dim)
        return self.norm(states).transpose(1, 2)


class Inkling2LoopGatedOutputProjection(nn.Module):
    """Attention output projection, gated by a sigmoid of the attention input."""

    def __init__(self, config: Inkling2LoopTextConfig, num_heads: int, head_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
        self.linear = nn.Linear(num_heads * head_dim, config.hidden_size, bias=False)

    def forward(self, attn_output: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(hidden_states).float())
        return self.linear((attn_output.float() * gate).to(attn_output.dtype))


class _Inkling2LoopAttention(nn.Module):
    """Shared attention body; ``log_scaling`` and ``dynamic_conv`` select the layer flavour."""

    log_scaling = False
    dynamic_conv = False

    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.is_sliding_layer(layer_idx)
        if self.is_sliding:
            self.head_dim = config.swa_head_dim
            self.num_heads = config.swa_num_attention_heads
            self.num_key_value_heads = config.swa_num_key_value_heads
            self.sliding_window = config.sliding_window_size
            self.rel_extent = config.sliding_window_size
        else:
            self.head_dim = config.head_dim
            self.num_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.sliding_window = None
            self.rel_extent = config.rel_extent
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = Inkling2LoopQueryProjection(
            config, self.num_heads, self.head_dim, log_scaling=self.log_scaling
        )
        self.k_proj = Inkling2LoopKeyValueProjection(
            config,
            self.num_key_value_heads,
            self.head_dim,
            layer_idx,
            conv_idx=0,
            is_key=True,
            dynamic=self.dynamic_conv,
        )
        self.v_proj = Inkling2LoopKeyValueProjection(
            config,
            self.num_key_value_heads,
            self.head_dim,
            layer_idx,
            conv_idx=1,
            is_key=False,
            dynamic=self.dynamic_conv,
        )
        self.o_proj = Inkling2LoopGatedOutputProjection(config, self.num_heads, self.head_dim)
        self.sink_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.relative_position_bias = Inkling2LoopRelativePositionBias(
            config, self.num_heads, self.rel_extent, log_scaling=self.log_scaling
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        relative_distance: torch.Tensor,
        conv_mask: Optional[torch.Tensor] = None,
        cache: Optional[Cache] = None,
        key_hidden_states: Optional[torch.Tensor] = None,
        value_hidden_states: Optional[torch.Tensor] = None,
        cache_offset: int = 0,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
        key_hidden_states = hidden_states if key_hidden_states is None else key_hidden_states
        value_hidden_states = hidden_states if value_hidden_states is None else value_hidden_states
        input_shape = hidden_states.shape[:-1]

        query_states = self.q_proj(hidden_states, position_ids)
        key_states = self.k_proj(
            key_hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs
        )
        value_states = self.v_proj(
            value_hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs
        )
        if cache is not None:
            key_states, value_states = cache.update(
                key_states, value_states, self.layer_idx + cache_offset
            )

        position_bias = self.relative_position_bias(hidden_states, relative_distance, position_ids)
        sinks = self.sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)
        attn_output = _attend(
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.scaling,
            position_bias=position_bias,
            sinks=sinks,
            num_key_value_groups=self.num_key_value_groups,
            dropout=0.0 if not self.training else self.attention_dropout,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output, hidden_states), None


class Inkling2LoopSlidingAttention(_Inkling2LoopAttention):
    """Sliding-window layers: static short convolutions, no log scaling."""


class Inkling2LoopFullAttention(_Inkling2LoopAttention):
    """Full-context layers: input-conditioned short convolutions and log scaling."""

    log_scaling = True
    dynamic_conv = True


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #


class Inkling2LoopTopKRouter(nn.Module):
    """Sigmoid router with a learned score-correction bias used only for the top-k choice."""

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, dtype=torch.float32)
        )
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.route_scale
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(self.num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states: torch.Tensor):
        states = hidden_states.float().reshape(-1, self.hidden_dim)
        logits = F.linear(states, self.weight)
        scores = self.score_fn(logits)
        indices = torch.topk(
            scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False
        ).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class Inkling2LoopSparseMoeBlock(nn.Module):
    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int):
        super().__init__()
        self.experts = build_experts(config, layer_idx)
        self.gate = Inkling2LoopTopKRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected = self.gate(flat)
        routed = self.experts(flat, selected, routing_weights)
        return routed.reshape(batch, seq_len, hidden_dim)


# --------------------------------------------------------------------------- #
# attention residuals and message gates
# --------------------------------------------------------------------------- #


class Inkling2LoopBlockAttentionResidual(nn.Module):
    """Retrieve one layer input from the bank of block-boundary states."""

    def __init__(self, config: Inkling2LoopTextConfig, num_queries: int = 1):
        super().__init__()
        shape = (config.hidden_size,) if num_queries == 1 else (num_queries, config.hidden_size)
        self.query = nn.Parameter(torch.zeros(*shape, dtype=torch.float32))
        self.norm = Inkling2LoopUnweightedRMSNorm()
        self.config = config

    def forward(self, block_states: torch.Tensor, partial_block_states: torch.Tensor):
        stacked = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2).float()
        stacked = self.norm(stacked)
        hidden_size = stacked.shape[-1]
        queries = self.query.float().view(-1, hidden_size)
        queries = queries.expand(*stacked.shape[:-2], -1, -1)
        attended = _attend(queries, stacked, stacked, None, hidden_size**-0.5)
        return attended.transpose(1, 2).unbind(-2)


class Inkling2LoopMessageAttentionResidual(nn.Module):
    """Retrieve the q/k/v/mlp messages a loop layer receives from the previous pass."""

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__()
        # The checkpoint stores two flat rows per loop layer: (query, mlp) in
        # `..._queries` and (key, value) in `..._kv_queries`. Keeping the
        # parameters flat makes the checkpoint load 1:1; the forward interleaves
        # them back into the four message slots.
        num_loop_layers = config.loop_layer_index_end - config.loop_layer_index_start
        shape = (2 * num_loop_layers, config.hidden_size)
        self.query = nn.Parameter(torch.zeros(*shape, dtype=torch.float32))
        self.kv_query = nn.Parameter(torch.zeros(*shape, dtype=torch.float32))
        self.norm = Inkling2LoopUnweightedRMSNorm()
        self.config = config

    def forward(
        self,
        block_states: torch.Tensor,
        partial_block_states: torch.Tensor,
        loop_layer_idx: int,
    ) -> torch.Tensor:
        stacked = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2).float()
        stacked = self.norm(stacked)
        hidden_size = stacked.shape[-1]
        query = self.query[2 * loop_layer_idx].float()
        mlp_query = self.query[2 * loop_layer_idx + 1].float()
        key_value = self.kv_query[2 * loop_layer_idx : 2 * loop_layer_idx + 2].float()
        queries = torch.cat([query[None], key_value, mlp_query[None]], dim=0)
        queries = queries.expand(*stacked.shape[:-2], -1, -1)
        attended = _attend(queries, stacked, stacked, None, hidden_size**-0.5)
        return attended.transpose(1, 2)


class Inkling2LoopMessageGate(nn.Module):
    """``(1 - forget) * hidden_states + message_gate * message`` from a shared low-rank gate."""

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__()
        self.norm = Inkling2LoopUnweightedRMSNorm(eps=config.rms_norm_eps)
        self.down_proj = nn.Linear(2 * config.hidden_size, config.loop_gate_rank, bias=False)
        self.forget_gate_proj = nn.Linear(
            config.loop_gate_rank, config.hidden_size, bias=False, dtype=torch.float32
        )
        self.message_gate_proj = nn.Linear(
            config.loop_gate_rank, config.hidden_size, bias=False, dtype=torch.float32
        )

    def forward(self, hidden_states: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        hidden_states, message = hidden_states.float(), message.float()
        gate_input = torch.cat([self.norm(hidden_states), self.norm(message)], dim=-1)
        gate_input = self.down_proj(gate_input.to(self.down_proj.weight.dtype)).float()
        forget = torch.tanh(self.forget_gate_proj(gate_input))
        write = torch.tanh(self.message_gate_proj(gate_input))
        return (1 - forget) * hidden_states + write * message


# --------------------------------------------------------------------------- #
# decoder layers
# --------------------------------------------------------------------------- #


class _Inkling2LoopDecoderLayer(GradientCheckpointingLayer):
    is_full_attention = False
    has_messages = False

    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.attention_type = "full_attention" if self.is_full_attention else "sliding_attention"

        num_queries = 3 if self.is_full_attention else 1
        self.attention_input = (
            Inkling2LoopBlockAttentionResidual(config, num_queries=num_queries)
            if layer_idx > 0
            else None
        )
        self.mlp_input = Inkling2LoopBlockAttentionResidual(config)
        if self.is_full_attention:
            self.self_attn = Inkling2LoopFullAttention(config, layer_idx)
            self.q_input_layernorm = Inkling2LoopRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.k_input_layernorm = Inkling2LoopRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.v_input_layernorm = Inkling2LoopRMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.self_attn = Inkling2LoopSlidingAttention(config, layer_idx)
            self.input_layernorm = Inkling2LoopRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Inkling2LoopSparseMoeBlock(config, layer_idx)
        self.post_attention_layernorm = Inkling2LoopRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_convolution = Inkling2LoopShortConvolution(
            config.hidden_size, config.conv_kernel_size, layer_idx=layer_idx, conv_idx=2
        ).to(dtype=torch.float32)
        self.post_mlp_convolution = Inkling2LoopShortConvolution(
            config.hidden_size, config.conv_kernel_size, layer_idx=layer_idx, conv_idx=3
        ).to(dtype=torch.float32)
        if self.has_messages:
            self.q_message_gate = Inkling2LoopMessageGate(config)
            self.k_message_gate = Inkling2LoopMessageGate(config)
            self.v_message_gate = Inkling2LoopMessageGate(config)
            self.mlp_message_gate = Inkling2LoopMessageGate(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        messages: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        relative_distance: Optional[torch.Tensor] = None,
        conv_mask: Optional[torch.Tensor] = None,
        cache: Optional[Cache] = None,
        cache_offset: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        block_residual = hidden_states

        if self.attention_input is None:
            query_states = key_states = value_states = hidden_states
        else:
            retrieved = self.attention_input(block_states, hidden_states)
            if self.is_full_attention:
                query_states, key_states, value_states = retrieved
            else:
                query_states = key_states = value_states = retrieved[0]

        mlp_message = None
        if messages is not None:
            query_message, key_message, value_message, mlp_message = messages.unbind(-2)
            query_states = self.q_message_gate(query_states, query_message)
            key_states = self.k_message_gate(key_states, key_message)
            value_states = self.v_message_gate(value_states, value_message)

        if self.is_full_attention:
            query_states = self.q_input_layernorm(query_states)
            key_states = self.k_input_layernorm(key_states)
            value_states = self.v_input_layernorm(value_states)
        else:
            query_states = self.input_layernorm(query_states)
            key_states = self.input_layernorm(key_states)
            value_states = self.input_layernorm(value_states)

        hidden_states, _ = self.self_attn(
            hidden_states=query_states.to(dtype),
            attention_mask=attention_mask,
            relative_distance=relative_distance,
            key_hidden_states=key_states.to(dtype),
            value_hidden_states=value_states.to(dtype),
            conv_mask=conv_mask,
            cache=cache,
            cache_offset=cache_offset,
            **kwargs,
        )
        hidden_states = self.post_attention_convolution(
            hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs
        )

        residual = hidden_states = hidden_states.float()
        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
        else:
            residual = hidden_states = block_residual.float() + hidden_states

        hidden_states = self.mlp_input(block_states, hidden_states)[0]
        if mlp_message is not None:
            hidden_states = self.mlp_message_gate(hidden_states, mlp_message)
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states).to(dtype))
        hidden_states = self.post_mlp_convolution(
            hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs
        )
        return (residual + hidden_states).to(dtype)


class Inkling2LoopSlidingDecoderLayer(_Inkling2LoopDecoderLayer):
    """Sliding-window layer outside the loop core."""


class Inkling2LoopSlidingLoopDecoderLayer(_Inkling2LoopDecoderLayer):
    """Sliding-window layer inside the loop core: adds the four message gates."""

    has_messages = True


class Inkling2LoopFullDecoderLayer(_Inkling2LoopDecoderLayer):
    """Full-context layer outside the loop core."""

    is_full_attention = True


class Inkling2LoopFullLoopDecoderLayer(_Inkling2LoopDecoderLayer):
    """Full-context layer inside the loop core: adds the four message gates."""

    is_full_attention = True
    has_messages = True


# --------------------------------------------------------------------------- #
# embeddings and the language model
# --------------------------------------------------------------------------- #


class Inkling2LoopNormedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int], norm_eps: float):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_norm = Inkling2LoopRMSNorm(embedding_dim, eps=norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(super().forward(input_ids))


def get_relative_distance(
    inputs_embeds: torch.Tensor, cache: Optional[Cache], is_sliding: bool
) -> torch.Tensor:
    """Distance from every query position to every key position, shaped ``(q_len, kv_len)``."""

    q_length = inputs_embeds.shape[1]
    if cache is None:
        kv_length, kv_offset, q_offset = q_length, 0, 0
    else:
        if not is_sliding:
            layer_idx = find_first_full_attention_index(cache)
        elif True in cache.is_sliding:
            layer_idx = cache.is_sliding.index(True)
        else:
            layer_idx = 0
        kv_length, kv_offset = cache.get_mask_sizes(q_length, layer_idx)
        q_offset = cache.get_query_offset(layer_idx)
    q_positions = torch.arange(q_length, device=inputs_embeds.device) + q_offset
    kv_positions = torch.arange(kv_length, device=inputs_embeds.device) + kv_offset
    return q_positions[:, None] - kv_positions[None, :]


class Inkling2LoopPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Inkling2LoopSlidingDecoderLayer",
        "Inkling2LoopFullDecoderLayer",
        "Inkling2LoopSlidingLoopDecoderLayer",
        "Inkling2LoopFullLoopDecoderLayer",
    ]
    # per-token attention sinks and per-layer cache offsets are not supported by
    # the fused attention backends
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = False
    _can_compile_fullgraph = False
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias",
        "mlp.gate.weight",
        "post_attention_convolution",
        "post_mlp_convolution",
        "sconv.conv1d",
        "attention_input",
        "mlp_input",
        "final_input",
        "message_input",
        "message_gate_proj",
        "forget_gate_proj",
    ]

    def _init_weights(self, module):
        super()._init_weights(module)
        config = getattr(module, "config", None)
        std = getattr(config, "initializer_range", 0.02) or 0.02
        if isinstance(module, Inkling2LoopRelativePositionBias):
            nn.init.normal_(module.proj, mean=0.0, std=std)
        elif isinstance(module, Inkling2LoopExperts):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, Inkling2LoopTopKRouter):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            nn.init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, (Inkling2LoopScaledL2Clip, Inkling2LoopLogScaledL2Clip)):
            nn.init.ones_(module.scale)
        elif isinstance(module, Inkling2LoopDynamicShortConvolution):
            nn.init.zeros_(module.static_weight)
        elif isinstance(
            module, (Inkling2LoopBlockAttentionResidual, Inkling2LoopMessageAttentionResidual)
        ):
            nn.init.zeros_(module.query)
            if isinstance(module, Inkling2LoopMessageAttentionResidual):
                nn.init.zeros_(module.kv_query)


class Inkling2LoopTextModel(Inkling2LoopPreTrainedModel):
    """The 24-layer looped decoder, without the language modeling head."""

    config_class = Inkling2LoopTextConfig
    config: Inkling2LoopTextConfig

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = Inkling2LoopNormedEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, config.rms_norm_eps
        )
        layer_classes = {
            (False, False): Inkling2LoopSlidingDecoderLayer,
            (True, False): Inkling2LoopSlidingLoopDecoderLayer,
            (False, True): Inkling2LoopFullDecoderLayer,
            (True, True): Inkling2LoopFullLoopDecoderLayer,
        }
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            layer_class = layer_classes[
                (config.is_loop_layer(layer_idx), not config.is_sliding_layer(layer_idx))
            ]
            self.layers.append(layer_class(config, layer_idx))
        self.norm = Inkling2LoopRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.final_input = Inkling2LoopBlockAttentionResidual(config)
        self.message_input = Inkling2LoopMessageAttentionResidual(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + seen
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            attention_mask = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }
        relative_distance = {
            "full_attention": get_relative_distance(inputs_embeds, past_key_values, False),
            "sliding_attention": get_relative_distance(inputs_embeds, past_key_values, True),
        }

        loop_start = self.config.loop_layer_index_start
        loop_end = self.config.loop_layer_index_end
        num_loop_layers = loop_end - loop_start
        num_passes = self.config.loop_num_passes

        if past_key_values is not None and len(past_key_values.layers) == self.config.num_hidden_layers:
            # every extra pass over the loop core needs its own cache layers
            extra = [
                copy.deepcopy(layer)
                for layer in past_key_values.layers[loop_start:loop_end] * (num_passes - 1)
            ]
            past_key_values.layers += extra

        hidden_states = inputs_embeds
        block_states = inputs_embeds.new_zeros(
            *inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1]
        )

        def run(layer, hidden_states, block_states, messages=None, cache_offset=0):
            block_residual = hidden_states
            hidden_states = layer(
                hidden_states,
                block_states,
                messages,
                attention_mask=attention_mask[layer.attention_type],
                relative_distance=relative_distance[layer.attention_type],
                conv_mask=attention_mask["linear_attention"],
                cache=past_key_values,
                cache_offset=cache_offset,
                position_ids=position_ids,
                **kwargs,
            )
            if layer.is_block_boundary:
                block_states = torch.cat(
                    [block_states.to(hidden_states.device), block_residual.unsqueeze(-2)], dim=-2
                )
            return hidden_states, block_states

        for layer in self.layers[:loop_start]:
            hidden_states, block_states = run(layer, hidden_states, block_states)

        pre_loop_hidden_states, pre_loop_block_states = hidden_states, block_states
        num_pre_loop_blocks = pre_loop_block_states.shape[-2]

        for layer in self.layers[loop_start:loop_end]:
            hidden_states, block_states = run(layer, hidden_states, block_states)

        for loop_idx in range(1, num_passes):
            # +1 to skip the block the first loop pass appended
            previous_block_states = block_states[..., num_pre_loop_blocks + 1:, :]
            previous_hidden_states = hidden_states
            hidden_states, block_states = pre_loop_hidden_states, pre_loop_block_states
            cache_offset = (
                self.config.num_hidden_layers + (loop_idx - 1) * num_loop_layers - loop_start
            )
            for loop_layer_idx, layer in enumerate(self.layers[loop_start:loop_end]):
                messages = self.message_input(
                    previous_block_states, previous_hidden_states, loop_layer_idx
                )
                hidden_states, block_states = run(
                    layer, hidden_states, block_states, messages, cache_offset
                )

        for layer in self.layers[loop_end:]:
            hidden_states, block_states = run(layer, hidden_states, block_states)

        hidden_states = self.final_input(block_states, hidden_states)[0]
        hidden_states = self.norm(hidden_states).to(inputs_embeds.dtype)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values
        )


@dataclass
class Inkling2LoopCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Any] = None
    attentions: Optional[Any] = None
    router_logits: Optional[Any] = None
    image_outputs: Optional[Any] = None


class Inkling2LoopForCausalLM(Inkling2LoopPreTrainedModel, GenerationMixin):
    """Text-only head: ``embed`` and ``unembed`` are separate checkpoint tensors."""

    config_class = Inkling2LoopTextConfig
    config: Inkling2LoopTextConfig
    _tied_weights_keys = {}

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__(config)
        self.model = Inkling2LoopTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logit_softcap = Inkling2LoopSoftcap(config.final_logit_softcapping)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> Inkling2LoopCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        logits = self.lm_head(hidden_states)[..., : self.config.unpadded_vocab_size]
        logits = self.logit_softcap(logits.float())
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=logits.shape[-1], **kwargs
            )
        return Inkling2LoopCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


@dataclass
class Inkling2LoopModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Any] = None
    attentions: Optional[Any] = None
    router_logits: Optional[Any] = None
    image_outputs: Optional[BaseModelOutputWithPooling] = None


class Inkling2LoopModel(Inkling2LoopPreTrainedModel):
    """Vision backbone plus language model, without the language modeling head."""

    config_class = Inkling2LoopConfig
    config: Inkling2LoopConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")

    def __init__(self, config: Inkling2LoopConfig):
        super().__init__(config)
        self.language_model = Inkling2LoopTextModel(config.text_config)
        self.vision_tower = Inkling2LoopVisionModel(config.vision_config)
        self.post_init()

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor, **kwargs
    ) -> BaseModelOutputWithPooling:
        image_grid_thw = image_grid_thw.cpu()
        outputs = self.vision_tower(
            pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs
        )
        outputs.pooler_output = outputs.pooler_output.split(image_grid_thw.prod(-1).tolist())
        return outputs

    def get_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.Tensor,
        features: torch.Tensor,
        token_id: int,
    ) -> torch.Tensor:
        if input_ids is None:
            token = torch.tensor(token_id, dtype=torch.long, device=inputs_embeds.device)
            mask = (inputs_embeds == self.get_input_embeddings()(token)).all(-1)
        else:
            mask = input_ids == token_id
        count = mask.sum()
        mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if inputs_embeds[mask].numel() != features.numel():
            raise ValueError(
                "Image features and image placeholder tokens do not match: "
                f"{count} placeholders, {features.shape[0]} features"
            )
        return mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Inkling2LoopModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_outputs = None
        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_outputs.pooler_output).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, image_embeds, self.config.image_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return Inkling2LoopModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            image_outputs=image_outputs,
        )


class Inkling2LoopForConditionalGeneration(Inkling2LoopPreTrainedModel, GenerationMixin):
    """Image-text model with a language modeling head."""

    config_class = Inkling2LoopConfig
    config: Inkling2LoopConfig
    _tied_weights_keys = {}
    input_modalities = ("image", "text")

    def __init__(self, config: Inkling2LoopConfig):
        super().__init__(config)
        self.model = Inkling2LoopModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.logit_softcap = Inkling2LoopSoftcap(config.text_config.final_logit_softcapping)
        self.post_init()

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor, **kwargs
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> Inkling2LoopCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        logits = self.lm_head(hidden_states)[..., : self.config.text_config.unpadded_vocab_size]
        logits = self.logit_softcap(logits.float())
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=logits.shape[-1], **kwargs
            )
        return Inkling2LoopCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            image_outputs=outputs.image_outputs,
        )


__all__ = [
    "Inkling2LoopCausalLMOutputWithPast",
    "Inkling2LoopForCausalLM",
    "Inkling2LoopForConditionalGeneration",
    "Inkling2LoopModel",
    "Inkling2LoopModelOutputWithPast",
    "Inkling2LoopPreTrainedModel",
    "Inkling2LoopTextModel",
    "get_relative_distance",
]
