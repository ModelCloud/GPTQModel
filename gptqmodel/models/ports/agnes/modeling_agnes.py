# Copyright 2026 Agnes AI. All rights reserved.
# Adapted for package-owned Transformers 5 loading by ModelCloud.ai, 2026.
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
"""Agnes 3.0 Flash.

A hybrid decoder: three delta-rule recurrent layers followed by one global
attention layer, repeated; every layer carries a main SwiGLU MLP plus a
parallel SwiGLU branch summed into the same residual.  A vision tower feeds
merged patch tokens into the language model through placeholder tokens.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicLayer, LinearAttentionLayer
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check
from transformers.utils.generic import (
    accepts_precomputed_kwargs,
    is_flash_attention_requested,
    maybe_autocast,
    merge_with_config_defaults,
)
from transformers.utils.output_capturing import capture_outputs
from transformers.vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)

from .configuration_agnes import LAYER_DELTA, LAYER_GLOBAL, AgnesConfig, AgnesTextConfig, AgnesVisionConfig


logger = logging.get_logger(__name__)

class AgnesDynamicCache(Cache):
    """Own the hybrid cache without modifying Transformers' global registries.

    Current Transformers stores linear states in dictionaries indexed by stream.
    Agnes has one stream per layer. Count tokens independently so all-delta and
    all-global layer plans remain valid as well as the released hybrid layout.
    A cache belongs to one request and must not be shared between threads.
    """

    def __init__(self, config: AgnesTextConfig):
        super().__init__(layers=[
            LinearAttentionLayer() if kind == LAYER_DELTA else DynamicLayer()
            for kind in config.layer_types
        ])
        self._agnes_seen_tokens = 0
        self._agnes_valid_tokens = None
        self._agnes_attention_mask = None
        self.rope_deltas = None
        self._in_forward = False

    def get_seq_length(self, layer_idx=0):
        return self._agnes_seen_tokens

    def reset(self):
        super().reset()
        self._agnes_seen_tokens = 0
        self._agnes_valid_tokens = None
        self._agnes_attention_mask = None
        self.rope_deltas = None
        self._in_forward = False

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        if self._agnes_valid_tokens is not None:
            self._agnes_valid_tokens = self._agnes_valid_tokens.index_select(0, beam_idx.to(self._agnes_valid_tokens.device))
        if self._agnes_attention_mask is not None:
            self._agnes_attention_mask = self._agnes_attention_mask.index_select(0, beam_idx.to(self._agnes_attention_mask.device))
        if self.rope_deltas is not None:
            self.rope_deltas = self.rope_deltas.index_select(0, beam_idx.to(self.rope_deltas.device))

    def batch_repeat_interleave(self, repeats):
        # LinearAttentionLayer has reorder_cache, but no batch repeat/select
        # methods in Transformers 5. Use the same indices for every state and
        # the request's rotary offsets.
        sample = self.rope_deltas
        if sample is None:
            for layer in self.layers:
                if isinstance(layer, LinearAttentionLayer):
                    sample = next(iter(layer.conv_states.values()), None)
                elif layer.is_initialized:
                    sample = layer.keys
                if sample is not None:
                    break
        if sample is not None:
            indices = torch.arange(sample.shape[0], device=sample.device).repeat_interleave(repeats)
            self.reorder_cache(indices)

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)

    def crop(self, max_length):
        raise ValueError("Agnes recurrent cache cannot crop; reset() and replay the prefix")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class _RMSNormFn(torch.autograd.Function):
    """RMS normalisation with a one-centred scale, y = x / rms(x) * (1 + w).

    The forward matches the reference op-for-op (fp32 statistics, cast back to
    the input dtype at the end); the backward is written out analytically
    instead of being traced through the forward graph.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        unit = xf * inv
        out = unit * (1.0 + weight.float())
        ctx.save_for_backward(unit, inv, weight)
        ctx.in_dtype = x.dtype
        return out.type_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        unit, inv, weight = ctx.saved_tensors
        g = grad_out.float()
        g_unit = g * (1.0 + weight.float())
        grad_x = inv * (g_unit - unit * (g_unit * unit).mean(-1, keepdim=True))
        grad_w = (g * unit).reshape(-1, unit.shape[-1]).sum(0)
        return grad_x.to(ctx.in_dtype), grad_w.to(weight.dtype), None


class AgnesRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return _RMSNormFn.apply(x, self.weight, self.eps)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class _GatedRMSNormFn(torch.autograd.Function):
    """RMS normalisation followed by a learned scale and a SiLU gate.

    Forward order (matters for bit-exactness): fp32 statistics, cast to the
    input dtype, multiply by the weight in that dtype, then multiply by the
    fp32 gate activation and cast back.
    """

    @staticmethod
    def forward(ctx, x, weight, gate, eps):
        in_dtype = x.dtype
        xf = x.to(torch.float32)
        inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        unit = xf * inv
        scaled = weight * unit.to(in_dtype)
        gate_f = gate.to(torch.float32)
        act = F.silu(gate_f)
        out = scaled * act
        ctx.save_for_backward(unit, inv, weight, gate_f, scaled)
        ctx.in_dtype = in_dtype
        return out.to(in_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        unit, inv, weight, gate_f, scaled = ctx.saved_tensors
        g = grad_out.float()
        sig = torch.sigmoid(gate_f)
        act = gate_f * sig
        g_scaled = g * act
        grad_w = (g_scaled * unit).reshape(-1, unit.shape[-1]).sum(0)
        g_unit = g_scaled * weight.float()
        grad_x = inv * (g_unit - unit * (g_unit * unit).mean(-1, keepdim=True))
        grad_gate = g * scaled.float() * (sig * (1.0 + gate_f * (1.0 - sig)))
        return grad_x.to(ctx.in_dtype), grad_w.to(weight.dtype), grad_gate.to(ctx.in_dtype), None


class AgnesGatedNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        return _GatedRMSNormFn.apply(hidden_states, self.weight, gate, self.variance_epsilon)


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------


class AgnesMLP(nn.Module):
    """SwiGLU block.  With `parallel_size > 0` a second, narrower SwiGLU runs
    on the same input and its output is added to the main one."""

    def __init__(self, config, intermediate_size: int, parallel_size: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.parallel_ffn = AgnesMLP(config, parallel_size) if parallel_size > 0 else None

    def forward(self, x):
        y = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        if self.parallel_ffn is not None:
            y = y + self.parallel_ffn(x)
        return y


# ---------------------------------------------------------------------------
# Rotary position encodings
# ---------------------------------------------------------------------------


def _swap_halves(x):
    """(a, b) -> (-b, a) along the last axis."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    """Rotate the leading `cos.shape[-1]` channels of q and k; the rest pass through."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    n_rot = cos.shape[-1]
    q_rot, q_rest = q[..., :n_rot], q[..., n_rot:]
    k_rot, k_rest = k[..., :n_rot], k[..., n_rot:]
    q_rot = (q_rot * cos) + (_swap_halves(q_rot) * sin)
    k_rot = (k_rot * cos) + (_swap_halves(k_rot) * sin)
    return torch.cat([q_rot, q_rest], dim=-1), torch.cat([k_rot, k_rest], dim=-1)


class AgnesRotaryEmbedding(nn.Module):
    """Three-axis (text / height / width) rotary tables with interleaved sections."""

    inv_freq: torch.Tensor

    def __init__(self, config: AgnesTextConfig, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_type = config.rope_parameters["rope_type"]
        builder: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            builder = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = builder(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(
        config: AgnesTextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        theta = config.rope_parameters["rope_theta"]
        fraction = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        n_rot = int(head_dim * fraction)
        exponents = torch.arange(0, n_rot, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / n_rot
        inv_freq = 1.0 / (theta**exponents)
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        freq_col = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        pos_row = position_ids[:, :, None, :].float()
        dev = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=dev, enabled=False):
            angles = (freq_col.float() @ pos_row.float()).transpose(2, 3)
            angles = self._interleave_axes(angles, self.mrope_section)
            table = torch.cat((angles, angles), dim=-1)
            cos = table.cos() * self.attention_scaling
            sin = table.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def _interleave_axes(angles, section):
        """Fold the H and W axes into the T table at every 2nd / 3rd frequency slot."""
        merged = angles[0]
        for axis, offset in ((1, 1), (2, 2)):
            take = slice(offset, section[axis] * 3, 3)
            merged[..., take] = angles[axis, ..., take]
        return merged


class AgnesVisionRotary(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


def _apply_vision_rope(q, k, cos, sin):
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q = (q * cos) + (_swap_halves(q) * sin)
    k = (k * cos) + (_swap_halves(k) * sin)
    return q.to(q_dtype), k.to(k_dtype)


# ---------------------------------------------------------------------------
# Global (softmax) attention
# ---------------------------------------------------------------------------


def _broadcast_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Repeat each kv head `groups` times along the head axis."""
    b, h, t, d = x.shape
    if groups == 1:
        return x
    return x[:, :, None, :, :].expand(b, h, groups, t, d).reshape(b, h * groups, t, d)


def _dense_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    k = _broadcast_kv(key, module.num_key_value_groups)
    v = _broadcast_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probs = nn.functional.dropout(probs, p=dropout, training=module.training)
    out = torch.matmul(probs, v).transpose(1, 2).contiguous()
    return out, probs


class AgnesGlobalAttention(nn.Module):
    """Grouped-query softmax attention with per-head q/k normalisation and a
    sigmoid output gate produced alongside the queries."""

    def __init__(self, config: AgnesTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        q_width = config.num_attention_heads * self.head_dim
        kv_width = config.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_width * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=config.attention_bias)
        self.q_norm = AgnesRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = AgnesRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        lead = hidden_states.shape[:-1]
        per_head = (*lead, -1, self.head_dim)

        q, gate = torch.chunk(self.q_proj(hidden_states).view(*lead, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*lead, -1)
        q = self.q_norm(q.view(per_head)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(per_head)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(per_head).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _apply_rope(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        attend: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, _dense_attention)
        out, probs = attend(
            self, q, k, v, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        out = out.reshape(*lead, -1).contiguous()
        out = out * torch.sigmoid(gate)
        return self.o_proj(out), probs


# ---------------------------------------------------------------------------
# Delta-rule (recurrent) attention
# ---------------------------------------------------------------------------


def _zero_padded_positions(hidden_states, attention_mask):
    """Mask out padding tokens so they leave no trace in the recurrent state."""
    if attention_mask is not None and attention_mask.shape[1] > 1:
        dt = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dt)
    return hidden_states


def _causal_conv_step(hidden_states, conv_state, weight, bias=None, activation=None):
    """One decode step of the depthwise causal conv, updating `conv_state` in place."""
    _, channels, steps = hidden_states.shape
    window = conv_state.shape[-1]
    joined = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(joined[:, :, -window:])
    y = F.conv1d(joined, weight.unsqueeze(1), bias, padding=0, groups=channels)
    y = F.silu(y[:, :, -steps:])
    return y.to(hidden_states.dtype)


def _unit_normalize(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _delta_rule_chunked(
    query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, **kwargs,
):
    """Chunk-parallel gated delta rule (prefill path)."""
    out_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _unit_normalize(query, dim=-1, eps=1e-6)
        key = _unit_normalize(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [t.transpose(1, 2).contiguous().to(torch.float32) for t in (query, key, value, beta, g)]

    bsz, heads, seq, dk = key.shape
    dv = value.shape[-1]
    pad = (chunk_size - seq % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    padded = seq + pad
    query = query * (1 / (query.shape[-1] ** 0.5))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        t.reshape(t.shape[0], t.shape[1], -1, chunk_size, t.shape[-1]) for t in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    solve = -((k_beta @ key.transpose(-1, -2)) * decay).masked_fill(upper, 0)
    for i in range(1, chunk_size):
        row = solve[..., i, :i].clone()
        block = solve[..., :i, :i].clone()
        solve[..., i, :i] = row + (row.unsqueeze(-1) * block).sum(-2)
    solve = solve + torch.eye(chunk_size, dtype=solve.dtype, device=solve.device)
    value = solve @ v_beta
    k_decayed = solve @ (k_beta * g.exp().unsqueeze(-1))
    state = (
        torch.zeros(bsz, heads, dk, dv, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    out = torch.zeros_like(value)

    for i in range(0, padded // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        local = q_i @ k_i.transpose(-1, -2) * decay[:, :, i]
        v_pred = (k_decayed[:, :, i]) @ state
        v_res = v_i - v_pred
        carried = (q_i * g[:, :, i, :, None].exp()) @ state
        out[:, :, i] = carried + local @ v_res
        state = (
            state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_res
        )

    if not output_final_state:
        state = None
    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])
    out = out[:, :, :seq]
    return out.transpose(1, 2).contiguous().to(out_dtype), state


def _delta_rule_stepwise(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False):
    """Token-by-token gated delta rule (decode path)."""
    out_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _unit_normalize(query, dim=-1, eps=1e-6)
        key = _unit_normalize(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [t.transpose(1, 2).contiguous().to(torch.float32) for t in (query, key, value, beta, g)]

    bsz, heads, seq, dk = key.shape
    dv = value.shape[-1]
    query = query * (1 / (query.shape[-1] ** 0.5))

    out = torch.zeros(bsz, heads, seq, dv, dtype=value.dtype, device=value.device)
    state = (
        torch.zeros(bsz, heads, dk, dv, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    for t in range(seq):
        q_t, k_t, v_t = query[:, :, t], key[:, :, t], value[:, :, t]
        decay_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, t].unsqueeze(-1)
        state = state * decay_t
        recalled = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        correction = (v_t - recalled) * beta_t
        state = state + k_t.unsqueeze(-1) * correction.unsqueeze(-2)
        out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        state = None
    return out.transpose(1, 2).contiguous().to(out_dtype), state


class AgnesDeltaAttention(nn.Module):
    """Gated delta-rule recurrent attention with a causal depthwise conv on
    the projected q/k/v and a gated RMS norm on the output."""

    def __init__(self, config: AgnesTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
            kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))

        # Preserve upstream's portable math without importing optional CUDA
        # packages during CPU/meta construction. Fused kernels are unqualified.
        self.norm = AgnesGatedNorm(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_fn = None
        self.causal_conv1d_update = _causal_conv_step
        self.chunk_gated_delta_rule = _delta_rule_chunked
        self.recurrent_gated_delta_rule = _delta_rule_stepwise

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        hidden_states = _zero_padded_positions(hidden_states, attention_mask)
        bsz, seq, _ = hidden_states.shape

        resume = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if resume:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            rec_state = cache_params.layers[self.layer_idx].recurrent_states[0]

        qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(bsz, seq, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if resume and seq == 1:
            qkv = self.causal_conv1d_update(qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)
        else:
            if cache_params is not None:
                context = cache_params.update_conv_state(
                    qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size,
                )
                # A fresh short prefill returns left-padded cache storage, not
                # current-token outputs. Only a resumed chunk needs that context.
                if resume:
                    qkv = context
            if self.causal_conv1d_fn is not None:
                qkv = self.causal_conv1d_fn(
                    x=qkv, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias,
                    activation=self.activation, seq_idx=kwargs.get("seq_idx"),
                )
            else:
                qkv = F.silu(self.conv1d(qkv)[:, :, : qkv.shape[-1]])
            if resume:
                qkv = qkv[:, :, -seq:]

        qkv = qkv.transpose(1, 2)
        query, key, value = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(bsz, seq, -1, self.head_k_dim)
        key = key.reshape(bsz, seq, -1, self.head_k_dim)
        value = value.reshape(bsz, seq, -1, self.head_v_dim)

        beta = b.sigmoid()
        # fp32 keeps exp(A_log) finite under fp16 weights
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        groups = self.num_v_heads // self.num_k_heads
        if groups > 1:
            query = query.repeat_interleave(groups, dim=2)
            key = key.repeat_interleave(groups, dim=2)

        if resume and seq == 1:
            mixed, rec_state = self.recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=rec_state,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
            )
        else:
            mixed, rec_state = self.chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=rec_state if resume else None,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.get("cu_seq_lens_q"),
            )
        if cache_params is not None:
            cache_params.update_recurrent_state(rec_state, self.layer_idx)

        mixed = self.norm(mixed.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(mixed.reshape(bsz, seq, -1))


# ---------------------------------------------------------------------------
# Decoder layer / base class
# ---------------------------------------------------------------------------


class AgnesDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: AgnesTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == LAYER_DELTA:
            self.delta_attn = AgnesDeltaAttention(config, layer_idx)
        elif self.layer_type == LAYER_GLOBAL:
            self.global_attn = AgnesGlobalAttention(config, layer_idx)
        self.mlp = AgnesMLP(config, config.intermediate_size, getattr(config, "parallel_ffn_intermediate_size", 0) or 0)
        self.input_layernorm = AgnesRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = AgnesRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        skip = hidden_states
        h = self.input_layernorm(hidden_states)
        if self.layer_type == LAYER_DELTA:
            h = self.delta_attn(hidden_states=h, cache_params=past_key_values, attention_mask=attention_mask, **kwargs)
        elif self.layer_type == LAYER_GLOBAL:
            h, _ = self.global_attn(
                hidden_states=h, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, position_embeddings=position_embeddings, **kwargs,
            )
        h = skip + h

        skip = h
        h = self.mlp(self.post_attention_layernorm(h))
        return skip + h


class AgnesPreTrainedModel(PreTrainedModel):
    config: AgnesConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AgnesDecoderLayer", "AgnesVisionBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    _can_record_outputs = {
        "hidden_states": AgnesDecoderLayer,
        "attentions": AgnesGlobalAttention,
    }
    _is_stateful = True

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length,
    ):
        if generation_config.cache_implementation not in (None, "dynamic"):
            raise ValueError("Agnes supports only its dynamic hybrid cache")
        cache = model_kwargs.get("past_key_values")
        if cache is not None and not isinstance(cache, AgnesDynamicCache):
            raise ValueError("Agnes requires AgnesDynamicCache")
        if generation_config.is_assistant:
            raise ValueError("Agnes recurrent cache rollback for assisted generation is not qualified")
        if generation_config.use_cache and cache is None:
            model_kwargs["past_key_values"] = AgnesDynamicCache(self.config.get_text_config())

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, AgnesDeltaAttention):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())
        elif isinstance(module, AgnesRMSNorm):
            # the norm scales by (1 + weight), so zero is the identity
            init.zeros_(module.weight)
        elif isinstance(module, AgnesVisionRotary):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------


class AgnesVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dt = self.proj.weight.dtype
        patches = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(patches.to(dtype=dt)).view(-1, self.embed_dim)


class AgnesVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class AgnesVisionAttention(nn.Module):
    def __init__(self, config: AgnesVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        n_tok = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(n_tok, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = _apply_vision_rope(q, k, cos, sin)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        attend: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, _dense_attention)
        drop = 0.0 if not self.training else self.attention_dropout
        if is_flash_attention_requested(self.config):
            longest = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            out, _ = attend(
                self, q, k, v, attention_mask=None, scaling=self.scaling, dropout=drop,
                cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens, max_length_q=longest, max_length_k=longest,
                is_causal=False, **kwargs,
            )
        else:
            spans = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            pieces = [torch.split(t, spans, dim=2) for t in (q, k, v)]
            out = torch.cat(
                [
                    attend(self, qi, ki, vi, attention_mask=None, scaling=self.scaling, dropout=drop, is_causal=False, **kwargs)[0]
                    for qi, ki, vi in zip(*pieces)
                ],
                dim=1,
            )
        return self.proj(out.reshape(n_tok, -1).contiguous())


class AgnesVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = AgnesVisionAttention(config=config)
        self.mlp = AgnesVisionMLP(config=config)

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        cu_seqlens (`torch.Tensor`):
            Cumulative sequence lengths used for packed variable-length attention.
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class AgnesVisionMerger(nn.Module):
    def __init__(self, config: AgnesVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class AgnesVisionModel(AgnesPreTrainedModel):
    config: AgnesVisionConfig
    config_class = AgnesVisionConfig
    input_modalities = ("image", "video")
    _can_record_outputs = {
        "hidden_states": AgnesVisionBlock,
        "attentions": AgnesVisionAttention,
    }
    _no_split_modules = ["AgnesVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = AgnesVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.rotary_pos_emb = AgnesVisionRotary((config.hidden_size // config.num_heads) // 2)
        self.blocks = nn.ModuleList([AgnesVisionBlock(config) for _ in range(config.depth)])
        self.merger = AgnesVisionMerger(config=config, use_postshuffle_norm=False)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                Flattened patches.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                Temporal / height / width extent of every image or video.
        """
        idx, w = get_vision_bilinear_indices_and_weights(
            grid_thw, num_grid_per_side=self.num_grid_per_side, spatial_merge_size=self.config.spatial_merge_size, kwargs=kwargs
        )
        pos_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens = get_vision_cu_seqlens(grid_thw, kwargs=kwargs)

        tokens = self.patch_embed(hidden_states)
        tokens = tokens + (self.pos_embed(idx) * w[:, :, None]).sum(0).to(tokens.dtype)
        angles = self.rotary_pos_emb(pos_ids)

        n_tok, _ = tokens.size()
        tokens = tokens.reshape(n_tok, -1)
        angles = angles.reshape(n_tok, -1)
        table = torch.cat((angles, angles), dim=-1)
        rope = (table.cos(), table.sin())

        for block in self.blocks:
            tokens = block(tokens, cu_seqlens=cu_seqlens, position_embeddings=rope, **kwargs)

        return BaseModelOutputWithPooling(last_hidden_state=tokens, pooler_output=self.merger(tokens))


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@auto_docstring
@dataclass
class AgnesModelOutput(BaseModelOutputWithPast):
    r"""
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        Offset between the multimodal rotary positions and plain sequence positions.
    """

    rope_deltas: torch.LongTensor | None = None


@auto_docstring
@dataclass
class AgnesCausalLMOutput(CausalLMOutputWithPast):
    r"""
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        Offset between the multimodal rotary positions and plain sequence positions.
    """

    rope_deltas: torch.LongTensor | None = None


def _text_positions_and_lengths(inputs_embeds, attention_mask, cache):
    """Current logical text positions and total valid lengths, before mutation.

    Keep padding counts with the request, so either a full-prefix mask or a
    chunk-local mask (or no subsequent mask) can extend its rotary coordinates.
    The scalar cache length still counts physical tokens for KV/mask indexing.
    """
    batch, seq, _ = inputs_embeds.shape
    past_len = 0 if cache is None else cache.get_seq_length()
    prefix = getattr(cache, "_agnes_valid_tokens", None)
    if prefix is None:
        prefix = torch.full((batch, 1), past_len, device=inputs_embeds.device, dtype=torch.long)
    if attention_mask is None:
        mask = torch.ones((batch, seq), device=inputs_embeds.device, dtype=torch.long)
    else:
        mask = attention_mask[:, -seq:].long().to(inputs_embeds.device)
        if attention_mask.shape[-1] > seq:
            prefix = attention_mask[:, :-seq].long().sum(-1, keepdim=True).to(inputs_embeds.device)
    positions = (mask.cumsum(-1) - 1 + prefix).masked_fill(mask == 0, 0)
    return positions, prefix + mask.sum(-1, keepdim=True)


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------


class AgnesTextModel(AgnesPreTrainedModel):
    config: AgnesTextConfig
    config_class = AgnesTextConfig

    def __init__(self, config: AgnesTextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([AgnesDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = AgnesRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = AgnesRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        _defer_cache_commit: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = AgnesDynamicCache(self.config)
        if past_key_values is not None and not isinstance(past_key_values, AgnesDynamicCache):
            raise ValueError("Agnes requires AgnesDynamicCache for its delta/global layer plan")
        if past_key_values is not None and past_key_values._in_forward:
            raise ValueError("Agnes cache has an incomplete forward; call reset() before retrying")

        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            if attention_mask is None or attention_mask.shape[-1] == inputs_embeds.shape[1]:
                current_mask = (torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
                                if attention_mask is None else attention_mask)
                attention_mask = torch.cat((past_key_values._agnes_attention_mask.to(current_mask.device), current_mask), dim=-1)

        text_positions, valid_lengths = _text_positions_and_lengths(inputs_embeds, attention_mask, past_key_values)
        # position ids carry four rows: plain text positions, then T / H / W
        if position_ids is None:
            position_ids = text_positions.unsqueeze(0).expand(4, -1, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_positions, position_ids = position_ids[0], position_ids[1:]
        else:
            text_positions = None

        global_mask = create_causal_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=text_positions,
        ) if LAYER_GLOBAL in self.config.layer_types else None
        delta_mask = self._recurrent_mask(attention_mask, past_key_values)

        h = inputs_embeds
        rope = self.rotary_emb(h, position_ids)
        if past_key_values is not None:
            # A failed layer can have partially advanced recurrent/KV states.
            # Keep the cache invalid until reset rather than silently reusing it.
            past_key_values._in_forward = True
        for i, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            mask = delta_mask if self.config.layer_types[i] == LAYER_DELTA else global_mask
            h = layer(
                h, position_embeddings=rope, attention_mask=mask, position_ids=text_positions,
                past_key_values=past_key_values, use_cache=use_cache, **kwargs,
            )
        h = self.norm(h)
        if past_key_values is not None:
            past_key_values._agnes_seen_tokens += inputs_embeds.shape[1]
            past_key_values._agnes_valid_tokens = valid_lengths
            past_key_values._agnes_attention_mask = (attention_mask.clone() if attention_mask is not None else
                torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long))
            if not _defer_cache_commit:
                past_key_values._in_forward = False
        return AgnesModelOutput(last_hidden_state=h, past_key_values=past_key_values)

    @staticmethod
    def _recurrent_mask(attention_mask, past_key_values):
        """Padding mask for the recurrent layers (left padding).  Not needed
        once a cache holds prior state, or when nothing is masked."""
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            return None
        if attention_mask is not None and torch.all(attention_mask == 1):
            return None
        return attention_mask


@auto_docstring
class AgnesModel(AgnesPreTrainedModel):
    config: AgnesConfig
    config_class = AgnesConfig
    base_model_prefix = "model"
    accepts_loss_kwargs = False
    _no_split_modules = ["AgnesDecoderLayer", "AgnesVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = AgnesVisionModel(config.vision_config)
        self.language_model = AgnesTextModel(config.text_config)
        self.post_init()

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list[int, int, int] | torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ):
        """Three-axis positions for the tokens of one image / video, offset by `start_position`."""
        n_t = grid_thw[0].item() // temp_merge_size
        n_h = grid_thw[1].item() // spatial_merge_size
        n_w = grid_thw[2].item() // spatial_merge_size

        t_axis = torch.arange(n_t, device=device) * time_interval
        w_axis = torch.arange(n_w, device=device) + start_position
        h_axis = torch.arange(n_h, device=device) + start_position

        # repeat patterns define the raster order; keep them as is
        w_axis = w_axis.repeat(n_h * n_t)
        h_axis = h_axis.repeat_interleave(n_w).repeat(n_t)
        t_axis = t_axis.repeat_interleave(n_h * n_w) + start_position
        return torch.stack([t_axis, h_axis, w_axis], dim=0)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (3, batch, seq) rotary positions: text runs advance all three
        axes together, vision runs get grid positions.  Videos are split per
        frame because frames are separated by timestamp tokens."""
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        merge = self.config.vision_config.spatial_merge_size

        deltas = []
        position_ids = torch.zeros(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
        grids = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for b, ids in enumerate(input_ids):
            kinds = mm_token_type_ids[b]
            if attention_mask is not None:
                keep = attention_mask[b].bool()
                ids, kinds = ids[keep], kinds[keep]

            runs = []
            for kind, members in itertools.groupby(enumerate(kinds.tolist()), lambda x: x[1]):
                members = list(members)
                runs.append((kind, members[0][0], members[-1][0] + 1))

            cursor = 0
            pieces = []
            for kind, lo, hi in runs:
                if kind == 0:
                    n = hi - lo
                    pieces.append(torch.arange(n, device=input_ids.device).view(1, -1).expand(3, -1) + cursor)
                    cursor += n
                else:
                    thw = next(grids[kind])
                    pieces.append(self.get_vision_position_ids(cursor, thw, 1, merge, device=input_ids.device))
                    cursor += max(thw[1], thw[2]) // merge
            pos = torch.cat(pieces, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, b, attention_mask[b].bool()] = pos.to(position_ids.device)
            else:
                position_ids[:, b] = pos.to(position_ids.device)
            deltas.append(pos.max() + 1 - len(ids))
        return position_ids, torch.tensor(deltas, device=input_ids.device).unsqueeze(1)

    @accepts_precomputed_kwargs(modality="video")
    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            Video frames as patches.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal / height / width extent of every video.
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            Images as patches.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal / height / width extent of every image.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        vis: BaseModelOutputWithPooling = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        per_image = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vis.pooler_output = torch.split(vis.pooler_output, per_image)
        return vis

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        """Boolean masks over the image / video placeholder tokens, checked
        against the number of visual features supplied."""
        if input_ids is None:
            embed = self.get_input_embeddings()
            dev = inputs_embeds.device
            image_mask = (inputs_embeds == embed(torch.tensor(self.config.image_token_id, dtype=torch.long, device=dev))).all(-1)
            video_mask = (inputs_embeds == embed(torch.tensor(self.config.video_token_id, dtype=torch.long, device=dev))).all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id
            video_mask = input_ids == self.config.video_token_id

        n_img = image_mask.sum()
        image_mask = image_mask.unsqueeze(-1).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                n_img * inputs_embeds.shape[-1] == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_img}, features: {image_features.shape[0]}",
            )
        n_vid = video_mask.sum()
        video_mask = video_mask.unsqueeze(-1).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                n_vid * inputs_embeds.shape[-1] == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_vid}, features: {video_features.shape[0]}",
            )
        return image_mask, video_mask

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        past_len = 0 if past_key_values is None else past_key_values.get_seq_length()
        rope_deltas = getattr(past_key_values, "rope_deltas", None)
        has_vision = image_grid_thw is not None or video_grid_thw is not None
        if has_vision and mm_token_type_ids is None and input_ids is not None:
            raise ValueError(
                "Multimodal data was passed (via `image_grid_thw` or `video_grid_thw`) but `mm_token_type_ids` is "
                "missing. Pass `mm_token_type_ids` (returned by the processor) so the 3-D rotary positions can be built."
            )
        text_positions, total_valid = _text_positions_and_lengths(inputs_embeds, attention_mask, past_key_values)
        seq = inputs_embeds.shape[1]
        current_mask = attention_mask[:, -seq:] if attention_mask is not None else None
        fresh = input_ids is not None and mm_token_type_ids is not None and has_vision
        if fresh:
            # Media may arrive in any chunk, including after an earlier image or
            # video. Build its local 3-D coordinates and translate by the prior
            # rotary cursor, not the physical number of cached tokens alone.
            position_ids, local_delta = self.get_rope_index(
                input_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                attention_mask=current_mask, mm_token_type_ids=mm_token_type_ids[:, -seq:],
            )
            previous_delta = rope_deltas if past_len and rope_deltas is not None else torch.zeros_like(local_delta)
            current_valid = current_mask.long().sum(-1, keepdim=True) if current_mask is not None else seq
            start = total_valid - current_valid + previous_delta
            position_ids = position_ids + start.unsqueeze(0)
            if current_mask is not None:
                position_ids = position_ids.masked_fill(~current_mask.bool().unsqueeze(0), 0)
            return position_ids, previous_delta + local_delta
        # continuing generation, or embeds-only input: reuse the stored deltas
        if rope_deltas is not None and (past_len > 0 or input_ids is None):
            bsz = inputs_embeds.shape[0]
            delta = rope_deltas.repeat_interleave(bsz // rope_deltas.shape[0], dim=0)
            position_ids = (text_positions + delta.to(inputs_embeds.device)).unsqueeze(0).expand(3, -1, -1)
            return position_ids, rope_deltas
        return None, None

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | AgnesModelOutput:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal / height / width extent of every image.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal / height / width extent of every video.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            feats = torch.cat(
                self.get_image_features(pixel_values, image_grid_thw, return_dict=True, **kwargs).pooler_output, dim=0
            ).to(inputs_embeds.device, inputs_embeds.dtype)
            mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=feats)
            inputs_embeds = inputs_embeds.masked_scatter(mask, feats)

        if pixel_values_videos is not None:
            feats = torch.cat(
                self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True, **kwargs).pooler_output, dim=0
            ).to(inputs_embeds.device, inputs_embeds.dtype)
            _, mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=feats)
            inputs_embeds = inputs_embeds.masked_scatter(mask, feats)

        if position_ids is None:
            position_ids, rope_deltas = self.compute_3d_position_ids(
                input_ids=input_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )
        else:
            # Explicit positions are authoritative, including when a caller
            # supplies the full-sequence slice for newly appended media.
            axes = position_ids[1:] if position_ids.ndim == 3 and position_ids.shape[0] == 4 else position_ids
            if axes.ndim == 2:
                axes = axes.unsqueeze(0)
            _, total_valid = _text_positions_and_lengths(inputs_embeds, attention_mask, past_key_values)
            rope_deltas = axes.amax(dim=(0, 2)).unsqueeze(1) + 1 - total_valid

        out = self.language_model(
            input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs,
        )
        if out.past_key_values is not None:
            out.past_key_values.rope_deltas = rope_deltas
        return AgnesModelOutput(**out, rope_deltas=rope_deltas)


@auto_docstring
class AgnesForCausalLM(AgnesPreTrainedModel, GenerationMixin):
    """Text-only head over `AgnesTextModel`."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: AgnesTextConfig
    config_class = AgnesTextConfig
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]

    def __init__(self, config):
        super().__init__(config)
        self.model = AgnesTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Targets for the language-modelling loss; -100 marks ignored positions.
        """
        out: BaseModelOutputWithPast = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
            _defer_cache_commit=True,
        )
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(out.last_hidden_state[:, keep, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        result = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=out.past_key_values,
            hidden_states=out.hidden_states, attentions=out.attentions,
        )
        if out.past_key_values is not None:
            out.past_key_values._in_forward = False
        return result


class AgnesForConditionalGeneration(AgnesPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False
    config: AgnesConfig
    config_class = AgnesConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AgnesModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            Video frames as patches.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal / height / width extent of every video.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            Images as patches.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal / height / width extent of every image.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | AgnesCausalLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Targets for the language-modelling loss; -100 marks ignored positions.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal / height / width extent of every image.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal / height / width extent of every video.
        """
        out = self.model(
            input_ids=input_ids, pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, position_ids=position_ids,
            attention_mask=attention_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids, **kwargs,
            _defer_cache_commit=True,
        )
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(out[0][:, keep, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
        result = AgnesCausalLMOutput(
            loss=loss, logits=logits, past_key_values=out.past_key_values,
            hidden_states=out.hidden_states, attentions=out.attentions, rope_deltas=out.rope_deltas,
        )
        if out.past_key_values is not None:
            out.past_key_values._in_forward = False
        return result

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # pixels are consumed on the first step only; later steps run on the cache
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            position_ids=position_ids, pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, use_cache=use_cache,
            is_first_iteration=is_first_iteration, **kwargs,
        )
        if not is_first_iteration and use_cache:
            inputs["pixel_values"] = None
            inputs["pixel_values_videos"] = None
        return inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # four-row positions: text row on top of the three rotary axes
        text_pos = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        cache = model_kwargs.get("past_key_values")
        past_len = 0 if cache is None else cache.get_seq_length()
        rope_deltas = getattr(cache, "rope_deltas", None)
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]
        is_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        has_vision = model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None
        mm_types = model_kwargs.get("mm_token_type_ids")
        if past_len and is_ids and has_vision and mm_types is not None:
            # Generation supplies full IDs; the cache already owns the prefix.
            # Only the unconsumed chunk carries new pixels/grids and needs fresh
            # spatial coordinates. Keep the prefix slots for HF's later slicing.
            current_ids = inputs_tensor[:, past_len:]
            current_types = mm_types[:, -current_ids.shape[1]:]
            if current_ids.shape[1] and current_types.ne(0).any():
                current_axes, _ = self.model.compute_3d_position_ids(
                    current_ids, self.get_input_embeddings()(current_ids),
                    image_grid_thw=model_kwargs.get("image_grid_thw"), video_grid_thw=model_kwargs.get("video_grid_thw"),
                    attention_mask=model_kwargs.get("attention_mask"), past_key_values=cache,
                    mm_token_type_ids=current_types,
                )
                axes = text_pos.unsqueeze(0).expand(3, -1, -1).clone()
                axes[:, :, -current_ids.shape[1]:] = current_axes
                return torch.cat([text_pos[None, ...], axes], dim=0)
        if past_len != 0 and rope_deltas is not None:
            return text_pos[None, ...] + rope_deltas

        if is_ids and model_kwargs.get("mm_token_type_ids") is not None and has_vision:
            rest = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
            axes, rope_deltas = self.model.get_rope_index(inputs_tensor, **rest)
        else:
            axes = text_pos.unsqueeze(0).expand(3, -1, -1)
            rope_deltas = torch.zeros(inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device)
        if cache is not None:
            cache.rope_deltas = rope_deltas
        return torch.cat([text_pos[None, ...], axes], dim=0)

    def _count_images_and_videos(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample number of images and of video frames, read off the
        vision-start / placeholder tokens."""
        img_id = self.config.image_token_id
        vid_id = self.config.video_token_id
        start_id = self.config.vision_start_token_id
        if inputs_embeds is not None:
            embed = self.get_input_embeddings()
            dev = inputs_embeds.device
            start = (inputs_embeds == embed(torch.tensor(start_id, dtype=torch.long, device=dev)))[..., 0]
            img = (inputs_embeds == embed(torch.tensor(img_id, dtype=torch.long, device=dev)))[..., 0]
            vid = (inputs_embeds == embed(torch.tensor(vid_id, dtype=torch.long, device=dev)))[..., 0]
        else:
            start = input_ids == start_id
            img = input_ids == img_id
            vid = input_ids == vid_id
        after_start = torch.roll(start, shifts=1, dims=1)
        return torch.sum(after_start & img, dim=1), torch.sum(after_start & vid, dim=1)

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # visual tensors have no batch axis (they are concatenated over samples),
        # so they are expanded sample by sample using the per-sample counts
        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

        def expand_visual(d):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            n_img, n_vid = self._count_images_and_videos(input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None))

            # n_vid counts frames (each frame carries a vision-start token); fold back to videos
            if video_grid_thw is not None:
                frame_cum = torch.cumsum(video_grid_thw[:, 0], dim=0)
                token_cum = torch.cumsum(n_vid, dim=0)
                edges = torch.searchsorted(frame_cum, token_cum)
                n_vid = torch.diff(torch.cat([-edges.new_ones(1), edges]))

            def tile(x, lengths, times):
                parts = torch.split(x, lengths)
                reps = [times] + [1] * (x.dim() - 1)
                return torch.cat([p.repeat(*reps) for p in parts], dim=0)

            for key in d:
                if key == "pixel_values":
                    per = torch.split(image_grid_thw, list(n_img))
                    d[key] = tile(d[key], [torch.prod(s, dim=1).sum() for s in per], expand_size)
                elif key == "image_grid_thw":
                    d[key] = tile(d[key], list(n_img), expand_size)
                elif key == "pixel_values_videos":
                    per = torch.split(video_grid_thw, list(n_vid))
                    d[key] = tile(d[key], [torch.prod(s, dim=1).sum() for s in per], expand_size)
                elif key == "video_grid_thw":
                    d[key] = tile(d[key], list(n_vid), expand_size)
            return d

        def expand_rest(d):
            for key in d:
                if key == "position_ids" and d[key].ndim == 3:
                    d[key] = d[key].repeat_interleave(expand_size, dim=1)
                elif d[key] is not None and isinstance(d[key], torch.Tensor) and key not in visual_keys:
                    d[key] = d[key].repeat_interleave(expand_size, dim=0)
            return d

        model_kwargs = expand_visual(model_kwargs)
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        model_kwargs = expand_rest(model_kwargs)
        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = expand_rest(model_kwargs["encoder_outputs"])
        return input_ids, model_kwargs


__all__ = [
    "AgnesPreTrainedModel",
    "AgnesVisionModel",
    "AgnesTextModel",
    "AgnesModel",
    "AgnesForCausalLM",
    "AgnesForConditionalGeneration",
]
