# SPDX-License-Identifier: Apache-2.0
"""Hugging Face ``transformers`` implementation of raon-1-mini.

Architecture (released 28.8B / 4.3B-active checkpoint): 25 decoder layers on a 4-stream Raon_mHC
residual, 18 Raon Delta Attention (RDA) layers + 7 NoPE gated
Multi-head Latent Attention (MLA) layers, a dense SiTU-GLU MLP in layer 0 and a 128-expert
top-6 sigmoid-routed MoE with one shared expert everywhere else, and an untied output head.

Every equation follows the training implementation's plain-PyTorch reference:

* Raon_mHC selector, router, SiTU nonlinearity, attention softmax, RDA gates and the RDA recurrent
  state are computed in fp32; GEMM activations live in the model dtype.
* RDA prefill/decode use a pure-PyTorch recurrence; no custom kernel package is required.
* The parameter names are the release checkpoint's tensor names.
* The released checkpoint stores routed experts as MXFP8 (``quant_method: mxfp8``) and loads them
  as E4M3 weights and byte exponents. This port uses portable per-call expansion, not
  upstream's unqualified native Blackwell grouped GEMM or process-global quantizer patch.
"""

from __future__ import annotations

import copy
import math
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_raon1_mini import Raon1MiniConfig


logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------------------------
class Raon1MiniRMSNorm(nn.Module):
    """RMSNorm with fp32 statistics, plain (non zero-centered) gamma, no bias."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden: torch.Tensor, *, keep_fp32: bool = False) -> torch.Tensor:
        value = hidden.float()
        normed = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return normed if keep_fp32 else normed.to(hidden.dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


def situ_glu(gate: torch.Tensor, value: torch.Tensor, gate_beta: float, value_beta: float) -> torch.Tensor:
    """SiTU-GLU in fp32: ``gate_beta*tanh(g/gate_beta)*sigmoid(g) * value_beta*tanh(v/value_beta)``."""
    g = gate.float()
    v = value.float()
    return (gate_beta * torch.tanh(g / gate_beta) * torch.sigmoid(g)) * (value_beta * torch.tanh(v / value_beta))


def helmert_basis(n: int) -> torch.Tensor:
    """Orthonormal ``Q`` in ``R^{n x (n-1)}`` with ``Q^T Q = I`` and ``Q^T 1 = 0``."""
    q = torch.zeros(n, n - 1, dtype=torch.float32)
    for k in range(1, n):
        q[:k, k - 1] = 1.0
        q[k, k - 1] = -float(k)
        q[:, k - 1] /= math.sqrt(k * (k + 1.0))
    return q


def _raon_mhc_constants(n: int, device: torch.device):
    eye = torch.eye(n, dtype=torch.float32, device=device)
    return helmert_basis(n).to(device), eye, 0.5 * (eye + torch.full_like(eye, 1.0 / n))


# ---------------------------------------------------------------------------------------------
# Raon_mHC multi-stream residual
# ---------------------------------------------------------------------------------------------
class RaonMHCStreamMix(nn.Module):
    """The ``res.*`` parameters of one Raon_mHC site: the stream-to-stream residual chart."""

    def __init__(self, num_streams: int, hidden_size: int) -> None:
        super().__init__()
        m = (num_streams - 1) ** 2
        self.phi_z = nn.Parameter(torch.zeros(m, num_streams * hidden_size))
        self.alpha_z = nn.Parameter(torch.ones(()))
        self.b_z = nn.Parameter(torch.zeros(m))
        self.eta = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.phi_z, self.alpha_z, self.b_z, self.eta


class RaonMHC(nn.Module):
    """One Raon_mHC site (attention or MLP): selects the branch input from the streams and writes back.

    ``route`` returns fp32 coefficients ``h_pre [N, n]``, ``h_post [N, n]`` and ``h_res [N, n, n]``
    for ``N = batch*seq`` rows (Raon_mHC chart:
    ``E = Q Z Q^T``, ``rho = max(0, max(-E/B))``, ``Phi = B + E/(1+rho)``, ``H = (1-gamma) I + gamma Phi``
    with ``B = (I + 11^T/n)/2`` and ``gamma = sigmoid(eta)``).
    """

    def __init__(self, config: Raon1MiniConfig) -> None:
        super().__init__()
        n, c = config.raon_mhc_streams, config.hidden_size
        self.num_streams = n
        self.eps = config.raon_mhc_selector_eps
        self.post_factor = config.raon_mhc_post_factor
        self.phi_pre = nn.Parameter(torch.zeros(n, n * c))
        self.phi_post = nn.Parameter(torch.zeros(n, n * c))
        self.alpha_pre = nn.Parameter(torch.ones(()))
        self.alpha_post = nn.Parameter(torch.ones(()))
        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.res = RaonMHCStreamMix(n, c)

    def route(self, stream: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(stream)

    def forward(self, stream: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi_z, alpha_z, b_z, eta = self.res()
        n = self.num_streams
        rows = stream.shape[0] * stream.shape[1]
        flat = stream.reshape(rows, n * stream.shape[-1])
        weight = torch.cat([self.phi_pre, self.phi_post, phi_z], dim=0)
        # RMS scale applied AFTER the (model-dtype) selector GEMM, as in training.
        inv = torch.rsqrt(flat.float().pow(2).mean(-1, keepdim=True) + self.eps)
        z = F.linear(flat, weight.to(flat.dtype)).float() * inv
        t_pre = self.alpha_pre.float() * z[:, :n] + self.b_pre.float()
        t_post = self.alpha_post.float() * z[:, n : 2 * n] + self.b_post.float()
        t_z = alpha_z.float() * z[:, 2 * n :] + b_z.float()
        h_pre = torch.sigmoid(t_pre)
        h_post = self.post_factor * torch.sigmoid(t_post)
        nm = n - 1
        helmert, eye, chart_b = _raon_mhc_constants(n, z.device)
        e = torch.einsum("im,bmk,jk->bij", helmert, t_z.reshape(rows, nm, nm), helmert)
        rho = (-e / chart_b).amax(dim=(-2, -1)).clamp_min(0.0)
        phi = chart_b + e / (1.0 + rho)[:, None, None]
        gamma = torch.sigmoid(eta.float())
        h_res = (1.0 - gamma) * eye + gamma * phi
        return h_pre, h_post, h_res

    @staticmethod
    def aggregate(stream_f32: torch.Tensor, h_pre: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``[N, n, C]`` fp32 streams -> ``[N, C]`` branch input in the model dtype."""
        return torch.einsum("bn,bnc->bc", h_pre, stream_f32).to(dtype)

    @staticmethod
    def mix(
        stream_f32: torch.Tensor,
        branch_out: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Residual mixing plus the branch write-back, rounded once to the model dtype."""
        mixed = torch.einsum("bij,bjc->bic", h_res, stream_f32)
        return (mixed + h_post.unsqueeze(-1) * branch_out.float().unsqueeze(1)).to(dtype)


# ---------------------------------------------------------------------------------------------
# RDA (Raon Delta Attention)
# ---------------------------------------------------------------------------------------------
class Raon1MiniShortConv(nn.Module):
    """Depthwise causal conv (no bias) whose weight is stored squeezed as ``[channels, kernel]``."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, kernel_size))
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """``x [B, T, C]`` raw rows -> silu(conv) ``[B, T, C]`` in x's dtype, plus the new raw window ``[B, C, K-1]``."""
        channels = x.shape[-1]
        k = self.kernel_size
        signal = x.transpose(1, 2)  # [B, C, T]
        padded = F.pad(signal, (k - 1, 0)) if state is None else torch.cat([state.to(signal.dtype), signal], dim=-1)
        # fp32 accumulation and silu, one rounding on the output (causal_conv1d semantics).
        out = F.conv1d(padded.float(), self.weight.float().unsqueeze(1), groups=channels)
        new_state = padded[..., -(k - 1) :] if k > 1 else padded[..., :0]
        return F.silu(out).to(x.dtype).transpose(1, 2), new_state


def torch_recurrent_rda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V-first delta rule, token serial, fp32 state ``[B, H, V, K]``. Reference for the fused kernels.

    ``q/k/v [B, T, H, D]`` (q already l2-normalised and scaled, k l2-normalised), ``log_alpha [B, T, H, K]``
    fp32 log decay, ``beta [B, T, H]``. Returns ``out [B, T, H, V]`` in v's dtype and the final state.
    """
    batch, seq, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    if state is None:
        state = torch.zeros(batch, heads, value_dim, key_dim, dtype=torch.float32, device=q.device)
    else:
        state = state.float()
    out = torch.empty(batch, seq, heads, value_dim, dtype=torch.float32, device=q.device)
    qf, kf, vf, bf = q.float(), k.float(), v.float(), beta.float()
    for t in range(seq):
        state = state * torch.exp(log_alpha[:, t]).unsqueeze(-2)
        predicted = torch.einsum("bhvk,bhk->bhv", state, kf[:, t])
        error = vf[:, t] - predicted
        state = state + bf[:, t][..., None, None] * torch.einsum("bhv,bhk->bhvk", error, kf[:, t])
        out[:, t] = torch.einsum("bhvk,bhk->bhv", state, qf[:, t])
    return out.to(v.dtype), state


class Raon1MiniRDA(nn.Module):
    """Raon Delta Attention layer: gated delta-rule linear attention with short convolutions and a write gate."""

    def __init__(self, config: Raon1MiniConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.rda_num_heads
        self.head_dim = config.rda_head_dim
        self.proj_size = self.num_heads * self.head_dim
        self.gate_lower_bound = config.rda_gate_lower_bound
        self.norm_eps = config.rms_norm_eps
        kernel = config.rda_conv_kernel_size
        self.q_proj = nn.Linear(hidden, self.proj_size, bias=False)
        self.k_proj = nn.Linear(hidden, self.proj_size, bias=False)
        self.v_proj = nn.Linear(hidden, self.proj_size, bias=False)
        self.q_conv1d = Raon1MiniShortConv(self.proj_size, kernel)
        self.k_conv1d = Raon1MiniShortConv(self.proj_size, kernel)
        self.v_conv1d = Raon1MiniShortConv(self.proj_size, kernel)
        self.f_a_proj = nn.Linear(hidden, config.rda_decay_rank, bias=False)
        self.f_b_proj = nn.Linear(config.rda_decay_rank, self.proj_size, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.rda_w_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.g_proj = nn.Linear(hidden, self.proj_size, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(self.proj_size, dtype=torch.float32))
        self.o_norm = Raon1MiniRMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(self.proj_size, hidden, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        cache: Raon1MiniDynamicCache | None,
        layer_idx: int,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        dtype = hidden.dtype
        if token_mask is not None:
            # Left padding: padded rows must not enter the conv window or the recurrence.
            hidden = hidden * token_mask[..., None].to(dtype)
        q = self.q_proj(hidden)
        k = self.k_proj(hidden)
        v = self.v_proj(hidden)
        gate = self.g_proj(hidden)
        beta_raw = self.b_proj(hidden)
        rda_delta = self.rda_w_proj(hidden)
        decay_raw = self.f_b_proj(self.f_a_proj(hidden))

        conv_state = cache.conv_states[layer_idx] if cache is not None else None
        q_hist = k_hist = v_hist = None
        if conv_state is not None:
            q_hist, k_hist, v_hist = conv_state.split(self.proj_size, dim=1)
        q, q_hist = self.q_conv1d(q, q_hist)
        k, k_hist = self.k_conv1d(k, k_hist)
        v, v_hist = self.v_conv1d(v, v_hist)
        if cache is not None:
            cache.conv_states[layer_idx] = torch.cat([q_hist, k_hist, v_hist], dim=1)

        q = q.reshape(batch, seq, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq, self.num_heads, self.head_dim)
        v = v.reshape(batch, seq, self.num_heads, self.head_dim)
        # _prepare_after_conv: l2-normalise q/k in fp32 (eps 1e-6), scale q by key_dim**-0.5.
        q = F.normalize(q.float(), dim=-1, eps=1e-6).to(dtype) * (self.head_dim**-0.5)
        k = F.normalize(k.float(), dim=-1, eps=1e-6).to(dtype)
        # Log decay (per-head scale, lower bound): fp32.
        decay = decay_raw.float().reshape(batch, seq, self.num_heads, self.head_dim) + self.dt_bias.float().reshape(
            self.num_heads, self.head_dim
        )
        scaled = torch.exp(self.A_log.float())[None, None, :, None] * decay
        log_alpha = (
            -F.softplus(scaled) if self.gate_lower_bound is None else self.gate_lower_bound * torch.sigmoid(scaled)
        )
        beta = torch.sigmoid(beta_raw.float()).to(dtype)
        # RDA: write gate sigmoid(beta+delta), erase gate sigmoid(beta); realised as a fp32 value scale.
        ratio = torch.exp(F.logsigmoid(beta_raw.float() + rda_delta.float()) - F.logsigmoid(beta_raw.float()))
        v = (v.float() * ratio.unsqueeze(-1)).to(dtype)

        recurrent_state = cache.recurrent_states[layer_idx] if cache is not None else None
        out, final_state = self._core(q, k, v, log_alpha.contiguous(), beta, recurrent_state)
        if cache is not None:
            cache.recurrent_states[layer_idx] = final_state
        # Per-head RMSNorm then sigmoid output gate, both in fp32, one rounding.
        normed = self.o_norm(out.reshape(-1, self.head_dim), keep_fp32=True)
        gated = (normed * torch.sigmoid(gate.float().reshape(-1, self.head_dim))).to(dtype)
        return self.o_proj(gated.reshape(batch, seq, self.proj_size))

    def _core(self, q, k, v, log_alpha, beta, state):
        return torch_recurrent_rda(q, k, v, log_alpha, beta, state)


# ---------------------------------------------------------------------------------------------
# NoPE gated MLA
# ---------------------------------------------------------------------------------------------
class Raon1MiniMLA(nn.Module):
    """Multi-head latent attention without positional encoding and with a sigmoid output gate.

    The 64 "rope" channels are never rotated: they are extra key/query channels shared across heads
    (``k_pe`` is expanded over heads), so scores are 192-wide dot products scaled by ``1/sqrt(192)``.
    """

    def __init__(self, config: Raon1MiniConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_nope = config.qk_nope_head_dim
        self.qk_pe = config.qk_rope_head_dim
        self.v_dim = config.v_head_dim
        self.config = config
        self.q_head_dim = self.qk_nope + self.qk_pe
        self.kv_lora_rank = config.kv_lora_rank
        self.softmax_scale = self.q_head_dim**-0.5
        self.use_output_gate = config.mla_use_output_gate
        self.q_a_proj = nn.Linear(hidden, config.q_lora_rank, bias=False)
        self.q_a_layernorm = Raon1MiniRMSNorm(config.q_lora_rank, config.mla_latent_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden, config.kv_lora_rank + self.qk_pe, bias=False)
        self.kv_a_layernorm = Raon1MiniRMSNorm(config.kv_lora_rank, config.mla_latent_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.qk_nope + self.v_dim),
            bias=False,
        )
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden, self.num_heads * self.v_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_dim, hidden, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        cache: Raon1MiniDynamicCache | None,
        layer_idx: int,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        dtype = hidden.dtype
        q_lat = self.q_a_layernorm(self.q_a_proj(hidden))
        q = self.q_b_proj(q_lat).reshape(batch, seq, self.num_heads, self.q_head_dim)
        kv_full = self.kv_a_proj_with_mqa(hidden)
        kv_lat, k_pe = torch.split(kv_full, [self.kv_lora_rank, self.qk_pe], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_lat)).reshape(batch, seq, self.num_heads, self.qk_nope + self.v_dim)
        k_nope, value = torch.split(kv, [self.qk_nope, self.v_dim], dim=-1)
        key = torch.cat(
            [k_nope, k_pe.unsqueeze(2).expand(batch, seq, self.num_heads, self.qk_pe)],
            dim=-1,
        )

        query = q.transpose(1, 2)  # [B, H, T, 192]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if cache is not None:
            key, value = cache.update(key, value, layer_idx)
        kv_len = key.shape[2]
        if attention_mask is None and seq > 1 and kv_len != seq:
            # Chunked prefill after cached tokens: explicit causal mask with the cache offset.
            offset = kv_len - seq
            attention_mask = torch.ones(seq, kv_len, dtype=torch.bool, device=hidden.device).tril(offset)[None, None]
        if self.config._attn_implementation == "eager":
            scores = query.float() @ key.float().transpose(-1, -2) * self.softmax_scale
            if attention_mask is None and seq > 1:
                attention_mask = torch.ones(seq, kv_len, dtype=torch.bool, device=hidden.device).tril(kv_len - seq)
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~attention_mask, float("-inf"))
                else:
                    scores = scores + attention_mask
            context = torch.nan_to_num(scores.softmax(-1)).to(dtype) @ value
        else:
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=attention_mask is None and seq > 1,
                scale=self.softmax_scale,
            )
        context = context.transpose(1, 2).reshape(batch, seq, self.num_heads * self.v_dim)
        if self.use_output_gate:
            # The training NoPE path evaluates the gate in the activation dtype.
            context = context * torch.sigmoid(self.g_proj(hidden))
        return self.o_proj(context.to(dtype))


# ---------------------------------------------------------------------------------------------
# MLPs
# ---------------------------------------------------------------------------------------------
class Raon1MiniMLP(nn.Module):
    """SiTU-GLU MLP: ``down(situ_glu(gate(x), up(x)))``. Used for the dense layer and the shared expert."""

    def __init__(self, config: Raon1MiniConfig, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.gate_beta = config.activation_situ_beta
        self.value_beta = config.activation_situ_linear_beta

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        act = situ_glu(
            self.gate_proj(hidden),
            self.up_proj(hidden),
            self.gate_beta,
            self.value_beta,
        )
        return self.down_proj(act.to(hidden.dtype))


def dequantize_mxfp8(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Expand E4M3 blocks using E8M0 exponents (255 is NaN), then cast to the activation dtype."""
    rows, cols = weight.shape
    blocks = weight.float().view(rows, cols // 32, 32)
    exponents = _scale_bytes(scale)
    factor = torch.exp2(exponents.float() - 127.0).masked_fill(exponents == 255, float("nan")).unsqueeze(-1)
    return (blocks * factor).view(rows, cols).to(dtype)


def _scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 scales as their raw exponent bytes, whether stored as uint8 or ``float8_e8m0fnu``."""
    return scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)


class Raon1MiniRoutedLinear(nn.Linear):
    """Dense projection or E4M3 + byte-exponent blocks, expanded only for this call."""

    def __init__(self, in_features, out_features, compressed=False):
        if compressed:
            nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
            self.register_parameter("bias", None)
            self.register_buffer("weight_scale_inv", torch.zeros(out_features, in_features // 32, dtype=torch.uint8))
        else:
            super().__init__(in_features, out_features, bias=False)

    def _apply(self, fn, recurse=True):
        if self.weight.dtype != torch.float8_e4m3fn:
            return super()._apply(fn, recurse)
        # Device moves are allowed, but dtype moves must not expand compressed storage.
        device = fn(torch.empty(0, device=self.weight.device)).device
        self.weight = self.weight.to(device=device)
        self.weight_scale_inv = self.weight_scale_inv.to(device=device)
        return self

    def forward(self, hidden):
        weight = self.weight
        if weight.dtype == torch.float8_e4m3fn:
            weight = dequantize_mxfp8(weight, self.weight_scale_inv, hidden.dtype)
        return F.linear(hidden, weight)


def _routed_projection(config, in_features, out_features):
    return Raon1MiniRoutedLinear(in_features, out_features, bool(config.raon_mxfp8_config))


class Raon1MiniExpert(nn.Module):
    """One routed SiTU-GLU expert: ``down_proj(situ_glu(gate_proj(x), up_proj(x)))``."""

    def __init__(self, config: Raon1MiniConfig) -> None:
        super().__init__()
        self.gate_proj = _routed_projection(config, config.hidden_size, config.moe_intermediate_size)
        self.up_proj = _routed_projection(config, config.hidden_size, config.moe_intermediate_size)
        self.down_proj = _routed_projection(config, config.moe_intermediate_size, config.hidden_size)
        self.gate_beta = config.activation_situ_beta
        self.value_beta = config.activation_situ_linear_beta

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        act = situ_glu(
            self.gate_proj(hidden),
            self.up_proj(hidden),
            self.gate_beta,
            self.value_beta,
        )
        return self.down_proj(act.to(hidden.dtype))


class Raon1MiniRouter(nn.Module):
    """Sigmoid router with the aux-loss-free expert bias (``e_score_correction_bias``) used for selection only."""

    def __init__(self, config: Raon1MiniConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.num_experts, dtype=torch.float32))
        self.top_k = config.num_experts_per_token
        self.renormalize = config.moe_renormalize
        self.scaling = config.routed_scaling_factor
        if config.moe_router_activation_func != "sigmoid":
            raise NotImplementedError("raon-1-mini uses the sigmoid router.")

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[N, C]`` -> (top-k expert indices ``[N, k]``, fp32 routing weights ``[N, k]``)."""
        logits = F.linear(hidden.float(), self.weight.float())
        scores = torch.sigmoid(logits)
        _, top = torch.topk(
            scores + self.e_score_correction_bias.float(),
            self.top_k,
            dim=-1,
            sorted=True,
        )
        chosen = torch.gather(scores, 1, top)
        if self.renormalize:
            chosen = chosen / (chosen.sum(-1, keepdim=True) + 1e-20)
        return top, chosen * self.scaling


class Raon1MiniMoE(nn.Module):
    """Routed experts plus the always-on shared expert.

    Each used MXFP8 expert is expanded to the activation dtype for its current
    projection only. No packed-stack rebinding or persistent dense cache is used.
    """

    def __init__(self, config: Raon1MiniConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.intermediate_size = config.moe_intermediate_size
        self.gate_beta = config.activation_situ_beta
        self.value_beta = config.activation_situ_linear_beta
        self.gate = Raon1MiniRouter(config)
        self.experts = nn.ModuleList([Raon1MiniExpert(config) for _ in range(config.num_experts)])
        shared_size = config.moe_intermediate_size * config.num_shared_experts
        self.shared_experts = Raon1MiniMLP(config, shared_size) if config.num_shared_experts else None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = hidden.shape
        flat = hidden.reshape(-1, dim)
        top, probs = self.gate(flat)
        out = torch.zeros(flat.shape[0], dim, dtype=torch.float32, device=flat.device)
        hit = torch.bincount(top.flatten(), minlength=self.num_experts)
        for expert_idx in hit.nonzero().flatten().tolist():
            rows, slot = torch.where(top == expert_idx)
            expert_out = self.experts[expert_idx](flat.index_select(0, rows))
            out.index_add_(0, rows, expert_out.float() * probs[rows, slot].unsqueeze(-1))
        out = out.to(hidden.dtype)
        if self.shared_experts is not None:
            out = out + self.shared_experts(flat)
        return out.reshape(batch, seq, dim)


# ---------------------------------------------------------------------------------------------
# decoder layer
# ---------------------------------------------------------------------------------------------
class Raon1MiniDecoderLayer(nn.Module):
    """One Raon_mHC-wired layer: attention site then MLP site, each with its own selector."""

    def __init__(self, config: Raon1MiniConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_rda = config.is_rda_layer(layer_idx)
        self.is_moe = config.is_moe_layer(layer_idx)
        self.layer_type = "linear_attention" if self.is_rda else "full_attention"
        self.input_layernorm = Raon1MiniRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Raon1MiniRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Raon1MiniRDA(config) if self.is_rda else Raon1MiniMLA(config)
        self.self_attn_raon_mhc = RaonMHC(config)
        self.mlp_raon_mhc = RaonMHC(config)
        if self.is_moe:
            self.moe = Raon1MiniMoE(config)
        else:
            self.mlp = Raon1MiniMLP(config, config.intermediate_size)

    def forward(
        self,
        stream: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        cache: Raon1MiniDynamicCache | None,
    ) -> torch.Tensor:
        batch, seq, n, dim = stream.shape
        dtype = stream.dtype
        rows = stream.reshape(batch * seq, n, dim)

        h_pre, h_post, h_res = self.self_attn_raon_mhc.route(stream)
        sf = rows.float()
        branch_in = self.input_layernorm(RaonMHC.aggregate(sf, h_pre, dtype)).reshape(batch, seq, dim)
        if self.is_rda:
            branch_out = self.self_attn(branch_in, cache, self.layer_idx, token_mask)
        else:
            branch_out = self.self_attn(branch_in, cache, self.layer_idx, attention_mask)
        rows = RaonMHC.mix(sf, branch_out.reshape(batch * seq, dim), h_post, h_res, dtype)

        h_pre, h_post, h_res = self.mlp_raon_mhc.route(rows.reshape(batch, seq, n, dim))
        sf = rows.float()
        branch_in = self.post_attention_layernorm(RaonMHC.aggregate(sf, h_pre, dtype)).reshape(batch, seq, dim)
        branch_out = self.moe(branch_in) if self.is_moe else self.mlp(branch_in)
        rows = RaonMHC.mix(sf, branch_out.reshape(batch * seq, dim), h_post, h_res, dtype)
        return rows.reshape(batch, seq, n, dim)


# ---------------------------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------------------------
class Raon1MiniDynamicCache:
    """Hybrid cache: RDA conv windows + fp32 recurrent states, MLA key/value tensors.

    Per layer: ``conv_states[i]`` is ``[B, 3*proj, K-1]`` of RAW (pre-conv) q|k|v rows,
    ``recurrent_states[i]`` is ``[B, H, V, K]`` fp32; ``key_cache[i]`` / ``value_cache[i]`` are
    ``[B, H, T, 192]`` / ``[B, H, T, 128]``. Implements the hybrid-cache interface ``generate`` uses.
    """

    is_compileable = False

    def __init__(self, config: Raon1MiniConfig) -> None:
        self.layer_types = config.layer_types
        self.transformer_layers = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        linear_layers = [i for i, kind in enumerate(self.layer_types) if kind == "linear_attention"]
        self.last_linear_layer = linear_layers[-1] if linear_layers else None
        self._seen_tokens = 0
        num = config.num_hidden_layers
        self.conv_states: list[torch.Tensor | None] = [None] * num
        self.recurrent_states: list[torch.Tensor | None] = [None] * num
        self.key_cache: list[torch.Tensor | None] = [None] * num
        self.value_cache: list[torch.Tensor | None] = [None] * num

    def __len__(self) -> int:
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ):
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for i in range(len(self.layer_types)):
            for store in (
                self.key_cache,
                self.value_cache,
                self.conv_states,
                self.recurrent_states,
            ):
                if store[i] is not None:
                    store[i] = store[i].index_select(0, beam_idx.to(store[i].device))

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return self._seen_tokens

    def reset(self) -> None:
        for store in (self.key_cache, self.value_cache, self.conv_states, self.recurrent_states):
            store[:] = [None] * len(self.layer_types)
        self._seen_tokens = 0

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return query_length + self.get_seq_length(layer_idx), 0

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    @property
    def has_previous_state(self) -> bool:
        return self._seen_tokens > 0


# ---------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------
class Raon1MiniPreTrainedModel(PreTrainedModel):
    config: Raon1MiniConfig
    config_class = Raon1MiniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules: ClassVar[list[str]] = ["Raon1MiniDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flash_attn = False
    _is_stateful = True
    # The checkpoint stores these in fp32 and the math reads them in fp32; never downcast them on load.
    _keep_in_fp32_modules: ClassVar[list[str]] = [
        "A_log",
        "dt_bias",
        "e_score_correction_bias",
        "eta",
    ]
    _keep_in_fp32_modules_strict: ClassVar[list[str]] = [
        "A_log",
        "dt_bias",
        "e_score_correction_bias",
        "eta",
    ]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        # Go through transformers.initialization: those initialisers are no-ops for parameters that
        # from_pretrained already loaded, so a plain in-place normal_() here would clobber checkpoints.
        if isinstance(module, Raon1MiniRoutedLinear) and module.weight.dtype == torch.float8_e4m3fn:
            init.zeros_(module.weight)
            init.constant_(module.weight_scale_inv, 127)
            return
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, Raon1MiniRMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, Raon1MiniShortConv):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, Raon1MiniRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Raon1MiniRDA):
            init.zeros_(module.A_log)
            init.zeros_(module.dt_bias)
        elif isinstance(module, RaonMHCStreamMix):
            init.normal_(module.phi_z, mean=0.0, std=std)
            init.ones_(module.alpha_z)
            init.zeros_(module.b_z)
            init.zeros_(module.eta)
        elif isinstance(module, RaonMHC):
            init.normal_(module.phi_pre, mean=0.0, std=std)
            init.normal_(module.phi_post, mean=0.0, std=std)
            init.ones_(module.alpha_pre)
            init.ones_(module.alpha_post)
            init.zeros_(module.b_pre)
            init.zeros_(module.b_post)


class Raon1MiniModel(Raon1MiniPreTrainedModel):
    """Embedding -> 4-stream expansion -> decoder layers -> mean reduction -> final norm."""

    def __init__(self, config: Raon1MiniConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Raon1MiniDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Raon1MiniRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Raon1MiniDynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast | tuple:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if past_key_values is not None and not isinstance(past_key_values, Raon1MiniDynamicCache):
            raise TypeError("Raon requires its own hybrid RDA/MLA cache.")
        length = inputs_embeds.shape[1]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if attention_mask is not None and (
            attention_mask.ndim != 2 or attention_mask.shape != (inputs_embeds.shape[0], past_length + length)
        ):
            raise ValueError("Raon attention_mask must cover the complete cached key sequence.")
        if past_key_values is not None and not use_cache:
            past_key_values = copy.deepcopy(past_key_values)
        if use_cache and past_key_values is None:
            past_key_values = Raon1MiniDynamicCache(self.config)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        token_mask = None
        if attention_mask is not None and attention_mask.dim() == 2 and not bool(attention_mask.all()):
            token_mask = attention_mask[:, -inputs_embeds.shape[1] :]

        n = self.config.raon_mhc_streams
        stream = inputs_embeds.unsqueeze(2).expand(-1, -1, n, -1).contiguous()
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                # Expose [batch, tokens, hidden] representations by reducing the Raon_mHC streams,
                # as at the model output. Intermediate states are not final-normalized.
                all_hidden_states += (stream.float().mean(2).to(inputs_embeds.dtype),)
            stream = layer(
                stream,
                attention_mask=causal_mask,
                token_mask=token_mask,
                cache=past_key_values,
            )
        if past_key_values is not None:
            past_key_values._seen_tokens += length
        hidden = self.norm(stream.float().mean(2).to(inputs_embeds.dtype))
        if output_hidden_states:
            all_hidden_states += (hidden,)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )
        return output if return_dict else output.to_tuple()


class Raon1MiniPresencePenaltyLogitsProcessor:
    """Subtract ``penalty`` from the logits of every token generated so far.

    Matches vLLM's presence penalty: prompt tokens are not penalized and a token
    is penalized once however often it appeared. ``prompt_length`` is the padded
    prompt width, the same for every row of a batch.
    """

    def __init__(self, penalty: float, prompt_length: int) -> None:
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[:, self.prompt_length :]
        if generated.shape[1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, generated, True)
        return scores - self.penalty * seen.to(scores.dtype)


class Raon1MiniForCausalLM(Raon1MiniPreTrainedModel, GenerationMixin):
    """raon-1-mini causal language model with the untied ``lm_head``."""

    _tied_weights_keys = None

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # RDA conv windows and recurrent state require the Raon hybrid cache.
        return False

    def generate(self, *args, presence_penalty: float | None = None, **kwargs):
        # GenerationConfig rejects keys it does not define, so carry this one in ourselves.
        if presence_penalty is not None:
            config = copy.deepcopy(kwargs.get("generation_config") or self.generation_config)
            config.presence_penalty = presence_penalty
            kwargs["generation_config"] = config
        return super().generate(*args, **kwargs)

    def _get_logits_processor(self, generation_config, input_ids_seq_length=None, *args, **kwargs):
        # Transformers has no presence penalty; generation_config.json sets one (1.5)
        # as the serving default, so apply it ahead of temperature/top-k/top-p as vLLM does.
        processors = super()._get_logits_processor(generation_config, input_ids_seq_length, *args, **kwargs)
        penalty = getattr(generation_config, "presence_penalty", None)
        if penalty:
            processors.insert(0, Raon1MiniPresencePenaltyLogitsProcessor(float(penalty), input_ids_seq_length))
        return processors

    def __init__(self, config: Raon1MiniConfig) -> None:
        super().__init__(config)
        self.model = Raon1MiniModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Raon1MiniDynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple:
        return_dict = self.config.return_dict if return_dict is None else return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
        return output if return_dict else output.to_tuple()


__all__ = [
    "Raon1MiniDecoderLayer",
    "Raon1MiniDynamicCache",
    "Raon1MiniForCausalLM",
    "Raon1MiniRDA",
    "RaonMHC",
    "Raon1MiniMLA",
    "Raon1MiniMoE",
    "Raon1MiniModel",
    "Raon1MiniPreTrainedModel",
    "torch_recurrent_rda",
]
