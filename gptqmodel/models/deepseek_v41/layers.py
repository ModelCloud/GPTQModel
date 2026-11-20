# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Operators for the published DeepSeek-V4.1-Flash weight format."""

import torch
from torch import nn
from torch.nn import functional as F

from ...quantization.activation_floatx import _e2m1_nearest_values
from ...quantization.dtype import dequantize_fp8
from ...utils.macaq_triton import mxfp8_scaled_linear
from ...utils.marlin import (
    gptq_marlin_gemm,
    marlin_is_tile_aligned,
    marlin_make_workspace_new,
    mxfp4_marlin_prepare,
    mxfp8_marlin_prepare,
)
from ...utils.marlin_scalar_type import scalar_types
from ...utils.model_dequant import dequantize_deepseek_v4_fp4_expert


def _quantize_fp8_blocks(x):
    blocks = x.float().unflatten(-1, (-1, 32))
    scale = (blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448).log2().ceil().exp2()
    codes = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return codes, scale


def fp8_roundtrip(x):
    codes, scale = _quantize_fp8_blocks(x)
    return (codes.float() * scale).flatten(-2).to(x.dtype)


def fp4_roundtrip(x, *, compressed_kv=False):
    # CSA2 uses 16-value E4M3-scaled KV blocks and 32-value E8M0-scaled index blocks.
    block_size = 16 if compressed_kv else 32
    blocks = x.float().unflatten(-1, (x.shape[-1] // block_size, block_size))
    scale = blocks.abs().amax(-1, keepdim=True) / 6
    if compressed_kv:
        scale = scale.clamp(2**-9, 448).to(torch.float8_e4m3fn).float()
    else:
        scale = scale.clamp_min(2**-126).log2().ceil().exp2()
    return (_e2m1_nearest_values(blocks / scale) * scale).flatten(-2).to(x.dtype)


class V41PackedLinear(nn.Module):
    """Store the published weights for the Torch reference path."""

    def __init__(self, in_features, out_features, *, fp4=False, groups=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.fp4 = fp4
        if in_features % 32 or out_features % 32 or out_features % groups:
            raise ValueError("V4.1 packed linear dimensions must align with its blocks and groups")
        rows = out_features if fp4 else out_features // 32
        self.register_buffer("scale", torch.empty(rows, in_features // 32, dtype=torch.float8_e8m0fnu))
        columns = in_features // 2 if fp4 else in_features
        dtype = torch.int8 if fp4 else torch.float8_e4m3fn
        self.register_buffer("weight", torch.empty(out_features, columns, dtype=dtype))

    @property
    def marlin_aligned(self):
        return marlin_is_tile_aligned(self.out_features // self.groups, self.in_features)

    def forward(self, x):
        x = fp8_roundtrip(x)
        if self.fp4:
            weight = dequantize_deepseek_v4_fp4_expert(self.weight, self.scale, target_dtype=x.dtype)
        else:
            weight = dequantize_fp8(self.weight, scale=self.scale, target_dtype=x.dtype)
        if self.groups == 1 and x.ndim != 4:
            return F.linear(x, weight)
        weight = weight.unflatten(0, (self.groups, -1))
        return torch.einsum("bsgd,grd->bsgr", x, weight)


class V41NativeMXLinear(nn.Module):
    """Run block-scaled MMA directly from the checkpoint's packed bytes."""

    def __init__(self, source):
        super().__init__()
        self.groups = source.groups
        self.register_buffer("weight", source.weight if source.fp4 else source.weight.view(torch.uint8))
        self.register_buffer("scale", source.scale.view(torch.uint8))

    def forward(self, x):
        codes, scale = _quantize_fp8_blocks(x)
        # These scales are powers of two. The FP32 exponent is the E8M0 byte.
        exponents = (scale.view(torch.int32) >> 23).to(torch.uint8).squeeze(-1)
        return mxfp8_scaled_linear(
            codes.flatten(-2), exponents, self.weight, self.scale, groups=self.groups, dtype=x.dtype,
        )


class V41MarlinLinear(nn.Module):
    """Execute the released MX weight codes in Marlin's packed layout."""

    def __init__(self, source):
        super().__init__()
        self.groups = source.groups
        self.in_features = source.in_features
        self.group_width = source.out_features // source.groups
        self.weight_type = scalar_types.float4_e2m1f if source.fp4 else scalar_types.float8_e4m3fn
        if source.fp4:
            prepare = mxfp4_marlin_prepare
            scales = source.scale
        else:
            prepare = mxfp8_marlin_prepare
            # DeepSeek shares each 32x32 exponent across 32 output rows.
            # Marlin reads one exponent byte per output row and K group.
            scales = source.scale.view(torch.uint8).repeat_interleave(32, dim=0).view(source.scale.dtype)
        if self.groups == 1:
            qweight, scales = prepare(source.weight, scales)
        else:
            weights = source.weight.split(self.group_width)
            packed = [prepare(weight, scale) for weight, scale in zip(weights, scales.split(self.group_width), strict=True)]
            qweights, group_scales = zip(*packed, strict=True)
            qweight, scales = torch.stack(qweights), torch.stack(group_scales)
        self.register_buffer("qweight", qweight)
        # Keep the exponent bytes opaque to dtype-changing module moves.
        self.register_buffer("scales", scales.view(torch.uint8))

    def _project(self, x, qweight, scales, workspace):
        flat = x.reshape(-1, self.in_features).contiguous()
        output = gptq_marlin_gemm(
            a=flat, c=None, b_q_weight=qweight, b_bias=None, b_scales=scales,
            global_scale=None, b_zeros=None, g_idx=None, perm=None, workspace=workspace,
            b_q_type=self.weight_type, size_m=flat.size(0), size_n=self.group_width,
            size_k=self.in_features, use_fp32_reduce=True,
        )
        return output.reshape(*x.shape[:-1], self.group_width)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            raise ValueError("DeepSeek V4.1 Marlin projections require CUDA bfloat16 activations")
        x = fp8_roundtrip(x)
        # Each forward owns its locks so concurrent streams do not share them.
        workspace = marlin_make_workspace_new(x.device)
        scales = self.scales.view(torch.float8_e8m0fnu)
        if self.groups == 1:
            return self._project(x, self.qweight, scales, workspace)
        return torch.stack([
            self._project(group, qweight, group_scales, workspace)
            for group, qweight, group_scales in zip(x.unbind(-2), self.qweight, scales, strict=True)
        ], dim=-2)


class V41RMSNorm(nn.Module):
    def __init__(self, width, eps, dtype):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(width, dtype=dtype))

    def forward(self, x):
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(x.dtype)


class V41HyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.streams = config.hc_mult
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.steps = config.hc_sinkhorn_iters
        mixes = self.streams * (self.streams + 2)
        self.fn = nn.Parameter(torch.empty(mixes, self.streams * config.hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(mixes, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def forward(self, residual):
        flat = residual.flatten(0, 1).flatten(-2)
        # Stabilize coefficient rounding across prefill and decode. Keep the
        # FP64 activation scratch bounded when a prompt has many tokens.
        weights = self.fn.double()
        mix_chunks = []
        for chunk in flat.split(max(1, 2**20 // flat.size(-1))):
            values = chunk.double()
            inv_rms = torch.rsqrt(values.square().mean(-1, keepdim=True) + self.norm_eps)
            mix_chunks.append((F.linear(values, weights) * inv_rms).float())
        mixes = torch.cat(mix_chunks).unflatten(0, residual.shape[:2])
        pre, post, residual_mix = mixes.split((self.streams, self.streams, self.streams**2), dim=-1)
        pre_base, post_base, mix_base = self.base.split((self.streams, self.streams, self.streams**2))
        pre = (pre * self.scale[0] + pre_base).sigmoid() + self.eps
        post = (post * self.scale[1] + post_base).sigmoid() * 2
        mix = residual_mix * self.scale[2] + mix_base
        mix = mix.unflatten(-1, (self.streams, self.streams)).softmax(dim=-1) + self.eps
        mix = mix / (mix.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.steps - 1):
            mix = mix / (mix.sum(dim=-1, keepdim=True) + self.eps)
            mix = mix / (mix.sum(dim=-2, keepdim=True) + self.eps)
        return pre, post, mix


def collapse_streams(hidden, pre_mix):
    return (hidden.float() * pre_mix.unsqueeze(-1)).sum(dim=-2).to(hidden.dtype)


def expand_streams(hidden, residual, post_mix, residual_mix):
    mixed = (residual.unsqueeze(-2) * residual_mix.unsqueeze(-1)).sum(dim=-3)
    return (post_mix.unsqueeze(-1) * hidden.unsqueeze(-2) + mixed).to(hidden.dtype)


class V41Expert(nn.Module):
    _route_owner = None

    def __init__(self, config, *, routed):
        super().__init__()
        width, intermediate = config.hidden_size, config.moe_intermediate_size
        self.w1 = V41PackedLinear(width, intermediate, fp4=routed)
        self.w2 = V41PackedLinear(intermediate, width, fp4=routed)
        self.w3 = V41PackedLinear(width, intermediate, fp4=routed)
        self.limit = config.swiglu_limit

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ("w1", "w2", "w3", "limit") and self._route_owner is not None:
            owner = self._route_owner()
            if owner is not None:
                owner._q4nx_tables = None

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_route_owner", None)
        return state

    def forward(self, x, weights=None):
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.limit > 0:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(-self.limit, self.limit)
        hidden = F.silu(gate) * up
        if weights is not None:
            hidden = hidden * weights.unsqueeze(-1)
        return self.w2(hidden.to(x.dtype))


class V41Router(nn.Module):
    def __init__(self, config, dtype, *, vision):
        super().__init__()
        count = config.n_routed_experts
        self.weight = nn.Parameter(torch.empty(count, config.hidden_size, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(count, dtype=torch.float32))
        self.bias_vl = nn.Parameter(torch.empty(count, dtype=torch.float32)) if vision else None
        self.top_k = config.num_experts_per_tok
        self.normalize = config.norm_topk_prob
        self.scale = config.routed_scaling_factor

    def forward(self, x, image_mask):
        scores = F.softplus(F.linear(x.float(), self.weight.float())).sqrt()
        bias = self.bias
        if image_mask is not None:
            bias = torch.where(image_mask.reshape(-1, 1), self.bias_vl, bias)
        indices = (scores + bias).topk(self.top_k, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.normalize and self.top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return indices, weights * self.scale


class V41MoE(nn.Module):
    def __init__(self, config, dtype, *, vision):
        super().__init__()
        if config.n_shared_experts != 1 or config.scoring_func != "sqrtsoftplus":
            raise ValueError("DeepSeek V4.1 Flash requires one shared expert and sqrtsoftplus routing")
        self.gate = V41Router(config, dtype, vision=vision)
        self.experts = nn.ModuleList(V41Expert(config, routed=True) for _ in range(config.n_routed_experts))
        self.shared_experts = V41Expert(config, routed=False)
        self._q4nx_routing = False
        self._q4nx_tables = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_q4nx_tables"] = None
        return state

    def _apply(self, fn, recurse=True):
        self._q4nx_tables = None
        return super()._apply(fn, recurse=recurse)

    def forward(self, hidden, image_mask):
        x = hidden.flatten(0, 1)
        indices, weights = self.gate(x, image_mask)
        if self._q4nx_routing:
            from .exl3 import routed_q4nx

            result = routed_q4nx(self, x, indices, weights)
            if result is not None:
                return (result + self.shared_experts(x)).to(x.dtype).reshape_as(hidden)
        result = torch.zeros_like(x, dtype=torch.float32)
        for index in indices.unique(sorted=True).tolist():
            rows, slots = torch.where(indices == index)
            values = self.experts[index](x[rows], weights[rows, slots])
            result.index_add_(0, rows, values.float())
        return (result + self.shared_experts(x)).to(x.dtype).reshape_as(hidden)
