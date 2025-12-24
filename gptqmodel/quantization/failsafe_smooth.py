# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .config import (
    FailSafe,
    QuantizeConfig,
    SmoothLog,
    SmoothMAD,
    SmoothOutlier,
    SmoothPercentile,
    SmoothPercentileAsymmetric,
    SmoothRowCol,
    SmoothSoftNorm,
)


def _quantile(abs_block: torch.Tensor, percentile: float) -> torch.Tensor:
    if percentile <= 0.0:
        return abs_block.min(dim=1, keepdim=True).values
    if percentile >= 100.0:
        return abs_block.max(dim=1, keepdim=True).values
    q = max(0.0, min(percentile / 100.0, 1.0))
    return torch.quantile(abs_block, q, dim=1, keepdim=True)


def _clamp_block(block: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(block, hi), lo)


def smooth_block(
    block: torch.Tensor,
    failsafe: FailSafe,
    *,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    method = getattr(failsafe, "smooth", None)
    if method is None:
        return block, None

    if isinstance(method, SmoothRowCol):
        axis = (method.axis or "row").lower()
        if axis == "col":
            col_rms = block.pow(2).mean(dim=0, keepdim=True).sqrt().clamp(min=eps)
            scale_factor = col_rms.mean().view(1, 1)
        else:
            scale_factor = block.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=eps)
        return block / scale_factor, scale_factor

    block_f = block.float()

    if isinstance(method, SmoothSoftNorm):
        mean = block_f.mean(dim=1, keepdim=True)
        rms = (block_f - mean).pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=eps)
        k = float(method.k)
        block_norm = (block_f - mean) / rms
        block_norm = torch.clamp(block_norm, -k, k)
        return (block_norm * rms + mean).to(block.dtype), None

    if isinstance(method, SmoothPercentile):
        abs_block = block_f.abs()
        threshold = _quantile(abs_block, float(method.percentile))
        return torch.clamp(block_f, -threshold, threshold).to(block.dtype), None

    if isinstance(method, SmoothPercentileAsymmetric):
        low = float(method.low)
        high = float(method.high)
        lo = torch.quantile(block_f, max(0.0, min(low / 100.0, 1.0)), dim=1, keepdim=True)
        hi = torch.quantile(block_f, max(0.0, min(high / 100.0, 1.0)), dim=1, keepdim=True)
        return _clamp_block(block_f, lo, hi).to(block.dtype), None

    if isinstance(method, SmoothMAD):
        median = block_f.median(dim=1, keepdim=True).values
        mad = (block_f - median).abs().median(dim=1, keepdim=True).values
        k = float(method.k)
        lo = median - k * mad
        hi = median + k * mad
        return _clamp_block(block_f, lo, hi).to(block.dtype), None

    if isinstance(method, SmoothOutlier):
        pct = float(method.pct)
        if pct <= 0.0:
            return block, None
        abs_block = block_f.abs()
        k = max(1, int(round(abs_block.shape[1] * (1.0 - pct / 100.0))))
        k = min(k, abs_block.shape[1])
        threshold = torch.kthvalue(abs_block, k, dim=1, keepdim=True).values
        return torch.clamp(block_f, -threshold, threshold).to(block.dtype), None

    if isinstance(method, SmoothLog):
        mu = max(float(method.mu), eps)
        abs_block = block_f.abs()
        log_mu = math.log1p(mu)
        log_vals = torch.log1p(abs_block * mu) / log_mu
        threshold = _quantile(log_vals, float(method.percentile))
        lin_threshold = (torch.exp(threshold * log_mu) - 1.0) / mu
        return torch.clamp(block_f, -lin_threshold, lin_threshold).to(block.dtype), None

    return block, None


def mse_optimal_quant(
    block: torch.Tensor,
    qcfg: QuantizeConfig,
    maxq: int,
    *,
    steps: int,
    maxshrink: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_f = block.float()
    rows = block_f.shape[0]
    cols = block_f.shape[1]

    if qcfg.sym:
        max_abs = block_f.abs().max(dim=1, keepdim=True).values
        min_val = -max_abs
        max_val = max_abs
        base_zero = torch.full_like(max_abs, (maxq + 1) / 2.0)
    else:
        min_val = block_f.min(dim=1, keepdim=True).values
        max_val = block_f.max(dim=1, keepdim=True).values
        base_zero = None

    steps = max(int(steps), 1)
    shrink = max(min(maxshrink, 1.0), 1e-3)
    p = torch.linspace(1.0, shrink, steps, device=block_f.device, dtype=block_f.dtype).view(1, steps, 1)

    min_p = min_val.unsqueeze(1) * p
    max_p = max_val.unsqueeze(1) * p
    scale = (max_p - min_p) / maxq
    scale = torch.clamp(scale, min=eps)

    if qcfg.sym:
        zero = base_zero.unsqueeze(1).expand(rows, steps, 1)
    else:
        zero = torch.round(-min_p / scale)

    block_expanded = block_f.unsqueeze(1)
    q = torch.round(block_expanded / scale + zero)
    q = torch.clamp(q, 0, maxq)
    dequant = (q - zero) * scale
    err = (dequant - block_expanded).pow(2).mean(dim=2)

    best_idx = err.argmin(dim=1)
    best_idx_exp = best_idx.view(rows, 1, 1).expand(rows, 1, cols)
    dequant_best = torch.gather(dequant, 1, best_idx_exp).squeeze(1)

    best_idx_scale = best_idx.view(rows, 1, 1)
    scale_best = torch.gather(scale, 1, best_idx_scale).squeeze(1)
    zero_best = torch.gather(zero, 1, best_idx_scale).squeeze(1)

    return dequant_best, scale_best, zero_best
