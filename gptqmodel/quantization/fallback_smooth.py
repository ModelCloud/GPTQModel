# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .config import (
    Fallback,
    QuantizeConfig,
    SmoothLog,
    SmoothMAD,
    SmoothMSE,
    SmoothOutlier,
    SmoothPercentile,
    SmoothPercentileAsymmetric,
    SmoothRowCol,
    SmoothSoftNorm,
)


# For Gaussian-like rows, raw MAD ~= 0.67449 * sigma. Normalize MAD so the
# configured `k` behaves like a sigma-width window instead of clipping far more
# aggressively than intended.
MAD_TO_STD_SCALE = 1.4826


def affine_range_with_zero(
    lo: torch.Tensor, hi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extend an affine code range so that it includes zero.

    An affine code range only represents zero points in ``[0, maxq]``.
    Fitting the raw group minimum and maximum makes the derived zero point
    fall outside that range for a group whose values do not cross zero
    (positive-only gives a negative zero point, negative-only one above
    ``maxq``), and the packer then stores a different grid than the one the
    quantizer scored. Including zero in the fitted range keeps every zero
    point representable and preserves full group coverage.
    """

    zero = torch.zeros_like(lo)
    return torch.minimum(lo, zero), torch.maximum(hi, zero)


def symmetric_range(block: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the zero-centered range every symmetric kernel expects.

    Symmetric GPTQ stores a fixed zero point of ``2 ** (bits - 1)`` and the
    kernels that ignore ``qzeros`` (Marlin) assume it. The fallback strategies
    therefore cannot fit their own midpoint/mean/median/STDCLIP endpoints under
    ``sym=True``: they must all use ``[-max|x|, max|x|]`` so the derived zero
    point is that fixed value. An all-zero group gets the same ``+-1`` sentinel
    as ``Quantizer.find_params`` so the scale stays finite.
    """

    max_abs = block.abs().amax(dim=1, keepdim=True)
    max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
    return -max_abs, max_abs


def _quantile(block: torch.Tensor, percentile: float) -> torch.Tensor:
    if percentile <= 0.0:
        return block.min(dim=1, keepdim=True).values
    if percentile >= 100.0:
        return block.max(dim=1, keepdim=True).values

    q = max(0.0, min(percentile / 100.0, 1.0))
    position = q * (block.shape[1] - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower = block.kthvalue(lower_index + 1, dim=1, keepdim=True).values

    if upper_index == lower_index:
        return lower

    upper = block.kthvalue(upper_index + 1, dim=1, keepdim=True).values
    return lower + (upper - lower) * (position - lower_index)


def _clamp_block(block: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(block, hi), lo)


def smooth_block(
    block: torch.Tensor,
    fallback: Fallback,
    *,
    eps: float = 1e-8,
    group_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    method = getattr(fallback, "smooth", None)
    if method is None:
        return block, None
    if group_size is not None and group_size < 0:
        group_size = block.shape[1]
    if group_size is not None and group_size < getattr(method, "group_size_threshold", 0):
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
        lo = _quantile(block_f, low)
        hi = _quantile(block_f, high)
        return _clamp_block(block_f, lo, hi).to(block.dtype), None

    if isinstance(method, SmoothMAD):
        median = block_f.median(dim=1, keepdim=True).values
        mad = (block_f - median).abs().median(dim=1, keepdim=True).values * MAD_TO_STD_SCALE
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

    if isinstance(method, SmoothMSE):
        # The MSE search needs the quantizer's code grid and saliency, so only
        # the RTN/GPTQ/QQQ loops can apply it; they branch before calling this
        # helper. The direct GGUF/BitsAndBytes packers call it too, and a silent
        # no-op saved un-smoothed weights under MSE metadata (#5383).
        raise ValueError(
            "SmoothMSE requires the quantizer's MSE search and is not available "
            "to the direct packers; use a clipping smoother such as SmoothMAD, "
            "or quantize through RTN/GPTQ/QQQ."
        )

    # Fail closed: silently returning the block unchanged would advertise a
    # preprocessor the packed weights never received.
    raise ValueError(f"smooth_block does not implement {type(method).__name__}")


def _eval_mse(block_f, min_val, max_val, base_zero, qcfg, maxq, shrink, eps, weights=None):
    """Compute MSE for shrinkage factors [rows, n, 1].

    ``weights`` has shape ``[rows, 1, columns]`` and carries the per-element
    saliency used to reduce over the group. It represents how much a weight's
    quantization error is expected to reach the layer output.
    """
    scale = torch.clamp((max_val.unsqueeze(1) * shrink - min_val.unsqueeze(1) * shrink) / maxq, min=eps)
    zero = base_zero.unsqueeze(1).expand_as(scale) if qcfg.sym else torch.round(-min_val.unsqueeze(1) * shrink / scale)
    q = torch.clamp(torch.round(block_f.unsqueeze(1) / scale + zero), 0, maxq)

    err = ((q - zero) * scale - block_f.unsqueeze(1)).pow(2)
    if weights is None:
        return err.mean(dim=2)
    return (err * weights).sum(dim=2)


def _normalize_weights(weights: Optional[torch.Tensor], rows: int, columns: int, *, device, dtype, eps: float = 1e-12):
    """Broadcast, sanitize and per-row normalize a saliency tensor.

    A one-dimensional input is a per-input-channel weight (the Hessian
    diagonal). A two-dimensional input is a per-element weight, which lets the
    calibration-free path penalize relative errors on large weights. Each row
    is normalized to sum to one so rows with different scales contribute
    comparably; an all-zero row falls back to a uniform weight.
    """
    if weights is None:
        return None
    weights = torch.as_tensor(weights, device=device, dtype=dtype)
    if weights.dim() == 1:
        if weights.numel() != columns:
            raise ValueError(
                f"weights must have one entry per input column ({columns}), got {weights.numel()}"
            )
        weights = weights.reshape(1, 1, columns).expand(rows, 1, columns)
    elif weights.dim() == 2:
        if tuple(weights.shape) != (rows, columns):
            raise ValueError(
                f"weights must have shape ({rows}, {columns}), got {tuple(weights.shape)}"
            )
        weights = weights.reshape(rows, 1, columns)
    else:
        raise ValueError(
            f"weights must be 1-D or 2-D, got {weights.dim()}-D"
        )
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = weights.sum(dim=2, keepdim=True)
    uniform = torch.full_like(weights, 1.0 / columns)
    return torch.where(total > 0, weights / total.clamp_min(eps), uniform)


def element_saliency_weights(block: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Calibration-free per-element saliency for the SmoothMSE objective.

    Without captured activations the best available proxy for "this weight
    matters" is its own magnitude: a large weight dominates its row's dot
    product, so a relative error on it is more visible at the layer output
    than the same relative error on a small weight. Weighting the squared
    error by ``|w| + mean(|w|)`` blends that relative view with a uniform
    floor. A pure unweighted mean instead rewards clipping a row's largest
    weight, because one huge element's error is cheap compared with the
    smaller step it buys for the other 127 elements.
    """
    magnitude = block.float().abs()
    return magnitude + magnitude.mean(dim=1, keepdim=True).clamp_min(eps)


def hessian_column_weights(hessian: Optional[torch.Tensor], columns: int, *, device=None) -> Optional[torch.Tensor]:
    """Return the input-channel saliency stored on a GPTQ/QQQ Hessian diagonal.

    ``H[i, i]`` accumulates the squared input activation of channel ``i`` over
    the calibration batch, so it is the exact per-channel weight for the
    diagonal approximation of the layer's output error. A caller that already
    reduced the Hessian to its diagonal may pass that 1-D vector directly;
    only the diagonal is needed, so a fallback can avoid materializing the
    ``columns ** 2`` matrix (#5300). Return ``None`` when no Hessian was
    captured so the caller keeps the calibration-free path.
    """
    if hessian is None:
        return None
    if hessian.dim() == 2:
        if hessian.shape[0] != columns or hessian.shape[1] != columns:
            return None
        diag = hessian.diagonal()
    elif hessian.dim() == 1 and hessian.numel() == columns:
        diag = hessian
    else:
        return None
    if device is not None:
        diag = diag.to(device=device)
    return torch.nan_to_num(diag.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def mse_optimal_quant(
    block: torch.Tensor,
    qcfg: QuantizeConfig,
    maxq: int,
    *,
    steps: int,
    maxshrink: float,
    eps: float = 1e-8,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ternary search over shrink factors: O(log steps).

    The unshrunk range (``p = 1``) is the plain quantization baseline, so it is
    scored explicitly and starts as the incumbent. Only a candidate whose
    measured error is strictly lower replaces it, which keeps the returned grid
    no worse than that baseline under the same weights even when the ternary
    probes and their refinement never reach the right endpoint.

    ``weights`` supplies saliency instead of the uniform mean: either a
    per-input-channel vector (the GPTQ/QQQ Hessian diagonal) or a per-element
    matrix (the calibration-free ``element_saliency_weights`` proxy). The
    unweighted mean is the right objective only when every weight is equally
    visible at the layer output; in a trained transformer it is not, and
    clipping a row's largest weights can lower the unweighted error while
    raising the layer's output error.
    """
    block_f = block.float()
    rows = block_f.shape[0]
    weights = _normalize_weights(
        weights, rows, block_f.shape[1], device=block_f.device, dtype=block_f.dtype
    )

    if qcfg.sym:
        max_abs = block_f.abs().max(dim=1, keepdim=True).values
        min_val = -max_abs
        max_val = max_abs
        base_zero = torch.full_like(max_abs, (maxq + 1) / 2.0)
    else:
        min_val = block_f.min(dim=1, keepdim=True).values
        max_val = block_f.max(dim=1, keepdim=True).values
        min_val, max_val = affine_range_with_zero(min_val, max_val)
        base_zero = None

    steps = max(int(math.log(max(steps, 2)) / math.log(1.5)) + 1, 3)
    shrink = max(min(maxshrink, 1.0), 1e-3)

    # Left, Right pointer
    l, r = torch.full((rows, 1), shrink, device=block_f.device, dtype=block_f.dtype), torch.ones((rows, 1), device=block_f.device, dtype=block_f.dtype)
    # Score both endpoints before searching. The interior probes are computed
    # from `l`/`r`, so they never equal `p = 1`; seeding an unscored incumbent
    # let the first interior candidate hide the unshrunk grid entirely.
    endpoint_err = _eval_mse(
        block_f, min_val, max_val, base_zero, qcfg, maxq,
        torch.stack([l, r], dim=1).view(rows, 2, 1), eps,
        weights=weights,
    )
    shrunk_better = endpoint_err[:, 0] < endpoint_err[:, 1]
    best_err = torch.where(shrunk_better, endpoint_err[:, 0], endpoint_err[:, 1])
    best_p = torch.where(shrunk_better.unsqueeze(1), l, r)

    for _ in range(steps):
        mid1, mid2 = l + (r - l) / 3.0, r - (r - l) / 3.0
        err = _eval_mse(
            block_f, min_val, max_val, base_zero, qcfg, maxq,
            torch.stack([mid1, mid2], dim=1).view(rows, 2, 1), eps,
            weights=weights,
        )

        for i, p in enumerate([mid1, mid2]):
            better = err[:, i] < best_err
            best_err, best_p = torch.where(better, err[:, i], best_err), torch.where(better.unsqueeze(1), p, best_p)

        move_r = err[:, 0] < err[:, 1]
        r, l = torch.where(move_r.unsqueeze(1), mid2, r), torch.where(move_r.unsqueeze(1), l, mid1)

    # Refine
    delta = (r - l) * 0.1
    refinement = torch.stack([torch.clamp(best_p - delta, shrink, 1.0), best_p, torch.clamp(best_p + delta, shrink, 1.0)], dim=1).view(rows, 3, 1)
    refine_err = _eval_mse(
        block_f, min_val, max_val, base_zero, qcfg, maxq, refinement, eps,
        weights=weights,
    )
    refine_best_err, refine_index = refine_err.min(dim=1)
    # Refinement always contains the incumbent, but keep the comparison
    # explicit so a re-evaluated candidate cannot displace a better endpoint.
    improved = refine_best_err < best_err
    best_p = torch.where(
        improved.unsqueeze(1),
        torch.gather(refinement.squeeze(2), 1, refine_index.unsqueeze(1)),
        best_p,
    )

    # Final quantization
    scale_best = torch.clamp((max_val - min_val) * best_p / maxq, min=eps)
    zero_best = base_zero if qcfg.sym else torch.round(-min_val * best_p / scale_best)
    q = torch.clamp(torch.round(block_f / scale_best + zero_best), 0, maxq)
    dequant_best = (q - zero_best) * scale_best

    return dequant_best, scale_best, zero_best
