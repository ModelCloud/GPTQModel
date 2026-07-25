# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Asymmetric vs. symmetric quantization analysis from Section 3.1 of arXiv:2605.02404.

Provides the centering inefficiency ``gamma``, the gamma-squared variance law,
and helper quantizer step/noise calculations.
"""

import torch


def centering_inefficiency(tensor: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Compute the centering inefficiency (skewness) ``gamma = 2M / R``.

    ``R = U - L`` is the dynamic range and ``M = max(|L|, |U|)`` is the largest
    absolute magnitude. When weights are symmetric about zero ``gamma = 1``;
    skewed intervals yield ``gamma > 1``.

    Args:
        tensor: Weight tensor (any shape).
        eps: Small constant for numerical stability.

    Returns:
        Scalar tensor containing ``gamma``.
    """

    min_v = tensor.min()
    max_v = tensor.max()
    dynamic_range = (max_v - min_v).clamp(min=eps)
    max_abs = torch.maximum(min_v.abs(), max_v.abs()).clamp(min=eps)
    gamma = 2.0 * max_abs / dynamic_range
    return gamma


def step_size_sym(bits: int, max_abs: float | torch.Tensor) -> torch.Tensor:
    """Symmetric quantizer step size ``Δ_sym = 2M / (2^bits - 1)``.

    Args:
        bits: Quantization bitwidth.
        max_abs: Largest absolute magnitude ``M``.

    Returns:
        The symmetric step size.
    """

    levels = (2 ** bits) - 1
    return 2.0 * max_abs / levels


def step_size_asym(bits: int, min_v: float | torch.Tensor, max_v: float | torch.Tensor) -> torch.Tensor:
    """Asymmetric quantizer step size ``Δ_asym = (U - L) / (2^bits - 1)``.

    Args:
        bits: Quantization bitwidth.
        min_v: Minimum value ``L``.
        max_v: Maximum value ``U``.

    Returns:
        The asymmetric step size.
    """

    levels = (2 ** bits) - 1
    return (max_v - min_v) / levels


def quantization_noise_variance_sym(bits: int, max_abs: float | torch.Tensor) -> torch.Tensor:
    """Quantization noise variance for a symmetric ``bits``-wide grid.

    Uses Bennett's high-rate approximation ``σ² = Δ² / 12``.
    """

    delta = step_size_sym(bits, max_abs)
    return (delta ** 2) / 12.0


def quantization_noise_variance_asym(bits: int, min_v: float | torch.Tensor, max_v: float | torch.Tensor) -> torch.Tensor:
    """Quantization noise variance for an asymmetric ``bits``-wide grid.

    Uses Bennett's high-rate approximation ``σ² = Δ² / 12``.
    """

    delta = step_size_asym(bits, min_v, max_v)
    return (delta ** 2) / 12.0


def gamma_squared_variance_law(
    tensor: torch.Tensor,
    bits: int,
    *,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Validate the gamma-squared variance law for a weight tensor.

    Computes the ratio ``σ_sym² / σ_asym²`` and compares it to ``gamma²``.
    Per Lemma 3.2, the ratio should equal ``gamma²`` (up to the high-rate
    approximation.

    Args:
        tensor: Weight tensor.
        bits: Quantization bitwidth.
        eps: Small constant.

    Returns:
        Dictionary with ``gamma``, ``gamma_sq``, ``var_sym``, ``var_asym``,
        ``var_ratio``, and ``law_error`` (``var_ratio / gamma_sq - 1``).
    """

    min_v = tensor.min()
    max_v = tensor.max()
    gamma = centering_inefficiency(tensor, eps=eps)
    gamma_sq = gamma ** 2

    max_abs = torch.maximum(min_v.abs(), max_v.abs()).clamp(min=eps)
    var_sym = quantization_noise_variance_sym(bits, max_abs)
    var_asym = quantization_noise_variance_asym(bits, min_v, max_v)
    var_ratio = var_sym / var_asym.clamp(min=eps)
    law_error = (var_ratio / gamma_sq.clamp(min=eps)) - 1.0

    return {
        "gamma": gamma,
        "gamma_sq": gamma_sq,
        "var_sym": var_sym,
        "var_asym": var_asym,
        "var_ratio": var_ratio,
        "law_error": law_error,
    }


def quantize_uniform(
    tensor: torch.Tensor,
    bits: int,
    *,
    symmetric: bool = True,
    keep_scale: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference uniform scalar quantizer used to demonstrate the gamma law.

    Args:
        tensor: Tensor to quantize.
        bits: Bitwidth.
        symmetric: If True, use zero-centered grid; otherwise use [min, max].
        keep_scale: If True, also return ``scale`` and ``zero``.

    Returns:
        Quantized tensor, or ``(quantized, scale, zero)`` when ``keep_scale``.
    """

    if symmetric:
        max_abs = tensor.abs().max()
        scale = step_size_sym(bits, max_abs)
        qmax = (2 ** (bits - 1)) - 1
        qmin = -(2 ** (bits - 1))
        q = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int64)
        quantized = scale * q.to(tensor.dtype)
        zero = torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)
    else:
        min_v = tensor.min()
        max_v = tensor.max()
        scale = step_size_asym(bits, min_v, max_v)
        zero = -min_v / scale
        qmax = (2 ** bits) - 1
        q = torch.clamp(torch.round(tensor / scale + zero), 0, qmax).to(torch.int64)
        quantized = scale * (q.to(tensor.dtype) - zero)

    if keep_scale:
        return quantized, scale, zero
    return quantized
