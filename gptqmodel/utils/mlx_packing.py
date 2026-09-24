# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Layout reference: MLX-LM (Apple Inc., MIT), mlx_lm/utils.py.
"""Transfer 4-bit GPTQ/AWQ weights into MLX's affine quantized layout."""

import numpy as np


_SHIFTS = np.arange(8, dtype=np.uint32) * 4
_AWQ_SHIFTS = np.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.uint32) * 4


def _pack_rows(codes):
    out_features, in_features = codes.shape
    if in_features % 8:
        raise ValueError("MLX 4-bit packing requires input features divisible by 8")
    return np.bitwise_or.reduce(
        codes.reshape(out_features, in_features // 8, 8).astype(np.uint32) << _SHIFTS,
        axis=-1,
    )


def repack_gptq_4bit(qweight, qzeros, scales, in_features, out_features):
    """Return (weight, scales, biases) for a GPTQ v2 linear layer."""
    if qweight.shape != (in_features // 8, out_features):
        raise ValueError("Unsupported GPTQ qweight shape")
    if qzeros.shape != (scales.shape[0], out_features // 8):
        raise ValueError("Unsupported GPTQ qzeros shape")
    if scales.shape[1] != out_features:
        raise ValueError("Unsupported GPTQ scales shape")

    codes = ((qweight.astype(np.uint32)[:, None, :] >> _SHIFTS[None, :, None]) & 15)
    codes = codes.reshape(in_features, out_features).T
    zeros = ((qzeros.astype(np.uint32)[:, :, None] >> _SHIFTS) & 15)
    zeros = zeros.reshape(scales.shape).T
    mlx_scales = np.ascontiguousarray(scales.T)
    biases = (-zeros.astype(np.float32) * mlx_scales.astype(np.float32)).astype(mlx_scales.dtype)
    return _pack_rows(codes), mlx_scales, biases


def repack_awq_4bit(qweight, qzeros, scales, in_features, out_features):
    """Return (weight, scales, biases) for an AWQ GEMM linear layer."""
    if qweight.shape != (in_features, out_features // 8):
        raise ValueError("Unsupported AWQ qweight shape")
    if qzeros.shape != (scales.shape[0], out_features // 8):
        raise ValueError("Unsupported AWQ qzeros shape")
    if scales.shape[1] != out_features:
        raise ValueError("Unsupported AWQ scales shape")

    codes = ((qweight.astype(np.uint32)[:, :, None] >> _AWQ_SHIFTS) & 15)
    codes = codes.reshape(in_features, out_features).T
    zeros = ((qzeros.astype(np.uint32)[:, :, None] >> _AWQ_SHIFTS) & 15)
    zeros = zeros.reshape(scales.shape).T
    mlx_scales = np.ascontiguousarray(scales.T)
    biases = (-zeros.astype(np.float32) * mlx_scales.astype(np.float32)).astype(mlx_scales.dtype)
    return _pack_rows(codes), mlx_scales, biases
