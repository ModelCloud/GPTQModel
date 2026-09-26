# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx

"""Native MLX fallback quantization for EXL3 weights."""

from .mlx_exl3 import exl3_quantize_tiles_mlx
from .mlx_exl3_tiles import (
    exl3_from_tensor_core_tiles_mlx,
    exl3_to_tensor_core_tiles_mlx,
)


def exl3_fallback_quantize_mlx(
    weight,
    *,
    bits: int,
    codebook: str = "mcg",
    workspace_bytes: int = 256 << 20,
):
    """Quantize a regularized EXL3 weight without Hessian compensation.

    EXL3 weights use ``(input_features, output_features)`` layout and float32
    quantization arithmetic. The reconstructed float32 weight and unpacked
    int16 trellis states are returned in matrix and tensor-core tile order,
    respectively.
    """
    import mlx.core as mx

    weight = mx.array(weight)
    if weight.dtype != mx.float32:
        raise ValueError("weight must have float32 dtype")
    if weight.ndim != 2 or any(dimension == 0 for dimension in weight.shape):
        raise ValueError("weight must be a nonempty rank-two array")
    if weight.shape[0] % 16:
        raise ValueError("input features must be divisible by 16")
    if weight.shape[1] % 128:
        raise ValueError("output features must be divisible by 128")

    tiles = exl3_to_tensor_core_tiles_mlx(weight)
    quantized_tiles, encoded = exl3_quantize_tiles_mlx(
        tiles,
        bits=bits,
        codebook=codebook,
        workspace_bytes=workspace_bytes,
    )
    quantized_weight = exl3_from_tensor_core_tiles_mlx(quantized_tiles)
    return quantized_weight, encoded


__all__ = ["exl3_fallback_quantize_mlx"]
