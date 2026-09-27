# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""End-to-end native MLX orchestration for EXL3 weight quantization."""

import math
import platform
import sys
from functools import lru_cache

import torch

from .mlx_exl3 import exl3_pack_trellis_mlx
from .mlx_exl3_fallback import exl3_fallback_quantize_mlx
from .mlx_exl3_gss import exl3_global_scale_search_mlx
from .mlx_exl3_hadamard import exl3_hadamard_128_mlx
from .mlx_exl3_hessian import exl3_finalize_hessian_mlx
from .mlx_exl3_ldlq import exl3_ldlq_quantize_mlx
from .mlx_exl3_regularize import exl3_regularize_transforms_mlx
from .mlx_native import _accurate_matmul_mlx


_EXL3_MCG_MARKER = 0xCBAC1FED
_EXL3_MUL1_MARKER = 0x83DCD12D


@lru_cache(maxsize=1)
def exl3_mlx_quantization_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def _mlx_array_to_torch(value, device: torch.device) -> torch.Tensor:
    return torch.from_dlpack(value).to(device=device).contiguous()


def exl3_quantize_weight_mlx_to_torch(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    sample_count: int,
    bits: int,
    codebook: str,
    force_output_scales,
    sigma_reg: float,
    seed: int,
):
    """Bridge Torch processor tensors through the native MLX EXL3 pipeline."""
    import mlx.core as mx

    if weight.device.type == "mps":
        torch.mps.synchronize()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_signs = (
        (torch.randn(weight.shape[0], generator=generator).sign() + 1e-5)
        .sign()
        .float()
    )
    generator.manual_seed(seed + 1)
    output_signs = (
        (torch.randn(weight.shape[1], generator=generator).sign() + 1e-5)
        .sign()
        .float()
    )
    result = exl3_quantize_weight_mlx(
        mx.from_dlpack(weight.contiguous()),
        mx.from_dlpack(hessian.contiguous()),
        mx.array(input_signs[:, None].numpy()),
        mx.array(output_signs[None, :].numpy()),
        sample_count=sample_count,
        bits=bits,
        codebook=codebook,
        force_output_scales=force_output_scales,
        sigma_reg=sigma_reg,
    )
    reconstructed, proxy_error, trellis, suh, svh = (
        _mlx_array_to_torch(value, weight.device) for value in result[:5]
    )
    out_tensors = {"suh": suh, "svh": svh, "trellis": trellis}
    if codebook == "mcg":
        out_tensors["mcg"] = torch.tensor(
            _EXL3_MCG_MARKER, dtype=torch.uint32
        ).view(torch.int32)
    elif codebook == "mul1":
        out_tensors["mul1"] = torch.tensor(
            _EXL3_MUL1_MARKER, dtype=torch.uint32
        ).view(torch.int32)
    return reconstructed, float(proxy_error.item()), out_tensors


def _exl3_proxy_error_mlx(weight, quantized, hessian, *, chunk_columns=1024):
    """Return EXL3's Hessian-weighted relative reconstruction error."""
    import mlx.core as mx

    numerator = mx.zeros((), dtype=mx.float32)
    denominator = mx.zeros((), dtype=mx.float32)
    error = weight - quantized
    for start in range(0, weight.shape[1], chunk_columns):
        stop = min(start + chunk_columns, weight.shape[1])
        weight_chunk = weight[:, start:stop]
        error_chunk = error[:, start:stop]
        numerator = numerator + mx.sum(
            error_chunk * _accurate_matmul_mlx(hessian, error_chunk)
        )
        denominator = denominator + mx.sum(
            weight_chunk * _accurate_matmul_mlx(hessian, weight_chunk)
        )
        mx.eval(numerator, denominator)
    return numerator / mx.maximum(denominator, mx.array(1e-8, dtype=mx.float32))


def exl3_quantize_weight_mlx(
    weight,
    hessian,
    input_signs,
    output_signs,
    *,
    sample_count: int,
    bits: int,
    codebook: str = "3inst",
    force_output_scales=None,
    sigma_reg: float = 0.025,
    workspace_bytes: int = 256 << 20,
):
    """Run EXL3's calibrated quantization pipeline with native MLX kernels.

    The weight uses EXL3's ``(input_features, output_features)`` layout and all
    matrix inputs use float32. Returned values are the reconstructed weight,
    proxy error, packed trellis, FP16 input/output scales, fallback decision,
    selected global scale, and output-scale decision.
    """
    import mlx.core as mx

    weight = mx.array(weight)
    hessian = mx.array(hessian)
    input_signs = mx.array(input_signs)
    output_signs = mx.array(output_signs)
    if weight.ndim != 2 or any(dimension == 0 for dimension in weight.shape):
        raise ValueError("weight must be a nonempty rank-two array")
    if weight.dtype != mx.float32:
        raise ValueError("weight must have float32 dtype")
    rows, columns = weight.shape
    if rows % 128 or columns % 128:
        raise ValueError("EXL3 dimensions must be divisible by 128")
    if hessian.shape != (rows, rows) or hessian.dtype != mx.float32:
        raise ValueError("hessian must be a matching float32 square matrix")
    if input_signs.shape != (rows, 1) or input_signs.dtype != mx.float32:
        raise ValueError("input_signs must be float32 with shape (input_features, 1)")
    if output_signs.shape != (1, columns) or output_signs.dtype != mx.float32:
        raise ValueError("output_signs must be float32 with shape (1, output_features)")
    if codebook not in {"3inst", "mcg", "mul1"}:
        raise ValueError("codebook must be '3inst', 'mcg', or 'mul1'")
    if not isinstance(bits, int) or isinstance(bits, bool) or not 1 <= bits <= 8:
        raise ValueError("bits must be an integer between 1 and 8")
    if not isinstance(workspace_bytes, int) or workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be a positive integer")

    fallback, transformed_hessian, factor, diagonal = exl3_finalize_hessian_mlx(
        hessian,
        input_signs[:, 0],
        sample_count=sample_count,
        sigma_reg=sigma_reg,
    )
    apply_output_scales, regularized, input_scales, output_scales = (
        exl3_regularize_transforms_mlx(
            weight,
            input_signs,
            output_signs,
            force_output_scales=force_output_scales,
            hessian_diagonal=diagonal,
            fallback=fallback,
        )
    )
    global_scale, _ = exl3_global_scale_search_mlx(
        regularized,
        bits=bits,
        codebook=codebook,
        workspace_bytes=workspace_bytes,
    )
    regularized = regularized * global_scale
    input_scales = input_scales / global_scale
    mx.eval(regularized, input_scales)

    if fallback:
        quantized, encoded = exl3_fallback_quantize_mlx(
            regularized,
            bits=bits,
            codebook=codebook,
            workspace_bytes=workspace_bytes,
        )
        proxy_error = mx.zeros((), dtype=mx.float32)
    else:
        quantized, encoded = exl3_ldlq_quantize_mlx(
            regularized,
            factor,
            bits=bits,
            codebook=codebook,
            workspace_bytes=workspace_bytes,
        )
        proxy_error = _exl3_proxy_error_mlx(
            regularized,
            quantized,
            transformed_hessian,
        )

    reconstructed = exl3_hadamard_128_mlx(quantized, axis=0) * input_scales
    reconstructed = exl3_hadamard_128_mlx(reconstructed, axis=1) * output_scales
    trellis = exl3_pack_trellis_mlx(encoded, bits=bits)
    input_scales = mx.contiguous(input_scales.reshape(-1).astype(mx.float16))
    output_scales = mx.contiguous(output_scales.reshape(-1).astype(mx.float16))
    reconstructed = mx.contiguous(reconstructed)
    proxy_error = mx.contiguous(proxy_error)
    mx.eval(reconstructed, proxy_error, trellis, input_scales, output_scales)
    if not math.isfinite(float(proxy_error.item())):
        raise ValueError("EXL3 proxy error is nonfinite")
    return (
        reconstructed,
        proxy_error,
        trellis,
        input_scales,
        output_scales,
        fallback,
        global_scale,
        apply_output_scales,
    )


__all__ = [
    "exl3_mlx_quantization_available",
    "exl3_quantize_weight_mlx",
    "exl3_quantize_weight_mlx_to_torch",
]
