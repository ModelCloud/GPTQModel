# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Test-only reference encoder for FP8 decode kernels.

Production GPT-QModel is FP8 load-only: it reads real checkpoints and never
quantizes or exports FP8 weights. Kernel tests still need synthetic FP8
payloads, so this module keeps the small reference encoder the tests used to
reach through ``TorchFP8Linear.pack_original``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.fp8 import _compute_scale_inv, _fp8_dtype_from_name, _weight_to_matrix
from gptqmodel.quantization.config import (
    _normalize_fp8_fmt,
    _normalize_fp8_weight_block_size,
    _normalize_fp8_weight_scale_method,
)


def quantize_fp8_weight(
    weight: torch.Tensor,
    *,
    format: str = "float8_e4m3fn",
    weight_scale_method: str = "row",
    weight_block_size: Optional[Tuple[int, int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one dense matrix as a synthetic FP8 checkpoint payload."""

    if weight.ndim != 2:
        raise ValueError(f"FP8 quantization expects a 2D weight matrix, got shape {tuple(weight.shape)}.")

    format = _normalize_fp8_fmt(format)
    if format == "float8_e8m0fnu":
        raise ValueError(
            "TorchFP8Linear does not quantize dense weights to float8_e8m0fnu; "
            "use that format only for dequantization of existing checkpoints."
        )
    block_size = _normalize_fp8_weight_block_size(weight_block_size)
    weight_scale_method = _normalize_fp8_weight_scale_method(
        weight_scale_method,
        weight_block_size=block_size,
    )
    fp8_dtype = _fp8_dtype_from_name(format)
    fp8_max = torch.finfo(fp8_dtype).max

    weight = weight.to(device="cpu", dtype=torch.float32).contiguous()

    if weight_scale_method == "tensor":
        scale_inv = _compute_scale_inv(weight.abs().amax(), fp8_max)
        quantized = torch.clamp(weight * scale_inv, min=-fp8_max, max=fp8_max).to(fp8_dtype)
        return quantized.contiguous(), scale_inv.to(torch.float32)

    if weight_scale_method == "row":
        scale_inv = _compute_scale_inv(weight.abs().amax(dim=1), fp8_max)
        quantized = torch.clamp(
            weight * scale_inv.unsqueeze(1),
            min=-fp8_max,
            max=fp8_max,
        ).to(fp8_dtype)
        return quantized.contiguous(), scale_inv.to(torch.float32).contiguous()

    if block_size is None:
        raise ValueError("FP8 block quantization requires `weight_block_size`.")

    block_rows, block_cols = block_size
    rows, cols = weight.shape
    if rows % block_rows != 0 or cols % block_cols != 0:
        raise ValueError(
            f"FP8 block quantization expects shape {tuple(weight.shape)} to be divisible by block size "
            f"{block_size}."
        )

    row_blocks = rows // block_rows
    col_blocks = cols // block_cols
    blocks = weight.reshape(row_blocks, block_rows, col_blocks, block_cols)
    scale_inv = _compute_scale_inv(blocks.abs().amax(dim=(1, 3)), fp8_max)
    scaled = blocks * scale_inv.unsqueeze(1).unsqueeze(3)
    quantized = torch.clamp(scaled, min=-fp8_max, max=fp8_max).to(fp8_dtype).reshape(rows, cols)
    return quantized.contiguous(), scale_inv.to(torch.float32).contiguous()


def _install(module, weight: torch.Tensor, scale_inv: torch.Tensor, linear: nn.Module) -> None:
    if "weight" in module._buffers:
        module.weight = weight
    else:
        module.register_buffer("weight", weight)
    if "weight_scale_inv" in module._buffers:
        module.weight_scale_inv = scale_inv
    else:
        module.register_buffer("weight_scale_inv", scale_inv)

    if linear.bias is not None:
        bias = linear.bias.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if "bias" in module._buffers:
            module.bias = bias
        else:
            module.register_buffer("bias", bias)
    else:
        module.bias = None

    module._scaled_mm_hard_disabled = False
    module._native_dequant_disabled = False
    module._invalidate_compiled()


def pack_fp8_module(module, linear: nn.Module, *, smooth=None) -> None:
    """Install a synthetic FP8 payload computed from ``linear``'s dense weight."""

    weight = _weight_to_matrix(linear).to(device="cpu", dtype=torch.float32)
    if smooth is not None:
        from gptqmodel.nn_modules.qlinear.gguf import _apply_optional_smoother

        weight = _apply_optional_smoother(
            weight,
            smooth=smooth,
            group_size=module.smooth_block_size(),
        ).contiguous()
    qweight, scale_inv = quantize_fp8_weight(
        weight,
        format=module.fp8_format,
        weight_scale_method=module.weight_scale_method,
        weight_block_size=module.weight_block_size,
    )
    _install(module, qweight, scale_inv, linear)


def install_fp8_payload(module, linear: nn.Module, weight: torch.Tensor, scale_inv: torch.Tensor) -> None:
    """Install an explicit FP8 payload without recomputing its scales."""

    if weight.shape != (module.out_features, module.in_features) or weight.dtype != module.fp8_dtype:
        raise ValueError("Calibrated FP8 weight does not match packed storage geometry or dtype")
    if scale_inv.shape != module._scale_shape() or scale_inv.dtype != torch.float32:
        raise ValueError("Calibrated FP8 scale does not match packed storage geometry or dtype")
    if not torch.isfinite(scale_inv).all() or (scale_inv <= 0).any():
        raise ValueError("Calibrated FP8 inverse scales must be finite and positive")
    _install(
        module,
        weight.detach().to(device="cpu").contiguous(),
        scale_inv.detach().to(device="cpu").contiguous(),
        linear,
    )
