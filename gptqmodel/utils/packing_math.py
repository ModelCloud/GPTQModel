# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch


def scaled_zero_offset(scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    """Build an affine offset for backends that store it in the scale dtype."""
    offset = (-scales.float() * zeros.float()).to(scales.dtype)
    if not torch.isfinite(offset).all():
        raise ValueError(
            f"Scaled zero-point offset is not finite in {scales.dtype}; "
            "use a wider compute dtype or a backend that stores integer zero points."
        )
    return offset.contiguous()


def awq_storage_dtype(scale_dtype: torch.dtype, *, gemm: bool) -> torch.dtype:
    """Resolve the checkpoint scale dtype exactly as the AWQ packers store it.

    AWQ GEMM keeps an FP16 or BF16 source scale and otherwise stores FP16.
    AWQ GEMV and GEMV-fast always store FP16.
    """
    if gemm and scale_dtype in (torch.float16, torch.bfloat16):
        return scale_dtype
    return torch.float16


def awq_packed_codes(
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    g_idx: torch.Tensor,
    *,
    bits: int,
    scale_dtype: torch.dtype,
) -> torch.Tensor:
    """Recover the integer codes the AWQ packers store for ``weight``.

    The AWQ GEMM, GEMV and GEMV-fast packers form the affine offset from the
    source scales in FP32, divide by the checkpoint-stored scale in FP32,
    round, and saturate to the code range. Use the same operand order,
    precision and saturation so a predicted code always names the exported
    code instead of a neighbouring value.
    """
    stored = scales.to(scale_dtype)
    offset = zeros.float() * scales.float()
    values = (weight.float() + offset[:, g_idx]) / stored.float()[:, g_idx]
    return values.round().clamp_(0, (1 << bits) - 1)
