# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch


def scale_to_storage_dtype(
    scales: torch.Tensor, dtype: torch.dtype, *, label: str = "AWQ scales"
) -> torch.Tensor:
    """Cast scales to the dtype a kernel stores, rejecting lossy endpoints.

    A scale that overflows to infinity or underflows to zero is not
    representable in the storage dtype.  Every code derived from that scale is
    then wrong, and a later clamp cannot recover the lost information.  Check
    the actual cast result so the decision matches what the kernel will load.
    """
    converted = scales.to(dtype)
    if not bool(torch.isfinite(converted).all()):
        raise ValueError(
            f"{label} cannot be stored in {dtype}: the converted scale is not "
            "finite; use a wider scale dtype or a backend that keeps FP32 scales."
        )
    if bool(((scales != 0) & (converted == 0)).any()):
        raise ValueError(
            f"{label} cannot be stored in {dtype}: a nonzero scale underflows to "
            "zero; use a wider scale dtype or a backend that keeps FP32 scales."
        )
    return converted


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
    checkpoint-stored scale in FP32, divide by that same scale in FP32, round,
    and saturate to the code range. Use the same operand order, precision and
    saturation so a predicted code always names the exported code instead of a
    neighbouring value.
    """
    stored = scales.to(scale_dtype)
    offset = zeros.float() * stored.float()
    values = (weight.float() + offset[:, g_idx]) / stored.float()[:, g_idx]
    return values.round().clamp_(0, (1 << bits) - 1)
