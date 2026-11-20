# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Range-safe decode dtype selection for dense quant fallbacks.

A checkpoint stores one scale per group and integer codes in
``[0, 2**bits - 1]``, so a decoded weight is ``(code - zero) * scale`` with
magnitude at most ``(2**bits - 1) * |scale|``. Decoding in the scale's own
dtype is only safe when that bound stays finite and a nonzero scale does not
round to zero there; otherwise the product overflows to ``inf`` (or the weight
becomes ``0``) before any wider accumulation can help.

:func:`decode_dtype_for_scales` returns a dtype that holds every decoded
weight. It is shared by the CPU int4pack runtime decode and the GPTQ dense
dequantize fallback. A caller with an activation dtype asks for a decode that
does not narrow below the stored scale precision.
"""

from __future__ import annotations

import torch

from ...utils.torch import dynamo_is_compiling


_HALF_DTYPES = (torch.float16, torch.bfloat16)


def scales_fit_dtype(scales, bits: int, dtype: torch.dtype) -> bool:
    """Whether every decoded weight from ``scales`` stays in ``dtype``."""

    if not isinstance(scales, torch.Tensor) or scales.numel == 0 or scales.device.type == "meta":
        return True
    span = (1 << bits) - 1
    magnitude = scales.detach().abs()
    converted = magnitude.to(dtype)
    representable = (
        converted.isfinite()
        & (converted * span).isfinite()
        & ~((magnitude > 0) & (converted == 0))
    )
    return bool(representable.all())


def decode_dtype_for_scales(
    scales, bits: int, activation_dtype: torch.dtype | None = None
) -> torch.dtype:
    """Return a dtype that decodes every stored weight without range loss.

    ``activation_dtype`` is the forward's operand dtype. A wide activation
    decodes in its own dtype; a half activation decodes in the stored scale
    dtype while that is safe, and in FP32 (or FP64 for a pathological BF16
    scale) otherwise. ``None`` asks for a decode that is safe for the stored
    scales alone.
    """

    if activation_dtype is not None and activation_dtype not in _HALF_DTYPES:
        return activation_dtype
    if not isinstance(scales, torch.Tensor):
        return activation_dtype or torch.float32
    if dynamo_is_compiling():
        # A host reduction is not traceable, and a fullgraph trace must not
        # graph-break on one. Read only the dtype (a Python constant). A
        # checkpoint-layout caller (``activation_dtype is None``) keeps the
        # historical scale dtype; the runtime decode passes the activation
        # dtype and gets the wide fallback, because that path has already
        # recorded that a half decode is unsafe (#5577).
        if activation_dtype is None:
            return scales.dtype if scales.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
        return torch.float64 if scales.dtype == torch.bfloat16 else torch.float32
    if scales.numel == 0 or scales.device.type == "meta":
        return activation_dtype or torch.float32
    candidate = scales.dtype if scales.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    if scales_fit_dtype(scales, bits, candidate):
        return candidate
    if scales_fit_dtype(scales, bits, torch.float32):
        return torch.float32
    return torch.float64


__all__ = ["decode_dtype_for_scales", "scales_fit_dtype"]
