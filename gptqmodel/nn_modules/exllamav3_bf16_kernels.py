# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Fused BF16 -> FP16 pre-scale and FP16/FP32 -> BF16 re-scale for EXL3.

The native EXL3 GEMM consumes FP16 activations.  A BF16 checkpoint therefore
maps each activation row onto FP16 before the GEMM and undoes the per-row
power-of-two pre-scale afterwards.  The eager implementation of that mapping
(``ExllamaV3Linear._native_bf16_input`` / ``_rescale_native_output``) is a chain
of about a dozen tiny CUDA launches per linear.  At decode batch size the GEMM
is short enough that those launches, not the GPU, dominate the BF16 forward
(#5148).

``gptqmodel_ext/exllamav3/bf16_scale.cu`` keeps the same arithmetic but does
each half in one launch:

* ``bf16_prescale`` reduces the row ``amax``, derives the power-of-two scale
  from ``frexp(amax / bound)``, divides and casts to FP16, and writes the FP32
  per-row scale.  It never reads a value back to the host (#4955).
* ``bf16_rescale.raw`` applies ``out * scale + bias`` in FP32 and writes BF16.
  The original ``bf16_rescale`` overload accepts a result that includes bias
  and removes it before multiplying by the row scale.

Every floating-point step uses explicit round-to-nearest operations, so the
result is bit-identical to the eager torch sequence: the division by a power of
two is exact, the FP16 cast is RTNE, and the rescale multiply and bias add
round separately in FP32. Rows that do not need scaling keep a scale of exactly
1.0.

Both entry points return ``None`` when the fused path cannot run; the caller
then uses the eager reference.  The fallbacks are a non-CUDA device, an
unsupported dtype/shape, a non-contiguous input, or an extension build that
does not provide the op (for example a stale prebuilt library).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


_FP16_MAX = float(torch.finfo(torch.float16).max)
# Keep in sync with ``gptqmodel.nn_modules.exllamav3._FP16_SAFE_AMAX``.
FP16_SAFE_AMAX = _FP16_MAX / math.sqrt(128.0)

# Resolved torch.ops handles.  A miss is not cached so the first forward (which
# runs before the JIT extension is loaded) can fall back and later calls pick
# the kernel up once ``ExllamaV3Linear._ensure_inner`` has loaded it.
_OPS: dict[str, object] = {}


def _op(name: str):
    op = _OPS.get(name)
    if op is not None:
        return op
    try:
        op = getattr(torch.ops.gptqmodel_exllamav3, name)
    except Exception:
        return None
    _OPS[name] = op
    return op


def fused_available() -> bool:
    """Whether the compiled extension exposes both fused kernels."""
    return _op("bf16_prescale") is not None and _op("bf16_rescale") is not None


def _can_prescale(x: torch.Tensor) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() >= 1
        and x.is_contiguous()
        and x.numel() > 0
    )


def try_prescale(x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused ``_native_bf16_input``; ``None`` means use the eager reference."""
    if not _can_prescale(x):
        return None
    op = _op("bf16_prescale")
    if op is None:
        return None
    out, scale = op(x, FP16_SAFE_AMAX)
    return out, scale


def _can_rescale(
    result: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_features: int,
) -> bool:
    if not isinstance(result, torch.Tensor):
        return False
    if not result.is_cuda or result.dtype not in (torch.float16, torch.float32):
        return False
    if result.dim() < 1 or not result.is_contiguous():
        return False
    if result.shape[-1] < out_features or out_features <= 0:
        return False
    if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32:
        return False
    if not scale.is_cuda or not scale.is_contiguous() or scale.device != result.device:
        return False
    rows = result.numel() // result.shape[-1]
    if scale.numel() != rows:
        return False
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or bias.device != result.device:
            return False
        if bias.dim() != 1 or not bias.is_contiguous() or bias.numel() < out_features:
            return False
        if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
    return True


def try_rescale(
    result: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_features: int,
    *,
    result_has_bias: bool = True,
) -> Optional[torch.Tensor]:
    """Fused ``_rescale_native_output`` + BF16 cast; ``None`` means fallback."""
    if not _can_rescale(result, scale, bias, out_features):
        return None
    op = _op("bf16_rescale")
    if bias is not None and not result_has_bias:
        op = getattr(op, "raw", None)
    if op is None:
        return None
    return op(result, scale, bias, out_features)
