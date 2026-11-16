# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pick the width of the bias buffer handed to an AllSpark W8A16 kernel.

Both AllSpark kernels add the bias inside their FP32 epilogue and accept the
bias either in the activation dtype or in FP32. A 16-bit buffer is what keeps a
biased layer on the fast path: the HIP prefill branch only reaches the
half-precision ``torch.addmm`` when the bias width matches the activation, and
the fused CUDA kernels read half as many bias bytes.

Narrowing is only safe when every value survives the round trip exactly. An
FP32 ``-65536`` is ``-inf`` in FP16, and ``-100000`` is not representable in
either 16-bit format, so those buffers must stay FP32.
"""

from __future__ import annotations

from typing import Optional

import torch


def narrow_bias_for_kernel(
    bias: Optional[torch.Tensor], dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Return ``bias`` in ``dtype`` when that is lossless, else unchanged.

    ``dtype`` must be the kernel's activation dtype. Values are compared after
    widening the candidate back to FP32, which also rejects an overflow to
    ``inf`` and a NaN (``torch.equal`` is false for NaN).
    """
    if bias is None or bias.dtype == dtype:
        return bias
    if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return bias
    if dtype not in (torch.float16, torch.bfloat16):
        return bias
    candidate = bias.to(dtype)
    if torch.equal(candidate.to(torch.float32), bias.to(torch.float32)):
        return candidate
    return bias


__all__ = ["narrow_bias_for_kernel"]
