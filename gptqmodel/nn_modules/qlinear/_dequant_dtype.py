# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Resolve the dense-weight dtype used by ``dequantize_model``.

GPTQ checkpoints store ``scales`` in FP16 even when the loaded model computes
in BF16, and the loader deliberately restores that checkpoint dtype.  Taking
the dequantized ``nn.Linear`` dtype from ``scales`` therefore produced FP16
dense layers inside a BF16 model, and the first matmul failed with a
Half/BFloat16 mismatch.  The dense weights must follow the model's compute
dtype instead.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import BaseQuantLinear


def _non_quantized_dtype(model: torch.nn.Module, accessor: str) -> Optional[torch.dtype]:
    """Most common floating dtype among non-quantized tensors of one kind."""
    counts: dict[torch.dtype, int] = {}
    for module in model.modules():
        if isinstance(module, BaseQuantLinear):
            continue
        for tensor in getattr(module, accessor)(recurse=False):
            if tensor is not None and tensor.is_floating_point():
                counts[tensor.dtype] = counts.get(tensor.dtype, 0) + 1
    if not counts:
        return None
    # ``max`` keeps the first inserted dtype on a tie, matching module order
    # (embeddings and norms usually precede the quantized stack).
    return max(counts, key=counts.get)


def resolve_dequant_dtype(
    model: torch.nn.Module,
    *,
    fallback: torch.dtype = torch.float16,
) -> torch.dtype:
    """Return the dtype the model computes in, for dequantized dense weights."""
    # Non-quantized parameters are the tensors the rest of the forward
    # multiplies with, so they are the strongest signal.
    dtype = _non_quantized_dtype(model, "parameters")
    if dtype is not None:
        return dtype

    model_dtype = getattr(model, "dtype", None)
    if isinstance(model_dtype, torch.dtype) and model_dtype.is_floating_point:
        return model_dtype

    # Buffers come last: models often keep FP32 rotary/attention buffers
    # alongside BF16 compute weights.
    dtype = _non_quantized_dtype(model, "buffers")
    if dtype is not None:
        return dtype

    return fallback
