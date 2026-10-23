# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Small cache-key helpers shared by CPU quantized-linear runtimes."""

from __future__ import annotations

from typing import Optional

import torch


def tensor_cache_key(tensor: Optional[torch.Tensor]):
    """Return a cheap identity/version key for a tensor or ``None``.

    ``id`` catches replacement while ``_version`` catches in-place writes. The
    shape/device/dtype fields keep a cache from surviving a ``.to()`` or
    ``resize_()`` operation that reuses the same Python object.

    Inference tensors do not expose a version counter. PyTorch also rejects
    in-place updates to them outside ``InferenceMode``, which is where model
    weights are normally packed and served. Use their stable storage identity
    here so the real packed-weight caches can hit. The base quantized-linear
    class clears these caches on state-dict load, device/dtype moves and
    training transitions; code that mutates an inference tensor from inside
    ``InferenceMode`` must call ``clear_weight_cache()`` explicitly.
    """

    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        version = None if tensor.is_inference() else tensor._version
    except RuntimeError:
        version = None
    return (
        id(tensor),
        version,
        tensor.data_ptr(),
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tensor.is_inference(),
    )


__all__ = ["tensor_cache_key"]
