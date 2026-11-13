# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import ctypes
import math

import torch


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return a tensor's logical payload as raw bytes, without NumPy.

    ``ctypes.string_at`` copies the contiguous CPU buffer in one C call, which
    matches ``Tensor.numpy().tobytes()`` throughput (it is the same memcpy) and
    avoids both a NumPy import and the very slow ``tolist()`` fallback.
    """
    view = tensor.detach()
    if view.is_conj() or view.is_neg():
        view = view.resolve_conj().resolve_neg()
    view = view.to(device="cpu").contiguous().reshape(-1)
    nbytes = view.nbytes
    if nbytes == 0:
        return b""
    return ctypes.string_at(view.data_ptr(), nbytes)


# helper method to get accurate parameter count for a gptq model
def tensor_parameters(
        tensor_name: str,  # name of tensor weight in model
        tensor_shape: torch.Size,  # shape of tensor
        bits: int,  # gptq bits
) -> int:
    # only .qweight is relevant for `parameters` in gptq model
    if tensor_name.endswith(".qweight"):
        real_infeatures = math.ceil(tensor_shape[0] / bits * 32)
        real_tensor_shape = (real_infeatures,) + tensor_shape[1:]
        return math.prod(real_tensor_shape)
    # .scales and .qzeros are not model parameters but aux data for .qweight
    elif tensor_name.endswith((".scales", ".qzeros", ".g_idx")):
        return 0
    else:
        return math.prod(tensor_shape)
