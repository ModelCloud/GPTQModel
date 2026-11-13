# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import ctypes
import math

import torch


def tensor_buffer(tensor: torch.Tensor):
    """Return ``(owner, memoryview)`` over a contiguous CPU tensor payload.

    The view aliases the tensor's storage (no copy), so ``owner`` must stay
    alive while the view is used. ``ctypes`` provides the buffer protocol
    without importing NumPy.
    """
    view = tensor.detach()
    if view.is_conj() or view.is_neg():
        view = view.resolve_conj().resolve_neg()
    view = view.to(device="cpu").contiguous().reshape(-1)
    nbytes = view.nbytes
    if nbytes == 0:
        return view, memoryview(b"")
    return view, memoryview((ctypes.c_char * nbytes).from_address(view.data_ptr()))


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return a tensor's logical payload as raw bytes, without NumPy.

    Callers that only hash or stream the payload should use
    :func:`tensor_buffer` to avoid materializing a second copy.
    """
    _owner, buffer = tensor_buffer(tensor)
    return bytes(buffer)


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
