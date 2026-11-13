# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math

import torch


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return a tensor's logical payload as raw bytes.

    NumPy is used when it is importable because its buffer bridge is the fast
    path for multi-hundred-MiB weight hashing; when it is absent the same bytes
    are produced from a Torch copy. GPT-QModel does not depend on NumPy, so this
    must not fail in a NumPy-free install.
    """
    view = tensor.detach().to(device="cpu").contiguous().reshape(-1).view(torch.uint8)
    try:
        import numpy  # noqa: F401  # presence probe only; the bridge below is Torch's
    except ModuleNotFoundError:
        return bytes(view.tolist())
    # ``Tensor.numpy`` is a zero-copy bridge; copying once through it is the
    # fastest NumPy-free-agnostic path and matches the old call sites exactly.
    return view.numpy().tobytes()


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
