import torch
import math
import numpy as np

# helper method to get accurate parameter count for a gptq model
def tensor_parameters(
        tensor_name: str,  # name of tensor weight in model
        tensor_shape: torch.Size,  # shape of tensor
        bits: int,  # gptq bits
):
    # only .qweight is relevant for `parameters` in gptq model
    if tensor_name.endswith(".qweight"):
        real_infeatures = math.ceil(tensor_shape[0] / bits * 32)
        real_tensor_shape = (real_infeatures,) + tensor_shape[1:]
        return np.prod(real_tensor_shape)
    # .scales and .qzeros are not model parameters but aux data for .qweight
    elif tensor_name.endswith((".scales", ".qzeros", ".g_idx")):
        return 0
    else:
        return np.prod(tensor_shape)
