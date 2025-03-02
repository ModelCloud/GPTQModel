import os

import torch
from accelerate.utils import find_tied_parameters
from safetensors import safe_open

from ..utils.model import recurse_getattr, recurse_setattr


# debug print all safetensor files in a directory and print its properties
def inspect_safetensors(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(directory, filename)
            print(f"Safetensor File: {filename}")

            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    print(f"  Key: {key}")
                    print(f"    Shape: {tensor.shape}")
                    print(f"    Dtype: {tensor.dtype}")
            print("-" * 40)


# Safetensors is unable to save tied weights, so we untie them here. Reference: https://github.com/huggingface/safetensors/issues/202
def untie_weights(model):
    tied_params = find_tied_parameters(model)

    for weight_group in tied_params:
        for param_name in weight_group:
            if isinstance(recurse_getattr(model, param_name), torch.nn.Parameter):
                recurse_setattr(
                    model,
                    param_name,
                    torch.nn.Parameter(recurse_getattr(model, param_name).clone()),
                )
            else:
                recurse_setattr(
                    model,
                    param_name,
                    recurse_getattr(model, param_name).clone(),
                )
