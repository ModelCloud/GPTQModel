from typing import Optional

import torch


def torch_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def torch_empty_cache(device: torch.device = None):
    # check all backends
    if device is None:
        torch.cuda.empty_cache()
        if hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
        return

    # if device passed, only execute for device backend
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()