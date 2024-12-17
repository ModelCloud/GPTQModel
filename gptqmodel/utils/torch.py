import torch
import gc as py_gc

HAS_CUDA = False
HAS_XPU = False
HAS_MPS = False

if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
    HAS_CUDA = True

if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
    HAS_XPU = True

if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
    HAS_MPS = True


def torch_sync(device: torch.device = None):
    # check all backends
    if device is None:
        if HAS_CUDA:
            torch.cuda.synchronize()
        if HAS_XPU:
            torch.xpu.synchronize()
        if HAS_MPS:
            torch.mps.synchronize()
        return

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def torch_empty_cache(device: torch.device = None, gc: bool = True):
    if gc:
        py_gc.collect()

    # check all backends
    if device is None:
        if HAS_CUDA:
            torch.cuda.empty_cache()
        if HAS_XPU:
            torch.xpu.empty_cache()
        if HAS_MPS:
            torch.mps.empty_cache()
        return

    # if device passed, only execute for device backend
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()