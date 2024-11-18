import os

from DeviceSMI import DeviceSMI
from ._const import CPU, CUDA_0
import torch


def check_cuda(raise_exception: bool = True) -> bool:
    at_least_one_cuda_v6 = any(torch.cuda.get_device_capability(i)[0] >= 6 for i in range(torch.cuda.device_count()))

    if not at_least_one_cuda_v6:
        if raise_exception:
            raise EnvironmentError("GPTQModel requires at least one GPU device with CUDA compute capability >= `6.0`.")
        else:
            return False
    else:
        return True

# unit: GiB
def get_gpu_usage_memory():
    smi = DeviceSMI(CUDA_0)
    info = smi.get_info()
    return info.__dict__["memory_used"] / 1024 / 1024 / 1024 #GB

# unit: GiB
def get_cpu_usage_memory():
    smi = DeviceSMI(CPU)
    info = smi.get_info()
    return info.__dict__["memory_used"] / 1024 / 1024 / 1024 #GB
