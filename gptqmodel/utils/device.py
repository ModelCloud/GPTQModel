
import torch
from device_smi import Device
from gptqmodel.models._const import CPU, CUDA_0


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
    smi = Device(CUDA_0)
    return smi.memory_used() / 1024 / 1024 / 1024 #GB

# unit: GiB
def get_cpu_usage_memory():
    smi = Device(CPU)
    return smi.memory_used() / 1024 / 1024 / 1024 #GB
