import os

import psutil
import torch
import GPUtil

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
def get_GPU_memory():
    index = [int(s.strip()) for s in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if s][0]
    gpu = GPUtil.getGPUs()[index]
    free = gpu.memoryFree
    total = gpu.memoryTotal
    return (total - free) / 1024

# unit: GiB
def get_cpu_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024 / 1024