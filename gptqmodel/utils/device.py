
from device_smi import Device
from gptqmodel.models._const import CPU, CUDA_0


# unit: GiB
def get_gpu_usage_memory():
    smi = Device(CUDA_0)
    return smi.memory_used() / 1024 / 1024 / 1024 #GB

# unit: GiB
def get_cpu_usage_memory():
    smi = Device(CPU)
    return smi.memory_used() / 1024 / 1024 / 1024 #GB
