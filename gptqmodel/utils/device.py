# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from device_smi import Device

from ..models._const import CPU, CUDA_0


# unit: GiB
def get_gpu_usage_memory():
    smi = Device(CUDA_0)
    return smi.memory_used() / 1024 / 1024 / 1024 #GB

# unit: GiB
def get_cpu_usage_memory():
    smi = Device(CPU)
    return smi.memory_used() / 1024 / 1024 / 1024 #GB
