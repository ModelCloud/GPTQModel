# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from __future__ import annotations

import torch
from device_smi import Device
from torch import nn as nn

from ..models._const import CPU, CUDA_0


# unit: GiB
def get_gpu_usage_memory():
    smi = Device(CUDA_0)
    return smi.memory_used() / 1024 / 1024 / 1024 #GB

# unit: GiB
def get_cpu_usage_memory():
    smi = Device(CPU)
    return smi.memory_used() / 1024 / 1024 / 1024 #GB


def get_device(obj: torch.Tensor | nn.Module) -> torch.device:
    if isinstance(obj, torch.Tensor):
        return obj.device

    params = list(obj.parameters())
    buffers = list(obj.buffers())
    if len(params) > 0:
        return params[0].device
    elif len(buffers) > 0:
        return buffers[0].device
    else:
        return CPU
