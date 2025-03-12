# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

import torch
import torch.nn as nn
import transformers
from torch import device

from ..utils import BACKEND
from ..utils.rocm import IS_ROCM
from ..utils.torch import HAS_CUDA, HAS_MPS, HAS_XPU

CPU = device("cpu")
CUDA = device("cuda")
CUDA_0 = device("cuda:0")
XPU = device("xpu")
XPU_0 = device("xpu:0")
MPS = device("mps")
ROCM = device("cuda:0")  # rocm maps to fake cuda

SUPPORTS_MODULE_TYPES = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]

DEFAULT_MAX_SHARD_SIZE = "4GB"

class DEVICE(str, Enum):
    ALL = "all"  # All device
    CPU = "cpu"  # All CPU: Optimized for IPEX is CPU has AVX, AVX512, AMX, or XMX instructions
    CUDA = "cuda"  # Nvidia GPU: Optimized for Ampere+
    XPU = "xpu"  # Intel GPU: Datacenter Max + Arc
    MPS = "mps"  # MacOS GPU: Apple Silion/Metal)
    ROCM = "rocm"  # AMD GPU: ROCm maps to fake cuda

    @classmethod
    # conversion method called for init when string is passed, i.e. Device("CUDA")
    def _missing_(cls, value):
        if IS_ROCM and f"{value}".lower() == "rocm":
            return cls.ROCM
        return super()._missing_(value)

    def to_device_map(self):
        return {"": DEVICE.CUDA if self == DEVICE.ROCM else self}


class PLATFORM(str, Enum):
    ALL = "all"  # All platform
    LINUX = "linux"  # linux
    WIN32 = "win32"  # windows
    DARWIN = "darwin"  # macos


def validate_cuda_support(raise_exception: bool = False):
    got_cuda = HAS_CUDA
    if got_cuda:
        at_least_one_cuda_v6 = any(
            torch.cuda.get_device_capability(i)[0] >= 6 for i in range(torch.cuda.device_count()))

        if not at_least_one_cuda_v6:
            if raise_exception:
                raise EnvironmentError(
                    "GPTQModel cuda requires Pascal or later gpu with compute capability >= `6.0`.")
            else:
                got_cuda = False

    return got_cuda


def normalize_device(type_value: str | DEVICE | int | torch.device) -> DEVICE:
    if isinstance(type_value, int):
        if HAS_CUDA:
            return DEVICE.CUDA
        elif HAS_XPU:
            return DEVICE.XPU
        elif HAS_MPS:
            return DEVICE.MPS
        else:
            return DEVICE.CPU

    if isinstance(type_value, torch.device):
        type_value = type_value.type

    # remove device index
    split_results = [s.strip() for s in type_value.split(":") if s]
    if len(split_results) > 1:
        type_value = split_results[0]

    if isinstance(type_value, DEVICE):
        return type_value

    if not isinstance(type_value, str):
        raise ValueError(f"Invalid device type_value type: {type(type_value)}")

    return DEVICE(type_value.lower())


def get_best_device(backend: BACKEND = BACKEND.AUTO) -> torch.device:
    if backend == BACKEND.IPEX:
        return XPU_0 if HAS_XPU else CPU
    elif HAS_CUDA:
        return CUDA_0
    elif HAS_XPU:
        return XPU_0
    elif HAS_MPS:
        return MPS
    else:
        return CPU

EXLLAMA_DEFAULT_MAX_INPUT_LENGTH = 2048

EXPERT_INDEX_PLACEHOLDER = "{expert_index}"

CALIBRATION_DATASET_CONCAT_CHAR = " "
