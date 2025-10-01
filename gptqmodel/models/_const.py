# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import transformers
from torch import device

from ..utils import BACKEND
from ..utils.rocm import IS_ROCM
from ..utils.torch import HAS_CUDA, HAS_MPS, HAS_XPU


CPU = device("cpu")
META = device("meta")
CUDA = device("cuda")
CUDA_0 = device("cuda:0")
XPU = device("xpu")
XPU_0 = device("xpu:0")
MPS = device("mps")
ROCM = device("cuda:0")  # rocm maps to fake cuda

SUPPORTS_MODULE_TYPES = [nn.Linear, nn.Conv1d, nn.Conv2d, transformers.Conv1D]

DEFAULT_MAX_SHARD_SIZE = "4GB"

class DEVICE(str, Enum):
    ALL = "all"  # All device
    CPU = "cpu"  # All CPU: Optimized for IPEX is CPU has AVX, AVX512, AMX, or XMX instructions
    CUDA = "cuda"  # Nvidia GPU: Optimized for Ampere+
    XPU = "xpu"  # Intel GPU: Datacenter Max + Arc
    MPS = "mps"  # MacOS GPU: Apple Silicon/Metal)
    ROCM = "rocm"  # AMD GPU: ROCm maps to fake cuda

    @classmethod
    # conversion method called for init when string is passed, i.e. Device("CUDA")
    def _missing_(cls, value):
        if IS_ROCM and f"{value}".lower() == "rocm":
            return cls.ROCM
        return super()._missing_(value)

    @property
    def type(self) -> str:
        """Return the backend type compatible with torch.device semantics."""
        if self == DEVICE.ROCM:
            return "cuda"
        return self.value

    @property
    def index(self) -> int | None:
        """Default index used when materialising a torch.device from this enum."""
        if self in (DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU):
            return 0
        return None

    def to_torch_device(self) -> torch.device:
        """Convert the enum to a concrete torch.device, defaulting to index 0."""
        idx = self.index
        return torch.device(self.type if idx is None else f"{self.type}:{idx}")

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
    if HAS_CUDA:
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
