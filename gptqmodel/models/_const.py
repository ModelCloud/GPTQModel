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
from ..utils.torch import HAS_CUDA, HAS_MPS, HAS_NPU, HAS_XPU


def _optional_torch_device(spec: str, fallback: torch.device) -> torch.device:
    try:
        return device(spec)
    except (RuntimeError, ValueError):
        return fallback


CPU = device("cpu")
META = device("meta")
CUDA = device("cuda")
CUDA_0 = device("cuda:0")
XPU = device("xpu")
XPU_0 = device("xpu:0")
NPU = _optional_torch_device("npu", CPU)
NPU_0 = _optional_torch_device("npu:0", CPU)
MPS = device("mps")
ROCM = device("cuda:0")  # Concrete HIP runtime target; use DEVICE.ROCM for the AMD family.

SUPPORTS_MODULE_TYPES = [nn.Embedding, nn.Linear, nn.Conv1d, nn.Conv2d, transformers.Conv1D]

DEFAULT_MAX_SHARD_SIZE = "4GB"

class DEVICE(str, Enum):
    """GPTQModel hardware families, distinct from PyTorch's runtime namespace.

    In SUPPORTS_DEVICES, CUDA declares NVIDIA and ROCM declares AMD. HIP also
    calls its runtime devices "cuda"; normalize requests with normalize_device
    and translate back only at Torch boundaries, never in support declarations.
    """

    ALL = "all"  # All device
    CPU = "cpu"  # All CPU: Optimized for IPEX is CPU has AVX, AVX512, AMX, or XMX instructions
    CUDA = "cuda"  # Nvidia GPU: Optimized for Ampere+
    XPU = "xpu"  # Intel GPU: Datacenter Max + Arc
    NPU = "npu"  # Ascend NPU: portable Torch kernels
    MPS = "mps"  # MacOS GPU: Apple Silicon/Metal)
    ROCM = "rocm"  # AMD GPU family; HIP's Torch runtime namespace is "cuda".

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
        if self in (DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU, DEVICE.NPU):
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
    if got_cuda and IS_ROCM:
        # ROCm reuses the ``cuda`` device namespace, where
        # ``get_device_capability`` returns an unrelated numeric pair. Every
        # supported gfx target is newer than the Pascal-era floor below, so the
        # SM check only applies to NVIDIA builds.
        return got_cuda
    if got_cuda:
        at_least_one_cuda_v6 = any(
            torch.cuda.get_device_capability(i)[0] >= 6 for i in range(torch.cuda.device_count()))

        if not at_least_one_cuda_v6:
            if raise_exception:
                raise EnvironmentError(
                    "GPT-QModel cuda requires Pascal or later gpu with compute capability >= `6.0`.")
            else:
                got_cuda = False

    return got_cuda


def normalize_device(type_value: str | DEVICE | int | torch.device) -> DEVICE:
    """Return the kernel support family for a public device value.

    HIP reuses torch's ``cuda`` namespace, so on a ROCm host an integer ordinal
    and every ``cuda`` spelling name the AMD accelerator.  They must map to
    ``DEVICE.ROCM``; keeping them as ``DEVICE.CUDA`` lets CUDA-only kernels
    (Triton V2, W4A-FP8/NVFP4, ...) pass the support-family gate on ROCm.
    ``DEVICE.CUDA`` is the CUDA support family, which cannot exist on a ROCm
    host, so it maps to ``DEVICE.ROCM`` there as well.
    """
    if isinstance(type_value, int):
        if HAS_CUDA:
            return DEVICE.ROCM if IS_ROCM else DEVICE.CUDA
        elif HAS_XPU:
            return DEVICE.XPU
        elif HAS_NPU:
            return DEVICE.NPU
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
        if type_value == DEVICE.CUDA and IS_ROCM:
            return DEVICE.ROCM
        return type_value

    if not isinstance(type_value, str):
        raise ValueError(f"Invalid device type_value type: {type(type_value)}")

    device = DEVICE(type_value.lower())
    if device == DEVICE.CUDA and IS_ROCM:
        # HIP exposes AMD devices through the ``cuda`` namespace.
        return DEVICE.ROCM
    return device


def get_best_device(backend: BACKEND = BACKEND.AUTO) -> torch.device:
    if HAS_CUDA:
        return CUDA_0
    elif HAS_XPU:
        return XPU_0
    elif HAS_NPU:
        return NPU_0
    elif HAS_MPS:
        return MPS
    else:
        return CPU

EXPERT_INDEX_PLACEHOLDER = "{expert_index}"
