# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native CPTC extension dispatch with a backend-independent checkpoint layout."""

from pathlib import Path

import torch

from ..models._const import DEVICE, normalize_device
from .arch import cuda_capability, rocm_arch
from .cpp import (
    TorchOpsJitExtension,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
    jit_disable_fma_contraction_flag,
)
from .logger import setup_logger


log = setup_logger()
_RDNA4 = ("gfx1200", "gfx1201")


def _sources():
    root = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "cptc"
    return [str(root / name) for name in ("cptc_ops.cpp", "cptc_kernels.cu")]


def _device_flags():
    return default_jit_cuda_cflags() + [jit_disable_fma_contraction_flag()]


_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_cptc_ops",
    namespace="gptqmodel_cptc",
    required_ops=("decode", "mma", "encode"),
    sources=_sources,
    build_root_env="GPTQMODEL_CPTC_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("cptc"),
    display_name="CPTC (CUDA / ROCm RDNA4)",
    extra_cuda_cflags=_device_flags,
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)


def native_supported(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    family = normalize_device(device)
    if family == DEVICE.ROCM:
        return rocm_arch(device) in _RDNA4
    if family == DEVICE.CUDA:
        capability = cuda_capability(device)
        return capability is not None and capability >= (8, 0)
    return False


def host_supported() -> bool:
    return any(native_supported(torch.device("cuda", index)) for index in range(torch.cuda.device_count()))


def native_ops(device: torch.device):
    """Return the native ops for this ordinal, or use the portable codec."""
    if not native_supported(device):
        return None
    if not _EXTENSION.load():
        log.warn.once(f"CPTC native extension unavailable, using Torch: {_EXTENSION.last_error_message()}")
        return None
    return _EXTENSION.namespace_object()
