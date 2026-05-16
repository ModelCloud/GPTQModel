# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import torch

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
    is_nvcc_compatible,
)


_VECQUANT3_OPS_NAME = "gptqmodel_vecquant3_ops"
_VECQUANT3_NAMESPACE = "gptqmodel_vecquant3"
_VECQUANT3_REQUIRED_CUDA_HEADERS = ("cuda_runtime_api.h",)


def _vecquant3_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "vecquant3"


def _vecquant3_sources() -> list[str]:
    root = _vecquant3_root()
    return [
        str(root / "vecquant3.cpp"),
        str(root / "vecquant3_kernel.cu"),
    ]


def _vecquant3_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_vecquant3_root())],
        required_header_names=_VECQUANT3_REQUIRED_CUDA_HEADERS,
    )


def _vecquant3_extra_cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags(
        enable_bf16=False,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_ptxas_verbosity=False,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )
    if is_nvcc_compatible():
        flags.insert(0, "-static-global-template-stub=false")
    return flags


_VECQUANT3_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_VECQUANT3_OPS_NAME,
    namespace=_VECQUANT3_NAMESPACE,
    required_ops=("gemv", "gemv_lora"),
    sources=_vecquant3_sources,
    build_root_env="GPTQMODEL_VECQUANT3_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("vecquant3"),
    display_name="VecQuant3 GPTQ grouped GEMV",
    extra_cflags=lambda: default_jit_cflags(enable_bf16=False),
    extra_cuda_cflags=_vecquant3_extra_cuda_cflags,
    extra_include_paths=_vecquant3_include_paths,
    force_rebuild_env="GPTQMODEL_VECQUANT3_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def vecquant3_supported() -> bool:
    return torch.cuda.is_available()


def vecquant3_runtime_error() -> str:
    if not torch.cuda.is_available():
        return "VecQuant3 GPTQ grouped GEMV requires CUDA."
    return _VECQUANT3_TORCH_OPS_EXTENSION.last_error_message()


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def vecquant3_runtime_available() -> bool:
    if not vecquant3_supported():
        return False
    return _extension_api().is_available("vecquant3")


def gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    ops = _extension_api().namespace(name="vecquant3")
    return ops.gemv(x, qweight, scales, qzeros, int(group_size))


def gemv_lora(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    ops = _extension_api().namespace(name="vecquant3")
    return ops.gemv_lora(x, qweight, scales, qzeros, down, up, int(group_size))


__all__ = [
    "_VECQUANT3_TORCH_OPS_EXTENSION",
    "gemv",
    "gemv_lora",
    "vecquant3_runtime_available",
    "vecquant3_runtime_error",
    "vecquant3_supported",
]
