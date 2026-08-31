# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

# JIT wrapper around the vendored fast-hadamard-transform kernel from
# https://github.com/Dao-AILab/fast-hadamard-transform (Tri Dao, BSD-3-Clause).

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)

_HADAMARD_OPS_NAME = "gptqmodel_hadamard_ops"
_HADAMARD_NAMESPACE = "gptqmodel_hadamard"


def _hadamard_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "hadamard"


def _hadamard_sources() -> list[str]:
    root = _hadamard_root()
    return [
        str(root / "hadamard.cpp"),
        str(root / "fast_hadamard_transform_cuda.cu"),
    ]


def _hadamard_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback([str(_hadamard_root())])


def _hadamard_extra_cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=False,
        include_ptxas_verbosity=False,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )
    return flags


_HADAMARD_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_HADAMARD_OPS_NAME,
    namespace=_HADAMARD_NAMESPACE,
    required_ops=(
        "fast_hadamard_transform",
        "fast_hadamard_transform_12N",
        "fast_hadamard_transform_20N",
        "fast_hadamard_transform_28N",
        "fast_hadamard_transform_40N",
    ),
    sources=_hadamard_sources,
    build_root_env="GPTQMODEL_HADAMARD_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("hadamard"),
    display_name="Fast Hadamard transform",
    extra_cflags=lambda: default_jit_cflags(enable_bf16=True),
    extra_cuda_cflags=_hadamard_extra_cuda_cflags,
    extra_include_paths=_hadamard_include_paths,
    force_rebuild_env="GPTQMODEL_HADAMARD_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)


def hadamard_supported() -> bool:
    return torch.cuda.is_available()


def hadamard_runtime_error() -> str:
    if not torch.cuda.is_available():
        return "Fast Hadamard transform requires CUDA."
    return _HADAMARD_TORCH_OPS_EXTENSION.last_error_message()


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def hadamard_available() -> bool:
    if not hadamard_supported():
        return False
    return _extension_api().is_available("hadamard")


def hadamard_op(op_name: str = "fast_hadamard_transform"):
    return _extension_api().op("hadamard", op_name)


def hadamard_transform(x: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """Apply the fast Hadamard transform to the last dimension of `x`.

    Args:
        x: Tensor of shape (..., dim) on CUDA with dtype float16, bfloat16, or float32.
        scale: Scalar multiplier applied to the result. Defaults to 1.0.

    Returns:
        Tensor of the same shape and dtype as `x`.
    """
    if scale is None:
        scale = 1.0
    op = hadamard_op("fast_hadamard_transform")
    return op(x, float(scale))
