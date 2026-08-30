# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Hopper-only CuTe RS-WGMMA prototype for the QVQ local-ring kernel."""

from __future__ import annotations

from pathlib import Path

import torch

from .cpp import (
    TorchOpsJitExtension,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
    is_nvcc_compatible,
)
from .machete import _ensure_cutlass_source


_QVQ_WGMMA_NAME = "gptqmodel_qvq_wgmma_ops"
_QVQ_WGMMA_NAMESPACE = "gptqmodel_qvq_wgmma"
_SM90A_FLAGS = (
    "-gencode=arch=compute_90a,code=sm_90a",
    "-gencode=arch=compute_90a,code=compute_90a",
)
_TORCH_NVCC_UNDEFINES = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source() -> list[str]:
    return [str(_project_root() / "gptqmodel_ext" / "qvq" / "qvq_wgmma_cuda.cu")]


def _include_paths() -> list[str]:
    cutlass_root = _ensure_cutlass_source()
    return [str((cutlass_root / "include").resolve())]


def _cuda_flags() -> list[str]:
    flags = [
        *_TORCH_NVCC_UNDEFINES,
        *default_jit_cuda_cflags(
            enable_bf16=True,
            include_lineinfo=True,
            include_nvcc_threads=True,
            nvcc_threads="2",
            include_split_compile=True,
            include_ptxas_optimizations=True,
            include_ptxas_verbosity=False,
            include_fatbin_compression=True,
            include_diag_suppress=True,
        ),
        *_SM90A_FLAGS,
    ]
    if is_nvcc_compatible():
        flags.insert(0, "-static-global-template-stub=false")
    return flags


_QVQ_WGMMA_EXTENSION = TorchOpsJitExtension(
    name=_QVQ_WGMMA_NAME,
    namespace=_QVQ_WGMMA_NAMESPACE,
    required_ops=("w3_m16", "w3_m16_tma", "p32_window_w3_m16", "p32_window_w3_m16_tma"),
    sources=_source,
    build_root_env="GPTQMODEL_QVQ_WGMMA_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("qvq_wgmma"),
    display_name="QVQ Hopper RS-WGMMA prototype",
    extra_cflags=lambda: default_jit_cflags(enable_bf16=True),
    extra_cuda_cflags=_cuda_flags,
    extra_include_paths=_include_paths,
    extra_ldflags=("-lcuda",),
    force_rebuild_env="GPTQMODEL_QVQ_WGMMA_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    merge_visible_cuda_arch_override=False,
)


def qvq_wgmma_w3_m16(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    *,
    out_features: int,
    bank_alt_id: int = 3,
    split_count: int = 1,
) -> torch.Tensor:
    return _QVQ_WGMMA_EXTENSION.op("w3_m16")(
        input,
        trellis,
        levels,
        bank_ids,
        out_features,
        bank_alt_id,
        split_count,
    )


def qvq_wgmma_w3_m16_tma(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    *,
    out_features: int,
    bank_alt_id: int = 3,
    split_count: int = 1,
) -> torch.Tensor:
    return _QVQ_WGMMA_EXTENSION.op("w3_m16_tma")(
        input,
        trellis,
        levels,
        bank_ids,
        out_features,
        bank_alt_id,
        split_count,
    )


def qvq_p32_window_wgmma_w3_m16(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    *,
    out_features: int,
    bank_alt_id: int = 3,
    split_count: int = 1,
) -> torch.Tensor:
    """Run the direct-window standard-P32 W3 RS-WGMMA prototype."""

    return _QVQ_WGMMA_EXTENSION.op("p32_window_w3_m16")(
        input,
        trellis,
        levels,
        bank_ids,
        out_features,
        bank_alt_id,
        split_count,
    )


def qvq_p32_window_wgmma_w3_m16_tma(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    *,
    out_features: int,
    bank_alt_id: int = 3,
    split_count: int = 1,
) -> torch.Tensor:
    """Run the two-stage TMA direct-window P32 W3 RS-WGMMA prototype."""

    return _QVQ_WGMMA_EXTENSION.op("p32_window_w3_m16_tma")(
        input,
        trellis,
        levels,
        bank_ids,
        out_features,
        bank_alt_id,
        split_count,
    )


__all__ = [
    "qvq_p32_window_wgmma_w3_m16",
    "qvq_p32_window_wgmma_w3_m16_tma",
    "qvq_wgmma_w3_m16",
    "qvq_wgmma_w3_m16_tma",
]
