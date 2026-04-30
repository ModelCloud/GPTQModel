# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

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
    is_nvcc_compatible,
)
from .rocm import IS_ROCM


_GPTQ_GEMM_REQUIRED_CUDA_HEADERS = (
    "cuda_runtime_api.h",
    "cuda_fp16.h",
)


def _gptq_gemm_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "gptq_gemm"


def _gptq_gemm_sources() -> list[str]:
    root = _gptq_gemm_root()
    return [
        str(root / "gptq_gemm_torch.cpp"),
        str(root / "gptq_gemm_kernel.cu"),
    ]


def _gptq_gemm_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_gptq_gemm_root())],
        required_header_names=_GPTQ_GEMM_REQUIRED_CUDA_HEADERS,
    )


def _gptq_gemm_extra_cflags() -> list[str]:
    return default_jit_cflags(opt_level="O3")


def _gptq_gemm_extra_cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags(
        opt_level="O3",
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


def _validate_gptq_gemm_device_support() -> bool:
    return (
        torch.cuda.is_available()
        and not IS_ROCM
        and all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(torch.cuda.device_count()))
    )


def gptq_gemm_runtime_error() -> str:
    if IS_ROCM:
        return "GPTQ-GEMM kernel is not supported on ROCm."
    if not torch.cuda.is_available():
        return "GPTQ-GEMM kernel requires CUDA."
    unsupported = [
        f"{major}.{minor}"
        for major, minor in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))
        if major < 8
    ]
    if unsupported:
        return "GPTQ-GEMM kernel requires compute capability >= 8.0; found " + ", ".join(unsupported) + "."

    extension_api = _extension_api()
    if extension_api.is_available("gptq_gemm"):
        return ""
    return extension_api.error("gptq_gemm") or "GPTQ-GEMM runtime unavailable."


_GPTQ_GEMM_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_gptq_gemm_ops",
    namespace="gptqmodel_gptq_gemm",
    required_ops=("gptq_gemm",),
    sources=_gptq_gemm_sources,
    build_root_env="GPTQMODEL_GPTQ_GEMM_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("gptq_gemm"),
    display_name="GPTQ-GEMM",
    extra_cflags=_gptq_gemm_extra_cflags,
    extra_cuda_cflags=_gptq_gemm_extra_cuda_cflags,
    extra_include_paths=_gptq_gemm_include_paths,
    force_rebuild_env="GPTQMODEL_GPTQ_GEMM_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def clear_gptq_gemm_extension_cache() -> None:
    _GPTQ_GEMM_TORCH_OPS_EXTENSION.clear_cache()


def gptq_gemm_runtime_available() -> bool:
    return _extension_api().is_available("gptq_gemm")


def prewarm_gptq_gemm_extension() -> bool:
    return _extension_api().load(name="gptq_gemm")["gptq_gemm"]


def gptq_gemm_qweight_to_b_packed(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.dtype != torch.int32:
        raise ValueError(f"Expected int32 qweight tensor, got `{qweight.dtype}`.")
    if qweight.dim() != 2:
        raise ValueError(f"Expected 2D qweight tensor, got shape `{tuple(qweight.shape)}`.")

    qweight = qweight.contiguous()
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=qweight.dtype).view(1, 8, 1)
    unpacked = torch.bitwise_and(torch.bitwise_right_shift(qweight.unsqueeze(1), shifts), 0xF).to(torch.uint8)
    unpacked = unpacked.reshape(-1, qweight.shape[1])
    return (unpacked[0::2] | (unpacked[1::2] << 4)).contiguous()


def apply_gptq_gemm_linear(
    input: torch.Tensor,
    b_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    return _extension_api().op("gptq_gemm", "gptq_gemm")(input, b_packed, scales, int(group_size))


__all__ = [
    "_GPTQ_GEMM_TORCH_OPS_EXTENSION",
    "_validate_gptq_gemm_device_support",
    "apply_gptq_gemm_linear",
    "clear_gptq_gemm_extension_cache",
    "gptq_gemm_qweight_to_b_packed",
    "gptq_gemm_runtime_available",
    "gptq_gemm_runtime_error",
    "prewarm_gptq_gemm_extension",
]
