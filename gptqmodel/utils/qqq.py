# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import torch

from .cpp import TorchOpsJitExtension, default_torch_ops_build_root


_QQQ_OPS_NAME = "gptqmodel_qqq_ops"
_QQQ_OPS_NAMESPACE = "gptqmodel_qqq"


def _qqq_sources() -> list[str]:
    root = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "qqq"
    return [
        str(root / "qqq.cpp"),
        str(root / "qqq_gemm.cu"),
    ]


def _qqq_include_paths() -> list[str]:
    root = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "qqq"
    return [str(root)]


def _qqq_cxx11_abi_flag() -> int:
    return int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 1))


def _qqq_extra_cflags() -> list[str]:
    abi = _qqq_cxx11_abi_flag()
    return [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
    ]


def _qqq_extra_cuda_cflags() -> list[str]:
    abi = _qqq_cxx11_abi_flag()
    return [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--threads",
        os.getenv("NVCC_THREADS", "2"),
        "--optimize=3",
        "-Xptxas",
        "-v,-O3,-dlcm=ca",
        "-lineinfo",
        "-Xfatbin",
        "-compress-all",
        "-diag-suppress=179,39,177",
    ]


_QQQ_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_QQQ_OPS_NAME,
    namespace=_QQQ_OPS_NAMESPACE,
    required_ops=("qqq_gemm",),
    sources=_qqq_sources,
    build_root_env="GPTQMODEL_QQQ_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("qqq"),
    display_name="QQQ",
    extra_cflags=_qqq_extra_cflags,
    extra_cuda_cflags=_qqq_extra_cuda_cflags,
    extra_include_paths=_qqq_include_paths,
    force_rebuild_env="GPTQMODEL_QQQ_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def clear_qqq_extension_cache() -> None:
    _QQQ_TORCH_OPS_EXTENSION.clear_cache()


def qqq_runtime_available() -> bool:
    return _QQQ_TORCH_OPS_EXTENSION.load()


def qqq_runtime_error() -> str:
    if _QQQ_TORCH_OPS_EXTENSION.load():
        return ""
    return _QQQ_TORCH_OPS_EXTENSION.last_error_message() or "QQQ CUDA runtime unavailable."


def prewarm_qqq_extension() -> bool:
    return qqq_runtime_available()


def qqq_gemm(
    A,
    B,
    C,
    D,
    s1,
    s2,
    s3,
    workspace,
    thread_k=-1,
    thread_n=-1,
    sms=-1,
    max_par=16,
):
    return _QQQ_TORCH_OPS_EXTENSION.op("qqq_gemm")(
        A,
        B,
        C,
        D,
        s1,
        s2,
        s3,
        workspace,
        thread_k,
        thread_n,
        sms,
        max_par,
    )


__all__ = [
    "clear_qqq_extension_cache",
    "prewarm_qqq_extension",
    "qqq_gemm",
    "qqq_runtime_available",
    "qqq_runtime_error",
]
