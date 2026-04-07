# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from .cpp import (
    TorchOpsJitExtension,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


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


def _qqq_extra_cflags() -> list[str]:
    return default_jit_cflags(enable_bf16=True)


def _qqq_extra_cuda_cflags() -> list[str]:
    return default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_diag_suppress=True,
    )


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


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def clear_qqq_extension_cache() -> None:
    _QQQ_TORCH_OPS_EXTENSION.clear_cache()


def qqq_runtime_available() -> bool:
    return _extension_api().is_available("qqq")


def qqq_runtime_error() -> str:
    extension_api = _extension_api()
    if extension_api.is_available("qqq"):
        return ""
    return extension_api.error("qqq") or "QQQ CUDA runtime unavailable."


def prewarm_qqq_extension() -> bool:
    return _extension_api().load(name="qqq")["qqq"]


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
    return _extension_api().op("qqq", "qqq_gemm")(
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
