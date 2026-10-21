# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import torch

from .cpp import (
    TorchOpsJitExtension,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


_VPTQ_OPS_NAME = "gptqmodel_vptq_ops"
_VPTQ_OPS_NAMESPACE = "gptqmodel_vptq"


def _vptq_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "vptq"


def _vptq_sources() -> list[str]:
    root = _vptq_root()
    return [
        str(root / "vptq.cpp"),
        str(root / "vptq_kernel.cu"),
    ]


def _vptq_extra_cflags() -> list[str]:
    return default_jit_cflags(enable_bf16=True)


def _vptq_extra_cuda_cflags() -> list[str]:
    return default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_diag_suppress=True,
    )


_VPTQ_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_VPTQ_OPS_NAME,
    namespace=_VPTQ_OPS_NAMESPACE,
    required_ops=("vptq_dequantize", "vptq_linear"),
    sources=_vptq_sources,
    build_root_env="GPTQMODEL_VPTQ_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("vptq"),
    display_name="VPTQ CUDA",
    extra_cflags=_vptq_extra_cflags,
    extra_cuda_cflags=_vptq_extra_cuda_cflags,
    extra_include_paths=lambda: [str(_vptq_root())],
    force_rebuild_env="GPTQMODEL_VPTQ_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def clear_vptq_extension_cache() -> None:
    _VPTQ_TORCH_OPS_EXTENSION.clear_cache()


def vptq_cuda_runtime_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return _extension_api().is_available("vptq")


def vptq_cuda_runtime_error() -> str:
    if not torch.cuda.is_available():
        return "VPTQ CUDA requires an available CUDA device."
    extension_api = _extension_api()
    if extension_api.is_available("vptq"):
        return ""
    return extension_api.error("vptq") or "VPTQ CUDA runtime unavailable."


def prewarm_vptq_extension() -> bool:
    if not torch.cuda.is_available():
        return False
    return _extension_api().load(name="vptq")["vptq"]


def vptq_cuda_dequantize(
    indices,
    centroids,
    res_centroids,
    outlier_indices,
    outlier_centroids,
    inverse_perm,
    weight_scale,
    weight_bias,
    *,
    group_size: int,
    in_features: int,
    out_features: int,
    vector_len: int,
    outlier_size: int,
    outlier_vector_len: int,
    num_centroids: int,
    num_res_centroids: int,
    index_bits: int,
    res_bits: int,
    norm_dim: int,
):
    return _extension_api().op("vptq", "vptq_dequantize")(
        indices,
        centroids,
        res_centroids,
        outlier_indices,
        outlier_centroids,
        inverse_perm,
        weight_scale,
        weight_bias,
        group_size,
        in_features,
        out_features,
        vector_len,
        outlier_size,
        outlier_vector_len,
        num_centroids,
        num_res_centroids,
        index_bits,
        res_bits,
        norm_dim,
    )


def vptq_cuda_linear(
    x,
    indices,
    centroids,
    res_centroids,
    outlier_indices,
    outlier_centroids,
    inverse_perm,
    weight_scale,
    weight_bias,
    bias,
    *,
    group_size: int,
    in_features: int,
    out_features: int,
    vector_len: int,
    outlier_size: int,
    outlier_vector_len: int,
    num_centroids: int,
    num_res_centroids: int,
    index_bits: int,
    res_bits: int,
    norm_dim: int,
):
    return _extension_api().op("vptq", "vptq_linear")(
        x,
        indices,
        centroids,
        res_centroids,
        outlier_indices,
        outlier_centroids,
        inverse_perm,
        weight_scale,
        weight_bias,
        bias,
        group_size,
        in_features,
        out_features,
        vector_len,
        outlier_size,
        outlier_vector_len,
        num_centroids,
        num_res_centroids,
        index_bits,
        res_bits,
        norm_dim,
    )


__all__ = [
    "clear_vptq_extension_cache",
    "prewarm_vptq_extension",
    "vptq_cuda_dequantize",
    "vptq_cuda_linear",
    "vptq_cuda_runtime_available",
    "vptq_cuda_runtime_error",
]
