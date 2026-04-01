# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import torch

from .cpp import TorchOpsJitExtension, default_torch_ops_build_root


_AWQ_OPS_NAME = "gptqmodel_awq_ops"
_AWQ_OPS_NAMESPACE = "gptqmodel_awq"


def _awq_sources() -> list[str]:
    root = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "awq"
    return [
        str(root / "torch_bind.cpp"),
        str(root / "quantization" / "gemm_cuda_gen.cu"),
        str(root / "quantization" / "gemv_cuda.cu"),
        str(root / "gemm_fast_cuda_entry.cu"),
        str(root / "gemv_fast_cuda_entry.cu"),
    ]


def _awq_include_paths() -> list[str]:
    root = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "awq"
    return [str(root)]


def _awq_cxx11_abi_flag() -> int:
    return int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 1))


def _awq_extra_cflags() -> list[str]:
    abi = _awq_cxx11_abi_flag()
    return [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
    ]


def _awq_extra_cuda_cflags() -> list[str]:
    abi = _awq_cxx11_abi_flag()
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


# Shared singleton so every AWQ/ParoQuant caller uses the same torch.ops JIT
# cache, force-rebuild controls, and user-facing compile spinner.
_AWQ_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_AWQ_OPS_NAME,
    namespace=_AWQ_OPS_NAMESPACE,
    required_ops=(
        "gemm_forward",
        "gemm_forward_fp32_reduce",
        "gemmv2_forward",
        "gemv_forward",
        "gemm_fast_forward_prefill",
        "gemv_fast_forward_decode",
        "dequantize_weights",
    ),
    sources=_awq_sources,
    build_root_env="GPTQMODEL_AWQ_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("awq"),
    display_name="AWQ",
    extra_cflags=_awq_extra_cflags,
    extra_cuda_cflags=_awq_extra_cuda_cflags,
    extra_include_paths=_awq_include_paths,
    force_rebuild_env="GPTQMODEL_AWQ_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def clear_awq_extension_cache() -> None:
    _AWQ_TORCH_OPS_EXTENSION.clear_cache()


def awq_runtime_available() -> bool:
    return _AWQ_TORCH_OPS_EXTENSION.load()


def awq_runtime_error() -> str:
    if _AWQ_TORCH_OPS_EXTENSION.load():
        return ""
    return _AWQ_TORCH_OPS_EXTENSION.last_error_message() or "CUDA AWQ runtime unavailable."


def prewarm_awq_extension() -> bool:
    return awq_runtime_available()


def _awq_runtime_namespace():
    return _AWQ_TORCH_OPS_EXTENSION.namespace_object()


def awq_gemm_forward(input, qweight, scales, qzeros, split_k_iters, fp32_accum: bool):
    return _AWQ_TORCH_OPS_EXTENSION.op("gemm_forward")(input, qweight, scales, qzeros, split_k_iters, fp32_accum)


def awq_dequantize_weights(qweight, scales, qzeros, split_k_iters, thx, thy, dbg):
    return _AWQ_TORCH_OPS_EXTENSION.op("dequantize_weights")(qweight, scales, qzeros, split_k_iters, thx, thy, dbg)


def awq_gemmv2_forward(input, qweight, scales, qzeros, group_size, split_k_iters):
    return _AWQ_TORCH_OPS_EXTENSION.op("gemmv2_forward")(input, qweight, scales, qzeros, group_size, split_k_iters)


def awq_gemv_forward(input, qweight, scales, qzeros, group_size):
    return _AWQ_TORCH_OPS_EXTENSION.op("gemv_forward")(input, qweight, scales, qzeros, group_size)


def awq_fast_gemm_forward_prefill(input, qweight, scales, qzeros):
    return _AWQ_TORCH_OPS_EXTENSION.op("gemm_fast_forward_prefill")(input, qweight, scales, qzeros)


def awq_fast_gemv_forward_decode(input, qweight, scales, qzeros, m, n, k, group_size):
    return _AWQ_TORCH_OPS_EXTENSION.op("gemv_fast_forward_decode")(input, qweight, scales, qzeros, m, n, k, group_size)


__all__ = [
    "awq_dequantize_weights",
    "awq_fast_gemm_forward_prefill",
    "awq_fast_gemv_forward_decode",
    "awq_gemm_forward",
    "awq_gemmv2_forward",
    "awq_gemv_forward",
    "awq_runtime_available",
    "awq_runtime_error",
    "clear_awq_extension_cache",
    "prewarm_awq_extension",
]
