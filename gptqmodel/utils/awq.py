# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


_AWQ_OPS_NAME = "gptqmodel_awq_ops"
_AWQ_OPS_NAMESPACE = "gptqmodel_awq"


def _awq_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "awq"


def _awq_sources() -> list[str]:
    root = _awq_root()
    return [
        str(root / "torch_bind.cpp"),
        str(root / "quantization" / "gemm_cuda_gen.cu"),
        str(root / "quantization" / "gemv_cuda.cu"),
        str(root / "gemm_fast_cuda_entry.cu"),
        str(root / "gemv_fast_cuda_entry.cu"),
    ]


def _awq_required_cuda_headers() -> tuple[str, ...]:
    return ("cusparse.h",)


def _awq_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_awq_root())],
        required_header_names=_awq_required_cuda_headers(),
    )


def _awq_extra_cflags() -> list[str]:
    return default_jit_cflags(enable_bf16=True)


def _awq_extra_cuda_cflags() -> list[str]:
    return default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_ptxas_verbosity=False,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )


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


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def clear_awq_extension_cache() -> None:
    _AWQ_TORCH_OPS_EXTENSION.clear_cache()


def awq_runtime_available() -> bool:
    return _extension_api().is_available("awq")


def awq_runtime_error() -> str:
    extension_api = _extension_api()
    if extension_api.is_available("awq"):
        return ""
    return extension_api.error("awq") or "CUDA AWQ runtime unavailable."


def prewarm_awq_extension() -> bool:
    return _extension_api().load(name="awq")["awq"]


def _awq_runtime_namespace():
    return _extension_api().namespace("awq")


def awq_gemm_forward(input, qweight, scales, qzeros, split_k_iters, fp32_accum: bool):
    return _extension_api().op("awq", "gemm_forward")(input, qweight, scales, qzeros, split_k_iters, fp32_accum)


def awq_dequantize_weights(qweight, scales, qzeros, split_k_iters, thx, thy, dbg):
    return _extension_api().op("awq", "dequantize_weights")(qweight, scales, qzeros, split_k_iters, thx, thy, dbg)


def awq_gemmv2_forward(input, qweight, scales, qzeros, group_size, split_k_iters):
    return _extension_api().op("awq", "gemmv2_forward")(input, qweight, scales, qzeros, group_size, split_k_iters)


def awq_gemv_forward(input, qweight, scales, qzeros, group_size):
    return _extension_api().op("awq", "gemv_forward")(input, qweight, scales, qzeros, group_size)


def awq_fast_gemm_forward_prefill(input, qweight, scales, qzeros):
    return _extension_api().op("awq", "gemm_fast_forward_prefill")(input, qweight, scales, qzeros)


def awq_fast_gemv_forward_decode(input, qweight, scales, qzeros, m, n, k, group_size):
    return _extension_api().op("awq", "gemv_fast_forward_decode")(input, qweight, scales, qzeros, m, n, k, group_size)


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
