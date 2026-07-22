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


_TRILIN_OPS_NAME = "gptqmodel_trilin_ops"
_TRILIN_NAMESPACE = "gptqmodel_trilin"
_TRILIN_REQUIRED_CUDA_HEADERS = ("cuda_bf16.h", "cuda_runtime_api.h", "mma.h")


def _trilin_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "trilin"


def _trilin_sources() -> list[str]:
    return [str(_trilin_root() / "trilin_3bit_wmma.cu")]


def _trilin_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_trilin_root())],
        required_header_names=_TRILIN_REQUIRED_CUDA_HEADERS,
    )


def _trilin_extra_cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=False,
        include_ptxas_verbosity=False,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )
    flags.append("--use_fast_math")
    if is_nvcc_compatible():
        flags.insert(0, "-static-global-template-stub=false")
    return flags


_TRILIN_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_TRILIN_OPS_NAME,
    namespace=_TRILIN_NAMESPACE,
    required_ops=("matmul", "matmul_eora", "qkv", "silu_mul"),
    sources=_trilin_sources,
    build_root_env="GPTQMODEL_TRILIN_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("trilin"),
    display_name="Trilin native 3-bit CUDA",
    extra_cflags=lambda: default_jit_cflags(enable_bf16=True),
    extra_cuda_cflags=_trilin_extra_cuda_cflags,
    extra_include_paths=_trilin_include_paths,
    force_rebuild_env="GPTQMODEL_TRILIN_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def trilin_runtime_available() -> bool:
    return torch.cuda.is_available() and _extension_api().is_available("trilin")


def trilin_runtime_error() -> str:
    if not torch.cuda.is_available():
        return "Trilin native 3-bit CUDA requires CUDA."
    extension_api = _extension_api()
    if extension_api.is_available("trilin"):
        return ""
    return extension_api.error("trilin") or "Trilin native 3-bit CUDA runtime unavailable."


def prewarm_trilin_extension() -> bool:
    return _extension_api().load(name="trilin")["trilin"]


def select_trilin_split_k(m: int, k: int) -> int:
    """Select the measured small-M split while keeping at least 128 K values per slice."""
    if m <= 0 or k <= 0 or k % 128 != 0:
        raise ValueError(f"Trilin split selection requires M>0 and K divisible by 128, got M={m}, K={k}")
    if m > 16:
        return 1
    split = 32
    while split > 1 and (k % (split * 32) != 0 or k // split < 128):
        split //= 2
    return split


def trilin_matmul(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    split_k = select_trilin_split_k(input.shape[0], input.shape[1])
    return _extension_api().op("trilin", "matmul")(input, qweight, scales, bias, split_k, group_size)


def trilin_matmul_eora(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    workspace: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse exact-shape TriLin decode with a supported EoRA rank using caller-exclusive scratch."""
    return _extension_api().op("trilin", "matmul_eora")(
        input,
        qweight,
        scales,
        bias,
        lora_a,
        lora_b,
        workspace,
    )


def trilin_silu_mul(
    input: torch.Tensor,
    gate_qweight: torch.Tensor,
    gate_scales: torch.Tensor,
    up_qweight: torch.Tensor,
    up_scales: torch.Tensor,
) -> torch.Tensor:
    """Fuse two exact-shape native 3-bit projections with SwiGLU for one decode row."""
    return _extension_api().op("trilin", "silu_mul")(
        input,
        gate_qweight,
        gate_scales,
        up_qweight,
        up_scales,
    )


def trilin_qkv(
    input: torch.Tensor,
    q_qweight: torch.Tensor,
    q_scales: torch.Tensor,
    k_qweight: torch.Tensor,
    k_scales: torch.Tensor,
    v_qweight: torch.Tensor,
    v_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run exact-shape native 3-bit Q/K/V projections in one launch and return zero-copy views."""
    combined = _extension_api().op("trilin", "qkv")(
        input,
        q_qweight,
        q_scales,
        k_qweight,
        k_scales,
        v_qweight,
        v_scales,
    )
    kv_size = k_scales.shape[1]
    return torch.split(combined, (4096, kv_size, kv_size), dim=-1)


__all__ = [
    "prewarm_trilin_extension",
    "select_trilin_split_k",
    "trilin_matmul",
    "trilin_matmul_eora",
    "trilin_qkv",
    "trilin_runtime_available",
    "trilin_runtime_error",
    "trilin_silu_mul",
]
