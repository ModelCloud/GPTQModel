# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Native CUDA decode-regime GEMV for the planar (gptq_p) format at 3/5/6/7 bits.

Register-level decode: each 32-code block loads its `bits` packed words once
and derives all 32 codes with compile-time shifts/masks, so weight-side DRAM
traffic is only the packed words (no dense fp16 round-trip). Requires every
32-row block of `g_idx` to map to a single group; callers fall back to the
Triton paths otherwise.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import torch
import torch.utils.weak

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


_PANGOLIN_OPS_NAME = "gptqmodel_pangolin_ops"
_PANGOLIN_NAMESPACE = "gptqmodel_pangolin"
_PANGOLIN_REQUIRED_CUDA_HEADERS = ("cuda_runtime_api.h",)

PANGOLIN_BITS = (3, 5, 6, 7)
PANGOLIN_MAX_M = 32
PANGOLIN_SUPPORTED_M = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32)


def _pangolin_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "planar"


def _pangolin_sources() -> list[str]:
    root = _pangolin_root()
    return [
        str(root / "planar_gemv.cpp"),
        str(root / "planar_gemv_kernel.cu"),
    ]


def _pangolin_cpu_sources() -> list[str]:
    return [str(_pangolin_root() / "planar_gemv_cpu.cpp")]


def _pangolin_cpu_supported() -> bool:
    import platform

    return platform.machine().lower() in ("x86_64", "amd64")


def _pangolin_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_pangolin_root())],
        required_header_names=_PANGOLIN_REQUIRED_CUDA_HEADERS,
    )


_PANGOLIN_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_PANGOLIN_OPS_NAME,
    namespace=_PANGOLIN_NAMESPACE,
    required_ops=("gemv",),
    sources=_pangolin_sources,
    build_root_env="GPTQMODEL_PANGOLIN_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("pangolin"),
    display_name="Pangolin planar gptq_p GEMV",
    extra_cflags=lambda: default_jit_cflags(enable_bf16=True),
    extra_cuda_cflags=lambda: default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_ptxas_verbosity=False,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    ),
    extra_include_paths=_pangolin_include_paths,
    force_rebuild_env="GPTQMODEL_PANGOLIN_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)

def _pangolin_cpu_extra_cflags() -> list[str]:
    import platform

    flags = list(default_jit_cflags(enable_bf16=True))
    if platform.system() == "Linux":
        flags.append("-fopenmp")
    return flags


def _pangolin_cpu_extra_ldflags() -> list[str]:
    import platform

    return ["-fopenmp"] if platform.system() == "Linux" else []


_PANGOLIN_CPU_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_pangolin_cpu_ops",
    namespace=_PANGOLIN_NAMESPACE,
    required_ops=("gemv_cpu",),
    sources=_pangolin_cpu_sources,
    build_root_env="GPTQMODEL_PANGOLIN_CPU_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("pangolin_cpu"),
    display_name="Pangolin planar gptq_p GEMV CPU",
    extra_cflags=_pangolin_cpu_extra_cflags,
    extra_ldflags=_pangolin_cpu_extra_ldflags,
    extra_include_paths=lambda: [str(_pangolin_root())],
    force_rebuild_env="GPTQMODEL_PANGOLIN_CPU_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=False,
)


def _sm80_or_newer_device_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return any(
            torch.cuda.get_device_capability(index) >= (8, 0)
            for index in range(torch.cuda.device_count())
        )
    except (RuntimeError, AssertionError):
        return False


def pangolin_supported() -> bool:
    return _sm80_or_newer_device_available()


def pangolin_cpu_supported() -> bool:
    return _pangolin_cpu_supported()


def pangolin_runtime_error() -> str:
    if not torch.cuda.is_available():
        return "Pangolin requires CUDA."
    if not _sm80_or_newer_device_available():
        return "Pangolin requires a CUDA compute capability >= 8.0 device."
    return _PANGOLIN_TORCH_OPS_EXTENSION.last_error_message()


def pangolin_cpu_runtime_error() -> str:
    if not _pangolin_cpu_supported():
        return "Pangolin CPU kernel requires x86-64 (AMD64)."
    return _PANGOLIN_CPU_TORCH_OPS_EXTENSION.last_error_message()


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def pangolin_runtime_available() -> bool:
    if not pangolin_supported():
        return False
    return _extension_api().is_available("pangolin")


def pangolin_cpu_runtime_available() -> bool:
    if not pangolin_cpu_supported():
        return False
    return _extension_api().is_available("pangolin_cpu")


_PANGOLIN_RUNTIME_AVAILABLE: bool | None = None
_PANGOLIN_CPU_RUNTIME_AVAILABLE: bool | None = None
_PANGOLIN_GEMV_OP: Callable | None = None
_PANGOLIN_CPU_GEMV_OP: Callable | None = None
_PANGOLIN_INIT_LOCK = threading.Lock()


def ensure_pangolin_runtime_available() -> bool:
    """Cache the (expensive) JIT availability check so hot paths pay it once."""
    global _PANGOLIN_RUNTIME_AVAILABLE
    if _PANGOLIN_RUNTIME_AVAILABLE is None:
        with _PANGOLIN_INIT_LOCK:
            if _PANGOLIN_RUNTIME_AVAILABLE is None:
                _PANGOLIN_RUNTIME_AVAILABLE = pangolin_runtime_available()
    return _PANGOLIN_RUNTIME_AVAILABLE


def ensure_pangolin_cpu_runtime_available() -> bool:
    """Cache the CPU JIT availability check."""
    global _PANGOLIN_CPU_RUNTIME_AVAILABLE
    if _PANGOLIN_CPU_RUNTIME_AVAILABLE is None:
        with _PANGOLIN_INIT_LOCK:
            if _PANGOLIN_CPU_RUNTIME_AVAILABLE is None:
                _PANGOLIN_CPU_RUNTIME_AVAILABLE = pangolin_cpu_runtime_available()
    return _PANGOLIN_CPU_RUNTIME_AVAILABLE


def _gemv_op() -> Callable:
    global _PANGOLIN_GEMV_OP
    if _PANGOLIN_GEMV_OP is None:
        with _PANGOLIN_INIT_LOCK:
            if _PANGOLIN_GEMV_OP is None:
                _PANGOLIN_GEMV_OP = _extension_api().op("pangolin", "gemv")
    return _PANGOLIN_GEMV_OP


def _gemv_cpu_op() -> Callable:
    global _PANGOLIN_CPU_GEMV_OP
    if _PANGOLIN_CPU_GEMV_OP is None:
        with _PANGOLIN_INIT_LOCK:
            if _PANGOLIN_CPU_GEMV_OP is None:
                _PANGOLIN_CPU_GEMV_OP = _extension_api().op("pangolin_cpu", "gemv_cpu")
    return _PANGOLIN_CPU_GEMV_OP


_G_IDX_BLOCK_UNIFORM_CACHE = torch.utils.weak.WeakTensorKeyDictionary()


def _g_idx_block_uniform(g_idx: torch.Tensor) -> bool:
    """True when every 32-row block maps to a single group (cached per g_idx tensor)."""
    if g_idx.numel() % 32 != 0:
        return False
    cached = _G_IDX_BLOCK_UNIFORM_CACHE.get(g_idx)
    if cached is None:
        blocks = g_idx.reshape(-1, 32)
        cached = bool((blocks == blocks[:, :1]).all().item())
        _G_IDX_BLOCK_UNIFORM_CACHE[g_idx] = cached
    return cached


def pangolin_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Run the native planar GEMV for a 2D `x[M, K]` with M <= PANGOLIN_MAX_M.

    The CUDA path accepts both FP16 and BF16 activations. The CPU path currently
    requires BF16 input and returns BF16 output (with FP32 accumulation).
    """
    if not x.is_contiguous():
        x = x.contiguous()
    if x.is_cuda:
        return _gemv_op()(x, qweight, scales, qzeros, g_idx, bits)
    if not ensure_pangolin_cpu_runtime_available():
        raise RuntimeError(pangolin_cpu_runtime_error())
    K = x.size(1)
    if K % 32 != 0 or g_idx.numel() % 32 != 0:
        raise RuntimeError(
            f"Pangolin CPU kernel requires K and g_idx length divisible by 32, got K={K}"
        )
    if not _g_idx_block_uniform(g_idx):
        raise RuntimeError("Pangolin CPU kernel requires g_idx to be uniform across 32-row blocks")
    return _gemv_cpu_op()(x, qweight, scales, qzeros, g_idx, bits).to(x.dtype)
