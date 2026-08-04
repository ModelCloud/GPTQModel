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

import os
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
_QWEIGHT_CPU_PACKED_CACHE = torch.utils.weak.WeakTensorKeyDictionary()
_QWEIGHT_CPU_PREEXPAND_CACHE = torch.utils.weak.WeakTensorKeyDictionary()


def _prepack_planar_to_cpu(qweight: torch.Tensor, bits: int) -> torch.Tensor:
    """Transpose planar qweight from [K/32*bits, N] to [N/16, K/32, bits, 16]."""
    num_k_blocks = qweight.size(0) // bits
    n = qweight.size(1)
    q = qweight.reshape(num_k_blocks, bits, n // 16, 16)
    return q.permute(2, 0, 1, 3).contiguous()


def _preexpand_qweight_uint8(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Decode packed planar qweight into uint8 codes and re-pack 4 codes per int32 word.

    Input shape:  [N/16, K/32, bits, 16] int32 (planar packed words).
    Output shape: [N/16, K/32, 8, 16] int32 (each word holds 4 consecutive uint8 codes).

    The conversion is done in K-block chunks so transient int32 intermediates
    (``selected``, ``codes``) stay bounded by the chunk size instead of the full
    layer size.
    """
    device = packed.device
    num_cb, num_kb, _, width16 = packed.shape
    # Per-bit-width plane descriptors: (word_width, start_word, bit_offset) for each plane.
    plane_specs = {
        3: [(2, 0, 0), (1, 2, 2)],
        5: [(4, 0, 0), (1, 4, 4)],
        6: [(4, 0, 0), (2, 4, 4)],
        7: [(4, 0, 0), (2, 4, 4), (1, 6, 6)],
    }
    specs = plane_specs[bits]
    out = torch.empty((num_cb, num_kb, 8, width16), dtype=torch.int32, device=device)
    k = torch.arange(32, device=device, dtype=torch.int32)
    # Process a few K-blocks at a time to keep peak transient memory low.
    kb_chunk = 4
    for kb_start in range(0, num_kb, kb_chunk):
        kb_end = min(num_kb, kb_start + kb_chunk)
        chunk = packed[:, kb_start:kb_end, :, :]
        # codes is int32 so bit shifts << 8/16/24 stay in-range for the final pack.
        codes = torch.zeros((num_cb, kb_end - kb_start, 32, width16), dtype=torch.int32, device=device)
        for w, start, off in specs:
            pack_factor = 32 // w
            word_idx = (start + k // pack_factor).long()
            shift = (w * (k % pack_factor)).view(1, 1, 32, 1)
            mask = (1 << w) - 1
            # selected shape: [num_cb, chunk_kb, 32, width16]
            selected = chunk[:, :, word_idx, :]
            part = ((selected >> shift) & mask) << off
            codes |= part
        c0 = codes[:, :, 0::4, :]
        c1 = codes[:, :, 1::4, :]
        c2 = codes[:, :, 2::4, :]
        c3 = codes[:, :, 3::4, :]
        out[:, kb_start:kb_end, :, :] = (c0 | (c1 << 8) | (c2 << 16) | (c3 << 24)).to(torch.int32)
    return out.contiguous()


def _should_preexpand_qweight(qweight: torch.Tensor, bits: int) -> bool:
    """Default the pre-expanded layout on only when the extra memory is small
    relative to available physical RAM.

    The pre-expanded layout needs ``8 / bits`` times as much memory as the packed
    planar qweight, so for very large 3-bit layers the resident memory growth
    can be substantial.  ``GPTQMODEL_PANGOLIN_CPU_PREEXPAND_QWEIGHT`` still
    overrides this heuristic.
    """
    try:
        page_size = os.sysconf("SC_PAGESIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        if page_size <= 0 or avail_pages <= 0:
            return True
    except (ValueError, OSError):
        return True
    extra_bytes = qweight.numel() * 4 * (8.0 / bits - 1.0)
    return extra_bytes <= page_size * avail_pages * 0.25


def _prepack_qweight_for_cpu(qweight: torch.Tensor, bits: int, preexpand: bool = True) -> torch.Tensor:
    """Transpose planar qweight for the CPU kernel.

    When `preexpand` is True, decode the packed planar words into a uint8
    pre-expanded layout ([N/16, K/32, 8, 16]) so the SIMD kernel loads one word
    per K value instead of `bits` words per K-block.  Cached weakly per qweight
    tensor, preexpand flag, and bit width.
    """
    expected_planes = 8 if preexpand else bits
    if qweight.dim() == 4:
        if qweight.size(2) == expected_planes:
            return qweight
    cache = _QWEIGHT_CPU_PREEXPAND_CACHE if preexpand else _QWEIGHT_CPU_PACKED_CACHE
    cached = cache.get(qweight)
    if cached is not None and cached.size(2) == expected_planes:
        return cached
    packed = _prepack_planar_to_cpu(qweight, bits)
    if preexpand:
        packed = _preexpand_qweight_uint8(packed, bits)
    cache[qweight] = packed
    return packed


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
    preexpand_env = os.environ.get("GPTQMODEL_PANGOLIN_CPU_PREEXPAND_QWEIGHT")
    if preexpand_env is not None:
        preexpand = preexpand_env.lower() in ("1", "true", "on")
    else:
        preexpand = _should_preexpand_qweight(qweight, bits)
    qweight_packed = _prepack_qweight_for_cpu(qweight, bits, preexpand=preexpand)
    return _gemv_cpu_op()(x, qweight_packed, scales, qzeros, g_idx, bits).to(x.dtype)
