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

import functools
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
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "pangolin"


def _pangolin_sources() -> list[str]:
    root = _pangolin_root()
    return [
        str(root / "pangolin_gemv.cpp"),
        str(root / "pangolin_gemv_kernel.cu"),
    ]


def _pangolin_cpu_sources() -> list[str]:
    return [str(_pangolin_root() / "pangolin_gemv_cpu.cpp")]


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
    python_abi_dependent=False,
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
    python_abi_dependent=False,
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
_QWEIGHT_CPU_VNNI_CACHE = torch.utils.weak.WeakTensorKeyDictionary()


def _prepack_planar_to_cpu(qweight: torch.Tensor, bits: int) -> torch.Tensor:
    """Transpose planar qweight from [K/32*bits, N] to [N/16, K/32, bits, 16]."""
    num_k_blocks = qweight.size(0) // bits
    n = qweight.size(1)
    q = qweight.reshape(num_k_blocks, bits, n // 16, 16)
    return q.permute(2, 0, 1, 3).contiguous()


@functools.lru_cache(maxsize=None)
def _cpu_has_avx512_vnni() -> bool:
    """Runtime probe for AVX-512 VNNI on Linux hosts."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = line.split()
                    return "avx512vnni" in flags or "avx512_vnni" in flags
    except (OSError, ValueError):
        pass
    return False


def _decode_zero_codes(qzeros: torch.Tensor, bits: int) -> torch.Tensor:
    """Decode packed planar qzeros to per-column zero codes.

    Input shape: [num_groups, (N/32)*bits] int32.
    Output shape: [num_groups, N] int32 (zero code for each group/column).
    """
    num_groups, n_words = qzeros.shape
    n32 = n_words // bits
    n = n32 * 32
    out = torch.zeros((num_groups, n), dtype=torch.int32, device=qzeros.device)
    plane_specs = {
        3: [(2, 0, 0), (1, 2, 2)],
        5: [(4, 0, 0), (1, 4, 4)],
        6: [(4, 0, 0), (2, 4, 4)],
        7: [(4, 0, 0), (2, 4, 4), (1, 6, 6)],
    }
    specs = plane_specs[bits]
    pos = torch.arange(32, device=qzeros.device, dtype=torch.int64)
    for cb in range(n32):
        base = cb * bits
        for w, start, off in specs:
            pack_factor = 32 // w
            word_idx = start + pos // pack_factor
            shift = (w * (pos % pack_factor)).view(1, 32)
            mask = (1 << w) - 1
            selected = qzeros[:, base + word_idx]
            part = ((selected >> shift) & mask) << off
            out[:, cb * 32 : (cb + 1) * 32] |= part
    return out.to(torch.int8)


def _prepack_qweight_vnni(packed: torch.Tensor, qzeros: torch.Tensor, g_idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Convert the uint8 pre-expanded layout into a signed int8 (code - zero) layout.

    Input packed shape: [N/16, K/32, 8, 16] int32 (uint8 codes, 4 per word).
    Output shape: same layout, but each byte is a signed int8 (code - zero).
    """
    num_cb, num_kb, _, width16 = packed.shape
    num_groups = qzeros.size(0)
    zero_codes = _decode_zero_codes(qzeros, bits)
    g_idx_norm = g_idx.to(torch.int64)
    g_idx_norm = torch.where(g_idx_norm < 0, g_idx_norm + num_groups, g_idx_norm)
    zero_per_k = zero_codes[g_idx_norm]  # [K, N]
    zero_per_k = zero_per_k.view(num_kb, 32, num_cb, width16)
    zero_per_k = zero_per_k.view(num_kb, 8, 4, num_cb, width16)
    zero_per_k = zero_per_k.permute(3, 0, 1, 4, 2)  # [num_cb, num_kb, 8, width16, 4]
    packed_i8 = packed.view(torch.int8).view(num_cb, num_kb, 8, width16, 4)
    diff_i8 = (packed_i8.to(torch.int16) - zero_per_k.to(torch.int16)).to(torch.int8)
    return diff_i8.view(torch.int32).view(num_cb, num_kb, 8, width16).contiguous()


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


def _prepack_qweight_for_cpu(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
    preexpand: bool = True,
    vnni: bool = False,
) -> torch.Tensor:
    """Transpose planar qweight for the CPU kernel.

    The CPU kernel accepts two packed layouts for `kernel_bits == 8`:

    * Generic FP32 path: uint8 ``code`` values packed 4 per 32-bit word.  The
      kernel subtracts ``zero * scale`` in the FMA loop.
    * VNNI path: signed int8 ``code - zero`` values packed 4 per 32-bit word.
      The kernel multiplies by ``scale`` directly and uses ``vpdpwssd`` for
      int16 dot products.

    When ``preexpand`` is True, decode the packed planar words into the uint8
    pre-expanded layout ``[N/16, K/32, 8, 16]`` so the SIMD kernel loads one word
    per K value instead of ``bits`` words per K-block.  When ``vnni`` is True,
    the same shape is returned but each byte is signed int8 ``(code - zero)``
    for the AVX-512 VNNI path.  Python passes a matching ``use_vnni`` flag to the
    C++ dispatcher so the kernel selects the same interpretation.  Cached weakly
    per qweight tensor and mode.
    """
    expected_planes = 8
    if vnni:
        cache = _QWEIGHT_CPU_VNNI_CACHE
    elif preexpand:
        cache = _QWEIGHT_CPU_PREEXPAND_CACHE
    else:
        expected_planes = bits
        cache = _QWEIGHT_CPU_PACKED_CACHE
    if qweight.dim() == 4 and qweight.size(2) == expected_planes:
        return qweight
    cached = cache.get(qweight)
    if cached is not None and cached.size(2) == expected_planes:
        return cached
    packed = _prepack_planar_to_cpu(qweight, bits)
    if vnni:
        packed = _preexpand_qweight_uint8(packed, bits)
        packed = _prepack_qweight_vnni(packed, qzeros, g_idx, bits)
    elif preexpand:
        packed = _preexpand_qweight_uint8(packed, bits)
    cache[qweight] = packed
    return packed


def g_idx_block_uniform(g_idx: torch.Tensor) -> bool:
    """True when every 32-row block maps to a single group (cached per g_idx tensor)."""
    if g_idx.numel() % 32 != 0:
        return False
    cached = _G_IDX_BLOCK_UNIFORM_CACHE.get(g_idx)
    version = g_idx._version
    if cached is None or cached[0] != version:
        blocks = g_idx.reshape(-1, 32)
        cached = (version, bool((blocks == blocks[:, :1]).all().item()))
        _G_IDX_BLOCK_UNIFORM_CACHE[g_idx] = cached
    return cached[1]


_g_idx_block_uniform = g_idx_block_uniform


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

    On VNNI-capable x86 CPUs the CPU kernel uses an int16 dot-product micro-kernel
    for M <= 8.  M == 16 and M == 32 are sliced into M == 8 VNNI chunks in Python
    to keep the accumulator set in ZMM registers.  It can be disabled with
    ``GPTQMODEL_PANGOLIN_CPU_DISABLE_VNNI=1``.

    The packed-planar VNNI path (``GPTQMODEL_PANGOLIN_CPU_ENABLE_VNNI_PACKED=1``)
    decodes the original ``[N/16, K/32, bits, 16]`` qweight layout on-the-fly
    instead of pre-expanding it to int8, saving weight memory traffic.  This is
    especially useful for 3-bit weights and is currently experimental.

    The VPDPBUSD uint8/int8 dot-product path (``GPTQMODEL_PANGOLIN_CPU_ENABLE_VNNI_BUSD=1``)
    is opt-in.  It uses a 4-K-row activation scale word to keep 8-bit quantization
    error within the default rtol=0.02/atol=0.01 tolerance on real laguna/glm45
    shapes for bits 3/5/6/7.  The smaller scale word trades some throughput for
    accuracy, so it is not yet the default VNNI path.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    if x.is_cuda:
        return _gemv_op()(x, qweight, scales, qzeros, g_idx, bits)
    if x.device.type == "mps":
        from .pangolin_mps import pangolin_mps_gemv

        return pangolin_mps_gemv(
            x, qweight, scales, qzeros, g_idx, bits,
            planar=bits in (3, 5, 6, 7),
        )
    if not ensure_pangolin_cpu_runtime_available():
        raise RuntimeError(pangolin_cpu_runtime_error())
    M = x.size(0)
    K = x.size(1)
    if K % 32 != 0 or g_idx.numel() % 32 != 0:
        raise RuntimeError(
            f"Pangolin CPU kernel requires K and g_idx length divisible by 32, got K={K}"
        )
    if not g_idx_block_uniform(g_idx):
        raise RuntimeError("Pangolin CPU kernel requires g_idx to be uniform across 32-row blocks")
    preexpand_env = os.environ.get("GPTQMODEL_PANGOLIN_CPU_PREEXPAND_QWEIGHT")
    if preexpand_env is not None:
        preexpand = preexpand_env.lower() in ("1", "true", "on")
    else:
        preexpand = _should_preexpand_qweight(qweight, bits)
    # VNNI is the default for small batches: it pre-subtracts the zero point from
    # each code, quantizes the activation per group to int16, and uses VPDPBWSSD.
    # The `use_vnni` flag is passed explicitly to the C++ kernel so the Python
    # dispatcher and the kernel always agree on the signedness of the prepacked
    # qweight (signed int8 for VNNI, uint8 code for the generic FP32 path).
    disable_avx512 = os.environ.get("GPTQMODEL_PANGOLIN_CPU_DISABLE_AVX512")
    disable_vnni = os.environ.get("GPTQMODEL_PANGOLIN_CPU_DISABLE_VNNI")
    avx512_disabled = disable_avx512 is not None and disable_avx512.lower() in ("1", "true", "on")
    vnni_disabled = disable_vnni is not None and disable_vnni.lower() in ("1", "true", "on")
    has_vnni = _cpu_has_avx512_vnni() and not avx512_disabled and not vnni_disabled
    vnni_packed_env = os.environ.get("GPTQMODEL_PANGOLIN_CPU_ENABLE_VNNI_PACKED")
    use_vnni_packed = (
        vnni_packed_env is not None and vnni_packed_env.lower() in ("1", "true", "on")
        and has_vnni
        and (M <= 8 or M in (16, 32))
    )
    # VNNI prepack is used for M <= 8 (int16 or busd dot-product) and for M in
    # {16, 32} when the C++ kernel handles the larger batch directly.
    use_vnni_prepack = (M <= 8 or M in (16, 32)) and has_vnni and not use_vnni_packed
    use_vnni = use_vnni_prepack
    # VPDPBUSD (uint8 activation * int8 weight) is faster but exceeds the default
    # numerical tolerance on real shapes, so it is gated off by default and can be
    # enabled explicitly with GPTQMODEL_PANGOLIN_CPU_ENABLE_VNNI_BUSD=1 for
    # experiments.  The int16 VNNI path is the safe default for M <= 8.
    busd_env = os.environ.get("GPTQMODEL_PANGOLIN_CPU_ENABLE_VNNI_BUSD")
    use_vnni_busd = busd_env is not None and busd_env.lower() in ("1", "true", "on")
    if use_vnni_packed:
        preexpand = False
    elif use_vnni_prepack:
        preexpand = True
    qweight_packed = _prepack_qweight_for_cpu(
        qweight, qzeros, g_idx, bits, preexpand=preexpand, vnni=use_vnni_prepack
    )
    # Slice M=16/32 only when the int16 VNNI path is being used; busd can process
    # up to M=32 in one C++ call when explicitly enabled.
    if M in (16, 32) and has_vnni and not use_vnni_busd:
        out_chunks: list[torch.Tensor] = []
        for start in range(0, M, 8):
            chunk = x[start : start + 8]
            out_chunks.append(
                _gemv_cpu_op()(
                    chunk, qweight_packed, scales, qzeros, g_idx, bits,
                    False if use_vnni_packed else True, use_vnni_packed, use_vnni_busd,
                    None,
                )
            )
        return torch.cat(out_chunks, dim=0).to(x.dtype)
    return _gemv_cpu_op()(
        x, qweight_packed, scales, qzeros, g_idx, bits, use_vnni, use_vnni_packed, use_vnni_busd,
        None,
    ).to(x.dtype)


__all__ = [
    "PANGOLIN_BITS",
    "PANGOLIN_MAX_M",
    "PANGOLIN_SUPPORTED_M",
    "ensure_pangolin_cpu_runtime_available",
    "ensure_pangolin_runtime_available",
    "g_idx_block_uniform",
    "pangolin_cpu_runtime_available",
    "pangolin_cpu_runtime_error",
    "pangolin_cpu_supported",
    "pangolin_gemv",
    "pangolin_runtime_available",
    "pangolin_runtime_error",
    "pangolin_supported",
]
