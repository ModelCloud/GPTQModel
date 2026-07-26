# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import contextlib
import json
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
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
from .marlin import (
    _marlin_resolve_op,
    gptq_marlin_repack,
    marlin_make_workspace_new,
    marlin_permute_scales,
    marlin_runtime_available,
)
from .marlin_scalar_type import scalar_types


_AMPLIN_OPS_NAME = "gptqmodel_amplin_ops"
_AMPLIN_NAMESPACE = "gptqmodel_amplin"
_AMPLIN_REQUIRED_CUDA_HEADERS = ("cuda_runtime_api.h",)
HMMA_K_TILE = 128
HMMA_N_TILE = 64
HMMA_PACKED_K_WORDS = HMMA_K_TILE // 8
MMA_LANE_K_STEPS = HMMA_K_TILE // 16
MMA_LANE_N_WARPS = HMMA_N_TILE // 16
MMA_LANES = 32
_GPTQ_LOGICAL_ZERO_WORD = -2004318072  # int32 bit pattern 0x88888888


def _amplin_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "amplin"


def _amplin_sources() -> list[str]:
    root = _amplin_root()
    return [
        str(root / "amplin.cpp"),
        str(root / "amplin_kernel.cu"),
    ]


def _amplin_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_amplin_root())],
        required_header_names=_AMPLIN_REQUIRED_CUDA_HEADERS,
    )


def _amplin_extra_cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags(
        enable_bf16=True,
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


_AMPLIN_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_AMPLIN_OPS_NAME,
    namespace=_AMPLIN_NAMESPACE,
    required_ops=(
        "gemv",
        "gemv_k12288_wide",
        "gemv_multirow",
        "gemm_hmma",
        "gemm_hmma_v0",
        "gemm_hmma_m64_v1",
        "gemm_hmma_m64_v2",
        "gemm_hmma_m64_v2_sync_a128",
        "gemm_hmma_m64_v3",
        "gemm_hmma_m32_n128_pipeline4",
        "mma_lane_m64",
        "mma_lane_m64_global_a",
        "mma_lane_m32_global_a",
        "mma_lane_m32_n32_global_a",
        "mma_lane_m16_n64_shared_a",
        "mma_lane_m16_n64_tile4_shared_a",
        "mma_lane_m16_n64_tile8_shared_a",
        "mma_lane_m32_n64_tile2_interleaved_dequant",
        "mma_lane_m32_n64_tile4_shared_a",
        "mma_lane_m32_n64_tile8_shared_a",
        "mma_lane_m32_n64_tile2_splitk2",
        "mma_lane_m32_n64_tile1_splitk4",
        "mma_lane_m32_n64_tile1_splitk8",
        "mma_lane_m32_n64_tile2_splitk4",
        "mma_lane_m32_n64_shared_a",
        "mma_lane_m32_n64_splitk12x2_coop_interleaved",
        "mma_lane_m16_n16_padded",
        "mma_lane_m16_n16_splitk4",
        "mma_lane_m16_n16_splitk8",
        "mma_lane_m16_n16_splitk12",
        "mma_lane_m16_n32_splitk12",
        "mma_lane_m16_n32_splitk16",
        "mma_lane_m16_n32_splitk8",
        "mma_lane_m16_n32_splitk8_pipe2",
        "mma_lane_m16_n32_splitk12_pipe2",
        "mma_lane_m16_n32_splitk12_pipe2_interleaved",
        "mma_lane_m16_n64_splitk24_pipe2_interleaved",
        "mma_lane_m16_n64_splitk4_pipe2_interleaved",
        "mma_lane_m16_n64_splitk20_pipe2_interleaved",
        "mma_lane_m32_n64_splitk24_pipe2_interleaved",
        "mma_lane_m32_n64_splitk12_pipe2_interleaved",
        "mma_lane_m32_n64_splitk16_pipe2_interleaved",
        "mma_lane_m32_n64_splitk20_pipe2_interleaved",
        "mma_lane_m16_n64_splitk12x2_coop_interleaved",
        "mma_lane_m16_n32_splitk16_pipe2",
        "mma_lane_m16_n16_splitk16",
        "mma_lane_tile",
        "mma_lane_tile_global_a",
        "marlin_style_run",
    ),
    sources=_amplin_sources,
    build_root_env="GPTQMODEL_AMPLIN_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("amplin"),
    display_name="Amplin Ampere GPTQ W4A16 GEMV",
    extra_cflags=lambda: default_jit_cflags(enable_bf16=True),
    extra_cuda_cflags=_amplin_extra_cuda_cflags,
    extra_include_paths=_amplin_include_paths,
    force_rebuild_env="GPTQMODEL_AMPLIN_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def _sm80_device_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return any(torch.cuda.get_device_capability(index) == (8, 0) for index in range(torch.cuda.device_count()))
    except (RuntimeError, AssertionError):
        return False


def amplin_supported() -> bool:
    return _sm80_device_available()


def amplin_runtime_error() -> str:
    if not torch.cuda.is_available():
        return "Amplin requires CUDA."
    if not _sm80_device_available():
        return "Amplin V0 requires at least one CUDA compute capability 8.0 device."
    return _AMPLIN_TORCH_OPS_EXTENSION.last_error_message()


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def amplin_runtime_available() -> bool:
    if not amplin_supported():
        return False
    return _extension_api().is_available("amplin")


# Cache the runtime-availability check and individual torch.ops handles so the
# dynamic router does not pay extension-loading overhead on every call.
_AMPLIN_RUNTIME_AVAILABLE: bool | None = None
_AMPLIN_INIT_LOCK = threading.Lock()


def _ensure_amplin_runtime_available() -> bool:
    with _AMPLIN_INIT_LOCK:
        global _AMPLIN_RUNTIME_AVAILABLE
        if _AMPLIN_RUNTIME_AVAILABLE is None:
            _AMPLIN_RUNTIME_AVAILABLE = amplin_runtime_available()
        return _AMPLIN_RUNTIME_AVAILABLE


def _get_amplin_op(op_name: str) -> Callable:
    caches = _thread_caches()
    op = caches.candidate_op_cache.get(op_name)
    if op is None:
        with _AMPLIN_INIT_LOCK:
            op = _extension_api().op("amplin", op_name)
        caches.candidate_op_cache[op_name] = op
    return op


def pack_hmma_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """Reorder canonical GPTQ W4 words into [N/64, K/128, 64, 16] Ampere execution tiles."""

    if qweight.dtype != torch.int32:
        raise ValueError("Amplin HMMA qweight must use torch.int32")
    if qweight.dim() != 2:
        raise ValueError("Amplin HMMA qweight must be two-dimensional [K/8, N]")
    packed_rows, size_n = qweight.shape
    if packed_rows <= 0 or packed_rows % HMMA_PACKED_K_WORDS != 0:
        raise ValueError("Amplin HMMA qweight K must be positive and divisible by 128")
    if size_n <= 0:
        raise ValueError("Amplin HMMA qweight N must be positive")

    num_groups = packed_rows // HMMA_PACKED_K_WORDS
    num_n_tiles = (size_n + HMMA_N_TILE - 1) // HMMA_N_TILE
    padded_n = num_n_tiles * HMMA_N_TILE
    padded = torch.full(
        (packed_rows, padded_n),
        _GPTQ_LOGICAL_ZERO_WORD,
        dtype=qweight.dtype,
        device=qweight.device,
    )
    padded[:, :size_n].copy_(qweight)
    return (
        padded.view(num_groups, HMMA_PACKED_K_WORDS, num_n_tiles, HMMA_N_TILE)
        .permute(2, 0, 3, 1)
        .contiguous()
    )


def pack_mma_lane_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """Pack W4 codes as [N64, K128, K16, N16, lane] native Ampere B-fragment words."""

    if qweight.dtype != torch.int32:
        raise ValueError("Amplin MMA-lane qweight must use torch.int32")
    if qweight.dim() != 2:
        raise ValueError("Amplin MMA-lane qweight must be two-dimensional [K/8, N]")
    packed_rows, size_n = qweight.shape
    if packed_rows <= 0 or packed_rows % HMMA_PACKED_K_WORDS != 0:
        raise ValueError("Amplin MMA-lane qweight K must be positive and divisible by 128")
    if size_n <= 0:
        raise ValueError("Amplin MMA-lane qweight N must be positive")

    num_groups = packed_rows // HMMA_PACKED_K_WORDS
    num_n_tiles = (size_n + HMMA_N_TILE - 1) // HMMA_N_TILE
    padded_n = num_n_tiles * HMMA_N_TILE
    padded = torch.full(
        (packed_rows, padded_n),
        _GPTQ_LOGICAL_ZERO_WORD,
        dtype=qweight.dtype,
        device=qweight.device,
    )
    padded[:, :size_n].copy_(qweight)
    words = (
        padded.view(
            num_groups,
            MMA_LANE_K_STEPS,
            2,
            num_n_tiles,
            MMA_LANE_N_WARPS,
            16,
        ).to(torch.int64)
        & 0xFFFFFFFF
    )
    even_shifts = (
        torch.arange(4, dtype=torch.int64, device=qweight.device).mul_(8).view(1, 1, 1, 1, 1, 4)
    )

    def codes(word_half: int, column_start: int, *, odd: bool) -> torch.Tensor:
        selected = words[:, :, word_half, :, :, column_start : column_start + 8].unsqueeze(-1)
        shifts = even_shifts + (4 if odd else 0)
        return (selected >> shifts) & 0xF

    packed = (
        codes(0, 0, odd=False)
        | (codes(1, 0, odd=False) << 4)
        | (codes(0, 8, odd=False) << 8)
        | (codes(1, 8, odd=False) << 12)
        | (codes(0, 0, odd=True) << 16)
        | (codes(1, 0, odd=True) << 20)
        | (codes(0, 8, odd=True) << 24)
        | (codes(1, 8, odd=True) << 28)
    )
    return (
        packed.reshape(
            num_groups,
            MMA_LANE_K_STEPS,
            num_n_tiles,
            MMA_LANE_N_WARPS,
            MMA_LANES,
        )
        .permute(2, 0, 1, 3, 4)
        .contiguous()
        .to(torch.int32)
    )


def unpack_mma_lane_qweight(
    packed_qweight: torch.Tensor,
    *,
    size_n: int,
) -> torch.Tensor:
    """Restore canonical [K/8, N] words from the Ampere B-fragment lane layout."""

    if packed_qweight.dtype != torch.int32:
        raise ValueError("Amplin MMA-lane packed qweight must use torch.int32")
    if packed_qweight.dim() != 5:
        raise ValueError("Amplin MMA-lane qweight must have shape [N64, K128, K16, N16, lane]")
    num_n_tiles, num_groups, k_steps, n_warps, lanes = packed_qweight.shape
    if (
        num_n_tiles <= 0
        or num_groups <= 0
        or k_steps != MMA_LANE_K_STEPS
        or n_warps != MMA_LANE_N_WARPS
        or lanes != MMA_LANES
    ):
        raise ValueError("Amplin MMA-lane qweight must have shape [N64, K128, K16, N16, lane]")
    padded_n = num_n_tiles * HMMA_N_TILE
    if size_n <= 0 or size_n > padded_n:
        raise ValueError("Amplin MMA-lane logical N must be positive and no larger than the packed N")

    lane_words = (
        packed_qweight.permute(1, 2, 0, 3, 4)
        .contiguous()
        .view(
            num_groups,
            MMA_LANE_K_STEPS,
            num_n_tiles,
            MMA_LANE_N_WARPS,
            8,
            4,
        )
        .to(torch.int64)
        & 0xFFFFFFFF
    )
    nibble_shifts = (
        torch.arange(4, dtype=torch.int64, device=packed_qweight.device).mul_(8).view(1, 1, 1, 1, 1, 4)
    )

    def rebuild_word(even_bit: int, odd_bit: int) -> torch.Tensor:
        even_codes = (lane_words >> even_bit) & 0xF
        odd_codes = (lane_words >> odd_bit) & 0xF
        return ((even_codes << nibble_shifts) | (odd_codes << (nibble_shifts + 4))).sum(dim=-1)

    word_0 = torch.cat((rebuild_word(0, 16), rebuild_word(8, 24)), dim=-1)
    word_1 = torch.cat((rebuild_word(4, 20), rebuild_word(12, 28)), dim=-1)
    canonical_padded = (
        torch.stack((word_0, word_1), dim=2)
        .contiguous()
        .view(num_groups * HMMA_PACKED_K_WORDS, padded_n)
        .to(torch.int32)
    )
    return canonical_padded[:, :size_n].contiguous()


def unpack_hmma_qweight(packed_qweight: torch.Tensor, *, size_n: int) -> torch.Tensor:
    """Restore canonical [K/8, N] GPTQ words from Amplin's N64/K128 execution tiles."""

    if packed_qweight.dtype != torch.int32:
        raise ValueError("Amplin HMMA packed qweight must use torch.int32")
    if packed_qweight.dim() != 4:
        raise ValueError("Amplin HMMA packed qweight must have shape [N/64, K/128, 64, 16]")
    num_n_tiles, num_groups, tile_n, packed_k_words = packed_qweight.shape
    if num_n_tiles <= 0 or num_groups <= 0 or tile_n != HMMA_N_TILE or packed_k_words != HMMA_PACKED_K_WORDS:
        raise ValueError("Amplin HMMA packed qweight must have shape [N/64, K/128, 64, 16]")
    padded_n = num_n_tiles * HMMA_N_TILE
    if size_n <= 0 or size_n > padded_n:
        raise ValueError("Amplin HMMA logical N must be positive and no larger than the packed N")

    canonical_padded = (
        packed_qweight.permute(1, 3, 0, 2)
        .contiguous()
        .view(num_groups * HMMA_PACKED_K_WORDS, padded_n)
    )
    return canonical_padded[:, :size_n].contiguous()


def pack_mma_lane_n32_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """Interleave each lane's two N32 words as [N32, K128, K16, lane, 2]."""

    packed = pack_mma_lane_qweight(qweight)
    num_n64_tiles, num_groups, k_steps, n16_tiles, lanes = packed.shape
    if n16_tiles != MMA_LANE_N_WARPS or MMA_LANE_N_WARPS != 4:
        raise AssertionError("Amplin N32 interleave requires four N16 tiles per N64 tile")
    return (
        packed.view(num_n64_tiles, num_groups, k_steps, 2, 2, lanes)
        .permute(0, 3, 1, 2, 5, 4)
        .contiguous()
        .view(num_n64_tiles * 2, num_groups, k_steps, lanes, 2)
    )


def unpack_mma_lane_n32_qweight(
    packed_qweight: torch.Tensor,
    *,
    size_n: int,
) -> torch.Tensor:
    """Restore canonical [K/8, N] words from the interleaved N32 lane layout."""

    if packed_qweight.dtype != torch.int32:
        raise ValueError("Amplin MMA-lane N32 packed qweight must use torch.int32")
    if packed_qweight.dim() != 5:
        raise ValueError("Amplin MMA-lane N32 qweight must have shape [N32, K128, K16, lane, 2]")
    num_n32_tiles, num_groups, k_steps, lanes, words = packed_qweight.shape
    if (
        num_n32_tiles <= 0
        or num_n32_tiles % 2 != 0
        or num_groups <= 0
        or k_steps != MMA_LANE_K_STEPS
        or lanes != MMA_LANES
        or words != 2
    ):
        raise ValueError("Amplin MMA-lane N32 qweight must have shape [N32, K128, K16, lane, 2]")
    num_n64_tiles = num_n32_tiles // 2
    packed_n16 = (
        packed_qweight.view(num_n64_tiles, 2, num_groups, k_steps, lanes, 2)
        .permute(0, 2, 3, 1, 5, 4)
        .contiguous()
        .view(num_n64_tiles, num_groups, k_steps, MMA_LANE_N_WARPS, lanes)
    )
    return unpack_mma_lane_qweight(packed_n16, size_n=size_n)


def pack_mma_lane_n64_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """Interleave each lane's four N64 words as [N64, K128, K16, lane, 4]."""

    return pack_mma_lane_qweight(qweight).permute(0, 1, 2, 4, 3).contiguous()


def unpack_mma_lane_n64_qweight(
    packed_qweight: torch.Tensor,
    *,
    size_n: int,
) -> torch.Tensor:
    """Restore canonical [K/8, N] words from the interleaved N64 lane layout."""

    if packed_qweight.dtype != torch.int32:
        raise ValueError("Amplin MMA-lane N64 packed qweight must use torch.int32")
    if packed_qweight.dim() != 5:
        raise ValueError("Amplin MMA-lane N64 qweight must have shape [N64, K128, K16, lane, 4]")
    num_n64_tiles, num_groups, k_steps, lanes, words = packed_qweight.shape
    if (
        num_n64_tiles <= 0
        or num_groups <= 0
        or k_steps != MMA_LANE_K_STEPS
        or lanes != MMA_LANES
        or words != MMA_LANE_N_WARPS
    ):
        raise ValueError("Amplin MMA-lane N64 qweight must have shape [N64, K128, K16, lane, 4]")
    packed_n16 = packed_qweight.permute(0, 1, 2, 4, 3).contiguous()
    return unpack_mma_lane_qweight(packed_n16, size_n=size_n)


def pack_hmma_scales(scales: torch.Tensor) -> torch.Tensor:
    """Tile group-128 scales as [N/64, K/128, 64], padding masked output channels with zero."""

    if not scales.is_floating_point():
        raise ValueError("Amplin HMMA scales must use a floating-point dtype")
    if scales.dim() != 2:
        raise ValueError("Amplin HMMA scales must be two-dimensional [K/128, N]")
    num_groups, size_n = scales.shape
    if num_groups <= 0 or size_n <= 0:
        raise ValueError("Amplin HMMA scales dimensions must be positive")

    num_n_tiles = (size_n + HMMA_N_TILE - 1) // HMMA_N_TILE
    padded_n = num_n_tiles * HMMA_N_TILE
    padded = torch.zeros((num_groups, padded_n), dtype=scales.dtype, device=scales.device)
    padded[:, :size_n].copy_(scales)
    return padded.view(num_groups, num_n_tiles, HMMA_N_TILE).permute(1, 0, 2).contiguous()


def unpack_hmma_scales(packed_scales: torch.Tensor, *, size_n: int) -> torch.Tensor:
    """Restore canonical [K/128, N] scales from Amplin's N64 execution tiles."""

    if not packed_scales.is_floating_point():
        raise ValueError("Amplin HMMA packed scales must use a floating-point dtype")
    if packed_scales.dim() != 3:
        raise ValueError("Amplin HMMA packed scales must have shape [N/64, K/128, 64]")
    num_n_tiles, num_groups, tile_n = packed_scales.shape
    if num_n_tiles <= 0 or num_groups <= 0 or tile_n != HMMA_N_TILE:
        raise ValueError("Amplin HMMA packed scales must have shape [N/64, K/128, 64]")
    padded_n = num_n_tiles * HMMA_N_TILE
    if size_n <= 0 or size_n > padded_n:
        raise ValueError("Amplin HMMA logical N must be positive and no larger than the packed N")

    canonical_padded = packed_scales.permute(1, 0, 2).contiguous().view(num_groups, padded_n)
    return canonical_padded[:, :size_n].contiguous()


def pack_hmma_weights(
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and pack a canonical GPTQ W4/group-128 weight pair for the Amplin HMMA prototype."""

    if qweight.dim() != 2 or scales.dim() != 2:
        raise ValueError("Amplin HMMA qweight and scales must be two-dimensional")
    expected_groups = qweight.size(0) // HMMA_PACKED_K_WORDS
    if qweight.size(0) % HMMA_PACKED_K_WORDS != 0 or scales.shape != (expected_groups, qweight.size(1)):
        raise ValueError("Amplin HMMA scales must have canonical shape [K/128, N]")
    return pack_hmma_qweight(qweight), pack_hmma_scales(scales)


def gemv(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Run the raw Amplin GPTQ W4, group-128 linear operator on supported sm_80 shapes."""

    return _extension_api().op("amplin", "gemv")(input, qweight, scales)


def gemv_k12288_wide(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Run the barrier-free 16-warp research path for M1 K12288 projections."""

    return _extension_api().op("amplin", "gemv_k12288_wide")(input, qweight, scales)


def gemv_multirow(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Run the 2/4-row weight-reuse research path for large MLP projections."""

    return _extension_api().op("amplin", "gemv_multirow")(input, qweight, scales)


def gemm_hmma(
    input: torch.Tensor,
    packed_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the raw Amplin W4-N64-K128 HMMA prototype on an exact sm_80 device."""

    return _extension_api().op("amplin", "gemm_hmma")(input, packed_qweight, packed_scales, logical_n)


def gemm_hmma_v0(
    input: torch.Tensor,
    packed_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the retained M16xN64 HMMA control without M64 schedule selection."""

    return _extension_api().op("amplin", "gemm_hmma_v0")(input, packed_qweight, packed_scales, logical_n)


def gemm_hmma_m64_v1(
    input: torch.Tensor,
    packed_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the retained scalar-store M64 HMMA control without automatic schedule selection."""

    return _extension_api().op("amplin", "gemm_hmma_m64_v1")(
        input,
        packed_qweight,
        packed_scales,
        logical_n,
    )


def gemm_hmma_m64_v2(
    input: torch.Tensor,
    packed_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the retained wide-store, synchronous-A M64 HMMA control."""

    return _extension_api().op("amplin", "gemm_hmma_m64_v2")(
        input,
        packed_qweight,
        packed_scales,
        logical_n,
    )


def gemm_hmma_m64_v2_sync_a128(
    input: torch.Tensor,
    packed_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the synchronous 128-bit A-copy M64 HMMA control."""

    return _extension_api().op("amplin", "gemm_hmma_m64_v2_sync_a128")(
        input,
        packed_qweight,
        packed_scales,
        logical_n,
    )


def gemm_hmma_m64_v3(
    input: torch.Tensor,
    packed_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the retained cp.async A-copy M64 HMMA V3 control."""

    return _extension_api().op("amplin", "gemm_hmma_m64_v3")(
        input,
        packed_qweight,
        packed_scales,
        logical_n,
    )


def gemm_hmma_m32_n128_pipeline4(
    input: torch.Tensor,
    packed_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the Marlin-style 4-warp 4-stage cp.async pipeline for M=32, N multiple of 64."""

    return _extension_api().op("amplin", "gemm_hmma_m32_n128_pipeline4")(
        input,
        packed_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_tile(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Run the one-warp K16xN16 Ampere register-fragment mapping proof."""

    return _extension_api().op("amplin", "mma_lane_tile")(
        input,
        packed_lane_qweight,
        scales,
    )


def mma_lane_tile_global_a(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Run the direct global-to-register A-fragment mapping proof."""

    return _extension_api().op("amplin", "mma_lane_tile_global_a")(
        input,
        packed_lane_qweight,
        scales,
    )


def mma_lane_m64(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the forced M64 register-fed B-fragment research path."""

    return _extension_api().op("amplin", "mma_lane_m64")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m64_global_a(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the forced M64 path with direct global A and register-fed B fragments."""

    return _extension_api().op("amplin", "mma_lane_m64_global_a")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_global_a(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the forced M32xN64 direct-A path with M16xN32 warp ownership."""

    return _extension_api().op("amplin", "mma_lane_m32_global_a")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n32_global_a(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the forced M32xN32 direct-A path with guarded N8 output fragments."""

    return _extension_api().op("amplin", "mma_lane_m32_n32_global_a")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n64_shared_a(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M16 N64 with A resident in shared memory (128 threads)."""

    return _extension_api().op("amplin", "mma_lane_m16_n64_shared_a")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n64_tile4_shared_a(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M16 N64 with A resident in shared memory, 4 N64 tiles per block (128 threads)."""

    return _extension_api().op("amplin", "mma_lane_m16_n64_tile4_shared_a")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n64_tile8_shared_a(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M16 N64 with A resident in shared memory, 8 N64 tiles per block (256 threads)."""

    return _extension_api().op("amplin", "mma_lane_m16_n64_tile8_shared_a")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_tile2_shared_a(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with A resident in shared memory, 2 N64 tiles per block (128 threads)."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_tile2_shared_a")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_tile2_interleaved_dequant(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 tile2 with word-level interleaved dequantization (dequant w+1 while MMA w)."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_tile2_interleaved_dequant")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_tile4_shared_a(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with A resident in shared memory, 4 N64 tiles per block (256 threads)."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_tile4_shared_a")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_tile8_shared_a(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with A resident in shared memory, 8 N64 tiles per block (512 threads)."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_tile8_shared_a")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_tile2_splitk2(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with 2 N64 tiles and K-split=2 in shared memory (256 threads, 49 KiB)."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_tile2_splitk2")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_tile1_splitk4(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with 1 N64 tile and K-split=4 in shared memory (256 threads, 66 KiB)."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_tile1_splitk4")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_tile1_splitk8(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with 1 N64 tile and K-split=8 in shared memory (512 threads, 132 KiB)."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_tile1_splitk8")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_tile2_splitk4(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with 2 N64 tiles and K-split=4 in shared memory (512 threads, 98 KiB)."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_tile2_splitk4")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_shared_a(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with A resident in shared memory (256 threads)."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_shared_a")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_splitk12x2_coop_interleaved(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with two cooperative K12 CTAs and device-memory partials."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_splitk12x2_coop_interleaved")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n16_padded(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the M=2/4/8/16 padded-M16 path with one direct-A N16 warp per CTA."""

    return _extension_api().op("amplin", "mma_lane_m16_n16_padded")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n16_splitk4(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the K12288 four-warp intra-CTA split-K padded-M16 control."""

    return _extension_api().op("amplin", "mma_lane_m16_n16_splitk4")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n16_splitk8(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the K12288 eight-warp intra-CTA split-K padded-M16 control."""

    return _extension_api().op("amplin", "mma_lane_m16_n16_splitk8")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n16_splitk12(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the K12288 12-warp intra-CTA split-K padded-M16 control."""

    return _extension_api().op("amplin", "mma_lane_m16_n16_splitk12")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n32_splitk8(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the eight-warp split-K path with N32 activation reuse."""

    return _extension_api().op("amplin", "mma_lane_m16_n32_splitk8")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n32_splitk8_pipe2(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the eight-warp N32 path with two-stage register prefetching."""

    return _extension_api().op("amplin", "mma_lane_m16_n32_splitk8_pipe2")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n32_splitk12(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the K12288 12-warp split-K path with N32 activation reuse."""

    return _extension_api().op("amplin", "mma_lane_m16_n32_splitk12")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n32_splitk16(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the K12288 16-warp split-K path with N32 activation reuse."""

    return _extension_api().op("amplin", "mma_lane_m16_n32_splitk16")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n32_splitk12_pipe2(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the K12288 12-warp N32 path with two-stage register prefetching."""

    return _extension_api().op("amplin", "mma_lane_m16_n32_splitk12_pipe2")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n32_splitk12_pipe2_interleaved(
    input: torch.Tensor,
    packed_lane_n32_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run K12 pipe2 with each lane's N32 int32 words fetched as one aligned pair."""

    return _extension_api().op("amplin", "mma_lane_m16_n32_splitk12_pipe2_interleaved")(
        input,
        packed_lane_n32_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n64_splitk24_pipe2_interleaved(
    input: torch.Tensor,
    packed_lane_n64_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run K24/N64 pipe2 with each lane's four int32 words fetched as one aligned vector."""

    return _extension_api().op("amplin", "mma_lane_m16_n64_splitk24_pipe2_interleaved")(
        input,
        packed_lane_n64_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n64_splitk4_pipe2_interleaved(
    input: torch.Tensor,
    packed_lane_n64_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run K4/N64 pipe2 with single-CTA split-K for small M and large K/N shapes."""

    return _extension_api().op("amplin", "mma_lane_m16_n64_splitk4_pipe2_interleaved")(
        input,
        packed_lane_n64_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n64_splitk8_pipe2_interleaved(
    input: torch.Tensor,
    packed_lane_n64_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run K8/N64 pipe2 with single-CTA split-K for small M and large K/N shapes."""

    return _extension_api().op("amplin", "mma_lane_m16_n64_splitk8_pipe2_interleaved")(
        input,
        packed_lane_n64_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n64_splitk12_pipe2_interleaved(
    input: torch.Tensor,
    packed_lane_n64_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run K12/N64 pipe2 with single-CTA split-K for small M and large K/N shapes."""

    return _extension_api().op("amplin", "mma_lane_m16_n64_splitk12_pipe2_interleaved")(
        input,
        packed_lane_n64_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n64_splitk16_pipe2_interleaved(
    input: torch.Tensor,
    packed_lane_n64_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run K16/N64 pipe2 with single-CTA split-K for small M and large K/N shapes."""

    return _extension_api().op("amplin", "mma_lane_m16_n64_splitk16_pipe2_interleaved")(
        input,
        packed_lane_n64_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n64_splitk20_pipe2_interleaved(
    input: torch.Tensor,
    packed_lane_n64_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run K20/N64 pipe2 with single-CTA split-K for small M and large K/N shapes."""

    return _extension_api().op("amplin", "mma_lane_m16_n64_splitk20_pipe2_interleaved")(
        input,
        packed_lane_n64_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_splitk24_pipe2_interleaved(
    input: torch.Tensor,
    packed_lane_n64_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with single-CTA K24 split and shared-memory partials."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_splitk24_pipe2_interleaved")(
        input,
        packed_lane_n64_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_splitk12_pipe2_interleaved(
    input: torch.Tensor,
    packed_lane_n64_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with single-CTA K12 split and shared-memory partials (lower shared, higher occupancy)."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_splitk12_pipe2_interleaved")(
        input,
        packed_lane_n64_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_splitk16_pipe2_interleaved(
    input: torch.Tensor,
    packed_lane_n64_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with single-CTA K16 split and shared-memory partials (16 warps, 64 KiB partials)."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_splitk16_pipe2_interleaved")(
        input,
        packed_lane_n64_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m32_n64_splitk20_pipe2_interleaved(
    input: torch.Tensor,
    packed_lane_n64_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run M32 N64 with single-CTA K20 split and shared-memory partials (20 warps, 80 KiB partials)."""

    return _extension_api().op("amplin", "mma_lane_m32_n64_splitk20_pipe2_interleaved")(
        input,
        packed_lane_n64_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n64_splitk12x2_coop_interleaved(
    input: torch.Tensor,
    packed_lane_n64_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run two cooperative K12 CTAs per N64 tile with FP32 cross-CTA reduction."""

    return _extension_api().op("amplin", "mma_lane_m16_n64_splitk12x2_coop_interleaved")(
        input,
        packed_lane_n64_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n32_splitk16_pipe2(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the K12288 16-warp N32 path with two-stage register prefetching."""

    return _extension_api().op("amplin", "mma_lane_m16_n32_splitk16_pipe2")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


def mma_lane_m16_n16_splitk16(
    input: torch.Tensor,
    packed_lane_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    logical_n: int,
) -> torch.Tensor:
    """Run the K12288 16-warp intra-CTA split-K padded-M16 control."""

    return _extension_api().op("amplin", "mma_lane_m16_n16_splitk16")(
        input,
        packed_lane_qweight,
        packed_scales,
        logical_n,
    )


# Dynamic routing state -------------------------------------------------

# Key: (size_m, size_k, size_n, dtype_name)
# Value: selected kernel name (str)
_STATIC_ROUTING_TABLE: dict[tuple[int, int, int, str], str] = {}
_DYNAMIC_ROUTING_TABLE: dict[tuple[int, int, int, str], str] = {}
_ROUTING_LOCK = threading.Lock()

# Packed-weight caches are per-thread (see _ThreadLocalAmplinCaches), so no
# global cache lock is required.  Existing ``with _CACHE_LOCK:`` sites are kept
# as no-ops below to minimize churn; they can be removed in a future cleanup.
_CACHE_LOCK = contextlib.nullcontext()


@dataclass(frozen=True)
class _CandidateSpec:
    name: str
    fn_name: str
    packer: str
    needs_logical_n: bool = True
    min_m: int = 1
    max_m: int | None = None
    exact_ms: tuple[int, ...] | None = None
    m_multiple: int = 1
    n_multiple: int = 64
    k_multiple: int = 128
    exact_ks: tuple[int, ...] | None = None
    # Unique layout ID used for the packed-weight cache.  Defaults to the
    # packer name because kernels that share a packer share the same layout.
    layout_id: str = ""


# Full candidate list for M <= 32.  Each kernel is shape-dependent; the router
# picks the fastest legal one per (M, K, N, dtype).
_DYNAMIC_CANDIDATES: tuple[_CandidateSpec, ...] = (
    # GEMV-style paths ------------------------------------------------------
    _CandidateSpec("gemv", "gemv", "none", needs_logical_n=False, max_m=16, n_multiple=1),
    _CandidateSpec(
        "gemv_k12288_wide",
        "gemv_k12288_wide",
        "none",
        needs_logical_n=False,
        exact_ks=(12288,),
        max_m=1,
        n_multiple=16,
    ),
    _CandidateSpec(
        "gemv_multirow",
        "gemv_multirow",
        "none",
        needs_logical_n=False,
        exact_ms=(2, 4, 8, 16),
        exact_ks=(3072, 4096, 12288),
        n_multiple=16,
    ),
    # HMMA / WMMA paths ----------------------------------------------------
    _CandidateSpec("gemm_hmma", "gemm_hmma", "hmma", min_m=16, max_m=32, m_multiple=16),
    _CandidateSpec("gemm_hmma_v0", "gemm_hmma_v0", "hmma", min_m=16, max_m=32, m_multiple=16),
    _CandidateSpec(
        "gemm_hmma_m32_n128_pipeline4",
        "gemm_hmma_m32_n128_pipeline4",
        "hmma",
        min_m=17,
        max_m=32,
    ),
    # HMMA / WMMA large-M paths ---------------------------------------------
    _CandidateSpec(
        "gemm_hmma_m64_v3_large",
        "gemm_hmma_m64_v3",
        "hmma",
        min_m=64,
        max_m=None,
        m_multiple=64,
    ),
    # M <= 16, N16 / N32 mma_lane split-K paths -----------------------------
    _CandidateSpec(
        "mma_lane_m16_n16_padded", "mma_lane_m16_n16_padded", "lane", max_m=16, n_multiple=16
    ),
    _CandidateSpec(
        "mma_lane_m16_n16_splitk4", "mma_lane_m16_n16_splitk4", "lane", max_m=16, n_multiple=16, k_multiple=512
    ),
    _CandidateSpec(
        "mma_lane_m16_n16_splitk8", "mma_lane_m16_n16_splitk8", "lane", max_m=16, n_multiple=16, k_multiple=1024
    ),
    _CandidateSpec(
        "mma_lane_m16_n16_splitk12", "mma_lane_m16_n16_splitk12", "lane", max_m=16, n_multiple=16, k_multiple=1536
    ),
    _CandidateSpec(
        "mma_lane_m16_n16_splitk16", "mma_lane_m16_n16_splitk16", "lane", max_m=16, n_multiple=16, k_multiple=2048
    ),
    _CandidateSpec(
        "mma_lane_m16_n32_splitk8", "mma_lane_m16_n32_splitk8", "lane", max_m=16, n_multiple=32, k_multiple=1024
    ),
    _CandidateSpec(
        "mma_lane_m16_n32_splitk12", "mma_lane_m16_n32_splitk12", "lane", max_m=16, n_multiple=32, k_multiple=1536
    ),
    _CandidateSpec(
        "mma_lane_m16_n32_splitk16", "mma_lane_m16_n32_splitk16", "lane", max_m=16, n_multiple=32, k_multiple=2048
    ),
    _CandidateSpec(
        "mma_lane_m16_n32_splitk8_pipe2", "mma_lane_m16_n32_splitk8_pipe2", "lane", max_m=16, n_multiple=32, k_multiple=1024
    ),
    _CandidateSpec(
        "mma_lane_m16_n32_splitk12_pipe2", "mma_lane_m16_n32_splitk12_pipe2", "lane", max_m=16, n_multiple=32, k_multiple=1536
    ),
    _CandidateSpec(
        "mma_lane_m16_n32_splitk16_pipe2", "mma_lane_m16_n32_splitk16_pipe2", "lane", max_m=16, n_multiple=32, k_multiple=2048
    ),
    _CandidateSpec(
        "mma_lane_m16_n32_splitk12_pipe2_interleaved",
        "mma_lane_m16_n32_splitk12_pipe2_interleaved",
        "n32_interleaved",
        max_m=16,
        n_multiple=32,
        k_multiple=1536,
    ),
    # M <= 16, N64 mma_lane paths -------------------------------------------
    _CandidateSpec(
        "mma_lane_m16_n64_splitk24_pipe2_interleaved",
        "mma_lane_m16_n64_splitk24_pipe2_interleaved",
        "n64",
        max_m=16,
    ),
    _CandidateSpec(
        "mma_lane_m16_n64_splitk4_pipe2_interleaved",
        "mma_lane_m16_n64_splitk4_pipe2_interleaved",
        "n64",
        max_m=16,
        k_multiple=512,
    ),
    _CandidateSpec(
        "mma_lane_m16_n64_splitk8_pipe2_interleaved",
        "mma_lane_m16_n64_splitk8_pipe2_interleaved",
        "n64",
        max_m=16,
        k_multiple=1024,
    ),
    _CandidateSpec(
        "mma_lane_m16_n64_splitk12_pipe2_interleaved",
        "mma_lane_m16_n64_splitk12_pipe2_interleaved",
        "n64",
        max_m=16,
        k_multiple=1536,
    ),
    _CandidateSpec(
        "mma_lane_m16_n64_splitk16_pipe2_interleaved",
        "mma_lane_m16_n64_splitk16_pipe2_interleaved",
        "n64",
        max_m=16,
        k_multiple=2048,
    ),
    _CandidateSpec(
        "mma_lane_m16_n64_splitk20_pipe2_interleaved",
        "mma_lane_m16_n64_splitk20_pipe2_interleaved",
        "n64",
        max_m=16,
    ),
    _CandidateSpec(
        "mma_lane_m16_n64_splitk12x2_coop_interleaved",
        "mma_lane_m16_n64_splitk12x2_coop_interleaved",
        "n64",
        max_m=16,
    ),
    _CandidateSpec("mma_lane_m16_n64_shared_a", "mma_lane_m16_n64_shared_a", "n64", max_m=16),
    _CandidateSpec(
        "mma_lane_m16_n64_tile4_shared_a",
        "mma_lane_m16_n64_tile4_shared_a",
        "n64",
        max_m=16,
        n_multiple=256,
    ),
    _CandidateSpec(
        "mma_lane_m16_n64_tile8_shared_a",
        "mma_lane_m16_n64_tile8_shared_a",
        "n64",
        max_m=16,
        n_multiple=512,
    ),
    _CandidateSpec(
        "mma_lane_m16_n64_tile8_shared_a_large",
        "mma_lane_m16_n64_tile8_shared_a",
        "n64",
        min_m=64,
        max_m=None,
        m_multiple=16,
        n_multiple=512,
    ),
    # M32 mma_lane paths ----------------------------------------------------
    _CandidateSpec(
        "mma_lane_m32_global_a",
        "mma_lane_m32_global_a",
        "lane",
        exact_ms=(32,),
    ),
    _CandidateSpec(
        "mma_lane_m32_n32_global_a",
        "mma_lane_m32_n32_global_a",
        "lane",
        exact_ms=(32,),
        n_multiple=8,
    ),
    _CandidateSpec(
        "mma_lane_m32_global_a_large",
        "mma_lane_m32_global_a",
        "lane",
        min_m=32,
        max_m=None,
        m_multiple=32,
    ),
    _CandidateSpec(
        "mma_lane_m32_n32_global_a_large",
        "mma_lane_m32_n32_global_a",
        "lane",
        min_m=32,
        max_m=None,
        m_multiple=32,
        n_multiple=8,
    ),
    _CandidateSpec(
        "mma_lane_m64",
        "mma_lane_m64",
        "lane",
        min_m=64,
        max_m=None,
        m_multiple=64,
    ),
    _CandidateSpec(
        "mma_lane_m64_global_a",
        "mma_lane_m64_global_a",
        "lane",
        min_m=64,
        max_m=None,
        m_multiple=64,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_splitk24_pipe2_interleaved",
        "mma_lane_m32_n64_splitk24_pipe2_interleaved",
        "n64",
        min_m=17,
        max_m=32,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_splitk12_pipe2_interleaved",
        "mma_lane_m32_n64_splitk12_pipe2_interleaved",
        "n64",
        min_m=17,
        max_m=32,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_splitk16_pipe2_interleaved",
        "mma_lane_m32_n64_splitk16_pipe2_interleaved",
        "n64",
        min_m=17,
        max_m=32,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_splitk20_pipe2_interleaved",
        "mma_lane_m32_n64_splitk20_pipe2_interleaved",
        "n64",
        min_m=17,
        max_m=32,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_splitk12x2_coop_interleaved",
        "mma_lane_m32_n64_splitk12x2_coop_interleaved",
        "n64",
        max_m=32,
    ),
    _CandidateSpec("mma_lane_m32_n64_shared_a", "mma_lane_m32_n64_shared_a", "n64", max_m=32),
    _CandidateSpec(
        "mma_lane_m32_n64_tile2_shared_a",
        "mma_lane_m32_n64_tile2_shared_a",
        "n64",
        max_m=32,
        n_multiple=128,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_tile2_interleaved_dequant",
        "mma_lane_m32_n64_tile2_interleaved_dequant",
        "n64",
        min_m=17,
        max_m=32,
        n_multiple=128,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_tile4_shared_a",
        "mma_lane_m32_n64_tile4_shared_a",
        "n64",
        max_m=32,
        n_multiple=256,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_tile2_shared_a_large",
        "mma_lane_m32_n64_tile2_shared_a",
        "n64",
        min_m=64,
        max_m=None,
        m_multiple=32,
        n_multiple=128,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_tile4_shared_a_large",
        "mma_lane_m32_n64_tile4_shared_a",
        "n64",
        min_m=64,
        max_m=None,
        m_multiple=32,
        n_multiple=256,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_tile8_shared_a",
        "mma_lane_m32_n64_tile8_shared_a",
        "n64",
        max_m=32,
        n_multiple=512,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_tile8_shared_a_large",
        "mma_lane_m32_n64_tile8_shared_a",
        "n64",
        min_m=64,
        max_m=None,
        m_multiple=32,
        n_multiple=512,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_tile2_splitk2",
        "mma_lane_m32_n64_tile2_splitk2",
        "n64",
        min_m=17,
        max_m=32,
        n_multiple=128,
        k_multiple=256,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_tile1_splitk4",
        "mma_lane_m32_n64_tile1_splitk4",
        "n64",
        min_m=17,
        max_m=32,
        n_multiple=64,
        k_multiple=512,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_tile1_splitk8",
        "mma_lane_m32_n64_tile1_splitk8",
        "n64",
        min_m=17,
        max_m=32,
        n_multiple=64,
        k_multiple=1024,
    ),
    _CandidateSpec(
        "mma_lane_m32_n64_tile2_splitk4",
        "mma_lane_m32_n64_tile2_splitk4",
        "n64",
        min_m=17,
        max_m=32,
        n_multiple=128,
        k_multiple=512,
    ),
    # Marlin-style N256/8-warp cp.async pipeline with register-fragment LOP3
    # dequantization.  This candidate is routed to the existing Marlin GEMM on
    # large-M shapes where that schedule currently wins.
    _CandidateSpec(
        "marlin_style",
        "marlin_style",
        "none",
        min_m=1,
        max_m=None,
        m_multiple=1,
        n_multiple=64,
        k_multiple=128,
        layout_id="marlin_style",
    ),
)


# Sentinel used in the weight cache for the canonical (un-packed) "none" layout
# so the original qweight tensor is not duplicated when spilling to CPU RAM.
_NONE_WEIGHT_SENTINEL = object()

# Maximum number of packed layouts that stay GPU-resident for one layer.  Inactive
# layouts are spilled to host RAM and moved back on demand.
_MAX_GPU_PACKED_LAYOUTS_PER_LAYER = 1


@dataclass
class _PendingCpuLayout:
    """In-flight GPU->CPU weight transfer for native Amplin layouts."""
    cpu_tensors: tuple[object, torch.Tensor]
    gpu_tensors: tuple[object, torch.Tensor]
    event: torch.cuda.Event
    stream: torch.cuda.Stream


@dataclass
class _PendingCpuMarlinLayout:
    """In-flight GPU->CPU weight transfer for Marlin-style layouts."""
    cpu_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]
    gpu_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]
    event: torch.cuda.Event
    stream: torch.cuda.Stream


@dataclass
class _ThreadLocalAmplinCaches:
    """Per-thread packed-weight and op caches.

    Normal inference runs a given module forward on a fixed thread, so each
    thread can own its own layout cache without cross-thread synchronization.
    """
    weight_cache: dict = field(default_factory=dict)
    weight_cache_pending: dict = field(default_factory=dict)
    weight_cache_resident: dict = field(default_factory=dict)
    weight_copy_streams: dict = field(default_factory=dict)
    fast_dispatch_cache: dict = field(default_factory=dict)
    marlin_pack_cache: dict = field(default_factory=dict)
    marlin_pack_cache_pending: dict = field(default_factory=dict)
    marlin_pack_cache_resident: dict = field(default_factory=dict)
    marlin_copy_streams: dict = field(default_factory=dict)
    candidate_op_cache: dict = field(default_factory=dict)
    marlin_gemm_op_cache: dict = field(default_factory=dict)
    marlin_available_cache: dict = field(default_factory=dict)

    def clear(self) -> None:
        """Release every tensor/op reference held by this thread's caches."""
        self.weight_cache.clear()
        self.weight_cache_pending.clear()
        self.weight_cache_resident.clear()
        self.weight_copy_streams.clear()
        self.fast_dispatch_cache.clear()
        self.marlin_pack_cache.clear()
        self.marlin_pack_cache_pending.clear()
        self.marlin_pack_cache_resident.clear()
        self.marlin_copy_streams.clear()
        self.candidate_op_cache.clear()
        self.marlin_gemm_op_cache.clear()
        self.marlin_available_cache.clear()


_THREAD_LOCAL_CACHES = threading.local()


def _thread_caches() -> _ThreadLocalAmplinCaches:
    """Return (creating if necessary) this thread's isolated cache set."""
    caches = getattr(_THREAD_LOCAL_CACHES, "caches", None)
    if caches is None:
        caches = _ThreadLocalAmplinCaches()
        _THREAD_LOCAL_CACHES.caches = caches
        # Thread-local values are not released when the thread dies.  Attach a
        # finalizer to the Thread object so GPU memory is dropped as soon as the
        # thread is collected (for long-lived inference threads this never fires).
        weakref.finalize(threading.current_thread(), caches.clear)
    return caches


@dataclass
class _OriginalWeightResidency:
    """Global per-layer residency manager for the canonical GPTQ qweight/scales.

    Amplin keeps at most one execution representation of a layer in GPU VRAM:
    either the original `qweight`/`scales` (when the ``none``/gemv layout is
    active) or one packed layout.  After a non-``none`` layout is produced, the
    original weights are copied to CPU RAM and the GPU copy is dropped once no
    kernel is still reading it.

    This object is shared by all threads, so all mutations of `qweight.data`
    and `scales.data` are protected by `lock`.  Active kernels that read the
    original weights register a CUDA event so their GPU tensor stays alive even
    if another thread moves the module's `qweight`/`scales` to CPU in the
    meantime.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    cpu_qweight: torch.Tensor | None = None
    cpu_scales: torch.Tensor | None = None
    # Each tuple is (event, gpu_qweight, gpu_scales) for an in-flight kernel
    # that reads the canonical weights.  Completed events are reaped by
    # `_finalize_original_gpu_users`.
    gpu_users: list[tuple[torch.cuda.Event, torch.Tensor, torch.Tensor]] = field(
        default_factory=list
    )


# Maps (id(qweight), id(scales)) to the shared residency record.
_ORIGINAL_WEIGHT_RESIDENCY: dict[tuple[int, int], _OriginalWeightResidency] = {}
_ORIGINAL_RESIDENCY_LOCK = threading.Lock()


def _original_weight_residency(
    qweight: torch.Tensor, scales: torch.Tensor
) -> _OriginalWeightResidency:
    """Return (creating if necessary) the shared residency record for a layer."""
    key = (id(qweight), id(scales))
    with _ORIGINAL_RESIDENCY_LOCK:
        r = _ORIGINAL_WEIGHT_RESIDENCY.get(key)
        if r is None:
            r = _OriginalWeightResidency()
            _ORIGINAL_WEIGHT_RESIDENCY[key] = r
    return r


def _finalize_original_gpu_users(r: _OriginalWeightResidency) -> None:
    """Drop GPU tensor references whose kernels have finished."""
    still_active: list[tuple[torch.cuda.Event, torch.Tensor, torch.Tensor]] = []
    for event, gq, gs in r.gpu_users:
        if event.query():
            del gq, gs
            continue
        still_active.append((event, gq, gs))
    r.gpu_users = still_active


def _ensure_original_in_vram(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    device: torch.device,
    move_to_cpu: bool = True,
    return_tensors: bool = True,
) -> tuple[_OriginalWeightResidency, torch.Tensor | None, torch.Tensor | None]:
    """Move/restore canonical ``qweight``/``scales`` storage and return GPU views.

    For non-``none`` layouts (``move_to_cpu=True``) the canonical weights are kept
    in CPU RAM and temporary GPU copies are returned for packing.  For
    ``none``/gemv layouts (``move_to_cpu=False``) the canonical weights are kept in
    GPU RAM and a detached view is returned so the gemv kernel can reference
    them safely while another thread may later move the canonical storage.

    The whole operation is performed under the per-layer residency lock so the
    returned GPU tensors are captured before another thread can change the
    canonical storage location.

    During ``torch.compile`` tracing / CUDA Graph capture this function returns GPU
    copies without moving the canonical tensors, because the graph expects its
    inputs to stay on the original device.
    """
    r = _original_weight_residency(qweight, scales)
    with r.lock:
        _finalize_original_gpu_users(r)

        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            if return_tensors:
                return (
                    r,
                    qweight.to(device, non_blocking=False).detach().contiguous(),
                    scales.to(device, non_blocking=False).detach().contiguous(),
                )
            return r, None, None

        if not move_to_cpu:
            # gemv consumes the original weights directly; keep them GPU-resident.
            if qweight.device.type != "cuda" or scales.device.type != "cuda":
                if r.cpu_qweight is not None and r.cpu_scales is not None:
                    qweight.data = r.cpu_qweight.to(device, non_blocking=False).contiguous()
                    scales.data = r.cpu_scales.to(device, non_blocking=False).contiguous()
                else:
                    qweight.data = qweight.to(device, non_blocking=False).contiguous()
                    scales.data = scales.to(device, non_blocking=False).contiguous()
            if return_tensors:
                return r, qweight.detach().contiguous(), scales.detach().contiguous()
            return r, None, None

        if r.cpu_qweight is None or r.cpu_scales is None:
            # First packed use of this layer: move the canonical weights to CPU.
            # The original GPU storage is released once no in-flight kernel holds it.
            if qweight.device.type == "cuda":
                r.cpu_qweight = qweight.to("cpu", non_blocking=False).contiguous()
                r.cpu_scales = scales.to("cpu", non_blocking=False).contiguous()
            else:
                r.cpu_qweight = qweight.detach().contiguous()
                r.cpu_scales = scales.detach().contiguous()
            qweight.data = r.cpu_qweight
            scales.data = r.cpu_scales
        elif qweight.device.type == "cuda":
            # Another call already canonicalized; point this Parameter at CPU home.
            qweight.data = r.cpu_qweight
            scales.data = r.cpu_scales

        if return_tensors:
            return (
                r,
                r.cpu_qweight.to(device, non_blocking=False).detach().contiguous(),
                r.cpu_scales.to(device, non_blocking=False).detach().contiguous(),
            )
        return r, None, None


def _register_original_gpu_user(
    r: _OriginalWeightResidency,
    event: torch.cuda.Event,
    gpu_qweight: torch.Tensor,
    gpu_scales: torch.Tensor,
) -> None:
    """Record an in-flight kernel that reads a temporary GPU copy of the
    canonical weights.  Completed events are reaped when new copies are requested."""
    with r.lock:
        _finalize_original_gpu_users(r)
        r.gpu_users.append((event, gpu_qweight, gpu_scales))


def _layout_id_for_spec(spec: _CandidateSpec) -> str:
    """Return the cache layout ID for a candidate; defaults to its packer."""
    return spec.layout_id or spec.packer


def _weight_cache_layer_key(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], torch.dtype]:
    """Layer key shared by all packed layouts for one set of weights.

    Use object identity (`id`) rather than `data_ptr` because the residency
    manager may move the canonical `qweight`/`scales` between CPU and GPU,
    changing their data pointer while the logical layer stays the same.
    """
    return (
        id(qweight),
        id(scales),
        tuple(qweight.shape),
        tuple(scales.shape),
        dtype,
    )


def _get_weight_copy_stream(device: torch.device) -> torch.cuda.Stream:
    """Lazily create one GPU->CPU copy stream per device."""
    with _CACHE_LOCK:
        stream = _thread_caches().weight_copy_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            _thread_caches().weight_copy_streams[device] = stream
        return stream


def _finalize_weight_cache_copies() -> None:
    """Move completed GPU->CPU weight transfers from pending to CPU cache.

    Querying the recorded event tells us the copy has finished and the GPU
    source memory is no longer needed by the copy engine.  This is polled on
    every ``dynamic()`` call so cold hits see valid CPU tensors and completed
    transfers release their source GPU memory promptly.
    """
    with _CACHE_LOCK:
        completed = [key for key, pending in _thread_caches().weight_cache_pending.items() if pending.event.query()]
        for key in completed:
            pending = _thread_caches().weight_cache_pending.pop(key)
            _thread_caches().weight_cache[key] = pending.cpu_tensors
            # Release the source GPU tensors now; the copy engine is done with them.
            pending.gpu_tensors = None


def _evict_weight_layout_to_cpu_async(
    key: tuple,
    device: torch.device,
) -> None:
    """Kick off an asynchronous GPU -> CPU copy for one cached packed layout.

    The copy is issued on the per-device copy stream.  The source GPU tensors
    remain referenced in ``_thread_caches().weight_cache_pending`` until the recorded event
    completes, at which point ``_finalize_weight_cache_copies`` drops them.
    """
    if key not in _thread_caches().weight_cache:
        return
    q, s = _thread_caches().weight_cache.pop(key)
    is_gpu = (q is not _NONE_WEIGHT_SENTINEL and q.device.type == "cuda") or (
        q is _NONE_WEIGHT_SENTINEL and s.device.type == "cuda"
    )
    if not is_gpu:
        # Already CPU-resident; put it back and skip the async copy.
        _thread_caches().weight_cache[key] = (q, s)
        return
    stream = _get_weight_copy_stream(device)
    # Capture the default stream state before we switch to the copy stream so
    # the copy waits for any kernels (e.g. packing ops) that produced this tensor.
    default_event = torch.cuda.Event()
    default_event.record()
    with torch.cuda.stream(stream):
        stream.wait_event(default_event)
        if q is _NONE_WEIGHT_SENTINEL:
            cpu_q = _NONE_WEIGHT_SENTINEL
            gpu_q = _NONE_WEIGHT_SENTINEL
            cpu_s = s.to("cpu", non_blocking=True)
        else:
            cpu_q = q.to("cpu", non_blocking=True)
            gpu_q = q
            cpu_s = s.to("cpu", non_blocking=True)
    event = stream.record_event()
    _thread_caches().weight_cache_pending[key] = _PendingCpuLayout(
        cpu_tensors=(cpu_q, cpu_s),
        gpu_tensors=(gpu_q, s),
        event=event,
        stream=stream,
    )


def _move_weight_layout_to_gpu(
    key: tuple,
    qweight: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bring a spilled (CPU) or in-flight packed layout back to GPU.

    If a GPU->CPU copy is still in flight for this key, we first synchronize
    its recorded event so the CPU tensor is fully populated before we copy it
    back to GPU.  This is the "cold hit" path the user asked about.
    """
    pending = _thread_caches().weight_cache_pending.pop(key, None)
    if pending is not None:
        pending.event.synchronize()
        cpu_q, cpu_s = pending.cpu_tensors
        # After synchronization the source GPU tensors are no longer needed by
        # the copy engine; drop them explicitly so the pending object can be GC'd.
        pending.gpu_tensors = None
    else:
        cpu_q, cpu_s = _thread_caches().weight_cache.pop(key)

    if cpu_q is _NONE_WEIGHT_SENTINEL:
        # The ``none`` layout references the canonical qweight; keep the sentinel
        # so the caller fetches a fresh GPU view via the residency manager.
        moved = (_NONE_WEIGHT_SENTINEL, cpu_s.to(device, non_blocking=False))
    else:
        moved = (
            cpu_q.to(device, non_blocking=False),
            cpu_s.to(device, non_blocking=False),
        )
    _thread_caches().weight_cache[key] = moved
    return moved


def _evict_other_layer_layouts(
    layer_key: tuple,
    keep_key: tuple,
    device: torch.device,
) -> None:
    """Spill every GPU-resident layout for ``layer_key`` except ``keep_key`` to CPU.

    This scans both native and Marlin caches (and pending transfers) and is the
    mechanism that enforces the single-GPU-representation invariant per layer.
    """
    caches = _thread_caches()
    prefix_len = len(layer_key)

    # Native weight cache.
    for key in list(caches.weight_cache.keys()):
        if key[:prefix_len] == layer_key and key != keep_key:
            _evict_weight_layout_to_cpu_async(key, device)
    for key in list(caches.weight_cache_pending.keys()):
        if key[:prefix_len] == layer_key and key != keep_key:
            pending = caches.weight_cache_pending.pop(key)
            pending.event.synchronize()
            caches.weight_cache[key] = pending.cpu_tensors
            pending.gpu_tensors = None

    caches.weight_cache_resident[layer_key] = (
        {keep_key}
        if keep_key in caches.weight_cache or keep_key in caches.weight_cache_pending
        else set()
    )

    # Marlin pack cache.
    for key in list(caches.marlin_pack_cache.keys()):
        if key[:prefix_len] == layer_key and key != keep_key:
            _evict_marlin_pack_to_cpu_async(key, device)
    for key in list(caches.marlin_pack_cache_pending.keys()):
        if key[:prefix_len] == layer_key and key != keep_key:
            pending = caches.marlin_pack_cache_pending.pop(key)
            pending.event.synchronize()
            caches.marlin_pack_cache[key] = pending.cpu_tensors
            pending.gpu_tensors = None

    caches.marlin_pack_cache_resident[layer_key] = (
        {keep_key}
        if keep_key in caches.marlin_pack_cache or keep_key in caches.marlin_pack_cache_pending
        else set()
    )


def _make_layout_resident(
    layer_key: tuple,
    key: tuple,
    device: torch.device,
    evict: bool = True,
) -> None:
    """Mark ``key`` as the active GPU-resident layout for ``layer_key``.

    If ``evict`` is true, every other layout for the same layer is spilled to
    CPU RAM so at most one packed representation stays in VRAM.
    """
    caches = _thread_caches()
    if key in caches.weight_cache or key in caches.weight_cache_pending:
        caches.weight_cache_resident.setdefault(layer_key, set()).add(key)
    elif key in caches.marlin_pack_cache or key in caches.marlin_pack_cache_pending:
        caches.marlin_pack_cache_resident.setdefault(layer_key, set()).add(key)
    if not evict:
        return
    _evict_other_layer_layouts(layer_key, key, device)


def _pack_with_cache(
    spec: _CandidateSpec,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    evict: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GPU-resident packed weights for ``spec``, spilling old layouts to CPU.

    The eviction path is asynchronous: the transfer runs on a dedicated CUDA
    copy stream and the source GPU tensors are released once the recorded event
    reports completion.  Cold hits synchronize the event before moving the cached
    CPU copy back to GPU.

    Only one packed layout per layer stays GPU-resident; inactive layouts in either
    the native or the Marlin cache are spilled to CPU RAM.

    The canonical `qweight`/`scales` are stored in CPU RAM the first time this
    layer is touched; each call creates temporary GPU copies for packing or for
    the ``none``/gemv kernels.  Those temporary copies are held alive by CUDA
    events until the consuming kernel finishes, so only the packed layout remains
    in VRAM.

    The caches are per-thread, so calls from different threads are isolated.
    """
    with _CACHE_LOCK:
        layout_id = _layout_id_for_spec(spec)
        layer_key = _weight_cache_layer_key(qweight, scales, dtype)
        key = (*layer_key, layout_id)

        # Complete any outstanding copies so CPU tensors become valid and source
        # GPU memory is released before we allocate more.
        _finalize_weight_cache_copies()

        if key in _thread_caches().weight_cache or key in _thread_caches().weight_cache_pending:
            packed = _move_weight_layout_to_gpu(key, qweight, device)
            _make_layout_resident(layer_key, key, device, evict=evict)
            if packed[0] is _NONE_WEIGHT_SENTINEL:
                # gemv-style kernels consume the original weights directly.  Avoid the
                # residency-manager overhead on the steady-state path by using the
                # GPU-resident canonical tensors as long as they are already on the
                # target device; otherwise restore them once and keep them there.
                if qweight.device.type != "cuda" or scales.device.type != "cuda":
                    _ensure_original_in_vram(qweight, scales, device, move_to_cpu=False, return_tensors=False)
                return (qweight.detach(), packed[1].to(device, non_blocking=False))
            # A non-``none`` layout is resident; keep the canonical weights on CPU.
            _ensure_original_in_vram(qweight, scales, device, move_to_cpu=True, return_tensors=False)
            return packed

        if layout_id == "none":
            # The "none" layout does not repack; keep the original qweight/scales
            # GPU-resident and cache only the dtype-converted scales tensor.
            if qweight.device.type == "cuda" and scales.device.type == "cuda":
                gpu_q, gpu_s = qweight, scales
            else:
                _, gpu_q, gpu_s = _ensure_original_in_vram(
                    qweight, scales, device, move_to_cpu=False
                )
            scales_t = gpu_s.to(dtype).contiguous().detach()
            packed = (_NONE_WEIGHT_SENTINEL, scales_t)
            _thread_caches().weight_cache[key] = packed
            _make_layout_resident(layer_key, key, device, evict=evict)
            return (gpu_q.detach(), scales_t)

        # Need temporary GPU copies of the canonical weights to create a layout.
        r, gpu_q, gpu_s = _ensure_original_in_vram(
            qweight, scales, device, move_to_cpu=True
        )
        if spec.packer == "hmma":
            packed = (pack_hmma_qweight(gpu_q), pack_hmma_scales(gpu_s.to(dtype)))
        elif spec.packer == "lane":
            packed = (pack_mma_lane_qweight(gpu_q), pack_hmma_scales(gpu_s.to(dtype)))
        elif spec.packer == "n32_interleaved":
            packed = (pack_mma_lane_n32_qweight(gpu_q), pack_hmma_scales(gpu_s.to(dtype)))
        elif spec.packer == "n64":
            packed = (pack_mma_lane_n64_qweight(gpu_q), pack_hmma_scales(gpu_s.to(dtype)))
        else:
            raise ValueError(f"Unknown packer {spec.packer!r}")
        # The packer just read these temporary GPU copies; keep them alive until
        # the packer kernel finishes, then they are freed and only the packed
        # layout remains in VRAM.
        event = torch.cuda.Event()
        event.record()
        _register_original_gpu_user(r, event, gpu_q, gpu_s)
        _thread_caches().weight_cache[key] = packed
        _make_layout_resident(layer_key, key, device, evict=evict)

        return packed


def _get_marlin_copy_stream(device: torch.device) -> torch.cuda.Stream:
    """Lazily create one GPU->CPU copy stream per device for Marlin layouts."""
    with _CACHE_LOCK:
        stream = _thread_caches().marlin_copy_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            _thread_caches().marlin_copy_streams[device] = stream
        return stream


def _finalize_marlin_cache_copies() -> None:
    """Move completed GPU->CPU Marlin transfers from pending to CPU cache."""
    with _CACHE_LOCK:
        completed = [key for key, pending in _thread_caches().marlin_pack_cache_pending.items() if pending.event.query()]
        for key in completed:
            pending = _thread_caches().marlin_pack_cache_pending.pop(key)
            _thread_caches().marlin_pack_cache[key] = pending.cpu_tensors
            pending.gpu_tensors = None


def _evict_marlin_pack_to_cpu_async(
    key: tuple,
    device: torch.device,
) -> None:
    """Kick off an async GPU -> CPU copy for one Marlin packed layout."""
    if key not in _thread_caches().marlin_pack_cache:
        return
    q, s, w, b = _thread_caches().marlin_pack_cache.pop(key)
    if q.device.type != "cuda":
        # Already CPU-resident; put it back and skip the async copy.
        _thread_caches().marlin_pack_cache[key] = (q, s, w, b)
        return
    stream = _get_marlin_copy_stream(device)
    # Ensure the copy waits for any default-stream kernels that produced these tensors.
    default_event = torch.cuda.Event()
    default_event.record()
    with torch.cuda.stream(stream):
        stream.wait_event(default_event)
        cpu_q = q.to("cpu", non_blocking=True)
        cpu_s = s.to("cpu", non_blocking=True)
        cpu_w = w.to("cpu", non_blocking=True)
    event = stream.record_event()
    _thread_caches().marlin_pack_cache_pending[key] = _PendingCpuMarlinLayout(
        cpu_tensors=(cpu_q, cpu_s, cpu_w, b),
        gpu_tensors=(q, s, w, b),
        event=event,
        stream=stream,
    )


def _move_marlin_pack_to_gpu(
    key: tuple,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
    """Bring a spilled or in-flight Marlin layout back to GPU, syncing first."""
    pending = _thread_caches().marlin_pack_cache_pending.pop(key, None)
    if pending is not None:
        pending.event.synchronize()
        cpu_tensors = pending.cpu_tensors
        pending.gpu_tensors = None
    else:
        cpu_tensors = _thread_caches().marlin_pack_cache.pop(key)
    moved = tuple(
        t.to(device, non_blocking=False) if isinstance(t, torch.Tensor) else t
        for t in cpu_tensors
    )
    _thread_caches().marlin_pack_cache[key] = moved
    return moved


def _get_marlin_packed(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    size_n: int,
    dtype: torch.dtype,
    device: torch.device,
    evict: bool = True,
    cache: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
    """Repack canonical GPTQ weights/scales into Marlin's execution layout.

    The cache is GPU-resident by default; inactive Marlin layouts for a layer
    are spilled to CPU RAM asynchronously on a dedicated copy stream.  The
    canonical weights are kept in CPU RAM and temporary GPU copies are created
    for repacking.  Those temporary copies are held alive by a CUDA event until
    the repack kernel finishes, leaving only the Marlin-packed tensors in VRAM.

    When ``cache=False`` (used during micro-benchmarking) the packed tensors are
    returned without entering the persistent ``marlin_pack_cache``.
    """
    with _CACHE_LOCK:
        layer_key = _weight_cache_layer_key(qweight, scales, dtype)
        key = (*layer_key, size_n, _dtype_name(dtype))

        _finalize_marlin_cache_copies()

        if cache and (key in _thread_caches().marlin_pack_cache or key in _thread_caches().marlin_pack_cache_pending):
            packed = _move_marlin_pack_to_gpu(key, device)
            _ensure_original_in_vram(qweight, scales, device, move_to_cpu=True, return_tensors=False)
            _make_layout_resident(layer_key, key, device, evict=evict)
            return packed

        # Need temporary GPU copies of the canonical weights for repacking.
        r, gpu_q, gpu_s = _ensure_original_in_vram(qweight, scales, device)
        size_k = gpu_q.size(0) * 8
        perm = torch.empty(0, dtype=torch.int, device=device)
        marlin_qweight = gptq_marlin_repack(
            gpu_q.contiguous(),
            perm,
            size_k=size_k,
            size_n=size_n,
            num_bits=4,
            dtype=dtype,
        )
        marlin_scales = marlin_permute_scales(
            gpu_s.to(dtype).contiguous(),
            size_k=size_k,
            size_n=size_n,
            group_size=128,
        )
        workspace = marlin_make_workspace_new(device)
        b_q_type = scalar_types.uint4b8
        packed = (marlin_qweight, marlin_scales, workspace, b_q_type)
        if cache:
            _thread_caches().marlin_pack_cache[key] = packed

        # Keep the temporary GPU copies alive until the repack finishes.
        event = torch.cuda.Event()
        event.record()
        _register_original_gpu_user(r, event, gpu_q, gpu_s)

        _make_layout_resident(layer_key, key, device, evict=evict)

        if not torch.compiler.is_compiling() and not torch.cuda.is_current_stream_capturing():
            # Wait for the repack kernels (and any async CPU->GPU restore in
            # _make_layout_resident) to finish, then release the temporary
            # canonical-weight copies.  Only the Marlin-packed tensors should
            # remain in VRAM after this.
            torch.cuda.synchronize(device)
            _finalize_original_gpu_users(r)

        return packed


def _get_marlin_gemm_op(dtype: torch.dtype) -> Callable:
    caches = _thread_caches()
    op = caches.marlin_gemm_op_cache.get(dtype)
    if op is None:
        op_name = "gptq_marlin_gemm_fp16" if dtype == torch.float16 else "gptq_marlin_gemm_bf16"
        with _AMPLIN_INIT_LOCK:
            op = _marlin_resolve_op(dtype=dtype, op_name=op_name)
        caches.marlin_gemm_op_cache[dtype] = op
    return op


def _marlin_available_cached(dtype: torch.dtype) -> bool:
    caches = _thread_caches()
    cached = caches.marlin_available_cache.get(dtype)
    if cached is None:
        cached = marlin_runtime_available(dtype)
        caches.marlin_available_cache[dtype] = cached
    return cached


def _get_marlin_style_fast_runner(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    size_n: int,
    dtype: torch.dtype,
) -> Callable[..., torch.Tensor]:
    """Return a closure that runs the C++ marlin_style fast path.

    The closure re-fetches the Marlin packed layout from the cache on each call
    so the cache can migrate it between GPU and CPU between requests.
    """
    cpp_op = _get_amplin_op("marlin_style_run")
    size_k = qweight.size(0) * 8

    def _marlin_style_fast_run(
        input: torch.Tensor, *_args: object, **_kwargs: object
    ) -> torch.Tensor:
        marlin_qweight, marlin_scales, workspace, b_q_type = _get_marlin_packed(
            qweight, scales, size_n, dtype, input.device
        )
        return cpp_op(
            input,
            marlin_qweight,
            marlin_scales,
            workspace,
            b_q_type.id,
            size_n,
            size_k,
        )

    return _marlin_style_fast_run


def _run_marlin_style(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    logical_n: int,
    evict: bool = True,
) -> torch.Tensor:
    size_k = input.size(-1)
    size_m = input.numel() // size_k
    size_n = logical_n
    device = input.device
    input_2d = input.reshape(-1, size_k).contiguous()
    marlin_qweight, marlin_scales, workspace, b_q_type = _get_marlin_packed(
        qweight, scales, size_n, input.dtype, device, evict=evict
    )
    op = _get_marlin_gemm_op(input.dtype)
    output = op(
        input_2d,
        None,
        marlin_qweight,
        None,
        marlin_scales,
        None,
        None,
        None,
        None,
        workspace,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        True,
        False,
        True,
        False,
        False,
        0,
    )
    if input.dim() == 2:
        return output
    return output.reshape(*input.shape[:-1], size_n)


def _dtype_name(dtype: torch.dtype) -> str:
    return "fp16" if dtype == torch.float16 else "bf16"


def _dynamic_key(input: torch.Tensor, logical_n: int) -> tuple[int, int, int, str]:
    size_k = input.size(-1)
    size_m = input.numel() // size_k
    return (size_m, size_k, logical_n, _dtype_name(input.dtype))


def _is_candidate_legal(spec: _CandidateSpec, size_m: int, size_k: int, size_n: int) -> bool:
    if spec.exact_ms is not None and size_m not in spec.exact_ms:
        return False
    if size_m < spec.min_m or (spec.max_m is not None and size_m > spec.max_m):
        return False
    if spec.m_multiple > 1 and size_m % spec.m_multiple != 0:
        return False
    if spec.exact_ks is not None and size_k not in spec.exact_ks:
        return False
    if size_k % spec.k_multiple != 0 or size_n % spec.n_multiple != 0:
        return False
    return True


def _run_op_with_original(
    spec: _CandidateSpec,
    op: Callable,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    packed: tuple[torch.Tensor, torch.Tensor],
    size_n: int,
) -> torch.Tensor:
    """Run ``op`` on ``packed`` and record a GPU-user event for ``none`` layouts.

    The ``none``/gemv kernels consume the original `qweight` directly, so we must
    keep the temporary GPU copy alive until the kernel finishes even if another
    thread requests a different layout for the same layer.
    """
    layout_id = _layout_id_for_spec(spec)
    if spec.needs_logical_n:
        output = op(input, *packed, logical_n=size_n)
    else:
        output = op(input, *packed)
    if layout_id == "none":
        # gemv-style kernels use the original GPU-resident weights.  The converted
        # scales tensor is kept alive by the weight cache, and the caller/module
        # keeps the original qweight alive, so skip the per-call event overhead.
        return output
    return output


@dataclass(frozen=True)
class _KernelDispatch:
    """Prepared kernel invocation state returned by the packing pipeline."""

    op: Callable[..., torch.Tensor]
    packed: tuple[torch.Tensor, ...]
    name: str
    needs_logical_n: bool
    is_marlin_style: bool
    size_n: int
    size_k: int
    min_m: int
    max_m: int | None
    m_multiple: int
    exact_ms: tuple[int, ...] | None
    packer: str
    spec: _CandidateSpec


@dataclass(frozen=True)
class _KernelDispatchFamily:
    """All micro-kernels that share the same packed-weight layout for one layer.

    ``post_init`` pre-builds this family so ``forward`` can switch from the
    small-M decode kernel to the largest-M family member that fits a prefill
    batch without repacking the weights.
    """

    packed: tuple[torch.Tensor, ...]
    layout_id: str
    members: tuple[_KernelDispatch, ...]
    default: _KernelDispatch


def _call_kernel(dispatch: _KernelDispatch, input: torch.Tensor) -> torch.Tensor:
    """Launch the prepared kernel with the correct positional/keyword arguments."""
    if dispatch.is_marlin_style:
        return dispatch.op(input, *dispatch.packed, dispatch.size_n, dispatch.size_k)
    if dispatch.needs_logical_n:
        return dispatch.op(input, *dispatch.packed, logical_n=dispatch.size_n)
    return dispatch.op(input, *dispatch.packed)


def _dispatch_supports_batch(dispatch: _KernelDispatch, batch: int) -> bool:
    """Return whether ``dispatch`` can consume ``batch`` rows in one call."""
    if dispatch.exact_ms is not None and batch not in dispatch.exact_ms:
        return False
    if batch < dispatch.min_m or (dispatch.max_m is not None and batch > dispatch.max_m):
        return False
    if batch % dispatch.m_multiple != 0:
        return False
    return True


def _select_family_member(family: _KernelDispatchFamily, batch: int) -> _KernelDispatch | None:
    """Pick the largest-M family member that can consume ``batch`` in one call.

    Preference is: larger ``max_m`` (``None`` is largest), larger ``m_multiple``
    (coarser M tile), then larger ``n_multiple`` (fewer N blocks).  If no member
    supports the full batch, ``None`` is returned so the caller can fall back to
    chunking with the decode dispatch.
    """
    for member in family.members:
        if _dispatch_supports_batch(member, batch):
            return member
    return None


def _next_valid_batch(dispatch: _KernelDispatch, batch: int) -> int:
    """Return the smallest valid batch size ``>= batch`` for ``dispatch``.

    The caller must ensure ``batch`` does not exceed ``dispatch.max_m``.
    """
    if dispatch.exact_ms is not None:
        valid = [m for m in dispatch.exact_ms if m >= batch]
        if valid:
            return min(valid)
        return max(dispatch.exact_ms)

    if batch < dispatch.min_m:
        batch = dispatch.min_m
    if batch % dispatch.m_multiple != 0:
        batch += dispatch.m_multiple - (batch % dispatch.m_multiple)
    if dispatch.max_m is not None and batch > dispatch.max_m:
        batch = dispatch.max_m - (dispatch.max_m % dispatch.m_multiple)
    return batch


def _call_kernel_chunked(dispatch: _KernelDispatch, input: torch.Tensor) -> torch.Tensor:
    """Launch ``dispatch`` on ``input``, chunking if the batch exceeds ``max_m``."""
    batch = input.size(0)
    if _dispatch_supports_batch(dispatch, batch):
        return _call_kernel(dispatch, input)

    max_m = dispatch.max_m
    if max_m is None:
        if dispatch.exact_ms:
            max_m = max(dispatch.exact_ms)
        else:
            raise RuntimeError(
                f"Amplin kernel {dispatch.name!r} has no max_m/exact_ms and cannot chunk batch {batch}"
            )

    outputs: list[torch.Tensor] = []
    start = 0
    remaining = batch
    while remaining > 0:
        # Pick the largest valid chunk size not exceeding ``max_m``.
        chunk_size = min(remaining, max_m)
        while chunk_size >= dispatch.min_m and not _dispatch_supports_batch(dispatch, chunk_size):
            chunk_size -= 1

        if chunk_size < dispatch.min_m:
            # Pad the final/in-between remainder up to the next valid size.
            chunk_size = _next_valid_batch(dispatch, remaining)
            tail = input[start:]
            pad_rows = chunk_size - tail.size(0)
            pad = torch.zeros(
                (pad_rows, tail.size(1)),
                dtype=tail.dtype,
                device=tail.device,
            )
            padded = torch.cat([tail, pad], dim=0)
            output = _call_kernel(dispatch, padded)
            outputs.append(output[: tail.size(0)])
            break

        outputs.append(_call_kernel(dispatch, input[start : start + chunk_size]))
        start += chunk_size
        remaining -= chunk_size

    return torch.cat(outputs, dim=0)


def _build_kernel_dispatch(
    spec: _CandidateSpec,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    size_n: int,
    size_k: int,
    cache: bool = True,
) -> _KernelDispatch:
    """Pack weights for ``spec`` and return everything needed to launch the kernel."""
    if spec.fn_name == "marlin_style":
        op = _get_amplin_op("marlin_style_run")
    else:
        op = _get_amplin_op(spec.fn_name)
    packed = _pack_for_spec(spec, qweight, scales, dtype, device, cache=cache)
    return _KernelDispatch(
        op=op,
        packed=packed,
        name=spec.name,
        needs_logical_n=spec.needs_logical_n,
        is_marlin_style=spec.fn_name == "marlin_style",
        size_n=size_n,
        size_k=size_k,
        min_m=spec.min_m,
        max_m=spec.max_m,
        m_multiple=spec.m_multiple,
        exact_ms=spec.exact_ms,
        packer=spec.packer,
        spec=spec,
    )


def _build_kernel_family(decode_dispatch: _KernelDispatch) -> _KernelDispatchFamily:
    """Return every micro-kernel that shares ``decode_dispatch``'s packed layout.

    Members are sorted so the largest-M legal variant is tried first: larger
    ``max_m``, then larger ``m_multiple`` (coarser M tile), then larger
    ``n_multiple`` (fewer N blocks).  All members reuse the same ``packed``
    tensors, so switching between them at inference time costs only an op
    lookup + kernel launch.
    """
    layout_id = _layout_id_for_spec(decode_dispatch.spec)
    members: list[_KernelDispatch] = []
    for spec in _DYNAMIC_CANDIDATES:
        if _layout_id_for_spec(spec) != layout_id:
            continue
        if spec.fn_name == "marlin_style":
            op = _get_amplin_op("marlin_style_run")
        else:
            op = _get_amplin_op(spec.fn_name)
        members.append(
            _KernelDispatch(
                op=op,
                packed=decode_dispatch.packed,
                name=spec.name,
                needs_logical_n=spec.needs_logical_n,
                is_marlin_style=spec.fn_name == "marlin_style",
                size_n=decode_dispatch.size_n,
                size_k=decode_dispatch.size_k,
                min_m=spec.min_m,
                max_m=spec.max_m,
                m_multiple=spec.m_multiple,
                exact_ms=spec.exact_ms,
                packer=spec.packer,
                spec=spec,
            )
        )

    def _priority(m: _KernelDispatch) -> tuple[int, int, int]:
        return (
            m.max_m if m.max_m is not None else 1_000_000_000,
            m.spec.m_multiple,
            m.spec.n_multiple,
        )

    members.sort(key=_priority, reverse=True)
    return _KernelDispatchFamily(
        packed=decode_dispatch.packed,
        layout_id=layout_id,
        members=tuple(members),
        default=decode_dispatch,
    )


def _pack_for_spec(
    spec: _CandidateSpec,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    cache: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Pack (or expose) weights for ``spec`` without any persistent cache.

    The packed tensors are owned by the caller; no CPU/GPU migration or eviction
    is performed here.  ``cache`` controls whether the Marlin-style repack is
    stored in the per-thread ``marlin_pack_cache`` (enabled during steady-state
    dispatch, disabled during candidate micro-benchmarks).
    """
    if spec.fn_name == "marlin_style":
        size_n = qweight.size(-1)
        marlin_qweight, marlin_scales, workspace, b_q_type = _get_marlin_packed(
            qweight, scales, size_n, dtype, device, evict=False, cache=cache
        )
        return (marlin_qweight, marlin_scales, workspace, b_q_type.id)

    q = qweight.to(device, non_blocking=False).contiguous().detach()
    s = scales.to(device, dtype=dtype).contiguous().detach()
    if spec.packer == "none":
        return (q, s)

    # Non-``none`` packed layouts own their own GPU tensors; move the canonical
    # (unpacked) weights to CPU to avoid duplicate VRAM.  Callers that want to
    # fully discard the canonical copy can meta them after ``_pack_for_spec``.
    if qweight.device.type == "cuda":
        qweight.data = qweight.to("cpu", non_blocking=False).contiguous()
    if scales.device.type == "cuda":
        scales.data = scales.to("cpu", non_blocking=False).contiguous()

    if spec.packer == "hmma":
        return (pack_hmma_qweight(q), pack_hmma_scales(s))
    if spec.packer == "lane":
        return (pack_mma_lane_qweight(q), pack_hmma_scales(s))
    if spec.packer == "n32_interleaved":
        return (pack_mma_lane_n32_qweight(q), pack_hmma_scales(s))
    if spec.packer == "n64":
        return (pack_mma_lane_n64_qweight(q), pack_hmma_scales(s))
    raise ValueError(f"Unknown packer {spec.packer!r}")


def _time_kernel(
    dispatch: _KernelDispatch,
    input: torch.Tensor,
    warmup: int,
    iters: int,
) -> float | None:
    """Time ``dispatch`` with warmup and median timing."""
    try:
        for _ in range(warmup):
            _call_kernel(dispatch, input)
            torch.cuda.synchronize(input.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _call_kernel(dispatch, input)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iters
    except Exception:
        return None


def _select_best_kernel(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype,
    size_m: int,
    size_k: int,
    size_n: int,
    warmup: int,
    iters: int,
) -> tuple[str, _KernelDispatch, torch.Tensor]:
    """Benchmark every legal candidate and return the fastest dispatch/output."""
    best_time = float("inf")
    best_spec: _CandidateSpec | None = None
    best_dispatch: _KernelDispatch | None = None

    for spec in _DYNAMIC_CANDIDATES:
        if not _is_candidate_legal(spec, size_m, size_k, size_n):
            continue
        if spec.fn_name == "marlin_style" and not _marlin_available_cached(dtype):
            continue
        try:
            # Micro-benchmark candidates must not pollute the persistent caches;
            # only the winning layout should stay in VRAM.
            dispatch = _build_kernel_dispatch(
                spec, qweight, scales, dtype, input.device, size_n, size_k, cache=False
            )
            elapsed = _time_kernel(dispatch, input, warmup, iters)
        except Exception:
            elapsed = None
        if elapsed is not None and elapsed < best_time:
            best_time = elapsed
            best_spec = spec
            best_dispatch = dispatch

    if best_spec is None or best_dispatch is None:
        raise RuntimeError(
            f"No legal Amplin dynamic candidate for M={size_m}, K={size_k}, N={size_n}"
        )

    torch.cuda.synchronize(input.device)
    return best_spec.name, best_dispatch, _call_kernel(best_dispatch, input)


def _dynamic_impl(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    logical_n: int | None = None,
    warmup: int = 5,
    iters: int | None = 10,
    update_dynamic_table: bool = True,
) -> torch.Tensor:
    """Backend implementation for ``amplin.dynamic``.  See ``dynamic`` for docs."""
    if iters is None:
        iters = 10

    if not _ensure_amplin_runtime_available():
        raise RuntimeError(amplin_runtime_error())

    if logical_n is None:
        size_n = qweight.size(-1)
    else:
        size_n = logical_n

    device = input.device
    dtype = input.dtype

    # Fast path: repeated calls with the same (shape, dtype, weights) skip
    # validation, routing-table locks, and spec/op lookups by launching the cached
    # kernel directly.  ``torch.Size`` and ``torch.dtype`` are hashable, so avoid
    # allocating tuples or converting dtype to a string here.
    if input.dim() >= 2 and input.is_contiguous():
        fast_key = (input.shape, dtype, id(qweight), id(scales), qweight.shape, scales.shape)
        fast_dispatch_cache = _thread_caches().fast_dispatch_cache
        dispatch = fast_dispatch_cache.get(fast_key)
        if dispatch is not None:
            if dispatch.is_marlin_style:
                return dispatch.op(input, *dispatch.packed, dispatch.size_n, dispatch.size_k)
            if dispatch.needs_logical_n:
                return dispatch.op(input, *dispatch.packed, logical_n=dispatch.size_n)
            return dispatch.op(input, *dispatch.packed)

    input = input.contiguous()
    if input.dim() < 2:
        raise ValueError("Amplin dynamic input must have at least two dimensions")

    size_k = input.size(-1)
    size_m = input.numel() // size_k
    if qweight.dim() != 2 or qweight.size(0) * 8 != size_k:
        raise ValueError(
            "Amplin dynamic qweight must have shape [K/8, N] and match input K"
        )
    if scales.dim() != 2 or scales.size(0) != size_k // HMMA_K_TILE or scales.size(1) != size_n:
        raise ValueError(
            "Amplin dynamic scales must have shape [K/128, N] and match input K/output N"
        )

    fast_key = (input.shape, dtype, id(qweight), id(scales), qweight.shape, scales.shape)
    key = _dynamic_key(input, size_n)
    with _ROUTING_LOCK:
        choice = _DYNAMIC_ROUTING_TABLE.get(key) or _STATIC_ROUTING_TABLE.get(key)
        if choice == "marlin_style" and not _marlin_available_cached(dtype):
            choice = None
        if choice is None:
            choice, dispatch, output = _select_best_kernel(
                input, qweight, scales, dtype, size_m, size_k, size_n, warmup, iters
            )
            if update_dynamic_table:
                _DYNAMIC_ROUTING_TABLE[key] = choice
            _thread_caches().fast_dispatch_cache[fast_key] = dispatch
            return output

        fast_dispatch_cache = _thread_caches().fast_dispatch_cache
        dispatch = fast_dispatch_cache.get(fast_key)
        if dispatch is not None:
            if dispatch.is_marlin_style:
                return dispatch.op(input, *dispatch.packed, dispatch.size_n, dispatch.size_k)
            if dispatch.needs_logical_n:
                return dispatch.op(input, *dispatch.packed, logical_n=dispatch.size_n)
            return dispatch.op(input, *dispatch.packed)

        spec = _CANDIDATE_BY_NAME.get(choice)
        if spec is None:
            raise RuntimeError(f"Amplin dynamic routing table contains unknown kernel {choice!r}")
        dispatch = _build_kernel_dispatch(spec, qweight, scales, dtype, device, size_n, size_k)
        _thread_caches().fast_dispatch_cache[fast_key] = dispatch
        if dispatch.is_marlin_style:
            return dispatch.op(input, *dispatch.packed, dispatch.size_n, dispatch.size_k)
        if dispatch.needs_logical_n:
            return dispatch.op(input, *dispatch.packed, logical_n=dispatch.size_n)
        return dispatch.op(input, *dispatch.packed)

def _dynamic_impl_meta(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    logical_n: int | None = None,
    warmup: int = 5,
    iters: int = 10,
    update_dynamic_table: bool = True,
) -> torch.Tensor:
    """Meta/FakeTensor implementation for ``gptqmodel_amplin::dynamic``."""
    size_n = qweight.size(-1) if logical_n is None else logical_n
    out_shape = list(input.shape[:-1]) + [size_n]
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)


def select_kernel(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    size_m: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    logical_n: int | None = None,
    warmup: int = 5,
    iters: int = 10,
    update_dynamic_table: bool = True,
) -> _KernelDispatch:
    """Select, pack, and return prepared kernel invocation state for a fixed (M, K, N, dtype).

    ``AmplinLinear.post_init`` should store the returned dispatch object and use
    ``_call_kernel`` or ``_call_kernel_chunked`` in ``forward`` without going
    through ``amplin.dynamic`` for the prepared batch range.
    """
    if not _ensure_amplin_runtime_available():
        raise RuntimeError(amplin_runtime_error())

    if logical_n is None:
        size_n = qweight.size(-1)
    else:
        size_n = logical_n

    if qweight.dim() != 2:
        raise ValueError("Amplin select_kernel qweight must be 2D")
    size_k = qweight.size(0) * 8
    if scales.dim() != 2 or scales.size(0) != size_k // HMMA_K_TILE or scales.size(1) != size_n:
        raise ValueError(
            "Amplin select_kernel scales must have shape [K/128, N] and match input K/output N"
        )

    key = (size_m, size_k, size_n, _dtype_name(dtype))
    input = torch.randn((size_m, size_k), device=device, dtype=dtype).mul_(0.25).contiguous()

    with _ROUTING_LOCK:
        choice = _DYNAMIC_ROUTING_TABLE.get(key) or _STATIC_ROUTING_TABLE.get(key)
        if choice == "marlin_style" and not _marlin_available_cached(dtype):
            choice = None

        if choice is None:
            choice, dispatch, _ = _select_best_kernel(
                input, qweight, scales, dtype, size_m, size_k, size_n, warmup, iters
            )
            if update_dynamic_table:
                _DYNAMIC_ROUTING_TABLE[key] = choice
            return dispatch

        spec = _CANDIDATE_BY_NAME.get(choice)
        if spec is None:
            raise RuntimeError(f"Amplin routing table contains unknown kernel {choice!r}")
        return _build_kernel_dispatch(spec, qweight, scales, dtype, device, size_n, size_k)


# Register the Python dynamic router as a custom op so ``torch.compile(fullgraph=True)``
# treats it as an opaque kernel and CUDA Graph capture records its internal launches.
_AMPLIN_FRAGMENT_LIB = torch.library.Library("gptqmodel_amplin", "FRAGMENT")
_AMPLIN_FRAGMENT_LIB.define(
    "dynamic(Tensor input, Tensor qweight, Tensor scales, int? logical_n = None, "
    "int warmup = 5, int iters = 10, bool update_dynamic_table = True) -> Tensor"
)
_AMPLIN_FRAGMENT_LIB.impl("dynamic", _dynamic_impl, "CompositeExplicitAutograd")
_AMPLIN_FRAGMENT_LIB.impl("dynamic", _dynamic_impl_meta, "Meta")


def dynamic(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    logical_n: int | None = None,
    warmup: int = 5,
    iters: int | None = 10,
    update_dynamic_table: bool = True,
) -> torch.Tensor:
    """Dispatch to the best Amplin kernel for this (M, K, N, dtype).

    On the first call for a given shape, the function runs a fast micro-benchmark
    across the legal candidate kernels, picks the fastest, and stores the choice in
    the dynamic routing table.  Subsequent calls reuse the cached choice.  A
    static routing table (populated from benchmarks or ``set_routing_table``) is
    consulted first and the dynamic table overrides it for newly-discovered shapes.

    In eager mode the implementation is called directly to avoid an extra
    ``torch.ops`` dispatch round-trip.  Under ``torch.compile`` the router is
    invoked as the ``gptqmodel_amplin::dynamic`` custom op so the graph can be
    captured in fullgraph mode and by CUDA Graph.
    """
    if torch.compiler.is_compiling():
        return torch.ops.gptqmodel_amplin.dynamic(
            input,
            qweight,
            scales,
            logical_n=logical_n,
            warmup=warmup,
            iters=10 if iters is None else iters,
            update_dynamic_table=update_dynamic_table,
        )
    return _dynamic_impl(
        input,
        qweight,
        scales,
        logical_n=logical_n,
        warmup=warmup,
        iters=iters,
        update_dynamic_table=update_dynamic_table,
    )


def get_static_routing_table() -> dict[tuple[int, int, int, str], str]:
    with _ROUTING_LOCK:
        return _STATIC_ROUTING_TABLE.copy()


def get_dynamic_routing_table() -> dict[tuple[int, int, int, str], str]:
    with _ROUTING_LOCK:
        return _DYNAMIC_ROUTING_TABLE.copy()


def get_routing_table() -> dict[tuple[int, int, int, str], str]:
    """Return the merged static + dynamic routing table."""
    with _ROUTING_LOCK:
        return {**_STATIC_ROUTING_TABLE, **_DYNAMIC_ROUTING_TABLE}


def set_routing_table(entries: dict[tuple[int, int, int, str], str]) -> None:
    """Replace the static routing table.  Existing dynamic entries are preserved."""
    with _ROUTING_LOCK:
        _STATIC_ROUTING_TABLE.clear()
        _STATIC_ROUTING_TABLE.update(entries)
    _thread_caches().fast_dispatch_cache.clear()


def _parse_routing_key(raw_key: object) -> tuple[int, int, int, str]:
    """Accept JSON array keys or stringified Python tuple keys."""
    if isinstance(raw_key, str):
        parsed = ast.literal_eval(raw_key)
    else:
        parsed = raw_key
    m, k, n, dtype = parsed
    return int(m), int(k), int(n), str(dtype)


def load_routing_table(path: str | Path) -> None:
    """Load a static routing table from JSON.

    Accepts keys as JSON arrays ``[m, k, n, dtype]`` or stringified Python
    tuples (as written by the benchmark script).  dtype is "fp16" or "bf16".
    """
    path = Path(path)
    data = json.loads(path.read_text())
    entries = {
        _parse_routing_key(raw_key): name
        for raw_key, name in data.items()
    }
    set_routing_table(entries)


def save_routing_table(path: str | Path) -> None:
    """Save the merged routing table to JSON using stringified tuple keys."""
    path = Path(path)
    serializable = {str(k): v for k, v in get_routing_table().items()}
    path.write_text(json.dumps(serializable, indent=2))


def clear_dynamic_routing_table() -> None:
    with _ROUTING_LOCK:
        _DYNAMIC_ROUTING_TABLE.clear()
    _thread_caches().fast_dispatch_cache.clear()


def clear_thread_caches() -> None:
    """Release all packed weights, op handles, and dispatch caches for the calling thread."""
    _thread_caches().clear()


# Build a name -> spec lookup for cached dispatch.
_CANDIDATE_BY_NAME: dict[str, _CandidateSpec] = {
    spec.name: spec for spec in _DYNAMIC_CANDIDATES
}


__all__ = [
    "HMMA_K_TILE",
    "HMMA_N_TILE",
    "HMMA_PACKED_K_WORDS",
    "MMA_LANES",
    "MMA_LANE_K_STEPS",
    "MMA_LANE_N_WARPS",
    "amplin_runtime_available",
    "amplin_runtime_error",
    "amplin_supported",
    "clear_dynamic_routing_table",
    "clear_thread_caches",
    "dynamic",
    "get_dynamic_routing_table",
    "get_routing_table",
    "get_static_routing_table",
    "load_routing_table",
    "save_routing_table",
    "select_kernel",
    "set_routing_table",
    "gemm_hmma",
    "gemm_hmma_m64_v1",
    "gemm_hmma_m64_v2",
    "gemm_hmma_m64_v2_sync_a128",
    "gemm_hmma_m64_v3",
    "gemm_hmma_m32_n128_pipeline4",
    "gemm_hmma_v0",
    "gemv",
    "mma_lane_m16_n16_padded",
    "mma_lane_m16_n16_splitk4",
    "mma_lane_m16_n16_splitk8",
    "mma_lane_m16_n16_splitk12",
    "mma_lane_m16_n32_splitk12",
    "mma_lane_m16_n32_splitk16",
    "mma_lane_m16_n32_splitk8",
    "mma_lane_m16_n32_splitk8_pipe2",
    "mma_lane_m16_n32_splitk12_pipe2",
    "mma_lane_m16_n32_splitk12_pipe2_interleaved",
    "mma_lane_m16_n64_splitk24_pipe2_interleaved",
    "mma_lane_m16_n64_splitk12x2_coop_interleaved",
    "mma_lane_m16_n64_shared_a",
    "mma_lane_m16_n64_tile4_shared_a",
    "mma_lane_m16_n64_tile8_shared_a",
    "mma_lane_m32_n64_tile2_shared_a",
    "mma_lane_m32_n64_tile2_interleaved_dequant",
    "mma_lane_m32_n64_tile4_shared_a",
    "mma_lane_m32_n64_tile8_shared_a",
    "mma_lane_m32_n64_shared_a",
    "mma_lane_m32_n64_splitk12x2_coop_interleaved",
    "mma_lane_m16_n32_splitk16_pipe2",
    "mma_lane_m16_n16_splitk16",
    "mma_lane_m32_global_a",
    "mma_lane_m32_n32_global_a",
    "mma_lane_m64",
    "mma_lane_m64_global_a",
    "mma_lane_tile",
    "mma_lane_tile_global_a",
    "pack_hmma_qweight",
    "pack_hmma_scales",
    "pack_hmma_weights",
    "pack_mma_lane_qweight",
    "pack_mma_lane_n32_qweight",
    "pack_mma_lane_n64_qweight",
    "unpack_hmma_qweight",
    "unpack_hmma_scales",
    "unpack_mma_lane_qweight",
    "unpack_mma_lane_n32_qweight",
    "unpack_mma_lane_n64_qweight",
]


# Pre-populate the static routing table from a bundled benchmark snapshot when available.
# This avoids the first-call micro-benchmark for shapes that have already been profiled.
_DEFAULT_ROUTING_TABLE = Path(__file__).parent / "amplin_dynamic_routing_table.json"
if _DEFAULT_ROUTING_TABLE.exists():
    try:
        load_routing_table(_DEFAULT_ROUTING_TABLE)
    except Exception:
        pass
