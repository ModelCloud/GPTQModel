# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
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
_CANDIDATE_OP_CACHE: dict[str, Callable] = {}


def _ensure_amplin_runtime_available() -> bool:
    global _AMPLIN_RUNTIME_AVAILABLE
    if _AMPLIN_RUNTIME_AVAILABLE is None:
        _AMPLIN_RUNTIME_AVAILABLE = amplin_runtime_available()
    return _AMPLIN_RUNTIME_AVAILABLE


def _get_amplin_op(op_name: str) -> Callable:
    op = _CANDIDATE_OP_CACHE.get(op_name)
    if op is None:
        op = _extension_api().op("amplin", op_name)
        _CANDIDATE_OP_CACHE[op_name] = op
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
        "mma_lane_m32_n64_tile8_shared_a",
        "mma_lane_m32_n64_tile8_shared_a",
        "n64",
        max_m=32,
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
        min_m=17,
        max_m=32,
        n_multiple=64,
        k_multiple=128,
    ),
)


# Cache packed weights keyed by tensor memory address, shape, packer, and dtype so
# repeated calls for the same layer avoid re-packing without reusing stale entries
# when tensor object ids are recycled after deallocation.
_WEIGHT_CACHE: dict[
    tuple[int, int, tuple[int, ...], tuple[int, ...], str, torch.dtype],
    tuple[torch.Tensor, torch.Tensor],
] = {}

# Fast-dispatch cache keyed like _WEIGHT_CACHE plus the routing key.  When the
# routing choice and packed weights for a layer are known, repeated calls avoid
# spec/op lookup and the _run_candidate indirection.
_FAST_DISPATCH_CACHE: dict[
    tuple[
        tuple[int, int, int, str],
        int,
        int,
        tuple[int, ...],
        tuple[int, ...],
    ],
    tuple[Callable, torch.Tensor, torch.Tensor, bool, int],
] = {}

# Cache for the Marlin-style candidate's repacked weights, permuted scales and
# workspace.  Keyed by the canonical qweight/scales pointers so repeated calls
# for the same layer avoid the repack overhead.
_MARLIN_PACK_CACHE: dict[
    tuple[int, int, tuple[int, ...], tuple[int, ...], int, str],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, object],
] = {}
_MARLIN_GEMM_OP_CACHE: dict[torch.dtype, Callable] = {}
_MARLIN_AVAILABLE_CACHE: dict[torch.dtype, bool] = {}


def _pack_with_cache(
    spec: _CandidateSpec,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        qweight.data_ptr(),
        scales.data_ptr(),
        tuple(qweight.shape),
        tuple(scales.shape),
        spec.packer,
        dtype,
    )
    packed = _WEIGHT_CACHE.get(key)
    if packed is None:
        if spec.packer == "none":
            packed = (qweight, scales.to(dtype))
        elif spec.packer == "hmma":
            packed = (pack_hmma_qweight(qweight), pack_hmma_scales(scales.to(dtype)))
        elif spec.packer == "lane":
            packed = (pack_mma_lane_qweight(qweight), pack_hmma_scales(scales.to(dtype)))
        elif spec.packer == "n32_interleaved":
            packed = (pack_mma_lane_n32_qweight(qweight), pack_hmma_scales(scales.to(dtype)))
        elif spec.packer == "n64":
            packed = (pack_mma_lane_n64_qweight(qweight), pack_hmma_scales(scales.to(dtype)))
        else:
            raise ValueError(f"Unknown packer {spec.packer!r}")
        _WEIGHT_CACHE[key] = packed
    return packed


def _get_marlin_packed(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    size_n: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
    """Repack canonical GPTQ weights/scales into Marlin's execution layout."""
    size_k = qweight.size(0) * 8
    key = (
        qweight.data_ptr(),
        scales.data_ptr(),
        tuple(qweight.shape),
        tuple(scales.shape),
        size_n,
        _dtype_name(dtype),
    )
    cached = _MARLIN_PACK_CACHE.get(key)
    if cached is not None:
        return cached

    perm = torch.empty(0, dtype=torch.int, device=qweight.device)
    marlin_qweight = gptq_marlin_repack(
        qweight.contiguous(),
        perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=4,
        dtype=dtype,
    )
    marlin_scales = marlin_permute_scales(
        scales.to(dtype).contiguous(),
        size_k=size_k,
        size_n=size_n,
        group_size=128,
    )
    workspace = marlin_make_workspace_new(qweight.device)
    b_q_type = scalar_types.uint4b8
    packed = (marlin_qweight, marlin_scales, workspace, b_q_type)
    _MARLIN_PACK_CACHE[key] = packed
    return packed


def _get_marlin_gemm_op(dtype: torch.dtype) -> Callable:
    op = _MARLIN_GEMM_OP_CACHE.get(dtype)
    if op is None:
        op_name = "gptq_marlin_gemm_fp16" if dtype == torch.float16 else "gptq_marlin_gemm_bf16"
        op = _marlin_resolve_op(dtype=dtype, op_name=op_name)
        _MARLIN_GEMM_OP_CACHE[dtype] = op
    return op


def _marlin_available_cached(dtype: torch.dtype) -> bool:
    cached = _MARLIN_AVAILABLE_CACHE.get(dtype)
    if cached is None:
        cached = marlin_runtime_available(dtype)
        _MARLIN_AVAILABLE_CACHE[dtype] = cached
    return cached


def _get_marlin_style_fast_runner(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    size_n: int,
    dtype: torch.dtype,
) -> Callable[..., torch.Tensor]:
    """Return a closure that runs the C++ marlin_style fast path."""
    marlin_qweight, marlin_scales, workspace, b_q_type = _get_marlin_packed(
        qweight, scales, size_n, dtype
    )
    cpp_op = _get_amplin_op("marlin_style_run")
    size_k = qweight.size(0) * 8

    def _marlin_style_fast_run(
        input: torch.Tensor, *_args: object, **_kwargs: object
    ) -> torch.Tensor:
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
) -> torch.Tensor:
    size_k = input.size(-1)
    size_m = input.numel() // size_k
    size_n = logical_n
    input_2d = input.reshape(-1, size_k).contiguous()
    marlin_qweight, marlin_scales, workspace, b_q_type = _get_marlin_packed(
        qweight, scales, size_n, input.dtype
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


def _run_candidate(
    spec: _CandidateSpec,
    input: torch.Tensor,
    packed_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    size_n: int,
) -> torch.Tensor:
    if spec.fn_name == "marlin_style":
        return _run_marlin_style(input, packed_qweight, packed_scales, logical_n=size_n)
    op = _get_amplin_op(spec.fn_name)
    if spec.needs_logical_n:
        return op(input, packed_qweight, packed_scales, logical_n=size_n)
    return op(input, packed_qweight, packed_scales)


def _time_candidate(
    spec: _CandidateSpec,
    input: torch.Tensor,
    packed_qweight: torch.Tensor,
    packed_scales: torch.Tensor,
    size_n: int,
    warmup: int,
    iters: int,
) -> float | None:
    try:
        for _ in range(warmup):
            _run_candidate(spec, input, packed_qweight, packed_scales, size_n)
            torch.cuda.synchronize(input.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _run_candidate(spec, input, packed_qweight, packed_scales, size_n)
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
) -> tuple[str, torch.Tensor]:
    packed_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    best_spec: _CandidateSpec | None = None
    best_time = float("inf")
    best_output: torch.Tensor | None = None

    for spec in _DYNAMIC_CANDIDATES:
        if not _is_candidate_legal(spec, size_m, size_k, size_n):
            continue
        if spec.fn_name == "marlin_style" and not _marlin_available_cached(dtype):
            continue
        if spec.packer not in packed_cache:
            packed_cache[spec.packer] = _pack_with_cache(spec, qweight, scales, dtype)
        packed_qweight, packed_scales = packed_cache[spec.packer]
        elapsed = _time_candidate(spec, input, packed_qweight, packed_scales, size_n, warmup, iters)
        if elapsed is not None and elapsed < best_time:
            best_time = elapsed
            best_spec = spec
            best_output = _run_candidate(spec, input, packed_qweight, packed_scales, size_n)

    if best_spec is None:
        raise RuntimeError(
            f"No legal Amplin dynamic candidate for M={size_m}, K={size_k}, N={size_n}"
        )

    # The benchmark already produced the output for the fastest candidate; use it
    # directly and return the name so the dynamic table can be updated.
    if best_output is None:
        packed_qweight, packed_scales = packed_cache[best_spec.packer]
        best_output = _run_candidate(best_spec, input, packed_qweight, packed_scales, size_n)
    torch.cuda.synchronize(input.device)
    return best_spec.name, best_output


def dynamic(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    logical_n: int | None = None,
    warmup: int = 5,
    iters: int = 10,
    update_dynamic_table: bool = True,
) -> torch.Tensor:
    """Dispatch to the best Amplin kernel for this (M, K, N, dtype).

    On the first call for a given shape, the function runs a fast micro-benchmark
    across the legal candidate kernels, picks the fastest, and stores the choice in
    the dynamic routing table.  Subsequent calls reuse the cached choice.  A
    static routing table (populated from benchmarks or ``set_routing_table``) is
    consulted first and the dynamic table overrides it for newly-discovered shapes.
    """
    if not _ensure_amplin_runtime_available():
        raise RuntimeError(amplin_runtime_error())

    if logical_n is None:
        size_n = qweight.size(-1)
    else:
        size_n = logical_n

    # Fast path: repeated calls with the same (M, K, N, dtype) and weights skip
    # validation, routing-table locks, and spec/op lookups.
    if input.dim() >= 2 and input.is_contiguous():
        size_k = input.size(-1)
        size_m = input.numel() // size_k
        key = (size_m, size_k, size_n, _dtype_name(input.dtype))
        fast_key = (
            key,
            qweight.data_ptr(),
            scales.data_ptr(),
            qweight.shape,
            scales.shape,
        )
        cached = _FAST_DISPATCH_CACHE.get(fast_key)
        if cached is not None:
            op, packed_qweight, packed_scales, needs_logical_n, cached_size_n = cached
            if cached_size_n == size_n and (
                op is not _run_marlin_style or _marlin_available_cached(input.dtype)
            ):
                if needs_logical_n:
                    return op(input, packed_qweight, packed_scales, logical_n=size_n)
                return op(input, packed_qweight, packed_scales)

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

    dtype = input.dtype
    key = _dynamic_key(input, size_n)
    with _ROUTING_LOCK:
        choice = _DYNAMIC_ROUTING_TABLE.get(key) or _STATIC_ROUTING_TABLE.get(key)
        if choice == "marlin_style" and not _marlin_available_cached(dtype):
            choice = None
        if choice is None:
            choice, output = _select_best_kernel(
                input, qweight, scales, dtype, size_m, size_k, size_n, warmup, iters
            )
            if update_dynamic_table:
                _DYNAMIC_ROUTING_TABLE[key] = choice
            return output

        fast_key = (
            key,
            qweight.data_ptr(),
            scales.data_ptr(),
            qweight.shape,
            scales.shape,
        )
        cached = _FAST_DISPATCH_CACHE.get(fast_key)
        if cached is not None:
            op, packed_qweight, packed_scales, needs_logical_n, cached_size_n = cached
            if cached_size_n == size_n:
                if needs_logical_n:
                    return op(input, packed_qweight, packed_scales, logical_n=size_n)
                return op(input, packed_qweight, packed_scales)

        spec = _CANDIDATE_BY_NAME.get(choice)
        if spec is None:
            raise RuntimeError(f"Amplin dynamic routing table contains unknown kernel {choice!r}")
        packed_qweight, packed_scales = _pack_with_cache(spec, qweight, scales, dtype)
        if spec.fn_name == "marlin_style":
            op = _get_marlin_style_fast_runner(qweight, scales, size_n, dtype)
            needs_logical_n = False
        else:
            op = _get_amplin_op(spec.fn_name)
            needs_logical_n = spec.needs_logical_n
        _FAST_DISPATCH_CACHE[fast_key] = (
            op,
            packed_qweight,
            packed_scales,
            needs_logical_n,
            size_n,
        )
        if needs_logical_n:
            return op(input, packed_qweight, packed_scales, logical_n=size_n)
        return op(input, packed_qweight, packed_scales)


def get_static_routing_table() -> dict[tuple[int, int, int, str], str]:
    return _STATIC_ROUTING_TABLE.copy()


def get_dynamic_routing_table() -> dict[tuple[int, int, int, str], str]:
    return _DYNAMIC_ROUTING_TABLE.copy()


def get_routing_table() -> dict[tuple[int, int, int, str], str]:
    """Return the merged static + dynamic routing table."""
    return {**_STATIC_ROUTING_TABLE, **_DYNAMIC_ROUTING_TABLE}


def set_routing_table(entries: dict[tuple[int, int, int, str], str]) -> None:
    """Replace the static routing table.  Existing dynamic entries are preserved."""
    _STATIC_ROUTING_TABLE.clear()
    _STATIC_ROUTING_TABLE.update(entries)
    _FAST_DISPATCH_CACHE.clear()


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
    _DYNAMIC_ROUTING_TABLE.clear()
    _FAST_DISPATCH_CACHE.clear()


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
    "dynamic",
    "get_dynamic_routing_table",
    "get_routing_table",
    "get_static_routing_table",
    "load_routing_table",
    "save_routing_table",
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
