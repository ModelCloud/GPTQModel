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
        "mma_lane_m64",
        "mma_lane_m64_global_a",
        "mma_lane_m32_global_a",
        "mma_lane_m32_n32_global_a",
        "mma_lane_m16_n16_padded",
        "mma_lane_m16_n16_splitk4",
        "mma_lane_m16_n16_splitk8",
        "mma_lane_m16_n16_splitk12",
        "mma_lane_m16_n32_splitk12",
        "mma_lane_m16_n32_splitk16",
        "mma_lane_m16_n32_splitk12_pipe2",
        "mma_lane_m16_n32_splitk12_pipe2_interleaved",
        "mma_lane_m16_n64_splitk24_pipe2_interleaved",
        "mma_lane_m16_n64_splitk12x2_coop_interleaved",
        "mma_lane_m16_n32_splitk16_pipe2",
        "mma_lane_m16_n16_splitk16",
        "mma_lane_tile",
        "mma_lane_tile_global_a",
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
    "gemm_hmma",
    "gemm_hmma_m64_v1",
    "gemm_hmma_m64_v2",
    "gemm_hmma_m64_v2_sync_a128",
    "gemm_hmma_m64_v3",
    "gemm_hmma_v0",
    "gemv",
    "mma_lane_m16_n16_padded",
    "mma_lane_m16_n16_splitk4",
    "mma_lane_m16_n16_splitk8",
    "mma_lane_m16_n16_splitk12",
    "mma_lane_m16_n32_splitk12",
    "mma_lane_m16_n32_splitk16",
    "mma_lane_m16_n32_splitk12_pipe2",
    "mma_lane_m16_n32_splitk12_pipe2_interleaved",
    "mma_lane_m16_n64_splitk24_pipe2_interleaved",
    "mma_lane_m16_n64_splitk12x2_coop_interleaved",
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
