# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exact standard-P32 continuous-window kernels for MLX/Metal."""

from __future__ import annotations

import operator
import threading
from typing import Any

from ..quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    PGC16_V2B4_BANK_XOR_MASKS_BY_TRANSITION_BITS,
    pgc16_levels_for_version,
)
from ..quantization.qvq_rates import (
    normalize_qvq_rate,
    qvq_transition_bits,
    qvq_words_per_tile,
)

_KERNEL_LOCK = threading.Lock()
_REPACK_KERNELS: dict[int, Any] = {}
_REPACK_KERNEL_ERRORS: dict[int, str] = {}
_GEMV_KERNELS: dict[tuple[int, int, int, int], Any] = {}
_GEMV_KERNEL_ERRORS: dict[tuple[int, int, int, int], str] = {}


def _p32_constants_metal() -> str:
    """Embed immutable PGC16-v1 levels and V2 bank masks in Metal constants."""

    import torch

    masks = (
        "\n".join(
            "  {" + ",".join(
                f"0x{mask:04x}u" for mask in PGC16_V2B4_BANK_XOR_MASKS_BY_TRANSITION_BITS[bits]
            ) + "},"
            for bits in range(2, 8)
        )
    )
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).to(torch.float16).contiguous()
    level_bits = levels.view(torch.int16).tolist()
    encoded_levels = ",".join(f"0x{int(value) & 0xffff:04x}u" for value in level_bits)
    return (
        "constant ushort qvq_p32_bank_masks[6][4]={\n"
        + masks
        + "\n};\nconstant ushort qvq_p32_levels[256]={"
        + encoded_levels
        + "};"
    )


_P32_CONSTANTS_METAL = _p32_constants_metal()

_REPACK_HEADER = r"""
inline uint qvq_p32_plane_width(uint remaining) {
  if (remaining >= 16u) return 16u;
  if (remaining >= 8u) return 8u;
  if (remaining >= 4u) return 4u;
  if (remaining >= 2u) return 2u;
  return 1u;
}
inline uint qvq_p32_planar_edge(device const int* tile, uint edge, uint edge_bits) {
  uint block = edge >> 5u, lane = edge & 31u, base = block * edge_bits;
  uint remaining = edge_bits, row = 0u, offset = 0u, value = 0u;
  for (uint plane = 0u; plane < 4u && remaining; ++plane) {
    uint width = qvq_p32_plane_width(remaining), packed_per_word = 32u / width;
    uint word = as_type<uint>(tile[base + row + lane / packed_per_word]);
    uint code = (word >> (width * (lane % packed_per_word))) & ((1u << width) - 1u);
    value |= code << offset;
    remaining -= width;
    row += width;
    offset += width;
  }
  return value;
}
"""

_REPACK_SOURCE = r"""
uint index = thread_position_in_grid.x;
uint words_per_tile = 4u * EdgeBits;
uint tile_index = index / words_per_tile;
uint word_index = index - tile_index * words_per_tile;
device const int* tile = planar + tile_index * words_per_tile;
uint packed = 0u;
uint output_bit_base = word_index << 5u;
for (uint output_bit = 0u; output_bit < 32u; ++output_bit) {
  uint stream_bit = output_bit_base + output_bit;
  uint reversed_edge = stream_bit / EdgeBits;
  uint edge_bit = stream_bit - reversed_edge * EdgeBits;
  uint edge = 127u - reversed_edge;
  packed |= ((qvq_p32_planar_edge(tile, edge, EdgeBits) >> edge_bit) & 1u) << output_bit;
}
window[index] = as_type<int>(packed);
"""

_GEMV_HEADER = r"""
__QVQ_P32_CONSTANTS__
inline uint qvq_p32_window_state(device const int* words, uint pair, uint edge_bits) {
  uint words_per_tile = 4u * edge_bits;
  uint bit_position = (127u - pair) * edge_bits;
  uint word_index = bit_position >> 5u;
  uint shift = bit_position & 31u;
  uint value = as_type<uint>(words[word_index]) >> shift;
  if (shift > 16u) {
    uint next_word = word_index + 1u == words_per_tile ? 0u : word_index + 1u;
    value |= as_type<uint>(words[next_word]) << (32u - shift);
  }
  return value & 0xffffu;
}
inline float2 qvq_p32_decode(uint state, uint bank, uint edge_bits) {
  uint mixed = state ^ uint(qvq_p32_bank_masks[edge_bits - 2u][bank]);
  mixed ^= mixed >> 8u;
  mixed = (mixed * 40503u + 17011u) & 0xffffu;
  mixed ^= mixed >> 7u;
  half first = as_type<half>(qvq_p32_levels[mixed >> 8u]);
  half second = as_type<half>(qvq_p32_levels[mixed & 255u]);
  return float2(float(first), float(second));
}
""".replace("__QVQ_P32_CONSTANTS__", _P32_CONSTANTS_METAL)

_GEMV_SOURCE = r"""
uint group = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;
uint split = simdgroup_index_in_threadgroup;
uint M = dims[0], K = dims[1], N = dims[2], edge_bits = EdgeBits;
uint n_tiles = N >> 4u;
uint n_groups = (n_tiles + 3u) >> 2u;
uint row_block = group / n_groups;
uint n_group = group - row_block * n_groups;
uint row_base = row_block * RowTile;
uint tile_in_group = lane >> 3u;
uint pair_column = lane & 7u;
uint n_tile = (n_group << 2u) + tile_in_group;
uint k_tiles = K >> 4u;
uint k_begin = (k_tiles * split) / SplitCount;
uint k_end = (k_tiles * (split + 1u)) / SplitCount;
uint words_per_tile = 4u * edge_bits;
uint alt_mask = uint(qvq_p32_bank_masks[edge_bits - 2u][AltBank]);
float accum0[RowTile];
float accum1[RowTile];
for (uint row = 0u; row < RowTile; ++row) {
  accum0[row] = 0.0f;
  accum1[row] = 0.0f;
}
if (n_tile < n_tiles) {
  for (uint k_tile = k_begin; k_tile < k_end; ++k_tile) {
    uint tile_index = k_tile * n_tiles + n_tile;
    device const int* words = window + tile_index * words_per_tile;
    uint selectors = uint(bank_ids[tile_index]);
    uint input_base = k_tile << 4u;
    for (uint input_row = 0u; input_row < 8u; ++input_row) {
      uint pair0 = (input_row << 3u) + pair_column;
      uint pair8 = pair0 + 64u;
      uint bank0 = ((selectors >> (pair0 >> 4u)) & 1u) * AltBank;
      uint bank8 = ((selectors >> (pair8 >> 4u)) & 1u) * AltBank;
      float2 weight0 = qvq_p32_decode(qvq_p32_window_state(words, pair0, edge_bits), bank0, edge_bits);
      float2 weight8 = qvq_p32_decode(qvq_p32_window_state(words, pair8, edge_bits), bank8, edge_bits);
      for (uint row = 0u; row < RowTile; ++row) {
        uint output_row = row_base + row;
        if (output_row < M) {
          float input0 = float(x[output_row * K + input_base + input_row]);
          float input8 = float(x[output_row * K + input_base + input_row + 8u]);
          accum0[row] = fma(input0, weight0.x, accum0[row]);
          accum1[row] = fma(input0, weight0.y, accum1[row]);
          accum0[row] = fma(input8, weight8.x, accum0[row]);
          accum1[row] = fma(input8, weight8.y, accum1[row]);
        }
      }
    }
  }
}
threadgroup float partials[SplitCount * RowTile * 64];
uint output_pair = (tile_in_group << 4u) + (pair_column << 1u);
for (uint row = 0u; row < RowTile; ++row) {
  uint partial_base = (split * RowTile + row) * 64u + output_pair;
  partials[partial_base] = accum0[row];
  partials[partial_base + 1u] = accum1[row];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (split == 0u && n_tile < n_tiles) {
  uint output_column = (n_tile << 4u) + (pair_column << 1u);
  for (uint row = 0u; row < RowTile; ++row) {
    uint output_row = row_base + row;
    if (output_row < M) {
      float sum0 = 0.0f, sum1 = 0.0f;
      for (uint source_split = 0u; source_split < SplitCount; ++source_split) {
        uint partial_base = (source_split * RowTile + row) * 64u + output_pair;
        sum0 += partials[partial_base];
        sum1 += partials[partial_base + 1u];
      }
      out[output_row * N + output_column] = sum0;
      out[output_row * N + output_column + 1u] = sum1;
    }
  }
}
"""


def _repack_kernel(transition_bits: int):
    kernel = _REPACK_KERNELS.get(transition_bits)
    if kernel is not None:
        return kernel
    with _KERNEL_LOCK:
        kernel = _REPACK_KERNELS.get(transition_bits)
        if kernel is not None:
            return kernel
        if transition_bits in _REPACK_KERNEL_ERRORS:
            raise RuntimeError(_REPACK_KERNEL_ERRORS[transition_bits])
        import mlx.core as mx

        try:
            kernel = mx.fast.metal_kernel(
                name=f"gptqmodel_qvq_p32_repack_e{transition_bits}",
                input_names=["planar"],
                output_names=["window"],
                header=_REPACK_HEADER,
                source=_REPACK_SOURCE,
                ensure_row_contiguous=True,
            )
        except Exception as exc:
            error = f"QVQ MLX P32 repack kernel creation failed for E{transition_bits}: {exc}"
            _REPACK_KERNEL_ERRORS[transition_bits] = error
            raise RuntimeError(error) from exc
        _REPACK_KERNELS[transition_bits] = kernel
        return kernel


def qvq_mlx_repack_p32_planar_to_window(trellis, bits: float):
    """Losslessly repack canonical P32 words once for the MLX runtime layout."""

    import mlx.core as mx

    bits = normalize_qvq_rate(bits)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    if transition_bits > 7:
        raise ValueError("QVQ MLX P32 continuous-window layout supports W1 through W3.5")
    expected_words = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    if trellis.dtype != mx.int32 or trellis.ndim != 2 or trellis.shape[1] != expected_words:
        raise ValueError(
            f"QVQ MLX canonical P32 trellis must be int32 with shape [tiles, {expected_words}]"
        )
    if not trellis.size:
        return mx.empty(trellis.shape, dtype=mx.int32)
    return _repack_kernel(transition_bits)(
        inputs=[trellis],
        template=[("EdgeBits", transition_bits)],
        grid=(trellis.size, 1, 1),
        threadgroup=(min(256, trellis.size), 1, 1),
        output_shapes=[trellis.shape],
        output_dtypes=[mx.int32],
    )[0]


def _p32_gemv_kernel(transition_bits: int, row_tile: int, split_count: int, alt_bank_id: int):
    key = transition_bits, row_tile, split_count, alt_bank_id
    kernel = _GEMV_KERNELS.get(key)
    if kernel is not None:
        return kernel
    with _KERNEL_LOCK:
        kernel = _GEMV_KERNELS.get(key)
        if kernel is not None:
            return kernel
        if key in _GEMV_KERNEL_ERRORS:
            raise RuntimeError(_GEMV_KERNEL_ERRORS[key])
        import mlx.core as mx

        try:
            kernel = mx.fast.metal_kernel(
                name=f"gptqmodel_qvq_p32_window_e{transition_bits}_r{row_tile}_s{split_count}_a{alt_bank_id}",
                input_names=["x", "window", "bank_ids", "dims"],
                output_names=["out"],
                header=_GEMV_HEADER,
                source=_GEMV_SOURCE,
                ensure_row_contiguous=True,
            )
        except Exception as exc:
            error = f"QVQ MLX P32 window kernel creation failed for {key}: {exc}"
            _GEMV_KERNEL_ERRORS[key] = error
            raise RuntimeError(error) from exc
        _GEMV_KERNELS[key] = kernel
        return kernel


def _split_count(m: int, k: int, n: int) -> int:
    """Initial M4-Max split policy; every reduction remains inside one launch."""

    k_tiles = k // 16
    if m == 1:
        if k >= 8192 or n <= 2048:
            return min(32, k_tiles)
        return min(16, k_tiles)
    if m <= 4:
        return min(8, k_tiles)
    return min(4, k_tiles)


def qvq_mlx_p32_window_gemv(
    x,
    window,
    bits: float,
    *,
    out_features: int,
    bank_ids,
    bank_alt_id: int,
    _split_count_override: int | None = None,
):
    """Run exact standard-P32 from the storage-neutral continuous-window layout."""

    import mlx.core as mx

    bits = normalize_qvq_rate(bits)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    if transition_bits > 7:
        raise ValueError("QVQ MLX P32 continuous-window GEMV supports W1 through W3.5")
    if x.dtype not in (mx.float16, mx.float32) or x.ndim != 2:
        raise TypeError("QVQ MLX P32 continuous-window GEMV requires a rank-2 FP16 or FP32 activation")
    if window.dtype != mx.int32 or window.ndim != 2:
        raise TypeError("QVQ MLX P32 continuous-window GEMV requires rank-2 int32 words")
    if bank_ids.dtype != mx.uint8 or bank_ids.ndim != 1:
        raise TypeError("QVQ MLX P32 continuous-window GEMV requires packed uint8 selectors")
    if not isinstance(bank_alt_id, int) or isinstance(bank_alt_id, bool) or not 1 <= bank_alt_id <= 3:
        raise ValueError("QVQ MLX P32 alternative-bank ID must be an integer in [1, 3]")
    m, k = x.shape
    if isinstance(out_features, bool):
        raise TypeError("QVQ MLX P32 out_features must be an integer")
    try:
        n = operator.index(out_features)
    except TypeError as exc:
        raise TypeError("QVQ MLX P32 out_features must be an integer") from exc
    if k <= 0 or n <= 0 or k % 16 or n % 16:
        raise ValueError(f"QVQ MLX P32 requires positive K/N divisible by 16, got K={k}, N={n}")
    words_per_tile = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    tile_count = (k // 16) * (n // 16)
    if window.shape != (tile_count, words_per_tile):
        raise ValueError(f"QVQ MLX P32 window words must have shape {(tile_count, words_per_tile)}")
    if bank_ids.size != tile_count:
        raise ValueError(f"QVQ MLX P32 selectors must have shape {(tile_count,)}")
    if not m:
        return mx.empty((0, n), dtype=mx.float32)

    row_tile = 1 if m == 1 else 2 if m == 2 else 4
    split_count = _split_count(m, k, n)
    if _split_count_override is not None:
        if (
            not isinstance(_split_count_override, int)
            or isinstance(_split_count_override, bool)
            or _split_count_override < 1
            or _split_count_override > 32
            or _split_count_override > k // 16
        ):
            raise ValueError("QVQ MLX P32 split override must be in [1, min(K/16, 32)]")
        split_count = _split_count_override
    n_groups = ((n // 16) + 3) // 4
    row_blocks = (m + row_tile - 1) // row_tile
    dims = mx.array([m, k, n], dtype=mx.uint32)
    return _p32_gemv_kernel(transition_bits, row_tile, split_count, bank_alt_id)(
        inputs=[x, window, bank_ids, dims],
        template=[
            ("EdgeBits", transition_bits),
            ("RowTile", row_tile),
            ("SplitCount", split_count),
            ("AltBank", bank_alt_id),
        ],
        grid=(row_blocks * n_groups * split_count * 32, 1, 1),
        threadgroup=(split_count * 32, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32],
    )[0]


__all__ = [
    "qvq_mlx_p32_window_gemv",
    "qvq_mlx_repack_p32_planar_to_window",
]
