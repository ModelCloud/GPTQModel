# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fused planar QVQ inner GEMV for PyTorch MPS."""

from __future__ import annotations

import operator
import threading
from dataclasses import dataclass
from typing import Any

import torch

from ..quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS,
    pgc16_levels_for_version,
)
from ..quantization.qvq_rates import (
    QVQ_BITS,
    normalize_qvq_rate,
    qvq_transition_bits,
    qvq_words_per_tile,
)

QVQ_MPS_BITS = QVQ_BITS
_LIBRARY: Any | None = None
_LIBRARY_ERROR: str | None = None
_LIBRARY_LOCK = threading.Lock()
_HYB_REFERENCE_LIBRARY: Any | None = None
_HYB_REFERENCE_LIBRARY_ERROR: str | None = None
_HYB_REFERENCE_LIBRARY_LOCK = threading.Lock()
_PGC16_LEVELS: dict[tuple[torch.device, str], torch.Tensor] = {}
_PGC16_LEVELS_LOCK = threading.Lock()
_PGC16_LEVELS_HOT: tuple[torch.device, str, torch.Tensor] | None = None
_VITERBI_LIBRARY: Any | None = None
_VITERBI_LIBRARY_ERROR: str | None = None
_VITERBI_LIBRARY_LOCK = threading.Lock()


@dataclass(frozen=True)
class _QVQMPSPreparedCompander:
    levels: torch.Tensor


def _integer_argument(value: int, name: str) -> int:
    """Reject lossy ``int(...)`` coercions at the native-kernel boundary."""

    if isinstance(value, bool):
        raise TypeError(f"QVQ Metal {name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"QVQ Metal {name} must be an integer") from exc


def _qvq_v4_bank_masks_metal(name: str) -> str:
    rows = (
        "  {" + ", ".join(f"0x{mask:04x}u" for mask in PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS[bits]) + "},"
        for bits in range(4, 17, 2)
    )
    return f"constant ushort {name}[7][4] = {{\n" + "\n".join(rows) + "\n};"


_QVQ_V4_BANK_MASKS_METAL = _qvq_v4_bank_masks_metal("qvq_v4_bank_masks")


_COMMON_METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline uint qvq_plane_width(uint remaining) {
  if (remaining >= 16) return 16;
  if (remaining >= 8) return 8;
  if (remaining >= 4) return 4;
  if (remaining >= 2) return 2;
  return 1;
}

inline uint planar_transition(device const int* p, uint edge, uint transition_bits) {
  const uint block = edge >> 5, lane = edge & 31, base = block * transition_bits;
  uint remaining = transition_bits, row = 0, offset = 0, value = 0;
  for (uint plane = 0; plane < 4 && remaining; ++plane) {
    const uint width = qvq_plane_width(remaining);
    const uint pack_factor = 32 / width;
    const uint word = as_type<uint>(p[base + row + lane / pack_factor]);
    const uint code = (word >> (width * (lane % pack_factor))) & ((1u << width) - 1u);
    value |= code << offset;
    remaining -= width;
    row += width;
    offset += width;
  }
  return value;
}

inline uint qvq_state(device const int* tile, uint pair, uint transition_bits) {
  const uint edge_count = (15 + transition_bits) / transition_bits;
  const uint first = (pair + 128 - edge_count + 1) & 127;
  uint state = 0;
  for (uint j = 0; j < edge_count; ++j) {
    const uint edge = (first + j) & 127;
    const uint value = planar_transition(tile, edge, transition_bits);
    state = ((state << transition_bits) | value) & 0xffffu;
  }
  return state;
}

inline uint planar_transition_v4_e4(device const int* p, uint edge) {
  const uint block = edge >> 5, lane = edge & 31;
  const uint word = as_type<uint>(p[block * 4 + lane / 8]);
  return (word >> (4 * (lane & 7))) & 15u;
}

inline uint qvq_state_v4(device const int* tile, uint vector, uint transition_bits) {
  const uint edge_count = (15 + transition_bits) / transition_bits;
  const uint first = (vector + 64 - edge_count + 1) & 63;
  uint state = 0;
  for (uint j = 0; j < edge_count; ++j) {
    const uint edge = (first + j) & 63;
    const uint value = planar_transition(tile, edge, transition_bits);
    state = ((state << transition_bits) | value) & 0xffffu;
  }
  return state;
}

inline uint qvq_state_v4_e4(device const int* tile, uint vector) {
  const uint first = (vector + 61) & 63;
  uint state = 0;
  #pragma unroll
  for (uint j = 0; j < 4; ++j) {
    const uint edge = (first + j) & 63;
    state = (state << 4) | planar_transition_v4_e4(tile, edge);
  }
  return state;
}
"""

_SOURCE = (
    _COMMON_METAL_SOURCE
    + r"""
inline float2 qvq_levels_for_state(constant half* levels, uint state) {
  uint mixed = state ^ (state >> 8);
  mixed = (mixed * 40503u + 17011u) & 0xffffu;
  mixed ^= mixed >> 7;
  return float2(float(levels[mixed >> 8]), float(levels[mixed & 255u]));
}

inline float4 qvq_levels_v4_for_state(constant half* levels, uint state) {
  uint first = state ^ (state >> 8);
  first = (first * 40503u + 17011u) & 0xffffu;
  first ^= first >> 7;
  uint second = state ^ 0xa5a5u;
  second ^= second >> 8;
  second = (second * 40503u + 17011u) & 0xffffu;
  second ^= second >> 7;
  return float4(
      float(levels[first >> 8]), float(levels[first & 255u]),
      float(levels[second >> 8]), float(levels[second & 255u]));
}

__QVQ_V4_BANK_MASKS_METAL__

inline uint qvq_v4_bank_id(device const uchar* bank_ids, uint tile_idx) {
  return (uint(bank_ids[tile_idx >> 2]) >> ((tile_idx & 3u) << 1)) & 3u;
}

inline float4 qvq_levels_v4_for_state_banked(
    constant half* levels, uint state, uint bank, uint transition_bits) {
  uint first = state ^ (state >> 8);
  first = (first * 40503u + 17011u) & 0xffffu;
  first ^= first >> 7;
  const uint rate_index = (transition_bits - 4u) >> 1;
  uint second = state ^ uint(qvq_v4_bank_masks[rate_index][bank]);
  second ^= second >> 8;
  second = (second * 40503u + 17011u) & 0xffffu;
  second ^= second >> 7;
  return float4(
      float(levels[first >> 8]), float(levels[first & 255u]),
      float(levels[second >> 8]), float(levels[second & 255u]));
}

inline float2 qvq_pair(
    device const int* trellis, constant half* levels,
    uint k, uint n, uint N, uint transition_bits) {
  const uint tile_cols = N >> 4;
  const uint tile_idx = (k >> 4) * tile_cols + (n >> 4);
  const uint words_per_tile = 4 * transition_bits;
  const uint local = (k & 15) * 16 + (n & 15);
  const uint state = qvq_state(trellis + tile_idx * words_per_tile, local >> 1, transition_bits);
  return qvq_levels_for_state(levels, state);
}

inline float4 qvq_quad(
    device const int* trellis, constant half* levels,
    uint k, uint n, uint N, uint transition_bits) {
  const uint tile_cols = N >> 4;
  const uint tile_idx = (k >> 4) * tile_cols + (n >> 4);
  const uint words_per_tile = 4 * transition_bits;
  device const int* tile = trellis + tile_idx * words_per_tile;
  const uint local = (k & 15) * 16 + (n & 15);
  const uint pair = local >> 1;
  const uint state0 = qvq_state(tile, pair, transition_bits);
  const uint next_pair = pair + 1;
  const uint next_edge = planar_transition(tile, next_pair, transition_bits);
  const uint state1 = ((state0 << transition_bits) | next_edge) & 0xffffu;
  return float4(qvq_levels_for_state(levels, state0), qvq_levels_for_state(levels, state1));
}

inline float4 qvq_v4_quad(
    device const int* trellis, constant half* levels,
    uint k, uint n, uint N, uint transition_bits) {
  const uint tile_cols = N >> 4;
  const uint tile_idx = (k >> 4) * tile_cols + (n >> 4);
  const uint words_per_tile = 2 * transition_bits;
  const uint local = (k & 15) * 16 + (n & 15);
  const uint state = qvq_state_v4(
      trellis + tile_idx * words_per_tile, local >> 2, transition_bits);
  return qvq_levels_v4_for_state(levels, state);
}

inline float4 qvq_v4_quad_banked(
    device const int* trellis, device const uchar* bank_ids, constant half* levels,
    uint k, uint n, uint N, uint transition_bits) {
  const uint tile_cols = N >> 4;
  const uint tile_idx = (k >> 4) * tile_cols + (n >> 4);
  const uint words_per_tile = 2 * transition_bits;
  const uint local = (k & 15) * 16 + (n & 15);
  const uint state = qvq_state_v4(
      trellis + tile_idx * words_per_tile, local >> 2, transition_bits);
  return qvq_levels_v4_for_state_banked(
      levels, state, qvq_v4_bank_id(bank_ids, tile_idx), transition_bits);
}

inline float4 qvq_v4_quad_e4(
    device const int* trellis, constant half* levels,
    uint k, uint n, uint N) {
  const uint tile_cols = N >> 4;
  const uint tile_idx = (k >> 4) * tile_cols + (n >> 4);
  const uint local = (k & 15) * 16 + (n & 15);
  const uint state = qvq_state_v4_e4(trellis + tile_idx * 8, local >> 2);
  return qvq_levels_v4_for_state(levels, state);
}

inline float4 qvq_v4_quad_e4_banked(
    device const int* trellis, device const uchar* bank_ids, constant half* levels,
    uint k, uint n, uint N) {
  const uint tile_cols = N >> 4;
  const uint tile_idx = (k >> 4) * tile_cols + (n >> 4);
  const uint local = (k & 15) * 16 + (n & 15);
  const uint state = qvq_state_v4_e4(trellis + tile_idx * 8, local >> 2);
  return qvq_levels_v4_for_state_banked(
      levels, state, qvq_v4_bank_id(bank_ids, tile_idx), 4u);
}

template <uint TransitionBits, bool Banked>
inline float4 qvq_v4_quad_select(
    device const int* trellis, device const uchar* bank_ids, constant half* levels,
    uint k, uint n, uint N) {
  if constexpr (Banked) {
    return qvq_v4_quad_banked(trellis, bank_ids, levels, k, n, N, TransitionBits);
  }
  return qvq_v4_quad(trellis, levels, k, n, N, TransitionBits);
}

template <bool Banked>
inline float4 qvq_v4_quad_e4_select(
    device const int* trellis, device const uchar* bank_ids, constant half* levels,
    uint k, uint n, uint N) {
  if constexpr (Banked) {
    return qvq_v4_quad_e4_banked(trellis, bank_ids, levels, k, n, N);
  }
  return qvq_v4_quad_e4(trellis, levels, k, n, N);
}

template <uint TransitionBits, bool Banked>
inline void qvq_planar_v4_fp16_impl(
    device const half* x, device const int* trellis, device const uchar* bank_ids, constant half* levels,
    device half* output, uint M, uint K, uint N, uint group, uint lane) {
  const uint vector_cols = N >> 2;
  const uint m = group / vector_cols, vector = group - m * vector_cols;
  const uint n = vector << 2;
  float4 sum = 0.0f;
  for (uint k = lane; k < K; k += 32) {
    const float input = float(x[m * K + k]);
    sum += input * qvq_v4_quad_select<TransitionBits, Banked>(trellis, bank_ids, levels, k, n, N);
  }
  sum.x = simd_sum(sum.x); sum.y = simd_sum(sum.y);
  sum.z = simd_sum(sum.z); sum.w = simd_sum(sum.w);
  if (lane == 0) {
    *reinterpret_cast<device half4*>(output + m * N + n) = half4(sum);
  }
}

template <bool Banked>
inline void qvq_planar_v4_fp16_e4_fast_impl(
    device const half* x, device const int* trellis, device const uchar* bank_ids, constant half* levels,
    device half* output, uint M, uint K, uint N, uint group, uint lane) {
  const uint vector_cols = N >> 2;
  const uint m = group / vector_cols, vector = group - m * vector_cols;
  const uint n = vector << 2;
  float4 sum = 0.0f;
  for (uint k = lane; k < K; k += 32) {
    sum += float(x[m * K + k]) * qvq_v4_quad_e4_select<Banked>(trellis, bank_ids, levels, k, n, N);
  }
  sum.x = simd_sum(sum.x); sum.y = simd_sum(sum.y);
  sum.z = simd_sum(sum.z); sum.w = simd_sum(sum.w);
  if (lane == 0) {
    *reinterpret_cast<device half4*>(output + m * N + n) = half4(sum);
  }
}

template <uint TransitionBits, bool Banked>
inline void qvq_planar_v4_fp16_multirow_impl(
    device const half* x, device const int* trellis, device const uchar* bank_ids, constant half* levels,
    device half* output, threadgroup float4* decoded,
    uint M, uint K, uint N, uint row_tile,
    uint group, uint lane, uint simd) {
  const uint vector_cols = N >> 2;
  const uint row_block = group / vector_cols, vector = group - row_block * vector_cols;
  const uint row_base = row_block * row_tile, n = vector << 2;
  const uint simdgroups = row_tile >> 2;
  const uint r0 = row_base + simd, r1 = r0 + simdgroups;
  const uint r2 = r1 + simdgroups, r3 = r2 + simdgroups;
  float4 sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  for (uint base = 0; base < K; base += 32) {
    const uint k = base + lane;
    if (simd == 0) {
      decoded[lane] = k < K
          ? qvq_v4_quad_select<TransitionBits, Banked>(trellis, bank_ids, levels, k, n, N)
          : float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k < K) {
      const float4 value = decoded[lane];
      if (r0 < M) sum0 += float(x[r0 * K + k]) * value;
      if (r1 < M) sum1 += float(x[r1 * K + k]) * value;
      if (r2 < M) sum2 += float(x[r2 * K + k]) * value;
      if (r3 < M) sum3 += float(x[r3 * K + k]) * value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  sum0.x = simd_sum(sum0.x); sum0.y = simd_sum(sum0.y);
  sum0.z = simd_sum(sum0.z); sum0.w = simd_sum(sum0.w);
  sum1.x = simd_sum(sum1.x); sum1.y = simd_sum(sum1.y);
  sum1.z = simd_sum(sum1.z); sum1.w = simd_sum(sum1.w);
  sum2.x = simd_sum(sum2.x); sum2.y = simd_sum(sum2.y);
  sum2.z = simd_sum(sum2.z); sum2.w = simd_sum(sum2.w);
  sum3.x = simd_sum(sum3.x); sum3.y = simd_sum(sum3.y);
  sum3.z = simd_sum(sum3.z); sum3.w = simd_sum(sum3.w);
  if (lane == 0) {
    if (r0 < M) *reinterpret_cast<device half4*>(output + r0 * N + n) = half4(sum0);
    if (r1 < M) *reinterpret_cast<device half4*>(output + r1 * N + n) = half4(sum1);
    if (r2 < M) *reinterpret_cast<device half4*>(output + r2 * N + n) = half4(sum2);
    if (r3 < M) *reinterpret_cast<device half4*>(output + r3 * N + n) = half4(sum3);
  }
}

template <uint TransitionBits, bool Banked>
inline void qvq_planar_v4_fp16_multirow_k64_impl(
    device const half* x, device const int* trellis, device const uchar* bank_ids, constant half* levels,
    device half* output, threadgroup float4* decoded0, threadgroup float4* decoded1,
    uint M, uint K, uint N, uint row_tile,
    uint group, uint lane, uint simd) {
  const uint vector_cols = N >> 2;
  const uint row_block = group / vector_cols, vector = group - row_block * vector_cols;
  const uint row_base = row_block * row_tile, n = vector << 2;
  const uint simdgroups = row_tile >> 2;
  const uint r0 = row_base + simd, r1 = r0 + simdgroups;
  const uint r2 = r1 + simdgroups, r3 = r2 + simdgroups;
  float4 sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  for (uint base = 0; base < K; base += 64) {
    const uint k0 = base + lane, k1 = k0 + 32;
    if (simd == 0) {
      decoded0[lane] = k0 < K
          ? qvq_v4_quad_select<TransitionBits, Banked>(trellis, bank_ids, levels, k0, n, N)
          : float4(0.0f);
      decoded1[lane] = k1 < K
          ? qvq_v4_quad_select<TransitionBits, Banked>(trellis, bank_ids, levels, k1, n, N)
          : float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 < K) {
      const float4 value = decoded0[lane];
      if (r0 < M) sum0 += float(x[r0 * K + k0]) * value;
      if (r1 < M) sum1 += float(x[r1 * K + k0]) * value;
      if (r2 < M) sum2 += float(x[r2 * K + k0]) * value;
      if (r3 < M) sum3 += float(x[r3 * K + k0]) * value;
    }
    if (k1 < K) {
      const float4 value = decoded1[lane];
      if (r0 < M) sum0 += float(x[r0 * K + k1]) * value;
      if (r1 < M) sum1 += float(x[r1 * K + k1]) * value;
      if (r2 < M) sum2 += float(x[r2 * K + k1]) * value;
      if (r3 < M) sum3 += float(x[r3 * K + k1]) * value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  sum0.x = simd_sum(sum0.x); sum0.y = simd_sum(sum0.y);
  sum0.z = simd_sum(sum0.z); sum0.w = simd_sum(sum0.w);
  sum1.x = simd_sum(sum1.x); sum1.y = simd_sum(sum1.y);
  sum1.z = simd_sum(sum1.z); sum1.w = simd_sum(sum1.w);
  sum2.x = simd_sum(sum2.x); sum2.y = simd_sum(sum2.y);
  sum2.z = simd_sum(sum2.z); sum2.w = simd_sum(sum2.w);
  sum3.x = simd_sum(sum3.x); sum3.y = simd_sum(sum3.y);
  sum3.z = simd_sum(sum3.z); sum3.w = simd_sum(sum3.w);
  if (lane == 0) {
    if (r0 < M) *reinterpret_cast<device half4*>(output + r0 * N + n) = half4(sum0);
    if (r1 < M) *reinterpret_cast<device half4*>(output + r1 * N + n) = half4(sum1);
    if (r2 < M) *reinterpret_cast<device half4*>(output + r2 * N + n) = half4(sum2);
    if (r3 < M) *reinterpret_cast<device half4*>(output + r3 * N + n) = half4(sum3);
  }
}

template <uint TransitionBits, bool Banked>
inline void qvq_planar_v4_fp16_multirow_k128_impl(
    device const half* x, device const int* trellis, device const uchar* bank_ids, constant half* levels,
    device half* output, threadgroup float4* decoded0, threadgroup float4* decoded1,
    threadgroup float4* decoded2, threadgroup float4* decoded3,
    uint M, uint K, uint N, uint row_tile,
    uint group, uint lane, uint simd) {
  const uint vector_cols = N >> 2;
  const uint row_block = group / vector_cols, vector = group - row_block * vector_cols;
  const uint row_base = row_block * row_tile, n = vector << 2;
  const uint simdgroups = row_tile >> 2;
  const uint r0 = row_base + simd, r1 = r0 + simdgroups;
  const uint r2 = r1 + simdgroups, r3 = r2 + simdgroups;
  float4 sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  for (uint base = 0; base < K; base += 128) {
    const uint k0 = base + lane, k1 = k0 + 32;
    const uint k2 = k0 + 64, k3 = k0 + 96;
    if (simd == 0) {
      decoded0[lane] = k0 < K
          ? qvq_v4_quad_select<TransitionBits, Banked>(trellis, bank_ids, levels, k0, n, N) : float4(0.0f);
      decoded1[lane] = k1 < K
          ? qvq_v4_quad_select<TransitionBits, Banked>(trellis, bank_ids, levels, k1, n, N) : float4(0.0f);
      decoded2[lane] = k2 < K
          ? qvq_v4_quad_select<TransitionBits, Banked>(trellis, bank_ids, levels, k2, n, N) : float4(0.0f);
      decoded3[lane] = k3 < K
          ? qvq_v4_quad_select<TransitionBits, Banked>(trellis, bank_ids, levels, k3, n, N) : float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#define QVQ_ACCUMULATE_V4(K_VALUE, DECODED) \
    if (K_VALUE < K) { \
      const float4 value = DECODED[lane]; \
      if (r0 < M) sum0 += float(x[r0 * K + K_VALUE]) * value; \
      if (r1 < M) sum1 += float(x[r1 * K + K_VALUE]) * value; \
      if (r2 < M) sum2 += float(x[r2 * K + K_VALUE]) * value; \
      if (r3 < M) sum3 += float(x[r3 * K + K_VALUE]) * value; \
    }
    QVQ_ACCUMULATE_V4(k0, decoded0)
    QVQ_ACCUMULATE_V4(k1, decoded1)
    QVQ_ACCUMULATE_V4(k2, decoded2)
    QVQ_ACCUMULATE_V4(k3, decoded3)
#undef QVQ_ACCUMULATE_V4
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  sum0.x = simd_sum(sum0.x); sum0.y = simd_sum(sum0.y);
  sum0.z = simd_sum(sum0.z); sum0.w = simd_sum(sum0.w);
  sum1.x = simd_sum(sum1.x); sum1.y = simd_sum(sum1.y);
  sum1.z = simd_sum(sum1.z); sum1.w = simd_sum(sum1.w);
  sum2.x = simd_sum(sum2.x); sum2.y = simd_sum(sum2.y);
  sum2.z = simd_sum(sum2.z); sum2.w = simd_sum(sum2.w);
  sum3.x = simd_sum(sum3.x); sum3.y = simd_sum(sum3.y);
  sum3.z = simd_sum(sum3.z); sum3.w = simd_sum(sum3.w);
  if (lane == 0) {
    if (r0 < M) *reinterpret_cast<device half4*>(output + r0 * N + n) = half4(sum0);
    if (r1 < M) *reinterpret_cast<device half4*>(output + r1 * N + n) = half4(sum1);
    if (r2 < M) *reinterpret_cast<device half4*>(output + r2 * N + n) = half4(sum2);
    if (r3 < M) *reinterpret_cast<device half4*>(output + r3 * N + n) = half4(sum3);
  }
}

template <uint TransitionBits>
inline void qvq_planar_fp16_impl(
    device const half* x, device const int* trellis, constant half* levels,
    device half* output, uint M, uint K, uint N, uint group, uint lane) {
  const uint pair_cols = N >> 1;
  const uint m = group / pair_cols, pair = group - m * pair_cols;
  const uint n = pair << 1;
  float sum0 = 0.0f, sum1 = 0.0f;
  for (uint k = lane; k < K; k += 32) {
    const float input = float(x[m * K + k]);
    const float2 values = qvq_pair(trellis, levels, k, n, N, TransitionBits);
    sum0 += input * values.x;
    sum1 += input * values.y;
  }
  sum0 = simd_sum(sum0);
  sum1 = simd_sum(sum1);
  if (lane == 0) {
    output[m * N + n] = half(sum0);
    output[m * N + n + 1] = half(sum1);
  }
}

template <uint TransitionBits>
inline void qvq_planar_fp32_impl(
    device const half* x, device const int* trellis, constant half* levels,
    device float* output, uint M, uint K, uint N, uint group, uint lane) {
  const uint pair_cols = N >> 1;
  const uint m = group / pair_cols, pair = group - m * pair_cols;
  const uint n = pair << 1;
  float sum0 = 0.0f, sum1 = 0.0f;
  for (uint k = lane; k < K; k += 32) {
    const float input = float(x[m * K + k]);
    const float2 values = qvq_pair(trellis, levels, k, n, N, TransitionBits);
    sum0 += input * values.x;
    sum1 += input * values.y;
  }
  sum0 = simd_sum(sum0);
  sum1 = simd_sum(sum1);
  if (lane == 0) {
    output[m * N + n] = sum0;
    output[m * N + n + 1] = sum1;
  }
}

template <uint TransitionBits, bool Banked>
inline void qvq_planar_v4_fp32_impl(
    device const half* x, device const int* trellis, device const uchar* bank_ids, constant half* levels,
    device float* output, uint M, uint K, uint N, uint group, uint lane) {
  const uint vector_cols = N >> 2;
  const uint m = group / vector_cols, vector = group - m * vector_cols;
  const uint n = vector << 2;
  float4 sum = 0.0f;
  for (uint k = lane; k < K; k += 32) {
    const float input = float(x[m * K + k]);
    sum += input * qvq_v4_quad_select<TransitionBits, Banked>(trellis, bank_ids, levels, k, n, N);
  }
  sum.x = simd_sum(sum.x); sum.y = simd_sum(sum.y);
  sum.z = simd_sum(sum.z); sum.w = simd_sum(sum.w);
  if (lane == 0) {
    *reinterpret_cast<device float4*>(output + m * N + n) = sum;
  }
}

template <uint TransitionBits>
inline void qvq_planar_fp16_multirow_impl(
    device const half* x, device const int* trellis, constant half* levels,
    device half* output, threadgroup float4* decoded,
    uint M, uint K, uint N, uint row_tile,
    uint group, uint lane, uint simd) {
  const uint vector_cols = N >> 2;
  const uint row_block = group / vector_cols, vector = group - row_block * vector_cols;
  const uint row_base = row_block * row_tile, n = vector << 2;
  const uint simdgroups = row_tile >> 2;
  const uint r0 = row_base + simd, r1 = r0 + simdgroups;
  const uint r2 = r1 + simdgroups, r3 = r2 + simdgroups;
  float4 sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  for (uint base = 0; base < K; base += 32) {
    const uint k = base + lane;
    if (simd == 0) {
      decoded[lane] = k < K ? qvq_quad(trellis, levels, k, n, N, TransitionBits) : float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k < K) {
      const float4 value = decoded[lane];
      if (r0 < M) sum0 += float(x[r0 * K + k]) * value;
      if (r1 < M) sum1 += float(x[r1 * K + k]) * value;
      if (r2 < M) sum2 += float(x[r2 * K + k]) * value;
      if (r3 < M) sum3 += float(x[r3 * K + k]) * value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  sum0.x = simd_sum(sum0.x); sum0.y = simd_sum(sum0.y);
  sum0.z = simd_sum(sum0.z); sum0.w = simd_sum(sum0.w);
  sum1.x = simd_sum(sum1.x); sum1.y = simd_sum(sum1.y);
  sum1.z = simd_sum(sum1.z); sum1.w = simd_sum(sum1.w);
  sum2.x = simd_sum(sum2.x); sum2.y = simd_sum(sum2.y);
  sum2.z = simd_sum(sum2.z); sum2.w = simd_sum(sum2.w);
  sum3.x = simd_sum(sum3.x); sum3.y = simd_sum(sum3.y);
  sum3.z = simd_sum(sum3.z); sum3.w = simd_sum(sum3.w);
  if (lane == 0) {
    if (r0 < M) *reinterpret_cast<device half4*>(output + r0 * N + n) = half4(sum0);
    if (r1 < M) *reinterpret_cast<device half4*>(output + r1 * N + n) = half4(sum1);
    if (r2 < M) *reinterpret_cast<device half4*>(output + r2 * N + n) = half4(sum2);
    if (r3 < M) *reinterpret_cast<device half4*>(output + r3 * N + n) = half4(sum3);
  }
}

template <uint TransitionBits>
inline void qvq_planar_fp16_multirow_n8_impl(
    device const half* x, device const int* trellis, constant half* levels,
    device half* output, threadgroup float4* decoded0, threadgroup float4* decoded1,
    uint M, uint K, uint N, uint row_tile,
    uint group, uint lane, uint simd) {
  const uint vector_cols = N >> 3;
  const uint row_block = group / vector_cols, vector = group - row_block * vector_cols;
  const uint row_base = row_block * row_tile, n = vector << 3;
  const uint simdgroups = row_tile >> 1;
  const uint r0 = row_base + simd, r1 = r0 + simdgroups;
  float4 sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
  for (uint base = 0; base < K; base += 32) {
    const uint k = base + lane;
    if (simd == 0) {
      const float2 value0 = k < K ? qvq_pair(trellis, levels, k, n, N, TransitionBits) : float2(0.0f);
      const float2 value1 = k < K ? qvq_pair(trellis, levels, k, n + 2, N, TransitionBits) : float2(0.0f);
      const float2 value2 = k < K ? qvq_pair(trellis, levels, k, n + 4, N, TransitionBits) : float2(0.0f);
      const float2 value3 = k < K ? qvq_pair(trellis, levels, k, n + 6, N, TransitionBits) : float2(0.0f);
      decoded0[lane] = float4(value0, value1);
      decoded1[lane] = float4(value2, value3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k < K) {
      const float4 value0 = decoded0[lane], value1 = decoded1[lane];
      if (r0 < M) {
        const float input = float(x[r0 * K + k]);
        sum00 += input * value0; sum01 += input * value1;
      }
      if (r1 < M) {
        const float input = float(x[r1 * K + k]);
        sum10 += input * value0; sum11 += input * value1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  sum00.x = simd_sum(sum00.x); sum00.y = simd_sum(sum00.y);
  sum00.z = simd_sum(sum00.z); sum00.w = simd_sum(sum00.w);
  sum01.x = simd_sum(sum01.x); sum01.y = simd_sum(sum01.y);
  sum01.z = simd_sum(sum01.z); sum01.w = simd_sum(sum01.w);
  sum10.x = simd_sum(sum10.x); sum10.y = simd_sum(sum10.y);
  sum10.z = simd_sum(sum10.z); sum10.w = simd_sum(sum10.w);
  sum11.x = simd_sum(sum11.x); sum11.y = simd_sum(sum11.y);
  sum11.z = simd_sum(sum11.z); sum11.w = simd_sum(sum11.w);
  if (lane == 0) {
    if (r0 < M) {
      *reinterpret_cast<device half4*>(output + r0 * N + n) = half4(sum00);
      *reinterpret_cast<device half4*>(output + r0 * N + n + 4) = half4(sum01);
    }
    if (r1 < M) {
      *reinterpret_cast<device half4*>(output + r1 * N + n) = half4(sum10);
      *reinterpret_cast<device half4*>(output + r1 * N + n + 4) = half4(sum11);
    }
  }
}

#define QVQ_DEFINE_PLANAR_KERNEL(EDGE_BITS) \
kernel void qvq_planar_fp16_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    constant half* levels [[buffer(2)]], device half* output [[buffer(3)]], \
    constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]], \
    constant uint& N [[buffer(6)]], uint group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  qvq_planar_fp16_impl<EDGE_BITS>(x, trellis, levels, output, M, K, N, group, lane); \
} \
kernel void qvq_planar_fp16_multirow_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    constant half* levels [[buffer(2)]], device half* output [[buffer(3)]], \
    constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]], \
    constant uint& N [[buffer(6)]], constant uint& row_tile [[buffer(7)]], \
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float4 decoded[32]; \
  qvq_planar_fp16_multirow_impl<EDGE_BITS>( \
      x, trellis, levels, output, decoded, M, K, N, row_tile, group, lane, simd); \
} \
kernel void qvq_planar_fp16_multirow_n8_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    constant half* levels [[buffer(2)]], device half* output [[buffer(3)]], \
    constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]], \
    constant uint& N [[buffer(6)]], constant uint& row_tile [[buffer(7)]], \
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float4 decoded0[32], decoded1[32]; \
  qvq_planar_fp16_multirow_n8_impl<EDGE_BITS>( \
      x, trellis, levels, output, decoded0, decoded1, M, K, N, row_tile, group, lane, simd); \
}

QVQ_DEFINE_PLANAR_KERNEL(2)
QVQ_DEFINE_PLANAR_KERNEL(3)
QVQ_DEFINE_PLANAR_KERNEL(4)
QVQ_DEFINE_PLANAR_KERNEL(5)
QVQ_DEFINE_PLANAR_KERNEL(6)
QVQ_DEFINE_PLANAR_KERNEL(7)
QVQ_DEFINE_PLANAR_KERNEL(8)
QVQ_DEFINE_PLANAR_KERNEL(9)
QVQ_DEFINE_PLANAR_KERNEL(10)
QVQ_DEFINE_PLANAR_KERNEL(11)
QVQ_DEFINE_PLANAR_KERNEL(12)
QVQ_DEFINE_PLANAR_KERNEL(13)
QVQ_DEFINE_PLANAR_KERNEL(14)
QVQ_DEFINE_PLANAR_KERNEL(15)
QVQ_DEFINE_PLANAR_KERNEL(16)
#undef QVQ_DEFINE_PLANAR_KERNEL

#define QVQ_DEFINE_PLANAR_FP32_KERNEL(EDGE_BITS) \
kernel void qvq_planar_fp32_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    constant half* levels [[buffer(2)]], device float* output [[buffer(3)]], \
    constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]], \
    constant uint& N [[buffer(6)]], uint group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  qvq_planar_fp32_impl<EDGE_BITS>(x, trellis, levels, output, M, K, N, group, lane); \
}

QVQ_DEFINE_PLANAR_FP32_KERNEL(2)
QVQ_DEFINE_PLANAR_FP32_KERNEL(3)
QVQ_DEFINE_PLANAR_FP32_KERNEL(4)
QVQ_DEFINE_PLANAR_FP32_KERNEL(5)
QVQ_DEFINE_PLANAR_FP32_KERNEL(6)
QVQ_DEFINE_PLANAR_FP32_KERNEL(7)
QVQ_DEFINE_PLANAR_FP32_KERNEL(8)
QVQ_DEFINE_PLANAR_FP32_KERNEL(9)
QVQ_DEFINE_PLANAR_FP32_KERNEL(10)
QVQ_DEFINE_PLANAR_FP32_KERNEL(11)
QVQ_DEFINE_PLANAR_FP32_KERNEL(12)
QVQ_DEFINE_PLANAR_FP32_KERNEL(13)
QVQ_DEFINE_PLANAR_FP32_KERNEL(14)
QVQ_DEFINE_PLANAR_FP32_KERNEL(15)
QVQ_DEFINE_PLANAR_FP32_KERNEL(16)
#undef QVQ_DEFINE_PLANAR_FP32_KERNEL

#define QVQ_DEFINE_PLANAR_V4_KERNEL(EDGE_BITS) \
kernel void qvq_planar_v4_fp16_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    constant half* levels [[buffer(2)]], device half* output [[buffer(3)]], \
    constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]], \
    constant uint& N [[buffer(6)]], uint group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  qvq_planar_v4_fp16_impl<EDGE_BITS, false>( \
      x, trellis, reinterpret_cast<device const uchar*>(trellis), levels, output, M, K, N, group, lane); \
} \
kernel void qvq_planar_v4_fp16_multirow_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    constant half* levels [[buffer(2)]], device half* output [[buffer(3)]], \
    constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]], \
    constant uint& N [[buffer(6)]], constant uint& row_tile [[buffer(7)]], \
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float4 decoded[32]; \
  qvq_planar_v4_fp16_multirow_impl<EDGE_BITS, false>( \
      x, trellis, reinterpret_cast<device const uchar*>(trellis), levels, output, decoded, \
      M, K, N, row_tile, group, lane, simd); \
} \
kernel void qvq_planar_v4_fp16_multirow_k64_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    constant half* levels [[buffer(2)]], device half* output [[buffer(3)]], \
    constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]], \
    constant uint& N [[buffer(6)]], constant uint& row_tile [[buffer(7)]], \
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float4 decoded0[32], decoded1[32]; \
  qvq_planar_v4_fp16_multirow_k64_impl<EDGE_BITS, false>( \
      x, trellis, reinterpret_cast<device const uchar*>(trellis), levels, output, decoded0, decoded1, \
      M, K, N, row_tile, group, lane, simd); \
} \
kernel void qvq_planar_v4_fp16_multirow_k128_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    constant half* levels [[buffer(2)]], device half* output [[buffer(3)]], \
    constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]], \
    constant uint& N [[buffer(6)]], constant uint& row_tile [[buffer(7)]], \
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float4 decoded0[32], decoded1[32], decoded2[32], decoded3[32]; \
  qvq_planar_v4_fp16_multirow_k128_impl<EDGE_BITS, false>( \
      x, trellis, reinterpret_cast<device const uchar*>(trellis), levels, \
      output, decoded0, decoded1, decoded2, decoded3, \
      M, K, N, row_tile, group, lane, simd); \
}

QVQ_DEFINE_PLANAR_V4_KERNEL(4)
QVQ_DEFINE_PLANAR_V4_KERNEL(6)
QVQ_DEFINE_PLANAR_V4_KERNEL(8)
QVQ_DEFINE_PLANAR_V4_KERNEL(10)
QVQ_DEFINE_PLANAR_V4_KERNEL(12)
QVQ_DEFINE_PLANAR_V4_KERNEL(14)
QVQ_DEFINE_PLANAR_V4_KERNEL(16)
#undef QVQ_DEFINE_PLANAR_V4_KERNEL

kernel void qvq_planar_v4_fp16_e4_fast(
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]],
    constant half* levels [[buffer(2)]], device half* output [[buffer(3)]],
    constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  qvq_planar_v4_fp16_e4_fast_impl<false>(
      x, trellis, reinterpret_cast<device const uchar*>(trellis), levels, output, M, K, N, group, lane);
}

#define QVQ_DEFINE_PLANAR_V4_BANKED_KERNEL(EDGE_BITS) \
kernel void qvq_planar_v4_banked_fp16_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    device const uchar* bank_ids [[buffer(2)]], constant half* levels [[buffer(3)]], \
    device half* output [[buffer(4)]], constant uint& M [[buffer(5)]], \
    constant uint& K [[buffer(6)]], constant uint& N [[buffer(7)]], \
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
  qvq_planar_v4_fp16_impl<EDGE_BITS, true>( \
      x, trellis, bank_ids, levels, output, M, K, N, group, lane); \
} \
kernel void qvq_planar_v4_banked_fp16_multirow_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    device const uchar* bank_ids [[buffer(2)]], constant half* levels [[buffer(3)]], \
    device half* output [[buffer(4)]], constant uint& M [[buffer(5)]], \
    constant uint& K [[buffer(6)]], constant uint& N [[buffer(7)]], \
    constant uint& row_tile [[buffer(8)]], uint group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float4 decoded[32]; \
  qvq_planar_v4_fp16_multirow_impl<EDGE_BITS, true>( \
      x, trellis, bank_ids, levels, output, decoded, M, K, N, row_tile, group, lane, simd); \
} \
kernel void qvq_planar_v4_banked_fp16_multirow_k64_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    device const uchar* bank_ids [[buffer(2)]], constant half* levels [[buffer(3)]], \
    device half* output [[buffer(4)]], constant uint& M [[buffer(5)]], \
    constant uint& K [[buffer(6)]], constant uint& N [[buffer(7)]], \
    constant uint& row_tile [[buffer(8)]], uint group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float4 decoded0[32], decoded1[32]; \
  qvq_planar_v4_fp16_multirow_k64_impl<EDGE_BITS, true>( \
      x, trellis, bank_ids, levels, output, decoded0, decoded1, \
      M, K, N, row_tile, group, lane, simd); \
} \
kernel void qvq_planar_v4_banked_fp16_multirow_k128_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    device const uchar* bank_ids [[buffer(2)]], constant half* levels [[buffer(3)]], \
    device half* output [[buffer(4)]], constant uint& M [[buffer(5)]], \
    constant uint& K [[buffer(6)]], constant uint& N [[buffer(7)]], \
    constant uint& row_tile [[buffer(8)]], uint group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float4 decoded0[32], decoded1[32], decoded2[32], decoded3[32]; \
  qvq_planar_v4_fp16_multirow_k128_impl<EDGE_BITS, true>( \
      x, trellis, bank_ids, levels, output, decoded0, decoded1, decoded2, decoded3, \
      M, K, N, row_tile, group, lane, simd); \
}

QVQ_DEFINE_PLANAR_V4_BANKED_KERNEL(4)
QVQ_DEFINE_PLANAR_V4_BANKED_KERNEL(6)
QVQ_DEFINE_PLANAR_V4_BANKED_KERNEL(8)
QVQ_DEFINE_PLANAR_V4_BANKED_KERNEL(10)
QVQ_DEFINE_PLANAR_V4_BANKED_KERNEL(12)
QVQ_DEFINE_PLANAR_V4_BANKED_KERNEL(14)
QVQ_DEFINE_PLANAR_V4_BANKED_KERNEL(16)
#undef QVQ_DEFINE_PLANAR_V4_BANKED_KERNEL

#define QVQ_DEFINE_PLANAR_V4_FP32_KERNEL(EDGE_BITS) \
kernel void qvq_planar_v4_fp32_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    constant half* levels [[buffer(2)]], device float* output [[buffer(3)]], \
    constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]], \
    constant uint& N [[buffer(6)]], uint group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  qvq_planar_v4_fp32_impl<EDGE_BITS, false>( \
      x, trellis, reinterpret_cast<device const uchar*>(trellis), levels, output, M, K, N, group, lane); \
} \
kernel void qvq_planar_v4_banked_fp32_e##EDGE_BITS( \
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]], \
    device const uchar* bank_ids [[buffer(2)]], constant half* levels [[buffer(3)]], \
    device float* output [[buffer(4)]], constant uint& M [[buffer(5)]], \
    constant uint& K [[buffer(6)]], constant uint& N [[buffer(7)]], \
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
  qvq_planar_v4_fp32_impl<EDGE_BITS, true>( \
      x, trellis, bank_ids, levels, output, M, K, N, group, lane); \
}

QVQ_DEFINE_PLANAR_V4_FP32_KERNEL(4)
QVQ_DEFINE_PLANAR_V4_FP32_KERNEL(6)
QVQ_DEFINE_PLANAR_V4_FP32_KERNEL(8)
QVQ_DEFINE_PLANAR_V4_FP32_KERNEL(10)
QVQ_DEFINE_PLANAR_V4_FP32_KERNEL(12)
QVQ_DEFINE_PLANAR_V4_FP32_KERNEL(14)
QVQ_DEFINE_PLANAR_V4_FP32_KERNEL(16)
#undef QVQ_DEFINE_PLANAR_V4_FP32_KERNEL

kernel void qvq_planar_v4_banked_fp16_e4_fast(
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]],
    device const uchar* bank_ids [[buffer(2)]], constant half* levels [[buffer(3)]],
    device half* output [[buffer(4)]], constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]], constant uint& N [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  qvq_planar_v4_fp16_e4_fast_impl<true>(
      x, trellis, bank_ids, levels, output, M, K, N, group, lane);
}

    """.replace("__QVQ_V4_BANK_MASKS_METAL__", _QVQ_V4_BANK_MASKS_METAL)
)

_VITERBI_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint kStateCount = 65536;

inline float qvq_emission(device const float* target, device const float* codebook, uint state, float weight) {
  const float x = target[0], y = target[1];
  const float cx = codebook[2 * state], cy = codebook[2 * state + 1];
  const float target_norm = x * x + y * y;
  const float codebook_norm = cx * cx + cy * cy;
  const float dot = fma(y, cy, x * cx);
  return max(target_norm + codebook_norm - 2.0f * dot, 0.0f) * weight;
}

template <uint TransitionBits>
inline void qvq_viterbi_lowrate(
    device const float* sequences,
    device const float* codebook,
    device const long* overlap,
    device const float* step_weights,
    device float* costs,
    device float* next_costs,
    device ushort* backpointers,
    device long* states,
    device float* squared_error,
    constant uint* dimensions,
    threadgroup float* reduction_cost,
    threadgroup uint* reduction_state,
    uint batch,
    uint tid) {
  constexpr uint prefix_count = 1u << TransitionBits;
  constexpr uint suffix_count = kStateCount / prefix_count;
  constexpr uint overlap_bits = 16 - TransitionBits;
  constexpr uint overlap_mask = (1u << overlap_bits) - 1u;
  const uint steps = dimensions[1];
  const bool constrained = dimensions[2] != 0;
  const bool weighted = dimensions[3] != 0;
  const ulong sequence_base = ulong(batch) * steps * 2;
  const ulong cost_base = ulong(batch) * kStateCount;
  const ulong pointer_base = ulong(batch) * (steps - 1) * suffix_count;
  const ulong state_base = ulong(batch) * steps;
  const long required_overlap = constrained ? overlap[batch] : 0;
  device float* current = costs + cost_base;
  device float* following = next_costs + cost_base;
  constexpr uint thread_count = TransitionBits == 2 ? 1024 : 512;

  const float first_weight = weighted ? step_weights[ulong(batch) * steps] : 1.0f;
  for (uint state = tid; state < kStateCount; state += thread_count) {
    float value = qvq_emission(sequences + sequence_base, codebook, state, first_weight);
    if (constrained && long(state >> TransitionBits) != required_overlap) value = INFINITY;
    current[state] = value;
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (uint step = 1; step < steps; ++step) {
    device const float* target = sequences + sequence_base + ulong(step) * 2;
    const float step_weight = weighted ? step_weights[ulong(batch) * steps + step] : 1.0f;
    const float target_norm = fma(target[1], target[1], target[0] * target[0]);
    for (uint suffix = tid; suffix < suffix_count; suffix += thread_count) {
      float best = current[suffix];
      ushort best_prefix = 0;
      for (uint prefix = 1; prefix < prefix_count; ++prefix) {
        const float candidate = current[ulong(prefix) * suffix_count + suffix];
        if (candidate < best) { best = candidate; best_prefix = ushort(prefix); }
      }
      backpointers[pointer_base + ulong(step - 1) * suffix_count + suffix] = best_prefix;
      if constexpr (TransitionBits == 2) {
        const uint state = suffix * 4;
        const float4 lo = *reinterpret_cast<device const float4*>(codebook + 2 * state);
        const float4 hi = *reinterpret_cast<device const float4*>(codebook + 2 * state + 4);
        const float4 cx = float4(lo.x, lo.z, hi.x, hi.z);
        const float4 cy = float4(lo.y, lo.w, hi.y, hi.w);
        const float4 codebook_norm = fma(cy, cy, cx * cx);
        const float4 dot = fma(float4(target[1]), cy, float4(target[0]) * cx);
        const float4 emission = max(float4(target_norm) + codebook_norm - 2.0f * dot, float4(0.0f));
        *reinterpret_cast<device float4*>(following + state) = float4(best) + emission * step_weight;
      } else if constexpr (TransitionBits == 3) {
        const uint state = suffix * 8;
        for (uint vector_index = 0; vector_index < 2; ++vector_index) {
          const uint base = state + vector_index * 4;
          const float4 lo = *reinterpret_cast<device const float4*>(codebook + 2 * base);
          const float4 hi = *reinterpret_cast<device const float4*>(codebook + 2 * base + 4);
          const float4 cx = float4(lo.x, lo.z, hi.x, hi.z);
          const float4 cy = float4(lo.y, lo.w, hi.y, hi.w);
          const float4 codebook_norm = fma(cy, cy, cx * cx);
          const float4 dot = fma(float4(target[1]), cy, float4(target[0]) * cx);
          const float4 emission = max(float4(target_norm) + codebook_norm - 2.0f * dot, float4(0.0f));
          *reinterpret_cast<device float4*>(following + base) = float4(best) + emission * step_weight;
        }
      } else {
        const uint state = suffix * prefix_count;
        for (uint edge = 0; edge < prefix_count; edge += 4) {
          const uint base = state + edge;
          const float4 lo = *reinterpret_cast<device const float4*>(codebook + 2 * base);
          const float4 hi = *reinterpret_cast<device const float4*>(codebook + 2 * base + 4);
          const float4 cx = float4(lo.x, lo.z, hi.x, hi.z);
          const float4 cy = float4(lo.y, lo.w, hi.y, hi.w);
          const float4 codebook_norm = fma(cy, cy, cx * cx);
          const float4 dot = fma(float4(target[1]), cy, float4(target[0]) * cx);
          const float4 emission = max(float4(target_norm) + codebook_norm - 2.0f * dot, float4(0.0f));
          *reinterpret_cast<device float4*>(following + base) = float4(best) + emission * step_weight;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_device);
    device float* swap = current; current = following; following = swap;
  }

  float best = INFINITY;
  uint best_state = kStateCount;
  for (uint state = tid; state < kStateCount; state += thread_count) {
    if (constrained && (state & overlap_mask) != uint(required_overlap)) continue;
    const float candidate = current[state];
    if (candidate < best || (candidate == best && state < best_state)) {
      best = candidate; best_state = state;
    }
  }
  reduction_cost[tid] = best;
  reduction_state[tid] = best_state;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    for (uint lane = 1; lane < thread_count; ++lane) {
      const float candidate = reduction_cost[lane];
      const uint candidate_state = reduction_state[lane];
      if (candidate < best || (candidate == best && candidate_state < best_state)) {
        best = candidate; best_state = candidate_state;
      }
    }
    squared_error[batch] = best;
    states[state_base + steps - 1] = long(best_state);
    for (uint step = steps - 1; step > 0; --step) {
      const uint suffix = best_state >> TransitionBits;
      const uint prefix = uint(backpointers[pointer_base + ulong(step - 1) * suffix_count + suffix]);
      best_state = prefix * suffix_count + suffix;
      states[state_base + step - 1] = long(best_state);
    }
  }
}

#define QVQ_VITERBI_KERNEL(BITS) \
kernel void qvq_viterbi_e##BITS( \
    device const float* sequences [[buffer(0)]], device const float* codebook [[buffer(1)]], \
    device const long* overlap [[buffer(2)]], device const float* step_weights [[buffer(3)]], \
    device float* costs [[buffer(4)]], device float* next_costs [[buffer(5)]], \
    device ushort* backpointers [[buffer(6)]], device long* states [[buffer(7)]], \
    device float* squared_error [[buffer(8)]], constant uint* dimensions [[buffer(9)]], \
    uint batch [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
  threadgroup float reduction_cost[1024]; \
  threadgroup uint reduction_state[1024]; \
  qvq_viterbi_lowrate<BITS>(sequences, codebook, overlap, step_weights, costs, next_costs, \
      backpointers, states, squared_error, dimensions, reduction_cost, reduction_state, batch, tid); \
}
QVQ_VITERBI_KERNEL(2)
QVQ_VITERBI_KERNEL(3)
QVQ_VITERBI_KERNEL(4)
QVQ_VITERBI_KERNEL(5)
QVQ_VITERBI_KERNEL(6)
QVQ_VITERBI_KERNEL(7)
QVQ_VITERBI_KERNEL(8)

template <uint TransitionBits>
inline void qvq_viterbi_highrate(
    device const float* sequences, device const float* codebook, device const long* overlap,
    device const float* step_weights, device float* costs, device float* next_costs,
    device ushort* backpointers, device long* states, device float* squared_error,
    constant uint* dimensions, threadgroup float* reduction_cost, threadgroup uint* reduction_state,
    threadgroup float* suffix_cost, uint batch, uint tid, uint simd_lane, uint simd_group) {
  constexpr uint prefix_count = 1u << TransitionBits;
  constexpr uint suffix_count = kStateCount / prefix_count;
  constexpr uint lanes_per_suffix = 1024 / suffix_count;
  constexpr uint overlap_mask = suffix_count - 1;
  const uint steps = dimensions[1];
  const bool constrained = dimensions[2] != 0, weighted = dimensions[3] != 0;
  const ulong sequence_base = ulong(batch) * steps * 2, cost_base = ulong(batch) * kStateCount;
  const ulong pointer_base = ulong(batch) * (steps - 1) * suffix_count, state_base = ulong(batch) * steps;
  const uint required = constrained ? uint(overlap[batch]) : 0;
  device float* current = costs + cost_base; device float* following = next_costs + cost_base;
  const float first_weight = weighted ? step_weights[ulong(batch) * steps] : 1.0f;
  device const float* first_target = sequences + sequence_base;
  const float first_target_norm = fma(first_target[1], first_target[1], first_target[0] * first_target[0]);
  for (uint state = tid * 4; state < kStateCount; state += 4096) {
    const float4 lo = *reinterpret_cast<device const float4*>(codebook + 2 * state);
    const float4 hi = *reinterpret_cast<device const float4*>(codebook + 2 * state + 4);
    const float4 cx = float4(lo.x, lo.z, hi.x, hi.z), cy = float4(lo.y, lo.w, hi.y, hi.w);
    const float4 dot = fma(float4(first_target[1]), cy, float4(first_target[0]) * cx);
    float4 value = max(float4(first_target_norm) + fma(cy, cy, cx * cx) - 2.0f * dot, float4(0.0f));
    value *= first_weight;
    if (constrained && (state >> TransitionBits) != required) value = float4(INFINITY);
    *reinterpret_cast<device float4*>(current + state) = value;
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint step = 1; step < steps; ++step) {
    const uint suffix = tid / lanes_per_suffix, lane = tid - suffix * lanes_per_suffix;
    float best = INFINITY; uint best_prefix = prefix_count;
    for (uint prefix = lane; prefix < prefix_count; prefix += lanes_per_suffix) {
      const float candidate = current[ulong(prefix) * suffix_count + suffix];
      if (candidate < best || (candidate == best && prefix < best_prefix)) {
        best = candidate; best_prefix = prefix;
      }
    }
    if constexpr (TransitionBits == 9 || TransitionBits == 10) {
      for (uint offset = 1; offset < lanes_per_suffix; offset <<= 1) {
        const float candidate = simd_shuffle_xor(best, offset);
        const uint prefix = simd_shuffle_xor(best_prefix, offset);
        if (candidate < best || (candidate == best && prefix < best_prefix)) {
          best = candidate; best_prefix = prefix;
        }
      }
      if (lane == 0) {
        suffix_cost[suffix] = best;
        backpointers[pointer_base + ulong(step - 1) * suffix_count + suffix] = ushort(best_prefix);
      }
    } else if constexpr (TransitionBits == 12) {
      const float simd_best = simd_min(best);
      const uint simd_prefix = simd_min(best == simd_best ? best_prefix : prefix_count);
      if (simd_lane == 0) {
        reduction_cost[simd_group] = simd_best;
        reduction_state[simd_group] = simd_prefix;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane == 0) {
        const uint group_base = suffix * 2;
        best = reduction_cost[group_base]; best_prefix = reduction_state[group_base];
        const float candidate = reduction_cost[group_base + 1];
        const uint prefix = reduction_state[group_base + 1];
        if (candidate < best || (candidate == best && prefix < best_prefix)) {
          best = candidate; best_prefix = prefix;
        }
        suffix_cost[suffix] = best;
        backpointers[pointer_base + ulong(step - 1) * suffix_count + suffix] = ushort(best_prefix);
      }
    } else if constexpr (TransitionBits == 11) {
      const float simd_best = simd_min(best);
      const uint simd_prefix = simd_min(best == simd_best ? best_prefix : prefix_count);
      if (simd_lane == 0) {
        suffix_cost[suffix] = simd_best;
        backpointers[pointer_base + ulong(step - 1) * suffix_count + suffix] = ushort(simd_prefix);
      }
    } else {
      reduction_cost[tid] = best; reduction_state[tid] = best_prefix;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane == 0) {
        for (uint other = 1; other < lanes_per_suffix; ++other) {
          const uint index = tid + other; const float candidate = reduction_cost[index];
          const uint prefix = reduction_state[index];
          if (candidate < best || (candidate == best && prefix < best_prefix)) {
            best = candidate; best_prefix = prefix;
          }
        }
        suffix_cost[suffix] = best;
        backpointers[pointer_base + ulong(step - 1) * suffix_count + suffix] = ushort(best_prefix);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* target = sequences + sequence_base + ulong(step) * 2;
    const float sw = weighted ? step_weights[ulong(batch) * steps + step] : 1.0f;
    const float target_norm = fma(target[1], target[1], target[0] * target[0]);
    for (uint state = tid * 4; state < kStateCount; state += 4096) {
      const float4 lo = *reinterpret_cast<device const float4*>(codebook + 2 * state);
      const float4 hi = *reinterpret_cast<device const float4*>(codebook + 2 * state + 4);
      const float4 cx = float4(lo.x, lo.z, hi.x, hi.z), cy = float4(lo.y, lo.w, hi.y, hi.w);
      const float4 dot = fma(float4(target[1]), cy, float4(target[0]) * cx);
      const float4 emission = max(float4(target_norm) + fma(cy, cy, cx * cx) - 2.0f * dot, float4(0.0f));
      *reinterpret_cast<device float4*>(following + state) = float4(suffix_cost[state >> TransitionBits]) + emission * sw;
    }
    threadgroup_barrier(mem_flags::mem_device);
    device float* swap = current; current = following; following = swap;
  }
  float best = INFINITY; uint best_state = kStateCount;
  for (uint state = tid * 4; state < kStateCount; state += 4096) {
    for (uint state_lane = 0; state_lane < 4; ++state_lane) {
      const uint candidate_state = state + state_lane;
      if (constrained && (candidate_state & overlap_mask) != required) continue;
      const float candidate = current[candidate_state];
      if (candidate < best || (candidate == best && candidate_state < best_state)) {
        best = candidate; best_state = candidate_state;
      }
    }
  }
  reduction_cost[tid] = best; reduction_state[tid] = best_state;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    for (uint lane = 1; lane < 1024; ++lane) {
      const float candidate = reduction_cost[lane]; const uint state = reduction_state[lane];
      if (candidate < best || (candidate == best && state < best_state)) { best = candidate; best_state = state; }
    }
    squared_error[batch] = best; states[state_base + steps - 1] = long(best_state);
    for (uint step = steps - 1; step > 0; --step) {
      const uint suffix = best_state >> TransitionBits;
      const uint prefix = uint(backpointers[pointer_base + ulong(step - 1) * suffix_count + suffix]);
      best_state = prefix * suffix_count + suffix; states[state_base + step - 1] = long(best_state);
    }
  }
}

#define QVQ_HIGHRATE_KERNEL(BITS) \
kernel void qvq_viterbi_e##BITS( \
    device const float* sequences [[buffer(0)]], device const float* codebook [[buffer(1)]], \
    device const long* overlap [[buffer(2)]], device const float* step_weights [[buffer(3)]], \
    device float* costs [[buffer(4)]], device float* next_costs [[buffer(5)]], \
    device ushort* backpointers [[buffer(6)]], device long* states [[buffer(7)]], \
    device float* squared_error [[buffer(8)]], constant uint* dimensions [[buffer(9)]], \
    uint batch [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float reduction_cost[1024]; threadgroup uint reduction_state[1024]; \
  threadgroup float suffix_cost[128]; \
  qvq_viterbi_highrate<BITS>(sequences, codebook, overlap, step_weights, costs, next_costs, backpointers, \
      states, squared_error, dimensions, reduction_cost, reduction_state, suffix_cost, batch, tid, simd_lane, \
      simd_group); \
}
QVQ_HIGHRATE_KERNEL(9)
QVQ_HIGHRATE_KERNEL(10)
QVQ_HIGHRATE_KERNEL(11)
QVQ_HIGHRATE_KERNEL(12)
QVQ_HIGHRATE_KERNEL(13)
QVQ_HIGHRATE_KERNEL(14)
QVQ_HIGHRATE_KERNEL(15)

kernel void qvq_viterbi_e16_independent(
    device const float* sequences [[buffer(0)]], device const float* codebook [[buffer(1)]],
    device const float* step_weights [[buffer(2)]], device long* states [[buffer(3)]],
    device float* step_losses [[buffer(4)]], constant uint* dimensions [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint steps = dimensions[1], batch = group / steps, step = group - batch * steps;
  const bool weighted = dimensions[3] != 0;
  const ulong index = ulong(batch) * steps + step;
  const float weight = weighted ? step_weights[index] : 1.0f;
  device const float* target = sequences + index * 2;
  float best = INFINITY; uint best_state = kStateCount;
  const float target_norm = fma(target[1], target[1], target[0] * target[0]);
  for (uint state = tid * 4; state < kStateCount; state += 1024) {
    const float4 lo = *reinterpret_cast<device const float4*>(codebook + 2 * state);
    const float4 hi = *reinterpret_cast<device const float4*>(codebook + 2 * state + 4);
    const float4 cx = float4(lo.x, lo.z, hi.x, hi.z), cy = float4(lo.y, lo.w, hi.y, hi.w);
    const float4 dot = fma(float4(target[1]), cy, float4(target[0]) * cx);
    const float4 candidates = max(float4(target_norm) + fma(cy, cy, cx * cx) - 2.0f * dot, float4(0.0f)) * weight;
    for (uint lane = 0; lane < 4; ++lane) {
      const float candidate = candidates[lane]; const uint candidate_state = state + lane;
      if (candidate < best || (candidate == best && candidate_state < best_state)) {
        best = candidate; best_state = candidate_state;
      }
    }
  }
  threadgroup float reduction_cost[256]; threadgroup uint reduction_state[256];
  reduction_cost[tid] = best; reduction_state[tid] = best_state;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    for (uint lane = 1; lane < 256; ++lane) {
      const float candidate = reduction_cost[lane]; const uint state = reduction_state[lane];
      if (candidate < best || (candidate == best && state < best_state)) {
        best = candidate; best_state = state;
      }
    }
    states[index] = long(best_state); step_losses[index] = best;
  }
}

kernel void qvq_viterbi_e16_finalize(
    device const float* step_losses [[buffer(0)]], device float* squared_error [[buffer(1)]],
    constant uint* dimensions [[buffer(2)]], uint batch [[thread_position_in_grid]]) {
  const uint steps = dimensions[1]; float total = 0.0f;
  for (uint step = 0; step < steps; ++step) total += step_losses[ulong(batch) * steps + step];
  squared_error[batch] = total;
}

kernel void qvq_viterbi_e15_transitions(
    device const float* sequences [[buffer(0)]], device const float* codebook [[buffer(1)]],
    device const float* step_weights [[buffer(2)]], device float* transition_costs [[buffer(3)]],
    device uint* transition_states [[buffer(4)]], constant uint* dimensions [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  const uint steps = dimensions[1], category = group & 3, item = group >> 2;
  const uint batch = item / steps, step = item - batch * steps;
  const bool weighted = dimensions[3] != 0; const ulong index = ulong(batch) * steps + step;
  const float weight = weighted ? step_weights[index] : 1.0f;
  device const float* target = sequences + index * 2;
  const uint previous = category >> 1, next = category & 1;
  float best = INFINITY; uint best_state = kStateCount;
  for (uint middle = tid; middle < 16384; middle += 64) {
    const uint state = (previous << 15) | (middle << 1) | next;
    const float candidate = qvq_emission(target, codebook, state, weight);
    if (candidate < best || (candidate == best && state < best_state)) { best = candidate; best_state = state; }
  }
  const float simd_best = simd_min(best);
  const uint simd_state = simd_min(best == simd_best ? best_state : kStateCount);
  threadgroup float reduction_cost[2]; threadgroup uint reduction_state[2];
  if (lane == 0) { reduction_cost[simdgroup] = simd_best; reduction_state[simdgroup] = simd_state; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    best = reduction_cost[0]; best_state = reduction_state[0];
    const float candidate = reduction_cost[1]; const uint state = reduction_state[1];
    if (candidate < best || (candidate == best && state < best_state)) {
      best = candidate; best_state = state;
    }
    transition_costs[index * 4 + category] = best; transition_states[index * 4 + category] = best_state;
  }
}

kernel void qvq_viterbi_e15_finalize(
    device const float* transition_costs [[buffer(0)]], device const uint* transition_states [[buffer(1)]],
    device const long* overlap [[buffer(2)]], device long* states [[buffer(3)]],
    device float* squared_error [[buffer(4)]], device uchar* context_backpointers [[buffer(5)]],
    constant uint* dimensions [[buffer(6)]], uint batch [[thread_position_in_grid]]) {
  const uint steps = dimensions[1]; const bool constrained = dimensions[2] != 0;
  const uint required = constrained ? uint(overlap[batch]) : 0; const ulong base = ulong(batch) * steps;
  float cost[2] = {INFINITY, INFINITY}; uint end_state[2] = {kStateCount, kStateCount};
  for (uint next = 0; next < 2; ++next) {
    for (uint previous = 0; previous < 2; ++previous) {
      if (constrained && previous != required) continue;
      const uint category = previous * 2 + next; const float candidate = transition_costs[base * 4 + category];
      const uint state = transition_states[base * 4 + category];
      if (candidate < cost[next] || (candidate == cost[next] && state < end_state[next])) {
        cost[next] = candidate; end_state[next] = state;
        context_backpointers[base * 2 + next] = uchar(previous);
      }
    }
  }
  for (uint step = 1; step < steps; ++step) {
    float next_cost[2] = {INFINITY, INFINITY}; uint next_state[2] = {kStateCount, kStateCount};
    for (uint next = 0; next < 2; ++next) {
      uint best_previous = 0;
      for (uint previous = 0; previous < 2; ++previous) {
        const uint category = previous * 2 + next;
        const float candidate = cost[previous] + transition_costs[(base + step) * 4 + category];
        const uint predecessor_state = end_state[previous];
        if (candidate < next_cost[next]) {
          next_cost[next] = candidate; next_state[next] = transition_states[(base + step) * 4 + category];
          best_previous = previous;
        }
      }
      context_backpointers[base * 2 + ulong(step) * 2 + next] = uchar(best_previous);
    }
    cost[0] = next_cost[0]; cost[1] = next_cost[1]; end_state[0] = next_state[0]; end_state[1] = next_state[1];
  }
  uint context = 0;
  if (constrained) context = required;
  else if (cost[1] < cost[0] || (cost[1] == cost[0] && end_state[1] < end_state[0])) context = 1;
  squared_error[batch] = cost[context];
  for (uint step = steps; step-- > 0;) {
    uint previous = uint(context_backpointers[base * 2 + ulong(step) * 2 + context]);
    states[base + step] = long(transition_states[(base + step) * 4 + previous * 2 + context]);
    context = previous;
  }
}

kernel void qvq_viterbi_e14_transitions(
    device const float* sequences [[buffer(0)]], device const float* codebook [[buffer(1)]],
    device const float* step_weights [[buffer(2)]], device float* transition_costs [[buffer(3)]],
    device uint* transition_states [[buffer(4)]], constant uint* dimensions [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  const uint steps = dimensions[1], category = group & 15, item = group >> 4;
  const uint batch = item / steps, step = item - batch * steps;
  const uint previous = category >> 2, next = category & 3;
  const bool weighted = dimensions[3] != 0; const ulong index = ulong(batch) * steps + step;
  const float weight = weighted ? step_weights[index] : 1.0f;
  device const float* target = sequences + index * 2;
  float best = INFINITY; uint best_state = kStateCount;
  for (uint middle = tid; middle < 4096; middle += 64) {
    const uint state = (previous << 14) | (middle << 2) | next;
    const float candidate = qvq_emission(target, codebook, state, weight);
    if (candidate < best || (candidate == best && state < best_state)) { best = candidate; best_state = state; }
  }
  const float simd_best = simd_min(best);
  const uint simd_state = simd_min(best == simd_best ? best_state : kStateCount);
  threadgroup float reduction_cost[2]; threadgroup uint reduction_state[2];
  if (lane == 0) { reduction_cost[simdgroup] = simd_best; reduction_state[simdgroup] = simd_state; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    best = reduction_cost[0]; best_state = reduction_state[0];
    const float candidate = reduction_cost[1]; const uint state = reduction_state[1];
    if (candidate < best || (candidate == best && state < best_state)) { best = candidate; best_state = state; }
    transition_costs[index * 16 + category] = best; transition_states[index * 16 + category] = best_state;
  }
}

kernel void qvq_viterbi_e14_finalize(
    device const float* transition_costs [[buffer(0)]], device const uint* transition_states [[buffer(1)]],
    device const long* overlap [[buffer(2)]], device long* states [[buffer(3)]],
    device float* squared_error [[buffer(4)]], device uchar* context_backpointers [[buffer(5)]],
    constant uint* dimensions [[buffer(6)]], uint batch [[thread_position_in_grid]]) {
  const uint steps = dimensions[1]; const bool constrained = dimensions[2] != 0;
  const uint required = constrained ? uint(overlap[batch]) : 0; const ulong base = ulong(batch) * steps;
  float cost[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
  uint end_state[4] = {kStateCount, kStateCount, kStateCount, kStateCount};
  for (uint next = 0; next < 4; ++next) {
    for (uint previous = 0; previous < 4; ++previous) {
      if (constrained && previous != required) continue;
      const uint category = previous * 4 + next; const float candidate = transition_costs[base * 16 + category];
      const uint state = transition_states[base * 16 + category];
      if (candidate < cost[next] || (candidate == cost[next] && state < end_state[next])) {
        cost[next] = candidate; end_state[next] = state;
        context_backpointers[base * 4 + next] = uchar(previous);
      }
    }
  }
  for (uint step = 1; step < steps; ++step) {
    float next_cost[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
    uint next_state[4] = {kStateCount, kStateCount, kStateCount, kStateCount};
    for (uint next = 0; next < 4; ++next) {
      uint best_previous = 0;
      for (uint previous = 0; previous < 4; ++previous) {
        const uint category = previous * 4 + next;
        const float candidate = cost[previous] + transition_costs[(base + step) * 16 + category];
        if (candidate < next_cost[next]) {
          next_cost[next] = candidate; next_state[next] = transition_states[(base + step) * 16 + category];
          best_previous = previous;
        }
      }
      context_backpointers[base * 4 + ulong(step) * 4 + next] = uchar(best_previous);
    }
    for (uint context = 0; context < 4; ++context) {
      cost[context] = next_cost[context]; end_state[context] = next_state[context];
    }
  }
  uint context = constrained ? required : 0;
  if (!constrained) for (uint candidate = 1; candidate < 4; ++candidate)
    if (cost[candidate] < cost[context] ||
        (cost[candidate] == cost[context] && end_state[candidate] < end_state[context])) context = candidate;
  squared_error[batch] = cost[context];
  for (uint step = steps; step-- > 0;) {
    const uint previous = uint(context_backpointers[base * 4 + ulong(step) * 4 + context]);
    states[base + step] = long(transition_states[(base + step) * 16 + previous * 4 + context]);
    context = previous;
  }
}

kernel void qvq_viterbi_e13_transitions(
    device const float* sequences [[buffer(0)]], device const float* codebook [[buffer(1)]],
    device const float* step_weights [[buffer(2)]], device float* transition_costs [[buffer(3)]],
    device uint* transition_states [[buffer(4)]], constant uint* dimensions [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint steps = dimensions[1], category = group & 63, item = group >> 6;
  const uint batch = item / steps, step = item - batch * steps;
  const uint previous = category >> 3, next = category & 7;
  const bool weighted = dimensions[3] != 0; const ulong index = ulong(batch) * steps + step;
  const float weight = weighted ? step_weights[index] : 1.0f;
  device const float* target = sequences + index * 2;
  float best = INFINITY; uint best_state = kStateCount;
  for (uint middle = tid; middle < 1024; middle += 32) {
    const uint state = (previous << 13) | (middle << 3) | next;
    const float candidate = qvq_emission(target, codebook, state, weight);
    if (candidate < best || (candidate == best && state < best_state)) { best = candidate; best_state = state; }
  }
  const float simd_best = simd_min(best);
  const uint simd_state = simd_min(best == simd_best ? best_state : kStateCount);
  if (lane == 0) {
    transition_costs[index * 64 + category] = simd_best;
    transition_states[index * 64 + category] = simd_state;
  }
}

kernel void qvq_viterbi_e13_finalize(
    device const float* transition_costs [[buffer(0)]], device const uint* transition_states [[buffer(1)]],
    device const long* overlap [[buffer(2)]], device long* states [[buffer(3)]],
    device float* squared_error [[buffer(4)]], device uchar* context_backpointers [[buffer(5)]],
    constant uint* dimensions [[buffer(6)]], uint batch [[thread_position_in_grid]]) {
  const uint steps = dimensions[1]; const bool constrained = dimensions[2] != 0;
  const uint required = constrained ? uint(overlap[batch]) : 0; const ulong base = ulong(batch) * steps;
  float cost[8]; uint end_state[8];
  for (uint context = 0; context < 8; ++context) { cost[context] = INFINITY; end_state[context] = kStateCount; }
  for (uint next = 0; next < 8; ++next) for (uint previous = 0; previous < 8; ++previous) {
    if (constrained && previous != required) continue;
    const uint category = previous * 8 + next; const float candidate = transition_costs[base * 64 + category];
    const uint state = transition_states[base * 64 + category];
    if (candidate < cost[next] || (candidate == cost[next] && state < end_state[next])) {
      cost[next] = candidate; end_state[next] = state;
      context_backpointers[base * 8 + next] = uchar(previous);
    }
  }
  for (uint step = 1; step < steps; ++step) {
    float next_cost[8]; uint next_state[8];
    for (uint context = 0; context < 8; ++context) { next_cost[context] = INFINITY; next_state[context] = kStateCount; }
    for (uint next = 0; next < 8; ++next) {
      uint best_previous = 0;
      for (uint previous = 0; previous < 8; ++previous) {
        const uint category = previous * 8 + next;
        const float candidate = cost[previous] + transition_costs[(base + step) * 64 + category];
        if (candidate < next_cost[next]) {
          next_cost[next] = candidate; next_state[next] = transition_states[(base + step) * 64 + category];
          best_previous = previous;
        }
      }
      context_backpointers[base * 8 + ulong(step) * 8 + next] = uchar(best_previous);
    }
    for (uint context = 0; context < 8; ++context) {
      cost[context] = next_cost[context]; end_state[context] = next_state[context];
    }
  }
  uint context = constrained ? required : 0;
  if (!constrained) for (uint candidate = 1; candidate < 8; ++candidate)
    if (cost[candidate] < cost[context] ||
        (cost[candidate] == cost[context] && end_state[candidate] < end_state[context])) context = candidate;
  squared_error[batch] = cost[context];
  for (uint step = steps; step-- > 0;) {
    const uint previous = uint(context_backpointers[base * 8 + ulong(step) * 8 + context]);
    states[base + step] = long(transition_states[(base + step) * 64 + previous * 8 + context]);
    context = previous;
  }
}

template <uint TransitionBits>
inline void qvq_overlap_scores_lowrate(
    device const float* sequences, device const float* codebook, device const float* step_weights,
    device float* costs, device float* next_costs, device float* forward_costs,
    device float* overlap_scores, constant uint* dimensions, uint batch, uint tid) {
  constexpr uint prefix_count = 1u << TransitionBits;
  constexpr uint suffix_count = kStateCount / prefix_count;
  constexpr uint overlap_mask = suffix_count - 1;
  const uint steps = dimensions[1], boundary = dimensions[2];
  const bool weighted = dimensions[3] != 0;
  const ulong sequence_base = ulong(batch) * steps * 2;
  const ulong cost_base = ulong(batch) * kStateCount;
  const ulong score_base = ulong(batch) * suffix_count;
  device float* current = costs + cost_base;
  device float* following = next_costs + cost_base;
  device float* saved_forward = forward_costs + cost_base;
  device float* scores = overlap_scores + score_base;
  constexpr uint thread_count = TransitionBits == 2 ? 1024 : 512;

  float sw = weighted ? step_weights[ulong(batch) * steps] : 1.0f;
  for (uint state = tid; state < kStateCount; state += thread_count)
    current[state] = qvq_emission(sequences + sequence_base, codebook, state, sw);
  threadgroup_barrier(mem_flags::mem_device);
  for (uint step = 1; step <= boundary; ++step) {
    device const float* target = sequences + sequence_base + ulong(step) * 2;
    sw = weighted ? step_weights[ulong(batch) * steps + step] : 1.0f;
    for (uint suffix = tid; suffix < suffix_count; suffix += thread_count) {
      float best = current[suffix];
      for (uint prefix = 1; prefix < prefix_count; ++prefix)
        best = min(best, current[ulong(prefix) * suffix_count + suffix]);
      for (uint edge = 0; edge < prefix_count; ++edge) {
        uint state = suffix * prefix_count + edge;
        following[state] = best + qvq_emission(target, codebook, state, sw);
      }
    }
    threadgroup_barrier(mem_flags::mem_device);
    device float* swap = current; current = following; following = swap;
  }
  for (uint state = tid; state < kStateCount; state += thread_count) saved_forward[state] = current[state];
  threadgroup_barrier(mem_flags::mem_device);

  current = costs + cost_base; following = next_costs + cost_base;
  for (uint state = tid; state < kStateCount; state += thread_count) current[state] = 0.0f;
  threadgroup_barrier(mem_flags::mem_device);
  for (uint step = steps - 1; step > boundary; --step) {
    device const float* target = sequences + sequence_base + ulong(step) * 2;
    sw = weighted ? step_weights[ulong(batch) * steps + step] : 1.0f;
    for (uint state = tid; state < kStateCount; state += thread_count)
      following[state] = qvq_emission(target, codebook, state, sw) + current[state];
    threadgroup_barrier(mem_flags::mem_device);
    for (uint overlap_index = tid; overlap_index < suffix_count; overlap_index += thread_count) {
      float best = following[overlap_index * prefix_count];
      for (uint edge = 1; edge < prefix_count; ++edge)
        best = min(best, following[overlap_index * prefix_count + edge]);
      scores[overlap_index] = best;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint state = tid; state < kStateCount; state += thread_count) current[state] = scores[state & overlap_mask];
    threadgroup_barrier(mem_flags::mem_device);
  }
  for (uint overlap_index = tid; overlap_index < suffix_count; overlap_index += thread_count) {
    float best = saved_forward[overlap_index * prefix_count] + current[overlap_index * prefix_count];
    for (uint edge = 1; edge < prefix_count; ++edge) {
      uint state = overlap_index * prefix_count + edge;
      best = min(best, saved_forward[state] + current[state]);
    }
    scores[overlap_index] = best;
  }
}

#define QVQ_OVERLAP_KERNEL(BITS) \
kernel void qvq_overlap_scores_e##BITS( \
    device const float* sequences [[buffer(0)]], device const float* codebook [[buffer(1)]], \
    device const float* step_weights [[buffer(2)]], device float* costs [[buffer(3)]], \
    device float* next_costs [[buffer(4)]], device float* forward_costs [[buffer(5)]], \
    device float* overlap_scores [[buffer(6)]], constant uint* dimensions [[buffer(7)]], \
    uint batch [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
  qvq_overlap_scores_lowrate<BITS>(sequences, codebook, step_weights, costs, next_costs, forward_costs, \
      overlap_scores, dimensions, batch, tid); \
}
QVQ_OVERLAP_KERNEL(2)
QVQ_OVERLAP_KERNEL(3)
"""

_HYB_REFERENCE_SOURCE = (
    _COMMON_METAL_SOURCE
    + r"""
inline float qvq_hyb_value(
    device const int* trellis, device const half* lut,
    uint k, uint n, uint N, uint transition_bits) {
  const uint tile_cols = N >> 4;
  const uint tile_idx = (k >> 4) * tile_cols + (n >> 4);
  const uint words_per_tile = 4 * transition_bits;
  const uint local = (k & 15) * 16 + (n & 15);
  const uint state = qvq_state(trellis + tile_idx * words_per_tile, local >> 1, transition_bits);
  const uint hashed = state * state + state;
  const uint lut_idx = (hashed >> 6) & 511u;
  float value = float(lut[lut_idx * 2 + (local & 1)]);
  if ((local & 1) && (hashed & (1u << 15))) value = -value;
  return value;
}

kernel void qvq_hyb_reference_fp16(
    device const half* x [[buffer(0)]], device const int* trellis [[buffer(1)]],
    device const half* lut [[buffer(2)]], device half* output [[buffer(3)]],
    constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]], constant uint& transition_bits [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint m = group / N, n = group - m * N;
  float sum = 0.0f;
  for (uint k = lane; k < K; k += 32) {
    sum += float(x[m * K + k]) * qvq_hyb_value(trellis, lut, k, n, N, transition_bits);
  }
  sum = simd_sum(sum);
  if (lane == 0) output[m * N + n] = half(sum);
}
"""
)


def qvq_mps_supported() -> bool:
    return bool(torch.backends.mps.is_available() and callable(getattr(torch.mps, "compile_shader", None)))


def _library():
    global _LIBRARY, _LIBRARY_ERROR
    if _LIBRARY is None:
        with _LIBRARY_LOCK:
            if _LIBRARY is None:
                if _LIBRARY_ERROR is not None:
                    raise RuntimeError(_LIBRARY_ERROR)
                if not qvq_mps_supported():
                    raise RuntimeError("QVQ Metal requires torch.mps.compile_shader")
                try:
                    _LIBRARY = torch.mps.compile_shader(_SOURCE)
                except Exception as exc:
                    _LIBRARY_ERROR = f"QVQ Metal shader compilation failed: {exc}"
                    raise RuntimeError(_LIBRARY_ERROR) from exc
    return _LIBRARY


def _hyb_reference_library():
    global _HYB_REFERENCE_LIBRARY, _HYB_REFERENCE_LIBRARY_ERROR
    if _HYB_REFERENCE_LIBRARY is None:
        with _HYB_REFERENCE_LIBRARY_LOCK:
            if _HYB_REFERENCE_LIBRARY is None:
                if _HYB_REFERENCE_LIBRARY_ERROR is not None:
                    raise RuntimeError(_HYB_REFERENCE_LIBRARY_ERROR)
                if not qvq_mps_supported():
                    raise RuntimeError("QVQ HYB reference Metal requires torch.mps.compile_shader")
                try:
                    _HYB_REFERENCE_LIBRARY = torch.mps.compile_shader(_HYB_REFERENCE_SOURCE)
                except Exception as exc:
                    _HYB_REFERENCE_LIBRARY_ERROR = f"QVQ HYB reference Metal compilation failed: {exc}"
                    raise RuntimeError(_HYB_REFERENCE_LIBRARY_ERROR) from exc
    return _HYB_REFERENCE_LIBRARY


def _viterbi_library():
    global _VITERBI_LIBRARY, _VITERBI_LIBRARY_ERROR
    if _VITERBI_LIBRARY is None:
        with _VITERBI_LIBRARY_LOCK:
            if _VITERBI_LIBRARY is None:
                if _VITERBI_LIBRARY_ERROR is not None:
                    raise RuntimeError(_VITERBI_LIBRARY_ERROR)
                if not qvq_mps_supported():
                    raise RuntimeError("QVQ Viterbi Metal requires torch.mps.compile_shader")
                try:
                    _VITERBI_LIBRARY = torch.mps.compile_shader(_VITERBI_SOURCE)
                except Exception as exc:
                    _VITERBI_LIBRARY_ERROR = f"QVQ Viterbi Metal compilation failed: {exc}"
                    raise RuntimeError(_VITERBI_LIBRARY_ERROR) from exc
    return _VITERBI_LIBRARY


def qvq_mps_viterbi(
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    bits: float,
    overlap: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
    *,
    _trusted_inputs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the persistent exact-width W1 through W8 Viterbi recurrence on Metal."""

    transition_bits = qvq_transition_bits(bits)
    if transition_bits not in range(2, 17):
        raise ValueError("QVQ Metal Viterbi currently supports W1 through W8")
    if sequences.ndim != 3 or sequences.shape[2] != 2:
        raise ValueError("QVQ Metal Viterbi sequences must have shape [batch, steps, 2]")
    if tuple(codebook.shape) != (1 << 16, 2):
        raise ValueError("QVQ Metal Viterbi codebook must have shape [65536, 2]")
    if sequences.device.type != "mps" or codebook.device != sequences.device:
        raise ValueError("QVQ Metal Viterbi tensors must share one MPS device")
    if sequences.dtype != torch.float32 or codebook.dtype != torch.float32:
        raise TypeError("QVQ Metal Viterbi sequences and codebook must use float32")
    if not sequences.is_contiguous() or not codebook.is_contiguous():
        raise ValueError("QVQ Metal Viterbi sequences and codebook must be contiguous")
    batch, steps, _ = sequences.shape
    if batch < 1 or steps < 1:
        raise ValueError("QVQ Metal Viterbi requires a nonempty batch and sequence")
    if not _trusted_inputs and (not torch.isfinite(sequences).all() or not torch.isfinite(codebook).all()):
        raise ValueError("QVQ Metal Viterbi sequences and codebook must contain only finite values")
    constrained = overlap is not None
    weighted = step_weights is not None
    if constrained:
        if overlap.device != sequences.device or overlap.dtype != torch.int64 or not overlap.is_contiguous():
            raise TypeError("QVQ Metal Viterbi overlap must be contiguous int64 on the sequence device")
        if tuple(overlap.shape) != (batch,):
            raise ValueError("QVQ Metal Viterbi overlap must have shape [batch]")
    else:
        overlap = torch.empty(1, dtype=torch.int64, device=sequences.device)
    if weighted:
        if (
            step_weights.device != sequences.device
            or step_weights.dtype != torch.float32
            or not step_weights.is_contiguous()
        ):
            raise TypeError("QVQ Metal Viterbi step weights must be contiguous float32 on the sequence device")
        if tuple(step_weights.shape) != (batch, steps):
            raise ValueError("QVQ Metal Viterbi step weights must have shape [batch, steps]")
        if not _trusted_inputs and (not torch.isfinite(step_weights).all() or torch.any(step_weights < 0)):
            raise ValueError("QVQ Metal Viterbi step weights must be finite and nonnegative")
    else:
        step_weights = torch.empty(1, dtype=torch.float32, device=sequences.device)

    if constrained and not _trusted_inputs:
        overlap_limit = 1 << (16 - transition_bits)
        if torch.any((overlap < 0) | (overlap >= overlap_limit)):
            raise ValueError(f"QVQ Metal Viterbi overlap must be in [0, {overlap_limit - 1}]")

    states = torch.empty((batch, steps), dtype=torch.int64, device=sequences.device)
    squared_error = torch.empty(batch, dtype=torch.float32, device=sequences.device)
    dimensions = torch.tensor(
        [batch, steps, int(constrained), int(weighted)], dtype=torch.uint32, device=sequences.device
    )
    if transition_bits == 16:
        step_losses = torch.empty((batch, steps), dtype=torch.float32, device=sequences.device)
        library = _viterbi_library()
        library.qvq_viterbi_e16_independent(
            sequences,
            codebook,
            step_weights,
            states,
            step_losses,
            dimensions,
            threads=batch * steps * 256,
            group_size=256,
        )
        library.qvq_viterbi_e16_finalize(
            step_losses,
            squared_error,
            dimensions,
            threads=batch,
            group_size=1,
        )
        return states, squared_error
    if transition_bits == 15:
        transition_costs = torch.empty((batch, steps, 4), dtype=torch.float32, device=sequences.device)
        transition_states = torch.empty((batch, steps, 4), dtype=torch.uint32, device=sequences.device)
        context_backpointers = torch.empty((batch, steps, 2), dtype=torch.uint8, device=sequences.device)
        library = _viterbi_library()
        library.qvq_viterbi_e15_transitions(
            sequences,
            codebook,
            step_weights,
            transition_costs,
            transition_states,
            dimensions,
            threads=batch * steps * 4 * 64,
            group_size=64,
        )
        library.qvq_viterbi_e15_finalize(
            transition_costs,
            transition_states,
            overlap,
            states,
            squared_error,
            context_backpointers,
            dimensions,
            threads=batch,
            group_size=1,
        )
        return states, squared_error
    if transition_bits == 14:
        transition_costs = torch.empty((batch, steps, 16), dtype=torch.float32, device=sequences.device)
        transition_states = torch.empty((batch, steps, 16), dtype=torch.uint32, device=sequences.device)
        context_backpointers = torch.empty((batch, steps, 4), dtype=torch.uint8, device=sequences.device)
        library = _viterbi_library()
        library.qvq_viterbi_e14_transitions(
            sequences,
            codebook,
            step_weights,
            transition_costs,
            transition_states,
            dimensions,
            threads=batch * steps * 16 * 64,
            group_size=64,
        )
        library.qvq_viterbi_e14_finalize(
            transition_costs,
            transition_states,
            overlap,
            states,
            squared_error,
            context_backpointers,
            dimensions,
            threads=batch,
            group_size=1,
        )
        return states, squared_error
    if transition_bits == 13:
        transition_costs = torch.empty((batch, steps, 64), dtype=torch.float32, device=sequences.device)
        transition_states = torch.empty((batch, steps, 64), dtype=torch.uint32, device=sequences.device)
        context_backpointers = torch.empty((batch, steps, 8), dtype=torch.uint8, device=sequences.device)
        library = _viterbi_library()
        library.qvq_viterbi_e13_transitions(
            sequences,
            codebook,
            step_weights,
            transition_costs,
            transition_states,
            dimensions,
            threads=batch * steps * 64 * 32,
            group_size=32,
        )
        library.qvq_viterbi_e13_finalize(
            transition_costs,
            transition_states,
            overlap,
            states,
            squared_error,
            context_backpointers,
            dimensions,
            threads=batch,
            group_size=1,
        )
        return states, squared_error
    suffix_count = (1 << 16) >> transition_bits
    costs = torch.empty((batch, 1 << 16), dtype=torch.float32, device=sequences.device)
    next_costs = torch.empty_like(costs)
    backpointers = torch.empty((batch, steps - 1, suffix_count), dtype=torch.uint16, device=sequences.device)
    kernel = getattr(_viterbi_library(), f"qvq_viterbi_e{transition_bits}")
    kernel(
        sequences,
        codebook,
        overlap,
        step_weights,
        costs,
        next_costs,
        backpointers,
        states,
        squared_error,
        dimensions,
        threads=batch * (1024 if transition_bits == 2 or transition_bits >= 9 else 512),
        group_size=1024 if transition_bits == 2 or transition_bits >= 9 else 512,
    )
    return states, squared_error


def qvq_mps_overlap_scores(
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    bits: float,
    boundary: int,
    step_weights: torch.Tensor | None = None,
    *,
    _trusted_inputs: bool = False,
) -> torch.Tensor:
    """Score every W1/W1.5 boundary overlap in one persistent Metal launch."""

    transition_bits = qvq_transition_bits(bits)
    if transition_bits not in (2, 3):
        raise ValueError("QVQ Metal overlap scoring currently supports W1 and W1.5")
    if sequences.ndim != 3 or sequences.shape[2] != 2 or tuple(codebook.shape) != (1 << 16, 2):
        raise ValueError("QVQ Metal overlap scoring requires [batch, steps, 2] and [65536, 2]")
    if sequences.device.type != "mps" or codebook.device != sequences.device:
        raise ValueError("QVQ Metal overlap-scoring tensors must share one MPS device")
    if sequences.dtype != torch.float32 or codebook.dtype != torch.float32:
        raise TypeError("QVQ Metal overlap scoring requires float32 sequences and codebook")
    if not sequences.is_contiguous() or not codebook.is_contiguous():
        raise ValueError("QVQ Metal overlap-scoring tensors must be contiguous")
    if not _trusted_inputs and (not torch.isfinite(sequences).all() or not torch.isfinite(codebook).all()):
        raise ValueError("QVQ Metal overlap-scoring sequences and codebook must contain only finite values")
    batch, steps, _ = sequences.shape
    if batch < 1 or steps < 1:
        raise ValueError("QVQ Metal overlap scoring requires a nonempty batch and sequence")
    if boundary < 0 or boundary >= steps:
        raise ValueError("QVQ Metal overlap boundary must index the sequence")
    weighted = step_weights is not None
    if weighted:
        if (
            step_weights.device != sequences.device
            or step_weights.dtype != torch.float32
            or not step_weights.is_contiguous()
        ):
            raise TypeError("QVQ Metal overlap step weights must be contiguous float32 on the sequence device")
        if tuple(step_weights.shape) != (batch, steps):
            raise ValueError("QVQ Metal overlap step weights must have shape [batch, steps]")
        if not _trusted_inputs and (not torch.isfinite(step_weights).all() or torch.any(step_weights < 0)):
            raise ValueError("QVQ Metal overlap step weights must be finite and nonnegative")
    else:
        step_weights = torch.empty(1, dtype=torch.float32, device=sequences.device)
    suffix_count = (1 << 16) >> transition_bits
    costs = torch.empty((batch, 1 << 16), dtype=torch.float32, device=sequences.device)
    next_costs = torch.empty_like(costs)
    forward_costs = torch.empty_like(costs)
    scores = torch.empty((batch, suffix_count), dtype=torch.float32, device=sequences.device)
    dimensions = torch.tensor([batch, steps, boundary, int(weighted)], dtype=torch.uint32, device=sequences.device)
    kernel = getattr(_viterbi_library(), f"qvq_overlap_scores_e{transition_bits}")
    kernel(
        sequences,
        codebook,
        step_weights,
        costs,
        next_costs,
        forward_costs,
        scores,
        dimensions,
        threads=batch * (1024 if transition_bits == 2 else 512),
        group_size=1024 if transition_bits == 2 else 512,
    )
    return scores


def _pgc16_levels(
    device: torch.device,
    codebook_version: str,
) -> torch.Tensor:
    global _PGC16_LEVELS_HOT
    hot = _PGC16_LEVELS_HOT
    if hot is not None and device == hot[0] and codebook_version == hot[1]:
        return hot[2]
    key = (device, str(codebook_version).strip().lower())
    levels = _PGC16_LEVELS.get(key)
    if levels is None:
        with _PGC16_LEVELS_LOCK:
            levels = _PGC16_LEVELS.get(key)
            if levels is None:
                levels = pgc16_levels_for_version(key[1]).to(device=device).contiguous()
                _PGC16_LEVELS[key] = levels
    _PGC16_LEVELS_HOT = (device, key[1], levels)
    return levels


def _prepare_qvq_mps_compander(
    device: torch.device,
    codebook_version: str,
) -> _QVQMPSPreparedCompander:
    return _QVQMPSPreparedCompander(levels=_pgc16_levels(device, codebook_version))


def _v4_row_tile(transition_bits: int, m: int, k: int, n: int) -> int:
    if k >= 8192 and n >= 8192 and transition_bits == 4:
        return 4
    if k >= 8192 and n <= 2048:
        if m == 16 and transition_bits < 16:
            return 8
        if m >= 24:
            return 4 if transition_bits == 16 else 8
    if k <= 2048 and n >= 8192 and transition_bits == 16:
        if m == 16:
            return 16
        if m >= 32:
            return 4
    if m <= 4:
        return 8 if n <= 2048 else 4
    if m <= 8:
        return 4
    if m <= 16:
        return 16 if n <= 2048 else 8
    if m < 20:
        return 4
    if m < 32:
        return 8
    return 16 if n <= 2048 else 8


def _v4_use_k64(m: int, k: int, n: int) -> bool:
    return (m >= 16 and n >= 8192) or (m >= 24 and k >= 8192)


def _v4_use_k128(transition_bits: int, m: int, k: int, n: int) -> bool:
    if m >= 24 and k >= 8192 and n < 8192:
        return True
    if m >= 16 and n >= 8192 and k < 8192:
        return transition_bits < 16 or m == 17 or m >= 32
    return False


def _run_multirow(
    x: torch.Tensor,
    trellis: torch.Tensor,
    bank_ids: torch.Tensor | None,
    levels: torch.Tensor,
    output: torch.Tensor,
    transition_bits: int,
    *,
    m: int,
    k: int,
    n: int,
    vector_width: int,
    vector_size: int,
) -> None:
    if vector_size == 4:
        if vector_width != 4:
            raise ValueError("QVQ V4 Metal multi-row kernels require vector width 4")
        row_tile = _v4_row_tile(transition_bits, m, k, n)
        group_size = (row_tile // 4) * 32
        row_blocks = (m + row_tile - 1) // row_tile
        if _v4_use_k128(transition_bits, m, k, n):
            suffix = "_k128"
        else:
            suffix = "_k64" if _v4_use_k64(m, k, n) else ""
        prefix = "qvq_planar_v4_banked_fp16" if bank_ids is not None else "qvq_planar_v4_fp16"
        kernel = getattr(_library(), f"{prefix}_multirow{suffix}_e{transition_bits}")
        args = (x, trellis, levels, output) if bank_ids is None else (x, trellis, bank_ids, levels, output)
        kernel(
            *args,
            m,
            k,
            n,
            row_tile,
            threads=row_blocks * (n // vector_width) * group_size,
            group_size=group_size,
        )
        return
    if vector_width not in (4, 8):
        raise ValueError(f"QVQ Metal multi-row vector width must be 4 or 8, got {vector_width}")
    row_tile = 16 if vector_width == 8 or m <= 16 else 32
    group_size = (row_tile // (2 if vector_width == 8 else 4)) * 32
    row_blocks = (m + row_tile - 1) // row_tile
    suffix = "_n8" if vector_width == 8 else ""
    kernel = getattr(_library(), f"qvq_planar_fp16_multirow{suffix}_e{transition_bits}")
    kernel(
        x,
        trellis,
        levels,
        output,
        m,
        k,
        n,
        row_tile,
        threads=row_blocks * (n // vector_width) * group_size,
        group_size=group_size,
    )


def _multirow_vector_width(transition_bits: int, m: int, k: int, n: int) -> int:
    if transition_bits == 16 and 15 <= m <= 16 and max(k, n) >= 8192:
        return 8
    return 4


def qvq_mps_gemv(
    x: torch.Tensor,
    trellis: torch.Tensor,
    bits: float,
    *,
    out_features: int,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
    vector_size: int = 2,
    bank_ids: torch.Tensor | None = None,
    output_fp32: bool = False,
    _prepared_compander: _QVQMPSPreparedCompander | None = None,
) -> torch.Tensor:
    """Multiply by planar PGC16 tiles without expanding weights."""

    bits = normalize_qvq_rate(bits)
    if not isinstance(output_fp32, bool):
        raise TypeError("QVQ Metal output_fp32 must be a bool")
    if vector_size not in (2, 4) or (vector_size == 4 and bits > 4):
        raise ValueError("QVQ Metal vector_size must be 2, or 4 for rates W1 through W4")
    if bank_ids is not None and vector_size != 4:
        raise ValueError("QVQ Metal bank selectors require vector_size=4")
    transition_bits = qvq_transition_bits(bits, vector_size=vector_size)
    if x.ndim != 2 or trellis.ndim != 2:
        raise ValueError("QVQ Metal expects 2D x and trellis tensors")
    if x.device.type != "mps" or trellis.device != x.device:
        raise ValueError("QVQ Metal tensors must share one MPS device")
    if x.dtype != torch.float16 or trellis.dtype != torch.int32:
        raise TypeError("QVQ Metal requires float16 x and int32 planar trellis words")
    if not x.is_contiguous() or not trellis.is_contiguous():
        raise ValueError("QVQ Metal tensors must be contiguous")
    m, k = x.shape
    n = _integer_argument(out_features, "out_features")
    if k <= 0 or n <= 0 or k % 16 or n % 16:
        raise ValueError(f"QVQ Metal requires positive K/N divisible by 16, got K={k}, N={n}")
    expected = (
        (k // 16) * (n // 16),
        qvq_words_per_tile(bits, vector_size=vector_size),
    )
    if tuple(trellis.shape) != expected:
        raise ValueError(f"QVQ planar trellis must have shape {expected}, got {tuple(trellis.shape)}")
    if bank_ids is not None:
        packed_count = (expected[0] + 3) // 4
        if bank_ids.device != x.device or bank_ids.dtype != torch.uint8:
            raise TypeError("QVQ Metal bank selectors must be packed uint8 on the input MPS device")
        if bank_ids.ndim != 1 or bank_ids.numel() != packed_count:
            raise ValueError(f"QVQ Metal bank selectors must have packed shape {(packed_count,)}")
        if not bank_ids.is_contiguous():
            raise ValueError("QVQ Metal bank selectors must be contiguous")
    levels = (
        _pgc16_levels(x.device, codebook_version)
        if _prepared_compander is None
        else _prepared_compander.levels
    )
    if levels.device != x.device:
        raise ValueError("QVQ Metal prepared compander must share the input MPS device")
    if m == 0:
        dtype = torch.float32 if output_fp32 else x.dtype
        return torch.empty((0, n), dtype=dtype, device=x.device)
    selector_numel = 0 if bank_ids is None else bank_ids.numel()
    if max(m, k, n, m * n, x.numel(), trellis.numel(), levels.numel(), selector_numel) > 2**32 - 1:
        raise ValueError("QVQ Metal dimensions exceed the uint32 kernel limit")
    output = torch.empty(
        (m, n), dtype=torch.float32 if output_fp32 else torch.float16, device=x.device
    )
    if output_fp32:
        if vector_size == 4:
            prefix = "qvq_planar_v4_banked_fp32" if bank_ids is not None else "qvq_planar_v4_fp32"
        else:
            prefix = "qvq_planar_fp32"
        kernel = getattr(_library(), f"{prefix}_e{transition_bits}")
        args = (x, trellis, levels, output) if bank_ids is None else (x, trellis, bank_ids, levels, output)
        kernel(
            *args,
            m,
            k,
            n,
            threads=m * (n // vector_size) * 32,
            group_size=32,
        )
    elif m >= 4:
        _run_multirow(
            x,
            trellis,
            bank_ids,
            levels,
            output,
            transition_bits,
            m=m,
            k=k,
            n=n,
            vector_width=(4 if vector_size == 4 else _multirow_vector_width(transition_bits, m, k, n)),
            vector_size=vector_size,
        )
    else:
        if vector_size == 4:
            prefix = "qvq_planar_v4_banked_fp16" if bank_ids is not None else "qvq_planar_v4_fp16"
            name = f"{prefix}_e4_fast" if transition_bits == 4 else f"{prefix}_e{transition_bits}"
        else:
            name = f"qvq_planar_fp16_e{transition_bits}"
        kernel = getattr(_library(), name)
        args = (x, trellis, levels, output) if bank_ids is None else (x, trellis, bank_ids, levels, output)
        kernel(
            *args,
            m,
            k,
            n,
            threads=m * (n // vector_size) * 32,
            group_size=32,
        )
    return output


def qvq_hyb_reference_mps_gemv(
    x: torch.Tensor,
    trellis: torch.Tensor,
    lut: torch.Tensor,
    bits: float,
    *,
    out_features: int,
) -> torch.Tensor:
    """Run the former scalar HYB kernel as a non-loadable benchmark oracle."""

    bits = normalize_qvq_rate(bits)
    transition_bits = qvq_transition_bits(bits)
    if x.ndim != 2 or trellis.ndim != 2 or lut.ndim != 2:
        raise ValueError("QVQ HYB reference expects 2D x, trellis, and LUT tensors")
    if x.device.type != "mps" or any(tensor.device != x.device for tensor in (trellis, lut)):
        raise ValueError("QVQ HYB reference tensors must share one MPS device")
    if x.dtype != torch.float16 or lut.dtype != torch.float16 or trellis.dtype != torch.int32:
        raise TypeError("QVQ HYB reference requires float16 x/LUT and int32 trellis words")
    if any(not tensor.is_contiguous() for tensor in (x, trellis, lut)):
        raise ValueError("QVQ HYB reference tensors must be contiguous")
    m, k = x.shape
    n = int(out_features)
    if k <= 0 or n <= 0 or k % 16 or n % 16:
        raise ValueError(f"QVQ HYB reference requires positive aligned K/N, got K={k}, N={n}")
    expected = ((k // 16) * (n // 16), qvq_words_per_tile(bits))
    if tuple(trellis.shape) != expected:
        raise ValueError(f"QVQ HYB reference trellis must have shape {expected}")
    if tuple(lut.shape) != (512, 2):
        raise ValueError("QVQ HYB reference LUT must have shape (512, 2)")
    if m == 0:
        return torch.empty((0, n), dtype=x.dtype, device=x.device)
    output = torch.empty((m, n), dtype=torch.float16, device=x.device)
    _hyb_reference_library().qvq_hyb_reference_fp16(
        x,
        trellis,
        lut,
        output,
        m,
        k,
        n,
        transition_bits,
        threads=m * n * 32,
        group_size=32,
    )
    return output


__all__ = [
    "QVQ_MPS_BITS",
    "qvq_hyb_reference_mps_gemv",
    "qvq_mps_gemv",
    "qvq_mps_overlap_scores",
    "qvq_mps_supported",
    "qvq_mps_viterbi",
]
