# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fused Apple Metal GEMV for GPTQ/Pangolin packed weights."""

from __future__ import annotations

import threading
from typing import Any

import torch

PANGOLIN_MPS_BITS = (2, 3, 4, 5, 6, 7, 8)
_LIBRARY: Any | None = None
_LIBRARY_ERROR: str | None = None
_LIBRARY_LOCK = threading.Lock()

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline uint planar_code(device const int* packed, uint k, uint n, uint N, uint bits) {
  const uint block = k >> 5;
  const uint lane = k & 31;
  const uint base = block * bits;
  uint w0, w1, w2 = 0, width0, width1, width2 = 0;
  if (bits == 3) { width0 = 2; width1 = 1; }
  else if (bits == 5) { width0 = 4; width1 = 1; }
  else if (bits == 6) { width0 = 4; width1 = 2; }
  else { width0 = 4; width1 = 2; width2 = 1; }
  const uint pf0 = 32 / width0;
  const uint pf1 = 32 / width1;
  w0 = as_type<uint>(packed[(base + lane / pf0) * N + n]);
  w1 = as_type<uint>(packed[(base + width0 + lane / pf1) * N + n]);
  uint code = ((w0 >> (width0 * (lane % pf0))) & ((1u << width0) - 1u));
  code |= ((w1 >> (width1 * (lane % pf1))) & ((1u << width1) - 1u)) << width0;
  if (width2 != 0) {
    const uint pf2 = 32 / width2;
    w2 = as_type<uint>(packed[(base + width0 + width1 + lane / pf2) * N + n]);
    code |= ((w2 >> (width2 * (lane % pf2))) & ((1u << width2) - 1u)) << (width0 + width1);
  }
  return code;
}

inline uint continuous_row_code(device const int* packed, uint k, uint n, uint N, uint bits) {
  const uint pf = 32 / bits;
  const uint word = as_type<uint>(packed[(k / pf) * N + n]);
  return (word >> (bits * (k % pf))) & ((1u << bits) - 1u);
}

inline uint planar_zero(device const int* packed, uint g, uint n, uint N, uint bits) {
  const uint block = n >> 5;
  const uint lane = n & 31;
  const uint words = (N >> 5) * bits;
  const uint base = g * words + block * bits;
  uint width0, width1, width2 = 0;
  if (bits == 3) { width0 = 2; width1 = 1; }
  else if (bits == 5) { width0 = 4; width1 = 1; }
  else if (bits == 6) { width0 = 4; width1 = 2; }
  else { width0 = 4; width1 = 2; width2 = 1; }
  const uint pf0 = 32 / width0;
  const uint pf1 = 32 / width1;
  uint w0 = as_type<uint>(packed[base + lane / pf0]);
  uint w1 = as_type<uint>(packed[base + width0 + lane / pf1]);
  uint code = (w0 >> (width0 * (lane % pf0))) & ((1u << width0) - 1u);
  code |= ((w1 >> (width1 * (lane % pf1))) & ((1u << width1) - 1u)) << width0;
  if (width2 != 0) {
    const uint pf2 = 32 / width2;
    uint w2 = as_type<uint>(packed[base + width0 + width1 + lane / pf2]);
    code |= ((w2 >> (width2 * (lane % pf2))) & 1u) << (width0 + width1);
  }
  return code;
}

inline uint continuous_zero(device const int* packed, uint g, uint n, uint N, uint bits) {
  const uint pf = 32 / bits;
  const uint words = N / pf;
  const uint word = as_type<uint>(packed[g * words + n / pf]);
  return (word >> (bits * (n % pf))) & ((1u << bits) - 1u);
}

inline uint4 load_uint4(device const int* packed, uint index) {
  return as_type<uint4>(*reinterpret_cast<device const int4*>(packed + index));
}

inline uint4 planar_code4(device const int* packed, uint k, uint n, uint N, uint bits) {
  const uint block = k >> 5, lane = k & 31, base = block * bits;
  uint width0, width1, width2 = 0;
  if (bits == 3) { width0 = 2; width1 = 1; }
  else if (bits == 5) { width0 = 4; width1 = 1; }
  else if (bits == 6) { width0 = 4; width1 = 2; }
  else { width0 = 4; width1 = 2; width2 = 1; }
  const uint pf0 = 32 / width0, pf1 = 32 / width1;
  const uint4 w0 = load_uint4(packed, (base + lane / pf0) * N + n);
  const uint4 w1 = load_uint4(packed, (base + width0 + lane / pf1) * N + n);
  uint4 code = (w0 >> (width0 * (lane % pf0))) & ((1u << width0) - 1u);
  code |= ((w1 >> (width1 * (lane % pf1))) & ((1u << width1) - 1u)) << width0;
  if (width2 != 0) {
    const uint pf2 = 32 / width2;
    const uint4 w2 = load_uint4(packed, (base + width0 + width1 + lane / pf2) * N + n);
    code |= ((w2 >> (width2 * (lane % pf2))) & 1u) << (width0 + width1);
  }
  return code;
}

inline uint4 continuous_row_code4(device const int* packed, uint k, uint n, uint N, uint bits) {
  uint word_row, shift;
  if (bits == 2) { word_row = k >> 4; shift = (k & 15) << 1; }
  else if (bits == 4) { word_row = k >> 3; shift = (k & 7) << 2; }
  else { word_row = k >> 2; shift = (k & 3) << 3; }
  const uint4 word = load_uint4(packed, word_row * N + n);
  return (word >> shift) & ((1u << bits) - 1u);
}

inline uint4 planar_zero4(device const int* packed, uint g, uint n, uint N, uint bits) {
  const uint block = n >> 5, lane = n & 31, words = (N >> 5) * bits;
  const uint base = g * words + block * bits;
  uint width0, width1, width2 = 0;
  if (bits == 3) { width0 = 2; width1 = 1; }
  else if (bits == 5) { width0 = 4; width1 = 1; }
  else if (bits == 6) { width0 = 4; width1 = 2; }
  else { width0 = 4; width1 = 2; width2 = 1; }
  const uint4 lanes = uint4(lane) + uint4(0, 1, 2, 3);
  const uint pf0 = 32 / width0, pf1 = 32 / width1;
  const uint w0 = as_type<uint>(packed[base + lane / pf0]);
  const uint w1 = as_type<uint>(packed[base + width0 + lane / pf1]);
  uint4 code = (uint4(w0) >> (width0 * (lanes % pf0))) & ((1u << width0) - 1u);
  code |= ((uint4(w1) >> (width1 * (lanes % pf1))) & ((1u << width1) - 1u)) << width0;
  if (width2 != 0) {
    const uint w2 = as_type<uint>(packed[base + width0 + width1 + lane / 32]);
    code |= ((uint4(w2) >> (lanes & 31)) & 1u) << (width0 + width1);
  }
  return code;
}

inline uint4 continuous_zero4(device const int* packed, uint g, uint n, uint N, uint bits) {
  uint words, word_col, lane, lane_shift;
  if (bits == 2) { words = N >> 4; word_col = n >> 4; lane = n & 15; lane_shift = 1; }
  else if (bits == 4) { words = N >> 3; word_col = n >> 3; lane = n & 7; lane_shift = 2; }
  else { words = N >> 2; word_col = n >> 2; lane = n & 3; lane_shift = 3; }
  const uint word = as_type<uint>(packed[g * words + word_col]);
  const uint4 shifts = (uint4(lane) + uint4(0, 1, 2, 3)) << lane_shift;
  return (uint4(word) >> shifts) & ((1u << bits) - 1u);
}

kernel void pangolin_fp16_m3_n4(
    device const half* x [[buffer(0)]], device const int* qweight [[buffer(1)]],
    device const half* scales [[buffer(2)]], device const int* qzeros [[buffer(3)]],
    device const int* g_idx [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& M [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]], constant uint& groups [[buffer(9)]],
    constant uint& bits [[buffer(10)]], constant uint& planar [[buffer(11)]],
    constant uint& block_uniform [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint groups_n = N / 4;
  const uint m = group / groups_n;
  const uint n = (group - m * groups_n) * 4;
  float4 sum = 0.0f;
  bool valid = true;
  for (uint k = lane; k < K; k += 32) {
    int gi = g_idx[k];
    if (gi < 0) gi += int(groups);
    if (gi < 0 || gi >= int(groups)) { valid = false; break; }
    const uint g = uint(gi);
    const uint4 code = planar ? planar_code4(qweight, k, n, N, bits)
                              : continuous_row_code4(qweight, k, n, N, bits);
    const uint4 zero = planar ? planar_zero4(qzeros, g, n, N, bits)
                              : continuous_zero4(qzeros, g, n, N, bits);
    const half4 scale_h = *reinterpret_cast<device const half4*>(scales + g * N + n);
    sum += float(x[m * K + k]) * (float4(int4(code) - int4(zero)) * float4(scale_h));
  }
  valid = simd_all(valid);
  sum.x = simd_sum(sum.x); sum.y = simd_sum(sum.y);
  sum.z = simd_sum(sum.z); sum.w = simd_sum(sum.w);
  if (lane == 0) {
    *reinterpret_cast<device half4*>(output + m * N + n) = valid ? half4(sum) : half4(NAN);
  }
}

kernel void pangolin_fp16_m3_n4_planar_uniform(
    device const half* x [[buffer(0)]], device const int* qweight [[buffer(1)]],
    device const half* scales [[buffer(2)]], device const int* qzeros [[buffer(3)]],
    device const int* g_idx [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& M [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]], constant uint& groups [[buffer(9)]],
    constant uint& bits [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint groups_n = N / 4;
  const uint m = group / groups_n;
  const uint n = (group - m * groups_n) * 4;
  float4 sum = 0.0f;
  bool valid = true;
  for (uint k = lane; k < K; k += 32) {
    int gi = lane == 0 ? g_idx[k] : 0;
    gi = simd_broadcast_first(gi);
    if (gi < 0) gi += int(groups);
    if (gi < 0 || gi >= int(groups)) { valid = false; break; }
    const uint g = uint(gi);
    const uint4 code = planar_code4(qweight, k, n, N, bits);
    uint4 zero = 0;
    float4 scale = 0.0f;
    if (lane == 0) {
      zero = planar_zero4(qzeros, g, n, N, bits);
      scale = float4(*reinterpret_cast<device const half4*>(scales + g * N + n));
    }
    zero.x = simd_broadcast_first(zero.x); zero.y = simd_broadcast_first(zero.y);
    zero.z = simd_broadcast_first(zero.z); zero.w = simd_broadcast_first(zero.w);
    scale.x = simd_broadcast_first(scale.x); scale.y = simd_broadcast_first(scale.y);
    scale.z = simd_broadcast_first(scale.z); scale.w = simd_broadcast_first(scale.w);
    sum += float(x[m * K + k]) * (float4(int4(code) - int4(zero)) * scale);
  }
  valid = simd_all(valid);
  sum.x = simd_sum(sum.x); sum.y = simd_sum(sum.y);
  sum.z = simd_sum(sum.z); sum.w = simd_sum(sum.w);
  if (lane == 0) {
    *reinterpret_cast<device half4*>(output + m * N + n) = valid ? half4(sum) : half4(NAN);
  }
}

// The common contiguous GPTQ group layout gives every lane in a K block the
// same zero point and scale. Load those values once for continuous 2/4/8-bit
// packing and broadcast them while retaining the four-column vector decode.
kernel void pangolin_fp16_m3_n4_continuous_uniform(
    device const half* x [[buffer(0)]], device const int* qweight [[buffer(1)]],
    device const half* scales [[buffer(2)]], device const int* qzeros [[buffer(3)]],
    device const int* g_idx [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& M [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]], constant uint& groups [[buffer(9)]],
    constant uint& bits [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint groups_n = N / 4;
  const uint m = group / groups_n;
  const uint n = (group - m * groups_n) * 4;
  float4 sum = 0.0f;
  bool valid = true;
  for (uint k = lane; k < K; k += 32) {
    int gi = lane == 0 ? g_idx[k] : 0;
    gi = simd_broadcast_first(gi);
    if (gi < 0) gi += int(groups);
    if (gi < 0 || gi >= int(groups)) { valid = false; break; }
    const uint g = uint(gi);
    const uint4 code = continuous_row_code4(qweight, k, n, N, bits);
    uint4 zero = 0;
    float4 scale = 0.0f;
    if (lane == 0) {
      zero = continuous_zero4(qzeros, g, n, N, bits);
      scale = float4(*reinterpret_cast<device const half4*>(scales + g * N + n));
    }
    zero.x = simd_broadcast_first(zero.x); zero.y = simd_broadcast_first(zero.y);
    zero.z = simd_broadcast_first(zero.z); zero.w = simd_broadcast_first(zero.w);
    scale.x = simd_broadcast_first(scale.x); scale.y = simd_broadcast_first(scale.y);
    scale.z = simd_broadcast_first(scale.z); scale.w = simd_broadcast_first(scale.w);
    sum += float(x[m * K + k]) * (float4(int4(code) - int4(zero)) * scale);
  }
  valid = simd_all(valid);
  sum.x = simd_sum(sum.x); sum.y = simd_sum(sum.y);
  sum.z = simd_sum(sum.z); sum.w = simd_sum(sum.w);
  if (lane == 0) {
    *reinterpret_cast<device half4*>(output + m * N + n) = valid ? half4(sum) : half4(NAN);
  }
}

kernel void pangolin_fp16(
    device const half* x [[buffer(0)]],
    device const int* qweight [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const int* qzeros [[buffer(3)]],
    device const int* g_idx [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& M [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]], constant uint& groups [[buffer(9)]],
    constant uint& bits [[buffer(10)]], constant uint& planar [[buffer(11)]],
    constant uint& block_uniform [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (group >= M * N) return;
  const uint m = group / N;
  const uint n = group - m * N;
  float sum = 0.0f;
  bool valid = true;
  for (uint k = lane; k < K; k += 32) {
    int gi = g_idx[block_uniform ? k - lane : k];
    if (gi < 0) gi += int(groups);
    if (gi < 0 || gi >= int(groups)) { valid = false; break; }
    const uint g = uint(gi);
    const uint code = planar ? planar_code(qweight, k, n, N, bits)
                             : continuous_row_code(qweight, k, n, N, bits);
    uint zero = 0;
    float scale = 0.0f;
    if (!block_uniform || lane == 0) {
      zero = planar ? planar_zero(qzeros, g, n, N, bits)
                    : continuous_zero(qzeros, g, n, N, bits);
      scale = float(scales[g * N + n]);
    }
    if (block_uniform) {
      zero = simd_broadcast_first(zero);
      scale = simd_broadcast_first(scale);
    }
    sum += float(x[m * K + k]) * (float(int(code) - int(zero)) * scale);
  }
  valid = simd_all(valid);
  sum = simd_sum(sum);
  if (lane == 0) output[group] = valid ? half(sum) : half(NAN);
}

// Decode each K lane once and share it across up to eight activation rows.
// One threadgroup owns one output column; each SIMD-group owns one M row.
kernel void pangolin_fp16_m8(
    device const half* x [[buffer(0)]],
    device const int* qweight [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const int* qzeros [[buffer(3)]],
    device const int* g_idx [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& M [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]], constant uint& groups [[buffer(9)]],
    constant uint& bits [[buffer(10)]], constant uint& planar [[buffer(11)]],
    constant uint& block_uniform [[buffer(12)]],
    uint n [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float decoded[32];
  threadgroup atomic_uint valid;
  if (tid == 0) atomic_store_explicit(&valid, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0.0f;
  const bool active = simd < M;
  for (uint base = 0; base < K; base += 32) {
    if (simd == 0) {
      const uint k = base + lane;
      int gi = g_idx[block_uniform ? base : k];
      if (gi < 0) gi += int(groups);
      if (gi < 0 || gi >= int(groups)) {
        atomic_store_explicit(&valid, 0u, memory_order_relaxed);
        decoded[lane] = 0.0f;
      } else {
        const uint g = uint(gi);
        const uint code = planar ? planar_code(qweight, k, n, N, bits)
                                 : continuous_row_code(qweight, k, n, N, bits);
        uint zero = 0;
        float scale = 0.0f;
        if (!block_uniform || lane == 0) {
          zero = planar ? planar_zero(qzeros, g, n, N, bits)
                        : continuous_zero(qzeros, g, n, N, bits);
          scale = float(scales[g * N + n]);
        }
        if (block_uniform) {
          zero = simd_broadcast_first(zero);
          scale = simd_broadcast_first(scale);
        }
        decoded[lane] = float(int(code) - int(zero)) * scale;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) sum += float(x[simd * K + base + lane]) * decoded[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (active) {
    sum = simd_sum(sum);
    if (lane == 0) {
      output[simd * N + n] = atomic_load_explicit(&valid, memory_order_relaxed) != 0
          ? half(sum) : half(NAN);
    }
  }
}

// Vectorize four adjacent output columns and share their decoded values across
// up to eight activation rows. This combines coalesced packed-weight loads with
// row reuse while retaining one float32 accumulation and one half conversion
// per output element.
kernel void pangolin_fp16_m8_n4(
    device const half* x [[buffer(0)]],
    device const int* qweight [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const int* qzeros [[buffer(3)]],
    device const int* g_idx [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& M [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]], constant uint& groups [[buffer(9)]],
    constant uint& bits [[buffer(10)]], constant uint& planar [[buffer(11)]],
    constant uint& block_uniform [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint n = group * 4;
  threadgroup float4 decoded[32];
  threadgroup atomic_uint valid;
  if (tid == 0) atomic_store_explicit(&valid, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 sum = 0.0f;
  const bool active = simd < M;
  for (uint base = 0; base < K; base += 32) {
    if (simd == 0) {
      const uint k = base + lane;
      int gi = g_idx[block_uniform ? base : k];
      if (gi < 0) gi += int(groups);
      if (gi < 0 || gi >= int(groups)) {
        atomic_store_explicit(&valid, 0u, memory_order_relaxed);
        decoded[lane] = 0.0f;
      } else {
        const uint g = uint(gi);
        const uint4 code = planar ? planar_code4(qweight, k, n, N, bits)
                                  : continuous_row_code4(qweight, k, n, N, bits);
        uint4 zero = 0;
        float4 scale = 0.0f;
        if (!block_uniform || lane == 0) {
          zero = planar ? planar_zero4(qzeros, g, n, N, bits)
                        : continuous_zero4(qzeros, g, n, N, bits);
          scale = float4(*reinterpret_cast<device const half4*>(scales + g * N + n));
        }
        if (block_uniform) {
          zero.x = simd_broadcast_first(zero.x); zero.y = simd_broadcast_first(zero.y);
          zero.z = simd_broadcast_first(zero.z); zero.w = simd_broadcast_first(zero.w);
          scale.x = simd_broadcast_first(scale.x); scale.y = simd_broadcast_first(scale.y);
          scale.z = simd_broadcast_first(scale.z); scale.w = simd_broadcast_first(scale.w);
        }
        decoded[lane] = float4(int4(code) - int4(zero)) * scale;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) sum += float(x[simd * K + base + lane]) * decoded[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (active) {
    sum.x = simd_sum(sum.x); sum.y = simd_sum(sum.y);
    sum.z = simd_sum(sum.z); sum.w = simd_sum(sum.w);
    if (lane == 0) {
      const bool is_valid = atomic_load_explicit(&valid, memory_order_relaxed) != 0;
      *reinterpret_cast<device half4*>(output + simd * N + n) =
          is_valid ? half4(sum) : half4(NAN);
    }
  }
}

// Process eight adjacent output columns per group. Two float4 accumulators
// halve the N4 launch count without changing any per-output reduction order.
kernel void pangolin_fp16_m8_n8(
    device const half* x [[buffer(0)]],
    device const int* qweight [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const int* qzeros [[buffer(3)]],
    device const int* g_idx [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& M [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]], constant uint& groups [[buffer(9)]],
    constant uint& bits [[buffer(10)]], constant uint& planar [[buffer(11)]],
    constant uint& block_uniform [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint n = group * 8;
  threadgroup float4 decoded0[32], decoded1[32];
  threadgroup atomic_uint valid;
  if (tid == 0) atomic_store_explicit(&valid, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 sum0 = 0.0f, sum1 = 0.0f;
  const bool active = simd < M;
  for (uint base = 0; base < K; base += 32) {
    if (simd == 0) {
      const uint k = base + lane;
      int gi = g_idx[block_uniform ? base : k];
      if (gi < 0) gi += int(groups);
      if (gi < 0 || gi >= int(groups)) {
        atomic_store_explicit(&valid, 0u, memory_order_relaxed);
        decoded0[lane] = 0.0f; decoded1[lane] = 0.0f;
      } else {
        const uint g = uint(gi);
        const uint4 code0 = planar ? planar_code4(qweight, k, n, N, bits)
                                   : continuous_row_code4(qweight, k, n, N, bits);
        const uint4 code1 = planar ? planar_code4(qweight, k, n + 4, N, bits)
                                   : continuous_row_code4(qweight, k, n + 4, N, bits);
        uint4 zero0 = 0, zero1 = 0;
        float4 scale0 = 0.0f, scale1 = 0.0f;
        if (!block_uniform || lane == 0) {
          zero0 = planar ? planar_zero4(qzeros, g, n, N, bits)
                         : continuous_zero4(qzeros, g, n, N, bits);
          zero1 = planar ? planar_zero4(qzeros, g, n + 4, N, bits)
                         : continuous_zero4(qzeros, g, n + 4, N, bits);
          scale0 = float4(*reinterpret_cast<device const half4*>(scales + g * N + n));
          scale1 = float4(*reinterpret_cast<device const half4*>(scales + g * N + n + 4));
        }
        if (block_uniform) {
          zero0.x = simd_broadcast_first(zero0.x); zero0.y = simd_broadcast_first(zero0.y);
          zero0.z = simd_broadcast_first(zero0.z); zero0.w = simd_broadcast_first(zero0.w);
          zero1.x = simd_broadcast_first(zero1.x); zero1.y = simd_broadcast_first(zero1.y);
          zero1.z = simd_broadcast_first(zero1.z); zero1.w = simd_broadcast_first(zero1.w);
          scale0.x = simd_broadcast_first(scale0.x); scale0.y = simd_broadcast_first(scale0.y);
          scale0.z = simd_broadcast_first(scale0.z); scale0.w = simd_broadcast_first(scale0.w);
          scale1.x = simd_broadcast_first(scale1.x); scale1.y = simd_broadcast_first(scale1.y);
          scale1.z = simd_broadcast_first(scale1.z); scale1.w = simd_broadcast_first(scale1.w);
        }
        decoded0[lane] = float4(int4(code0) - int4(zero0)) * scale0;
        decoded1[lane] = float4(int4(code1) - int4(zero1)) * scale1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      const float value = float(x[simd * K + base + lane]);
      sum0 += value * decoded0[lane]; sum1 += value * decoded1[lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (active) {
    sum0.x = simd_sum(sum0.x); sum0.y = simd_sum(sum0.y);
    sum0.z = simd_sum(sum0.z); sum0.w = simd_sum(sum0.w);
    sum1.x = simd_sum(sum1.x); sum1.y = simd_sum(sum1.y);
    sum1.z = simd_sum(sum1.z); sum1.w = simd_sum(sum1.w);
    if (lane == 0) {
      const bool is_valid = atomic_load_explicit(&valid, memory_order_relaxed) != 0;
      *reinterpret_cast<device half4*>(output + simd * N + n) =
          is_valid ? half4(sum0) : half4(NAN);
      *reinterpret_cast<device half4*>(output + simd * N + n + 4) =
          is_valid ? half4(sum1) : half4(NAN);
    }
  }
}

// Four SIMD-groups share each decoded K lane across up to sixteen rows. This
// keeps all 128 threads useful for M=9..16 and avoids two independent M8
// launches (and two packed-weight decodes) for the same output column.
kernel void pangolin_fp16_m16(
    device const half* x [[buffer(0)]],
    device const int* qweight [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const int* qzeros [[buffer(3)]],
    device const int* g_idx [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& M [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]], constant uint& groups [[buffer(9)]],
    constant uint& bits [[buffer(10)]], constant uint& planar [[buffer(11)]],
    constant uint& block_uniform [[buffer(12)]],
    uint n [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float decoded[32];
  threadgroup atomic_uint valid;
  if (tid == 0) atomic_store_explicit(&valid, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  const uint row0 = simd, row1 = simd + 4, row2 = simd + 8, row3 = simd + 12;
  for (uint base = 0; base < K; base += 32) {
    if (simd == 0) {
      const uint k = base + lane;
      int gi = g_idx[block_uniform ? base : k];
      if (gi < 0) gi += int(groups);
      if (gi < 0 || gi >= int(groups)) {
        atomic_store_explicit(&valid, 0u, memory_order_relaxed);
        decoded[lane] = 0.0f;
      } else {
        const uint g = uint(gi);
        const uint code = planar ? planar_code(qweight, k, n, N, bits)
                                 : continuous_row_code(qweight, k, n, N, bits);
        uint zero = 0;
        float scale = 0.0f;
        if (!block_uniform || lane == 0) {
          zero = planar ? planar_zero(qzeros, g, n, N, bits)
                        : continuous_zero(qzeros, g, n, N, bits);
          scale = float(scales[g * N + n]);
        }
        if (block_uniform) {
          zero = simd_broadcast_first(zero);
          scale = simd_broadcast_first(scale);
        }
        decoded[lane] = float(int(code) - int(zero)) * scale;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float value = decoded[lane];
    if (row0 < M) sum0 += float(x[row0 * K + base + lane]) * value;
    if (row1 < M) sum1 += float(x[row1 * K + base + lane]) * value;
    if (row2 < M) sum2 += float(x[row2 * K + base + lane]) * value;
    if (row3 < M) sum3 += float(x[row3 * K + base + lane]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  sum0 = simd_sum(sum0); sum1 = simd_sum(sum1);
  sum2 = simd_sum(sum2); sum3 = simd_sum(sum3);
  if (lane == 0) {
    const bool is_valid = atomic_load_explicit(&valid, memory_order_relaxed) != 0;
    if (row0 < M) output[row0 * N + n] = is_valid ? half(sum0) : half(NAN);
    if (row1 < M) output[row1 * N + n] = is_valid ? half(sum1) : half(NAN);
    if (row2 < M) output[row2 * N + n] = is_valid ? half(sum2) : half(NAN);
    if (row3 < M) output[row3 * N + n] = is_valid ? half(sum3) : half(NAN);
  }
}

// Eight SIMD-groups each accumulate two rows and four adjacent columns. The
// 256-thread group keeps register pressure bounded while reducing both packed
// decode work and total launched threads for M=9..16.
kernel void pangolin_fp16_m16_n4(
    device const half* x [[buffer(0)]], device const int* qweight [[buffer(1)]],
    device const half* scales [[buffer(2)]], device const int* qzeros [[buffer(3)]],
    device const int* g_idx [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& M [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]], constant uint& groups [[buffer(9)]],
    constant uint& bits [[buffer(10)]], constant uint& planar [[buffer(11)]],
    constant uint& block_uniform [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  const uint n = group * 4;
  const uint row0 = simd, row1 = simd + 8;
  threadgroup float4 decoded[32];
  threadgroup atomic_uint valid;
  if (tid == 0) atomic_store_explicit(&valid, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 sum0 = 0.0f, sum1 = 0.0f;
  for (uint base = 0; base < K; base += 32) {
    if (simd == 0) {
      const uint k = base + lane;
      int gi = g_idx[block_uniform ? base : k];
      if (gi < 0) gi += int(groups);
      if (gi < 0 || gi >= int(groups)) {
        atomic_store_explicit(&valid, 0u, memory_order_relaxed);
        decoded[lane] = 0.0f;
      } else {
        const uint g = uint(gi);
        const uint4 code = planar ? planar_code4(qweight, k, n, N, bits)
                                  : continuous_row_code4(qweight, k, n, N, bits);
        uint4 zero = 0;
        float4 scale = 0.0f;
        if (!block_uniform || lane == 0) {
          zero = planar ? planar_zero4(qzeros, g, n, N, bits)
                        : continuous_zero4(qzeros, g, n, N, bits);
          scale = float4(*reinterpret_cast<device const half4*>(scales + g * N + n));
        }
        if (block_uniform) {
          zero.x = simd_broadcast_first(zero.x); zero.y = simd_broadcast_first(zero.y);
          zero.z = simd_broadcast_first(zero.z); zero.w = simd_broadcast_first(zero.w);
          scale.x = simd_broadcast_first(scale.x); scale.y = simd_broadcast_first(scale.y);
          scale.z = simd_broadcast_first(scale.z); scale.w = simd_broadcast_first(scale.w);
        }
        decoded[lane] = float4(int4(code) - int4(zero)) * scale;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float4 value = decoded[lane];
    sum0 += float(x[row0 * K + base + lane]) * value;
    if (row1 < M) sum1 += float(x[row1 * K + base + lane]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  sum0.x = simd_sum(sum0.x); sum0.y = simd_sum(sum0.y);
  sum0.z = simd_sum(sum0.z); sum0.w = simd_sum(sum0.w);
  sum1.x = simd_sum(sum1.x); sum1.y = simd_sum(sum1.y);
  sum1.z = simd_sum(sum1.z); sum1.w = simd_sum(sum1.w);
  if (lane == 0) {
    const bool is_valid = atomic_load_explicit(&valid, memory_order_relaxed) != 0;
    *reinterpret_cast<device half4*>(output + row0 * N + n) =
        is_valid ? half4(sum0) : half4(NAN);
    if (row1 < M) {
      *reinterpret_cast<device half4*>(output + row1 * N + n) =
          is_valid ? half4(sum1) : half4(NAN);
    }
  }
}

// Eight SIMD-groups each accumulate two rows and eight adjacent columns.
// This halves launch count for the largest decode projections; the N4 kernel
// remains preferable where the extra accumulators reduce occupancy.
kernel void pangolin_fp16_m16_n8(
    device const half* x [[buffer(0)]], device const int* qweight [[buffer(1)]],
    device const half* scales [[buffer(2)]], device const int* qzeros [[buffer(3)]],
    device const int* g_idx [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& M [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]], constant uint& groups [[buffer(9)]],
    constant uint& bits [[buffer(10)]], constant uint& planar [[buffer(11)]],
    constant uint& block_uniform [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  const uint n = group * 8;
  const uint row0 = simd, row1 = simd + 8;
  threadgroup float4 decoded0[32], decoded1[32];
  threadgroup atomic_uint valid;
  if (tid == 0) atomic_store_explicit(&valid, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
  for (uint base = 0; base < K; base += 32) {
    if (simd == 0) {
      const uint k = base + lane;
      int gi = g_idx[block_uniform ? base : k];
      if (gi < 0) gi += int(groups);
      if (gi < 0 || gi >= int(groups)) {
        atomic_store_explicit(&valid, 0u, memory_order_relaxed);
        decoded0[lane] = 0.0f; decoded1[lane] = 0.0f;
      } else {
        const uint g = uint(gi);
        const uint4 code0 = planar ? planar_code4(qweight, k, n, N, bits)
                                   : continuous_row_code4(qweight, k, n, N, bits);
        const uint4 code1 = planar ? planar_code4(qweight, k, n + 4, N, bits)
                                   : continuous_row_code4(qweight, k, n + 4, N, bits);
        uint4 zero0 = 0, zero1 = 0;
        float4 scale0 = 0.0f, scale1 = 0.0f;
        if (!block_uniform || lane == 0) {
          zero0 = planar ? planar_zero4(qzeros, g, n, N, bits)
                         : continuous_zero4(qzeros, g, n, N, bits);
          zero1 = planar ? planar_zero4(qzeros, g, n + 4, N, bits)
                         : continuous_zero4(qzeros, g, n + 4, N, bits);
          scale0 = float4(*reinterpret_cast<device const half4*>(scales + g * N + n));
          scale1 = float4(*reinterpret_cast<device const half4*>(scales + g * N + n + 4));
        }
        if (block_uniform) {
          zero0.x = simd_broadcast_first(zero0.x); zero0.y = simd_broadcast_first(zero0.y);
          zero0.z = simd_broadcast_first(zero0.z); zero0.w = simd_broadcast_first(zero0.w);
          zero1.x = simd_broadcast_first(zero1.x); zero1.y = simd_broadcast_first(zero1.y);
          zero1.z = simd_broadcast_first(zero1.z); zero1.w = simd_broadcast_first(zero1.w);
          scale0.x = simd_broadcast_first(scale0.x); scale0.y = simd_broadcast_first(scale0.y);
          scale0.z = simd_broadcast_first(scale0.z); scale0.w = simd_broadcast_first(scale0.w);
          scale1.x = simd_broadcast_first(scale1.x); scale1.y = simd_broadcast_first(scale1.y);
          scale1.z = simd_broadcast_first(scale1.z); scale1.w = simd_broadcast_first(scale1.w);
        }
        decoded0[lane] = float4(int4(code0) - int4(zero0)) * scale0;
        decoded1[lane] = float4(int4(code1) - int4(zero1)) * scale1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float4 value0 = decoded0[lane], value1 = decoded1[lane];
    const float x0 = float(x[row0 * K + base + lane]);
    sum00 += x0 * value0; sum01 += x0 * value1;
    if (row1 < M) {
      const float x1 = float(x[row1 * K + base + lane]);
      sum10 += x1 * value0; sum11 += x1 * value1;
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
    const bool is_valid = atomic_load_explicit(&valid, memory_order_relaxed) != 0;
    *reinterpret_cast<device half4*>(output + row0 * N + n) =
        is_valid ? half4(sum00) : half4(NAN);
    *reinterpret_cast<device half4*>(output + row0 * N + n + 4) =
        is_valid ? half4(sum01) : half4(NAN);
    if (row1 < M) {
      *reinterpret_cast<device half4*>(output + row1 * N + n) =
          is_valid ? half4(sum10) : half4(NAN);
      *reinterpret_cast<device half4*>(output + row1 * N + n + 4) =
          is_valid ? half4(sum11) : half4(NAN);
    }
  }
}

// Share one decoded K lane across as many as 32 activation rows. Apple GPUs
// expose eight SIMD-groups per 256-thread group, so each SIMD-group accumulates
// up to four rows. This avoids decoding the packed weight again in four
// separate launches for M=32.
kernel void pangolin_fp16_m32(
    device const half* x [[buffer(0)]],
    device const int* qweight [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const int* qzeros [[buffer(3)]],
    device const int* g_idx [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& M [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]], constant uint& groups [[buffer(9)]],
    constant uint& bits [[buffer(10)]], constant uint& planar [[buffer(11)]],
    constant uint& block_uniform [[buffer(12)]],
    uint n [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float decoded[32];
  threadgroup atomic_uint valid;
  if (tid == 0) atomic_store_explicit(&valid, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  const uint row0 = simd, row1 = simd + 8, row2 = simd + 16, row3 = simd + 24;
  for (uint base = 0; base < K; base += 32) {
    if (simd == 0) {
      const uint k = base + lane;
      int gi = g_idx[block_uniform ? base : k];
      if (gi < 0) gi += int(groups);
      if (gi < 0 || gi >= int(groups)) {
        atomic_store_explicit(&valid, 0u, memory_order_relaxed);
        decoded[lane] = 0.0f;
      } else {
        const uint g = uint(gi);
        const uint code = planar ? planar_code(qweight, k, n, N, bits)
                                 : continuous_row_code(qweight, k, n, N, bits);
        uint zero = 0;
        float scale = 0.0f;
        if (!block_uniform || lane == 0) {
          zero = planar ? planar_zero(qzeros, g, n, N, bits)
                        : continuous_zero(qzeros, g, n, N, bits);
          scale = float(scales[g * N + n]);
        }
        if (block_uniform) {
          zero = simd_broadcast_first(zero);
          scale = simd_broadcast_first(scale);
        }
        decoded[lane] = float(int(code) - int(zero)) * scale;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float value = decoded[lane];
    if (row0 < M) sum0 += float(x[row0 * K + base + lane]) * value;
    if (row1 < M) sum1 += float(x[row1 * K + base + lane]) * value;
    if (row2 < M) sum2 += float(x[row2 * K + base + lane]) * value;
    if (row3 < M) sum3 += float(x[row3 * K + base + lane]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  sum0 = simd_sum(sum0); sum1 = simd_sum(sum1);
  sum2 = simd_sum(sum2); sum3 = simd_sum(sum3);
  if (lane == 0) {
    const bool is_valid = atomic_load_explicit(&valid, memory_order_relaxed) != 0;
    if (row0 < M) output[row0 * N + n] = is_valid ? half(sum0) : half(NAN);
    if (row1 < M) output[row1 * N + n] = is_valid ? half(sum1) : half(NAN);
    if (row2 < M) output[row2 * N + n] = is_valid ? half(sum2) : half(NAN);
    if (row3 < M) output[row3 * N + n] = is_valid ? half(sum3) : half(NAN);
  }
}
"""


def pangolin_mps_supported() -> bool:
    return bool(
        torch.backends.mps.is_available()
        and callable(getattr(torch.mps, "compile_shader", None))
    )


def _library():
    global _LIBRARY, _LIBRARY_ERROR
    if _LIBRARY is None:
        with _LIBRARY_LOCK:
            if _LIBRARY is None:
                if _LIBRARY_ERROR is not None:
                    raise RuntimeError(_LIBRARY_ERROR)
                if not pangolin_mps_supported():
                    raise RuntimeError(
                        "Pangolin Metal requires torch.mps.compile_shader"
                    )
                try:
                    _LIBRARY = torch.mps.compile_shader(_SOURCE)
                except Exception as exc:
                    _LIBRARY_ERROR = f"Pangolin Metal shader compilation failed: {exc}"
                    raise RuntimeError(_LIBRARY_ERROR) from exc
    return _LIBRARY


def pangolin_mps_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
    *,
    planar: bool,
    _g_idx_validated: bool = False,
    _g_idx_block_uniform: bool | None = None,
) -> torch.Tensor:
    """Multiply ``x[M,K]`` by packed GPTQ weights without materializing them."""
    if bits not in PANGOLIN_MPS_BITS:
        raise ValueError(
            f"Pangolin Metal supports bits {PANGOLIN_MPS_BITS}, got {bits}"
        )
    if planar != (bits in (3, 5, 6, 7)):
        raise ValueError(f"invalid planar={planar} for {bits}-bit gptq_p")
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")
    if x.device.type != "mps":
        raise ValueError(f"Pangolin Metal tensors must be on MPS, got {x.device}")
    tensors = (x, qweight, scales, qzeros, g_idx)
    if any(t.device != x.device for t in tensors):
        raise ValueError("Pangolin Metal tensors must share one MPS device")
    if x.dtype != torch.float16 or scales.dtype != torch.float16:
        raise TypeError("Pangolin Metal currently requires float16 x and scales")
    if (
        qweight.dtype != torch.int32
        or qzeros.dtype != torch.int32
        or g_idx.dtype != torch.int32
    ):
        raise TypeError("Pangolin Metal requires int32 qweight, qzeros, and g_idx")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("Pangolin Metal tensors must be contiguous")
    m, k = x.shape
    groups, n = scales.shape
    if groups == 0 or k == 0 or n == 0 or k % 32 or n % 32:
        raise ValueError(
            f"Pangolin Metal requires positive groups and K/N divisible by 32, "
            f"got groups={groups}, K={k}, N={n}"
        )
    if (
        max(
            m,
            k,
            n,
            groups,
            m * n,
            *(tensor.numel() for tensor in tensors),
        )
        > 2**32 - 1
    ):
        raise ValueError("Pangolin Metal dimensions exceed the uint32 kernel limit")
    if g_idx.shape != (k,):
        raise ValueError(f"g_idx must have shape {(k,)}, got {tuple(g_idx.shape)}")
    expected_qweight = ((k // 32) * bits, n) if planar else (k // (32 // bits), n)
    expected_qzeros = (
        (groups, (n // 32) * bits) if planar else (groups, n // (32 // bits))
    )
    if qweight.shape != expected_qweight or qzeros.shape != expected_qzeros:
        raise ValueError(
            f"packed shapes must be qweight={expected_qweight}, qzeros={expected_qzeros}; "
            f"got {tuple(qweight.shape)}, {tuple(qzeros.shape)}"
        )
    if m == 0:
        return torch.empty((0, n), dtype=x.dtype, device=x.device)
    # Validate before the unchecked shader indexing. This synchronizes once per
    # direct call; qlinear validates the same invariant once during post_init.
    if not _g_idx_validated and (
        int(g_idx.min()) < -groups or int(g_idx.max()) >= groups
    ):
        raise ValueError(
            "g_idx contains a group index outside the scales/qzeros bounds"
        )
    if _g_idx_block_uniform is None:
        blocks = g_idx.reshape(-1, 32)
        _g_idx_block_uniform = bool((blocks == blocks[:, :1]).all().item())
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    library = _library()
    vector_loads_aligned = (
        qweight.storage_offset() % 4 == 0
        and scales.storage_offset() % 4 == 0
        and qzeros.storage_offset() % 4 == 0
    )
    if m <= 3 and (bits != 2 or k >= 2048) and vector_loads_aligned:
        if planar and _g_idx_block_uniform:
            library.pangolin_fp16_m3_n4_planar_uniform(
                x,
                qweight,
                scales,
                qzeros,
                g_idx,
                output,
                m,
                k,
                n,
                groups,
                bits,
                threads=m * (n // 4) * 32,
                group_size=32,
            )
        elif _g_idx_block_uniform:
            library.pangolin_fp16_m3_n4_continuous_uniform(
                x,
                qweight,
                scales,
                qzeros,
                g_idx,
                output,
                m,
                k,
                n,
                groups,
                bits,
                threads=m * (n // 4) * 32,
                group_size=32,
            )
        else:
            library.pangolin_fp16_m3_n4(
                x,
                qweight,
                scales,
                qzeros,
                g_idx,
                output,
                m,
                k,
                n,
                groups,
                bits,
                int(planar),
                int(_g_idx_block_uniform),
                threads=m * (n // 4) * 32,
                group_size=32,
            )
        return output
    # Sharing amortizes its two barriers per K block from M >= 4. For M=1/2/3
    # independent SIMD groups are faster because there is too little reuse.
    if 4 <= m <= 8:
        vectorize_columns = not planar and vector_loads_aligned
        vector_width = 8 if vectorize_columns and k >= 4096 and n >= 4096 else 4
        if vector_width == 8:
            kernel = library.pangolin_fp16_m8_n8
        elif vectorize_columns:
            kernel = library.pangolin_fp16_m8_n4
        else:
            kernel = library.pangolin_fp16_m8
        kernel(
            x,
            qweight,
            scales,
            qzeros,
            g_idx,
            output,
            m,
            k,
            n,
            groups,
            bits,
            int(planar),
            int(_g_idx_block_uniform),
            threads=(n // vector_width if vectorize_columns else n) * 256,
            group_size=256,
        )
        return output
    if 9 <= m <= 16:
        vectorize_columns = vector_loads_aligned
        use_n8 = (
            vectorize_columns
            and k >= 2048
            and ((not planar and n >= 2048) or (planar and n >= 4096))
        )
        vector_width = 8 if use_n8 else 4
        if use_n8:
            kernel = library.pangolin_fp16_m16_n8
        elif vectorize_columns:
            kernel = library.pangolin_fp16_m16_n4
        else:
            kernel = library.pangolin_fp16_m16
        kernel(
            x,
            qweight,
            scales,
            qzeros,
            g_idx,
            output,
            m,
            k,
            n,
            groups,
            bits,
            int(planar),
            int(_g_idx_block_uniform),
            threads=(n // vector_width) * 256 if vectorize_columns else n * 128,
            group_size=256 if vectorize_columns else 128,
        )
        return output
    if 17 <= m < 24:
        for start in range(0, m, 8):
            width = min(8, m - start)
            library.pangolin_fp16_m8(
                x.narrow(0, start, width),
                qweight,
                scales,
                qzeros,
                g_idx,
                output.narrow(0, start, width),
                width,
                k,
                n,
                groups,
                bits,
                int(planar),
                int(_g_idx_block_uniform),
                threads=n * 256,
                group_size=256,
            )
        return output
    if 24 <= m <= 32:
        library.pangolin_fp16_m32(
            x,
            qweight,
            scales,
            qzeros,
            g_idx,
            output,
            m,
            k,
            n,
            groups,
            bits,
            int(planar),
            int(_g_idx_block_uniform),
            threads=n * 256,
            group_size=256,
        )
        return output
    library.pangolin_fp16(
        x,
        qweight,
        scales,
        qzeros,
        g_idx,
        output,
        m,
        k,
        n,
        groups,
        bits,
        int(planar),
        int(_g_idx_block_uniform),
        threads=m * n * 32,
        group_size=32,
    )
    return output


__all__ = ["PANGOLIN_MPS_BITS", "pangolin_mps_gemv", "pangolin_mps_supported"]
