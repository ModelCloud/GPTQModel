// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
#include <immintrin.h>
#endif

namespace qvq_cpu {

namespace {

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
__attribute__((constructor))
static void qvq_viterbi_init_cpu_features() {
  __builtin_cpu_init();
}

inline bool cpu_has_avx512() {
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
}
#else
inline bool cpu_has_avx512() { return false; }
#endif

}  // namespace

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
static void emit_distance_avx512_v2(
    int64_t state_count,
    const float* __restrict__ codebook_t,
    const float* __restrict__ codebook_norm,
    const float* __restrict__ target,
    float target_norm,
    float weight,
    float* __restrict__ emission,
    int64_t begin,
    int64_t end) {
  const float* c0 = codebook_t + 0 * state_count;
  const float* c1 = codebook_t + 1 * state_count;
  const __m512 t0v = _mm512_set1_ps(target[0]);
  const __m512 t1v = _mm512_set1_ps(target[1]);
  const __m512 tnv = _mm512_set1_ps(target_norm);
  const __m512 wv = _mm512_set1_ps(weight);
  const __m512 two = _mm512_set1_ps(2.0f);
  const __m512 zero = _mm512_setzero_ps();
  int64_t s = begin;
  for (; s < end && (s & 15); ++s) {
    float dot = target[0] * c0[s] + target[1] * c1[s];
    float dist = target_norm + codebook_norm[s] - 2.0f * dot;
    if (dist < 0.0f) dist = 0.0f;
    emission[s] = dist * weight;
  }
  for (; s + 16 <= end; s += 16) {
    __m512 v0 = _mm512_loadu_ps(c0 + s);
    __m512 v1 = _mm512_loadu_ps(c1 + s);
    __m512 dot = _mm512_mul_ps(v1, t1v);
    dot = _mm512_fmadd_ps(v0, t0v, dot);
    __m512 cn = _mm512_loadu_ps(codebook_norm + s);
    __m512 dist = _mm512_fnmadd_ps(dot, two, _mm512_add_ps(tnv, cn));
    dist = _mm512_max_ps(dist, zero);
    if (weight != 1.0f) {
      dist = _mm512_mul_ps(dist, wv);
    }
    _mm512_storeu_ps(emission + s, dist);
  }
  for (; s < end; ++s) {
    float dot = target[0] * c0[s] + target[1] * c1[s];
    float dist = target_norm + codebook_norm[s] - 2.0f * dot;
    if (dist < 0.0f) dist = 0.0f;
    emission[s] = dist * weight;
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
static void emit_distance_avx512_v4(
    int64_t state_count,
    const float* __restrict__ codebook_t,
    const float* __restrict__ codebook_norm,
    const float* __restrict__ target,
    float target_norm,
    float weight,
    float* __restrict__ emission,
    int64_t begin,
    int64_t end) {
  const float* c0 = codebook_t + 0 * state_count;
  const float* c1 = codebook_t + 1 * state_count;
  const float* c2 = codebook_t + 2 * state_count;
  const float* c3 = codebook_t + 3 * state_count;
  const __m512 t0v = _mm512_set1_ps(target[0]);
  const __m512 t1v = _mm512_set1_ps(target[1]);
  const __m512 t2v = _mm512_set1_ps(target[2]);
  const __m512 t3v = _mm512_set1_ps(target[3]);
  const __m512 tnv = _mm512_set1_ps(target_norm);
  const __m512 wv = _mm512_set1_ps(weight);
  const __m512 two = _mm512_set1_ps(2.0f);
  const __m512 zero = _mm512_setzero_ps();
  int64_t s = begin;
  for (; s < end && (s & 15); ++s) {
    float dot = target[0] * c0[s] + target[1] * c1[s] + target[2] * c2[s] + target[3] * c3[s];
    float dist = target_norm + codebook_norm[s] - 2.0f * dot;
    if (dist < 0.0f) dist = 0.0f;
    emission[s] = dist * weight;
  }
  for (; s + 16 <= end; s += 16) {
    __m512 dot = _mm512_setzero_ps();
    dot = _mm512_fmadd_ps(_mm512_loadu_ps(c0 + s), t0v, dot);
    dot = _mm512_fmadd_ps(_mm512_loadu_ps(c1 + s), t1v, dot);
    dot = _mm512_fmadd_ps(_mm512_loadu_ps(c2 + s), t2v, dot);
    dot = _mm512_fmadd_ps(_mm512_loadu_ps(c3 + s), t3v, dot);
    __m512 cn = _mm512_loadu_ps(codebook_norm + s);
    __m512 dist = _mm512_fnmadd_ps(dot, two, _mm512_add_ps(tnv, cn));
    dist = _mm512_max_ps(dist, zero);
    if (weight != 1.0f) {
      dist = _mm512_mul_ps(dist, wv);
    }
    _mm512_storeu_ps(emission + s, dist);
  }
  for (; s < end; ++s) {
    float dot = target[0] * c0[s] + target[1] * c1[s] + target[2] * c2[s] + target[3] * c3[s];
    float dist = target_norm + codebook_norm[s] - 2.0f * dot;
    if (dist < 0.0f) dist = 0.0f;
    emission[s] = dist * weight;
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
static void column_argmin_avx512(
    const float* __restrict__ matrix,
    int64_t row_count,
    int64_t suffix_count,
    float* __restrict__ best_cost,
    int32_t* __restrict__ best_arg,
    int64_t col_begin,
    int64_t col_end) {
  const float inf = std::numeric_limits<float>::infinity();
  int64_t col = col_begin;
  for (; col + 16 <= col_end; col += 16) {
    __m512 best_v = _mm512_set1_ps(inf);
    __m512i best_a = _mm512_setzero_epi32();
    for (int64_t row = 0; row < row_count; ++row) {
      const float* row_ptr = matrix + row * suffix_count + col;
      __m512 vals = _mm512_loadu_ps(row_ptr);
      __mmask16 mask = _mm512_cmp_ps_mask(vals, best_v, _CMP_LT_OQ);
      best_v = _mm512_mask_blend_ps(mask, best_v, vals);
      best_a = _mm512_mask_blend_epi32(mask, best_a, _mm512_set1_epi32(static_cast<int>(row)));
    }
    _mm512_storeu_ps(best_cost + col, best_v);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(best_arg + col), best_a);
  }
  for (; col < col_end; ++col) {
    float best_v = inf;
    int32_t best_a = 0;
    for (int64_t row = 0; row < row_count; ++row) {
      float v = matrix[row * suffix_count + col];
      if (v < best_v) {
        best_v = v;
        best_a = static_cast<int32_t>(row);
      }
    }
    best_cost[col] = best_v;
    best_arg[col] = best_a;
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
static void broadcast_add_avx512(
    const float* __restrict__ emission,
    const float* __restrict__ best_cost,
    int64_t prefix_count,
    float* __restrict__ next_costs,
    int64_t begin,
    int64_t end) {
  int64_t block_begin = (begin + prefix_count - 1) / prefix_count;
  int64_t block_end = end / prefix_count;
  for (int64_t s = begin; s < block_begin * prefix_count; ++s) {
    next_costs[s] = emission[s] + best_cost[s / prefix_count];
  }
  for (int64_t block = block_begin; block < block_end; ++block) {
    __m512 trans = _mm512_set1_ps(best_cost[block]);
    int64_t s0 = block * prefix_count;
    int64_t s1 = s0 + prefix_count;
    for (int64_t s = s0; s < s1; s += 16) {
      __m512 e = _mm512_loadu_ps(emission + s);
      _mm512_storeu_ps(next_costs + s, _mm512_add_ps(e, trans));
    }
  }
  for (int64_t s = block_end * prefix_count; s < end; ++s) {
    next_costs[s] = emission[s] + best_cost[s / prefix_count];
  }
}
#endif

// Compute squared Euclidean emission distances for one batch into emission[begin:end].
// codebook_t is [vector_size, state_count] row-major (contiguous).
// codebook_norm is [state_count].
// target is [vector_size].
// If weight != 1.0f the distance is multiplied by weight (step weight).
// Output is clamped to [0, inf).
inline void emit_distance(
    int64_t state_count,
    int64_t vector_size,
    const float* __restrict__ codebook_t,
    const float* __restrict__ codebook_norm,
    const float* __restrict__ target,
    float target_norm,
    float weight,
    float* __restrict__ emission,
    int64_t begin,
    int64_t end) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
  if (cpu_has_avx512() && vector_size == 2) {
    emit_distance_avx512_v2(state_count, codebook_t, codebook_norm, target, target_norm, weight, emission, begin, end);
    return;
  }
  if (cpu_has_avx512() && vector_size == 4) {
    emit_distance_avx512_v4(state_count, codebook_t, codebook_norm, target, target_norm, weight, emission, begin, end);
    return;
  }
#endif

  for (int64_t s = begin; s < end; ++s) {
    float dot = 0.0f;
    for (int64_t v = 0; v < vector_size; ++v) {
      dot += target[v] * codebook_t[v * state_count + s];
    }
    float dist = target_norm + codebook_norm[s] - 2.0f * dot;
    if (dist < 0.0f) dist = 0.0f;
    emission[s] = dist * weight;
  }
}

// Compute per-column minimum and argmin for a matrix with `row_count` rows and
// `suffix_count` columns stored row-major at `matrix`.
// Only columns [col_begin, col_end) are processed.
// best_cost and best_arg must have at least suffix_count entries.
inline void column_argmin(
    const float* __restrict__ matrix,
    int64_t row_count,
    int64_t suffix_count,
    float* __restrict__ best_cost,
    int32_t* __restrict__ best_arg,
    int64_t col_begin,
    int64_t col_end) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
  if (cpu_has_avx512() && suffix_count >= 16) {
    column_argmin_avx512(matrix, row_count, suffix_count, best_cost, best_arg, col_begin, col_end);
    return;
  }
#endif

  const float inf = std::numeric_limits<float>::infinity();
  for (int64_t col = col_begin; col < col_end; ++col) {
    float best_v = inf;
    int32_t best_a = 0;
    for (int64_t row = 0; row < row_count; ++row) {
      float v = matrix[row * suffix_count + col];
      if (v < best_v) {
        best_v = v;
        best_a = static_cast<int32_t>(row);
      }
    }
    best_cost[col] = best_v;
    best_arg[col] = best_a;
  }
}

// Compute next_costs[s] = emission[s] + best_cost[s / prefix_count]
// for s in [begin, end).  prefix_count must be a power of two.
inline void broadcast_add(
    const float* __restrict__ emission,
    const float* __restrict__ best_cost,
    int64_t prefix_count,
    int64_t suffix_count,
    float* __restrict__ next_costs,
    int64_t begin,
    int64_t end) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
  if (cpu_has_avx512() && prefix_count >= 16) {
    broadcast_add_avx512(emission, best_cost, prefix_count, next_costs, begin, end);
    return;
  }
#endif

  int64_t block_begin = (begin + prefix_count - 1) / prefix_count;
  int64_t block_end = end / prefix_count;
  for (int64_t s = begin; s < block_begin * prefix_count; ++s) {
    next_costs[s] = emission[s] + best_cost[s / prefix_count];
  }
  for (int64_t block = block_begin; block < block_end; ++block) {
    float trans = best_cost[block];
    int64_t s0 = block * prefix_count;
    int64_t s1 = s0 + prefix_count;
    for (int64_t s = s0; s < s1; ++s) {
      next_costs[s] = emission[s] + trans;
    }
  }
  for (int64_t s = block_end * prefix_count; s < end; ++s) {
    next_costs[s] = emission[s] + best_cost[s / prefix_count];
  }
}

}  // namespace qvq_cpu
