// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Fused AVX-512 CPU Viterbi trellis quantization kernel.
//
// Mirrors the CUDA kernel design in qvq_viterbi_cuda.cu: the emission distance
// is computed inline inside the transition sweep instead of being materialized
// into a separate tensor, the tail-overlap constraint is applied inline, and
// int16 backpointers are packed directly from 32-bit lanes.
//
// Transition-group geometry (identical to the baseline kernel and the torch
// oracle): predecessors of state s are {h * suffix_count + (s >> shift)} for
// h in [0, 2**shift), so the per-step update is
//
//   next[s] = emission(s) + best[s >> shift],
//   best[k] = min_h costs[h * suffix_count + k].
//
// The baseline realizes this with five sweeps per step (emission materialize,
// argmin, broadcast add, end-mask fill, backpointer convert).  This kernel
// uses two sweeps: an argmin sweep that writes backpointers straight out of
// the 32-bit lanes, and a fused emission+add sweep whose best-cost lookup is
// a register gather.  Arithmetic order, clamping, weight handling, and
// first-index-wins tie breaking match the baseline bit-for-bit.

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <ATen/Parallel.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
#include <immintrin.h>
#endif

namespace qvq_cpu_opt {

namespace {

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
__attribute__((constructor))
static void qvq_viterbi_opt_init_cpu_features() {
  __builtin_cpu_init();
}

inline bool cpu_has_avx512() {
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
}
#else
inline bool cpu_has_avx512() { return false; }
#endif

inline int log2_state_count(int64_t state_count) {
  int l = 0;
  int64_t s = state_count;
  while (s > 1) {
    s >>= 1;
    ++l;
  }
  return l;
}

// ---------------------------------------------------------------------------
// Portable reference paths (used without AVX-512 and for tiny trellises).
// ---------------------------------------------------------------------------

static void emission_portable(
    int64_t vector_size,
    int64_t state_count,
    const float* __restrict__ codebook_t,
    const float* __restrict__ codebook_norm,
    const float* __restrict__ target,
    float target_norm,
    float weight,
    float* __restrict__ out,
    int64_t begin,
    int64_t end) {
  for (int64_t s = begin; s < end; ++s) {
    float dot = 0.0f;
    for (int64_t v = 0; v < vector_size; ++v) {
      dot += target[v] * codebook_t[v * state_count + s];
    }
    float dist = target_norm + codebook_norm[s] - 2.0f * dot;
    if (dist < 0.0f) dist = 0.0f;
    out[s] = dist * weight;
  }
}

static void argmin_keys_portable(
    int64_t prefix_count,
    int64_t suffix_count,
    const float* __restrict__ costs_b,
    float* __restrict__ best_cost,
    int32_t* __restrict__ best_prefix_i32,
    int16_t* __restrict__ best_prefix_i16,
    bool use_int16,
    int64_t key_begin,
    int64_t key_end) {
  const float inf = std::numeric_limits<float>::infinity();
  for (int64_t key = key_begin; key < key_end; ++key) {
    float best_v = inf;
    int32_t best_a = 0;
    for (int64_t h = 0; h < prefix_count; ++h) {
      float v = costs_b[h * suffix_count + key];
      if (v < best_v) {
        best_v = v;
        best_a = static_cast<int32_t>(h);
      }
    }
    best_cost[key] = best_v;
    if (use_int16) {
      best_prefix_i16[key] = static_cast<int16_t>(best_a);
    } else {
      best_prefix_i32[key] = best_a;
    }
  }
}

template <bool ConstrainedEnd>
static void emit_add_portable(
    int64_t state_count,
    const float* __restrict__ best_cost,
    float* __restrict__ next_b,
    int64_t overlap_shift,
    int64_t begin,
    int64_t end,
    int64_t overlap_value,
    int64_t overlap_mask) {
  (void)state_count;
  for (int64_t s = begin; s < end; ++s) {
    float value = next_b[s] + best_cost[s >> overlap_shift];
    if constexpr (ConstrainedEnd) {
      if ((s & overlap_mask) != overlap_value) {
        value = std::numeric_limits<float>::infinity();
      }
    }
    next_b[s] = value;
  }
}

// ---------------------------------------------------------------------------
// AVX-512 paths.
// ---------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)

// Squared Euclidean emission over the global state range [begin, end).
// Arithmetic is identical to emit_distance_avx512_v2/v4 in qvq_viterbi_simd.h.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
static void emission_avx512(
    int64_t vector_size,
    int64_t state_count,
    const float* __restrict__ codebook_t,
    const float* __restrict__ codebook_norm,
    const float* __restrict__ target,
    float target_norm,
    float weight,
    float* __restrict__ out,
    int64_t begin,
    int64_t end) {
  const float* c0 = codebook_t;
  const float* c1 = codebook_t + state_count;
  const __m512 tnv = _mm512_set1_ps(target_norm);
  const __m512 wv = _mm512_set1_ps(weight);
  const __m512 two = _mm512_set1_ps(2.0f);
  const __m512 zero = _mm512_setzero_ps();
  if (vector_size == 2) {
    const __m512 t0v = _mm512_set1_ps(target[0]);
    const __m512 t1v = _mm512_set1_ps(target[1]);
    int64_t s = begin;
    // Both lanes below accumulate from zero in coordinate order (c0 then c1).
    // A leading `_mm512_mul_ps` pre-rounds one product and leaves the choice of
    // which one to the compiler's contraction of the scalar expression, so the
    // 16-lane body and the scalar prologue/tail can end up rounding different
    // coordinates.  Chunk alignment decides which states take which lane, so
    // that desynchronisation can flip exact Viterbi winners.  Mirrors the fix
    // in `qvq_viterbi_cpu.cpp` and the V=4 vector body below.
    for (; s < end && (s & 15); ++s) {
      float dot = 0.0f;
      dot += target[0] * c0[s];
      dot += target[1] * c1[s];
      float dist = target_norm + codebook_norm[s] - 2.0f * dot;
      if (dist < 0.0f) dist = 0.0f;
      out[s] = dist * weight;
    }
    for (; s + 16 <= end; s += 16) {
      __m512 dot = _mm512_setzero_ps();
      dot = _mm512_fmadd_ps(_mm512_loadu_ps(c0 + s), t0v, dot);
      dot = _mm512_fmadd_ps(_mm512_loadu_ps(c1 + s), t1v, dot);
      __m512 cn = _mm512_loadu_ps(codebook_norm + s);
      __m512 dist = _mm512_fnmadd_ps(dot, two, _mm512_add_ps(tnv, cn));
      dist = _mm512_max_ps(dist, zero);
      if (weight != 1.0f) {
        dist = _mm512_mul_ps(dist, wv);
      }
      _mm512_storeu_ps(out + s, dist);
    }
    for (; s < end; ++s) {
      float dot = 0.0f;
      dot += target[0] * c0[s];
      dot += target[1] * c1[s];
      float dist = target_norm + codebook_norm[s] - 2.0f * dot;
      if (dist < 0.0f) dist = 0.0f;
      out[s] = dist * weight;
    }
    return;
  }
  const float* c2 = codebook_t + 2 * state_count;
  const float* c3 = codebook_t + 3 * state_count;
  const __m512 t0v = _mm512_set1_ps(target[0]);
  const __m512 t1v = _mm512_set1_ps(target[1]);
  const __m512 t2v = _mm512_set1_ps(target[2]);
  const __m512 t3v = _mm512_set1_ps(target[3]);
  int64_t s = begin;
  // Accumulate from zero in coordinate order (c0..c3) so these scalar lanes
  // match the zero-seeded FMA chain in the 16-lane body below.  Written as a
  // sum expression, gcc contracts the *first* product into an FMA and pre-rounds
  // one of the others, which desynchronises the two lanes -- the same defect
  // fixed in the V=2 arm above.  This head/tail is currently UNREACHABLE (the op
  // requires a power-of-two state_count >= 16 and partitions sweep B on
  // state_count alone, so every [begin, end) is 16-aligned with a 16-multiple
  // length; an instrumented build counted zero entries here across the whole
  // test matrix).  It is written this way so the agreement is structural rather
  // than a property of today's chunking -- the production V=2 defect stayed
  // hidden in exactly this way until a chunk size stopped being a multiple of 16.
  for (; s < end && (s & 15); ++s) {
    float dot = 0.0f;
    dot += target[0] * c0[s];
    dot += target[1] * c1[s];
    dot += target[2] * c2[s];
    dot += target[3] * c3[s];
    float dist = target_norm + codebook_norm[s] - 2.0f * dot;
    if (dist < 0.0f) dist = 0.0f;
    out[s] = dist * weight;
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
    _mm512_storeu_ps(out + s, dist);
  }
  for (; s < end; ++s) {
    float dot = 0.0f;
    dot += target[0] * c0[s];
    dot += target[1] * c1[s];
    dot += target[2] * c2[s];
    dot += target[3] * c3[s];
    float dist = target_norm + codebook_norm[s] - 2.0f * dot;
    if (dist < 0.0f) dist = 0.0f;
    out[s] = dist * weight;
  }
}

// Sweep A: transition-group min over prefix rows for keys [key_begin,
// key_end).  best_cost[key] = min_h costs[h * suffix_count + key]; the first
// (smallest h) row wins ties exactly like torch.min(dim=1).  Backpointers are
// packed straight from the 32-bit lanes.
template <bool UseInt16>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
static void argmin_keys_avx512(
    int64_t prefix_count,
    int64_t suffix_count,
    const float* __restrict__ costs_b,
    float* __restrict__ best_cost,
    int32_t* __restrict__ best_prefix_i32,
    int16_t* __restrict__ best_prefix_i16,
    int64_t key_begin,
    int64_t key_end) {
  constexpr int64_t kVec = 16;
  const float inf = std::numeric_limits<float>::infinity();
  int64_t key = key_begin;
  for (; key + kVec <= key_end; key += kVec) {
    __m512 best_v = _mm512_set1_ps(inf);
    __m512i best_a = _mm512_setzero_si512();
    for (int64_t h = 0; h < prefix_count; ++h) {
      const float* row_ptr = costs_b + h * suffix_count + key;
      __m512 vals = _mm512_loadu_ps(row_ptr);
      __mmask16 mask = _mm512_cmp_ps_mask(vals, best_v, _CMP_LT_OQ);
      best_v = _mm512_mask_blend_ps(mask, best_v, vals);
      best_a = _mm512_mask_blend_epi32(mask, best_a, _mm512_set1_epi32(static_cast<int>(h)));
    }
    _mm512_storeu_ps(best_cost + key, best_v);
    if constexpr (UseInt16) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(best_prefix_i16 + key), _mm512_cvtepi32_epi16(best_a));
    } else {
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(best_prefix_i32 + key), best_a);
    }
  }
  for (; key < key_end; ++key) {
    float best_v = inf;
    int32_t best_a = 0;
    for (int64_t h = 0; h < prefix_count; ++h) {
      float v = costs_b[h * suffix_count + key];
      if (v < best_v) {
        best_v = v;
        best_a = static_cast<int32_t>(h);
      }
    }
    best_cost[key] = best_v;
    if constexpr (UseInt16) {
      best_prefix_i16[key] = static_cast<int16_t>(best_a);
    } else {
      best_prefix_i32[key] = best_a;
    }
  }
}

// Sweep B: fused emission + transition add over global states [begin, end),
// which must be a multiple of 16.  The best-cost lookup is a register gather
// keyed by s >> shift; the tail-overlap constraint is blended inline.
template <bool ConstrainedEnd>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
static void emit_add_avx512(
    int64_t vector_size,
    int64_t state_count,
    int64_t overlap_shift,
    const float* __restrict__ best_cost,
    float* __restrict__ next_b,
    const float* __restrict__ codebook_t,
    const float* __restrict__ codebook_norm,
    const float* __restrict__ target,
    float target_norm,
    float weight,
    int64_t begin,
    int64_t end,
    int64_t overlap_value,
    int64_t overlap_mask) {
  emission_avx512(vector_size, state_count, codebook_t, codebook_norm, target, target_norm, weight, next_b, begin,
                  end);
  const __m512i shift_vec = _mm512_set1_epi32(static_cast<int>(overlap_shift));
  const __m512 inf_v = _mm512_set1_ps(std::numeric_limits<float>::infinity());
  const __m512i ov =
      ConstrainedEnd ? _mm512_set1_epi32(static_cast<int>(overlap_value & overlap_mask)) : _mm512_setzero_si512();
  const __m512i omask =
      ConstrainedEnd ? _mm512_set1_epi32(static_cast<int>(overlap_mask)) : _mm512_setzero_si512();
  __m512i s_vec = _mm512_add_epi32(
      _mm512_set1_epi32(static_cast<int>(begin)),
      _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
  const __m512i step_vec = _mm512_set1_epi32(16);
  for (int64_t s = begin; s + 16 <= end; s += 16) {
    __m512 added = _mm512_loadu_ps(next_b + s);
    const __m512i keys = _mm512_srlv_epi32(s_vec, shift_vec);
    const __m512 trans = _mm512_i32gather_ps(keys, best_cost, static_cast<int>(sizeof(float)));
    added = _mm512_add_ps(added, trans);
    if constexpr (ConstrainedEnd) {
      const __m512i low_bits = _mm512_and_si512(s_vec, omask);
      const __mmask16 illegal = _mm512_cmpneq_epi32_mask(low_bits, ov);
      added = _mm512_mask_blend_ps(illegal, added, inf_v);
    }
    _mm512_storeu_ps(next_b + s, added);
    s_vec = _mm512_add_epi32(s_vec, step_vec);
  }
}

// Vectorized final-state argmin: strict less-than with ascending scan order,
// so the lowest state id wins ties exactly like the scalar loops and
// torch.argmin.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
static void final_argmin_avx512(
    const float* __restrict__ final_costs,
    int64_t state_count,
    float* __restrict__ best_val_out,
    int64_t* __restrict__ best_idx_out) {
  constexpr int64_t kVec = 16;
  const float inf = std::numeric_limits<float>::infinity();
  __m512 best_v = _mm512_set1_ps(inf);
  __m512i best_idx = _mm512_setzero_si512();
  __m512i idx_vec = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i step_vec = _mm512_set1_epi32(kVec);
  int64_t s = 0;
  for (; s + kVec <= state_count; s += kVec) {
    __m512 vals = _mm512_loadu_ps(final_costs + s);
    __mmask16 mask = _mm512_cmp_ps_mask(vals, best_v, _CMP_LT_OQ);
    best_v = _mm512_mask_blend_ps(mask, best_v, vals);
    best_idx = _mm512_mask_blend_epi32(mask, best_idx, idx_vec);
    idx_vec = _mm512_add_epi32(idx_vec, step_vec);
  }
  alignas(64) float lane_v[kVec];
  alignas(64) int32_t lane_i[kVec];
  _mm512_store_ps(lane_v, best_v);
  _mm512_store_si512(reinterpret_cast<__m512i*>(lane_i), best_idx);
  // Reduce the per-lane minima lexicographically by (value, index): equal
  // values living in different lanes must resolve to the lowest global state
  // id, exactly like the scalar ascending scan and torch.argmin.
  float best_final = inf;
  int64_t end_state = 0;
  for (int64_t lane = 0; lane < kVec; ++lane) {
    const float v = lane_v[lane];
    const int64_t idx = lane_i[lane];
    if (v < best_final || (v == best_final && idx < end_state)) {
      best_final = v;
      end_state = idx;
    }
  }
  for (; s < state_count; ++s) {
    float c = final_costs[s];
    if (c < best_final) {
      best_final = c;
      end_state = s;
    }
  }
  *best_val_out = best_final;
  *best_idx_out = end_state;
}
#endif  // x86

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> qvq_viterbi_cpu_opt_impl(
    torch::Tensor sequences,
    torch::Tensor codebook,
    int64_t transition_bits,
    c10::optional<torch::Tensor> overlap,
    c10::optional<torch::Tensor> step_weights) {
  TORCH_CHECK(sequences.is_contiguous(), "qvq_viterbi_cpu_opt: sequences must be contiguous");
  TORCH_CHECK(codebook.is_contiguous(), "qvq_viterbi_cpu_opt: codebook must be contiguous");
  TORCH_CHECK(sequences.dim() == 3, "qvq_viterbi_cpu_opt: sequences must be [batch, steps, V]");
  TORCH_CHECK(codebook.dim() == 2, "qvq_viterbi_cpu_opt: codebook must be [state_count, V]");
  TORCH_CHECK(sequences.dtype() == torch::kFloat32, "qvq_viterbi_cpu_opt: sequences must be float32");
  TORCH_CHECK(codebook.dtype() == torch::kFloat32, "qvq_viterbi_cpu_opt: codebook must be float32");
  TORCH_CHECK(sequences.device().is_cpu(), "qvq_viterbi_cpu_opt: sequences must be on CPU");
  TORCH_CHECK(codebook.device().is_cpu(), "qvq_viterbi_cpu_opt: codebook must be on CPU");

  int64_t batch_size = sequences.size(0);
  int64_t step_count = sequences.size(1);
  int64_t vector_size = sequences.size(2);
  int64_t state_count = codebook.size(0);
  int64_t codebook_v = codebook.size(1);
  TORCH_CHECK(vector_size == codebook_v, "qvq_viterbi_cpu_opt: vector size mismatch");
  TORCH_CHECK(vector_size == 2 || vector_size == 4, "qvq_viterbi_cpu_opt: only V=2 or 4 supported");
  TORCH_CHECK(step_count > 0, "qvq_viterbi_cpu_opt: step_count must be positive");
  TORCH_CHECK(state_count > 0 && (state_count & (state_count - 1)) == 0,
              "qvq_viterbi_cpu_opt: state_count must be power of two");

  int l = log2_state_count(state_count);
  TORCH_CHECK(transition_bits >= 1 && transition_bits <= l, "qvq_viterbi_cpu_opt: transition_bits out of range");

  int64_t prefix_count = static_cast<int64_t>(1) << transition_bits;
  int64_t suffix_count = static_cast<int64_t>(1) << (l - transition_bits);
  int64_t overlap_bits = l - transition_bits;
  int64_t overlap_mask = (static_cast<int64_t>(1) << overlap_bits) - 1;

  bool has_overlap = overlap.has_value() && overlap->defined();
  bool has_step_weights = step_weights.has_value() && step_weights->defined();

  const int64_t* overlap_ptr = nullptr;
  if (has_overlap) {
    TORCH_CHECK(overlap->dim() == 1 && overlap->size(0) == batch_size,
                "qvq_viterbi_cpu_opt: overlap shape must be [batch]");
    TORCH_CHECK(overlap->dtype() == torch::kInt64, "qvq_viterbi_cpu_opt: overlap must be int64");
    TORCH_CHECK(overlap->is_contiguous(), "qvq_viterbi_cpu_opt: overlap must be contiguous");
    overlap_ptr = overlap->data_ptr<int64_t>();
  }

  const float* step_weights_ptr = nullptr;
  if (has_step_weights) {
    TORCH_CHECK(step_weights->dim() == 2 && step_weights->size(0) == batch_size &&
                    step_weights->size(1) == step_count,
                "qvq_viterbi_cpu_opt: step_weights shape must be [batch, steps]");
    TORCH_CHECK(step_weights->dtype() == torch::kFloat32, "qvq_viterbi_cpu_opt: step_weights must be float32");
    TORCH_CHECK(step_weights->is_contiguous(), "qvq_viterbi_cpu_opt: step_weights must be contiguous");
    step_weights_ptr = step_weights->data_ptr<float>();
  }

  const float* seq_ptr = sequences.data_ptr<float>();

  torch::Tensor states = torch::empty({batch_size, step_count}, torch::dtype(torch::kInt64).device(sequences.device()));
  torch::Tensor squared_error = torch::empty({batch_size}, torch::dtype(torch::kFloat32).device(sequences.device()));
  int64_t* states_ptr = states.data_ptr<int64_t>();
  float* se_ptr = squared_error.data_ptr<float>();

  torch::Tensor codebook_t = codebook.transpose(0, 1).contiguous();
  const float* codebook_t_ptr = codebook_t.data_ptr<float>();

  std::vector<float> codebook_norm(state_count, 0.0f);
  const float* cb_ptr = codebook.data_ptr<float>();
  for (int64_t s = 0; s < state_count; ++s) {
    float norm = 0.0f;
    for (int64_t v = 0; v < vector_size; ++v) {
      float c = cb_ptr[s * vector_size + v];
      norm += c * c;
    }
    codebook_norm[s] = norm;
  }

  const bool use_int16 = transition_bits <= 15;
  torch::Tensor backpointers;
  void* bp_ptr = nullptr;
  if (step_count > 1) {
    if (use_int16) {
      backpointers = torch::empty({step_count - 1, batch_size, suffix_count},
                                  torch::dtype(torch::kInt16).device(sequences.device()));
      bp_ptr = backpointers.data_ptr<int16_t>();
    } else {
      backpointers = torch::empty({step_count - 1, batch_size, suffix_count},
                                  torch::dtype(torch::kInt32).device(sequences.device()));
      bp_ptr = backpointers.data_ptr<int32_t>();
    }
  }

  // Ping-pong cost buffers plus the per-batch best-cost table.  Unlike the
  // baseline, no emission tensor and no intermediate int32 best-prefix buffer
  // exist; backpointers are written once, directly.
  torch::Tensor costs_tensor =
      torch::empty({batch_size, state_count}, torch::dtype(torch::kFloat32).device(sequences.device()));
  torch::Tensor next_tensor =
      torch::empty({batch_size, state_count}, torch::dtype(torch::kFloat32).device(sequences.device()));
  torch::Tensor best_tensor =
      torch::empty({batch_size, suffix_count}, torch::dtype(torch::kFloat32).device(sequences.device()));
  float* costs_a = costs_tensor.data_ptr<float>();
  float* next_a = next_tensor.data_ptr<float>();
  float* best_a = best_tensor.data_ptr<float>();

  const float inf = std::numeric_limits<float>::infinity();

  std::vector<float> target_norms(static_cast<size_t>(batch_size) * step_count, 0.0f);
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t step = 0; step < step_count; ++step) {
      const float* target = seq_ptr + (b * step_count + step) * vector_size;
      float norm = 0.0f;
      for (int64_t v = 0; v < vector_size; ++v) {
        norm += target[v] * target[v];
      }
      target_norms[b * step_count + step] = norm;
    }
  }

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
  const bool avx512_available = cpu_has_avx512();
#else
  const bool avx512_available = false;
#endif
  // The vectorized sweeps require at least one full 16-state vector.
  const bool use_avx512 = avx512_available && (state_count >= 16);

  for (int64_t step = 0; step < step_count; ++step) {
    const bool constrain_start = has_overlap && step == 0 && overlap_bits != 0;
    const bool constrain_end = has_overlap && step == step_count - 1 && overlap_bits != 0;

    if (step == 0) {
      at::parallel_for(0, batch_size, 1, [&](int64_t b_begin, int64_t b_end) {
        for (int64_t b = b_begin; b < b_end; ++b) {
          const float* target = seq_ptr + b * step_count * vector_size;
          float target_norm = target_norms[b * step_count];
          float w = has_step_weights ? step_weights_ptr[b * step_count] : 1.0f;
          float* costs_b = costs_a + b * state_count;
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
          if (use_avx512) {
            emission_avx512(vector_size, state_count, codebook_t_ptr, codebook_norm.data(), target, target_norm, w,
                            costs_b, 0, state_count);
          } else
#endif
          {
            emission_portable(vector_size, state_count, codebook_t_ptr, codebook_norm.data(), target, target_norm, w,
                              costs_b, 0, state_count);
          }
          if (constrain_start) {
            const int64_t overlap_val = overlap_ptr[b];
            const int64_t block_start = overlap_val * prefix_count;
            const int64_t block_end = block_start + prefix_count;
            for (int64_t s = 0; s < std::min<int64_t>(block_start, state_count); ++s) costs_b[s] = inf;
            for (int64_t s = std::max<int64_t>(block_end, static_cast<int64_t>(0)); s < state_count; ++s)
              costs_b[s] = inf;
          }
        }
      });
      continue;
    }

    // Sweep A: argmin over predecessor rows, backpointers written directly.
    {
      int64_t key_strip = suffix_count;
      if (use_avx512 || suffix_count >= 16) {
        const int64_t target_tasks = std::max<int64_t>(at::get_num_threads() * 4, 1);
        key_strip = (suffix_count + target_tasks - 1) / target_tasks;
        key_strip = ((key_strip + 127) / 128) * 128;
        key_strip = std::min(std::max(key_strip, static_cast<int64_t>(128)), suffix_count);
      }
      const int64_t strips_per_batch = (suffix_count + key_strip - 1) / key_strip;
      const int64_t total_tasks = batch_size * strips_per_batch;
      const int64_t grain = std::max<int64_t>(1, total_tasks / (at::get_num_threads() * 4));
      at::parallel_for(0, total_tasks, grain, [&](int64_t task_begin, int64_t task_end) {
        for (int64_t task = task_begin; task < task_end; ++task) {
          const int64_t b = task / strips_per_batch;
          const int64_t key_begin = (task % strips_per_batch) * key_strip;
          const int64_t key_end = std::min(key_begin + key_strip, suffix_count);
          const float* costs_b = costs_a + b * state_count;
          float* best_b = best_a + b * suffix_count;
          int32_t* bp32 = use_int16 ? nullptr
                                    : static_cast<int32_t*>(bp_ptr) +
                                          ((step - 1) * batch_size + b) * suffix_count;
          int16_t* bp16 = use_int16 ? static_cast<int16_t*>(bp_ptr) + ((step - 1) * batch_size + b) * suffix_count
                                    : nullptr;
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
          if (use_avx512) {
            if (use_int16) {
              argmin_keys_avx512<true>(prefix_count, suffix_count, costs_b, best_b, bp32, bp16, key_begin, key_end);
            } else {
              argmin_keys_avx512<false>(prefix_count, suffix_count, costs_b, best_b, bp32, bp16, key_begin, key_end);
            }
            continue;
          }
#endif
          argmin_keys_portable(prefix_count, suffix_count, costs_b, best_b, bp32, bp16, use_int16, key_begin,
                               key_end);
        }
      });
    }

    // Sweep B: fused emission + transition add (+ inline tail-overlap mask).
    {
      const int64_t chunks_per_batch =
          use_avx512 ? (state_count + 4095) / 4096 : (state_count + 1023) / 1024;
      const int64_t chunk_states = (state_count + chunks_per_batch - 1) / chunks_per_batch;
      const int64_t total_tasks = batch_size * chunks_per_batch;
      const int64_t grain = std::max<int64_t>(1, total_tasks / (at::get_num_threads() * 4));
      at::parallel_for(0, total_tasks, grain, [&](int64_t task_begin, int64_t task_end) {
        for (int64_t task = task_begin; task < task_end; ++task) {
          const int64_t b = task / chunks_per_batch;
          const int64_t begin = (task % chunks_per_batch) * chunk_states;
          const int64_t end = std::min(begin + chunk_states, state_count);
          const float* target = seq_ptr + (b * step_count + step) * vector_size;
          float target_norm = target_norms[b * step_count + step];
          float w = has_step_weights ? step_weights_ptr[b * step_count + step] : 1.0f;
          float* next_b = next_a + b * state_count;
          const float* best_b = best_a + b * suffix_count;
          const int64_t overlap_val = has_overlap ? overlap_ptr[b] : 0;
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
          if (use_avx512) {
            if (constrain_end) {
              emit_add_avx512<true>(vector_size, state_count, transition_bits, best_b, next_b, codebook_t_ptr,
                                    codebook_norm.data(), target, target_norm, w, begin, end, overlap_val,
                                    overlap_mask);
            } else {
              emit_add_avx512<false>(vector_size, state_count, transition_bits, best_b, next_b, codebook_t_ptr,
                                     codebook_norm.data(), target, target_norm, w, begin, end, overlap_val,
                                     overlap_mask);
            }
            continue;
          }
#endif
          emission_portable(vector_size, state_count, codebook_t_ptr, codebook_norm.data(), target, target_norm, w,
                            next_b, begin, end);
          if (constrain_end) {
            emit_add_portable<true>(state_count, best_b, next_b, transition_bits, begin, end, overlap_val,
                                    overlap_mask);
          } else {
            emit_add_portable<false>(state_count, best_b, next_b, transition_bits, begin, end, overlap_val,
                                     overlap_mask);
          }
        }
      });
    }

    std::swap(costs_a, next_a);
  }

  // Final argmin and traceback (identical semantics to the baseline kernel).
  for (int64_t b = 0; b < batch_size; ++b) {
    const float* final_costs = costs_a + b * state_count;
    float best_final = inf;
    int64_t end_state = 0;
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
    if (use_avx512) {
      final_argmin_avx512(final_costs, state_count, &best_final, &end_state);
    } else
#endif
    {
      for (int64_t s = 0; s < state_count; ++s) {
        float c = final_costs[s];
        if (c < best_final) {
          best_final = c;
          end_state = s;
        }
      }
    }
    se_ptr[b] = best_final;
    states_ptr[b * step_count + step_count - 1] = end_state;

    for (int64_t back_step = step_count - 1; back_step > 0; --back_step) {
      int64_t s = states_ptr[b * step_count + back_step];
      int64_t col = s >> transition_bits;
      int32_t prefix = 0;
      if (use_int16) {
        const int16_t* bp = static_cast<const int16_t*>(bp_ptr) + ((back_step - 1) * batch_size + b) * suffix_count;
        prefix = static_cast<int32_t>(bp[col]);
      } else {
        const int32_t* bp = static_cast<const int32_t*>(bp_ptr) + ((back_step - 1) * batch_size + b) * suffix_count;
        prefix = bp[col];
      }
      states_ptr[b * step_count + back_step - 1] = static_cast<int64_t>(prefix) * suffix_count + col;
    }
  }

  return std::make_tuple(states, squared_error);
}

}  // namespace qvq_cpu_opt

std::tuple<torch::Tensor, torch::Tensor> qvq_viterbi_cpu_opt(
    torch::Tensor sequences,
    torch::Tensor codebook,
    int64_t transition_bits,
    c10::optional<torch::Tensor> overlap,
    c10::optional<torch::Tensor> step_weights) {
  return qvq_cpu_opt::qvq_viterbi_cpu_opt_impl(sequences, codebook, transition_bits, overlap, step_weights);
}

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def(
      "viterbi_cpu_opt(Tensor sequences, Tensor codebook, int transition_bits, Tensor? overlap=None, "
      "Tensor? step_weights=None) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CPU, m) {
  m.impl("viterbi_cpu_opt", qvq_cpu_opt::qvq_viterbi_cpu_opt_impl);
}
