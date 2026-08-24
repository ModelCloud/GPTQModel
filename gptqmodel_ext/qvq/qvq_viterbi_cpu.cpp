// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <ATen/Parallel.h>

#include "qvq_viterbi_simd.h"

namespace qvq_cpu {

namespace {

inline int log2_state_count(int64_t state_count) {
  int l = 0;
  int64_t s = state_count;
  while (s > 1) {
    s >>= 1;
    ++l;
  }
  return l;
}

inline float fused_candidate_scalar(
    int64_t state_count,
    int64_t vector_size,
    const float* codebook_t,
    const float* codebook_norm,
    const float* target,
    float target_norm,
    float weight,
    const float* previous_g,
    int64_t state,
    int64_t transition_bits) {
  float dot = 0.0f;
  for (int64_t v = 0; v < vector_size; ++v) {
    dot += target[v] * codebook_t[v * state_count + state];
  }
  float distance = target_norm + codebook_norm[state] - 2.0f * dot;
  if (distance < 0.0f) distance = 0.0f;
  distance *= weight;
  return distance + (previous_g == nullptr ? 0.0f : previous_g[state >> transition_bits]);
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
static void fused_g_argmin_avx512(
    int64_t state_count,
    int64_t vector_size,
    const float* __restrict__ codebook_t,
    const float* __restrict__ codebook_norm,
    const float* __restrict__ target,
    float target_norm,
    float weight,
    const float* __restrict__ previous_g,
    int64_t transition_bits,
    int64_t prefix_count,
    int64_t suffix_count,
    bool constrain_initial,
    int64_t required_initial_overlap,
    float* __restrict__ next_g,
    int32_t* __restrict__ best_prefix,
    int64_t suffix_begin,
    int64_t suffix_end) {
  const float* c0 = codebook_t;
  const float* c1 = codebook_t + state_count;
  const float* c2 = vector_size == 4 ? codebook_t + 2 * state_count : nullptr;
  const float* c3 = vector_size == 4 ? codebook_t + 3 * state_count : nullptr;
  const __m512 t0 = _mm512_set1_ps(target[0]);
  const __m512 t1 = _mm512_set1_ps(target[1]);
  const __m512 t2 = vector_size == 4 ? _mm512_set1_ps(target[2]) : _mm512_setzero_ps();
  const __m512 t3 = vector_size == 4 ? _mm512_set1_ps(target[3]) : _mm512_setzero_ps();
  const __m512 tn = _mm512_set1_ps(target_norm);
  const __m512 w = _mm512_set1_ps(weight);
  const __m512 two = _mm512_set1_ps(2.0f);
  const __m512 zero = _mm512_setzero_ps();
  const __m512 inf = _mm512_set1_ps(std::numeric_limits<float>::infinity());
  const __m512i lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const bool initial_overlap_valid =
      !constrain_initial ||
      (required_initial_overlap >= 0 && required_initial_overlap < suffix_count);
  const __m512i required_initial_overlap_v = initial_overlap_valid
      ? _mm512_set1_epi32(static_cast<int>(required_initial_overlap))
      : _mm512_setzero_epi32();

  int64_t x = suffix_begin;
  for (; x + 16 <= suffix_end; x += 16) {
    __m512 best = inf;
    __m512i best_h = _mm512_setzero_epi32();
    for (int64_t h = 0; h < prefix_count; ++h) {
      const int64_t state = h * suffix_count + x;
      __m512 dot;
      if (vector_size == 2) {
        // Accumulate from zero in coordinate order so this matches
        // `fused_candidate_scalar` -- which serves both the remainder columns of
        // this function and the entire non-AVX fallback -- bit for bit. A
        // leading `_mm512_mul_ps` would pre-round the c1 product and make the
        // vector body disagree with the scalar tail. V=4 below already does this.
        dot = _mm512_setzero_ps();
        dot = _mm512_fmadd_ps(_mm512_loadu_ps(c0 + state), t0, dot);
        dot = _mm512_fmadd_ps(_mm512_loadu_ps(c1 + state), t1, dot);
      } else {
        dot = _mm512_setzero_ps();
        dot = _mm512_fmadd_ps(_mm512_loadu_ps(c0 + state), t0, dot);
        dot = _mm512_fmadd_ps(_mm512_loadu_ps(c1 + state), t1, dot);
        dot = _mm512_fmadd_ps(_mm512_loadu_ps(c2 + state), t2, dot);
        dot = _mm512_fmadd_ps(_mm512_loadu_ps(c3 + state), t3, dot);
      }
      __m512 value = _mm512_fnmadd_ps(
          dot, two, _mm512_add_ps(tn, _mm512_loadu_ps(codebook_norm + state)));
      value = _mm512_max_ps(value, zero);
      if (weight != 1.0f) {
        value = _mm512_mul_ps(value, w);
      }
      const __m512i states = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(state)), lanes);
      if (previous_g != nullptr) {
        const __m512i predecessors = _mm512_srli_epi32(states, static_cast<unsigned>(transition_bits));
        value = _mm512_add_ps(value, _mm512_i32gather_ps(predecessors, previous_g, 4));
      } else if (constrain_initial && initial_overlap_valid) {
        const __m512i predecessors = _mm512_srli_epi32(states, static_cast<unsigned>(transition_bits));
        const __mmask16 valid = _mm512_cmpeq_epi32_mask(predecessors, required_initial_overlap_v);
        value = _mm512_mask_mov_ps(inf, valid, value);
      } else if (constrain_initial) {
        value = inf;
      }
      const __mmask16 lower = _mm512_cmp_ps_mask(value, best, _CMP_LT_OQ);
      best = _mm512_mask_mov_ps(best, lower, value);
      best_h = _mm512_mask_mov_epi32(best_h, lower, _mm512_set1_epi32(static_cast<int>(h)));
    }
    _mm512_storeu_ps(next_g + x, best);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(best_prefix + x), best_h);
  }
  for (; x < suffix_end; ++x) {
    float best = std::numeric_limits<float>::infinity();
    int32_t best_h = 0;
    for (int64_t h = 0; h < prefix_count; ++h) {
      const int64_t state = h * suffix_count + x;
      float value = fused_candidate_scalar(
          state_count, vector_size, codebook_t, codebook_norm, target, target_norm,
          weight, previous_g, state, transition_bits);
      if (previous_g == nullptr && constrain_initial &&
          (!initial_overlap_valid || (state >> transition_bits) != required_initial_overlap)) {
        value = std::numeric_limits<float>::infinity();
      }
      if (value < best) {
        best = value;
        best_h = static_cast<int32_t>(h);
      }
    }
    next_g[x] = best;
    best_prefix[x] = best_h;
  }
}
#endif

static void fused_g_argmin(
    int64_t state_count,
    int64_t vector_size,
    const float* codebook_t,
    const float* codebook_norm,
    const float* target,
    float target_norm,
    float weight,
    const float* previous_g,
    int64_t transition_bits,
    int64_t prefix_count,
    int64_t suffix_count,
    bool constrain_initial,
    int64_t required_initial_overlap,
    float* next_g,
    int32_t* best_prefix,
    int64_t suffix_begin,
    int64_t suffix_end) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
  if (cpu_has_avx512()) {
    fused_g_argmin_avx512(
        state_count, vector_size, codebook_t, codebook_norm, target, target_norm,
        weight, previous_g, transition_bits, prefix_count, suffix_count,
        constrain_initial, required_initial_overlap, next_g, best_prefix, suffix_begin, suffix_end);
    return;
  }
#endif
  const bool initial_overlap_valid =
      !constrain_initial ||
      (required_initial_overlap >= 0 && required_initial_overlap < suffix_count);
  for (int64_t x = suffix_begin; x < suffix_end; ++x) {
    float best = std::numeric_limits<float>::infinity();
    int32_t best_h = 0;
    for (int64_t h = 0; h < prefix_count; ++h) {
      const int64_t state = h * suffix_count + x;
      float value = fused_candidate_scalar(
          state_count, vector_size, codebook_t, codebook_norm, target, target_norm,
          weight, previous_g, state, transition_bits);
      if (previous_g == nullptr && constrain_initial &&
          (!initial_overlap_valid || (state >> transition_bits) != required_initial_overlap)) {
        value = std::numeric_limits<float>::infinity();
      }
      if (value < best) {
        best = value;
        best_h = static_cast<int32_t>(h);
      }
    }
    next_g[x] = best;
    best_prefix[x] = best_h;
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> qvq_viterbi_cpu(
    torch::Tensor sequences,
    torch::Tensor codebook,
    int64_t transition_bits,
    c10::optional<torch::Tensor> overlap,
    c10::optional<torch::Tensor> step_weights) {
  TORCH_CHECK(sequences.is_contiguous(), "qvq_viterbi_cpu: sequences must be contiguous");
  TORCH_CHECK(codebook.is_contiguous(), "qvq_viterbi_cpu: codebook must be contiguous");
  TORCH_CHECK(sequences.dim() == 3, "qvq_viterbi_cpu: sequences must be [batch, steps, V]");
  TORCH_CHECK(codebook.dim() == 2, "qvq_viterbi_cpu: codebook must be [state_count, V]");
  TORCH_CHECK(sequences.dtype() == torch::kFloat32, "qvq_viterbi_cpu: sequences must be float32");
  TORCH_CHECK(codebook.dtype() == torch::kFloat32, "qvq_viterbi_cpu: codebook must be float32");
  TORCH_CHECK(sequences.device().is_cpu(), "qvq_viterbi_cpu: sequences must be on CPU");
  TORCH_CHECK(codebook.device().is_cpu(), "qvq_viterbi_cpu: codebook must be on CPU");

  int64_t batch_size = sequences.size(0);
  int64_t step_count = sequences.size(1);
  int64_t vector_size = sequences.size(2);
  int64_t state_count = codebook.size(0);
  int64_t codebook_v = codebook.size(1);
  TORCH_CHECK(vector_size == codebook_v, "qvq_viterbi_cpu: vector size mismatch");
  TORCH_CHECK(vector_size == 2 || vector_size == 4, "qvq_viterbi_cpu: only V=2 or 4 supported");
  TORCH_CHECK(step_count > 0, "qvq_viterbi_cpu: step_count must be positive");
  TORCH_CHECK(state_count > 0 && (state_count & (state_count - 1)) == 0, "qvq_viterbi_cpu: state_count must be power of two");

  int l = log2_state_count(state_count);
  TORCH_CHECK(transition_bits >= 1 && transition_bits <= l, "qvq_viterbi_cpu: transition_bits out of range");

  int64_t prefix_count = static_cast<int64_t>(1) << transition_bits;
  int64_t suffix_count = static_cast<int64_t>(1) << (l - transition_bits);
  bool has_overlap = overlap.has_value() && overlap->defined();
  bool has_step_weights = step_weights.has_value() && step_weights->defined();

  const int64_t* overlap_ptr = nullptr;
  if (has_overlap) {
    TORCH_CHECK(overlap->dim() == 1 && overlap->size(0) == batch_size, "qvq_viterbi_cpu: overlap shape must be [batch]");
    TORCH_CHECK(overlap->dtype() == torch::kInt64, "qvq_viterbi_cpu: overlap must be int64");
    TORCH_CHECK(overlap->is_contiguous(), "qvq_viterbi_cpu: overlap must be contiguous");
    overlap_ptr = overlap->data_ptr<int64_t>();
  }

  const float* step_weights_ptr = nullptr;
  if (has_step_weights) {
    TORCH_CHECK(step_weights->dim() == 2 && step_weights->size(0) == batch_size && step_weights->size(1) == step_count, "qvq_viterbi_cpu: step_weights shape must be [batch, steps]");
    TORCH_CHECK(step_weights->dtype() == torch::kFloat32, "qvq_viterbi_cpu: step_weights must be float32");
    TORCH_CHECK(step_weights->is_contiguous(), "qvq_viterbi_cpu: step_weights must be contiguous");
    step_weights_ptr = step_weights->data_ptr<float>();
  }

  const float* seq_ptr = sequences.data_ptr<float>();

  torch::Tensor states = torch::empty({batch_size, step_count}, torch::dtype(torch::kInt64).device(sequences.device()));
  torch::Tensor squared_error = torch::empty({batch_size}, torch::dtype(torch::kFloat32).device(sequences.device()));
  int64_t* states_ptr = states.data_ptr<int64_t>();
  float* se_ptr = squared_error.data_ptr<float>();

  // Transpose codebook to [vector_size, state_count] so each coordinate is
  // contiguous across states and can be loaded with contiguous AVX-512 vectors.
  torch::Tensor codebook_t = codebook.transpose(0, 1).contiguous();
  const float* codebook_t_ptr = codebook_t.data_ptr<float>();

  // Precompute codebook norms.
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

  // Backpointers: (step_count-1) x batch x suffix_count.
  const bool use_int16 = transition_bits <= 15;
  torch::Tensor backpointers;
  void* bp_ptr = nullptr;
  if (step_count > 1) {
    if (use_int16) {
      backpointers = torch::empty({step_count - 1, batch_size, suffix_count}, torch::dtype(torch::kInt16).device(sequences.device()));
      bp_ptr = backpointers.data_ptr<int16_t>();
    } else {
      backpointers = torch::empty({step_count - 1, batch_size, suffix_count}, torch::dtype(torch::kInt32).device(sequences.device()));
      bp_ptr = backpointers.data_ptr<int32_t>();
    }
  }

  std::vector<float> g_a(static_cast<size_t>(batch_size) * suffix_count, 0.0f);
  std::vector<float> g_b(static_cast<size_t>(batch_size) * suffix_count, 0.0f);
  std::vector<int32_t> best_prefix(static_cast<size_t>(batch_size) * suffix_count, 0);

  auto* previous_g = g_a.data();
  auto* next_g = g_b.data();
  auto* best_prefix_a = best_prefix.data();

  const float inf = std::numeric_limits<float>::infinity();

  // Precompute per-step targets and norms for quick access.
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

  // G-only recurrence: fuse emission, predecessor-G addition, suffix argmin,
  // and compressed backpointer production in one suffix-parallel pass.
  for (int64_t step = 0; step < step_count; ++step) {
    int64_t suffix_grain = std::max<int64_t>(16, suffix_count / (at::get_num_threads() * 4));
    at::parallel_for(0, suffix_count, suffix_grain, [&](int64_t start_suffix, int64_t end_suffix) {
      for (int64_t b = 0; b < batch_size; ++b) {
        const float* target = seq_ptr + (b * step_count + step) * vector_size;
        float target_norm = target_norms[b * step_count + step];
        float w = has_step_weights ? step_weights_ptr[b * step_count + step] : 1.0f;
        const float* previous_g_b = step == 0 ? nullptr : previous_g + b * suffix_count;
        const bool constrain_initial = step == 0 && has_overlap;
        const int64_t required_initial_overlap = constrain_initial ? overlap_ptr[b] : 0;
        fused_g_argmin(
            state_count, vector_size, codebook_t_ptr, codebook_norm.data(), target,
            target_norm, w, previous_g_b, transition_bits, prefix_count, suffix_count,
            constrain_initial, required_initial_overlap, next_g + b * suffix_count,
            best_prefix_a + b * suffix_count, start_suffix, end_suffix);
      }
    });

    if (step < step_count - 1) {
      for (int64_t b = 0; b < batch_size; ++b) {
        int64_t bp_step = step;
        if (use_int16) {
          int16_t* bp = static_cast<int16_t*>(bp_ptr) + (bp_step * batch_size + b) * suffix_count;
          const int32_t* bp_src = best_prefix_a + b * suffix_count;
          for (int64_t col = 0; col < suffix_count; ++col) {
            bp[col] = static_cast<int16_t>(bp_src[col]);
          }
        } else {
          int32_t* bp = static_cast<int32_t*>(bp_ptr) + (bp_step * batch_size + b) * suffix_count;
          const int32_t* bp_src = best_prefix_a + b * suffix_count;
          std::memcpy(bp, bp_src, static_cast<size_t>(suffix_count) * sizeof(int32_t));
        }
      }
    }
    std::swap(previous_g, next_g);
  }

  // Find end state per batch and traceback.
  for (int64_t b = 0; b < batch_size; ++b) {
    const float* final_g = previous_g + b * suffix_count;
    const int32_t* final_prefix = best_prefix_a + b * suffix_count;
    float best_final = inf;
    int64_t end_state = 0;
    const bool constrain_final = has_overlap && step_count > 1;
    int64_t suffix_begin = constrain_final ? overlap_ptr[b] : 0;
    int64_t suffix_end = suffix_count;
    if (suffix_begin < 0 || suffix_begin >= suffix_count) {
      // Match the old full-frontier mask: an invalid overlap leaves every
      // final cost at infinity and therefore retains end state zero.
      suffix_begin = 0;
      suffix_end = 0;
    } else if (constrain_final) {
      suffix_end = suffix_begin + 1;
    }
    for (int64_t x = suffix_begin; x < suffix_end; ++x) {
      float c = final_g[x];
      int64_t candidate_state = static_cast<int64_t>(final_prefix[x]) * suffix_count + x;
      if (c < best_final || (c == best_final && candidate_state < end_state)) {
        best_final = c;
        end_state = candidate_state;
      }
    }
    se_ptr[b] = best_final;
    states_ptr[b * step_count + step_count - 1] = end_state;

    for (int64_t step = step_count - 1; step > 0; --step) {
      int64_t s = states_ptr[b * step_count + step];
      int64_t col = s >> transition_bits;
      int32_t prefix = 0;
      if (use_int16) {
        const int16_t* bp = static_cast<const int16_t*>(bp_ptr) + ((step - 1) * batch_size + b) * suffix_count;
        prefix = static_cast<int32_t>(bp[col]);
      } else {
        const int32_t* bp = static_cast<const int32_t*>(bp_ptr) + ((step - 1) * batch_size + b) * suffix_count;
        prefix = bp[col];
      }
      states_ptr[b * step_count + step - 1] = static_cast<int64_t>(prefix) * suffix_count + col;
    }
  }

  return std::make_tuple(states, squared_error);
}

}  // namespace qvq_cpu

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def(
      "viterbi_cpu(Tensor sequences, Tensor codebook, int transition_bits, Tensor? overlap=None, "
      "Tensor? step_weights=None) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CPU, m) {
  m.impl("viterbi_cpu", qvq_cpu::qvq_viterbi_cpu);
}
