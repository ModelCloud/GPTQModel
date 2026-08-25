// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
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

// Parse a boolean-ish environment override. Matches the repository convention
// used by gptqmodel/utils/qvq_cpu.py: a variable that is present but set to an
// empty string, "0", "false" or "off" counts as DISABLED. Testing only for
// presence would make `QVQ_TEST_FORCE_BANKED_G_ONLY=0` silently enable the
// override.
inline bool qvq_env_enabled(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return false;
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return !(value.empty() || value == "0" || value == "false" || value == "off");
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
    bool constrain_final,
    int64_t required_final_state,
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
  const bool initial_valid =
      !constrain_initial || (required_initial_overlap >= 0 && required_initial_overlap < suffix_count);
  const __m512i required_v = initial_valid
      ? _mm512_set1_epi32(static_cast<int>(required_initial_overlap))
      : _mm512_setzero_epi32();

  int64_t x = suffix_begin;
  // The transition_bits >= 4 fast path broadcasts one predecessor over all
  // 16 lanes. An ATen chunk beginning mid-block therefore stays scalar rather
  // than letting a vector cross a predecessor boundary.
  for (; (x & 15) == 0 && x + 16 <= suffix_end; x += 16) {
    __m512 best = inf;
    __m512i best_h = _mm512_setzero_epi32();
    for (int64_t h = 0; h < prefix_count; ++h) {
      const int64_t state = h * suffix_count + x;
      __m512 dot;
      if (vector_size == 2) {
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
      if (weight != 1.0f) value = _mm512_mul_ps(value, w);
      const __m512i states = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(state)), lanes);
      if (previous_g != nullptr) {
        if (transition_bits >= 4) {
          value = _mm512_add_ps(
              value, _mm512_set1_ps(previous_g[state >> transition_bits]));
        } else {
          const __m512i predecessors = _mm512_srli_epi32(states, static_cast<unsigned>(transition_bits));
          value = _mm512_add_ps(value, _mm512_i32gather_ps(predecessors, previous_g, 4));
        }
      } else if (constrain_initial && initial_valid) {
        const __m512i predecessors = _mm512_srli_epi32(states, static_cast<unsigned>(transition_bits));
        const __mmask16 valid = _mm512_cmpeq_epi32_mask(predecessors, required_v);
        value = _mm512_mask_mov_ps(inf, valid, value);
      } else if (constrain_initial) {
        value = inf;
      }
      if (constrain_final) {
        const __mmask16 valid = _mm512_cmpeq_epi32_mask(
            states, _mm512_set1_epi32(static_cast<int>(required_final_state)));
        value = _mm512_mask_mov_ps(inf, valid, value);
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
          (!initial_valid || (state >> transition_bits) != required_initial_overlap)) {
        value = std::numeric_limits<float>::infinity();
      }
      if (constrain_final && state != required_final_state) {
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
    bool constrain_final,
    int64_t required_final_state,
    float* next_g,
    int32_t* best_prefix,
    int64_t suffix_begin,
    int64_t suffix_end) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
  if (cpu_has_avx512()) {
    fused_g_argmin_avx512(
        state_count, vector_size, codebook_t, codebook_norm, target, target_norm,
        weight, previous_g, transition_bits, prefix_count, suffix_count,
        constrain_initial, required_initial_overlap, constrain_final, required_final_state,
        next_g, best_prefix, suffix_begin, suffix_end);
    return;
  }
#endif
  const bool initial_valid =
      !constrain_initial || (required_initial_overlap >= 0 && required_initial_overlap < suffix_count);
  for (int64_t x = suffix_begin; x < suffix_end; ++x) {
    float best = std::numeric_limits<float>::infinity();
    int32_t best_h = 0;
    for (int64_t h = 0; h < prefix_count; ++h) {
      const int64_t state = h * suffix_count + x;
      float value = fused_candidate_scalar(
          state_count, vector_size, codebook_t, codebook_norm, target, target_norm,
          weight, previous_g, state, transition_bits);
      if (previous_g == nullptr && constrain_initial &&
          (!initial_valid || (state >> transition_bits) != required_initial_overlap)) {
        value = std::numeric_limits<float>::infinity();
      }
      if (constrain_final && state != required_final_state) {
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> qvq_viterbi_banked_cpu_legacy(
    torch::Tensor sequences,
    torch::Tensor codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    c10::optional<torch::Tensor> overlap,
    c10::optional<torch::Tensor> step_weights,
    c10::optional<torch::Tensor> entry_states,
    c10::optional<torch::Tensor> exit_states) {
  TORCH_CHECK(sequences.is_contiguous(), "qvq_viterbi_banked_cpu: sequences must be contiguous");
  TORCH_CHECK(codebooks.is_contiguous(), "qvq_viterbi_banked_cpu: codebooks must be contiguous");
  TORCH_CHECK(sequences.dim() == 3, "qvq_viterbi_banked_cpu: sequences must be [batch, steps, V]");
  TORCH_CHECK(codebooks.dim() == 3, "qvq_viterbi_banked_cpu: codebooks must be [bank_count, state_count, V]");
  TORCH_CHECK(sequences.dtype() == torch::kFloat32, "qvq_viterbi_banked_cpu: sequences must be float32");
  TORCH_CHECK(codebooks.dtype() == torch::kFloat32, "qvq_viterbi_banked_cpu: codebooks must be float32");
  TORCH_CHECK(sequences.device().is_cpu(), "qvq_viterbi_banked_cpu: sequences must be on CPU");
  TORCH_CHECK(codebooks.device().is_cpu(), "qvq_viterbi_banked_cpu: codebooks must be on CPU");

  int64_t batch_size = sequences.size(0);
  int64_t step_count = sequences.size(1);
  int64_t vector_size = sequences.size(2);
  int64_t bank_count = codebooks.size(0);
  int64_t state_count = codebooks.size(1);
  int64_t codebook_v = codebooks.size(2);
  TORCH_CHECK(vector_size == codebook_v, "qvq_viterbi_banked_cpu: vector size mismatch");
  TORCH_CHECK(vector_size == 2 || vector_size == 4, "qvq_viterbi_banked_cpu: only V=2 or 4 supported");
  TORCH_CHECK(bank_count >= 1 && bank_count <= 4, "qvq_viterbi_banked_cpu: bank_count must be 1..4");
  TORCH_CHECK(state_count > 0 && (state_count & (state_count - 1)) == 0, "qvq_viterbi_banked_cpu: state_count must be power of two");
  TORCH_CHECK(step_count > 0 && step_count % segment_steps == 0, "qvq_viterbi_banked_cpu: step_count must be divisible by segment_steps");

  int l = log2_state_count(state_count);
  TORCH_CHECK(transition_bits >= 1 && transition_bits <= l, "qvq_viterbi_banked_cpu: transition_bits out of range");

  int64_t prefix_count = static_cast<int64_t>(1) << transition_bits;
  int64_t suffix_count = static_cast<int64_t>(1) << (l - transition_bits);
  int64_t overlap_bits = l - transition_bits;
  int64_t overlap_mask = (static_cast<int64_t>(1) << overlap_bits) - 1;

  bool has_overlap = overlap.has_value() && overlap->defined();
  bool has_step_weights = step_weights.has_value() && step_weights->defined();
  bool has_entry = entry_states.has_value() && entry_states->defined();
  bool has_exit = exit_states.has_value() && exit_states->defined();

  const int64_t* overlap_ptr = nullptr;
  if (has_overlap) {
    TORCH_CHECK(overlap->dim() == 1 && overlap->size(0) == batch_size, "qvq_viterbi_banked_cpu: overlap shape must be [batch]");
    TORCH_CHECK(overlap->dtype() == torch::kInt64, "qvq_viterbi_banked_cpu: overlap must be int64");
    TORCH_CHECK(overlap->is_contiguous(), "qvq_viterbi_banked_cpu: overlap must be contiguous");
    overlap_ptr = overlap->data_ptr<int64_t>();
  }

  const float* step_weights_ptr = nullptr;
  if (has_step_weights) {
    TORCH_CHECK(step_weights->dim() == 2 && step_weights->size(0) == batch_size && step_weights->size(1) == step_count, "qvq_viterbi_banked_cpu: step_weights shape must be [batch, steps]");
    TORCH_CHECK(step_weights->dtype() == torch::kFloat32, "qvq_viterbi_banked_cpu: step_weights must be float32");
    TORCH_CHECK(step_weights->is_contiguous(), "qvq_viterbi_banked_cpu: step_weights must be contiguous");
    step_weights_ptr = step_weights->data_ptr<float>();
  }

  const int64_t* entry_ptr = nullptr;
  if (has_entry) {
    TORCH_CHECK(entry_states->dim() == 1 && entry_states->size(0) == batch_size, "qvq_viterbi_banked_cpu: entry_states shape must be [batch]");
    TORCH_CHECK(entry_states->dtype() == torch::kInt64, "qvq_viterbi_banked_cpu: entry_states must be int64");
    TORCH_CHECK(entry_states->is_contiguous(), "qvq_viterbi_banked_cpu: entry_states must be contiguous");
    entry_ptr = entry_states->data_ptr<int64_t>();
  }

  const int64_t* exit_ptr = nullptr;
  if (has_exit) {
    TORCH_CHECK(exit_states->dim() == 1 && exit_states->size(0) == batch_size, "qvq_viterbi_banked_cpu: exit_states shape must be [batch]");
    TORCH_CHECK(exit_states->dtype() == torch::kInt64, "qvq_viterbi_banked_cpu: exit_states must be int64");
    TORCH_CHECK(exit_states->is_contiguous(), "qvq_viterbi_banked_cpu: exit_states must be contiguous");
    exit_ptr = exit_states->data_ptr<int64_t>();
  }

  TORCH_CHECK(!(has_overlap && (has_entry || has_exit)), "qvq_viterbi_banked_cpu: overlap and entry/exit are mutually exclusive");

  const float* seq_ptr = sequences.data_ptr<float>();

  torch::Tensor states = torch::empty({batch_size, step_count}, torch::dtype(torch::kInt64).device(sequences.device()));
  torch::Tensor squared_error = torch::empty({batch_size}, torch::dtype(torch::kFloat32).device(sequences.device()));
  int64_t segment_count = step_count / segment_steps;
  torch::Tensor segment_bank_ids = torch::empty({batch_size, segment_count}, torch::dtype(torch::kUInt8).device(sequences.device()));

  int64_t* states_ptr = states.data_ptr<int64_t>();
  float* se_ptr = squared_error.data_ptr<float>();
  uint8_t* seg_bank_ptr = segment_bank_ids.data_ptr<uint8_t>();

  // Transpose codebooks to [bank_count, vector_size, state_count] for
  // contiguous per-coordinate AVX-512 loads.
  torch::Tensor codebooks_t = codebooks.permute({0, 2, 1}).contiguous();
  const float* codebooks_t_ptr = codebooks_t.data_ptr<float>();

  // Precompute codebook norms.
  std::vector<float> codebook_norm(static_cast<size_t>(bank_count) * state_count, 0.0f);
  for (int64_t bank = 0; bank < bank_count; ++bank) {
    const float* bank_t = codebooks_t_ptr + bank * (vector_size * state_count);
    for (int64_t s = 0; s < state_count; ++s) {
      float norm = 0.0f;
      for (int64_t v = 0; v < vector_size; ++v) {
        float c = bank_t[v * state_count + s];
        norm += c * c;
      }
      codebook_norm[static_cast<size_t>(bank) * state_count + s] = norm;
    }
  }

  // Backpointers.
  // Boundary backpointers flatten bank * prefix_count + prefix, so size
  // against the full flattened range rather than transition_bits alone.
  const bool use_int16 =
      bank_count * prefix_count - 1 <= std::numeric_limits<int16_t>::max();
  torch::Tensor prefix_backpointers;
  torch::Tensor boundary_backpointers;
  void* prefix_bp_ptr = nullptr;
  void* boundary_bp_ptr = nullptr;
  if (step_count > 1) {
    if (use_int16) {
      prefix_backpointers = torch::empty({step_count - 1, batch_size, bank_count, suffix_count}, torch::dtype(torch::kInt16).device(sequences.device()));
      boundary_backpointers = torch::empty({step_count - 1, batch_size, suffix_count}, torch::dtype(torch::kInt16).device(sequences.device()));
      prefix_bp_ptr = prefix_backpointers.data_ptr<int16_t>();
      boundary_bp_ptr = boundary_backpointers.data_ptr<int16_t>();
    } else {
      prefix_backpointers = torch::empty({step_count - 1, batch_size, bank_count, suffix_count}, torch::dtype(torch::kInt32).device(sequences.device()));
      boundary_backpointers = torch::empty({step_count - 1, batch_size, suffix_count}, torch::dtype(torch::kInt32).device(sequences.device()));
      prefix_bp_ptr = prefix_backpointers.data_ptr<int32_t>();
      boundary_bp_ptr = boundary_backpointers.data_ptr<int32_t>();
    }
  }

  const float inf = std::numeric_limits<float>::infinity();

  // Precompute per-step target norms.
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

  std::vector<float> best_final_costs(static_cast<size_t>(batch_size), inf);
  std::vector<int64_t> best_final_banks(static_cast<size_t>(batch_size), 0);
  std::vector<int64_t> final_end_states(static_cast<size_t>(batch_size), 0);

  auto run_tile = [&](int64_t tile_start, int64_t tile_end, bool parallel_inner) {
    int64_t tile_batch = tile_end - tile_start;
    std::vector<float> costs(static_cast<size_t>(tile_batch) * bank_count * state_count, 0.0f);
    std::vector<float> next_costs(static_cast<size_t>(tile_batch) * bank_count * state_count, 0.0f);
    std::vector<float> emission_buf(static_cast<size_t>(tile_batch) * bank_count * state_count, 0.0f);
    std::vector<float> best_cost(static_cast<size_t>(tile_batch) * bank_count * suffix_count, 0.0f);
    std::vector<int32_t> best_prefix(static_cast<size_t>(tile_batch) * bank_count * suffix_count, 0);
    std::vector<float> best_cost_boundary(static_cast<size_t>(tile_batch) * suffix_count, 0.0f);
    std::vector<int32_t> best_arg_boundary(static_cast<size_t>(tile_batch) * suffix_count, 0);

    auto* costs_a = costs.data();
    auto* next_costs_a = next_costs.data();
    auto* emission_a = emission_buf.data();
    auto* best_cost_a = best_cost.data();
    auto* best_prefix_a = best_prefix.data();
    auto* best_cost_boundary_a = best_cost_boundary.data();
    auto* best_arg_boundary_a = best_arg_boundary.data();

    for (int64_t step = 0; step < step_count; ++step) {
      bool is_boundary = (step > 0) && (step % segment_steps == 0);
      int64_t grain = std::max<int64_t>(256, state_count / (at::get_num_threads() * 4));
      auto run_emission = [&](int64_t start_state, int64_t end_state) {
        for (int64_t tb = 0; tb < tile_batch; ++tb) {
          int64_t b = tile_start + tb;
          const float* target = seq_ptr + (b * step_count + step) * vector_size;
          float target_norm = target_norms[b * step_count + step];
          float w = has_step_weights ? step_weights_ptr[b * step_count + step] : 1.0f;
          for (int64_t bank = 0; bank < bank_count; ++bank) {
            float* emission_b = emission_a + (tb * bank_count + bank) * state_count;
            const float* bank_codebook_t = codebooks_t_ptr + bank * (vector_size * state_count);
            const float* bank_norm = codebook_norm.data() + bank * state_count;
            emit_distance(
                state_count,
                vector_size,
                bank_codebook_t,
                bank_norm,
                target,
                target_norm,
                w,
                emission_b,
                start_state,
                end_state);
          }
        }
      };
      if (parallel_inner) {
        at::parallel_for(0, state_count, grain, run_emission);
      } else {
        run_emission(0, state_count);
      }

      if (step == 0) {
        for (int64_t tb = 0; tb < tile_batch; ++tb) {
          int64_t b = tile_start + tb;
          int64_t required = -1;
          if (has_entry) {
            required = entry_ptr[b] & overlap_mask;
          } else if (has_overlap) {
            required = overlap_ptr[b];
          }
          for (int64_t bank = 0; bank < bank_count; ++bank) {
            float* costs_b = costs_a + (tb * bank_count + bank) * state_count;
            const float* emission_b = emission_a + (tb * bank_count + bank) * state_count;
            std::memcpy(costs_b, emission_b, static_cast<size_t>(state_count) * sizeof(float));
            if (required >= 0) {
              for (int64_t s = 0; s < state_count; ++s) {
                if ((s >> transition_bits) != required) {
                  costs_b[s] = inf;
                }
              }
            }
          }
        }
        continue;
      }

      int64_t suffix_grain = std::max<int64_t>(16, suffix_count / (at::get_num_threads() * 4));
      auto run_column_argmin = [&](int64_t start_suffix, int64_t end_suffix) {
        for (int64_t tb = 0; tb < tile_batch; ++tb) {
          if (is_boundary) {
            column_argmin(
                costs_a + tb * bank_count * state_count,
                bank_count * prefix_count,
                suffix_count,
                best_cost_boundary_a + tb * suffix_count,
                best_arg_boundary_a + tb * suffix_count,
                start_suffix,
                end_suffix);
          } else {
            for (int64_t bank = 0; bank < bank_count; ++bank) {
              column_argmin(
                  costs_a + (tb * bank_count + bank) * state_count,
                  prefix_count,
                  suffix_count,
                  best_cost_a + (tb * bank_count + bank) * suffix_count,
                  best_prefix_a + (tb * bank_count + bank) * suffix_count,
                  start_suffix,
                  end_suffix);
            }
          }
        }
      };
      if (parallel_inner) {
        at::parallel_for(0, suffix_count, suffix_grain, run_column_argmin);
      } else {
        run_column_argmin(0, suffix_count);
      }

      int64_t prefix_grain = std::max<int64_t>(prefix_count, state_count / (at::get_num_threads() * 4));
      prefix_grain = ((prefix_grain + prefix_count - 1) / prefix_count) * prefix_count;
      auto run_broadcast_add = [&](int64_t start_state, int64_t end_state) {
        for (int64_t tb = 0; tb < tile_batch; ++tb) {
          if (is_boundary) {
            const float* best_cost_b = best_cost_boundary_a + tb * suffix_count;
            for (int64_t bank = 0; bank < bank_count; ++bank) {
              broadcast_add(
                  emission_a + (tb * bank_count + bank) * state_count,
                  best_cost_b,
                  prefix_count,
                  suffix_count,
                  next_costs_a + (tb * bank_count + bank) * state_count,
                  start_state,
                  end_state);
            }
          } else {
            for (int64_t bank = 0; bank < bank_count; ++bank) {
              broadcast_add(
                  emission_a + (tb * bank_count + bank) * state_count,
                  best_cost_a + (tb * bank_count + bank) * suffix_count,
                  prefix_count,
                  suffix_count,
                  next_costs_a + (tb * bank_count + bank) * state_count,
                  start_state,
                  end_state);
            }
          }
        }
      };
      if (parallel_inner) {
        at::parallel_for(0, state_count, prefix_grain, run_broadcast_add);
      } else {
        run_broadcast_add(0, state_count);
      }

      if (step == step_count - 1) {
        for (int64_t tb = 0; tb < tile_batch; ++tb) {
          int64_t b = tile_start + tb;
          for (int64_t bank = 0; bank < bank_count; ++bank) {
            float* next_b = next_costs_a + (tb * bank_count + bank) * state_count;
            if (has_exit) {
              int64_t required = exit_ptr[b];
              for (int64_t s = 0; s < state_count; ++s) {
                if (s != required) next_b[s] = inf;
              }
            } else if (has_overlap) {
              int64_t required = overlap_ptr[b];
              for (int64_t s = 0; s < state_count; ++s) {
                if ((s & overlap_mask) != required) next_b[s] = inf;
              }
            }
          }
        }
      }

      for (int64_t tb = 0; tb < tile_batch; ++tb) {
        int64_t b = tile_start + tb;
        int64_t bp_step = step - 1;
        if (is_boundary) {
          if (use_int16) {
            int16_t* bp = static_cast<int16_t*>(boundary_bp_ptr) + (bp_step * batch_size + b) * suffix_count;
            const int32_t* bp_src = best_arg_boundary_a + tb * suffix_count;
            for (int64_t col = 0; col < suffix_count; ++col) {
              bp[col] = static_cast<int16_t>(bp_src[col]);
            }
          } else {
            int32_t* bp = static_cast<int32_t*>(boundary_bp_ptr) + (bp_step * batch_size + b) * suffix_count;
            const int32_t* bp_src = best_arg_boundary_a + tb * suffix_count;
            std::memcpy(bp, bp_src, static_cast<size_t>(suffix_count) * sizeof(int32_t));
          }
        } else {
          for (int64_t bank = 0; bank < bank_count; ++bank) {
            const int32_t* bp_src = best_prefix_a + (tb * bank_count + bank) * suffix_count;
            if (use_int16) {
              int16_t* bp = static_cast<int16_t*>(prefix_bp_ptr) +
                  ((bp_step * batch_size + b) * bank_count + bank) * suffix_count;
              for (int64_t col = 0; col < suffix_count; ++col) {
                bp[col] = static_cast<int16_t>(bp_src[col]);
              }
            } else {
              int32_t* bp = static_cast<int32_t*>(prefix_bp_ptr) +
                  ((bp_step * batch_size + b) * bank_count + bank) * suffix_count;
              std::memcpy(bp, bp_src, static_cast<size_t>(suffix_count) * sizeof(int32_t));
            }
          }
        }
      }

      std::swap(costs_a, next_costs_a);
    }

    for (int64_t tb = 0; tb < tile_batch; ++tb) {
      int64_t b = tile_start + tb;
      float best_final = inf;
      int64_t best_bank = 0;
      int64_t end_state = 0;
      for (int64_t bank = 0; bank < bank_count; ++bank) {
        const float* costs_b = costs_a + (tb * bank_count + bank) * state_count;
        for (int64_t s = 0; s < state_count; ++s) {
          float c = costs_b[s];
          if (c < best_final) {
            best_final = c;
            best_bank = bank;
            end_state = s;
          }
        }
      }
      best_final_costs[b] = best_final;
      best_final_banks[b] = best_bank;
      final_end_states[b] = end_state;
    }
  };

  // Row-parallel once there are enough rows to keep a reasonable number of
  // threads busy; below that, fall back to the outer-serial, inner-parallel
  // tiled path. The clamp to 8 mirrors the G-only recurrence below and is what
  // keeps this from becoming a cliff: with a bare at::get_num_threads() every
  // batch under the thread count took the tiled path, which is ~6x slower per
  // row here, so on a 32-thread box batches 8..31 fell off a cliff and on a
  // 64-thread box batches 8..63 did.
  //
  // Both legs are output-identical: run_tile's reductions (the suffix-column
  // argmin and the final best-(bank,state) scan) run serially within one row
  // in either leg, and parallel_inner only ever partitions output dimensions.
  // tests/test_qvq.py::test_native_banked_viterbi_v2_is_thread_count_invariant
  // pins this by crossing the branch at batch 4 (2 threads -> row-parallel,
  // 16 threads -> tiled) on codebooks dense with exact cost ties.
  if (batch_size >= std::min<int64_t>(8, at::get_num_threads())) {
    at::parallel_for(0, batch_size, 1, [&](int64_t start_batch, int64_t end_batch) {
      for (int64_t b = start_batch; b < end_batch; ++b) {
        run_tile(b, b + 1, false);
      }
    });
  } else {
    constexpr int64_t cache_budget_bytes = 4 * 1024 * 1024;
    int64_t row_footprint_bytes = 3 * bank_count * state_count * static_cast<int64_t>(sizeof(float));
    int64_t tile_size = std::max<int64_t>(1, cache_budget_bytes / std::max<int64_t>(1, row_footprint_bytes));
    for (int64_t tile_start = 0; tile_start < batch_size; tile_start += tile_size) {
      int64_t tile_end = std::min(batch_size, tile_start + tile_size);
      run_tile(tile_start, tile_end, true);
    }
  }

  // Find best final (bank, state) per batch and traceback.
  at::parallel_for(0, batch_size, 0, [&](int64_t start_batch, int64_t end_batch) {
    for (int64_t b = start_batch; b < end_batch; ++b) {
      float best_final = best_final_costs[b];
      int64_t best_bank = best_final_banks[b];
      int64_t end_state = final_end_states[b];
      se_ptr[b] = best_final;
      states_ptr[b * step_count + step_count - 1] = end_state;
      int64_t current_bank = best_bank;

      std::vector<uint8_t> path_banks(static_cast<size_t>(step_count), 0);
      path_banks[step_count - 1] = static_cast<uint8_t>(current_bank);
      for (int64_t step = step_count - 1; step > 0; --step) {
        int64_t s = states_ptr[b * step_count + step];
        int64_t col = s >> transition_bits;
        int64_t prev_state;
        if (step % segment_steps == 0) {
          int32_t arg = 0;
          if (use_int16) {
            const int16_t* bp = static_cast<const int16_t*>(boundary_bp_ptr) + ((step - 1) * batch_size + b) * suffix_count;
            arg = static_cast<int32_t>(bp[col]);
          } else {
            const int32_t* bp = static_cast<const int32_t*>(boundary_bp_ptr) + ((step - 1) * batch_size + b) * suffix_count;
            arg = bp[col];
          }
          int64_t prev_bank = arg / prefix_count;
          int64_t prev_prefix = arg % prefix_count;
          prev_state = prev_prefix * suffix_count + col;
          current_bank = prev_bank;
        } else {
          int32_t prefix = 0;
          if (use_int16) {
            const int16_t* bp = static_cast<const int16_t*>(prefix_bp_ptr) + (((step - 1) * batch_size + b) * bank_count + current_bank) * suffix_count;
            prefix = static_cast<int32_t>(bp[col]);
          } else {
            const int32_t* bp = static_cast<const int32_t*>(prefix_bp_ptr) + (((step - 1) * batch_size + b) * bank_count + current_bank) * suffix_count;
            prefix = bp[col];
          }
          prev_state = static_cast<int64_t>(prefix) * suffix_count + col;
        }
        states_ptr[b * step_count + step - 1] = prev_state;
        path_banks[step - 1] = static_cast<uint8_t>(current_bank);
      }

      for (int64_t seg = 0; seg < segment_count; ++seg) {
        seg_bank_ptr[b * segment_count + seg] = path_banks[seg * segment_steps];
      }
    }
  });

  return std::make_tuple(states, squared_error, segment_bank_ids);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> qvq_viterbi_banked_cpu(
    torch::Tensor sequences,
    torch::Tensor codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    c10::optional<torch::Tensor> overlap,
    c10::optional<torch::Tensor> step_weights,
    c10::optional<torch::Tensor> entry_states,
    c10::optional<torch::Tensor> exit_states) {
  TORCH_CHECK(sequences.is_contiguous(), "qvq_viterbi_banked_cpu: sequences must be contiguous");
  TORCH_CHECK(codebooks.is_contiguous(), "qvq_viterbi_banked_cpu: codebooks must be contiguous");
  TORCH_CHECK(sequences.dim() == 3, "qvq_viterbi_banked_cpu: sequences must be [batch, steps, V]");
  TORCH_CHECK(codebooks.dim() == 3, "qvq_viterbi_banked_cpu: codebooks must be [bank_count, state_count, V]");
  TORCH_CHECK(sequences.dtype() == torch::kFloat32, "qvq_viterbi_banked_cpu: sequences must be float32");
  TORCH_CHECK(codebooks.dtype() == torch::kFloat32, "qvq_viterbi_banked_cpu: codebooks must be float32");
  TORCH_CHECK(sequences.device().is_cpu(), "qvq_viterbi_banked_cpu: sequences must be on CPU");
  TORCH_CHECK(codebooks.device().is_cpu(), "qvq_viterbi_banked_cpu: codebooks must be on CPU");

  int64_t batch_size = sequences.size(0);
  int64_t step_count = sequences.size(1);
  int64_t vector_size = sequences.size(2);
  int64_t bank_count = codebooks.size(0);
  int64_t state_count = codebooks.size(1);
  int64_t codebook_v = codebooks.size(2);
  TORCH_CHECK(vector_size == codebook_v, "qvq_viterbi_banked_cpu: vector size mismatch");
  TORCH_CHECK(vector_size == 2 || vector_size == 4, "qvq_viterbi_banked_cpu: only V=2 or 4 supported");
  TORCH_CHECK(bank_count >= 1 && bank_count <= 4, "qvq_viterbi_banked_cpu: bank_count must be 1..4");
  TORCH_CHECK(state_count > 0 && (state_count & (state_count - 1)) == 0, "qvq_viterbi_banked_cpu: state_count must be power of two");
  TORCH_CHECK(
      step_count > 0 && segment_steps > 0 && step_count % segment_steps == 0,
      "qvq_viterbi_banked_cpu: positive segment_steps must divide step_count");

  // At transition width 16 the suffix frontier has one element and legacy is
  // 2.7-2.9x faster than G-only (median; 2.9-3.7x on minima). Legacy used to
  // have a severe small-batch cliff, but that came from its own
  // `batch_size >= at::get_num_threads()` tiling predicate, which is now
  // clamped the same way the G-only recurrence already clamps its own. The
  // cliff is therefore fixed at its source. The V=2 legacy dispatch remains
  // shape-based; the V=4/t16 default comparison is recorded at the guard below.
  //
  // The two force overrides exist so the recurrences can be A/B measured on
  // one build. Both are scoped to `legacy_t16_shape`, i.e. neither can move a
  // shape this dispatcher deliberately excludes. That scope is load-bearing:
  // an unscoped QVQ_TEST_FORCE_BANKED_LEGACY diverts *every* banked call to
  // legacy, including the t15, bank_count 3-4, V=4 and overlap/entry/exit
  // shapes. Legacy implements all of those, so it does not crash -- it
  // silently returns a different answer (measured: one selected state and all
  // four squared errors change on a bank_count-3 t16 case).
  //
  // Keep the test-only G-only override scoped to the V=2 legacy-t16 predicate.
  // Do not infer recurrence ownership from the public V: MEASURED: default V=4
  // t16 output is unchanged by this scoping, 16 completed tie-rich configs,
  // 0 divergences in selected states, bank IDs, and packed words.
  const bool test_force_legacy = qvq_env_enabled("QVQ_TEST_FORCE_BANKED_LEGACY");
  const bool test_force_g_only = qvq_env_enabled("QVQ_TEST_FORCE_BANKED_G_ONLY");
  TORCH_CHECK(
      !(test_force_legacy && test_force_g_only),
      "qvq_viterbi_banked_cpu: forced legacy and G-only paths are mutually exclusive");
  if (test_force_legacy) {
    TORCH_WARN_ONCE(
        "qvq_viterbi_banked_cpu: QVQ_TEST_FORCE_BANKED_LEGACY is set; transition-width-16 "
        "banked calls are pinned to the legacy recurrence. This is a test/measurement "
        "override and must not be set in production.");
  }
  if (test_force_g_only) {
    TORCH_WARN_ONCE(
        "qvq_viterbi_banked_cpu: QVQ_TEST_FORCE_BANKED_G_ONLY is set; transition-width-16 "
        "banked calls are diverted to the G-only recurrence and may produce different "
        "FP32 near-tie winners. This is a test/measurement override and must not be set "
        "in production.");
  }
  const bool legacy_t16_shape = vector_size == 2 && transition_bits == 16 && bank_count <= 2 &&
      !(overlap.has_value() && overlap->defined()) &&
      !(entry_states.has_value() && entry_states->defined()) &&
      !(exit_states.has_value() && exit_states->defined());
  if (legacy_t16_shape && !test_force_g_only) {
    return qvq_viterbi_banked_cpu_legacy(
        sequences, codebooks, transition_bits, segment_steps, overlap, step_weights,
        entry_states, exit_states);
  }

  int l = log2_state_count(state_count);
  TORCH_CHECK(transition_bits >= 1 && transition_bits <= l, "qvq_viterbi_banked_cpu: transition_bits out of range");

  int64_t prefix_count = static_cast<int64_t>(1) << transition_bits;
  int64_t suffix_count = static_cast<int64_t>(1) << (l - transition_bits);
  int64_t overlap_bits = l - transition_bits;
  int64_t overlap_mask = (static_cast<int64_t>(1) << overlap_bits) - 1;

  bool has_overlap = overlap.has_value() && overlap->defined();
  bool has_step_weights = step_weights.has_value() && step_weights->defined();
  bool has_entry = entry_states.has_value() && entry_states->defined();
  bool has_exit = exit_states.has_value() && exit_states->defined();

  const int64_t* overlap_ptr = nullptr;
  if (has_overlap) {
    TORCH_CHECK(overlap->dim() == 1 && overlap->size(0) == batch_size, "qvq_viterbi_banked_cpu: overlap shape must be [batch]");
    TORCH_CHECK(overlap->dtype() == torch::kInt64, "qvq_viterbi_banked_cpu: overlap must be int64");
    TORCH_CHECK(overlap->is_contiguous(), "qvq_viterbi_banked_cpu: overlap must be contiguous");
    overlap_ptr = overlap->data_ptr<int64_t>();
  }

  const float* step_weights_ptr = nullptr;
  if (has_step_weights) {
    TORCH_CHECK(step_weights->dim() == 2 && step_weights->size(0) == batch_size && step_weights->size(1) == step_count, "qvq_viterbi_banked_cpu: step_weights shape must be [batch, steps]");
    TORCH_CHECK(step_weights->dtype() == torch::kFloat32, "qvq_viterbi_banked_cpu: step_weights must be float32");
    TORCH_CHECK(step_weights->is_contiguous(), "qvq_viterbi_banked_cpu: step_weights must be contiguous");
    step_weights_ptr = step_weights->data_ptr<float>();
  }

  const int64_t* entry_ptr = nullptr;
  if (has_entry) {
    TORCH_CHECK(entry_states->dim() == 1 && entry_states->size(0) == batch_size, "qvq_viterbi_banked_cpu: entry_states shape must be [batch]");
    TORCH_CHECK(entry_states->dtype() == torch::kInt64, "qvq_viterbi_banked_cpu: entry_states must be int64");
    TORCH_CHECK(entry_states->is_contiguous(), "qvq_viterbi_banked_cpu: entry_states must be contiguous");
    entry_ptr = entry_states->data_ptr<int64_t>();
  }

  const int64_t* exit_ptr = nullptr;
  if (has_exit) {
    TORCH_CHECK(exit_states->dim() == 1 && exit_states->size(0) == batch_size, "qvq_viterbi_banked_cpu: exit_states shape must be [batch]");
    TORCH_CHECK(exit_states->dtype() == torch::kInt64, "qvq_viterbi_banked_cpu: exit_states must be int64");
    TORCH_CHECK(exit_states->is_contiguous(), "qvq_viterbi_banked_cpu: exit_states must be contiguous");
    exit_ptr = exit_states->data_ptr<int64_t>();
  }

  TORCH_CHECK(!(has_overlap && (has_entry || has_exit)), "qvq_viterbi_banked_cpu: overlap and entry/exit are mutually exclusive");

  const float* seq_ptr = sequences.data_ptr<float>();

  torch::Tensor states = torch::empty({batch_size, step_count}, torch::dtype(torch::kInt64).device(sequences.device()));
  torch::Tensor squared_error = torch::empty({batch_size}, torch::dtype(torch::kFloat32).device(sequences.device()));
  int64_t segment_count = step_count / segment_steps;
  torch::Tensor segment_bank_ids = torch::empty({batch_size, segment_count}, torch::dtype(torch::kUInt8).device(sequences.device()));

  int64_t* states_ptr = states.data_ptr<int64_t>();
  float* se_ptr = squared_error.data_ptr<float>();
  uint8_t* seg_bank_ptr = segment_bank_ids.data_ptr<uint8_t>();

  // Transpose codebooks to [bank_count, vector_size, state_count] for
  // contiguous per-coordinate AVX-512 loads.
  torch::Tensor codebooks_t = codebooks.permute({0, 2, 1}).contiguous();
  const float* codebooks_t_ptr = codebooks_t.data_ptr<float>();

  // Precompute codebook norms.
  std::vector<float> codebook_norm(static_cast<size_t>(bank_count) * state_count, 0.0f);
  for (int64_t bank = 0; bank < bank_count; ++bank) {
    const float* bank_t = codebooks_t_ptr + bank * (vector_size * state_count);
    for (int64_t s = 0; s < state_count; ++s) {
      float norm = 0.0f;
      for (int64_t v = 0; v < vector_size; ++v) {
        float c = bank_t[v * state_count + s];
        norm += c * c;
      }
      codebook_norm[static_cast<size_t>(bank) * state_count + s] = norm;
    }
  }

  // Backpointers.
  const bool use_int16 = bank_count * prefix_count - 1 <= std::numeric_limits<int16_t>::max();
  torch::Tensor prefix_backpointers;
  torch::Tensor boundary_backpointers;
  void* prefix_bp_ptr = nullptr;
  void* boundary_bp_ptr = nullptr;
  if (step_count > 1) {
    if (use_int16) {
      prefix_backpointers = torch::empty({step_count - 1, batch_size, bank_count, suffix_count}, torch::dtype(torch::kInt16).device(sequences.device()));
      boundary_backpointers = torch::empty({step_count - 1, batch_size, suffix_count}, torch::dtype(torch::kInt16).device(sequences.device()));
      prefix_bp_ptr = prefix_backpointers.data_ptr<int16_t>();
      boundary_bp_ptr = boundary_backpointers.data_ptr<int16_t>();
    } else {
      prefix_backpointers = torch::empty({step_count - 1, batch_size, bank_count, suffix_count}, torch::dtype(torch::kInt32).device(sequences.device()));
      boundary_backpointers = torch::empty({step_count - 1, batch_size, suffix_count}, torch::dtype(torch::kInt32).device(sequences.device()));
      prefix_bp_ptr = prefix_backpointers.data_ptr<int32_t>();
      boundary_bp_ptr = boundary_backpointers.data_ptr<int32_t>();
    }
  }

  const float inf = std::numeric_limits<float>::infinity();

  // Precompute per-step target norms.
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

  std::vector<float> best_final_costs(static_cast<size_t>(batch_size), inf);
  std::vector<int64_t> best_final_banks(static_cast<size_t>(batch_size), 0);
  std::vector<int64_t> final_end_states(static_cast<size_t>(batch_size), 0);

  auto run_tile = [&](int64_t tile_start, int64_t tile_end, bool parallel_inner) {
    int64_t tile_batch = tile_end - tile_start;
    std::vector<float> g_a(static_cast<size_t>(tile_batch) * bank_count * suffix_count, 0.0f);
    std::vector<float> g_b(static_cast<size_t>(tile_batch) * bank_count * suffix_count, 0.0f);
    std::vector<int32_t> best_prefix(static_cast<size_t>(tile_batch) * bank_count * suffix_count, 0);
    std::vector<float> boundary_g(static_cast<size_t>(tile_batch) * suffix_count, 0.0f);

    auto* previous_g = g_a.data();
    auto* next_g = g_b.data();
    auto* best_prefix_a = best_prefix.data();

    for (int64_t step = 0; step < step_count; ++step) {
      bool is_boundary = (step > 0) && (step % segment_steps == 0);
      if (step > 0) {
        for (int64_t tb = 0; tb < tile_batch; ++tb) {
          int64_t b = tile_start + tb;
          int64_t bp_step = step - 1;
          if (is_boundary) {
            float* reduced = boundary_g.data() + tb * suffix_count;
            if (use_int16) {
              int16_t* bp = static_cast<int16_t*>(boundary_bp_ptr) +
                  (bp_step * batch_size + b) * suffix_count;
              for (int64_t x = 0; x < suffix_count; ++x) {
                float best = inf;
                int32_t arg = 0;
                for (int64_t bank = 0; bank < bank_count; ++bank) {
                  float value = previous_g[(tb * bank_count + bank) * suffix_count + x];
                  if (value < best) {
                    best = value;
                    arg = static_cast<int32_t>(bank * prefix_count) +
                        best_prefix_a[(tb * bank_count + bank) * suffix_count + x];
                  }
                }
                reduced[x] = best;
                bp[x] = static_cast<int16_t>(arg);
              }
            } else {
              int32_t* bp = static_cast<int32_t*>(boundary_bp_ptr) +
                  (bp_step * batch_size + b) * suffix_count;
              for (int64_t x = 0; x < suffix_count; ++x) {
                float best = inf;
                int32_t arg = 0;
                for (int64_t bank = 0; bank < bank_count; ++bank) {
                  float value = previous_g[(tb * bank_count + bank) * suffix_count + x];
                  if (value < best) {
                    best = value;
                    arg = static_cast<int32_t>(bank * prefix_count) +
                        best_prefix_a[(tb * bank_count + bank) * suffix_count + x];
                  }
                }
                reduced[x] = best;
                bp[x] = arg;
              }
            }
          } else {
            for (int64_t bank = 0; bank < bank_count; ++bank) {
              const int32_t* source = best_prefix_a + (tb * bank_count + bank) * suffix_count;
              if (use_int16) {
                int16_t* bp = static_cast<int16_t*>(prefix_bp_ptr) +
                    ((bp_step * batch_size + b) * bank_count + bank) * suffix_count;
                for (int64_t x = 0; x < suffix_count; ++x) bp[x] = static_cast<int16_t>(source[x]);
              } else {
                int32_t* bp = static_cast<int32_t*>(prefix_bp_ptr) +
                    ((bp_step * batch_size + b) * bank_count + bank) * suffix_count;
                std::memcpy(bp, source, static_cast<size_t>(suffix_count) * sizeof(int32_t));
              }
            }
          }
        }
      }

      int64_t suffix_grain = std::max<int64_t>(16, suffix_count / (at::get_num_threads() * 4));
      auto run_fused = [&](int64_t start_suffix, int64_t end_suffix) {
        for (int64_t tb = 0; tb < tile_batch; ++tb) {
          int64_t b = tile_start + tb;
          const float* target = seq_ptr + (b * step_count + step) * vector_size;
          float target_norm = target_norms[b * step_count + step];
          float w = has_step_weights ? step_weights_ptr[b * step_count + step] : 1.0f;
          bool constrain_initial = step == 0 && (has_entry || has_overlap);
          int64_t required_initial = has_entry ? (entry_ptr[b] & overlap_mask) :
              (has_overlap ? overlap_ptr[b] : 0);
          bool constrain_final = step_count > 1 && step == step_count - 1 && has_exit;
          int64_t required_final = constrain_final ? exit_ptr[b] : 0;
          if (constrain_final && (required_final < 0 || required_final >= state_count)) {
            constrain_final = false;
            required_final = state_count;
          }
          for (int64_t bank = 0; bank < bank_count; ++bank) {
            const float* prior = nullptr;
            if (step > 0) {
              prior = is_boundary ? boundary_g.data() + tb * suffix_count :
                  previous_g + (tb * bank_count + bank) * suffix_count;
            }
            fused_g_argmin(
                state_count, vector_size,
                codebooks_t_ptr + bank * vector_size * state_count,
                codebook_norm.data() + bank * state_count,
                target, target_norm, w, prior, transition_bits, prefix_count, suffix_count,
                constrain_initial, required_initial, constrain_final, required_final,
                next_g + (tb * bank_count + bank) * suffix_count,
                best_prefix_a + (tb * bank_count + bank) * suffix_count,
                start_suffix, end_suffix);
          }
        }
      };
      if (parallel_inner) {
        at::parallel_for(0, suffix_count, suffix_grain, run_fused);
      } else {
        run_fused(0, suffix_count);
      }
      std::swap(previous_g, next_g);
    }

    for (int64_t tb = 0; tb < tile_batch; ++tb) {
      int64_t b = tile_start + tb;
      float best_final = inf;
      int64_t best_bank = 0;
      int64_t end_state = 0;
      for (int64_t bank = 0; bank < bank_count; ++bank) {
        const float* final_g = previous_g + (tb * bank_count + bank) * suffix_count;
        const int32_t* final_prefix = best_prefix_a + (tb * bank_count + bank) * suffix_count;
        int64_t suffix_begin = 0;
        int64_t suffix_end = suffix_count;
        if (step_count > 1 && has_exit) {
          int64_t required = exit_ptr[b];
          if (required < 0 || required >= state_count) {
            suffix_end = 0;
          } else {
            suffix_begin = required & overlap_mask;
            suffix_end = suffix_begin + 1;
          }
        } else if (step_count > 1 && has_overlap) {
          int64_t required = overlap_ptr[b];
          if (required < 0 || required >= suffix_count) {
            suffix_end = 0;
          } else {
            suffix_begin = required;
            suffix_end = required + 1;
          }
        }
        for (int64_t x = suffix_begin; x < suffix_end; ++x) {
          int64_t candidate_state = static_cast<int64_t>(final_prefix[x]) * suffix_count + x;
          if (step_count > 1 && has_exit && candidate_state != exit_ptr[b]) continue;
          float c = final_g[x];
          if (c < best_final ||
              (c == best_final && bank == best_bank && candidate_state < end_state)) {
            best_final = c;
            best_bank = bank;
            end_state = candidate_state;
          }
        }
      }
      best_final_costs[b] = best_final;
      best_final_banks[b] = best_bank;
      final_end_states[b] = end_state;
    }
  };

  if (batch_size >= std::min<int64_t>(8, at::get_num_threads())) {
    at::parallel_for(0, batch_size, 1, [&](int64_t start_batch, int64_t end_batch) {
      for (int64_t b = start_batch; b < end_batch; ++b) {
        run_tile(b, b + 1, false);
      }
    });
  } else {
    constexpr int64_t cache_budget_bytes = 4 * 1024 * 1024;
    int64_t row_footprint_bytes =
        (2 * bank_count + 1) * suffix_count * static_cast<int64_t>(sizeof(float));
    int64_t tile_size = std::max<int64_t>(1, cache_budget_bytes / std::max<int64_t>(1, row_footprint_bytes));
    for (int64_t tile_start = 0; tile_start < batch_size; tile_start += tile_size) {
      int64_t tile_end = std::min(batch_size, tile_start + tile_size);
      run_tile(tile_start, tile_end, true);
    }
  }

  // Find best final (bank, state) per batch and traceback.
  at::parallel_for(0, batch_size, 0, [&](int64_t start_batch, int64_t end_batch) {
    for (int64_t b = start_batch; b < end_batch; ++b) {
      float best_final = best_final_costs[b];
      int64_t best_bank = best_final_banks[b];
      int64_t end_state = final_end_states[b];
      se_ptr[b] = best_final;
      states_ptr[b * step_count + step_count - 1] = end_state;
      int64_t current_bank = best_bank;

      std::vector<uint8_t> path_banks(static_cast<size_t>(step_count), 0);
      path_banks[step_count - 1] = static_cast<uint8_t>(current_bank);
      for (int64_t step = step_count - 1; step > 0; --step) {
        int64_t s = states_ptr[b * step_count + step];
        int64_t col = s >> transition_bits;
        int64_t prev_state;
        if (step % segment_steps == 0) {
          int32_t arg = 0;
          if (use_int16) {
            const int16_t* bp = static_cast<const int16_t*>(boundary_bp_ptr) + ((step - 1) * batch_size + b) * suffix_count;
            arg = static_cast<int32_t>(bp[col]);
          } else {
            const int32_t* bp = static_cast<const int32_t*>(boundary_bp_ptr) + ((step - 1) * batch_size + b) * suffix_count;
            arg = bp[col];
          }
          int64_t prev_bank = arg / prefix_count;
          int64_t prev_prefix = arg % prefix_count;
          prev_state = prev_prefix * suffix_count + col;
          current_bank = prev_bank;
        } else {
          int32_t prefix = 0;
          if (use_int16) {
            const int16_t* bp = static_cast<const int16_t*>(prefix_bp_ptr) + (((step - 1) * batch_size + b) * bank_count + current_bank) * suffix_count;
            prefix = static_cast<int32_t>(bp[col]);
          } else {
            const int32_t* bp = static_cast<const int32_t*>(prefix_bp_ptr) + (((step - 1) * batch_size + b) * bank_count + current_bank) * suffix_count;
            prefix = bp[col];
          }
          prev_state = static_cast<int64_t>(prefix) * suffix_count + col;
        }
        states_ptr[b * step_count + step - 1] = prev_state;
        path_banks[step - 1] = static_cast<uint8_t>(current_bank);
      }

      for (int64_t seg = 0; seg < segment_count; ++seg) {
        seg_bank_ptr[b * segment_count + seg] = path_banks[seg * segment_steps];
      }
    }
  });

  return std::make_tuple(states, squared_error, segment_bank_ids);
}

}  // namespace qvq_cpu

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def(
      "viterbi_banked_cpu(Tensor sequences, Tensor codebooks, int transition_bits, int segment_steps, "
      "Tensor? overlap=None, Tensor? step_weights=None, Tensor? entry_states=None, Tensor? exit_states=None) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CPU, m) {
  m.impl("viterbi_banked_cpu", qvq_cpu::qvq_viterbi_banked_cpu);
}
