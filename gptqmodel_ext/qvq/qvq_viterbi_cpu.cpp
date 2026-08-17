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
  TORCH_CHECK(state_count > 0 && (state_count & (state_count - 1)) == 0, "qvq_viterbi_cpu: state_count must be power of two");

  int l = log2_state_count(state_count);
  TORCH_CHECK(transition_bits >= 1 && transition_bits <= l, "qvq_viterbi_cpu: transition_bits out of range");

  int64_t prefix_count = static_cast<int64_t>(1) << transition_bits;
  int64_t suffix_count = static_cast<int64_t>(1) << (l - transition_bits);
  int64_t overlap_bits = l - transition_bits;
  int64_t overlap_mask = (static_cast<int64_t>(1) << overlap_bits) - 1;

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

  std::vector<float> costs(static_cast<size_t>(batch_size) * state_count, 0.0f);
  std::vector<float> next_costs(static_cast<size_t>(batch_size) * state_count, 0.0f);
  std::vector<float> emission_buf(static_cast<size_t>(batch_size) * state_count, 0.0f);
  std::vector<float> best_cost(static_cast<size_t>(batch_size) * suffix_count, 0.0f);
  std::vector<int32_t> best_prefix(static_cast<size_t>(batch_size) * suffix_count, 0);

  auto* costs_a = costs.data();
  auto* next_costs_a = next_costs.data();
  auto* emission_a = emission_buf.data();
  auto* best_cost_a = best_cost.data();
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

  // Process the DP step by step.  Emission and the per-step reductions are
  // parallelised over the state/suffix dimension so all CPU cores are used even
  // for batch_size == 1.
  for (int64_t step = 0; step < step_count; ++step) {
    // 1. Emission distances for all batches.
    int64_t grain = std::max<int64_t>(256, state_count / (at::get_num_threads() * 4));
    at::parallel_for(0, state_count, grain, [&](int64_t start_state, int64_t end_state) {
      for (int64_t b = 0; b < batch_size; ++b) {
        const float* target = seq_ptr + (b * step_count + step) * vector_size;
        float target_norm = target_norms[b * step_count + step];
        float w = has_step_weights ? step_weights_ptr[b * step_count + step] : 1.0f;
        float* emission_b = emission_a + b * state_count;
        emit_distance(
            state_count,
            vector_size,
            codebook_t_ptr,
            codebook_norm.data(),
            target,
            target_norm,
            w,
            emission_b,
            start_state,
            end_state);
      }
    });

    if (step == 0) {
      // First step: costs are just the (masked) emission.
      if (has_overlap) {
        for (int64_t b = 0; b < batch_size; ++b) {
          float* costs_b = costs_a + b * state_count;
          const float* emission_b = emission_a + b * state_count;
          int64_t overlap_val = overlap_ptr[b];
          // Only the block of states whose high bits equal overlap_val is legal.
          int64_t block_start = overlap_val * prefix_count;
          int64_t block_end = block_start + prefix_count;
          if (block_start < 0) block_start = 0;
          if (block_end > state_count) block_end = state_count;
          std::memcpy(costs_b, emission_b, static_cast<size_t>(state_count) * sizeof(float));
          for (int64_t s = 0; s < block_start; ++s) costs_b[s] = inf;
          for (int64_t s = block_end; s < state_count; ++s) costs_b[s] = inf;
        }
      } else {
        std::memcpy(costs_a, emission_a, costs.size() * sizeof(float));
      }
      continue;
    }

    // 2. Reduce previous costs over prefix dimension per (batch, suffix).
    int64_t suffix_grain = std::max<int64_t>(16, suffix_count / (at::get_num_threads() * 4));
    at::parallel_for(0, suffix_count, suffix_grain, [&](int64_t start_suffix, int64_t end_suffix) {
      for (int64_t b = 0; b < batch_size; ++b) {
        const float* costs_b = costs_a + b * state_count;
        float* best_cost_b = best_cost_a + b * suffix_count;
        int32_t* best_prefix_b = best_prefix_a + b * suffix_count;
        column_argmin(
            costs_b,
            prefix_count,
            suffix_count,
            best_cost_b,
            best_prefix_b,
            start_suffix,
            end_suffix);
      }
    });

    // 3. Combine transition cost with emission for the next costs.
    int64_t prefix_grain = std::max<int64_t>(prefix_count, state_count / (at::get_num_threads() * 4));
    // Round grain up to a multiple of prefix_count to keep broadcast_add block-aligned.
    prefix_grain = ((prefix_grain + prefix_count - 1) / prefix_count) * prefix_count;
    at::parallel_for(0, state_count, prefix_grain, [&](int64_t start_state, int64_t end_state) {
      for (int64_t b = 0; b < batch_size; ++b) {
        const float* emission_b = emission_a + b * state_count;
        const float* best_cost_b = best_cost_a + b * suffix_count;
        float* next_b = next_costs_a + b * state_count;
        broadcast_add(
            emission_b,
            best_cost_b,
            prefix_count,
            suffix_count,
            next_b,
            start_state,
            end_state);
      }
    });

    // 4. Apply end-of-sequence overlap mask if requested.
    if (has_overlap && step == step_count - 1) {
      for (int64_t b = 0; b < batch_size; ++b) {
        float* next_b = next_costs_a + b * state_count;
        int64_t overlap_val = overlap_ptr[b];
        for (int64_t s = 0; s < state_count; ++s) {
          if ((s & overlap_mask) != overlap_val) {
            next_b[s] = inf;
          }
        }
      }
    }

    // 5. Store backpointers.
    for (int64_t b = 0; b < batch_size; ++b) {
      int64_t bp_step = step - 1;
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

    std::swap(costs_a, next_costs_a);
  }

  // Find end state per batch and traceback.
  for (int64_t b = 0; b < batch_size; ++b) {
    float* final_costs = costs_a + b * state_count;
    float best_final = inf;
    int64_t end_state = 0;
    for (int64_t s = 0; s < state_count; ++s) {
      float c = final_costs[s];
      if (c < best_final) {
        best_final = c;
        end_state = s;
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
