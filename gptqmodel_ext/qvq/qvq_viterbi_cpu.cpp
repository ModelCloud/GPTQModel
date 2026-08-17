#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <ATen/Parallel.h>

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

float compute_distance(
    const float* target,
    const float* codebook_row,
    float target_norm,
    float codebook_norm,
    int64_t vector_size) {
  float dot = 0.0f;
  if (vector_size == 2) {
    dot += target[0] * codebook_row[0];
    dot += target[1] * codebook_row[1];
  } else if (vector_size == 4) {
    dot += target[0] * codebook_row[0];
    dot += target[1] * codebook_row[1];
    dot += target[2] * codebook_row[2];
    dot += target[3] * codebook_row[3];
  } else {
    for (int64_t v = 0; v < vector_size; ++v) {
      dot += target[v] * codebook_row[v];
    }
  }
  float dist = target_norm + codebook_norm - 2.0f * dot;
  return dist > 0.0f ? dist : 0.0f;
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
  const float* cb_ptr = codebook.data_ptr<float>();

  torch::Tensor states = torch::empty({batch_size, step_count}, torch::dtype(torch::kInt64).device(sequences.device()));
  torch::Tensor squared_error = torch::empty({batch_size}, torch::dtype(torch::kFloat32).device(sequences.device()));
  int64_t* states_ptr = states.data_ptr<int64_t>();
  float* se_ptr = squared_error.data_ptr<float>();

  // Precompute codebook norms.
  std::vector<float> codebook_norm(state_count, 0.0f);
  for (int64_t s = 0; s < state_count; ++s) {
    float norm = 0.0f;
    for (int64_t v = 0; v < vector_size; ++v) {
      float c = cb_ptr[s * vector_size + v];
      norm += c * c;
    }
    codebook_norm[s] = norm;
  }

  // Backpointers: (step_count-1) x batch x suffix_count.
  // Use int16 when prefix_count fits; otherwise int32.
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

  // Per-batch cost buffers.
  std::vector<float> costs(static_cast<size_t>(batch_size) * state_count, 0.0f);
  std::vector<float> next_costs(static_cast<size_t>(batch_size) * state_count, 0.0f);
  std::vector<float> best_cost(static_cast<size_t>(batch_size) * suffix_count, 0.0f);
  std::vector<int32_t> best_prefix(static_cast<size_t>(batch_size) * suffix_count, 0);

  auto* costs_a = costs.data();
  auto* next_costs_a = next_costs.data();
  auto* best_cost_a = best_cost.data();
  auto* best_prefix_a = best_prefix.data();

  const float inf = std::numeric_limits<float>::infinity();

  // We parallelize over batch slices. Each thread owns a contiguous slice
  // and does the full DP for that slice, avoiding cross-thread writes.
  at::parallel_for(0, batch_size, 0, [&](int64_t start_batch, int64_t end_batch) {
    for (int64_t b = start_batch; b < end_batch; ++b) {
      // Step 0.
      const float* target = seq_ptr + b * step_count * vector_size;
      float target_norm = 0.0f;
      for (int64_t v = 0; v < vector_size; ++v) {
        target_norm += target[v] * target[v];
      }
      float* b_costs = costs_a + b * state_count;
      for (int64_t s = 0; s < state_count; ++s) {
        float dist = compute_distance(target, cb_ptr + s * vector_size, target_norm, codebook_norm[s], vector_size);
        if (has_step_weights) {
          dist *= step_weights_ptr[b * step_count];
        }
        if (has_overlap) {
          int64_t overlap_val = overlap_ptr[b];
          int64_t state_high = s >> transition_bits;
          if (state_high != overlap_val) {
            dist = inf;
          }
        }
        b_costs[s] = dist;
      }

      for (int64_t step = 1; step < step_count; ++step) {
        target = seq_ptr + (b * step_count + step) * vector_size;
        target_norm = 0.0f;
        for (int64_t v = 0; v < vector_size; ++v) {
          target_norm += target[v] * target[v];
        }
        float w = has_step_weights ? step_weights_ptr[b * step_count + step] : 1.0f;

        // Reduce over prefix for each suffix column.
        float* b_best_cost = best_cost_a + b * suffix_count;
        int32_t* b_best_prefix = best_prefix_a + b * suffix_count;
        for (int64_t col = 0; col < suffix_count; ++col) {
          float best = inf;
          int32_t arg = 0;
          for (int64_t prefix = 0; prefix < prefix_count; ++prefix) {
            int64_t state = prefix * suffix_count + col;
            float c = b_costs[state];
            if (c < best) {
              best = c;
              arg = static_cast<int32_t>(prefix);
            }
          }
          b_best_cost[col] = best;
          b_best_prefix[col] = arg;
        }

        // Store backpointer for this transition.
        if (step > 0) {
          int64_t bp_step = step - 1;
          if (use_int16) {
            int16_t* bp = static_cast<int16_t*>(bp_ptr) + (bp_step * batch_size + b) * suffix_count;
            for (int64_t col = 0; col < suffix_count; ++col) {
              bp[col] = static_cast<int16_t>(b_best_prefix[col]);
            }
          } else {
            int32_t* bp = static_cast<int32_t*>(bp_ptr) + (bp_step * batch_size + b) * suffix_count;
            for (int64_t col = 0; col < suffix_count; ++col) {
              bp[col] = b_best_prefix[col];
            }
          }
        }

        float* b_next = next_costs_a + b * state_count;
        for (int64_t s = 0; s < state_count; ++s) {
          int64_t col = s >> transition_bits;
          float trans = b_best_cost[col];
          float dist = compute_distance(target, cb_ptr + s * vector_size, target_norm, codebook_norm[s], vector_size);
          dist = trans + dist * w;
          if (has_overlap && step == step_count - 1) {
            int64_t overlap_val = overlap_ptr[b];
            if ((s & overlap_mask) != overlap_val) {
              dist = inf;
            }
          }
          b_next[s] = dist;
        }
        std::swap(b_costs, b_next);
      }

      // Find end state.
      float* final_costs = b_costs;
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

      // Traceback.
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
  });

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