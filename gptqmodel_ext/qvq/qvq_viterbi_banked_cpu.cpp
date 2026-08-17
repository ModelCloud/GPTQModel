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
  const float* cb_ptr = codebooks.data_ptr<float>();

  torch::Tensor states = torch::empty({batch_size, step_count}, torch::dtype(torch::kInt64).device(sequences.device()));
  torch::Tensor squared_error = torch::empty({batch_size}, torch::dtype(torch::kFloat32).device(sequences.device()));
  int64_t segment_count = step_count / segment_steps;
  torch::Tensor segment_bank_ids = torch::empty({batch_size, segment_count}, torch::dtype(torch::kUInt8).device(sequences.device()));

  int64_t* states_ptr = states.data_ptr<int64_t>();
  float* se_ptr = squared_error.data_ptr<float>();
  uint8_t* seg_bank_ptr = segment_bank_ids.data_ptr<uint8_t>();

  // Precompute codebook norms.
  std::vector<float> codebook_norm(static_cast<size_t>(bank_count) * state_count, 0.0f);
  for (int64_t bank = 0; bank < bank_count; ++bank) {
    for (int64_t s = 0; s < state_count; ++s) {
      float norm = 0.0f;
      for (int64_t v = 0; v < vector_size; ++v) {
        float c = cb_ptr[(bank * state_count + s) * vector_size + v];
        norm += c * c;
      }
      codebook_norm[static_cast<size_t>(bank) * state_count + s] = norm;
    }
  }

  // Backpointers: (step_count - 1) x batch x ...
  const bool use_int16 = transition_bits <= 15;
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

  at::parallel_for(0, batch_size, 0, [&](int64_t start_batch, int64_t end_batch) {
    // Per-batch DP buffers.
    std::vector<float> costs(static_cast<size_t>(bank_count) * state_count, 0.0f);
    std::vector<float> next_costs(static_cast<size_t>(bank_count) * state_count, 0.0f);
    std::vector<float> best_cost(static_cast<size_t>(bank_count) * suffix_count, 0.0f);
    // For boundary steps we only need suffix_count entries; reuse the start.

    for (int64_t b = start_batch; b < end_batch; ++b) {
      // Step 0: emission for all banks.
      const float* target = seq_ptr + b * step_count * vector_size;
      float target_norm = 0.0f;
      for (int64_t v = 0; v < vector_size; ++v) {
        target_norm += target[v] * target[v];
      }
      float w0 = has_step_weights ? step_weights_ptr[b * step_count] : 1.0f;

      for (int64_t bank = 0; bank < bank_count; ++bank) {
        float* b_costs = costs.data() + bank * state_count;
        for (int64_t s = 0; s < state_count; ++s) {
          float dist = compute_distance(
              target,
              cb_ptr + ((bank * state_count + s) * vector_size),
              target_norm,
              codebook_norm[static_cast<size_t>(bank) * state_count + s],
              vector_size);
          dist *= w0;
          if (has_entry) {
            int64_t required = entry_ptr[b] & overlap_mask;
            if ((s >> transition_bits) != required) {
              dist = inf;
            }
          } else if (has_overlap) {
            int64_t ov = overlap_ptr[b];
            if ((s >> transition_bits) != ov) {
              dist = inf;
            }
          }
          b_costs[s] = dist;
        }
      }

      // Forward recurrence.
      for (int64_t step = 1; step < step_count; ++step) {
        target = seq_ptr + (b * step_count + step) * vector_size;
        target_norm = 0.0f;
        for (int64_t v = 0; v < vector_size; ++v) {
          target_norm += target[v] * target[v];
        }
        float w = has_step_weights ? step_weights_ptr[b * step_count + step] : 1.0f;
        bool is_boundary = (step % segment_steps == 0);

        if (is_boundary) {
          // Reduce over previous bank and prefix for each suffix.
          for (int64_t col = 0; col < suffix_count; ++col) {
            float best = inf;
            int32_t arg = 0;
            for (int64_t prev_bank = 0; prev_bank < bank_count; ++prev_bank) {
              const float* b_costs = costs.data() + prev_bank * state_count;
              for (int64_t prefix = 0; prefix < prefix_count; ++prefix) {
                int64_t state = prefix * suffix_count + col;
                float c = b_costs[state];
                if (c < best) {
                  best = c;
                  arg = static_cast<int32_t>(prev_bank * prefix_count + prefix);
                }
              }
            }
            best_cost[col] = best;
            if (use_int16) {
              int16_t* bp = static_cast<int16_t*>(boundary_bp_ptr) + ((step - 1) * batch_size + b) * suffix_count;
              bp[col] = static_cast<int16_t>(arg);
            } else {
              int32_t* bp = static_cast<int32_t*>(boundary_bp_ptr) + ((step - 1) * batch_size + b) * suffix_count;
              bp[col] = arg;
            }
          }

          for (int64_t bank = 0; bank < bank_count; ++bank) {
            float* b_next = next_costs.data() + bank * state_count;
            for (int64_t s = 0; s < state_count; ++s) {
              int64_t col = s >> transition_bits;
              float trans = best_cost[col];
              float dist = compute_distance(
                  target,
                  cb_ptr + ((bank * state_count + s) * vector_size),
                  target_norm,
                  codebook_norm[static_cast<size_t>(bank) * state_count + s],
                  vector_size);
              b_next[s] = trans + dist * w;
            }
          }
        } else {
          // Reduce over prefix per (bank, suffix).
          for (int64_t bank = 0; bank < bank_count; ++bank) {
            const float* b_costs = costs.data() + bank * state_count;
            float* b_best = best_cost.data() + bank * suffix_count;
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
              b_best[col] = best;
              if (use_int16) {
                int16_t* bp = static_cast<int16_t*>(prefix_bp_ptr) + ((step - 1) * batch_size + b) * bank_count * suffix_count + bank * suffix_count;
                bp[col] = static_cast<int16_t>(arg);
              } else {
                int32_t* bp = static_cast<int32_t*>(prefix_bp_ptr) + ((step - 1) * batch_size + b) * bank_count * suffix_count + bank * suffix_count;
                bp[col] = arg;
              }
            }
          }

          for (int64_t bank = 0; bank < bank_count; ++bank) {
            const float* b_best = best_cost.data() + bank * suffix_count;
            float* b_next = next_costs.data() + bank * state_count;
            for (int64_t s = 0; s < state_count; ++s) {
              int64_t col = s >> transition_bits;
              float trans = b_best[col];
              float dist = compute_distance(
                  target,
                  cb_ptr + ((bank * state_count + s) * vector_size),
                  target_norm,
                  codebook_norm[static_cast<size_t>(bank) * state_count + s],
                  vector_size);
              b_next[s] = trans + dist * w;
            }
          }
        }

        std::swap(costs, next_costs);
      }

      // Apply end constraints and find best final (bank, state).
      float best_final = inf;
      int64_t best_bank = 0;
      int64_t end_state = 0;
      for (int64_t bank = 0; bank < bank_count; ++bank) {
        const float* b_costs = costs.data() + bank * state_count;
        for (int64_t s = 0; s < state_count; ++s) {
          float c = b_costs[s];
          if (has_exit) {
            if (s != exit_ptr[b]) continue;
          } else if (has_overlap) {
            int64_t ov = overlap_ptr[b];
            if ((s & overlap_mask) != ov) continue;
          }
          if (c < best_final) {
            best_final = c;
            best_bank = bank;
            end_state = s;
          }
        }
      }

      se_ptr[b] = best_final;
      states_ptr[b * step_count + step_count - 1] = end_state;
      int64_t current_bank = best_bank;

      // Traceback to fill states and segment banks.
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
            const int16_t* bp = static_cast<const int16_t*>(prefix_bp_ptr) + ((step - 1) * batch_size + b) * bank_count * suffix_count + current_bank * suffix_count;
            prefix = static_cast<int32_t>(bp[col]);
          } else {
            const int32_t* bp = static_cast<const int32_t*>(prefix_bp_ptr) + ((step - 1) * batch_size + b) * bank_count * suffix_count + current_bank * suffix_count;
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
