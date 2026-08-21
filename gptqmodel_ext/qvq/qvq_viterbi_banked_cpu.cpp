// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

  if (batch_size >= at::get_num_threads()) {
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

}  // namespace qvq_cpu

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def(
      "viterbi_banked_cpu(Tensor sequences, Tensor codebooks, int transition_bits, int segment_steps, "
      "Tensor? overlap=None, Tensor? step_weights=None, Tensor? entry_states=None, Tensor? exit_states=None) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CPU, m) {
  m.impl("viterbi_banked_cpu", qvq_cpu::qvq_viterbi_banked_cpu);
}
