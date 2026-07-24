// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <tuple>

namespace {

constexpr int kMaxDecisions = 128;
constexpr int kMaxSplitDepth = 20;
constexpr int kThreadsPerBlock = 128;

__device__ __forceinline__ bool state_bit(uint64_t low, uint64_t high, int index) {
  return index < 64 ? ((low >> index) & uint64_t{1}) != 0
                    : ((high >> (index - 64)) & uint64_t{1}) != 0;
}

__device__ __forceinline__ void set_state_bit(
    uint64_t &low, uint64_t &high, int index, bool value) {
  uint64_t &word = index < 64 ? low : high;
  const int shift = index < 64 ? index : index - 64;
  const uint64_t mask = uint64_t{1} << shift;
  word = value ? word | mask : word & ~mask;
}

__device__ __forceinline__ double atomic_min_double(double *address, double value) {
  auto *integer_address = reinterpret_cast<unsigned long long *>(address);
  unsigned long long observed = *integer_address;
  while (value < __longlong_as_double(observed)) {
    const unsigned long long expected = observed;
    observed = atomicCAS(integer_address, expected, __double_as_longlong(value));
    if (observed == expected) {
      break;
    }
  }
  return __longlong_as_double(observed);
}

__device__ double evaluate_state(
    double constant,
    const double *__restrict__ linear,
    const double *__restrict__ interaction,
    int size,
    uint64_t low,
    uint64_t high) {
  double energy = constant;
  for (int i = 0; i < size; ++i) {
    if (!state_bit(low, high, i)) {
      continue;
    }
    energy += linear[i];
    for (int j = i + 1; j < size; ++j) {
      if (state_bit(low, high, j)) {
        energy += interaction[i * size + j];
      }
    }
  }
  return energy;
}

__device__ double admissible_lower_bound(
    double fixed_energy,
    const double *effective,
    const double *negative_incident,
    const double *positive_incident,
    double negative_edge_sum,
    double positive_edge_sum,
    int depth,
    int size) {
  // Bound 1 minimizes every unary and negative pair independently.
  double independent_edges = fixed_energy + negative_edge_sum;
  // Bound 2 splits q*x_i*x_j for q<0 equally across its two endpoints.
  double negative_split = fixed_energy;
  // Bound 3 adds the positive-edge envelope q*(x_i+x_j-1).
  double signed_split = fixed_energy - positive_edge_sum;
  // Bound 4 combines independent negative edges with the positive envelope.
  double positive_envelope =
      fixed_energy + negative_edge_sum - positive_edge_sum;

  for (int i = depth; i < size; ++i) {
    independent_edges += fmin(0.0, effective[i]);
    negative_split +=
        fmin(0.0, effective[i] + 0.5 * negative_incident[i]);
    signed_split += fmin(
        0.0,
        effective[i] + positive_incident[i] +
            0.5 * negative_incident[i]);
    positive_envelope +=
        fmin(0.0, effective[i] + positive_incident[i]);
  }
  return fmax(
      fmax(independent_edges, negative_split),
      fmax(signed_split, positive_envelope));
}

__global__ void adjacent_branch_bound_kernel(
    const double *__restrict__ constant,
    const double *__restrict__ linear,
    const double *__restrict__ interaction,
    int size,
    int split_depth,
    int64_t worker_count,
    int64_t max_nodes_per_worker,
    double certificate_tolerance,
    double *__restrict__ global_best,
    int64_t *__restrict__ candidate_states,
    double *__restrict__ candidate_costs,
    double *__restrict__ root_lower_bounds,
    int64_t *__restrict__ nodes_visited,
    uint8_t *__restrict__ completed) {
  const int64_t worker =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (worker >= worker_count) {
    return;
  }

  // These arrays intentionally live in thread-local memory. Each CUDA thread
  // owns one disjoint prefix and performs deterministic depth-first search.
  double effective[kMaxDecisions];
  double negative_incident[kMaxDecisions];
  double positive_incident[kMaxDecisions];
  double fixed_energy[kMaxDecisions + 1];
  double negative_edge_sum[kMaxDecisions + 1];
  double positive_edge_sum[kMaxDecisions + 1];
  uint8_t next_branch[kMaxDecisions];
  uint8_t decision[kMaxDecisions];

  double initial_negative_edges = 0.0;
  double initial_positive_edges = 0.0;
  for (int i = 0; i < size; ++i) {
    effective[i] = linear[i];
    double negative = 0.0;
    double positive = 0.0;
    for (int j = 0; j < size; ++j) {
      if (i == j) {
        continue;
      }
      const double coupling = interaction[i * size + j];
      negative += fmin(0.0, coupling);
      positive += fmax(0.0, coupling);
    }
    negative_incident[i] = negative;
    positive_incident[i] = positive;
    initial_negative_edges += negative;
    initial_positive_edges += positive;
  }
  initial_negative_edges *= 0.5;
  initial_positive_edges *= 0.5;

  fixed_energy[0] = constant[0];
  negative_edge_sum[0] = initial_negative_edges;
  positive_edge_sum[0] = initial_positive_edges;
  uint64_t state_low = 0;
  uint64_t state_high = 0;

  // Prefix tasks cover every assignment of the most influential variables.
  for (int depth = 0; depth < split_depth; ++depth) {
    const bool value = ((static_cast<uint64_t>(worker) >> depth) & 1u) != 0;
    decision[depth] = static_cast<uint8_t>(value);
    set_state_bit(state_low, state_high, depth, value);

    const double removed_negative = negative_incident[depth];
    const double removed_positive = positive_incident[depth];
    fixed_energy[depth + 1] =
        fixed_energy[depth] + (value ? effective[depth] : 0.0);
    negative_edge_sum[depth + 1] =
        negative_edge_sum[depth] - removed_negative;
    positive_edge_sum[depth + 1] =
        positive_edge_sum[depth] - removed_positive;
    for (int i = depth + 1; i < size; ++i) {
      const double coupling = interaction[depth * size + i];
      if (value) {
        effective[i] += coupling;
      }
      if (coupling < 0.0) {
        negative_incident[i] -= coupling;
      } else {
        positive_incident[i] -= coupling;
      }
    }
  }

  const double root_bound = admissible_lower_bound(
      fixed_energy[split_depth],
      effective,
      negative_incident,
      positive_incident,
      negative_edge_sum[split_depth],
      positive_edge_sum[split_depth],
      split_depth,
      size);
  root_lower_bounds[worker] = root_bound;

  double local_best = constant[0];
  uint64_t local_best_low = 0;
  uint64_t local_best_high = 0;
  int64_t nodes = 0;
  bool complete = true;
  bool entering = true;
  int depth = split_depth;

  while (true) {
    if (entering) {
      if (max_nodes_per_worker > 0 && nodes >= max_nodes_per_worker) {
        complete = false;
        break;
      }
      ++nodes;

      const double bound = admissible_lower_bound(
          fixed_energy[depth],
          effective,
          negative_incident,
          positive_incident,
          negative_edge_sum[depth],
          positive_edge_sum[depth],
          depth,
          size);
      const double incumbent = *global_best;
      const double tolerance =
          certificate_tolerance * (1.0 + fabs(incumbent));
      if (bound >= incumbent - tolerance) {
        if (depth < size) {
          next_branch[depth] = 2;
        }
        entering = false;
      } else if (depth == size) {
        const double candidate = evaluate_state(
            constant[0],
            linear,
            interaction,
            size,
            state_low,
            state_high);
        if (candidate < local_best) {
          local_best = candidate;
          local_best_low = state_low;
          local_best_high = state_high;
          atomic_min_double(global_best, candidate);
        }
        entering = false;
      } else {
        // Gauge transformation makes zero the supplied incumbent, so visit
        // the incumbent-compatible branch before its flip.
        next_branch[depth] = 0;
        entering = false;
      }
      continue;
    }

    if (depth < size && next_branch[depth] < 2) {
      const int branch_depth = depth;
      const bool value = next_branch[branch_depth]++ != 0;
      decision[branch_depth] = static_cast<uint8_t>(value);
      set_state_bit(state_low, state_high, branch_depth, value);

      const double removed_negative = negative_incident[branch_depth];
      const double removed_positive = positive_incident[branch_depth];
      fixed_energy[branch_depth + 1] =
          fixed_energy[branch_depth] +
          (value ? effective[branch_depth] : 0.0);
      negative_edge_sum[branch_depth + 1] =
          negative_edge_sum[branch_depth] - removed_negative;
      positive_edge_sum[branch_depth + 1] =
          positive_edge_sum[branch_depth] - removed_positive;
      for (int i = branch_depth + 1; i < size; ++i) {
        const double coupling = interaction[branch_depth * size + i];
        if (value) {
          effective[i] += coupling;
        }
        if (coupling < 0.0) {
          negative_incident[i] -= coupling;
        } else {
          positive_incident[i] -= coupling;
        }
      }
      ++depth;
      entering = true;
      continue;
    }

    if (depth == split_depth) {
      break;
    }

    --depth;
    const bool value = decision[depth] != 0;
    for (int i = depth + 1; i < size; ++i) {
      const double coupling = interaction[depth * size + i];
      if (value) {
        effective[i] -= coupling;
      }
      if (coupling < 0.0) {
        negative_incident[i] += coupling;
      } else {
        positive_incident[i] += coupling;
      }
    }
    if (value) {
      set_state_bit(state_low, state_high, depth, false);
    }
  }

  candidate_states[2 * worker] = static_cast<int64_t>(local_best_low);
  candidate_states[2 * worker + 1] = static_cast<int64_t>(local_best_high);
  candidate_costs[worker] = local_best;
  nodes_visited[worker] = nodes;
  completed[worker] = static_cast<uint8_t>(complete);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
adjacent_branch_bound_cuda(
    const at::Tensor &constant,
    const at::Tensor &linear,
    const at::Tensor &interaction,
    int64_t split_depth,
    int64_t max_nodes_per_worker,
    double certificate_tolerance) {
  TORCH_CHECK(constant.is_cuda(), "constant must be a CUDA tensor");
  TORCH_CHECK(linear.is_cuda(), "linear must be a CUDA tensor");
  TORCH_CHECK(interaction.is_cuda(), "interaction must be a CUDA tensor");
  TORCH_CHECK(constant.scalar_type() == at::kDouble, "constant must have dtype float64");
  TORCH_CHECK(linear.scalar_type() == at::kDouble, "linear must have dtype float64");
  TORCH_CHECK(interaction.scalar_type() == at::kDouble, "interaction must have dtype float64");
  TORCH_CHECK(constant.dim() == 0, "constant must be scalar");
  TORCH_CHECK(linear.dim() == 1, "linear must be rank one");
  TORCH_CHECK(
      linear.numel() >= 1 && linear.numel() <= kMaxDecisions,
      "linear must contain between 1 and ",
      kMaxDecisions,
      " decisions");
  TORCH_CHECK(
      interaction.dim() == 2 && interaction.size(0) == linear.numel() &&
          interaction.size(1) == linear.numel(),
      "interaction must be square and match linear");
  TORCH_CHECK(
      constant.device() == linear.device() &&
          interaction.device() == linear.device(),
      "all inputs must be on the same CUDA device");
  TORCH_CHECK(
      constant.is_contiguous() && linear.is_contiguous() &&
          interaction.is_contiguous(),
      "all inputs must be contiguous");
  TORCH_CHECK(
      split_depth >= 0 && split_depth <= linear.numel() &&
          split_depth <= kMaxSplitDepth,
      "split_depth must be in [0, min(linear.numel(), ",
      kMaxSplitDepth,
      ")]");
  TORCH_CHECK(
      max_nodes_per_worker >= 0,
      "max_nodes_per_worker must be zero (unlimited) or positive");
  TORCH_CHECK(
      std::isfinite(certificate_tolerance) && certificate_tolerance >= 0.0,
      "certificate_tolerance must be finite and non-negative");

  const c10::cuda::CUDAGuard device_guard(linear.device());
  const int size = static_cast<int>(linear.numel());
  const int64_t worker_count = int64_t{1} << split_depth;
  auto candidate_states =
      at::zeros({worker_count, 2}, linear.options().dtype(at::kLong));
  auto candidate_costs = at::empty({worker_count}, linear.options());
  auto root_lower_bounds = at::empty({worker_count}, linear.options());
  auto nodes_visited =
      at::zeros({worker_count}, linear.options().dtype(at::kLong));
  auto completed =
      at::zeros({worker_count}, linear.options().dtype(at::kByte));
  auto global_best = constant.clone();

  const int64_t blocks =
      (worker_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(linear.get_device());
  adjacent_branch_bound_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      constant.data_ptr<double>(),
      linear.data_ptr<double>(),
      interaction.data_ptr<double>(),
      size,
      static_cast<int>(split_depth),
      worker_count,
      max_nodes_per_worker,
      certificate_tolerance,
      global_best.data_ptr<double>(),
      candidate_states.data_ptr<int64_t>(),
      candidate_costs.data_ptr<double>(),
      root_lower_bounds.data_ptr<double>(),
      nodes_visited.data_ptr<int64_t>(),
      completed.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(
      candidate_states,
      candidate_costs,
      root_lower_bounds,
      nodes_visited,
      completed,
      global_best);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_adjacent_exact, m) {
  m.def(
      "branch_bound(Tensor constant, Tensor linear, Tensor interaction, "
      "int split_depth, int max_nodes_per_worker=0, "
      "float certificate_tolerance=1e-12) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gptqmodel_adjacent_exact, CUDA, m) {
  m.impl("branch_bound", &adjacent_branch_bound_cuda);
}
