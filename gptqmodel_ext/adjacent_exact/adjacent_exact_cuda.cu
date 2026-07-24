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
#include <cstdint>
#include <tuple>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int64_t kMaxWorkerWarps = int64_t{1} << 20;
// Recompute from the bit pattern frequently so incremental FP64 drift cannot
// accumulate across a long Gray-code range.
constexpr uint64_t kRebaseInterval = 64;

__device__ __forceinline__ uint32_t gray_code(uint64_t index) {
  return static_cast<uint32_t>(index ^ (index >> 1));
}

__device__ __forceinline__ double warp_sum(double value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void adjacent_exact_candidates_kernel(
    const double *__restrict__ constant,
    const double *__restrict__ linear,
    const double *__restrict__ interaction,
    int size,
    uint64_t total_states,
    uint64_t states_per_warp,
    int64_t worker_warps,
    int64_t *__restrict__ candidate_states,
    double *__restrict__ candidate_costs) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int64_t worker = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (worker >= worker_warps) {
    return;
  }

  const uint64_t begin = static_cast<uint64_t>(worker) * states_per_warp;
  const uint64_t end = min(begin + states_per_warp, total_states);
  if (begin >= end) {
    if (lane == 0) {
      candidate_states[worker] = 0;
      candidate_costs[worker] = DBL_MAX;
    }
    return;
  }

  uint32_t state = 0;
  uint32_t best_state = 0;
  double field = 0.0;
  double energy = 0.0;
  double best_energy = DBL_MAX;

  for (uint64_t index = begin; index < end; ++index) {
    const uint64_t local_index = index - begin;
    if ((local_index & (kRebaseInterval - 1)) == 0) {
      state = gray_code(index);
      if (lane < size) {
        field = linear[lane];
#pragma unroll 1
        for (int other = 0; other < size; ++other) {
          if ((state >> other) & 1u) {
            field += interaction[lane * size + other];
          }
        }
      } else {
        field = 0.0;
      }

      const double contribution =
          lane < size && ((state >> lane) & 1u) ? 0.5 * (linear[lane] + field) : 0.0;
      const double contribution_sum = warp_sum(contribution);
      if (lane == 0) {
        energy = constant[0] + contribution_sum;
      }
    }

    if (lane == 0 &&
        (energy < best_energy || (energy == best_energy && state < best_state))) {
      best_energy = energy;
      best_state = state;
    }

    if (index + 1 < end) {
      const uint32_t next_state = gray_code(index + 1);
      const uint32_t changed = state ^ next_state;
      const int flipped = __ffs(changed) - 1;
      const double direction = ((next_state >> flipped) & 1u) ? 1.0 : -1.0;
      const double flipped_field = __shfl_sync(0xffffffffu, field, flipped);
      if (lane == 0) {
        energy += direction * flipped_field;
      }
      if (lane < size) {
        field += direction * interaction[lane * size + flipped];
      }
      state = next_state;
    }
  }

  best_state = __shfl_sync(0xffffffffu, best_state, 0);
  if (lane < size) {
    field = linear[lane];
#pragma unroll 1
    for (int other = 0; other < size; ++other) {
      if ((best_state >> other) & 1u) {
        field += interaction[lane * size + other];
      }
    }
  } else {
    field = 0.0;
  }
  const double best_contribution =
      lane < size && ((best_state >> lane) & 1u) ? 0.5 * (linear[lane] + field) : 0.0;
  const double best_contribution_sum = warp_sum(best_contribution);
  if (lane == 0) {
    candidate_states[worker] = static_cast<int64_t>(best_state);
    candidate_costs[worker] = constant[0] + best_contribution_sum;
  }
}

std::tuple<at::Tensor, at::Tensor> adjacent_exact_candidates_cuda(
    const at::Tensor &constant,
    const at::Tensor &linear,
    const at::Tensor &interaction,
    int64_t requested_warps) {
  TORCH_CHECK(constant.is_cuda(), "constant must be a CUDA tensor");
  TORCH_CHECK(linear.is_cuda(), "linear must be a CUDA tensor");
  TORCH_CHECK(interaction.is_cuda(), "interaction must be a CUDA tensor");
  TORCH_CHECK(constant.scalar_type() == at::kDouble, "constant must have dtype float64");
  TORCH_CHECK(linear.scalar_type() == at::kDouble, "linear must have dtype float64");
  TORCH_CHECK(interaction.scalar_type() == at::kDouble, "interaction must have dtype float64");
  TORCH_CHECK(constant.dim() == 0, "constant must be scalar");
  TORCH_CHECK(linear.dim() == 1, "linear must be rank one");
  TORCH_CHECK(linear.numel() >= 1 && linear.numel() <= 32,
              "linear must contain between 1 and 32 decisions");
  TORCH_CHECK(interaction.dim() == 2 && interaction.size(0) == linear.numel() &&
                  interaction.size(1) == linear.numel(),
              "interaction must be square and match linear");
  TORCH_CHECK(constant.device() == linear.device() && interaction.device() == linear.device(),
              "all inputs must be on the same CUDA device");
  TORCH_CHECK(constant.is_contiguous() && linear.is_contiguous() && interaction.is_contiguous(),
              "all inputs must be contiguous");
  TORCH_CHECK(requested_warps >= 0 && requested_warps <= kMaxWorkerWarps,
              "requested_warps must be zero or no greater than ", kMaxWorkerWarps);

  const c10::cuda::CUDAGuard device_guard(linear.device());
  const int size = static_cast<int>(linear.numel());
  const uint64_t total_states = uint64_t{1} << size;
  const cudaDeviceProp *properties = at::cuda::getDeviceProperties(linear.get_device());
  const int64_t automatic_warps = static_cast<int64_t>(properties->multiProcessorCount) * 64;
  const int64_t worker_warps =
      std::max<int64_t>(1, std::min<int64_t>(
                               requested_warps > 0 ? requested_warps : automatic_warps,
                               static_cast<int64_t>(total_states)));
  const uint64_t states_per_warp =
      (total_states + static_cast<uint64_t>(worker_warps) - 1) /
      static_cast<uint64_t>(worker_warps);

  auto candidate_states = at::empty({worker_warps}, linear.options().dtype(at::kLong));
  auto candidate_costs = at::empty({worker_warps}, linear.options());
  const int64_t blocks = (worker_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(linear.get_device());
  adjacent_exact_candidates_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      constant.data_ptr<double>(),
      linear.data_ptr<double>(),
      interaction.data_ptr<double>(),
      size,
      total_states,
      states_per_warp,
      worker_warps,
      candidate_states.data_ptr<int64_t>(),
      candidate_costs.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(candidate_states, candidate_costs);
}

}  // namespace

TORCH_LIBRARY(gptqmodel_adjacent_exact, m) {
  m.def(
      "exact_candidates(Tensor constant, Tensor linear, Tensor interaction, int warps=0) "
      "-> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gptqmodel_adjacent_exact, CUDA, m) {
  m.impl("exact_candidates", &adjacent_exact_candidates_cuda);
}
