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
// Recompute from the bit pattern frequently so FP32 incremental drift cannot
// accumulate across a long Gray-code range. 1024 is large enough to amortize
// the O(size^2) rebase cost while small enough to keep FP32 rounding stable
// for the internal accumulator; the final candidate cost is always recomputed
// from the original FP64 QUBO values using the selected best state.
constexpr uint64_t kRebaseInterval = 1024;

// Internal accumulator type.  FP32 is much faster on Ampere than FP64 and is
// sufficient for the incremental Gray-code energy, because we recompute from
// the original FP64 data at every rebase point and again at the end.
using acc_t = float;

__device__ __forceinline__ uint32_t gray_code(uint64_t index) {
  return static_cast<uint32_t>(index ^ (index >> 1));
}

template <typename T>
__device__ __forceinline__ T warp_sum(T value) {
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

  // Load the small QUBO data into shared memory once per block and store
  // interaction transposed (column-major per row lane) to avoid 32-way bank
  // conflicts when every warp lane reads the same decision column.
  extern __shared__ char smem[];
  double* const smem_interaction = reinterpret_cast<double*>(smem);
  double* const smem_linear = smem_interaction + size * size;
  for (int idx = threadIdx.x; idx < size * size; idx += kThreadsPerBlock) {
    const int row = idx / size;
    const int col = idx % size;
    smem_interaction[col * size + row] = interaction[row * size + col];
  }
  for (int idx = threadIdx.x; idx < size; idx += kThreadsPerBlock) {
    smem_linear[idx] = linear[idx];
  }
  __syncthreads();

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
  acc_t field = 0.0;
  acc_t energy = 0.0;
  acc_t best_energy = FLT_MAX;

  for (uint64_t index = begin; index < end; ++index) {
    const uint64_t local_index = index - begin;
    if ((local_index & (kRebaseInterval - 1)) == 0) {
      state = gray_code(index);
      if (lane < size) {
        field = static_cast<acc_t>(smem_linear[lane]);
        uint32_t bits = state;
        while (bits) {
          const int other = __ffs(bits) - 1;
          field += static_cast<acc_t>(smem_interaction[other * size + lane]);
          bits &= bits - 1;
        }
      } else {
        field = 0.0;
      }

      const acc_t contribution =
          lane < size && ((state >> lane) & 1u)
              ? static_cast<acc_t>(0.5) * (static_cast<acc_t>(smem_linear[lane]) + field)
              : acc_t(0.0);
      const acc_t contribution_sum = warp_sum(contribution);
      if (lane == 0) {
        energy = static_cast<acc_t>(constant[0]) + contribution_sum;
      }
    }

    if (lane == 0 &&
        (energy < best_energy || (energy == best_energy && state < best_state))) {
      best_energy = energy;
      best_state = state;
    }

    if (index + 1 < end) {
      // In a binary-reflected Gray code the bit that changes when moving from
      // index to index+1 is the least-significant set bit of (index+1).
      const uint32_t n = static_cast<uint32_t>(index + 1);
      const int flipped = __ffs(n) - 1;
      const uint32_t bit = 1u << flipped;
      const acc_t direction = ((state >> flipped) & 1u) ? acc_t(-1.0) : acc_t(1.0);
      const acc_t flipped_field = __shfl_sync(0xffffffffu, field, flipped);
      if (lane == 0) {
        energy += direction * flipped_field;
      }
      if (lane < size) {
        field += direction * static_cast<acc_t>(smem_interaction[flipped * size + lane]);
      }
      state ^= bit;
    }
  }

  best_state = __shfl_sync(0xffffffffu, best_state, 0);
  if (lane < size) {
    field = smem_linear[lane];
    uint32_t bits = best_state;
    while (bits) {
      const int other = __ffs(bits) - 1;
      field += smem_interaction[other * size + lane];
      bits &= bits - 1;
    }
  } else {
    field = 0.0;
  }
  const double best_contribution =
      lane < size && ((best_state >> lane) & 1u) ? 0.5 * (smem_linear[lane] + field) : 0.0;
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
  const size_t adjacent_exact_smem = sizeof(double) * (static_cast<size_t>(size) * size + size);
  adjacent_exact_candidates_kernel<<<blocks, kThreadsPerBlock, adjacent_exact_smem, stream>>>(
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
