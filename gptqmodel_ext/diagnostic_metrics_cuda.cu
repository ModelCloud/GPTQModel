// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

// Fused row diagnostics for QVQ evaluation. One CTA owns one logical token
// row and streams its feature dimension without materializing probability
// tensors. The operator emits per-row FP64 moments, forward KL, deterministic
// Top-10 indices, and a tie flag used by Python to preserve historical CPU
// Top-K ordering only for genuinely ambiguous rows.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/types.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

namespace gptqmodel_diagnostic_metrics_cuda {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMomentCount = 10;
constexpr int kStatsColumns = 13;
constexpr int kTopCount = 11;

template <typename T>
__device__ __forceinline__ T warp_sum(T value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T warp_max(T value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = max(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
  }
  return value;
}

template <int Count>
__device__ __forceinline__ void block_sum(double (&values)[Count], double* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int index = 0; index < Count; ++index) {
    values[index] = warp_sum(values[index]);
    if (lane == 0) {
      shared[warp * Count + index] = values[index];
    }
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int index = 0; index < Count; ++index) {
      double value = lane < kWarps ? shared[lane * Count + index] : 0.0;
      value = warp_sum(value);
      if (lane == 0) {
        shared[index] = value;
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int index = 0; index < Count; ++index) {
    values[index] = shared[index];
  }
}

template <int Count>
__device__ __forceinline__ void block_sum(float (&values)[Count], float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int index = 0; index < Count; ++index) {
    values[index] = warp_sum(values[index]);
    if (lane == 0) {
      shared[warp * Count + index] = values[index];
    }
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int index = 0; index < Count; ++index) {
      float value = lane < kWarps ? shared[lane * Count + index] : 0.0f;
      value = warp_sum(value);
      if (lane == 0) {
        shared[index] = value;
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int index = 0; index < Count; ++index) {
    values[index] = shared[index];
  }
}

__device__ __forceinline__ float block_max(float value, float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_max(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < kWarps ? shared[lane] : -INFINITY;
    value = warp_max(value);
    if (lane == 0) {
      shared[0] = value;
    }
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ bool top_precedes(float left_value, int left_index, float right_value, int right_index) {
  return left_value > right_value || (left_value == right_value && left_index < right_index);
}

__device__ __forceinline__ void insert_top(
    float value,
    int index,
    float (&values)[kTopCount],
    int (&indices)[kTopCount]) {
  if (!top_precedes(value, index, values[kTopCount - 1], indices[kTopCount - 1])) {
    return;
  }
  int position = kTopCount - 1;
  while (position > 0 && top_precedes(value, index, values[position - 1], indices[position - 1])) {
    values[position] = values[position - 1];
    indices[position] = indices[position - 1];
    --position;
  }
  values[position] = value;
  indices[position] = index;
}

template <bool Normalize, bool IncludeTop>
__global__ __launch_bounds__(kThreads) void primary_metrics_kernel(
    const float* __restrict__ dense,
    const float* __restrict__ quantized,
    double* __restrict__ stats,
    int64_t* __restrict__ top_indices,
    bool* __restrict__ ambiguous,
    int rows,
    int columns) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t row_offset = static_cast<int64_t>(row) * columns;
  dense += row_offset;
  quantized += row_offset;
  __shared__ double moment_shared[kWarps * kMomentCount];
  __shared__ float float_shared[kWarps * 2];

  double moments[kMomentCount] = {};
  float maximum_error = 0.0f;
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float dense_value = dense[column];
    const float quantized_value = quantized[column];
    const double dense_double = static_cast<double>(dense_value);
    const double quantized_double = static_cast<double>(quantized_value);
    const double error = quantized_double - dense_double;
    moments[0] += dense_double;
    moments[1] += quantized_double;
    moments[2] += dense_double * dense_double;
    moments[3] += quantized_double * quantized_double;
    moments[4] += error;
    moments[5] += fabs(error);
    moments[6] += error * error;
    moments[7] += dense_double * quantized_double;
    moments[8] += ((dense_value >= 0.0f) == (quantized_value >= 0.0f)) ? 1.0 : 0.0;
    moments[9] += (isfinite(dense_value) && isfinite(quantized_value)) ? 0.0 : 1.0;
    maximum_error = fmaxf(maximum_error, fabsf(quantized_value - dense_value));
  }
  block_sum(moments, moment_shared);
  maximum_error = block_max(maximum_error, float_shared);

  const float dense_mean = Normalize ? static_cast<float>(moments[0] / columns) : 0.0f;
  const double dense_variance = fmax(0.0, moments[2] / columns - static_cast<double>(dense_mean) * dense_mean);
  const float dense_scale = Normalize ? fmaxf(static_cast<float>(sqrt(dense_variance)), 1e-6f) : 1.0f;
  float dense_maximum = -INFINITY;
  float quantized_maximum = -INFINITY;
  float dense_top[kTopCount];
  float quantized_top[kTopCount];
  int dense_top_indices[kTopCount];
  int quantized_top_indices[kTopCount];
#pragma unroll
  for (int index = 0; index < kTopCount; ++index) {
    dense_top[index] = -INFINITY;
    quantized_top[index] = -INFINITY;
    dense_top_indices[index] = std::numeric_limits<int>::max();
    quantized_top_indices[index] = std::numeric_limits<int>::max();
  }
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float dense_value = Normalize ? (dense[column] - dense_mean) / dense_scale : dense[column];
    const float quantized_value = Normalize ? (quantized[column] - dense_mean) / dense_scale : quantized[column];
    dense_maximum = fmaxf(dense_maximum, dense_value);
    quantized_maximum = fmaxf(quantized_maximum, quantized_value);
    if constexpr (IncludeTop) {
      insert_top(dense_value, column, dense_top, dense_top_indices);
      insert_top(quantized_value, column, quantized_top, quantized_top_indices);
    }
  }
  dense_maximum = block_max(dense_maximum, float_shared);
  quantized_maximum = block_max(quantized_maximum, float_shared);

  if constexpr (IncludeTop) {
    __shared__ float top_value_shared[2 * kThreads * kTopCount];
    __shared__ int top_index_shared[2 * kThreads * kTopCount];
#pragma unroll
    for (int index = 0; index < kTopCount; ++index) {
      const int offset = threadIdx.x * kTopCount + index;
      top_value_shared[offset] = dense_top[index];
      top_value_shared[kThreads * kTopCount + offset] = quantized_top[index];
      top_index_shared[offset] = dense_top_indices[index];
      top_index_shared[kThreads * kTopCount + offset] = quantized_top_indices[index];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float merged_dense[kTopCount];
      float merged_quantized[kTopCount];
      int merged_dense_indices[kTopCount];
      int merged_quantized_indices[kTopCount];
#pragma unroll
      for (int index = 0; index < kTopCount; ++index) {
        merged_dense[index] = -INFINITY;
        merged_quantized[index] = -INFINITY;
        merged_dense_indices[index] = std::numeric_limits<int>::max();
        merged_quantized_indices[index] = std::numeric_limits<int>::max();
      }
      for (int candidate = 0; candidate < kThreads * kTopCount; ++candidate) {
        insert_top(top_value_shared[candidate], top_index_shared[candidate], merged_dense, merged_dense_indices);
        insert_top(
            top_value_shared[kThreads * kTopCount + candidate],
            top_index_shared[kThreads * kTopCount + candidate],
            merged_quantized,
            merged_quantized_indices);
      }
      const int64_t dense_output = static_cast<int64_t>(row) * 10;
      const int64_t quantized_output = static_cast<int64_t>(rows + row) * 10;
      for (int index = 0; index < 10; ++index) {
        top_indices[dense_output + index] = merged_dense_indices[index];
        top_indices[quantized_output + index] = merged_quantized_indices[index];
      }
      ambiguous[row] =
          (columns > 1 && (merged_dense[0] == merged_dense[1] || merged_quantized[0] == merged_quantized[1]))
          || (columns > 5 && (merged_dense[4] == merged_dense[5] || merged_quantized[4] == merged_quantized[5]))
          || (columns > 10 && (merged_dense[9] == merged_dense[10] || merged_quantized[9] == merged_quantized[10]));
    }
  }

  float exponential_sums[2] = {};
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float dense_value = Normalize ? (dense[column] - dense_mean) / dense_scale : dense[column];
    const float quantized_value = Normalize ? (quantized[column] - dense_mean) / dense_scale : quantized[column];
    exponential_sums[0] += expf(dense_value - dense_maximum);
    exponential_sums[1] += expf(quantized_value - quantized_maximum);
  }
  block_sum(exponential_sums, float_shared);
  const float dense_log_normalizer = dense_maximum + logf(exponential_sums[0]);
  const float quantized_log_normalizer = quantized_maximum + logf(exponential_sums[1]);
  float kl_values[1] = {};
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float dense_value = Normalize ? (dense[column] - dense_mean) / dense_scale : dense[column];
    const float quantized_value = Normalize ? (quantized[column] - dense_mean) / dense_scale : quantized[column];
    const float dense_log_probability = dense_value - dense_log_normalizer;
    const float quantized_log_probability = quantized_value - quantized_log_normalizer;
    kl_values[0] += expf(dense_log_probability) * (dense_log_probability - quantized_log_probability);
  }
  block_sum(kl_values, float_shared);

  if (threadIdx.x == 0) {
    double* row_stats = stats + static_cast<int64_t>(row) * kStatsColumns;
    row_stats[0] = columns;
    row_stats[1] = moments[9] == 0.0 ? 1.0 : 0.0;
    row_stats[2] = moments[0];
    row_stats[3] = moments[1];
    row_stats[4] = moments[2];
    row_stats[5] = moments[3];
    row_stats[6] = moments[4];
    row_stats[7] = moments[5];
    row_stats[8] = moments[6];
    row_stats[9] = moments[7];
    row_stats[10] = maximum_error;
    row_stats[11] = moments[8];
    row_stats[12] = kl_values[0];
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> primary_metrics_cuda(
    const at::Tensor& dense,
    const at::Tensor& quantized,
    bool normalize_distribution,
    bool include_top10) {
  TORCH_CHECK(dense.is_cuda() && quantized.is_cuda(), "primary_metrics_cuda expects CUDA tensors");
  TORCH_CHECK(dense.scalar_type() == at::kFloat && quantized.scalar_type() == at::kFloat,
              "primary_metrics_cuda expects FP32 tensors");
  TORCH_CHECK(dense.dim() == 2 && quantized.dim() == 2, "primary_metrics_cuda expects rank-2 tensors");
  TORCH_CHECK(dense.sizes() == quantized.sizes(), "primary_metrics_cuda tensors must have identical shapes");
  TORCH_CHECK(dense.is_contiguous() && quantized.is_contiguous(), "primary_metrics_cuda tensors must be contiguous");
  TORCH_CHECK(dense.device() == quantized.device(), "primary_metrics_cuda tensors must share one device");
  TORCH_CHECK(dense.size(0) > 0 && dense.size(1) > 0, "primary_metrics_cuda tensors must be non-empty");
  TORCH_CHECK(dense.size(0) <= std::numeric_limits<int>::max() && dense.size(1) <= std::numeric_limits<int>::max(),
              "primary_metrics_cuda dimensions exceed int32 launch bounds");
  if (include_top10) {
    TORCH_CHECK(dense.size(1) >= 10, "primary_metrics_cuda Top-10 requires at least ten columns");
  }

  const c10::cuda::CUDAGuard device_guard(dense.device());
  const int rows = static_cast<int>(dense.size(0));
  const int columns = static_cast<int>(dense.size(1));
  at::Tensor stats = at::empty({rows, kStatsColumns}, dense.options().dtype(at::kDouble));
  at::Tensor top_indices = include_top10
      ? at::empty({2, rows, 10}, dense.options().dtype(at::kLong))
      : at::empty({0}, dense.options().dtype(at::kLong));
  at::Tensor ambiguous = include_top10
      ? at::empty({rows}, dense.options().dtype(at::kBool))
      : at::empty({0}, dense.options().dtype(at::kBool));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(rows);
  const dim3 block(kThreads);
#define LAUNCH(NORMALIZE, TOP) \
  primary_metrics_kernel<NORMALIZE, TOP><<<grid, block, 0, stream>>>( \
      dense.data_ptr<float>(), quantized.data_ptr<float>(), stats.data_ptr<double>(), \
      top_indices.data_ptr<int64_t>(), ambiguous.data_ptr<bool>(), rows, columns)
  if (normalize_distribution) {
    if (include_top10) {
      LAUNCH(true, true);
    } else {
      LAUNCH(true, false);
    }
  } else if (include_top10) {
    LAUNCH(false, true);
  } else {
    LAUNCH(false, false);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {stats, top_indices, ambiguous};
}

}  // namespace gptqmodel_diagnostic_metrics_cuda

TORCH_LIBRARY_FRAGMENT(gptqmodel_diagnostic_metrics, library) {
  library.def(
      "primary_metrics_cuda(Tensor dense, Tensor quantized, bool normalize_distribution, bool include_top10) "
      "-> (Tensor, Tensor, Tensor)");
  library.impl(
      "primary_metrics_cuda",
      c10::DispatchKey::CUDA,
      TORCH_FN(gptqmodel_diagnostic_metrics_cuda::primary_metrics_cuda));
}
