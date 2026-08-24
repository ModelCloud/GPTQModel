// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

// Exact greedy-token divergence counters.  The argmax kernel uses the same
// deterministic ordering as the Python reference: larger value first, then
// lower token index.  The operator emits only four int64 counters, so no
// vocabulary-sized intermediate or CPU logits copy is required.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace gptqmodel_divergence_metrics_cuda {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ bool precedes(float left_value, int left_index, float right_value, int right_index) {
  const bool left_nan = isnan(left_value);
  const bool right_nan = isnan(right_value);
  if (left_nan != right_nan) {
    return left_nan;
  }
  return left_nan || left_value > right_value || (left_value == right_value && left_index < right_index);
}

__device__ __forceinline__ void warp_argmax(float& value, int& index) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_value = __shfl_down_sync(0xFFFFFFFFu, value, offset);
    const int other_index = __shfl_down_sync(0xFFFFFFFFu, index, offset);
    if (precedes(other_value, other_index, value, index)) {
      value = other_value;
      index = other_index;
    }
  }
}

__global__ void argmax_kernel(
    const float* __restrict__ dense,
    const float* __restrict__ quantized,
    int64_t* __restrict__ indices,
    int token_count,
    int columns) {
  const int token = blockIdx.x;
  if (token >= token_count) {
    return;
  }
  const int64_t offset = static_cast<int64_t>(token) * columns;
  __shared__ float dense_values[kWarps];
  __shared__ float quantized_values[kWarps];
  __shared__ int dense_indices[kWarps];
  __shared__ int quantized_indices[kWarps];

  float dense_best = -INFINITY;
  float quantized_best = -INFINITY;
  int dense_index = std::numeric_limits<int>::max();
  int quantized_index = std::numeric_limits<int>::max();
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float dense_value = dense[offset + column];
    const float quantized_value = quantized[offset + column];
    if (precedes(dense_value, column, dense_best, dense_index)) {
      dense_best = dense_value;
      dense_index = column;
    }
    if (precedes(quantized_value, column, quantized_best, quantized_index)) {
      quantized_best = quantized_value;
      quantized_index = column;
    }
  }
  warp_argmax(dense_best, dense_index);
  warp_argmax(quantized_best, quantized_index);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    dense_values[warp] = dense_best;
    quantized_values[warp] = quantized_best;
    dense_indices[warp] = dense_index;
    quantized_indices[warp] = quantized_index;
  }
  __syncthreads();
  if (warp == 0) {
    dense_best = lane < kWarps ? dense_values[lane] : -INFINITY;
    quantized_best = lane < kWarps ? quantized_values[lane] : -INFINITY;
    dense_index = lane < kWarps ? dense_indices[lane] : std::numeric_limits<int>::max();
    quantized_index = lane < kWarps ? quantized_indices[lane] : std::numeric_limits<int>::max();
    warp_argmax(dense_best, dense_index);
    warp_argmax(quantized_best, quantized_index);
  }
  if (threadIdx.x == 0) {
    indices[static_cast<int64_t>(token) * 2] = dense_index;
    indices[static_cast<int64_t>(token) * 2 + 1] = quantized_index;
  }
}

__global__ void reduce_kernel(const int64_t* __restrict__ indices, int64_t* __restrict__ result, int token_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int64_t matches = 0;
  int64_t first_mismatch = static_cast<int64_t>(token_count) + 1;
  for (int token = 0; token < token_count; ++token) {
    const int64_t dense_index = indices[static_cast<int64_t>(token) * 2];
    const int64_t quantized_index = indices[static_cast<int64_t>(token) * 2 + 1];
    if (dense_index == quantized_index) {
      ++matches;
    } else if (first_mismatch == static_cast<int64_t>(token_count) + 1) {
      first_mismatch = static_cast<int64_t>(token) + 1;
    }
  }
  result[0] = matches;
  result[1] = matches == token_count ? 1 : 0;
  result[2] = first_mismatch;
  result[3] = token_count;
}

at::Tensor divergence_metrics_cuda(
    const at::Tensor& dense,
    const at::Tensor& quantized,
    int64_t token_count) {
  TORCH_CHECK(dense.is_cuda() && quantized.is_cuda(), "divergence_metrics_cuda expects CUDA tensors");
  TORCH_CHECK(dense.scalar_type() == at::kFloat && quantized.scalar_type() == at::kFloat,
              "divergence_metrics_cuda expects FP32 tensors");
  TORCH_CHECK(dense.dim() == 2 && quantized.dim() == 2, "divergence_metrics_cuda expects rank-2 tensors");
  TORCH_CHECK(dense.sizes() == quantized.sizes(), "divergence tensors must have identical shapes");
  TORCH_CHECK(dense.is_contiguous() && quantized.is_contiguous(), "divergence tensors must be contiguous");
  TORCH_CHECK(dense.device() == quantized.device(), "divergence tensors must share one device");
  TORCH_CHECK(token_count > 0 && token_count <= dense.size(0), "invalid divergence token horizon");
  TORCH_CHECK(dense.size(1) > 0 && dense.size(1) <= std::numeric_limits<int>::max(),
              "invalid divergence vocabulary width");
  TORCH_CHECK(token_count <= std::numeric_limits<int>::max(), "divergence horizon exceeds int32 bounds");

  const c10::cuda::CUDAGuard device_guard(dense.device());
  const auto options = dense.options().dtype(at::kLong);
  at::Tensor indices = at::empty({token_count, 2}, options);
  at::Tensor result = at::empty({4}, options);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  argmax_kernel<<<static_cast<int>(token_count), kThreads, 0, stream>>>(
      dense.data_ptr<float>(), quantized.data_ptr<float>(), indices.data_ptr<int64_t>(),
      static_cast<int>(token_count), static_cast<int>(dense.size(1)));
  reduce_kernel<<<1, 1, 0, stream>>>(indices.data_ptr<int64_t>(), result.data_ptr<int64_t>(), static_cast<int>(token_count));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

}  // namespace gptqmodel_divergence_metrics_cuda

TORCH_LIBRARY_FRAGMENT(gptqmodel_diagnostic_metrics, library) {
  library.def("divergence_metrics_cuda(Tensor dense, Tensor quantized, int token_count) -> Tensor");
  library.impl(
      "divergence_metrics_cuda",
      c10::DispatchKey::CUDA,
      TORCH_FN(gptqmodel_divergence_metrics_cuda::divergence_metrics_cuda));
}
