// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

// One warp owns one output row.  Its four lane groups compute the eight rank
// components cooperatively, so the hidden projection is evaluated once per
// row rather than once per output column.  The hidden values are rounded to
// FP16 before the FP32 expansion, matching the reference rank8 contract.
__global__ void qvq_rank8_fused_no_hadamard_kernel(
    const at::Half* __restrict__ transformed,
    int64_t transformed_stride,
    const float* __restrict__ rank8_a,
    const float* __restrict__ rank8_b,
    const float* __restrict__ base,
    int64_t base_stride,
    const float* __restrict__ scale_v,
    const float* __restrict__ bias,
    at::Half* __restrict__ output,
    int m,
    int k,
    int n) {
  constexpr int kRowsPerBlock = 4;
  constexpr int kRanks = 8;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int row = static_cast<int>(blockIdx.y) * kRowsPerBlock + warp;
  const int column = static_cast<int>(blockIdx.x) * 32 + lane;
  __shared__ float hidden[kRowsPerBlock * kRanks];

  if (row < m) {
    // Keep one deterministic FP32 accumulation order per rank.  Eight lanes
    // perform the eight projections while the remaining lanes wait; this
    // preserves the reference hidden rounding boundary and still exposes all
    // 32 lanes to the expansion/store phase below.
    if (lane < kRanks) {
      const int rank = lane;
      float partial = 0.0f;
      for (int index = 0; index < k; ++index) {
        partial += static_cast<float>(transformed[
            static_cast<int64_t>(row) * transformed_stride + index]) *
            rank8_a[static_cast<int64_t>(index) * kRanks + rank];
      }
      hidden[warp * kRanks + rank] = __half2float(__float2half_rn(partial));
    }
  }
  __syncthreads();

  if (row < m && column < n) {
    float value = base[static_cast<int64_t>(row) * base_stride + column];
#pragma unroll
    for (int rank = 0; rank < kRanks; ++rank) {
      value += hidden[warp * kRanks + rank] *
          rank8_b[static_cast<int64_t>(rank) * n + column];
    }
    value *= scale_v[column];
    if (bias != nullptr) {
      value += bias[column];
    }
    output[static_cast<int64_t>(row) * n + column] = __float2half_rn(value);
  }
}

} // namespace

extern "C" void qvq_rank8_fused_no_hadamard(
    const at::Tensor& transformed,
    const at::Tensor& rank8_a,
    const at::Tensor& rank8_b,
    const at::Tensor& base,
    const at::Tensor& scale_v,
    const at::Tensor& bias,
    at::Tensor& output,
    cudaStream_t stream) {
  TORCH_CHECK(transformed.is_cuda() && rank8_a.is_cuda() && rank8_b.is_cuda() &&
                  base.is_cuda() && scale_v.is_cuda() && output.is_cuda(),
              "native fused rank8 tensors must be CUDA tensors");
  TORCH_CHECK(transformed.scalar_type() == at::kHalf &&
                  rank8_a.scalar_type() == at::kFloat &&
                  rank8_b.scalar_type() == at::kFloat &&
                  base.scalar_type() == at::kFloat &&
                  scale_v.scalar_type() == at::kFloat &&
                  output.scalar_type() == at::kHalf,
              "native fused rank8 tensor dtypes are invalid");
  TORCH_CHECK(transformed.dim() == 2 && rank8_a.sizes() ==
                  at::IntArrayRef({transformed.size(1), 8}) &&
                  rank8_b.sizes() == at::IntArrayRef({8, base.size(1)}) &&
                  base.dim() == 2 && scale_v.numel() == base.size(1) &&
                  output.sizes() == base.sizes(),
              "native fused rank8 tensor shapes are invalid");
  TORCH_CHECK(transformed.is_contiguous() && rank8_a.is_contiguous() &&
                  rank8_b.is_contiguous() && scale_v.is_contiguous() &&
                  output.is_contiguous(),
              "native fused rank8 tensors must be contiguous");
  TORCH_CHECK(!bias.defined() ||
                  (bias.scalar_type() == at::kFloat && bias.numel() == base.size(1) &&
                   bias.is_contiguous()),
              "native fused rank8 bias must be contiguous FP32 or absent");
  const int m = static_cast<int>(base.size(0));
  const int k = static_cast<int>(transformed.size(1));
  const int n = static_cast<int>(base.size(1));
  constexpr int kThreads = 128;
  const dim3 grid(
      static_cast<unsigned>((n + 31) / 32),
      static_cast<unsigned>((m + 3) / 4),
      1);
  qvq_rank8_fused_no_hadamard_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const at::Half*>(transformed.data_ptr<at::Half>()),
      transformed.stride(0), rank8_a.data_ptr<float>(), rank8_b.data_ptr<float>(),
      base.data_ptr<float>(), base.stride(0), scale_v.data_ptr<float>(),
      bias.defined() ? bias.data_ptr<float>() : nullptr,
      reinterpret_cast<at::Half*>(output.data_ptr<at::Half>()), m, k, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
