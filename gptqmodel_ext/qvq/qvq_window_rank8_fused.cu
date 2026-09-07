// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

__global__ void qvq_rank8_epilogue_no_hadamard_kernel(
    const at::Half* __restrict__ hidden,
    const at::Half* __restrict__ rank8_b,
    const float* __restrict__ base,
    int64_t base_stride,
    const float* __restrict__ scale_v,
    const float* __restrict__ bias,
    at::Half* __restrict__ output,
    int m,
    int n) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int row = static_cast<int>(blockIdx.y) * 4 + warp;
  if (row >= m) return;
  for (int column = lane; column < n; column += 32) {
    float value = base[static_cast<int64_t>(row) * base_stride + column];
#pragma unroll
    for (int rank = 0; rank < 8; ++rank) {
      value += static_cast<float>(hidden[static_cast<int64_t>(row) * 8 + rank]) *
          static_cast<float>(rank8_b[static_cast<int64_t>(rank) * n + column]);
    }
    value *= scale_v[column];
    if (bias != nullptr) value += bias[column];
    output[static_cast<int64_t>(row) * n + column] = __float2half_rn(value);
  }
}

__device__ __forceinline__ float qvq_rank8_round_fp16(float value) {
  const half narrowed = __float2half_rn(value);
  const float restored = __half2float(narrowed);
  return isfinite(restored) ? restored : value;
}

// Power-of-two output Hadamard epilogue. This mirrors the existing native
// mode-3/4 butterfly ordering and FP16-emulation boundary, while folding the
// FP32 base + FP16 rank8 expansion into the same row tile. One block owns one
// output row; the padded shared layout avoids the 32-way bank conflicts used
// by the standalone Hadamard kernel.
__global__ void qvq_rank8_epilogue_hadamard_kernel(
    const at::Half* __restrict__ hidden,
    const at::Half* __restrict__ rank8_b,
    const float* __restrict__ base,
    int64_t base_stride,
    const float* __restrict__ scale_v,
    const float* __restrict__ bias,
    at::Half* __restrict__ output,
    int m,
    int n,
    bool normalize_first) {
  extern __shared__ float values[];
  const int row = static_cast<int>(blockIdx.x);
  if (row >= m) return;
  const auto padded = [](int index) { return index + (index >> 5); };
  const float divisor = __half2float(__float2half_rn(sqrtf(static_cast<float>(n))));
  const float reciprocal = 1.0f / sqrtf(static_cast<float>(n));
  for (int column = static_cast<int>(threadIdx.x); column < n; column += blockDim.x) {
    float value = base[static_cast<int64_t>(row) * base_stride + column];
#pragma unroll
    for (int rank = 0; rank < 8; ++rank) {
      value += static_cast<float>(hidden[static_cast<int64_t>(row) * 8 + rank]) *
          static_cast<float>(rank8_b[static_cast<int64_t>(rank) * n + column]);
    }
    value = qvq_rank8_round_fp16(value);
    if (normalize_first) value = qvq_rank8_round_fp16(value / divisor);
    values[padded(column)] = value;
  }
  __syncthreads();

  for (int bit = 1; bit < n; bit <<= 1) {
    for (int index = static_cast<int>(threadIdx.x); index < n; index += blockDim.x) {
      const int peer = index ^ bit;
      if (index < peer) {
        const float first = values[padded(index)];
        const float second = values[padded(peer)];
        values[padded(index)] = qvq_rank8_round_fp16(first + second);
        values[padded(peer)] = qvq_rank8_round_fp16(first - second);
      }
    }
    __syncthreads();
  }

  for (int column = static_cast<int>(threadIdx.x); column < n; column += blockDim.x) {
    float value = values[padded(column)];
    if (!normalize_first) value = qvq_rank8_round_fp16(value * reciprocal);
    value = qvq_rank8_round_fp16(value * scale_v[column]);
    if (bias != nullptr) value = qvq_rank8_round_fp16(value + bias[column]);
    output[static_cast<int64_t>(row) * n + column] = __float2half_rn(value);
  }
}

} // namespace

extern "C" void qvq_rank8_epilogue_no_hadamard(
    const at::Tensor& hidden,
    const at::Tensor& rank8_b,
    const at::Tensor& base,
    const at::Tensor& scale_v,
    const at::Tensor& bias,
    at::Tensor& output,
    cudaStream_t stream) {
  TORCH_CHECK(hidden.is_cuda() && rank8_b.is_cuda() && base.is_cuda() &&
                  scale_v.is_cuda() && output.is_cuda(),
              "native rank8 epilogue tensors must be CUDA tensors");
  TORCH_CHECK(hidden.scalar_type() == at::kHalf && rank8_b.scalar_type() == at::kHalf &&
                  base.scalar_type() == at::kFloat && scale_v.scalar_type() == at::kFloat &&
                  output.scalar_type() == at::kHalf,
              "native rank8 epilogue tensor dtypes are invalid");
  TORCH_CHECK(hidden.dim() == 2 && hidden.size(1) == 8 && rank8_b.dim() == 2 &&
                  rank8_b.size(0) == 8 && rank8_b.size(1) == base.size(1) &&
                  base.dim() == 2 && scale_v.numel() == base.size(1) &&
                  output.sizes() == base.sizes(),
              "native rank8 epilogue tensor shapes are invalid");
  TORCH_CHECK(hidden.is_contiguous() && rank8_b.is_contiguous() &&
                  scale_v.is_contiguous() && output.is_contiguous(),
              "native rank8 epilogue tensors must be contiguous");
  TORCH_CHECK(!bias.defined() ||
                  (bias.scalar_type() == at::kFloat && bias.numel() == base.size(1) &&
                   bias.is_contiguous()),
              "native rank8 epilogue bias must be contiguous FP32 or absent");
  const int m = static_cast<int>(base.size(0));
  const int n = static_cast<int>(base.size(1));
  const dim3 grid(1, static_cast<unsigned>((m + 3) / 4), 1);
  qvq_rank8_epilogue_no_hadamard_kernel<<<grid, 128, 0, stream>>>(
      hidden.data_ptr<at::Half>(), rank8_b.data_ptr<at::Half>(), base.data_ptr<float>(),
      base.stride(0), scale_v.data_ptr<float>(),
      bias.defined() ? bias.data_ptr<float>() : nullptr,
      output.data_ptr<at::Half>(), m, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

extern "C" void qvq_rank8_epilogue_hadamard(
    const at::Tensor& hidden,
    const at::Tensor& rank8_b,
    const at::Tensor& base,
    const at::Tensor& scale_v,
    const at::Tensor& bias,
    at::Tensor& output,
    cudaStream_t stream) {
  TORCH_CHECK(hidden.is_cuda() && rank8_b.is_cuda() && base.is_cuda() &&
                  scale_v.is_cuda() && output.is_cuda(),
              "native rank8 Hadamard epilogue tensors must be CUDA tensors");
  TORCH_CHECK(hidden.scalar_type() == at::kHalf && rank8_b.scalar_type() == at::kHalf &&
                  base.scalar_type() == at::kFloat && scale_v.scalar_type() == at::kFloat &&
                  output.scalar_type() == at::kHalf,
              "native rank8 Hadamard epilogue tensor dtypes are invalid");
  TORCH_CHECK(hidden.dim() == 2 && hidden.size(1) == 8 && rank8_b.dim() == 2 &&
                  rank8_b.size(0) == 8 && rank8_b.size(1) == base.size(1) &&
                  base.dim() == 2 && scale_v.numel() == base.size(1) &&
                  output.sizes() == base.sizes(),
              "native rank8 Hadamard epilogue tensor shapes are invalid");
  TORCH_CHECK(base.size(1) >= 16 && base.size(1) <= 16384 &&
                  (base.size(1) & (base.size(1) - 1)) == 0,
              "native rank8 Hadamard epilogue requires power-of-two N in [16,16384]");
  TORCH_CHECK(hidden.is_contiguous() && rank8_b.is_contiguous() &&
                  scale_v.is_contiguous() && output.is_contiguous(),
              "native rank8 Hadamard epilogue tensors must be contiguous");
  TORCH_CHECK(!bias.defined() ||
                  (bias.scalar_type() == at::kFloat && bias.numel() == base.size(1) &&
                   bias.is_contiguous()),
              "native rank8 Hadamard epilogue bias must be contiguous FP32 or absent");
  const int m = static_cast<int>(base.size(0));
  const int n = static_cast<int>(base.size(1));
  const size_t shared_bytes = static_cast<size_t>(n + n / 32) * sizeof(float);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      qvq_rank8_epilogue_hadamard_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_bytes)));
  const dim3 grid(static_cast<unsigned>(m), 1, 1);
  qvq_rank8_epilogue_hadamard_kernel<<<grid, 1024, shared_bytes, stream>>>(
      hidden.data_ptr<at::Half>(), rank8_b.data_ptr<at::Half>(), base.data_ptr<float>(),
      base.stride(0), scale_v.data_ptr<float>(),
      bias.defined() ? bias.data_ptr<float>() : nullptr,
      output.data_ptr<at::Half>(), m, n, n >= 2048);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
