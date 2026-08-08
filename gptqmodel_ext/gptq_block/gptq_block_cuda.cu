// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kMaxBlockColumns = 128;
constexpr int kWarpSize = 32;
// Four independent rows per block provide enough resident warps to hide the
// serial column dependency without introducing any cross-warp synchronization.
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

__device__ __forceinline__ float clamp_preserve_nan(float value, float lower, float upper) {
  value = value < lower ? lower : value;
  return value > upper ? upper : value;
}

__global__ __launch_bounds__(kThreadsPerBlock, 12) void gptq_block_kernel(
    const float *__restrict__ weights,
    const float *__restrict__ hessian_inverse,
    const float *__restrict__ scale,
    const float *__restrict__ zero,
    int rows,
    int count,
    int groups,
    int group_size,
    float maxq,
    bool groupwise,
    float *__restrict__ quantized,
    float *__restrict__ errors) {
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }

  const int64_t row_offset = static_cast<int64_t>(row) * count;
  const int column0 = lane;
  const int column1 = lane + kWarpSize;
  const int column2 = lane + 2 * kWarpSize;
  const int column3 = lane + 3 * kWarpSize;
  float working0 = column0 < count ? weights[row_offset + column0] : 0.0f;
  float working1 = column1 < count ? weights[row_offset + column1] : 0.0f;
  float working2 = column2 < count ? weights[row_offset + column2] : 0.0f;
  float working3 = column3 < count ? weights[row_offset + column3] : 0.0f;

  // GPTQ's error correction is ordered by column. Every row is independent,
  // but column i+1 must observe the complete correction from column i.
  for (int i = 0; i < count; ++i) {
    const int source_lane = i % kWarpSize;
    float lane_error = 0.0f;
    if (lane == source_lane) {
      const int group = i / group_size;
      const int64_t scale_offset = static_cast<int64_t>(row) * groups + group;
      const float scale_value = scale[scale_offset];
      const float zero_value = zero[scale_offset];
      const int source_slot = i / kWarpSize;
      const float weight_value =
          source_slot == 0 ? working0 : source_slot == 1 ? working1 : source_slot == 2 ? working2 : working3;
      const float ratio = __fdiv_rn(weight_value, scale_value);
      float quantized_integer = nearbyintf(ratio);
      float quantized_value;
      if (groupwise) {
        quantized_integer = clamp_preserve_nan(quantized_integer, -maxq, maxq);
        quantized_value = scale_value * quantized_integer;
      } else {
        quantized_integer = clamp_preserve_nan(quantized_integer + zero_value, 0.0f, maxq);
        quantized_value = scale_value * (quantized_integer - zero_value);
      }

      const int64_t output_offset = static_cast<int64_t>(row) * count + i;
      const float diagonal = hessian_inverse[static_cast<int64_t>(i) * count + i];
      const float error = __fdiv_rn(weight_value - quantized_value, diagonal);
      quantized[output_offset] = quantized_value;
      errors[output_offset] = error;
      lane_error = error;
    }
    const float column_error = __shfl_sync(0xffffffffu, lane_error, source_lane);

    // Each lane owns four fixed columns in registers. This preserves the exact
    // serial column dependency while avoiding shared-memory traffic entirely.
    const int64_t hessian_row = static_cast<int64_t>(i) * count;
    if (column0 > i && column0 < count) {
      const float correction = __fmul_rn(column_error, hessian_inverse[hessian_row + column0]);
      working0 = __fsub_rn(working0, correction);
    }
    if (column1 > i && column1 < count) {
      const float correction = __fmul_rn(column_error, hessian_inverse[hessian_row + column1]);
      working1 = __fsub_rn(working1, correction);
    }
    if (column2 > i && column2 < count) {
      const float correction = __fmul_rn(column_error, hessian_inverse[hessian_row + column2]);
      working2 = __fsub_rn(working2, correction);
    }
    if (column3 > i && column3 < count) {
      const float correction = __fmul_rn(column_error, hessian_inverse[hessian_row + column3]);
      working3 = __fsub_rn(working3, correction);
    }
  }
}

bool tensors_overlap(const at::Tensor &left, const at::Tensor &right) {
  const auto left_begin = reinterpret_cast<uintptr_t>(left.data_ptr());
  const auto right_begin = reinterpret_cast<uintptr_t>(right.data_ptr());
  const auto left_end = left_begin + static_cast<uintptr_t>(left.nbytes());
  const auto right_end = right_begin + static_cast<uintptr_t>(right.nbytes());
  return left_begin < right_end && right_begin < left_end;
}

void gptq_block_cuda(
    const at::Tensor &weights,
    const at::Tensor &hessian_inverse,
    const at::Tensor &scale,
    const at::Tensor &zero,
    int64_t maxq,
    int64_t group_size,
    bool groupwise,
    const at::Tensor &quantized,
    const at::Tensor &errors) {
  TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
  TORCH_CHECK(hessian_inverse.is_cuda(), "hessian_inverse must be a CUDA tensor");
  TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
  TORCH_CHECK(zero.is_cuda(), "zero must be a CUDA tensor");
  TORCH_CHECK(weights.scalar_type() == at::kFloat, "weights must have dtype float32");
  TORCH_CHECK(hessian_inverse.scalar_type() == at::kFloat, "hessian_inverse must have dtype float32");
  TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must have dtype float32");
  TORCH_CHECK(zero.scalar_type() == at::kFloat, "zero must have dtype float32");
  TORCH_CHECK(quantized.is_cuda() && errors.is_cuda(), "outputs must be CUDA tensors");
  TORCH_CHECK(
      quantized.scalar_type() == at::kFloat && errors.scalar_type() == at::kFloat,
      "outputs must have dtype float32");
  TORCH_CHECK(weights.dim() == 2, "weights must be rank two");

  const int64_t rows = weights.size(0);
  const int64_t count = weights.size(1);
  TORCH_CHECK(rows > 0 && count > 0, "weights dimensions must be positive");
  TORCH_CHECK(
      rows <= std::numeric_limits<int>::max(),
      "rows must be no greater than ",
      std::numeric_limits<int>::max());
  TORCH_CHECK(count <= kMaxBlockColumns, "count must be no greater than ", kMaxBlockColumns);
  TORCH_CHECK(group_size > 0 && count % group_size == 0, "group_size must be positive and divide count");
  TORCH_CHECK(maxq > 0, "maxq must be positive");
  TORCH_CHECK(maxq <= 255, "maxq must be no greater than 255");
  TORCH_CHECK(
      hessian_inverse.sizes() == at::IntArrayRef({count, count}),
      "hessian_inverse must be square and match the weight column count");

  const int64_t groups = count / group_size;
  TORCH_CHECK(
      scale.sizes() == at::IntArrayRef({rows, groups}),
      "scale must have shape (rows, count / group_size)");
  TORCH_CHECK(zero.sizes() == scale.sizes(), "zero must match scale shape");
  TORCH_CHECK(quantized.sizes() == weights.sizes(), "quantized output must match weights shape");
  TORCH_CHECK(errors.sizes() == weights.sizes(), "error output must match weights shape");
  TORCH_CHECK(
      weights.device() == hessian_inverse.device() && weights.device() == scale.device() &&
          weights.device() == zero.device() && weights.device() == quantized.device() &&
          weights.device() == errors.device(),
      "all inputs must be on the same CUDA device");
  TORCH_CHECK(
      weights.is_contiguous() && hessian_inverse.is_contiguous() && scale.is_contiguous() && zero.is_contiguous() &&
          quantized.is_contiguous() && errors.is_contiguous(),
      "all inputs and outputs must be contiguous");
  TORCH_CHECK(!tensors_overlap(quantized, errors), "output tensors must not overlap each other");
  TORCH_CHECK(
      !tensors_overlap(quantized, weights) && !tensors_overlap(quantized, hessian_inverse) &&
          !tensors_overlap(quantized, scale) && !tensors_overlap(quantized, zero) &&
          !tensors_overlap(errors, weights) && !tensors_overlap(errors, hessian_inverse) &&
          !tensors_overlap(errors, scale) && !tensors_overlap(errors, zero),
      "output tensors must not overlap any input");

  const c10::cuda::CUDAGuard device_guard(weights.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(weights.get_device());
  const unsigned int blocks = static_cast<unsigned int>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  gptq_block_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      weights.data_ptr<float>(),
      hessian_inverse.data_ptr<float>(),
      scale.data_ptr<float>(),
      zero.data_ptr<float>(),
      static_cast<int>(rows),
      static_cast<int>(count),
      static_cast<int>(groups),
      static_cast<int>(group_size),
      static_cast<float>(maxq),
      groupwise,
      quantized.data_ptr<float>(),
      errors.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(gptqmodel_gptq_block, m) {
  m.def(
      "quantize(Tensor weights, Tensor hessian_inverse, Tensor scale, Tensor zero, int maxq, int group_size, "
      "bool groupwise, Tensor(a!) quantized, Tensor(b!) errors) -> ()");
}

TORCH_LIBRARY_IMPL(gptqmodel_gptq_block, CUDA, m) {
  m.impl("quantize", &gptq_block_cuda);
}
