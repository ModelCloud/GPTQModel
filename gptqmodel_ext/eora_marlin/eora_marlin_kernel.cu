// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_runtime.h>
#include <torch/types.h>

namespace {

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void lora_up_add_kernel(const scalar_t* __restrict__ down,
                                   const scalar_t* __restrict__ up,
                                   scalar_t* __restrict__ out,
                                   int64_t rows,
                                   int64_t cols,
                                   int64_t rank) {
  const int64_t total = rows * cols;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total;
       linear += stride) {
    const int64_t row = linear / cols;
    const int64_t col = linear - row * cols;
    const scalar_t* down_row = down + row * rank;

    float acc = 0.0f;
    for (int64_t r = 0; r < rank; ++r) {
      acc += scalar_to_float(down_row[r]) * scalar_to_float(up[r * cols + col]);
    }

    out[linear] = float_to_scalar<scalar_t>(scalar_to_float(out[linear]) + acc);
  }
}

void validate_lora_up_add_inputs(const torch::Tensor& down,
                                 const torch::Tensor& up,
                                 const torch::Tensor& out) {
  TORCH_CHECK(down.is_cuda(), "down must be a CUDA tensor");
  TORCH_CHECK(up.is_cuda(), "up must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(down.is_contiguous(), "down must be contiguous");
  TORCH_CHECK(up.is_contiguous(), "up must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(down.dim() == 2, "down must have shape [rows, rank]");
  TORCH_CHECK(up.dim() == 2, "up must have shape [rank, cols]");
  TORCH_CHECK(out.dim() == 2, "out must have shape [rows, cols]");
  TORCH_CHECK(down.scalar_type() == up.scalar_type(),
              "down and up must have the same dtype");
  TORCH_CHECK(down.scalar_type() == out.scalar_type(),
              "down and out must have the same dtype");
  TORCH_CHECK(down.scalar_type() == at::ScalarType::Half ||
                  down.scalar_type() == at::ScalarType::BFloat16,
              "lora_up_add only supports float16 and bfloat16");
  TORCH_CHECK(down.size(1) == up.size(0),
              "shape mismatch: down.shape[1] must equal up.shape[0]");
  TORCH_CHECK(down.size(0) == out.size(0),
              "shape mismatch: down.shape[0] must equal out.shape[0]");
  TORCH_CHECK(up.size(1) == out.size(1),
              "shape mismatch: up.shape[1] must equal out.shape[1]");
}

}  // namespace

torch::Tensor eora_marlin_lora_up_add_cuda(torch::Tensor down,
                                           torch::Tensor up,
                                           torch::Tensor out) {
  validate_lora_up_add_inputs(down, up, out);

  const c10::cuda::OptionalCUDAGuard device_guard(at::device_of(out));
  const int64_t rows = out.size(0);
  const int64_t cols = out.size(1);
  const int64_t rank = down.size(1);
  if (rows == 0 || cols == 0 || rank == 0) {
    return out;
  }

  constexpr int threads = 256;
  int64_t blocks64 = (rows * cols + threads - 1) / threads;
  blocks64 = std::min<int64_t>(blocks64, 65535);
  const dim3 blocks(static_cast<unsigned int>(blocks64));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(out.get_device());

  if (out.scalar_type() == at::ScalarType::Half) {
    lora_up_add_kernel<at::Half><<<blocks, threads, 0, stream>>>(
        down.data_ptr<at::Half>(),
        up.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        rows,
        cols,
        rank);
  } else {
    lora_up_add_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
        down.data_ptr<at::BFloat16>(),
        up.data_ptr<at::BFloat16>(),
        out.data_ptr<at::BFloat16>(),
        rows,
        cols,
        rank);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
