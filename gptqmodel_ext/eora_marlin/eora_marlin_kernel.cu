// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <mutex>
#include <torch/types.h>
#include <vector>

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

template <typename scalar_t, int static_rank, bool static_single_row>
__global__ void lora_fused_add_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ down_weight,
    const scalar_t* __restrict__ up_weight, scalar_t* __restrict__ out,
    float* __restrict__ workspace, int64_t rows, int64_t in_features,
    int64_t out_features, int64_t rank, int blocks_per_row,
    int down_blocks, int cols_per_block) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __shared__ float down_partials[256];
  const int64_t kernel_rank = static_rank > 0 ? static_rank : rank;
  const int64_t kernel_rows = static_single_row ? 1 : rows;
  const int64_t global_thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t global_stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = global_thread; index < kernel_rows * kernel_rank;
       index += global_stride) {
    workspace[index] = 0.0f;
  }
  grid.sync();

  const int row = static_single_row ? 0 : blockIdx.x / blocks_per_row;
  const int row_block =
      static_single_row ? blockIdx.x : blockIdx.x - row * blocks_per_row;
  const scalar_t* x_row = x + static_cast<int64_t>(row) * in_features;
  float* down_row = workspace + static_cast<int64_t>(row) * kernel_rank;

  if (row_block < down_blocks && kernel_rank <= blockDim.x) {
    const int lanes_per_rank = blockDim.x / kernel_rank;
    const int active_threads = lanes_per_rank * kernel_rank;
    float partial = 0.0f;
    if constexpr (static_single_row && static_rank == 128) {
      // Pair adjacent lanes per rank. The even and odd lanes in each warp
      // each read one contiguous 32-byte rank segment, while a warp shuffle
      // replaces the shared-memory reduction and block barrier used by the
      // generic path.
      const int r = threadIdx.x >> 1;
      const int lane = threadIdx.x & 1;
      for (int64_t k = static_cast<int64_t>(row_block) * 2 + lane;
           k < in_features; k += static_cast<int64_t>(down_blocks) * 2) {
        partial += scalar_to_float(x_row[k]) *
                   scalar_to_float(down_weight[k * static_rank + r]);
      }
      const float peer = __shfl_down_sync(0xffffffff, partial, 1);
      if (lane == 0) {
        atomicAdd(down_row + r, partial + peer);
      }
    } else {
      if (threadIdx.x < active_threads) {
        const int r = threadIdx.x % kernel_rank;
        const int lane = threadIdx.x / kernel_rank;
        for (int64_t k =
                 static_cast<int64_t>(row_block) * lanes_per_rank + lane;
             k < in_features;
             k += static_cast<int64_t>(down_blocks) * lanes_per_rank) {
          partial += scalar_to_float(x_row[k]) *
                     scalar_to_float(down_weight[k * kernel_rank + r]);
        }
      }
      down_partials[threadIdx.x] = partial;
      __syncthreads();
      if (threadIdx.x < kernel_rank) {
        float block_sum = 0.0f;
        for (int lane = 0; lane < lanes_per_rank; ++lane) {
          block_sum += down_partials[lane * kernel_rank + threadIdx.x];
        }
        atomicAdd(down_row + threadIdx.x, block_sum);
      }
    }
  } else if (row_block < down_blocks) {
    for (int64_t r = threadIdx.x; r < kernel_rank; r += blockDim.x) {
      float partial = 0.0f;
      for (int64_t k = row_block; k < in_features; k += down_blocks) {
        partial += scalar_to_float(x_row[k]) *
                   scalar_to_float(down_weight[k * kernel_rank + r]);
      }
      atomicAdd(down_row + r, partial);
    }
  }
  grid.sync();

  const int64_t col_begin = static_cast<int64_t>(row_block) * cols_per_block;
  const int64_t col_end =
      min(out_features, col_begin + static_cast<int64_t>(cols_per_block));
  if constexpr (static_single_row &&
                (static_rank == 64 || static_rank == 128)) {
    // Give each 32-rank slice a full warp so every up-weight load stays
    // coalesced while independent warps hide the serial GEMV load latency.
    // The down-phase scratch is dead after grid.sync() and can hold the
    // per-warp partials without increasing shared memory.
    constexpr int up_warps = static_rank / 32;
    constexpr int ranks_per_warp = static_rank / up_warps;
    const int up_lane = threadIdx.x & 31;
    const int up_warp = threadIdx.x >> 5;
    const int64_t col = col_begin + up_lane;
    float update = 0.0f;
    if (up_warp < up_warps && col < col_end) {
      const int rank_begin = up_warp * ranks_per_warp;
      for (int r = rank_begin; r < rank_begin + ranks_per_warp; ++r) {
        update +=
            down_row[r] * scalar_to_float(up_weight[r * out_features + col]);
      }
      down_partials[threadIdx.x] = update;
    }
    __syncthreads();
    if (threadIdx.x < 32 && col < col_end) {
      const float low_half =
          down_partials[threadIdx.x] + down_partials[threadIdx.x + 32];
      if constexpr (up_warps == 4) {
        const float high_half =
            down_partials[threadIdx.x + 64] + down_partials[threadIdx.x + 96];
        update = low_half + high_half;
      } else {
        update = low_half;
      }
      out[col] =
          float_to_scalar<scalar_t>(scalar_to_float(out[col]) + update);
    }
  } else {
    for (int64_t col = col_begin + threadIdx.x; col < col_end;
         col += blockDim.x) {
      float update = 0.0f;
      for (int64_t r = 0; r < kernel_rank; ++r) {
        update +=
            down_row[r] * scalar_to_float(up_weight[r * out_features + col]);
      }
      const int64_t out_index = static_cast<int64_t>(row) * out_features + col;
      out[out_index] =
          float_to_scalar<scalar_t>(scalar_to_float(out[out_index]) + update);
    }
  }
}

void validate_lora_up_add_inputs(const torch::Tensor& down,
                                 const torch::Tensor& up,
                                 const torch::Tensor& out) {
  TORCH_CHECK(down.is_cuda(), "down must be a CUDA tensor");
  TORCH_CHECK(up.is_cuda(), "up must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(down.get_device() == up.get_device() &&
                  down.get_device() == out.get_device(),
              "down, up, and out must be on the same CUDA device");
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

void validate_lora_fused_add_inputs(const torch::Tensor& x,
                                    const torch::Tensor& down_weight,
                                    const torch::Tensor& up_weight,
                                    const torch::Tensor& out) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(down_weight.is_cuda(), "down_weight must be a CUDA tensor");
  TORCH_CHECK(up_weight.is_cuda(), "up_weight must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(x.get_device() == down_weight.get_device() &&
                  x.get_device() == up_weight.get_device() &&
                  x.get_device() == out.get_device(),
              "x, down_weight, up_weight, and out must be on the same CUDA "
              "device");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(down_weight.is_contiguous(),
              "down_weight must be contiguous");
  TORCH_CHECK(up_weight.is_contiguous(), "up_weight must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must have shape [rows, in_features]");
  TORCH_CHECK(down_weight.dim() == 2,
              "down_weight must have shape [in_features, rank]");
  TORCH_CHECK(up_weight.dim() == 2,
              "up_weight must have shape [rank, out_features]");
  TORCH_CHECK(out.dim() == 2, "out must have shape [rows, out_features]");
  TORCH_CHECK(x.scalar_type() == down_weight.scalar_type() &&
                  x.scalar_type() == up_weight.scalar_type() &&
                  x.scalar_type() == out.scalar_type(),
              "x, down_weight, up_weight, and out must have the same dtype");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half ||
                  x.scalar_type() == at::ScalarType::BFloat16,
              "lora_fused_add only supports float16 and bfloat16");
  TORCH_CHECK(x.size(1) == down_weight.size(0),
              "shape mismatch: x.shape[1] must equal down_weight.shape[0]");
  TORCH_CHECK(down_weight.size(1) == up_weight.size(0),
              "shape mismatch: down_weight.shape[1] must equal "
              "up_weight.shape[0]");
  TORCH_CHECK(x.size(0) == out.size(0),
              "shape mismatch: x.shape[0] must equal out.shape[0]");
  TORCH_CHECK(up_weight.size(1) == out.size(1),
              "shape mismatch: up_weight.shape[1] must equal out.shape[1]");
}

template <typename scalar_t, int static_rank, bool static_single_row>
int query_lora_fused_max_grid_blocks(int device) {
  int cooperative_launch = 0;
  cudaError_t status = cudaDeviceGetAttribute(
      &cooperative_launch, cudaDevAttrCooperativeLaunch, device);
  TORCH_CHECK(status == cudaSuccess,
              "failed to query cooperative launch support: ",
              cudaGetErrorString(status));
  TORCH_CHECK(cooperative_launch != 0,
              "lora_fused_add requires cooperative CUDA launches");

  int active_blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm,
      lora_fused_add_kernel<scalar_t, static_rank, static_single_row>, 256, 0);
  TORCH_CHECK(status == cudaSuccess,
              "failed to query lora_fused_add occupancy: ",
              cudaGetErrorString(status));
  int sm_count = 0;
  status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                  device);
  TORCH_CHECK(status == cudaSuccess,
              "failed to query CUDA multiprocessor count: ",
              cudaGetErrorString(status));
  return active_blocks_per_sm * sm_count;
}

template <typename scalar_t, int static_rank, bool static_single_row>
int get_lora_fused_max_grid_blocks(int device) {
  static std::mutex mutex;
  static std::vector<int> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (device >= static_cast<int>(cache.size())) {
    cache.resize(device + 1, 0);
  }
  int& max_grid_blocks = cache[device];
  if (max_grid_blocks == 0) {
    max_grid_blocks =
        query_lora_fused_max_grid_blocks<scalar_t, static_rank,
                                         static_single_row>(device);
  }
  return max_grid_blocks;
}

template <typename scalar_t, int static_rank, bool static_single_row>
void launch_lora_fused_add(const torch::Tensor& x,
                           const torch::Tensor& down_weight,
                           const torch::Tensor& up_weight,
                           torch::Tensor& out, torch::Tensor& workspace) {
  constexpr int threads = 256;
  int64_t in_features = x.size(-1);
  int64_t rows = x.numel() / in_features;
  int64_t out_features = out.size(-1);
  const int device = out.get_device();

  const int max_grid_blocks =
      get_lora_fused_max_grid_blocks<scalar_t, static_rank,
                                     static_single_row>(device);
  TORCH_CHECK(rows <= max_grid_blocks,
              "lora_fused_add rows exceed the cooperative grid capacity");
  const bool narrow_rank128 = static_single_row && static_rank == 128 &&
                              in_features >= 2 * out_features;
  const int target_cols_per_block = narrow_rank128 ? 16 : 32;
  int blocks_per_row = std::max(
      1,
      std::min(
          static_cast<int>((out_features + target_cols_per_block - 1) /
                           target_cols_per_block),
          max_grid_blocks / static_cast<int>(rows)));
  constexpr int max_down_blocks =
      static_single_row && static_rank == 128 ? 256 : 128;
  int down_blocks = std::min(blocks_per_row, max_down_blocks);
  int launch_blocks_per_row = blocks_per_row;
  if constexpr (static_single_row && static_rank == 128) {
    if (!narrow_rank128 && in_features == out_features &&
        blocks_per_row >= 128 && blocks_per_row < max_down_blocks) {
      down_blocks = std::min(192, max_grid_blocks);
      launch_blocks_per_row = std::max(blocks_per_row, down_blocks);
    }
  }
  int cols_per_block = static_cast<int>(
      (out_features + blocks_per_row - 1) / blocks_per_row);
  const dim3 blocks(
      static_cast<unsigned int>(rows * launch_blocks_per_row));
  const dim3 block_threads(threads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);

  const scalar_t* x_ptr = x.data_ptr<scalar_t>();
  const scalar_t* down_weight_ptr = down_weight.data_ptr<scalar_t>();
  const scalar_t* up_weight_ptr = up_weight.data_ptr<scalar_t>();
  scalar_t* out_ptr = out.data_ptr<scalar_t>();
  float* workspace_ptr = workspace.data_ptr<float>();
  int64_t rank = down_weight.size(1);
  void* args[] = {&x_ptr,
                  &down_weight_ptr,
                  &up_weight_ptr,
                  &out_ptr,
                  &workspace_ptr,
                  &rows,
                  &in_features,
                  &out_features,
                  &rank,
                  &blocks_per_row,
                  &down_blocks,
                  &cols_per_block};
  cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(
          lora_fused_add_kernel<scalar_t, static_rank, static_single_row>),
      blocks, block_threads, args, 0, stream);
  TORCH_CHECK(status == cudaSuccess, "lora_fused_add launch failed: ",
              cudaGetErrorString(status));
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

// Module post-initialization validates and owns every tensor passed here. Keep
// the fully validated entry point below for direct or externally assembled calls.
torch::Tensor eora_marlin_lora_fused_add_prepared_cuda(
    torch::Tensor x, torch::Tensor down_weight, torch::Tensor up_weight,
    torch::Tensor out, torch::Tensor workspace) {
  const c10::cuda::OptionalCUDAGuard device_guard(at::device_of(out));
  if (x.numel() == 0 || out.size(-1) == 0 || down_weight.size(1) == 0) {
    return out;
  }
  const int64_t rank = down_weight.size(1);
  const bool single_row = out.numel() == out.size(-1);
  if (out.scalar_type() == at::ScalarType::Half) {
    if (rank == 64) {
      if (single_row) {
        launch_lora_fused_add<at::Half, 64, true>(
            x, down_weight, up_weight, out, workspace);
      } else {
        launch_lora_fused_add<at::Half, 64, false>(
            x, down_weight, up_weight, out, workspace);
      }
    } else if (rank == 128) {
      if (single_row) {
        launch_lora_fused_add<at::Half, 128, true>(
            x, down_weight, up_weight, out, workspace);
      } else {
        launch_lora_fused_add<at::Half, 128, false>(
            x, down_weight, up_weight, out, workspace);
      }
    } else {
      launch_lora_fused_add<at::Half, 0, false>(
          x, down_weight, up_weight, out, workspace);
    }
  } else {
    if (rank == 64) {
      if (single_row) {
        launch_lora_fused_add<at::BFloat16, 64, true>(
            x, down_weight, up_weight, out, workspace);
      } else {
        launch_lora_fused_add<at::BFloat16, 64, false>(
            x, down_weight, up_weight, out, workspace);
      }
    } else if (rank == 128) {
      if (single_row) {
        launch_lora_fused_add<at::BFloat16, 128, true>(
            x, down_weight, up_weight, out, workspace);
      } else {
        launch_lora_fused_add<at::BFloat16, 128, false>(
            x, down_weight, up_weight, out, workspace);
      }
    } else {
      launch_lora_fused_add<at::BFloat16, 0, false>(
          x, down_weight, up_weight, out, workspace);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor eora_marlin_lora_fused_add_cuda(
    torch::Tensor x, torch::Tensor down_weight, torch::Tensor up_weight,
    torch::Tensor out, torch::Tensor workspace) {
  validate_lora_fused_add_inputs(x, down_weight, up_weight, out);
  TORCH_CHECK(workspace.is_cuda(), "workspace must be a CUDA tensor");
  TORCH_CHECK(workspace.get_device() == out.get_device(),
              "workspace must be on the same CUDA device as out");
  TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Float,
              "workspace must have dtype float32");
  TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
  TORCH_CHECK(workspace.numel() >= x.size(0) * down_weight.size(1),
              "workspace does not have enough elements for [rows, rank]");
  return eora_marlin_lora_fused_add_prepared_cuda(
      x, down_weight, up_weight, out, workspace);
}
