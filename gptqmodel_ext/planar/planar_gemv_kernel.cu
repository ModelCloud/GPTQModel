// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
// SPDX-License-Identifier: Apache-2.0
// Contact: qubitium@modelcloud.ai, x.com/qubitium

// Native decode-regime GEMV for the planar (gptq_p) GPTQ format at 3/5/6/7
// bits. Register-level decode: each 32-code block loads its `bits` packed
// words once into registers and derives all 32 codes with compile-time
// shifts/masks, so weight-side DRAM traffic is only the packed words.
//
// Layout contract (docs/gptq_planar.md):
// - qweight [K/32*bits, N] int32: every 32 logical rows use `bits` adjacent
//   word rows, low plane first; no code crosses a word boundary.
// - qzeros [G, N/32*bits] int32: same plane scheme along output columns.
// - scales [G, N] fp16/bf16, zeros use v2 semantics (no +1 bias).
// - g_idx must map each 32-row block to a single group (checked in Python;
//   the block group is read from g_idx[row0]).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/types.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>

namespace pangolin {

constexpr int kWarpSize = 32;
constexpr int kBlockN = 32;     // minimum column granularity (one qzeros column block)
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * kWarpSize;
constexpr int kMaxM = 32;       // decode-regime rows handled per launch

template <typename Scalar>
struct ScalarTraits;

template <>
struct ScalarTraits<half> {
  static __device__ __forceinline__ float to_float(half value) { return __half2float(value); }
  static __device__ __forceinline__ half from_float(float value) { return __float2half_rn(value); }
};

template <>
struct ScalarTraits<nv_bfloat16> {
  static __device__ __forceinline__ float to_float(nv_bfloat16 value) { return __bfloat162float(value); }
  static __device__ __forceinline__ nv_bfloat16 from_float(float value) { return __float2bfloat16_rn(value); }
};

// Plane widths per bit-width: bits = W0 + W1 + W2, planes packed low-first.
template <int Bits>
struct PlaneSpec;

template <>
struct PlaneSpec<3> {
  static constexpr int kW0 = 2, kW1 = 1, kW2 = 0;
};
template <>
struct PlaneSpec<5> {
  static constexpr int kW0 = 4, kW1 = 1, kW2 = 0;
};
template <>
struct PlaneSpec<6> {
  static constexpr int kW0 = 4, kW1 = 2, kW2 = 0;
};
template <>
struct PlaneSpec<7> {
  static constexpr int kW0 = 4, kW1 = 2, kW2 = 1;
};

template <int Width>
__device__ __forceinline__ int plane_code(const uint32_t* words, int k) {
  constexpr int kPackFactor = 32 / Width;
  constexpr uint32_t kMask = (1u << Width) - 1u;
  return static_cast<int>((words[k / kPackFactor] >> (Width * (k % kPackFactor))) & kMask);
}

// Decode the logical code for lane-column `r` of one qzeros column block.
template <int Bits>
__device__ __forceinline__ int decode_zero(const int32_t* qzeros_block, int r) {
  using Spec = PlaneSpec<Bits>;
  const uint32_t* words = reinterpret_cast<const uint32_t*>(qzeros_block);
  int code = plane_code<Spec::kW0>(words, r);
  code |= plane_code<Spec::kW1>(words + Spec::kW0, r) << Spec::kW0;
  if constexpr (Spec::kW2 > 0) {
    code |= plane_code<Spec::kW2>(words + Spec::kW0 + Spec::kW1, r) << (Spec::kW0 + Spec::kW1);
  }
  return code;
}

// One block owns kWarpSize * ColsPerLane output columns and a strided set of
// 32-row k-blocks. Each warp decodes whole k-blocks (`bits` packed word loads
// per owned column, reused for all 32 codes in registers); ColsPerLane == 2
// loads adjacent column pairs as one 8-byte `uint2` and shares each shuffled
// activation across both columns, halving shuffle traffic per output. Warp
// partials reduce through shared memory and each split-K block writes its
// partial to its own fp32 workspace slice (no atomics, no zero-init needed).
// The last block to finish a column block sums the slices and converts to the
// output dtype in-kernel, so the whole op is a single kernel launch plus one
// tiny counter memset.
template <typename Scalar, int Bits, int SizeM, int ColsPerLane>
__global__ __launch_bounds__(kThreads) void planar_gemv_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ qweight,
    const Scalar* __restrict__ scales,
    const int32_t* __restrict__ qzeros,
    const int32_t* __restrict__ g_idx,
    Scalar* __restrict__ output,
    float* __restrict__ workspace,
    int* __restrict__ counters,
    int size_k,
    int size_n,
    int num_groups) {
  using Spec = PlaneSpec<Bits>;
  constexpr int kBlockCols = kWarpSize * ColsPerLane;
  constexpr int kPartialCols = kBlockCols;
  // Dynamic shared-memory partials sized at launch to allow large SizeM.
  extern __shared__ float partials[];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  const int column_block = blockIdx.x;
  const int n0 = column_block * kBlockCols + lane * ColsPerLane;
  const int num_k_blocks = size_k / 32;
  const int qzeros_stride = (size_n / 32) * Bits;

  float acc[SizeM][ColsPerLane];
#pragma unroll
  for (int m = 0; m < SizeM; ++m) {
#pragma unroll
    for (int c = 0; c < ColsPerLane; ++c) {
      acc[m][c] = 0.0f;
    }
  }

  for (int blk = blockIdx.y * kWarpsPerBlock + warp; blk < num_k_blocks;
       blk += gridDim.y * kWarpsPerBlock) {
    const int row0 = blk * 32;
    int group = g_idx[row0];
    if (group < 0) {
      group += num_groups;
    }

    // Fold the zero point into the scale so each code costs one FMA:
    // (code - zero) * scale == code * scale - zero * scale.
    float scale[ColsPerLane];
    float zero_scale[ColsPerLane];
#pragma unroll
    for (int c = 0; c < ColsPerLane; ++c) {
      const int n = n0 + c;
      scale[c] = ScalarTraits<Scalar>::to_float(
          scales[static_cast<int64_t>(group) * size_n + n]);
      const int zero = decode_zero<Bits>(
          qzeros + static_cast<int64_t>(group) * qzeros_stride + (n / 32) * Bits, n % 32);
      zero_scale[c] = static_cast<float>(zero) * scale[c];
    }

    uint32_t words[ColsPerLane][Bits];
    const int32_t* qweight_block = qweight + static_cast<int64_t>(blk) * Bits * size_n + n0;
#pragma unroll
    for (int w = 0; w < Bits; ++w) {
      if constexpr (ColsPerLane == 2) {
        // n0 is even and size_n % 64 == 0, so the pair load is 8-byte aligned.
        const uint2 pair = *reinterpret_cast<const uint2*>(
            qweight_block + static_cast<int64_t>(w) * size_n);
        words[0][w] = pair.x;
        words[1][w] = pair.y;
      } else {
        words[0][w] = static_cast<uint32_t>(qweight_block[static_cast<int64_t>(w) * size_n]);
      }
    }

    // Every lane needs all 32 activations of the block: each lane loads one
    // (coalesced) and the unrolled loop broadcasts them with warp shuffles.
    float x_lane[SizeM];
#pragma unroll
    for (int m = 0; m < SizeM; ++m) {
      x_lane[m] = ScalarTraits<Scalar>::to_float(
          input[static_cast<int64_t>(m) * size_k + row0 + lane]);
    }

#pragma unroll
    for (int k = 0; k < 32; ++k) {
      float weight[ColsPerLane];
#pragma unroll
      for (int c = 0; c < ColsPerLane; ++c) {
        int code = plane_code<Spec::kW0>(words[c], k);
        code |= plane_code<Spec::kW1>(words[c] + Spec::kW0, k) << Spec::kW0;
        if constexpr (Spec::kW2 > 0) {
          code |= plane_code<Spec::kW2>(words[c] + Spec::kW0 + Spec::kW1, k)
                  << (Spec::kW0 + Spec::kW1);
        }
        weight[c] = __fmaf_rn(static_cast<float>(code), scale[c], -zero_scale[c]);
      }
#pragma unroll
      for (int m = 0; m < SizeM; ++m) {
        const float activation = __shfl_sync(0xffffffffu, x_lane[m], k);
#pragma unroll
        for (int c = 0; c < ColsPerLane; ++c) {
          acc[m][c] = __fmaf_rn(activation, weight[c], acc[m][c]);
        }
      }
    }
  }

#pragma unroll
  for (int m = 0; m < SizeM; ++m) {
#pragma unroll
    for (int c = 0; c < ColsPerLane; ++c) {
      partials[((warp * SizeM + m) * kPartialCols) + (lane * ColsPerLane + c)] = acc[m][c];
    }
  }
  __syncthreads();

  if (warp != 0) {
    return;
  }

  if (gridDim.y == 1) {
#pragma unroll
    for (int m = 0; m < SizeM; ++m) {
#pragma unroll
      for (int c = 0; c < ColsPerLane; ++c) {
        const int col = lane * ColsPerLane + c;
        float value = partials[((0 * SizeM + m) * kPartialCols) + col];
#pragma unroll
        for (int w = 1; w < kWarpsPerBlock; ++w) {
          value += partials[((w * SizeM + m) * kPartialCols) + col];
        }
        output[static_cast<int64_t>(m) * size_n + n0 + c] = ScalarTraits<Scalar>::from_float(value);
      }
    }
    return;
  }

  // Workspace layout: [split_k, SizeM, N]; each split-K block owns one slice
  // so plain stores replace global atomics and no zero-init pass is needed.
#pragma unroll
  for (int m = 0; m < SizeM; ++m) {
#pragma unroll
    for (int c = 0; c < ColsPerLane; ++c) {
      const int col = lane * ColsPerLane + c;
      float value = partials[((0 * SizeM + m) * kPartialCols) + col];
#pragma unroll
      for (int w = 1; w < kWarpsPerBlock; ++w) {
        value += partials[((w * SizeM + m) * kPartialCols) + col];
      }
      workspace[(static_cast<int64_t>(blockIdx.y) * SizeM + m) * size_n + n0 + c] = value;
    }
  }

  // Last split-K block for this column block sums the slices and converts to
  // the output dtype (release/acquire via threadfence + atomic counter).
  __threadfence();
  __shared__ int is_last;
  if (lane == 0) {
    is_last = (atomicAdd(counters + column_block, 1) == static_cast<int>(gridDim.y) - 1);
  }
  __syncwarp();
  if (is_last) {
#pragma unroll
    for (int m = 0; m < SizeM; ++m) {
#pragma unroll
      for (int c = 0; c < ColsPerLane; ++c) {
        float value = 0.0f;
        for (int y = 0; y < static_cast<int>(gridDim.y); ++y) {
          value += workspace[(static_cast<int64_t>(y) * SizeM + m) * size_n + n0 + c];
        }
        output[static_cast<int64_t>(m) * size_n + n0 + c] = ScalarTraits<Scalar>::from_float(value);
      }
    }
  }
}

template <typename Scalar, int Bits, int SizeM, int ColsPerLane>
int launch_cols(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& qzeros,
    const torch::Tensor& g_idx,
    torch::Tensor& output,
    float* workspace,
    int* counters,
    int size_k,
    int size_n,
    int num_groups,
    int split_cap,
    int sm_count,
    cudaStream_t stream,
    bool dry_run = false) {
  // Size split-K from this instantiation's real occupancy so the grid fills
  // exactly the resident-block wave (a partial second wave runs alone and
  // stretches the whole launch).
  const int column_blocks = size_n / (kWarpSize * ColsPerLane);
  // Dynamic shared memory for the partials reduction; must be requested before
  // occupancy/launch when it exceeds the default 48 KiB per block.
  constexpr int kPartialBytes =
      pangolin::kWarpsPerBlock * SizeM * (pangolin::kWarpSize * ColsPerLane) * sizeof(float);
  auto kernel = planar_gemv_kernel<Scalar, Bits, SizeM, ColsPerLane>;
  // Occupancy is fixed per (instantiation, device); cache it so the decode
  // path does not pay a driver query on every launch. The shared-memory
  // attribute is set inside the same cache miss because it is a per-function,
  // per-device sticky property.
  constexpr int kMaxDevices = 64;
  static std::array<std::atomic<int>, kMaxDevices> occupancy_cache{};
  int device = 0;
  cudaGetDevice(&device);
  int blocks_per_sm = 0;
  if (device >= 0 && device < kMaxDevices) {
    blocks_per_sm = occupancy_cache[static_cast<size_t>(device)].load(std::memory_order_relaxed);
  }
  if (blocks_per_sm == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kPartialBytes));
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, kThreads, static_cast<size_t>(kPartialBytes));
    blocks_per_sm = std::max(1, blocks_per_sm);
    if (device >= 0 && device < kMaxDevices) {
      occupancy_cache[static_cast<size_t>(device)].store(blocks_per_sm, std::memory_order_relaxed);
    }
  }
  const int wave_slots = std::max(1, blocks_per_sm) * sm_count;
  const int wanted = std::max(1, wave_slots / column_blocks);
  const int split_k = std::min(split_cap, wanted);
  if (dry_run) {
    return split_k;
  }
  const dim3 grid(column_blocks, split_k);
  kernel<<<grid, kThreads, kPartialBytes, stream>>>(
      reinterpret_cast<const Scalar*>(input.const_data_ptr()),
      qweight.const_data_ptr<int32_t>(),
      reinterpret_cast<const Scalar*>(scales.const_data_ptr()),
      qzeros.const_data_ptr<int32_t>(),
      g_idx.const_data_ptr<int32_t>(),
      reinterpret_cast<Scalar*>(output.mutable_data_ptr()),
      workspace,
      counters,
      size_k,
      size_n,
      num_groups);
  return split_k;
}

template <typename Scalar, int Bits, int SizeM>
int launch_size_m(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& qzeros,
    const torch::Tensor& g_idx,
    torch::Tensor& output,
    float* workspace,
    int* counters,
    int size_k,
    int size_n,
    int num_groups,
    int split_cap,
    int sm_count,
    cudaStream_t stream,
    bool dry_run = false) {
  // Adjacent column pairs load as one 8-byte word when N allows it: fewer,
  // wider weight loads and one shuffle broadcast feeding two output columns.
  if (size_n % (kWarpSize * 2) == 0) {
    return launch_cols<Scalar, Bits, SizeM, 2>(
        input, qweight, scales, qzeros, g_idx, output, workspace, counters,
        size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
  } else {
    return launch_cols<Scalar, Bits, SizeM, 1>(
        input, qweight, scales, qzeros, g_idx, output, workspace, counters,
        size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
  }
}

template <typename Scalar, int Bits>
int launch_bits(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& qzeros,
    const torch::Tensor& g_idx,
    torch::Tensor& output,
    float* workspace,
    int* counters,
    int size_m,
    int size_k,
    int size_n,
    int num_groups,
    int split_cap,
    int sm_count,
    cudaStream_t stream,
    bool dry_run = false) {
  switch (size_m) {
    case 1:
      return launch_size_m<Scalar, Bits, 1>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 2:
      return launch_size_m<Scalar, Bits, 2>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 3:
      return launch_size_m<Scalar, Bits, 3>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 4:
      return launch_size_m<Scalar, Bits, 4>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 5:
      return launch_size_m<Scalar, Bits, 5>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 6:
      return launch_size_m<Scalar, Bits, 6>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 7:
      return launch_size_m<Scalar, Bits, 7>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 8:
      return launch_size_m<Scalar, Bits, 8>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 16:
      return launch_size_m<Scalar, Bits, 16>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 32:
      return launch_size_m<Scalar, Bits, 32>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    default:
      TORCH_CHECK(false, "pangolin gemv supports 1..", kMaxM, " input rows, got ", size_m);
  }
}

template <typename Scalar>
int launch_scalar(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& qzeros,
    const torch::Tensor& g_idx,
    torch::Tensor& output,
    float* workspace,
    int* counters,
    int64_t bits,
    int size_m,
    int size_k,
    int size_n,
    int num_groups,
    int split_cap,
    int sm_count,
    cudaStream_t stream,
    bool dry_run = false) {
  switch (bits) {
    case 3:
      return launch_bits<Scalar, 3>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_m, size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 5:
      return launch_bits<Scalar, 5>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_m, size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 6:
      return launch_bits<Scalar, 6>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_m, size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 7:
      return launch_bits<Scalar, 7>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_m, size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    default:
      TORCH_CHECK(false, "pangolin gemv supports bits 3/5/6/7, got ", bits);
  }
}

}  // namespace pangolin

torch::Tensor pangolin_gemv_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor g_idx,
    int64_t bits) {
  TORCH_CHECK(input.is_cuda(), "pangolin input must be CUDA");
  TORCH_CHECK(
      qweight.is_cuda() && scales.is_cuda() && qzeros.is_cuda() && g_idx.is_cuda(),
      "pangolin weight tensors must be CUDA");
  TORCH_CHECK(
      input.device() == qweight.device() && input.device() == scales.device() &&
          input.device() == qzeros.device() && input.device() == g_idx.device(),
      "pangolin tensors must be on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "pangolin input must be FP16 or BF16");
  TORCH_CHECK(qweight.scalar_type() == at::kInt, "pangolin qweight must be int32");
  TORCH_CHECK(qzeros.scalar_type() == at::kInt, "pangolin qzeros must be int32");
  TORCH_CHECK(g_idx.scalar_type() == at::kInt, "pangolin g_idx must be int32");
  TORCH_CHECK(scales.scalar_type() == input.scalar_type(), "pangolin scales dtype must match input dtype");
  TORCH_CHECK(
      input.dim() == 2 && qweight.dim() == 2 && scales.dim() == 2 && qzeros.dim() == 2 && g_idx.dim() == 1,
      "pangolin expects input[M,K], qweight/scales/qzeros 2D, g_idx 1D");
  TORCH_CHECK(
      input.is_contiguous() && qweight.is_contiguous() && scales.is_contiguous() &&
          qzeros.is_contiguous() && g_idx.is_contiguous(),
      "pangolin tensors must be contiguous");
  TORCH_CHECK(bits == 3 || bits == 5 || bits == 6 || bits == 7, "pangolin supports bits 3/5/6/7");

  const int64_t size_m = input.size(0);
  const int64_t size_k = input.size(1);
  const int64_t size_n = scales.size(1);
  const int64_t num_groups = scales.size(0);
  constexpr std::array<int64_t, 10> kSupportedM = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32};
  TORCH_CHECK(
      std::find(kSupportedM.begin(), kSupportedM.end(), size_m) != kSupportedM.end(),
      "pangolin gemv supports M in {1,2,3,4,5,6,7,8,16,32}, got ", size_m);
  TORCH_CHECK(size_k % 32 == 0 && size_n % 32 == 0, "pangolin K and N must be divisible by 32");
  TORCH_CHECK(
      qweight.size(0) == (size_k / 32) * bits && qweight.size(1) == size_n,
      "pangolin qweight must have planar shape [K/32*bits, N]");
  TORCH_CHECK(
      qzeros.size(0) == num_groups && qzeros.size(1) == (size_n / 32) * bits,
      "pangolin qzeros must have planar shape [groups, N/32*bits]");
  TORCH_CHECK(g_idx.size(0) == size_k, "pangolin g_idx must have K entries");
  TORCH_CHECK(
      size_k <= std::numeric_limits<int>::max() && size_n <= std::numeric_limits<int>::max(),
      "pangolin dimensions exceed int32 kernel indexing limits");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());

  // Deep split-K supplies SM-level parallelism: the column blocks alone
  // underfill large devices at decode shapes. The exact split is chosen at
  // launch (per-instantiation occupancy); here only cap it so every warp
  // still owns whole k-blocks.
  const int num_k_blocks = static_cast<int>(size_k / 32);
  // Counters are sized for the finest column granularity (32 columns per block)
  // because the launch may pick ColsPerLane=1 for some SizeM/N combinations.
  const int column_blocks = static_cast<int>(size_n / pangolin::kBlockN);
  // 64 bounds the transient fp32 workspace; the occupancy-derived split
  // never usefully exceeds one wave of blocks per column block anyway.
  const int split_cap = std::min(64, std::max(1, num_k_blocks / pangolin::kWarpsPerBlock));
  const int sm_count = properties->multiProcessorCount;

  auto output = torch::empty({size_m, size_n}, input.options());

  // Sizing-only pass: compute the actual split_k for this instantiation so
  // the workspace is sized exactly. This also warms the per-device occupancy
  // cache and sets the dynamic-shared-memory attribute.
  int split_k = 1;
  if (input.scalar_type() == at::kHalf) {
    split_k = pangolin::launch_scalar<half>(
        input, qweight, scales, qzeros, g_idx, output, nullptr, nullptr, bits,
        static_cast<int>(size_m), static_cast<int>(size_k), static_cast<int>(size_n),
        static_cast<int>(num_groups), split_cap, sm_count, stream, true);
  } else {
    split_k = pangolin::launch_scalar<nv_bfloat16>(
        input, qweight, scales, qzeros, g_idx, output, nullptr, nullptr, bits,
        static_cast<int>(size_m), static_cast<int>(size_k), static_cast<int>(size_n),
        static_cast<int>(num_groups), split_cap, sm_count, stream, true);
  }

  // Split-K partials land in per-slice fp32 workspace (written, not
  // accumulated, so only the tiny counter tail needs zeroing) and the last
  // block reduces + converts in-kernel: one gemv launch, no zero/convert
  // kernels on the output tensor.
  torch::Tensor scratch;
  float* workspace = nullptr;
  int* counters = nullptr;
  if (split_k > 1) {
    const int64_t workspace_elems = static_cast<int64_t>(split_k) * size_m * size_n;
    scratch = torch::empty(
        {workspace_elems + column_blocks}, input.options().dtype(at::kFloat));
    workspace = scratch.mutable_data_ptr<float>();
    counters = reinterpret_cast<int*>(workspace + workspace_elems);
    C10_CUDA_CHECK(cudaMemsetAsync(counters, 0, sizeof(int) * column_blocks, stream));
  }

  if (input.scalar_type() == at::kHalf) {
    pangolin::launch_scalar<half>(
        input, qweight, scales, qzeros, g_idx, output, workspace, counters, bits,
        static_cast<int>(size_m), static_cast<int>(size_k), static_cast<int>(size_n),
        static_cast<int>(num_groups), split_k, sm_count, stream, false);
  } else {
    pangolin::launch_scalar<nv_bfloat16>(
        input, qweight, scales, qzeros, g_idx, output, workspace, counters, bits,
        static_cast<int>(size_m), static_cast<int>(size_k), static_cast<int>(size_n),
        static_cast<int>(num_groups), split_k, sm_count, stream, false);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}
