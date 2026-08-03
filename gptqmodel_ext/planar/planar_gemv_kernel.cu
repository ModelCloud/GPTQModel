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
#include <cstdint>
#include <limits>

namespace pangolin {

constexpr int kWarpSize = 32;
constexpr int kBlockN = 32;     // columns per block == one qzeros column block
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * kWarpSize;
constexpr int kMaxM = 8;        // decode-regime rows handled per launch

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

// One block owns 32 output columns and a strided set of 32-row k-blocks.
// Each warp decodes whole k-blocks (`bits` word loads per lane, reused for
// all 32 codes in registers); warp partials reduce through shared memory and
// each split-K block writes its partial to its own fp32 workspace slice (no
// atomics, no zero-init needed). The last block to finish a column block
// sums the slices and converts to the output dtype in-kernel, so the whole
// op is a single kernel launch plus one tiny counter memset.
template <typename Scalar, int Bits, int SizeM>
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
  __shared__ float partials[kWarpsPerBlock][SizeM][kBlockN];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  const int column_block = blockIdx.x;
  const int global_n = column_block * kBlockN + lane;
  const int num_k_blocks = size_k / 32;
  const int qzeros_stride = (size_n / 32) * Bits;

  float acc[SizeM];
#pragma unroll
  for (int m = 0; m < SizeM; ++m) {
    acc[m] = 0.0f;
  }

  for (int blk = blockIdx.y * kWarpsPerBlock + warp; blk < num_k_blocks;
       blk += gridDim.y * kWarpsPerBlock) {
    const int row0 = blk * 32;
    int group = g_idx[row0];
    if (group < 0) {
      group += num_groups;
    }

    const float scale = ScalarTraits<Scalar>::to_float(
        scales[static_cast<int64_t>(group) * size_n + global_n]);
    const int zero = decode_zero<Bits>(
        qzeros + static_cast<int64_t>(group) * qzeros_stride + column_block * Bits, lane);

    uint32_t words[Bits];
    const int32_t* qweight_block = qweight + static_cast<int64_t>(blk) * Bits * size_n + global_n;
#pragma unroll
    for (int w = 0; w < Bits; ++w) {
      words[w] = static_cast<uint32_t>(qweight_block[static_cast<int64_t>(w) * size_n]);
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
      int code = plane_code<Spec::kW0>(words, k);
      code |= plane_code<Spec::kW1>(words + Spec::kW0, k) << Spec::kW0;
      if constexpr (Spec::kW2 > 0) {
        code |= plane_code<Spec::kW2>(words + Spec::kW0 + Spec::kW1, k) << (Spec::kW0 + Spec::kW1);
      }
      const float weight = static_cast<float>(code - zero) * scale;
#pragma unroll
      for (int m = 0; m < SizeM; ++m) {
        const float activation = __shfl_sync(0xffffffffu, x_lane[m], k);
        acc[m] = __fmaf_rn(activation, weight, acc[m]);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < SizeM; ++m) {
    partials[warp][m][lane] = acc[m];
  }
  __syncthreads();

  if (warp != 0) {
    return;
  }

  if (gridDim.y == 1) {
#pragma unroll
    for (int m = 0; m < SizeM; ++m) {
      float value = partials[0][m][lane];
#pragma unroll
      for (int w = 1; w < kWarpsPerBlock; ++w) {
        value += partials[w][m][lane];
      }
      output[static_cast<int64_t>(m) * size_n + global_n] = ScalarTraits<Scalar>::from_float(value);
    }
    return;
  }

  // Workspace layout: [split_k, SizeM, N]; each split-K block owns one slice
  // so plain stores replace global atomics and no zero-init pass is needed.
#pragma unroll
  for (int m = 0; m < SizeM; ++m) {
    float value = partials[0][m][lane];
#pragma unroll
    for (int w = 1; w < kWarpsPerBlock; ++w) {
      value += partials[w][m][lane];
    }
    workspace[(static_cast<int64_t>(blockIdx.y) * SizeM + m) * size_n + global_n] = value;
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
      float value = 0.0f;
      for (int y = 0; y < static_cast<int>(gridDim.y); ++y) {
        value += workspace[(static_cast<int64_t>(y) * SizeM + m) * size_n + global_n];
      }
      output[static_cast<int64_t>(m) * size_n + global_n] = ScalarTraits<Scalar>::from_float(value);
    }
  }
}

template <typename Scalar, int Bits, int SizeM>
void launch_size_m(
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
    int split_k,
    cudaStream_t stream) {
  const dim3 grid(size_n / kBlockN, split_k);
  planar_gemv_kernel<Scalar, Bits, SizeM><<<grid, kThreads, 0, stream>>>(
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
}

template <typename Scalar, int Bits>
void launch_bits(
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
    int split_k,
    cudaStream_t stream) {
  switch (size_m) {
    case 1:
      launch_size_m<Scalar, Bits, 1>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_k, stream);
      break;
    case 2:
      launch_size_m<Scalar, Bits, 2>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_k, stream);
      break;
    case 3:
      launch_size_m<Scalar, Bits, 3>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_k, stream);
      break;
    case 4:
      launch_size_m<Scalar, Bits, 4>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_k, stream);
      break;
    case 5:
      launch_size_m<Scalar, Bits, 5>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_k, stream);
      break;
    case 6:
      launch_size_m<Scalar, Bits, 6>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_k, stream);
      break;
    case 7:
      launch_size_m<Scalar, Bits, 7>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_k, stream);
      break;
    case 8:
      launch_size_m<Scalar, Bits, 8>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_k, stream);
      break;
    default:
      TORCH_CHECK(false, "pangolin gemv supports 1..", kMaxM, " input rows, got ", size_m);
  }
}

template <typename Scalar>
void launch_scalar(
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
    int split_k,
    cudaStream_t stream) {
  switch (bits) {
    case 3:
      launch_bits<Scalar, 3>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_m, size_k, size_n, num_groups, split_k, stream);
      break;
    case 5:
      launch_bits<Scalar, 5>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_m, size_k, size_n, num_groups, split_k, stream);
      break;
    case 6:
      launch_bits<Scalar, 6>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_m, size_k, size_n, num_groups, split_k, stream);
      break;
    case 7:
      launch_bits<Scalar, 7>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_m, size_k, size_n, num_groups, split_k, stream);
      break;
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
  TORCH_CHECK(
      size_m >= 1 && size_m <= pangolin::kMaxM,
      "pangolin gemv supports 1..", pangolin::kMaxM, " input rows, got ", size_m);
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

  // Deep split-K supplies SM-level parallelism: N/32 column blocks alone
  // underfill large devices at decode shapes. Cover ~8 blocks per SM, capped
  // by available k-blocks per warp-slot.
  const int num_k_blocks = static_cast<int>(size_k / 32);
  const int column_blocks = static_cast<int>(size_n / pangolin::kBlockN);
  const int max_split = std::max(1, num_k_blocks / pangolin::kWarpsPerBlock);
  const int wanted = static_cast<int>(
      (8L * properties->multiProcessorCount + column_blocks - 1) / column_blocks);
  const int split_k = std::min(max_split, std::max(1, wanted));

  auto output = torch::empty({size_m, size_n}, input.options());

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
        static_cast<int>(num_groups), split_k, stream);
  } else {
    pangolin::launch_scalar<nv_bfloat16>(
        input, qweight, scales, qzeros, g_idx, output, workspace, counters, bits,
        static_cast<int>(size_m), static_cast<int>(size_k), static_cast<int>(size_n),
        static_cast<int>(num_groups), split_k, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}
