// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/library.h>

#include <algorithm>
#include <mutex>
#include <optional>
#include <vector>

namespace {

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

constexpr int kBlockN = 64;
constexpr int kTileK = 32;
constexpr int kGemvBlockN = 16;
constexpr int kBiasNone = 0;
constexpr int kBiasHalf = 1;
constexpr int kBiasBFloat16 = 2;

template <typename Scalar>
struct ScalarTraits;

template <>
struct ScalarTraits<half> {
  static constexpr bool kDecodeAdjacentPair = true;

  __device__ static __forceinline__ float to_float(half value) {
    return __half2float(value);
  }

  __device__ static __forceinline__ half from_float(float value) {
    return __float2half_rn(value);
  }

  __device__ static __forceinline__ half scaled_quant(int value, half scale) {
    return __hmul(__int2half_rn(value), scale);
  }

  __device__ static __forceinline__ float2 load2(const half* values) {
    return __half22float2(*reinterpret_cast<const half2*>(values));
  }
};

template <>
struct ScalarTraits<__nv_bfloat16> {
  static constexpr bool kDecodeAdjacentPair = false;

  __device__ static __forceinline__ float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }

  __device__ static __forceinline__ __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
  }

  __device__ static __forceinline__ __nv_bfloat16 scaled_quant(int value, half scale) {
    return __float2bfloat16_rn(static_cast<float>(value) * __half2float(scale));
  }

  __device__ static __forceinline__ float2 load2(const __nv_bfloat16* values) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(values));
  }
};

__device__ __forceinline__ float load_bias(const void* bias, int bias_type, int n) {
  if (bias_type == kBiasHalf) {
    return __half2float(reinterpret_cast<const half*>(bias)[n]);
  }
  if (bias_type == kBiasBFloat16) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(bias)[n]);
  }
  return 0.0f;
}

template <typename Scalar, int BlockM, int GroupSize>
__global__ void trilin_3bit_wmma_kernel(
    const Scalar* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const void* __restrict__ bias,
    Scalar* __restrict__ output,
    float* __restrict__ split_workspace,
    int size_m,
    int size_n,
    int size_k,
    int split_k,
    int bias_type) {
  constexpr int kWarpRows = BlockM / 16;
  constexpr int kWarpCols = kBlockN / 16;
  constexpr int kWarps = kWarpRows * kWarpCols;
  static_assert(kWarps == 4 || kWarps == 8);

  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int warp_m = warp / kWarpCols;
  const int warp_n = warp % kWarpCols;
  const int block_m = blockIdx.y * BlockM;
  const int block_n = blockIdx.x * kBlockN;
  const int split = blockIdx.z;
  const int split_size = size_k / split_k;
  const int begin_k = split * split_size;
  const int end_k = begin_k + split_size;

  extern __shared__ __align__(16) unsigned char shared_raw[];
  Scalar* shared_a = reinterpret_cast<Scalar*>(shared_raw);
  Scalar* shared_b = shared_a + BlockM * kTileK;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, Scalar, wmma::row_major> a_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, Scalar, wmma::row_major> b_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int tile_k = begin_k; tile_k < end_k; tile_k += kTileK) {
    constexpr int scalars_per_vector = sizeof(int4) / sizeof(Scalar);
    constexpr int vectors_per_row = kTileK / scalars_per_vector;
    for (int vector_index = thread; vector_index < BlockM * vectors_per_row; vector_index += blockDim.x) {
      const int local_m = vector_index / vectors_per_row;
      const int local_vector = vector_index % vectors_per_row;
      const int global_m = block_m + local_m;
      int4 value = {0, 0, 0, 0};
      if (global_m < size_m) {
        const Scalar* input_row = input + static_cast<int64_t>(global_m) * size_k + tile_k;
        value = reinterpret_cast<const int4*>(input_row)[local_vector];
      }
      reinterpret_cast<int4*>(shared_a)[vector_index] = value;
    }

    if (thread < kBlockN) {
      const int global_n = block_n + thread;
      uint32_t words[3] = {0, 0, 0};
      half first_scale = __float2half(0.0f);
      half second_scale = __float2half(0.0f);
      if (global_n < size_n) {
        const int packed_k = tile_k / 32 * 3;
        words[0] = qweight[static_cast<int64_t>(packed_k) * size_n + global_n];
        words[1] = qweight[static_cast<int64_t>(packed_k + 1) * size_n + global_n];
        words[2] = qweight[static_cast<int64_t>(packed_k + 2) * size_n + global_n];
        const int scale_group = tile_k / GroupSize;
        first_scale = scales[static_cast<int64_t>(scale_group) * size_n + global_n];
        if constexpr (GroupSize == 16) {
          second_scale = scales[static_cast<int64_t>(scale_group + 1) * size_n + global_n];
        }
      }
#pragma unroll
      for (int local_k = 0; local_k < 32; ++local_k) {
        const int bit = local_k * 3;
        const int word = bit / 32;
        const int shift = bit % 32;
        uint32_t code = words[word] >> shift;
        if (shift > 29) {
          code |= words[word + 1] << (32 - shift);
        }
        const half scale = GroupSize == 16 && local_k >= 16 ? second_scale : first_scale;
        shared_b[local_k * kBlockN + thread] =
            ScalarTraits<Scalar>::scaled_quant(static_cast<int>(code & 0x7) - 4, scale);
      }
    }
    __syncthreads();

#pragma unroll
    for (int sub_k = 0; sub_k < kTileK; sub_k += 16) {
      wmma::load_matrix_sync(
          a_fragment,
          shared_a + warp_m * 16 * kTileK + sub_k,
          kTileK);
      wmma::load_matrix_sync(
          b_fragment,
          shared_b + sub_k * kBlockN + warp_n * 16,
          kBlockN);
      wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    }
    __syncthreads();
  }

  float* shared_c = reinterpret_cast<float*>(shared_raw);
  wmma::store_matrix_sync(
      shared_c + warp_m * 16 * kBlockN + warp_n * 16,
      accumulator,
      kBlockN,
      wmma::mem_row_major);
  __syncthreads();

  for (int index = thread; index < BlockM * kBlockN; index += blockDim.x) {
    const int local_m = index / kBlockN;
    const int local_n = index % kBlockN;
    const int global_m = block_m + local_m;
    const int global_n = block_n + local_n;
    if (global_m >= size_m || global_n >= size_n) {
      continue;
    }
    const int64_t output_index = static_cast<int64_t>(global_m) * size_n + global_n;
    const float value = shared_c[index];
    if (split_k > 1) {
      split_workspace[static_cast<int64_t>(split) * size_m * size_n + output_index] = value;
    } else {
      output[output_index] = ScalarTraits<Scalar>::from_float(value + load_bias(bias, bias_type, global_n));
    }
  }

}

template <typename Scalar, int GroupSize>
__device__ __forceinline__ float trilin_3bit_gemv_accumulate_tile(
    const Scalar* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    int tile_k,
    int global_n,
    int size_n,
    float accumulator) {
  const int packed_k = tile_k / 32 * 3;
  uint32_t words[3];
  words[0] = qweight[static_cast<int64_t>(packed_k) * size_n + global_n];
  words[1] = qweight[static_cast<int64_t>(packed_k + 1) * size_n + global_n];
  words[2] = qweight[static_cast<int64_t>(packed_k + 2) * size_n + global_n];
  const int scale_group = tile_k / GroupSize;
  const float first_scale = __half2float(scales[static_cast<int64_t>(scale_group) * size_n + global_n]);
  float second_scale = first_scale;
  if constexpr (GroupSize == 16) {
    second_scale = __half2float(scales[static_cast<int64_t>(scale_group + 1) * size_n + global_n]);
  }
#pragma unroll
  for (int pair_k = 0; pair_k < 16; ++pair_k) {
    const int first_k = pair_k * 2;
    const int first_bit = first_k * 3;
    const int first_word = first_bit / 32;
    const int first_shift = first_bit % 32;
    uint32_t first_code = words[first_word] >> first_shift;
    if (first_shift > 29) {
      first_code |= words[first_word + 1] << (32 - first_shift);
    }

    const int second_k = first_k + 1;
    const int second_bit = second_k * 3;
    const int second_word = second_bit / 32;
    const int second_shift = second_bit % 32;
    uint32_t second_code = words[second_word] >> second_shift;
    if (second_shift > 29) {
      second_code |= words[second_word + 1] << (32 - second_shift);
    }

    const float2 input_pair = ScalarTraits<Scalar>::load2(input + tile_k + first_k);
    const float float_scale = GroupSize == 16 && pair_k >= 8 ? second_scale : first_scale;
    const float first_weight = static_cast<float>(static_cast<int>(first_code & 0x7) - 4) * float_scale;
    const float second_weight = static_cast<float>(static_cast<int>(second_code & 0x7) - 4) * float_scale;
    accumulator = __fmaf_rn(input_pair.x, first_weight, accumulator);
    accumulator = __fmaf_rn(input_pair.y, second_weight, accumulator);
  }
  return accumulator;
}

template <typename Scalar, int GroupSize>
__global__ void trilin_3bit_gemv_kernel(
    const Scalar* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const void* __restrict__ bias,
    Scalar* __restrict__ output,
    float* __restrict__ split_workspace,
    int size_n,
    int size_k,
    int split_k,
    int bias_type) {
  const int global_n = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_n >= size_n) {
    return;
  }
  const int split = blockIdx.y;
  const int split_size = size_k / split_k;
  const int begin_k = split * split_size;
  const int end_k = begin_k + split_size;
  float accumulator = 0.0f;

  for (int tile_k = begin_k; tile_k < end_k; tile_k += kTileK) {
    accumulator = trilin_3bit_gemv_accumulate_tile<Scalar, GroupSize>(
        input, qweight, scales, tile_k, global_n, size_n, accumulator);
  }

  if (split_k > 1) {
    split_workspace[static_cast<int64_t>(split) * size_n + global_n] = accumulator;
  } else {
    output[global_n] = ScalarTraits<Scalar>::from_float(accumulator + load_bias(bias, bias_type, global_n));
  }
}

template <typename Scalar, int GroupSize>
__global__ void trilin_3bit_gemv_cta_reduce_kernel(
    const Scalar* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const void* __restrict__ bias,
    Scalar* __restrict__ output,
    int size_n,
    int size_k,
    int split_k,
    int bias_type) {
  __shared__ float partials[32 * kGemvBlockN];

  const int column = threadIdx.x % kGemvBlockN;
  const int split = threadIdx.x / kGemvBlockN;
  const int global_n = blockIdx.x * kGemvBlockN + column;
  const int split_size = size_k / split_k;
  const int begin_k = split * split_size;
  const int end_k = begin_k + split_size;
  float accumulator = 0.0f;

  for (int tile_k = begin_k; tile_k < end_k; tile_k += kTileK) {
    accumulator = trilin_3bit_gemv_accumulate_tile<Scalar, GroupSize>(
        input, qweight, scales, tile_k, global_n, size_n, accumulator);
  }

  partials[split * kGemvBlockN + column] = accumulator;
  __syncthreads();

  if (split == 0) {
    float value = 0.0f;
#pragma unroll
    for (int current_split = 0; current_split < 32; ++current_split) {
      if (current_split < split_k) {
        value += partials[current_split * kGemvBlockN + column];
      }
    }
    value += load_bias(bias, bias_type, global_n);
    output[global_n] = ScalarTraits<Scalar>::from_float(value);
  }
}

template <typename Scalar, int PairBegin = 0, int PairCount = 16>
__device__ __forceinline__ void trilin_3bit_gemv_accumulate_tile_unscaled_prefetched(
    const Scalar* __restrict__ input,
    const uint32_t* words,
    int tile_k,
    float& even_accumulator,
    float& odd_accumulator) {
#pragma unroll
  for (int pair_offset = 0; pair_offset < PairCount; ++pair_offset) {
    const int pair_k = PairBegin + pair_offset;
    const int first_k = pair_k * 2;
    uint32_t first_code;
    uint32_t second_code;
    if constexpr (ScalarTraits<Scalar>::kDecodeAdjacentPair) {
      const int pair_bit = pair_k * 6;
      const int pair_word = pair_bit / 32;
      const int pair_shift = pair_bit % 32;
      uint32_t pair_codes = words[pair_word] >> pair_shift;
      if (pair_shift > 26) {
        pair_codes |= words[pair_word + 1] << (32 - pair_shift);
      }
      first_code = pair_codes;
      second_code = pair_codes >> 3;
    } else {
      const int first_bit = first_k * 3;
      const int first_word = first_bit / 32;
      const int first_shift = first_bit % 32;
      first_code = words[first_word] >> first_shift;
      if (first_shift > 29) {
        first_code |= words[first_word + 1] << (32 - first_shift);
      }

      const int second_k = first_k + 1;
      const int second_bit = second_k * 3;
      const int second_word = second_bit / 32;
      const int second_shift = second_bit % 32;
      second_code = words[second_word] >> second_shift;
      if (second_shift > 29) {
        second_code |= words[second_word + 1] << (32 - second_shift);
      }
    }

    const float2 input_pair = ScalarTraits<Scalar>::load2(input + tile_k + first_k);
    const float first_quant = static_cast<float>(static_cast<int>(first_code & 0x7) - 4);
    const float second_quant = static_cast<float>(static_cast<int>(second_code & 0x7) - 4);
    even_accumulator = __fmaf_rn(input_pair.x, first_quant, even_accumulator);
    odd_accumulator = __fmaf_rn(input_pair.y, second_quant, odd_accumulator);
  }
}

template <typename Scalar, int SizeN, int GroupSize = 128>
__device__ __forceinline__ void trilin_3bit_gemv_group_warp_reduce_body(
    const Scalar* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const void* __restrict__ bias,
    Scalar* __restrict__ output,
    int bias_type,
    int local_block,
    float* paired_partials) {
  constexpr int kSplitK = 32;
  constexpr int kSplitSize = 128;
  constexpr int kTilesPerGroup = GroupSize / kTileK;
  constexpr int kGroupsPerSplit = kSplitSize / GroupSize;
  constexpr int kPairedSplits = kSplitK / 2;
  constexpr int kSharedStride = kGemvBlockN + 2;
  static_assert(GroupSize >= 16 && (GroupSize == 16 || GroupSize % kTileK == 0));
  static_assert(
      GroupSize == 16 || (GroupSize <= kSplitSize ? kSplitSize % GroupSize == 0 : GroupSize % kSplitSize == 0));

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int column = lane & (kGemvBlockN - 1);
  const int split = warp * 2 + (lane >> 4);
  const int global_n = local_block * kGemvBlockN + column;
  const int begin_k = split * kSplitSize;
  uint32_t words[12];
#pragma unroll
  for (int tile_index = 0; tile_index < 4; ++tile_index) {
    const int packed_k = (begin_k / 32 + tile_index) * 3;
#pragma unroll
    for (int packed_offset = 0; packed_offset < 3; ++packed_offset) {
      words[tile_index * 3 + packed_offset] =
          qweight[static_cast<int64_t>(packed_k + packed_offset) * SizeN + global_n];
    }
  }
  float accumulator = 0.0f;
  if constexpr (GroupSize == 16) {
#pragma unroll
    for (int tile_index = 0; tile_index < 4; ++tile_index) {
      float even_accumulator = 0.0f;
      float odd_accumulator = 0.0f;
      trilin_3bit_gemv_accumulate_tile_unscaled_prefetched<Scalar, 0, 8>(
          input,
          words + tile_index * 3,
          begin_k + tile_index * kTileK,
          even_accumulator,
          odd_accumulator);
      const int first_scale_group = begin_k / GroupSize + tile_index * 2;
      const float first_scale =
          __half2float(scales[static_cast<int64_t>(first_scale_group) * SizeN + global_n]);
      accumulator = __fmaf_rn(even_accumulator + odd_accumulator, first_scale, accumulator);

      even_accumulator = 0.0f;
      odd_accumulator = 0.0f;
      trilin_3bit_gemv_accumulate_tile_unscaled_prefetched<Scalar, 8, 8>(
          input,
          words + tile_index * 3,
          begin_k + tile_index * kTileK,
          even_accumulator,
          odd_accumulator);
      const float second_scale =
          __half2float(scales[static_cast<int64_t>(first_scale_group + 1) * SizeN + global_n]);
      accumulator = __fmaf_rn(even_accumulator + odd_accumulator, second_scale, accumulator);
    }
  } else if constexpr (GroupSize >= kSplitSize) {
    float even_accumulator = 0.0f;
    float odd_accumulator = 0.0f;
#pragma unroll
    for (int tile_index = 0; tile_index < 4; ++tile_index) {
      trilin_3bit_gemv_accumulate_tile_unscaled_prefetched<Scalar>(
          input,
          words + tile_index * 3,
          begin_k + tile_index * kTileK,
          even_accumulator,
          odd_accumulator);
    }
    const int scale_group = begin_k / GroupSize;
    const float group_scale = __half2float(scales[static_cast<int64_t>(scale_group) * SizeN + global_n]);
    accumulator = (even_accumulator + odd_accumulator) * group_scale;
  } else {
#pragma unroll
    for (int local_group = 0; local_group < kGroupsPerSplit; ++local_group) {
      float even_accumulator = 0.0f;
      float odd_accumulator = 0.0f;
#pragma unroll
      for (int tile_in_group = 0; tile_in_group < kTilesPerGroup; ++tile_in_group) {
        const int tile_index = local_group * kTilesPerGroup + tile_in_group;
        trilin_3bit_gemv_accumulate_tile_unscaled_prefetched<Scalar>(
            input,
            words + tile_index * 3,
            begin_k + tile_index * kTileK,
            even_accumulator,
            odd_accumulator);
      }
      const int scale_group = split * kGroupsPerSplit + local_group;
      const float group_scale = __half2float(scales[static_cast<int64_t>(scale_group) * SizeN + global_n]);
      accumulator = __fmaf_rn(even_accumulator + odd_accumulator, group_scale, accumulator);
    }
  }

  const float adjacent_split = __shfl_down_sync(0xffffffff, accumulator, kGemvBlockN);
  if (lane < kGemvBlockN) {
    paired_partials[warp * kSharedStride + column] = accumulator + adjacent_split;
  }
  __syncthreads();

  if (warp == 0) {
    const int pair_begin = (lane >> 4) * (kPairedSplits / 2);
    float value = 0.0f;
#pragma unroll
    for (int pair_offset = 0; pair_offset < kPairedSplits / 2; ++pair_offset) {
      value += paired_partials[(pair_begin + pair_offset) * kSharedStride + column];
    }
    const float upper_value = __shfl_down_sync(0xffffffff, value, kGemvBlockN);
    if (lane < kGemvBlockN) {
      value += upper_value;
      value += load_bias(bias, bias_type, global_n);
      output[global_n] = ScalarTraits<Scalar>::from_float(value);
    }
  }
}

template <typename Scalar, int SizeN, int GroupSize>
__global__ void trilin_3bit_gemv_group_warp_reduce_kernel(
    const Scalar* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const void* __restrict__ bias,
    Scalar* __restrict__ output,
    int bias_type) {
  constexpr int kPairedSplits = 16;
  constexpr int kSharedStride = kGemvBlockN + 2;
  __shared__ float paired_partials[kPairedSplits * kSharedStride];
  trilin_3bit_gemv_group_warp_reduce_body<Scalar, SizeN, GroupSize>(
      input,
      qweight,
      scales,
      bias,
      output,
      bias_type,
      blockIdx.x,
      paired_partials);
}

template <typename Scalar, int Rank>
__global__ void trilin_3bit_eora_kernel(
    const Scalar* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const void* __restrict__ bias,
    Scalar* __restrict__ output,
    const Scalar* __restrict__ lora_a,
    const Scalar* __restrict__ lora_b,
    float* __restrict__ lora_down,
    int bias_type) {
  constexpr int kSizeK = 4096;
  constexpr int kSizeN = 4096;
  constexpr int kRank = Rank;
  constexpr int kBaseTiles = kSizeN / kGemvBlockN;
  constexpr int kDownTile = 16;
  constexpr int kDownBlocks = kRank / kDownTile;
  constexpr int kSharedFloats = 512;
  static_assert(kBaseTiles == 256);
  static_assert(kRank == 32 || kRank == 64 || kRank == 128 || kRank == 256);
  static_assert(kRank % kDownTile == 0);

  __shared__ float scratch[kSharedFloats];
  const int launch_blocks = static_cast<int>(gridDim.x);
  const int base_blocks = launch_blocks - kDownBlocks;
  const int block = static_cast<int>(blockIdx.x);

  // Base-output CTAs retain Trilin's exact 16-column reduction. The remaining
  // rank-specialized CTAs compute disjoint complete-K tiles at the same time.
  if (block < base_blocks) {
    for (int output_tile = block; output_tile < kBaseTiles; output_tile += base_blocks) {
      trilin_3bit_gemv_group_warp_reduce_body<Scalar, kSizeN, 128>(
          input,
          qweight,
          scales,
          bias,
          output,
          bias_type,
          output_tile,
          scratch);
      // The body needs no terminal barrier when it returns from an ordinary
      // kernel. Here the same CTA may immediately reuse its shared reduction
      // rows for another output tile.
      __syncthreads();
    }
  } else {
    const int down_block = block - base_blocks;
    const int rank = down_block * kDownTile + (threadIdx.x & (kDownTile - 1));
    const int k_lane = threadIdx.x / kDownTile;
    constexpr int kLanesPerRank = 512 / kDownTile;
    constexpr int kStride = kLanesPerRank;
    int k = k_lane;
    float partial_0 = 0.0f;
    float partial_1 = 0.0f;
    float partial_2 = 0.0f;
    float partial_3 = 0.0f;
    float partial_4 = 0.0f;
    float partial_5 = 0.0f;
    float partial_6 = 0.0f;
    float partial_7 = 0.0f;
    for (; k + kStride * 7 < kSizeK; k += kStride * 8) {
      partial_0 = __fmaf_rn(
          ScalarTraits<Scalar>::to_float(input[k]),
          ScalarTraits<Scalar>::to_float(lora_a[static_cast<int64_t>(k) * kRank + rank]),
          partial_0);
      const int k_1 = k + kStride;
      partial_1 = __fmaf_rn(
          ScalarTraits<Scalar>::to_float(input[k_1]),
          ScalarTraits<Scalar>::to_float(lora_a[static_cast<int64_t>(k_1) * kRank + rank]),
          partial_1);
      const int k_2 = k + kStride * 2;
      partial_2 = __fmaf_rn(
          ScalarTraits<Scalar>::to_float(input[k_2]),
          ScalarTraits<Scalar>::to_float(lora_a[static_cast<int64_t>(k_2) * kRank + rank]),
          partial_2);
      const int k_3 = k + kStride * 3;
      partial_3 = __fmaf_rn(
          ScalarTraits<Scalar>::to_float(input[k_3]),
          ScalarTraits<Scalar>::to_float(lora_a[static_cast<int64_t>(k_3) * kRank + rank]),
          partial_3);
      const int k_4 = k + kStride * 4;
      partial_4 = __fmaf_rn(
          ScalarTraits<Scalar>::to_float(input[k_4]),
          ScalarTraits<Scalar>::to_float(lora_a[static_cast<int64_t>(k_4) * kRank + rank]),
          partial_4);
      const int k_5 = k + kStride * 5;
      partial_5 = __fmaf_rn(
          ScalarTraits<Scalar>::to_float(input[k_5]),
          ScalarTraits<Scalar>::to_float(lora_a[static_cast<int64_t>(k_5) * kRank + rank]),
          partial_5);
      const int k_6 = k + kStride * 6;
      partial_6 = __fmaf_rn(
          ScalarTraits<Scalar>::to_float(input[k_6]),
          ScalarTraits<Scalar>::to_float(lora_a[static_cast<int64_t>(k_6) * kRank + rank]),
          partial_6);
      const int k_7 = k + kStride * 7;
      partial_7 = __fmaf_rn(
          ScalarTraits<Scalar>::to_float(input[k_7]),
          ScalarTraits<Scalar>::to_float(lora_a[static_cast<int64_t>(k_7) * kRank + rank]),
          partial_7);
    }
    for (; k < kSizeK; k += kStride) {
      partial_0 = __fmaf_rn(
          ScalarTraits<Scalar>::to_float(input[k]),
          ScalarTraits<Scalar>::to_float(lora_a[static_cast<int64_t>(k) * kRank + rank]),
          partial_0);
    }
    scratch[threadIdx.x] =
        ((partial_0 + partial_1) + (partial_2 + partial_3)) +
        ((partial_4 + partial_5) + (partial_6 + partial_7));
    __syncthreads();

    if (threadIdx.x < kDownTile) {
      float total = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kLanesPerRank; ++lane) {
        total += scratch[threadIdx.x + lane * kDownTile];
      }
      // torch.matmul produces a dtype-rounded LoRA-down tensor before the up
      // projection. Preserve that boundary while storing it in an exclusively
      // owned FP32 workspace that cannot alias another Trilin route's state.
      lora_down[down_block * kDownTile + threadIdx.x] =
          ScalarTraits<Scalar>::to_float(ScalarTraits<Scalar>::from_float(total));
    }
  }

  cg::this_grid().sync();

  // At most 128 CTAs own one 32-column LoRA-up tile. When the cooperative
  // capacity is smaller, the first CTAs use otherwise-idle warps for a second
  // tile, keeping the tail concurrent rather than serial.
  constexpr int kUpTile = 32;
  constexpr int kUpTiles = kSizeN / kUpTile;
  constexpr int kUpWarpsPerTile = kRank / 32;
  static_assert(kUpWarpsPerTile >= 1 && kUpWarpsPerTile <= 8);
  const int extra_tiles = launch_blocks < kUpTiles ? kUpTiles - launch_blocks : 0;
  const bool dual_tile_block = block < extra_tiles;
  const int primary_tile = dual_tile_block ? block * 2 : block + extra_tiles;
  const int secondary_tile = dual_tile_block ? primary_tile + 1 : kUpTiles;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_slot = warp / kUpWarpsPerTile;
  const int tile = tile_slot == 0 ? primary_tile : secondary_tile;
  const int column = tile * kUpTile + lane;
  const int rank_begin = (warp - tile_slot * kUpWarpsPerTile) * 32;
  const int tiles_for_block = dual_tile_block ? 2 : 1;
  const bool valid_up = tile_slot < tiles_for_block && tile < kUpTiles;
  if (valid_up) {
    float update = 0.0f;
#pragma unroll
    for (int rank_offset = 0; rank_offset < 32; ++rank_offset) {
      const int rank = rank_begin + rank_offset;
      update = __fmaf_rn(
          lora_down[rank],
          ScalarTraits<Scalar>::to_float(lora_b[static_cast<int64_t>(rank) * kSizeN + column]),
          update);
    }
    scratch[threadIdx.x] = update;
  }
  __syncthreads();

  if (threadIdx.x < 64) {
    const bool write_secondary = threadIdx.x >= 32;
    const int write_tile = write_secondary ? secondary_tile : primary_tile;
    if (write_tile < kUpTiles) {
      const int write_lane = threadIdx.x & 31;
      const int write_column = write_tile * kUpTile + write_lane;
      const int partial_base = write_secondary ? kUpWarpsPerTile * 32 : 0;
      float update = 0.0f;
#pragma unroll
      for (int rank_slice = 0; rank_slice < kUpWarpsPerTile; ++rank_slice) {
        update += scratch[partial_base + write_lane + rank_slice * 32];
      }
      output[write_column] = ScalarTraits<Scalar>::from_float(
          ScalarTraits<Scalar>::to_float(output[write_column]) + update);
    }
  }
}

template <typename Scalar, int KvSize>
__global__ void trilin_3bit_qkv_group_warp_reduce_kernel(
    const Scalar* __restrict__ input,
    const uint32_t* __restrict__ q_qweight,
    const half* __restrict__ q_scales,
    const uint32_t* __restrict__ k_qweight,
    const half* __restrict__ k_scales,
    const uint32_t* __restrict__ v_qweight,
    const half* __restrict__ v_scales,
    Scalar* __restrict__ output) {
  constexpr int kQSize = 4096;
  constexpr int kQBlocks = kQSize / kGemvBlockN;
  constexpr int kKvBlocks = KvSize / kGemvBlockN;
  constexpr int kPairedSplits = 16;
  constexpr int kSharedStride = kGemvBlockN + 2;
  __shared__ float paired_partials[kPairedSplits * kSharedStride];

  const int block = blockIdx.x;
  if constexpr (KvSize == kQSize) {
    const uint32_t* selected_qweight;
    const half* selected_scales;
    Scalar* selected_output;
    int local_block;
    if (block < kQBlocks) {
      selected_qweight = q_qweight;
      selected_scales = q_scales;
      selected_output = output;
      local_block = block;
    } else if (block < kQBlocks + kKvBlocks) {
      selected_qweight = k_qweight;
      selected_scales = k_scales;
      selected_output = output + kQSize;
      local_block = block - kQBlocks;
    } else {
      selected_qweight = v_qweight;
      selected_scales = v_scales;
      selected_output = output + kQSize + KvSize;
      local_block = block - kQBlocks - kKvBlocks;
    }
    trilin_3bit_gemv_group_warp_reduce_body<Scalar, kQSize>(
        input,
        selected_qweight,
        selected_scales,
        nullptr,
        selected_output,
        kBiasNone,
        local_block,
        paired_partials);
  } else if (block < kQBlocks) {
    trilin_3bit_gemv_group_warp_reduce_body<Scalar, kQSize>(
        input,
        q_qweight,
        q_scales,
        nullptr,
        output,
        kBiasNone,
        block,
        paired_partials);
  } else {
    const bool is_k = block < kQBlocks + kKvBlocks;
    const int local_block = is_k ? block - kQBlocks : block - kQBlocks - kKvBlocks;
    trilin_3bit_gemv_group_warp_reduce_body<Scalar, KvSize>(
        input,
        is_k ? k_qweight : v_qweight,
        is_k ? k_scales : v_scales,
        nullptr,
        output + kQSize + (is_k ? 0 : KvSize),
        kBiasNone,
        local_block,
        paired_partials);
  }
}

template <typename Scalar>
__device__ __forceinline__ void trilin_3bit_decode_pair(
    const uint32_t* words,
    int pair_k,
    uint32_t& first_code,
    uint32_t& second_code) {
  const int first_k = pair_k * 2;
  if constexpr (ScalarTraits<Scalar>::kDecodeAdjacentPair) {
    const int pair_bit = pair_k * 6;
    const int pair_word = pair_bit / 32;
    const int pair_shift = pair_bit % 32;
    uint32_t pair_codes = words[pair_word] >> pair_shift;
    if (pair_shift > 26) {
      pair_codes |= words[pair_word + 1] << (32 - pair_shift);
    }
    first_code = pair_codes;
    second_code = pair_codes >> 3;
  } else {
    const int first_bit = first_k * 3;
    const int first_word = first_bit / 32;
    const int first_shift = first_bit % 32;
    first_code = words[first_word] >> first_shift;
    if (first_shift > 29) {
      first_code |= words[first_word + 1] << (32 - first_shift);
    }

    const int second_k = first_k + 1;
    const int second_bit = second_k * 3;
    const int second_word = second_bit / 32;
    const int second_shift = second_bit % 32;
    second_code = words[second_word] >> second_shift;
    if (second_shift > 29) {
      second_code |= words[second_word + 1] << (32 - second_shift);
    }
  }
}

template <typename Scalar>
__device__ __forceinline__ void trilin_3bit_swiglu_accumulate_tile(
    const Scalar* __restrict__ input,
    const uint32_t* gate_words,
    const uint32_t* up_words,
    int tile_k,
    float& gate_even_accumulator,
    float& gate_odd_accumulator,
    float& up_even_accumulator,
    float& up_odd_accumulator) {
#pragma unroll
  for (int pair_k = 0; pair_k < 16; ++pair_k) {
    uint32_t gate_first_code;
    uint32_t gate_second_code;
    uint32_t up_first_code;
    uint32_t up_second_code;
    trilin_3bit_decode_pair<Scalar>(gate_words, pair_k, gate_first_code, gate_second_code);
    trilin_3bit_decode_pair<Scalar>(up_words, pair_k, up_first_code, up_second_code);

    const int first_k = pair_k * 2;
    const float2 input_pair = ScalarTraits<Scalar>::load2(input + tile_k + first_k);
    const float gate_first_quant = static_cast<float>(static_cast<int>(gate_first_code & 0x7) - 4);
    const float gate_second_quant = static_cast<float>(static_cast<int>(gate_second_code & 0x7) - 4);
    const float up_first_quant = static_cast<float>(static_cast<int>(up_first_code & 0x7) - 4);
    const float up_second_quant = static_cast<float>(static_cast<int>(up_second_code & 0x7) - 4);
    gate_even_accumulator = __fmaf_rn(input_pair.x, gate_first_quant, gate_even_accumulator);
    gate_odd_accumulator = __fmaf_rn(input_pair.y, gate_second_quant, gate_odd_accumulator);
    up_even_accumulator = __fmaf_rn(input_pair.x, up_first_quant, up_even_accumulator);
    up_odd_accumulator = __fmaf_rn(input_pair.y, up_second_quant, up_odd_accumulator);
  }
}

template <typename Scalar, int SizeN>
__global__ void trilin_3bit_swiglu_group_warp_reduce_kernel(
    const Scalar* __restrict__ input,
    const uint32_t* __restrict__ gate_qweight,
    const half* __restrict__ gate_scales,
    const uint32_t* __restrict__ up_qweight,
    const half* __restrict__ up_scales,
    Scalar* __restrict__ output) {
  constexpr int kSplitK = 32;
  constexpr int kSplitSize = 128;
  constexpr int kPairedSplits = kSplitK / 2;
  constexpr int kSharedStride = kGemvBlockN + 2;
  constexpr int kProjectionStride = kPairedSplits * kSharedStride;
  __shared__ float paired_partials[2 * kProjectionStride];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int column = lane & (kGemvBlockN - 1);
  const int split = warp * 2 + (lane >> 4);
  const int global_n = blockIdx.x * kGemvBlockN + column;
  const int begin_k = split * kSplitSize;
  uint32_t gate_words[12];
  uint32_t up_words[12];
#pragma unroll
  for (int tile_index = 0; tile_index < 4; ++tile_index) {
    const int packed_k = (begin_k / 32 + tile_index) * 3;
#pragma unroll
    for (int packed_offset = 0; packed_offset < 3; ++packed_offset) {
      const int word_index = tile_index * 3 + packed_offset;
      const int64_t qweight_index =
          static_cast<int64_t>(packed_k + packed_offset) * SizeN + global_n;
      gate_words[word_index] = gate_qweight[qweight_index];
      up_words[word_index] = up_qweight[qweight_index];
    }
  }
  const int64_t scale_index = static_cast<int64_t>(split) * SizeN + global_n;
  const float gate_group_scale = __half2float(gate_scales[scale_index]);
  const float up_group_scale = __half2float(up_scales[scale_index]);

  float gate_even_accumulator = 0.0f;
  float gate_odd_accumulator = 0.0f;
  float up_even_accumulator = 0.0f;
  float up_odd_accumulator = 0.0f;
#pragma unroll
  for (int tile_index = 0; tile_index < 4; ++tile_index) {
    trilin_3bit_swiglu_accumulate_tile<Scalar>(
        input,
        gate_words + tile_index * 3,
        up_words + tile_index * 3,
        begin_k + tile_index * kTileK,
        gate_even_accumulator,
        gate_odd_accumulator,
        up_even_accumulator,
        up_odd_accumulator);
  }
  const float gate_accumulator = (gate_even_accumulator + gate_odd_accumulator) * gate_group_scale;
  const float up_accumulator = (up_even_accumulator + up_odd_accumulator) * up_group_scale;

  const float adjacent_gate = __shfl_down_sync(0xffffffff, gate_accumulator, kGemvBlockN);
  const float adjacent_up = __shfl_down_sync(0xffffffff, up_accumulator, kGemvBlockN);
  if (lane < kGemvBlockN) {
    const int partial_index = warp * kSharedStride + column;
    paired_partials[partial_index] = gate_accumulator + adjacent_gate;
    paired_partials[kProjectionStride + partial_index] = up_accumulator + adjacent_up;
  }
  __syncthreads();

  if (warp == 0) {
    const int pair_begin = (lane >> 4) * (kPairedSplits / 2);
    float gate_value = 0.0f;
    float up_value = 0.0f;
#pragma unroll
    for (int pair_offset = 0; pair_offset < kPairedSplits / 2; ++pair_offset) {
      const int partial_index = (pair_begin + pair_offset) * kSharedStride + column;
      gate_value += paired_partials[partial_index];
      up_value += paired_partials[kProjectionStride + partial_index];
    }
    gate_value += __shfl_down_sync(0xffffffff, gate_value, kGemvBlockN);
    up_value += __shfl_down_sync(0xffffffff, up_value, kGemvBlockN);
    if (lane < kGemvBlockN) {
      const Scalar rounded_gate = ScalarTraits<Scalar>::from_float(gate_value);
      const Scalar rounded_up = ScalarTraits<Scalar>::from_float(up_value);
      const float gate = ScalarTraits<Scalar>::to_float(rounded_gate);
      const float up = ScalarTraits<Scalar>::to_float(rounded_up);
      const Scalar activated_gate = ScalarTraits<Scalar>::from_float(gate / (1.0f + __expf(-gate)));
      output[global_n] = ScalarTraits<Scalar>::from_float(ScalarTraits<Scalar>::to_float(activated_gate) * up);
    }
  }
}

template <typename Scalar, int SizeN>
void launch_trilin_3bit_swiglu_group_warp_reduce(
    const Scalar* input,
    const uint32_t* gate_qweight,
    const half* gate_scales,
    const uint32_t* up_qweight,
    const half* up_scales,
    Scalar* output,
    cudaStream_t stream) {
  static_assert(SizeN % kGemvBlockN == 0);
  constexpr int threads = 32 * kGemvBlockN;
  constexpr int blocks = SizeN / kGemvBlockN;
  trilin_3bit_swiglu_group_warp_reduce_kernel<Scalar, SizeN><<<blocks, threads, 0, stream>>>(
      input,
      gate_qweight,
      gate_scales,
      up_qweight,
      up_scales,
      output);
}

template <typename Scalar>
bool try_launch_trilin_3bit_swiglu_group_warp_reduce(
    const Scalar* input,
    const uint32_t* gate_qweight,
    const half* gate_scales,
    const uint32_t* up_qweight,
    const half* up_scales,
    Scalar* output,
    int size_n,
    cudaStream_t stream) {
  switch (size_n) {
    case 11008:
      launch_trilin_3bit_swiglu_group_warp_reduce<Scalar, 11008>(
          input, gate_qweight, gate_scales, up_qweight, up_scales, output, stream);
      return true;
    case 14336:
      launch_trilin_3bit_swiglu_group_warp_reduce<Scalar, 14336>(
          input, gate_qweight, gate_scales, up_qweight, up_scales, output, stream);
      return true;
    default:
      return false;
  }
}

template <typename Scalar, int SizeN, int GroupSize>
void launch_trilin_3bit_gemv_group_warp_reduce(
    const Scalar* input,
    const uint32_t* qweight,
    const half* scales,
    const void* bias,
    Scalar* output,
    int bias_type,
    cudaStream_t stream) {
  static_assert(SizeN % kGemvBlockN == 0);
  constexpr int threads = 32 * kGemvBlockN;
  constexpr int blocks = SizeN / kGemvBlockN;
  trilin_3bit_gemv_group_warp_reduce_kernel<Scalar, SizeN, GroupSize><<<blocks, threads, 0, stream>>>(
      input,
      qweight,
      scales,
      bias,
      output,
      bias_type);
}

template <typename Scalar, int KvSize>
void launch_trilin_3bit_qkv_group_warp_reduce(
    const Scalar* input,
    const uint32_t* q_qweight,
    const half* q_scales,
    const uint32_t* k_qweight,
    const half* k_scales,
    const uint32_t* v_qweight,
    const half* v_scales,
    Scalar* output,
    cudaStream_t stream) {
  constexpr int kQSize = 4096;
  constexpr int threads = 32 * kGemvBlockN;
  constexpr int blocks = (kQSize + 2 * KvSize) / kGemvBlockN;
  trilin_3bit_qkv_group_warp_reduce_kernel<Scalar, KvSize><<<blocks, threads, 0, stream>>>(
      input,
      q_qweight,
      q_scales,
      k_qweight,
      k_scales,
      v_qweight,
      v_scales,
      output);
}

template <typename Scalar>
bool try_launch_trilin_3bit_qkv_group_warp_reduce(
    const Scalar* input,
    const uint32_t* q_qweight,
    const half* q_scales,
    const uint32_t* k_qweight,
    const half* k_scales,
    const uint32_t* v_qweight,
    const half* v_scales,
    Scalar* output,
    int kv_size,
    cudaStream_t stream) {
  switch (kv_size) {
    case 1024:
      launch_trilin_3bit_qkv_group_warp_reduce<Scalar, 1024>(
          input, q_qweight, q_scales, k_qweight, k_scales, v_qweight, v_scales, output, stream);
      return true;
    case 4096:
      launch_trilin_3bit_qkv_group_warp_reduce<Scalar, 4096>(
          input, q_qweight, q_scales, k_qweight, k_scales, v_qweight, v_scales, output, stream);
      return true;
    default:
      return false;
  }
}

template <typename Scalar, int GroupSize>
bool try_launch_trilin_3bit_gemv_group_warp_reduce(
    const Scalar* input,
    const uint32_t* qweight,
    const half* scales,
    const void* bias,
    Scalar* output,
    int size_n,
    int bias_type,
    cudaStream_t stream) {
  switch (size_n) {
    case 1024:
      launch_trilin_3bit_gemv_group_warp_reduce<Scalar, 1024, GroupSize>(
          input, qweight, scales, bias, output, bias_type, stream);
      return true;
    case 4096:
      launch_trilin_3bit_gemv_group_warp_reduce<Scalar, 4096, GroupSize>(
          input, qweight, scales, bias, output, bias_type, stream);
      return true;
    case 11008:
      launch_trilin_3bit_gemv_group_warp_reduce<Scalar, 11008, GroupSize>(
          input, qweight, scales, bias, output, bias_type, stream);
      return true;
    case 14336:
      launch_trilin_3bit_gemv_group_warp_reduce<Scalar, 14336, GroupSize>(
          input, qweight, scales, bias, output, bias_type, stream);
      return true;
    default:
      return false;
  }
}

template <typename Scalar>
__global__ void trilin_3bit_reduce_kernel(
    const float* __restrict__ workspace,
    const void* __restrict__ bias,
    Scalar* __restrict__ output,
    int size_m,
    int size_n,
    int split_k,
    int bias_type) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(size_m) * size_n;
  if (index >= elements) {
    return;
  }
  float value = 0.0f;
#pragma unroll
  for (int split = 0; split < 32; ++split) {
    if (split < split_k) {
      value += workspace[static_cast<int64_t>(split) * elements + index];
    }
  }
  const int n = index % size_n;
  value += load_bias(bias, bias_type, n);
  output[index] = ScalarTraits<Scalar>::from_float(value);
}

template <typename Scalar, int GroupSize>
void launch_trilin_3bit(
    const Scalar* input,
    const uint32_t* qweight,
    const half* scales,
    const void* bias,
    Scalar* output,
    float* workspace,
    int size_m,
    int size_n,
    int size_k,
    int split_k,
    int bias_type,
    cudaStream_t stream) {
  constexpr bool kGroupWarpScaleAligned =
      GroupSize == 16 || (GroupSize <= 128 ? 128 % GroupSize == 0 : GroupSize % 128 == 0);
  if constexpr (kGroupWarpScaleAligned) {
    if (size_m == 1 && size_k == 4096 && split_k == 32 &&
        try_launch_trilin_3bit_gemv_group_warp_reduce<Scalar, GroupSize>(
            input, qweight, scales, bias, output, size_n, bias_type, stream)) {
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
  if (size_m == 1 && split_k > 1) {
    const int threads = split_k * kGemvBlockN;
    const int blocks = size_n / kGemvBlockN;
    trilin_3bit_gemv_cta_reduce_kernel<Scalar, GroupSize><<<blocks, threads, 0, stream>>>(
        input,
        qweight,
        scales,
        bias,
        output,
        size_n,
        size_k,
        split_k,
        bias_type);
  } else if (size_m == 1) {
    constexpr int threads = 128;
    const dim3 grid(
        static_cast<unsigned int>((size_n + threads - 1) / threads),
        static_cast<unsigned int>(split_k));
    trilin_3bit_gemv_kernel<Scalar, GroupSize><<<grid, threads, 0, stream>>>(
        input,
        qweight,
        scales,
        bias,
        output,
        workspace,
        size_n,
        size_k,
        split_k,
        bias_type);
  } else if (size_m <= 16) {
    const dim3 grid(
        static_cast<unsigned int>(size_n / kBlockN),
        1,
        static_cast<unsigned int>(split_k));
    constexpr int block_m = 16;
    constexpr int threads = 128;
    constexpr int shared_bytes = std::max(block_m * kTileK * static_cast<int>(sizeof(Scalar)) +
                                              kTileK * kBlockN * static_cast<int>(sizeof(Scalar)),
                                          block_m * kBlockN * static_cast<int>(sizeof(float)));
    trilin_3bit_wmma_kernel<Scalar, block_m, GroupSize><<<grid, threads, shared_bytes, stream>>>(
        input,
        qweight,
        scales,
        bias,
        output,
        workspace,
        size_m,
        size_n,
        size_k,
        split_k,
        bias_type);
  } else {
    TORCH_CHECK(split_k == 1, "Trilin split_k > 1 is only supported for M <= 16");
    const dim3 grid(
        static_cast<unsigned int>(size_n / kBlockN),
        static_cast<unsigned int>((size_m + 31) / 32),
        1);
    constexpr int block_m = 32;
    constexpr int threads = 256;
    constexpr int shared_bytes = std::max(block_m * kTileK * static_cast<int>(sizeof(Scalar)) +
                                              kTileK * kBlockN * static_cast<int>(sizeof(Scalar)),
                                          block_m * kBlockN * static_cast<int>(sizeof(float)));
    trilin_3bit_wmma_kernel<Scalar, block_m, GroupSize><<<grid, threads, shared_bytes, stream>>>(
        input,
        qweight,
        scales,
        bias,
        output,
        workspace,
        size_m,
        size_n,
        size_k,
        split_k,
        bias_type);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (split_k > 1 && size_m != 1) {
    constexpr int threads = 256;
    const int64_t elements = static_cast<int64_t>(size_m) * size_n;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    trilin_3bit_reduce_kernel<Scalar><<<blocks, threads, 0, stream>>>(
        workspace,
        bias,
        output,
        size_m,
        size_n,
        split_k,
        bias_type);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename Scalar>
void dispatch_trilin_3bit_group_size(
    const Scalar* input,
    const uint32_t* qweight,
    const half* scales,
    const void* bias,
    Scalar* output,
    float* workspace,
    int size_m,
    int size_n,
    int size_k,
    int split_k,
    int group_size,
    int bias_type,
    cudaStream_t stream) {
  switch (group_size) {
    case 16:
      launch_trilin_3bit<Scalar, 16>(
          input, qweight, scales, bias, output, workspace, size_m, size_n, size_k, split_k, bias_type, stream);
      return;
    case 32:
      launch_trilin_3bit<Scalar, 32>(
          input, qweight, scales, bias, output, workspace, size_m, size_n, size_k, split_k, bias_type, stream);
      return;
    case 64:
      launch_trilin_3bit<Scalar, 64>(
          input, qweight, scales, bias, output, workspace, size_m, size_n, size_k, split_k, bias_type, stream);
      return;
    case 96:
      launch_trilin_3bit<Scalar, 96>(
          input, qweight, scales, bias, output, workspace, size_m, size_n, size_k, split_k, bias_type, stream);
      return;
    case 128:
      launch_trilin_3bit<Scalar, 128>(
          input, qweight, scales, bias, output, workspace, size_m, size_n, size_k, split_k, bias_type, stream);
      return;
    case 192:
      launch_trilin_3bit<Scalar, 192>(
          input, qweight, scales, bias, output, workspace, size_m, size_n, size_k, split_k, bias_type, stream);
      return;
    case 256:
      launch_trilin_3bit<Scalar, 256>(
          input, qweight, scales, bias, output, workspace, size_m, size_n, size_k, split_k, bias_type, stream);
      return;
    case 384:
      launch_trilin_3bit<Scalar, 384>(
          input, qweight, scales, bias, output, workspace, size_m, size_n, size_k, split_k, bias_type, stream);
      return;
    case 512:
      launch_trilin_3bit<Scalar, 512>(
          input, qweight, scales, bias, output, workspace, size_m, size_n, size_k, split_k, bias_type, stream);
      return;
    case 1024:
      launch_trilin_3bit<Scalar, 1024>(
          input, qweight, scales, bias, output, workspace, size_m, size_n, size_k, split_k, bias_type, stream);
      return;
    default:
      TORCH_CHECK(
          false,
          "Trilin native group-size dispatch requires group_size 16, 32, 64, 96, 128, 192, 256, 384, 512, or 1024, got ",
          group_size);
  }
}

template <typename Scalar, int Rank>
int query_trilin_eora_grid_blocks(int device) {
  int cooperative_launch = 0;
  cudaError_t status = cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device);
  TORCH_CHECK(
      status == cudaSuccess,
      "Trilin fused EoRA failed to query cooperative-launch support: ",
      cudaGetErrorString(status));
  TORCH_CHECK(cooperative_launch != 0, "Trilin fused EoRA requires cooperative CUDA launches");

  constexpr int threads = 512;
  int active_blocks_per_sm = 0;
  auto kernel = trilin_3bit_eora_kernel<Scalar, Rank>;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm,
      kernel,
      threads,
      0);
  TORCH_CHECK(
      status == cudaSuccess,
      "Trilin fused EoRA failed to query kernel occupancy: ",
      cudaGetErrorString(status));

  int sm_count = 0;
  status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  TORCH_CHECK(
      status == cudaSuccess,
      "Trilin fused EoRA failed to query the multiprocessor count: ",
      cudaGetErrorString(status));
  constexpr int kDesiredBlocks = 256 + Rank / 16;
  constexpr int kMinimumBlocks = 64;
  const int grid_blocks = std::min(kDesiredBlocks, active_blocks_per_sm * sm_count);
  TORCH_CHECK(
      grid_blocks >= kMinimumBlocks,
      "Trilin fused EoRA requires cooperative capacity for at least ",
      kMinimumBlocks,
      " resident CTAs, got ",
      grid_blocks);
  return grid_blocks;
}

template <typename Scalar, int Rank>
int get_trilin_eora_grid_blocks(int device) {
  static std::mutex mutex;
  static std::vector<int> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (device >= static_cast<int>(cache.size())) {
    cache.resize(device + 1, 0);
  }
  int& grid_blocks = cache[device];
  if (grid_blocks == 0) {
    grid_blocks = query_trilin_eora_grid_blocks<Scalar, Rank>(device);
  }
  return grid_blocks;
}

template <typename Scalar, int Rank>
void launch_trilin_3bit_eora(
    const Scalar* input,
    const uint32_t* qweight,
    const half* scales,
    const void* bias,
    Scalar* output,
    const Scalar* lora_a,
    const Scalar* lora_b,
    float* lora_down,
    int bias_type,
    int device,
    cudaStream_t stream) {
  constexpr int threads = 512;
  const int grid_blocks = get_trilin_eora_grid_blocks<Scalar, Rank>(device);
  auto kernel = trilin_3bit_eora_kernel<Scalar, Rank>;
  void* args[] = {
      &input,
      &qweight,
      &scales,
      &bias,
      &output,
      &lora_a,
      &lora_b,
      &lora_down,
      &bias_type,
  };
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(kernel),
      dim3(static_cast<unsigned int>(grid_blocks)),
      dim3(threads),
      args,
      0,
      stream);
  TORCH_CHECK(
      status == cudaSuccess,
      "Trilin fused EoRA cooperative launch failed: ",
      cudaGetErrorString(status));
}

template <typename Scalar>
void dispatch_trilin_3bit_eora(
    int64_t rank,
    const Scalar* input,
    const uint32_t* qweight,
    const half* scales,
    const void* bias,
    Scalar* output,
    const Scalar* lora_a,
    const Scalar* lora_b,
    float* lora_down,
    int bias_type,
    int device,
    cudaStream_t stream) {
#define LAUNCH_TRILIN_EORA(RANK) \
  launch_trilin_3bit_eora<Scalar, RANK>( \
      input, qweight, scales, bias, output, lora_a, lora_b, lora_down, bias_type, device, stream)
  switch (rank) {
    case 32:
      LAUNCH_TRILIN_EORA(32);
      return;
    case 64:
      LAUNCH_TRILIN_EORA(64);
      return;
    case 128:
      LAUNCH_TRILIN_EORA(128);
      return;
    case 256:
      LAUNCH_TRILIN_EORA(256);
      return;
    default:
      TORCH_CHECK(false, "Trilin fused EoRA requires rank 32, 64, 128, or 256, got ", rank);
  }
#undef LAUNCH_TRILIN_EORA
}

at::Tensor trilin_3bit_wmma(
    at::Tensor input,
    at::Tensor qweight,
    at::Tensor scales,
    std::optional<at::Tensor> bias,
    int64_t split_k,
    int64_t group_size) {
  TORCH_CHECK(input.is_cuda(), "Trilin input must be CUDA");
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda(), "Trilin weight tensors must be CUDA");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Trilin input must be FP16 or BF16");
  TORCH_CHECK(qweight.scalar_type() == at::kInt, "Trilin qweight must be int32");
  TORCH_CHECK(scales.scalar_type() == at::kHalf, "Trilin scales must be FP16");
  TORCH_CHECK(input.dim() == 2 && qweight.dim() == 2 && scales.dim() == 2, "Trilin tensors must be 2D");
  TORCH_CHECK(input.is_contiguous(), "Trilin input must be contiguous");
  TORCH_CHECK(qweight.is_contiguous() && scales.is_contiguous(), "Trilin weights must be contiguous");
  TORCH_CHECK(input.device() == qweight.device() && input.device() == scales.device(), "Trilin tensors differ in device");

  const int64_t size_m = input.size(0);
  const int64_t size_k = input.size(1);
  const int64_t size_n = scales.size(1);
  TORCH_CHECK(size_m > 0 && size_k > 0 && size_n > 0, "Trilin dimensions must be positive");
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      properties->major >= 8,
      "Trilin native 3-bit CUDA requires compute capability >= 8.0, got ",
      properties->major,
      ".",
      properties->minor);
  TORCH_CHECK(size_k % 128 == 0, "Trilin K must be divisible by 128");
  TORCH_CHECK(size_n % kBlockN == 0, "Trilin N must be divisible by 64");
  TORCH_CHECK(qweight.size(0) == size_k / 32 * 3 && qweight.size(1) == size_n, "Trilin qweight shape mismatch");
  TORCH_CHECK(
      group_size == 16 || group_size == 32 || group_size == 64 || group_size == 96 || group_size == 128 ||
          group_size == 192 || group_size == 256 || group_size == 384 || group_size == 512 || group_size == 1024,
      "Trilin native group_size must be 16, 32, 64, 96, 128, 192, 256, 384, 512, or 1024, got ",
      group_size);
  TORCH_CHECK(size_k % group_size == 0, "Trilin group_size must divide K");
  TORCH_CHECK(scales.size(0) == size_k / group_size, "Trilin scales shape mismatch");
  TORCH_CHECK(
      split_k == 1 || split_k == 2 || split_k == 4 || split_k == 8 || split_k == 16 || split_k == 32,
      "Trilin split_k must be 1, 2, 4, 8, 16, or 32");
  TORCH_CHECK(size_k % (split_k * kTileK) == 0, "Trilin split_k must divide K in 32-row tiles");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  auto output = at::empty({size_m, size_n}, input.options());
  const bool needs_workspace = split_k > 1 && size_m != 1;
  auto workspace = needs_workspace
      ? at::empty({split_k, size_m, size_n}, input.options().dtype(at::kFloat))
      : at::empty({0}, input.options().dtype(at::kFloat));
  const void* bias_ptr = nullptr;
  int bias_type = kBiasNone;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == input.device(), "Trilin bias device mismatch");
    TORCH_CHECK(
        (bias->scalar_type() == at::kHalf || bias->scalar_type() == at::kBFloat16) && bias->is_contiguous(),
        "Trilin bias must be contiguous FP16 or BF16");
    TORCH_CHECK(bias->numel() == size_n, "Trilin bias shape mismatch");
    bias_ptr = bias->const_data_ptr();
    bias_type = bias->scalar_type() == at::kHalf ? kBiasHalf : kBiasBFloat16;
  }

  if (input.scalar_type() == at::kHalf) {
    dispatch_trilin_3bit_group_size<half>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        workspace.data_ptr<float>(),
        size_m,
        size_n,
        size_k,
        split_k,
        group_size,
        bias_type,
        stream);
  } else {
    dispatch_trilin_3bit_group_size<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        workspace.data_ptr<float>(),
        size_m,
        size_n,
        size_k,
        split_k,
        group_size,
        bias_type,
        stream);
  }
  return output;
}

at::Tensor trilin_3bit_eora(
    at::Tensor input,
    at::Tensor qweight,
    at::Tensor scales,
    std::optional<at::Tensor> bias,
    at::Tensor lora_a,
    at::Tensor lora_b,
    at::Tensor workspace) {
  constexpr int64_t kSize = 4096;
  TORCH_CHECK(input.is_cuda(), "Trilin fused EoRA input must be CUDA");
  TORCH_CHECK(
      qweight.is_cuda() && scales.is_cuda() && lora_a.is_cuda() && lora_b.is_cuda() && workspace.is_cuda(),
      "Trilin fused EoRA tensors must be CUDA");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Trilin fused EoRA input must be FP16 or BF16");
  TORCH_CHECK(qweight.scalar_type() == at::kInt, "Trilin fused EoRA qweight must be int32");
  TORCH_CHECK(scales.scalar_type() == at::kHalf, "Trilin fused EoRA scales must be FP16");
  TORCH_CHECK(
      lora_a.scalar_type() == input.scalar_type() && lora_b.scalar_type() == input.scalar_type(),
      "Trilin fused EoRA adapter tensors must match the input dtype");
  TORCH_CHECK(workspace.scalar_type() == at::kFloat, "Trilin fused EoRA workspace must be FP32");
  TORCH_CHECK(
      input.dim() == 2 && qweight.dim() == 2 && scales.dim() == 2 && lora_a.dim() == 2 && lora_b.dim() == 2,
      "Trilin fused EoRA input, weights, scales, and adapter tensors must be 2D");
  TORCH_CHECK(
      input.is_contiguous() && qweight.is_contiguous() && scales.is_contiguous() && lora_a.is_contiguous() &&
          lora_b.is_contiguous() && workspace.is_contiguous(),
      "Trilin fused EoRA tensors must be contiguous");
  TORCH_CHECK(
      input.device() == qweight.device() && input.device() == scales.device() && input.device() == lora_a.device() &&
          input.device() == lora_b.device() && input.device() == workspace.device(),
      "Trilin fused EoRA tensors differ in device");
  TORCH_CHECK(input.size(0) == 1 && input.size(1) == kSize, "Trilin fused EoRA requires input shape (1, 4096)");
  TORCH_CHECK(
      qweight.size(0) == kSize / 32 * 3 && qweight.size(1) == kSize,
      "Trilin fused EoRA qweight shape mismatch");
  TORCH_CHECK(scales.size(0) == kSize / 128 && scales.size(1) == kSize, "Trilin fused EoRA scale shape mismatch");
  const int64_t rank = lora_a.size(1);
  TORCH_CHECK(
      rank == 32 || rank == 64 || rank == 128 || rank == 256,
      "Trilin fused EoRA requires rank 32, 64, 128, or 256, got ",
      rank);
  TORCH_CHECK(
      lora_a.size(0) == kSize,
      "Trilin fused EoRA LoRA-A first dimension must be 4096");
  TORCH_CHECK(
      lora_b.size(0) == rank && lora_b.size(1) == kSize,
      "Trilin fused EoRA LoRA-B shape must be (rank, 4096)");
  TORCH_CHECK(
      workspace.numel() >= rank,
      "Trilin fused EoRA workspace must contain at least ",
      rank,
      " FP32 values");

  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 0,
      "Trilin fused EoRA requires compute capability 8.0, got ",
      properties->major,
      ".",
      properties->minor);

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaError_t status = cudaStreamIsCapturing(stream, &capture_status);
  TORCH_CHECK(
      status == cudaSuccess,
      "Trilin fused EoRA failed to query CUDA graph capture state: ",
      cudaGetErrorString(status));
  TORCH_CHECK(
      capture_status == cudaStreamCaptureStatusNone,
      "Trilin fused EoRA cooperative launch does not support CUDA graph capture");

  const void* bias_ptr = nullptr;
  int bias_type = kBiasNone;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == input.device(), "Trilin fused EoRA bias device mismatch");
    TORCH_CHECK(
        (bias->scalar_type() == at::kHalf || bias->scalar_type() == at::kBFloat16) && bias->is_contiguous(),
        "Trilin fused EoRA bias must be contiguous FP16 or BF16");
    TORCH_CHECK(bias->numel() == kSize, "Trilin fused EoRA bias shape mismatch");
    bias_ptr = bias->const_data_ptr();
    bias_type = bias->scalar_type() == at::kHalf ? kBiasHalf : kBiasBFloat16;
  }

  auto output = at::empty({1, kSize}, input.options());
  if (input.scalar_type() == at::kHalf) {
    dispatch_trilin_3bit_eora<half>(
        rank,
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(lora_a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(lora_b.data_ptr<at::Half>()),
        workspace.data_ptr<float>(),
        bias_type,
        input.get_device(),
        stream);
  } else {
    dispatch_trilin_3bit_eora<__nv_bfloat16>(
        rank,
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(lora_a.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(lora_b.data_ptr<at::BFloat16>()),
        workspace.data_ptr<float>(),
        bias_type,
        input.get_device(),
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor trilin_3bit_swiglu(
    at::Tensor input,
    at::Tensor gate_qweight,
    at::Tensor gate_scales,
    at::Tensor up_qweight,
    at::Tensor up_scales) {
  TORCH_CHECK(input.is_cuda(), "Trilin fused SwiGLU input must be CUDA");
  TORCH_CHECK(
      gate_qweight.is_cuda() && gate_scales.is_cuda() && up_qweight.is_cuda() && up_scales.is_cuda(),
      "Trilin fused SwiGLU weights must be CUDA");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Trilin fused SwiGLU input must be FP16 or BF16");
  TORCH_CHECK(
      gate_qweight.scalar_type() == at::kInt && up_qweight.scalar_type() == at::kInt,
      "Trilin fused SwiGLU qweights must be int32");
  TORCH_CHECK(
      gate_scales.scalar_type() == at::kHalf && up_scales.scalar_type() == at::kHalf,
      "Trilin fused SwiGLU scales must be FP16");
  TORCH_CHECK(
      input.dim() == 2 && gate_qweight.dim() == 2 && gate_scales.dim() == 2 && up_qweight.dim() == 2 &&
          up_scales.dim() == 2,
      "Trilin fused SwiGLU tensors must be 2D");
  TORCH_CHECK(
      input.is_contiguous() && gate_qweight.is_contiguous() && gate_scales.is_contiguous() &&
          up_qweight.is_contiguous() && up_scales.is_contiguous(),
      "Trilin fused SwiGLU tensors must be contiguous");
  TORCH_CHECK(
      input.device() == gate_qweight.device() && input.device() == gate_scales.device() &&
          input.device() == up_qweight.device() && input.device() == up_scales.device(),
      "Trilin fused SwiGLU tensors differ in device");
  TORCH_CHECK(input.size(0) == 1 && input.size(1) == 4096, "Trilin fused SwiGLU requires input shape (1, 4096)");

  const int64_t size_n = gate_scales.size(1);
  TORCH_CHECK(
      size_n == 11008 || size_n == 14336,
      "Trilin fused SwiGLU requires N=11008 or N=14336, got ",
      size_n);
  TORCH_CHECK(
      gate_qweight.size(0) == 384 && gate_qweight.size(1) == size_n,
      "Trilin fused SwiGLU gate qweight shape mismatch");
  TORCH_CHECK(
      up_qweight.size(0) == 384 && up_qweight.size(1) == size_n,
      "Trilin fused SwiGLU up qweight shape mismatch");
  TORCH_CHECK(
      gate_scales.size(0) == 32 && up_scales.size(0) == 32 && up_scales.size(1) == size_n,
      "Trilin fused SwiGLU scale shape mismatch");

  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      properties->major >= 8,
      "Trilin fused 3-bit SwiGLU requires compute capability >= 8.0, got ",
      properties->major,
      ".",
      properties->minor);

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  auto output = at::empty({1, size_n}, input.options());
  bool launched = false;
  if (input.scalar_type() == at::kHalf) {
    launched = try_launch_trilin_3bit_swiglu_group_warp_reduce<half>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(gate_qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(gate_scales.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(up_qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(up_scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        size_n,
        stream);
  } else {
    launched = try_launch_trilin_3bit_swiglu_group_warp_reduce<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint32_t*>(gate_qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(gate_scales.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(up_qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(up_scales.data_ptr<at::Half>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        size_n,
        stream);
  }
  TORCH_CHECK(launched, "Trilin fused SwiGLU launch dispatch failed for N=", size_n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor trilin_3bit_qkv(
    at::Tensor input,
    at::Tensor q_qweight,
    at::Tensor q_scales,
    at::Tensor k_qweight,
    at::Tensor k_scales,
    at::Tensor v_qweight,
    at::Tensor v_scales) {
  constexpr int64_t kInputSize = 4096;
  constexpr int64_t kPackedRows = kInputSize / 32 * 3;
  constexpr int64_t kScaleRows = kInputSize / 128;

  TORCH_CHECK(input.is_cuda(), "Trilin fused QKV input must be CUDA");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Trilin fused QKV input must be FP16 or BF16");
  TORCH_CHECK(input.dim() == 2, "Trilin fused QKV input must be 2D");
  TORCH_CHECK(input.is_contiguous(), "Trilin fused QKV input must be contiguous");
  TORCH_CHECK(
      input.size(0) == 1 && input.size(1) == kInputSize,
      "Trilin fused QKV requires input shape (1, 4096)");
  TORCH_CHECK(q_scales.dim() == 2, "Trilin fused QKV Q scales must be 2D");
  const int64_t q_size = q_scales.size(1);
  TORCH_CHECK(q_size == 4096, "Trilin fused QKV requires Q size 4096, got ", q_size);
  TORCH_CHECK(k_scales.dim() == 2 && v_scales.dim() == 2, "Trilin fused QKV K/V scales must be 2D");
  const int64_t kv_size = k_scales.size(1);
  TORCH_CHECK(
      kv_size == 1024 || kv_size == 4096,
      "Trilin fused QKV requires K/V size 1024 or 4096, got ",
      kv_size);
  TORCH_CHECK(v_scales.size(1) == kv_size, "Trilin fused QKV K/V widths differ");

  const auto validate_projection = [&](const at::Tensor& qweight,
                                       const at::Tensor& scales,
                                       int64_t size_n,
                                       const char* name) {
    TORCH_CHECK(qweight.is_cuda() && scales.is_cuda(), "Trilin fused QKV ", name, " tensors must be CUDA");
    TORCH_CHECK(qweight.scalar_type() == at::kInt, "Trilin fused QKV ", name, " qweight must be int32");
    TORCH_CHECK(scales.scalar_type() == at::kHalf, "Trilin fused QKV ", name, " scales must be FP16");
    TORCH_CHECK(qweight.dim() == 2 && scales.dim() == 2, "Trilin fused QKV ", name, " tensors must be 2D");
    TORCH_CHECK(
        qweight.is_contiguous() && scales.is_contiguous(),
        "Trilin fused QKV ",
        name,
        " tensors must be contiguous");
    TORCH_CHECK(
        qweight.device() == input.device() && scales.device() == input.device(),
        "Trilin fused QKV ",
        name,
        " tensors differ in device");
    TORCH_CHECK(
        qweight.size(0) == kPackedRows && qweight.size(1) == size_n,
        "Trilin fused QKV ",
        name,
        " qweight shape mismatch");
    TORCH_CHECK(
        scales.size(0) == kScaleRows && scales.size(1) == size_n,
        "Trilin fused QKV ",
        name,
        " scale shape mismatch");
  };
  validate_projection(q_qweight, q_scales, q_size, "Q");
  validate_projection(k_qweight, k_scales, kv_size, "K");
  validate_projection(v_qweight, v_scales, kv_size, "V");

  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      properties->major >= 8,
      "Trilin fused 3-bit QKV requires compute capability >= 8.0, got ",
      properties->major,
      ".",
      properties->minor);

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  auto output = at::empty({1, q_size + 2 * kv_size}, input.options());
  bool launched = false;
  if (input.scalar_type() == at::kHalf) {
    launched = try_launch_trilin_3bit_qkv_group_warp_reduce<half>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(q_qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(q_scales.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(k_qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(k_scales.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(v_qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(v_scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        kv_size,
        stream);
  } else {
    launched = try_launch_trilin_3bit_qkv_group_warp_reduce<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint32_t*>(q_qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(q_scales.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(k_qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(k_scales.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(v_qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(v_scales.data_ptr<at::Half>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        kv_size,
        stream);
  }
  TORCH_CHECK(launched, "Trilin fused QKV launch dispatch failed for K/V size ", kv_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY(gptqmodel_trilin, m) {
  m.def("matmul(Tensor input, Tensor qweight, Tensor scales, Tensor? bias, int split_k, int group_size=128) -> Tensor");
  m.def(
      "matmul_eora(Tensor input, Tensor qweight, Tensor scales, Tensor? bias, Tensor lora_a, Tensor lora_b, Tensor workspace) -> Tensor");
  m.def(
      "silu_mul(Tensor input, Tensor gate_qweight, Tensor gate_scales, Tensor up_qweight, Tensor up_scales) -> Tensor");
  m.def(
      "qkv(Tensor input, Tensor q_qweight, Tensor q_scales, Tensor k_qweight, Tensor k_scales, Tensor v_qweight, Tensor v_scales) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_trilin, CUDA, m) {
  m.impl("matmul", &trilin_3bit_wmma);
  m.impl("matmul_eora", &trilin_3bit_eora);
  m.impl("silu_mul", &trilin_3bit_swiglu);
  m.impl("qkv", &trilin_3bit_qkv);
}
