// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <climits>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int kBlockWidth = 256;
constexpr int kBlockHeight = 24;
constexpr int kKTileHalf2 = kBlockWidth / 2;
constexpr int kWideKTileHalf2 = kBlockWidth / 4;
constexpr int kNarrowDecodeKTileHalf2 = kBlockWidth / 8;
constexpr int kGemmBatchTileRows = 4;
constexpr int kWideGemmBatchTileRows = 8;
constexpr int kGemmBatchTileMinWidth = 2048;
constexpr int kWideGemmBatchTileMinFeature = 8192;
constexpr int kThreads = kBlockWidth;
constexpr int kAccumulationFloat32 = 0;
constexpr int kAccumulationInput = 1;
constexpr int kLoraNone = 0;
constexpr int kLoraDense = 1;
constexpr int kLoraInt8 = 2;

int select_gemv_ktile_half2(int64_t bits, int64_t width, int64_t batch_rows) {
  if (bits == 3) {
    return kKTileHalf2;
  }
  if (width >= kGemmBatchTileMinWidth) {
    return kWideKTileHalf2;
  }
  if (batch_rows == 1) {
    return kNarrowDecodeKTileHalf2;
  }
  return kKTileHalf2;
}

template <typename scalar_t>
struct scalar_traits;

template <>
struct scalar_traits<half> {
  using scalar2_t = half2;
  using torch_t = at::Half;

  __device__ __forceinline__ static half from_float(float value) {
    return __float2half(value);
  }

  __device__ __forceinline__ static half from_int(int value) {
    return __int2half_rn(value);
  }

  __device__ __forceinline__ static half2 make2(half value) {
    return __half2half2(value);
  }

  __device__ __forceinline__ static half2 make2(half low, half high) {
    return __halves2half2(low, high);
  }

  __device__ __forceinline__ static half mul(half a, half b) {
    return __hmul(a, b);
  }

  __device__ __forceinline__ static half2 mul2(half2 a, half2 b) {
    return __hmul2(a, b);
  }

  __device__ __forceinline__ static half2 fma2(half2 a, half2 b, half2 c) {
    return __hfma2(a, b, c);
  }

  __device__ __forceinline__ static float to_float(half value) {
    return __half2float(value);
  }
};

template <>
struct scalar_traits<__nv_bfloat16> {
  using scalar2_t = __nv_bfloat162;
  using torch_t = c10::BFloat16;

  __device__ __forceinline__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16(value);
  }

  __device__ __forceinline__ static __nv_bfloat16 from_int(int value) {
    return __float2bfloat16(static_cast<float>(value));
  }

  __device__ __forceinline__ static __nv_bfloat162 make2(__nv_bfloat16 value) {
    return __bfloat162bfloat162(value);
  }

  __device__ __forceinline__ static __nv_bfloat162 make2(__nv_bfloat16 low,
                                                         __nv_bfloat16 high) {
    return __halves2bfloat162(low, high);
  }

  __device__ __forceinline__ static __nv_bfloat16 mul(__nv_bfloat16 a,
                                                      __nv_bfloat16 b) {
    return __hmul(a, b);
  }

  __device__ __forceinline__ static __nv_bfloat162 mul2(__nv_bfloat162 a,
                                                        __nv_bfloat162 b) {
    return __hmul2(a, b);
  }

  __device__ __forceinline__ static __nv_bfloat162 fma2(__nv_bfloat162 a,
                                                        __nv_bfloat162 b,
                                                        __nv_bfloat162 c) {
    return __hfma2(a, b, c);
  }

  __device__ __forceinline__ static float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }
};

__device__ __forceinline__ unsigned int as_unsigned(int value) {
  return *reinterpret_cast<unsigned int *>(&value);
}

__device__ __forceinline__ int unpack_int3_zero(const int *__restrict__ qzeros,
                                                int group,
                                                int col,
                                                int qzeros_stride) {
  const int block = col >> 5;
  const int idx = col & 31;
  const int *base = qzeros + group * qzeros_stride + block * 3;
  const unsigned int word0 = as_unsigned(base[0]);
  const unsigned int word1 = as_unsigned(base[1]);
  const unsigned int word2 = as_unsigned(base[2]);

  if (idx <= 9) {
    return (word0 >> (idx * 3)) & 0x7;
  }
  if (idx == 10) {
    return ((word0 >> 30) & 0x3) | ((word1 << 2) & 0x4);
  }
  if (idx <= 20) {
    return (word1 >> (1 + 3 * (idx - 11))) & 0x7;
  }
  if (idx == 21) {
    return ((word1 >> 31) & 0x1) | ((word2 << 1) & 0x6);
  }
  return (word2 >> (2 + 3 * (idx - 22))) & 0x7;
}

template <int Bits>
__device__ __forceinline__ int unpack_zero(const int *__restrict__ qzeros,
                                           int group,
                                           int col,
                                           int qzeros_stride) {
  static_assert(Bits == 3 || Bits == 4 || Bits == 8,
                "GrassHopper supports 3, 4, and 8-bit GPTQ decode.");
  if constexpr (Bits == 3) {
    return unpack_int3_zero(qzeros, group, col, qzeros_stride);
  } else {
    constexpr int pack_factor = 32 / Bits;
    constexpr int mask = (1 << Bits) - 1;
    const int word_col = col / pack_factor;
    const int shift = (col - word_col * pack_factor) * Bits;
    return (as_unsigned(qzeros[group * qzeros_stride + word_col]) >> shift) &
           mask;
  }
}

template <typename scalar_t>
struct QuantGroupCache {
  using scalar2_t = typename scalar_traits<scalar_t>::scalar2_t;

  // One thread owns one output column, so scale/qzero only changes when K
  // crosses a quantization group boundary.
  int group;
  scalar_t scale;
  scalar2_t zero2;
};

template <typename scalar_t, int Bits, int GroupSize, bool FloatAccum>
__device__ __forceinline__ typename scalar_traits<scalar_t>::scalar2_t
decode_pair(
    unsigned int packed_pair,
    int absolute_half2_k,
    int col,
    int width,
    int qzeros_stride,
    const int *__restrict__ qzeros,
    const scalar_t *__restrict__ scales,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ deq2,
    int deq_offset,
    QuantGroupCache<scalar_t> &group_cache) {
  using traits = scalar_traits<scalar_t>;
  using scalar2_t = typename traits::scalar2_t;

  const int group = (absolute_half2_k * 2) / GroupSize;
  if (group != group_cache.group) {
    group_cache.group = group;
    group_cache.scale = scales[group * width + col];
    const int zero = unpack_zero<Bits>(qzeros, group, col, qzeros_stride);
    group_cache.zero2 = traits::make2(traits::mul(
        traits::from_float(-static_cast<float>(zero)), group_cache.scale));
  }
  const scalar2_t scale2 = traits::make2(group_cache.scale);
  scalar2_t deq;
  if constexpr (Bits == 3) {
    deq = deq2[(packed_pair & 0x3f) * 32 + deq_offset];
  } else {
    constexpr int mask = (1 << Bits) - 1;
    deq = traits::make2(traits::from_int(packed_pair & mask),
                        traits::from_int((packed_pair >> Bits) & mask));
  }
  return traits::fma2(deq, scale2, group_cache.zero2);
}

template <typename scalar_t, int Bits, int GroupSize, bool FloatAccum>
__device__ __forceinline__ void accumulate_pair(
    unsigned int packed_pair,
    int absolute_half2_k,
    int relative_half2_k,
    int col,
    int width,
    int qzeros_stride,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ blockvec,
    const int *__restrict__ qzeros,
    const scalar_t *__restrict__ scales,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ deq2,
    int deq_offset,
    QuantGroupCache<scalar_t> &group_cache,
    float &acc_float,
    typename scalar_traits<scalar_t>::scalar2_t &acc_input) {
  using traits = scalar_traits<scalar_t>;
  using scalar2_t = typename traits::scalar2_t;

  const scalar2_t weight = decode_pair<scalar_t, Bits, GroupSize, FloatAccum>(
      packed_pair, absolute_half2_k, col, width, qzeros_stride, qzeros, scales,
      deq2, deq_offset, group_cache);
  if constexpr (FloatAccum) {
    const scalar2_t product = traits::mul2(weight, blockvec[relative_half2_k]);
    acc_float += traits::to_float(product.x) + traits::to_float(product.y);
  } else {
    acc_input = traits::fma2(weight, blockvec[relative_half2_k], acc_input);
  }
}

template <typename scalar_t, int Bits, bool FloatAccum>
__device__ __forceinline__ void accumulate_pair_fixed_group(
    unsigned int packed_pair,
    int relative_half2_k,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ blockvec,
    typename scalar_traits<scalar_t>::scalar2_t scale2,
    typename scalar_traits<scalar_t>::scalar2_t zero2,
    float &acc_float,
    typename scalar_traits<scalar_t>::scalar2_t &acc_input) {
  static_assert(Bits == 4 || Bits == 8,
                "Fixed group decode is only used for 4/8-bit GPTQ.");
  using traits = scalar_traits<scalar_t>;
  using scalar2_t = typename traits::scalar2_t;
  constexpr int mask = (1 << Bits) - 1;
  const scalar2_t deq =
      traits::make2(traits::from_int(packed_pair & mask),
                    traits::from_int((packed_pair >> Bits) & mask));
  const scalar2_t weight = traits::fma2(deq, scale2, zero2);
  if constexpr (FloatAccum) {
    const scalar2_t product = traits::mul2(weight, blockvec[relative_half2_k]);
    acc_float += traits::to_float(product.x) + traits::to_float(product.y);
  } else {
    acc_input = traits::fma2(weight, blockvec[relative_half2_k], acc_input);
  }
}

template <typename scalar_t, int Bits, int GroupSize, bool FloatAccum,
          int BatchTileRows, int KTileHalf2>
__device__ __forceinline__ void accumulate_pair_batch(
    unsigned int packed_pair,
    int absolute_half2_k,
    int relative_half2_k,
    int col,
    int width,
    int qzeros_stride,
    int valid_rows,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ blockvec,
    const int *__restrict__ qzeros,
    const scalar_t *__restrict__ scales,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ deq2,
    int deq_offset,
    QuantGroupCache<scalar_t> &group_cache,
    float (&acc_float)[BatchTileRows],
    typename scalar_traits<scalar_t>::scalar2_t (&acc_input)[BatchTileRows]) {
  using traits = scalar_traits<scalar_t>;
  using scalar2_t = typename traits::scalar2_t;

  const scalar2_t weight = decode_pair<scalar_t, Bits, GroupSize, FloatAccum>(
      packed_pair, absolute_half2_k, col, width, qzeros_stride, qzeros, scales,
      deq2, deq_offset, group_cache);
  for (int row = 0; row < BatchTileRows; ++row) {
    if (row >= valid_rows) {
      break;
    }
    const scalar2_t input = blockvec[row * KTileHalf2 + relative_half2_k];
    if constexpr (FloatAccum) {
      const scalar2_t product = traits::mul2(weight, input);
      acc_float[row] += traits::to_float(product.x) + traits::to_float(product.y);
    } else {
      acc_input[row] = traits::fma2(weight, input, acc_input[row]);
    }
  }
}

template <typename scalar_t, int Bits, bool FloatAccum, int BatchTileRows,
          int KTileHalf2>
__device__ __forceinline__ void accumulate_pair_batch_fixed_group(
    unsigned int packed_pair,
    int relative_half2_k,
    int valid_rows,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ blockvec,
    typename scalar_traits<scalar_t>::scalar2_t scale2,
    typename scalar_traits<scalar_t>::scalar2_t zero2,
    float (&acc_float)[BatchTileRows],
    typename scalar_traits<scalar_t>::scalar2_t (&acc_input)[BatchTileRows]) {
  static_assert(Bits == 4 || Bits == 8,
                "Fixed group decode is only used for 4/8-bit GPTQ.");
  using traits = scalar_traits<scalar_t>;
  using scalar2_t = typename traits::scalar2_t;
  constexpr int mask = (1 << Bits) - 1;
  const scalar2_t deq =
      traits::make2(traits::from_int(packed_pair & mask),
                    traits::from_int((packed_pair >> Bits) & mask));
  const scalar2_t weight = traits::fma2(deq, scale2, zero2);
  for (int row = 0; row < BatchTileRows; ++row) {
    if (row >= valid_rows) {
      break;
    }
    const scalar2_t input = blockvec[row * KTileHalf2 + relative_half2_k];
    if constexpr (FloatAccum) {
      const scalar2_t product = traits::mul2(weight, input);
      acc_float[row] += traits::to_float(product.x) + traits::to_float(product.y);
    } else {
      acc_input[row] = traits::fma2(weight, input, acc_input[row]);
    }
  }
}

template <typename scalar_t, int Bits, int GroupSize, bool FloatAccum,
          int KTileHalf2>
__device__ __forceinline__ void accumulate_packed_word(
    unsigned int packed_word,
    int qrow,
    int absolute_half2_base,
    int col,
    int width,
    int qzeros_stride,
    int total_half2,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ blockvec,
    const int *__restrict__ qzeros,
    const scalar_t *__restrict__ scales,
    QuantGroupCache<scalar_t> &group_cache,
    float &acc_float,
    typename scalar_traits<scalar_t>::scalar2_t &acc_input) {
  static_assert(Bits == 4 || Bits == 8,
                "Packed word reuse is only used for 4/8-bit GPTQ.");
  constexpr int pairs_per_word = 16 / Bits;
  constexpr int pair_bits = Bits * 2;
  constexpr unsigned int pair_mask = (1u << pair_bits) - 1u;
  const int qblock = qrow / Bits;
  const int word_in_block = qrow - qblock * Bits;
  const int absolute_half2_word_base =
      qblock * 16 + word_in_block * pairs_per_word;

#pragma unroll
  for (int pair = 0; pair < pairs_per_word; ++pair) {
    const int absolute_half2 = absolute_half2_word_base + pair;
    if (absolute_half2 >= total_half2) {
      break;
    }
    const unsigned int packed_pair =
        (packed_word >> (pair * pair_bits)) & pair_mask;
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        packed_pair, absolute_half2, absolute_half2 - absolute_half2_base,
        col, width, qzeros_stride, blockvec, qzeros, scales, nullptr, 0,
        group_cache, acc_float, acc_input);
  }
}

template <typename scalar_t, int Bits, bool FloatAccum, int KTileHalf2>
__device__ __forceinline__ void accumulate_packed_word_fixed_group(
    unsigned int packed_word,
    int qrow,
    int absolute_half2_base,
    int total_half2,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ blockvec,
    typename scalar_traits<scalar_t>::scalar2_t scale2,
    typename scalar_traits<scalar_t>::scalar2_t zero2,
    float &acc_float,
    typename scalar_traits<scalar_t>::scalar2_t &acc_input) {
  static_assert(Bits == 4 || Bits == 8,
                "Fixed group decode is only used for 4/8-bit GPTQ.");
  constexpr int pairs_per_word = 16 / Bits;
  constexpr int pair_bits = Bits * 2;
  constexpr unsigned int pair_mask = (1u << pair_bits) - 1u;
  const int qblock = qrow / Bits;
  const int word_in_block = qrow - qblock * Bits;
  const int absolute_half2_word_base =
      qblock * 16 + word_in_block * pairs_per_word;

#pragma unroll
  for (int pair = 0; pair < pairs_per_word; ++pair) {
    const int absolute_half2 = absolute_half2_word_base + pair;
    if (absolute_half2 >= total_half2) {
      break;
    }
    const unsigned int packed_pair =
        (packed_word >> (pair * pair_bits)) & pair_mask;
    accumulate_pair_fixed_group<scalar_t, Bits, FloatAccum>(
        packed_pair, absolute_half2 - absolute_half2_base, blockvec, scale2,
        zero2, acc_float, acc_input);
  }
}

template <typename scalar_t, int Bits, int GroupSize, bool FloatAccum,
          int BatchTileRows, int KTileHalf2>
__device__ __forceinline__ void accumulate_packed_word_batch(
    unsigned int packed_word,
    int qrow,
    int absolute_half2_base,
    int col,
    int width,
    int qzeros_stride,
    int valid_rows,
    int total_half2,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ blockvec,
    const int *__restrict__ qzeros,
    const scalar_t *__restrict__ scales,
    QuantGroupCache<scalar_t> &group_cache,
    float (&acc_float)[BatchTileRows],
    typename scalar_traits<scalar_t>::scalar2_t (&acc_input)[BatchTileRows]) {
  static_assert(Bits == 4 || Bits == 8,
                "Packed word reuse is only used for 4/8-bit GPTQ.");
  constexpr int pairs_per_word = 16 / Bits;
  constexpr int pair_bits = Bits * 2;
  constexpr unsigned int pair_mask = (1u << pair_bits) - 1u;
  const int qblock = qrow / Bits;
  const int word_in_block = qrow - qblock * Bits;
  const int absolute_half2_word_base =
      qblock * 16 + word_in_block * pairs_per_word;

#pragma unroll
  for (int pair = 0; pair < pairs_per_word; ++pair) {
    const int absolute_half2 = absolute_half2_word_base + pair;
    if (absolute_half2 >= total_half2) {
      break;
    }
    const unsigned int packed_pair =
        (packed_word >> (pair * pair_bits)) & pair_mask;
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        packed_pair, absolute_half2, absolute_half2 - absolute_half2_base,
        col, width, qzeros_stride, valid_rows, blockvec, qzeros, scales,
        nullptr, 0, group_cache, acc_float, acc_input);
  }
}

template <typename scalar_t, int Bits, bool FloatAccum, int BatchTileRows,
          int KTileHalf2>
__device__ __forceinline__ void accumulate_packed_word_batch_fixed_group(
    unsigned int packed_word,
    int qrow,
    int absolute_half2_base,
    int valid_rows,
    int total_half2,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ blockvec,
    typename scalar_traits<scalar_t>::scalar2_t scale2,
    typename scalar_traits<scalar_t>::scalar2_t zero2,
    float (&acc_float)[BatchTileRows],
    typename scalar_traits<scalar_t>::scalar2_t (&acc_input)[BatchTileRows]) {
  static_assert(Bits == 4 || Bits == 8,
                "Fixed group decode is only used for 4/8-bit GPTQ.");
  constexpr int pairs_per_word = 16 / Bits;
  constexpr int pair_bits = Bits * 2;
  constexpr unsigned int pair_mask = (1u << pair_bits) - 1u;
  const int qblock = qrow / Bits;
  const int word_in_block = qrow - qblock * Bits;
  const int absolute_half2_word_base =
      qblock * 16 + word_in_block * pairs_per_word;

#pragma unroll
  for (int pair = 0; pair < pairs_per_word; ++pair) {
    const int absolute_half2 = absolute_half2_word_base + pair;
    if (absolute_half2 >= total_half2) {
      break;
    }
    const unsigned int packed_pair =
        (packed_word >> (pair * pair_bits)) & pair_mask;
    accumulate_pair_batch_fixed_group<scalar_t, Bits, FloatAccum,
                                      BatchTileRows, KTileHalf2>(
        packed_pair, absolute_half2 - absolute_half2_base, valid_rows,
        blockvec, scale2, zero2, acc_float, acc_input);
  }
}

template <typename scalar_t, int Bits, int GroupSize, int LoraMode,
          bool FloatAccum, int KTileHalf2>
__global__ void vecquant3_gptq_gemv_kernel(
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ vec,
    const int *__restrict__ qweight,
    const scalar_t *__restrict__ scales,
    const int *__restrict__ qzeros,
    const scalar_t *__restrict__ down,
    const scalar_t *__restrict__ up,
    const int8_t *__restrict__ up_qweight,
    const scalar_t *__restrict__ up_scales,
    float *__restrict__ out,
    int qweight_rows,
    int width,
    int qzeros_stride,
    int vec_stride_half2,
    int down_stride,
    int out_stride,
    int rank,
    int lora_group_size) {
  using traits = scalar_traits<scalar_t>;
  using scalar2_t = typename traits::scalar2_t;

  static_assert(Bits == 3 || Bits == 4 || Bits == 8,
                "GrassHopper supports 3, 4, and 8-bit GPTQ decode.");
  static_assert(KTileHalf2 % 16 == 0,
                "GrassHopper K tile must align to GPTQ packing.");
  constexpr int blockwidth2 = KTileHalf2;
  constexpr int blockheight = (KTileHalf2 * Bits) / 16;
  const int row = blockheight * blockIdx.x;
  const int col = kBlockWidth * blockIdx.y + threadIdx.x;
  const int batch_row = blockIdx.z;
  const bool valid_col = col < width;
  const scalar2_t *__restrict__ vec_row = vec + batch_row * vec_stride_half2;
  float *__restrict__ out_row = out + batch_row * out_stride;
  const scalar_t *__restrict__ down_row =
      LoraMode == kLoraNone ? nullptr : down + batch_row * down_stride;

  __shared__ scalar2_t blockvec[blockwidth2];
  const int total_half2 = (qweight_rows / Bits) * 16;
  for (int idx = threadIdx.x; idx < blockwidth2; idx += kThreads) {
    const int absolute_half2 = blockIdx.x * blockwidth2 + idx;
    blockvec[idx] = absolute_half2 < total_half2
                        ? vec_row[absolute_half2]
                        : traits::make2(traits::from_float(0.0f));
  }

  const int off = threadIdx.x % 32;
  __shared__ scalar2_t deq2[64][32];
  if constexpr (Bits == 3) {
    int val = threadIdx.x / 32;
    for (; val < 64; val += kBlockWidth / 32) {
      deq2[val][off] = traits::make2(traits::from_int(val & 0x7),
                                     traits::from_int(val >> 3));
    }
  }

  __syncthreads();

  if (!valid_col) {
    return;
  }

  int i = width * row + col;
  int k = 0;
  const int absolute_half2_base = blockIdx.x * blockwidth2;
  float acc_float = 0.0f;
  scalar2_t acc_input = traits::make2(traits::from_float(0.0f));
  QuantGroupCache<scalar_t> group_cache = {
      -1,
      traits::from_float(0.0f),
      traits::make2(traits::from_float(0.0f)),
  };

  if constexpr (Bits == 3) {
  while (k < blockwidth2 && (row + ((k * 3) >> 4)) < qweight_rows) {
    unsigned int tmp1 = as_unsigned(qweight[i]);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp1 >> 0, absolute_half2_base + k + 0, k + 0, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp1 >> 6, absolute_half2_base + k + 1, k + 1, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp1 >> 12, absolute_half2_base + k + 2, k + 2, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp1 >> 18, absolute_half2_base + k + 3, k + 3, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp1 >> 24, absolute_half2_base + k + 4, k + 4, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    i += width;
    unsigned int tmp2 = as_unsigned(qweight[i]);
    unsigned int tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp, absolute_half2_base + k + 5, k + 5, col, width, qzeros_stride,
        blockvec, qzeros, scales, &deq2[0][0], off, group_cache,
        acc_float, acc_input);
    tmp2 >>= 4;
    k += 6;

    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp2 >> 0, absolute_half2_base + k + 0, k + 0, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp2 >> 6, absolute_half2_base + k + 1, k + 1, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp2 >> 12, absolute_half2_base + k + 2, k + 2, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp2 >> 18, absolute_half2_base + k + 3, k + 3, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    i += width;
    tmp1 = as_unsigned(qweight[i]);
    tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp, absolute_half2_base + k + 4, k + 4, col, width, qzeros_stride,
        blockvec, qzeros, scales, &deq2[0][0], off, group_cache,
        acc_float, acc_input);
    tmp1 >>= 2;
    k += 5;

    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp1 >> 0, absolute_half2_base + k + 0, k + 0, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp1 >> 6, absolute_half2_base + k + 1, k + 1, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp1 >> 12, absolute_half2_base + k + 2, k + 2, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp1 >> 18, absolute_half2_base + k + 3, k + 3, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    accumulate_pair<scalar_t, Bits, GroupSize, FloatAccum>(
        tmp1 >> 24, absolute_half2_base + k + 4, k + 4, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off,
        group_cache, acc_float, acc_input);
    i += width;
    k += 5;
  }
  } else {
    const int row_end = min(row + blockheight, qweight_rows);
    if constexpr (GroupSize >= KTileHalf2 * 2 &&
                  GroupSize % (KTileHalf2 * 2) == 0) {
      const int group = (absolute_half2_base * 2) / GroupSize;
      const scalar_t scale = scales[group * width + col];
      const scalar2_t scale2 = traits::make2(scale);
      const int zero = unpack_zero<Bits>(qzeros, group, col, qzeros_stride);
      const scalar2_t zero2 = traits::make2(
          traits::mul(traits::from_float(-static_cast<float>(zero)), scale));
      for (int qrow = row; qrow < row_end; ++qrow) {
        const unsigned int packed_word =
            as_unsigned(qweight[qrow * width + col]);
        accumulate_packed_word_fixed_group<scalar_t, Bits, FloatAccum,
                                           KTileHalf2>(
            packed_word, qrow, absolute_half2_base, total_half2, blockvec,
            scale2, zero2, acc_float, acc_input);
      }
    } else {
      constexpr int qblocks_per_group = GroupSize / 32;
      scalar2_t scale2 = traits::make2(traits::from_float(0.0f));
      scalar2_t zero2 = traits::make2(traits::from_float(0.0f));
      for (int qrow = row; qrow < row_end; ++qrow) {
        const int group = (qrow / Bits) / qblocks_per_group;
        if (group != group_cache.group) {
          group_cache.group = group;
          const scalar_t scale = scales[group * width + col];
          scale2 = traits::make2(scale);
          const int zero = unpack_zero<Bits>(qzeros, group, col, qzeros_stride);
          zero2 = traits::make2(
              traits::mul(traits::from_float(-static_cast<float>(zero)), scale));
        }
        const unsigned int packed_word =
            as_unsigned(qweight[qrow * width + col]);
        accumulate_packed_word_fixed_group<scalar_t, Bits, FloatAccum,
                                           KTileHalf2>(
            packed_word, qrow, absolute_half2_base, total_half2, blockvec,
            scale2, zero2, acc_float, acc_input);
      }
    }
  }

  float acc;
  if constexpr (FloatAccum) {
    acc = acc_float;
  } else {
    acc = traits::to_float(acc_input.x) + traits::to_float(acc_input.y);
  }

  if constexpr (LoraMode == kLoraDense) {
    if (blockIdx.x == 0) {
      for (int r = 0; r < rank; ++r) {
        acc += traits::to_float(traits::mul(down_row[r], up[r * width + col]));
      }
    }
  } else if constexpr (LoraMode == kLoraInt8) {
    if (blockIdx.x == 0) {
      if (width % lora_group_size == 0) {
        const int scale_stride = width / lora_group_size;
        int idx = col;
        int scale_idx = col / lora_group_size;
        for (int r = 0; r < rank; ++r) {
          const scalar_t up_value = traits::mul(
              traits::from_int(static_cast<int>(up_qweight[idx])),
              up_scales[scale_idx]);
          acc += traits::to_float(traits::mul(down_row[r], up_value));
          idx += width;
          scale_idx += scale_stride;
        }
      } else {
        for (int r = 0; r < rank; ++r) {
          const int idx = r * width + col;
          const scalar_t up_value = traits::mul(
              traits::from_int(static_cast<int>(up_qweight[idx])),
              up_scales[idx / lora_group_size]);
          acc += traits::to_float(traits::mul(down_row[r], up_value));
        }
      }
    }
  }

  atomicAdd(&out_row[col], acc);
}

template <typename scalar_t, int Bits, int GroupSize, int LoraMode,
          bool FloatAccum, int KTileHalf2, int BatchTileRows>
__global__ void vecquant3_gptq_gemm_batch_kernel(
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ vec,
    const int *__restrict__ qweight,
    const scalar_t *__restrict__ scales,
    const int *__restrict__ qzeros,
    const scalar_t *__restrict__ down,
    const scalar_t *__restrict__ up,
    const int8_t *__restrict__ up_qweight,
    const scalar_t *__restrict__ up_scales,
    float *__restrict__ out,
    int qweight_rows,
    int width,
    int qzeros_stride,
    int vec_stride_half2,
    int down_stride,
    int out_stride,
    int batch_rows,
    int rank,
    int lora_group_size) {
  using traits = scalar_traits<scalar_t>;
  using scalar2_t = typename traits::scalar2_t;

  static_assert(Bits == 3 || Bits == 4 || Bits == 8,
                "GrassHopper supports 3, 4, and 8-bit GPTQ decode.");
  static_assert(KTileHalf2 % 16 == 0,
                "GrassHopper K tile must align to GPTQ packing.");
  constexpr int blockwidth2 = KTileHalf2;
  constexpr int blockheight = (KTileHalf2 * Bits) / 16;
  const int row = blockheight * blockIdx.x;
  const int col = kBlockWidth * blockIdx.y + threadIdx.x;
  const int batch_base = blockIdx.z * BatchTileRows;
  const int valid_rows = min(BatchTileRows, batch_rows - batch_base);
  const bool valid_col = col < width;

  __shared__ scalar2_t blockvec[BatchTileRows * blockwidth2];
  const int total_half2 = (qweight_rows / Bits) * 16;
  for (int idx = threadIdx.x; idx < BatchTileRows * blockwidth2;
       idx += kThreads) {
    const int batch_offset = idx / blockwidth2;
    const int k_offset = idx - batch_offset * blockwidth2;
    const int absolute_half2 = blockIdx.x * blockwidth2 + k_offset;
    const int batch_row = batch_base + batch_offset;
    blockvec[idx] =
        batch_offset < valid_rows && absolute_half2 < total_half2
            ? vec[batch_row * vec_stride_half2 + absolute_half2]
            : traits::make2(traits::from_float(0.0f));
  }

  const int off = threadIdx.x % 32;
  __shared__ scalar2_t deq2[64][32];
  if constexpr (Bits == 3) {
    int val = threadIdx.x / 32;
    for (; val < 64; val += kBlockWidth / 32) {
      deq2[val][off] = traits::make2(traits::from_int(val & 0x7),
                                     traits::from_int(val >> 3));
    }
  }

  __syncthreads();

  if (!valid_col || valid_rows <= 0) {
    return;
  }

  int i = width * row + col;
  int k = 0;
  const int absolute_half2_base = blockIdx.x * blockwidth2;
  float acc_float[BatchTileRows];
  scalar2_t acc_input[BatchTileRows];
  for (int batch_offset = 0; batch_offset < BatchTileRows; ++batch_offset) {
    acc_float[batch_offset] = 0.0f;
    acc_input[batch_offset] = traits::make2(traits::from_float(0.0f));
  }
  QuantGroupCache<scalar_t> group_cache = {
      -1,
      traits::from_float(0.0f),
      traits::make2(traits::from_float(0.0f)),
  };

  if constexpr (Bits == 3) {
  while (k < blockwidth2 && (row + ((k * 3) >> 4)) < qweight_rows) {
    unsigned int tmp1 = as_unsigned(qweight[i]);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp1 >> 0, absolute_half2_base + k + 0, k + 0, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp1 >> 6, absolute_half2_base + k + 1, k + 1, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp1 >> 12, absolute_half2_base + k + 2, k + 2, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp1 >> 18, absolute_half2_base + k + 3, k + 3, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp1 >> 24, absolute_half2_base + k + 4, k + 4, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    i += width;
    unsigned int tmp2 = as_unsigned(qweight[i]);
    unsigned int tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp, absolute_half2_base + k + 5, k + 5, col, width, qzeros_stride,
        valid_rows, blockvec, qzeros, scales, &deq2[0][0], off, group_cache,
        acc_float, acc_input);
    tmp2 >>= 4;
    k += 6;

    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp2 >> 0, absolute_half2_base + k + 0, k + 0, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp2 >> 6, absolute_half2_base + k + 1, k + 1, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp2 >> 12, absolute_half2_base + k + 2, k + 2, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp2 >> 18, absolute_half2_base + k + 3, k + 3, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    i += width;
    tmp1 = as_unsigned(qweight[i]);
    tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp, absolute_half2_base + k + 4, k + 4, col, width, qzeros_stride,
        valid_rows, blockvec, qzeros, scales, &deq2[0][0], off, group_cache,
        acc_float, acc_input);
    tmp1 >>= 2;
    k += 5;

    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp1 >> 0, absolute_half2_base + k + 0, k + 0, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp1 >> 6, absolute_half2_base + k + 1, k + 1, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp1 >> 12, absolute_half2_base + k + 2, k + 2, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp1 >> 18, absolute_half2_base + k + 3, k + 3, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    accumulate_pair_batch<scalar_t, Bits, GroupSize, FloatAccum,
                          BatchTileRows, KTileHalf2>(
        tmp1 >> 24, absolute_half2_base + k + 4, k + 4, col, width,
        qzeros_stride, valid_rows, blockvec, qzeros, scales, &deq2[0][0],
        off, group_cache, acc_float, acc_input);
    i += width;
    k += 5;
  }
  } else {
    const int row_end = min(row + blockheight, qweight_rows);
    if constexpr (GroupSize >= KTileHalf2 * 2 &&
                  GroupSize % (KTileHalf2 * 2) == 0) {
      const int group = (absolute_half2_base * 2) / GroupSize;
      const scalar_t scale = scales[group * width + col];
      const scalar2_t scale2 = traits::make2(scale);
      const int zero = unpack_zero<Bits>(qzeros, group, col, qzeros_stride);
      const scalar2_t zero2 = traits::make2(
          traits::mul(traits::from_float(-static_cast<float>(zero)), scale));
      for (int qrow = row; qrow < row_end; ++qrow) {
        const unsigned int packed_word =
            as_unsigned(qweight[qrow * width + col]);
        accumulate_packed_word_batch_fixed_group<
            scalar_t, Bits, FloatAccum, BatchTileRows, KTileHalf2>(
            packed_word, qrow, absolute_half2_base, valid_rows, total_half2,
            blockvec, scale2, zero2, acc_float, acc_input);
      }
    } else {
      if constexpr (GroupSize == 32 || (GroupSize == 64 && Bits == 8)) {
        constexpr int qrows_per_group = (GroupSize / 32) * Bits;
        for (int qrow = row; qrow < row_end;) {
          const int group = qrow / qrows_per_group;
          const int group_end = min(row_end, (group + 1) * qrows_per_group);
          const scalar_t scale = scales[group * width + col];
          const scalar2_t scale2 = traits::make2(scale);
          const int zero = unpack_zero<Bits>(qzeros, group, col, qzeros_stride);
          const scalar2_t zero2 = traits::make2(
              traits::mul(traits::from_float(-static_cast<float>(zero)), scale));
          for (; qrow < group_end; ++qrow) {
            const unsigned int packed_word =
                as_unsigned(qweight[qrow * width + col]);
            accumulate_packed_word_batch_fixed_group<
                scalar_t, Bits, FloatAccum, BatchTileRows, KTileHalf2>(
                packed_word, qrow, absolute_half2_base, valid_rows, total_half2,
                blockvec, scale2, zero2, acc_float, acc_input);
          }
        }
      } else {
        constexpr int qblocks_per_group = GroupSize / 32;
        scalar2_t scale2 = traits::make2(traits::from_float(0.0f));
        scalar2_t zero2 = traits::make2(traits::from_float(0.0f));
        for (int qrow = row; qrow < row_end; ++qrow) {
          const int group = (qrow / Bits) / qblocks_per_group;
          if (group != group_cache.group) {
            group_cache.group = group;
            const scalar_t scale = scales[group * width + col];
            scale2 = traits::make2(scale);
            const int zero =
                unpack_zero<Bits>(qzeros, group, col, qzeros_stride);
            zero2 = traits::make2(traits::mul(
                traits::from_float(-static_cast<float>(zero)), scale));
          }
          const unsigned int packed_word =
              as_unsigned(qweight[qrow * width + col]);
          accumulate_packed_word_batch_fixed_group<
              scalar_t, Bits, FloatAccum, BatchTileRows, KTileHalf2>(
              packed_word, qrow, absolute_half2_base, valid_rows, total_half2,
              blockvec, scale2, zero2, acc_float, acc_input);
        }
      }
    }
  }

  float lora_acc[BatchTileRows];
  if constexpr (LoraMode == kLoraInt8) {
    for (int batch_offset = 0; batch_offset < BatchTileRows; ++batch_offset) {
      lora_acc[batch_offset] = 0.0f;
    }
    if (blockIdx.x == 0) {
      if (width % lora_group_size == 0) {
        const int scale_stride = width / lora_group_size;
        int idx = col;
        int scale_idx = col / lora_group_size;
        for (int r = 0; r < rank; ++r) {
          const scalar_t up_value = traits::mul(
              traits::from_int(static_cast<int>(up_qweight[idx])),
              up_scales[scale_idx]);
          for (int batch_offset = 0; batch_offset < BatchTileRows;
               ++batch_offset) {
            if (batch_offset >= valid_rows) {
              break;
            }
            const int batch_row = batch_base + batch_offset;
            const scalar_t down_value = down[batch_row * down_stride + r];
            lora_acc[batch_offset] +=
                traits::to_float(traits::mul(down_value, up_value));
          }
          idx += width;
          scale_idx += scale_stride;
        }
      } else {
        for (int r = 0; r < rank; ++r) {
          const int idx = r * width + col;
          const scalar_t up_value = traits::mul(
              traits::from_int(static_cast<int>(up_qweight[idx])),
              up_scales[idx / lora_group_size]);
          for (int batch_offset = 0; batch_offset < BatchTileRows;
               ++batch_offset) {
            if (batch_offset >= valid_rows) {
              break;
            }
            const int batch_row = batch_base + batch_offset;
            const scalar_t down_value = down[batch_row * down_stride + r];
            lora_acc[batch_offset] +=
                traits::to_float(traits::mul(down_value, up_value));
          }
        }
      }
    }
  }

  for (int batch_offset = 0; batch_offset < BatchTileRows; ++batch_offset) {
    if (batch_offset >= valid_rows) {
      break;
    }
    float acc;
    if constexpr (FloatAccum) {
      acc = acc_float[batch_offset];
    } else {
      acc = traits::to_float(acc_input[batch_offset].x) +
            traits::to_float(acc_input[batch_offset].y);
    }
    if constexpr (LoraMode == kLoraDense) {
      if (blockIdx.x == 0) {
        const scalar_t *__restrict__ down_row =
            down + (batch_base + batch_offset) * down_stride;
        for (int r = 0; r < rank; ++r) {
          acc += traits::to_float(
              traits::mul(down_row[r], up[r * width + col]));
        }
      }
    } else if constexpr (LoraMode == kLoraInt8) {
      acc += lora_acc[batch_offset];
    }

    const int batch_row = batch_base + batch_offset;
    atomicAdd(&out[batch_row * out_stride + col], acc);
  }
}

void validate_common_inputs(const torch::Tensor &vec,
                            const torch::Tensor &qweight,
                            const torch::Tensor &scales,
                            const torch::Tensor &qzeros,
                            int64_t group_size,
                            int64_t accumulation_type,
                            int64_t bits) {
  TORCH_CHECK(vec.is_cuda(), "GrassHopper gemv requires CUDA vec");
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda() && qzeros.is_cuda(),
              "GrassHopper gemv inputs must be CUDA tensors");
  TORCH_CHECK(vec.scalar_type() == torch::kFloat16 ||
                  vec.scalar_type() == torch::kBFloat16,
              "GrassHopper gemv requires fp16 or bf16 vec");
  TORCH_CHECK(scales.scalar_type() == vec.scalar_type(),
              "GrassHopper gemv scales dtype must match vec dtype");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32,
              "GrassHopper gemv requires int32 qweight");
  TORCH_CHECK(qzeros.scalar_type() == torch::kInt32,
              "GrassHopper gemv requires int32 qzeros");
  TORCH_CHECK(accumulation_type == kAccumulationFloat32 ||
                  accumulation_type == kAccumulationInput,
              "GrassHopper gemv accumulation_type must be 0 (float32) or 1 (input dtype)");
  TORCH_CHECK(bits == 3 || bits == 4 || bits == 8,
              "GrassHopper gemv bits must be one of 3, 4, 8");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "GrassHopper gemv group_size must be one of 32, 64, 128");
  TORCH_CHECK(qweight.dim() == 2 && qweight.size(0) % bits == 0,
              "GrassHopper gemv qweight rows must be divisible by bits");
  TORCH_CHECK(vec.numel() == (qweight.size(0) / bits) * 32,
              "GrassHopper gemv vec length must match qweight packing");
  TORCH_CHECK(vec.numel() % 256 == 0,
              "GrassHopper gemv fast path requires in_features divisible by 256");
  const int64_t ktile_half2 =
      select_gemv_ktile_half2(bits, qweight.size(1), 1);
  const int64_t q_rows_per_tile = (ktile_half2 * bits) / 16;
  TORCH_CHECK(qweight.size(0) % q_rows_per_tile == 0,
              "GrassHopper gemv qweight rows must align to the K tile");
  TORCH_CHECK(scales.dim() == 2,
              "GrassHopper gemv scales must be [num_groups, out_features]");
  TORCH_CHECK(qzeros.dim() == 2,
              "GrassHopper gemv qzeros must be [num_groups, packed_zero_cols]");
  TORCH_CHECK(scales.size(1) == qweight.size(1),
              "GrassHopper gemv scales out_features mismatch");
  TORCH_CHECK(qweight.size(1) % 32 == 0,
              "GrassHopper gemv out_features must be divisible by 32");
  TORCH_CHECK(qzeros.size(0) == scales.size(0),
              "GrassHopper gemv qzeros/scales num_groups mismatch");
  TORCH_CHECK(qzeros.size(1) == (qweight.size(1) / 32) * bits,
              "GrassHopper gemv qzeros packed width mismatch");
  TORCH_CHECK(scales.size(0) == vec.numel() / group_size,
              "GrassHopper gemv scales num_groups must equal in_features / group_size");
  TORCH_CHECK(vec.is_contiguous() && qweight.is_contiguous() &&
                  scales.is_contiguous() && qzeros.is_contiguous(),
              "GrassHopper gemv inputs must be contiguous");
}

void validate_gemm_common_inputs(const torch::Tensor &vec,
                                 const torch::Tensor &qweight,
                                 const torch::Tensor &scales,
                                 const torch::Tensor &qzeros,
                                 int64_t group_size,
                                 int64_t accumulation_type,
                                 int64_t bits) {
  TORCH_CHECK(vec.is_cuda(), "GrassHopper gemm requires CUDA vec");
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda() && qzeros.is_cuda(),
              "GrassHopper gemm inputs must be CUDA tensors");
  TORCH_CHECK(vec.scalar_type() == torch::kFloat16 ||
                  vec.scalar_type() == torch::kBFloat16,
              "GrassHopper gemm requires fp16 or bf16 vec");
  TORCH_CHECK(scales.scalar_type() == vec.scalar_type(),
              "GrassHopper gemm scales dtype must match vec dtype");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32,
              "GrassHopper gemm requires int32 qweight");
  TORCH_CHECK(qzeros.scalar_type() == torch::kInt32,
              "GrassHopper gemm requires int32 qzeros");
  TORCH_CHECK(accumulation_type == kAccumulationFloat32 ||
                  accumulation_type == kAccumulationInput,
              "GrassHopper gemm accumulation_type must be 0 (float32) or 1 (input dtype)");
  TORCH_CHECK(bits == 3 || bits == 4 || bits == 8,
              "GrassHopper gemm bits must be one of 3, 4, 8");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "GrassHopper gemm group_size must be one of 32, 64, 128");
  TORCH_CHECK(vec.dim() == 2 && vec.size(0) > 0,
              "GrassHopper gemm vec must be [batch, in_features]");
  TORCH_CHECK(vec.size(0) <= INT32_MAX,
              "GrassHopper gemm batch size is too large");
  TORCH_CHECK(qweight.dim() == 2 && qweight.size(0) % bits == 0,
              "GrassHopper gemm qweight rows must be divisible by bits");
  TORCH_CHECK(vec.size(1) == (qweight.size(0) / bits) * 32,
              "GrassHopper gemm vec in_features must match qweight packing");
  TORCH_CHECK(vec.size(1) % 256 == 0,
              "GrassHopper gemm fast path requires in_features divisible by 256");
  const int64_t ktile_half2 =
      select_gemv_ktile_half2(bits, qweight.size(1), vec.size(0));
  const int64_t q_rows_per_tile = (ktile_half2 * bits) / 16;
  TORCH_CHECK(qweight.size(0) % q_rows_per_tile == 0,
              "GrassHopper gemm qweight rows must align to the K tile");
  TORCH_CHECK(scales.dim() == 2,
              "GrassHopper gemm scales must be [num_groups, out_features]");
  TORCH_CHECK(qzeros.dim() == 2,
              "GrassHopper gemm qzeros must be [num_groups, packed_zero_cols]");
  TORCH_CHECK(scales.size(1) == qweight.size(1),
              "GrassHopper gemm scales out_features mismatch");
  TORCH_CHECK(qweight.size(1) % 32 == 0,
              "GrassHopper gemm out_features must be divisible by 32");
  TORCH_CHECK(qzeros.size(0) == scales.size(0),
              "GrassHopper gemm qzeros/scales num_groups mismatch");
  TORCH_CHECK(qzeros.size(1) == (qweight.size(1) / 32) * bits,
              "GrassHopper gemm qzeros packed width mismatch");
  TORCH_CHECK(scales.size(0) == vec.size(1) / group_size,
              "GrassHopper gemm scales num_groups must equal in_features / group_size");
  TORCH_CHECK(vec.is_contiguous() && qweight.is_contiguous() &&
                  scales.is_contiguous() && qzeros.is_contiguous(),
              "GrassHopper gemm inputs must be contiguous");
}

template <typename scalar_t, int Bits, int LoraMode, bool FloatAccum,
          int KTileHalf2>
torch::Tensor launch_vecquant3_gptq_gemv_typed_bits_tile(
    torch::Tensor vec,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor down,
    torch::Tensor up,
    torch::Tensor up_qweight,
    torch::Tensor up_scales,
    int64_t group_size,
    int64_t lora_group_size,
    int64_t batch_rows,
    bool output_2d) {
  using traits = scalar_traits<scalar_t>;
  using scalar2_t = typename traits::scalar2_t;
  using torch_t = typename traits::torch_t;
  const c10::cuda::CUDAGuard device_guard(vec.device());
  if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
    const auto *props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major >= 8,
                "GrassHopper gemv bf16 requires CUDA compute capability >= 8.0");
  }

  auto out = output_2d
                 ? torch::empty({batch_rows, qweight.size(1)},
                                torch::TensorOptions()
                                    .device(vec.device())
                                    .dtype(torch::kFloat32))
                 : torch::empty({qweight.size(1)},
                                torch::TensorOptions()
                                    .device(vec.device())
                                    .dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream(vec.device().index());
  C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<float>(), 0,
                                 out.numel() * sizeof(float), stream));

  const int qweight_rows = static_cast<int>(qweight.size(0));
  const int width = static_cast<int>(qweight.size(1));
  const int qzeros_stride = static_cast<int>(qzeros.size(1));
  const int rows = static_cast<int>(batch_rows);
  const int rank_values =
      LoraMode == kLoraNone ? 0 : static_cast<int>(down.numel());
  const int rank = LoraMode == kLoraNone ? 0 : rank_values / rows;
  const int vec_stride_half2 = (qweight_rows / Bits) * 16;
  const int down_stride = rank;
  const int out_stride = width;
  const int lora_group =
      LoraMode == kLoraInt8 ? static_cast<int>(lora_group_size) : 1;
  constexpr int q_rows_per_tile = (KTileHalf2 * Bits) / 16;
  dim3 blocks((qweight_rows + q_rows_per_tile - 1) / q_rows_per_tile,
              (width + kBlockWidth - 1) / kBlockWidth, rows);
  dim3 threads(kThreads);

  const auto *vec_ptr =
      reinterpret_cast<const scalar2_t *>(vec.data_ptr<torch_t>());
  const auto *scale_ptr =
      reinterpret_cast<const scalar_t *>(scales.data_ptr<torch_t>());
  const auto *down_ptr = LoraMode == kLoraNone
                             ? nullptr
                             : reinterpret_cast<const scalar_t *>(
                                   down.data_ptr<torch_t>());
  const auto *up_ptr = LoraMode == kLoraDense
                           ? reinterpret_cast<const scalar_t *>(
                                 up.data_ptr<torch_t>())
                           : nullptr;
  const int8_t *up_qweight_ptr =
      LoraMode == kLoraInt8 ? up_qweight.data_ptr<int8_t>() : nullptr;
  const auto *up_scale_ptr = LoraMode == kLoraInt8
                                 ? reinterpret_cast<const scalar_t *>(
                                       up_scales.data_ptr<torch_t>())
                                 : nullptr;

  if (group_size == 32) {
    vecquant3_gptq_gemv_kernel<scalar_t, Bits, 32, LoraMode, FloatAccum,
                               KTileHalf2>
        <<<blocks, threads, 0, stream>>>(
        vec_ptr, qweight.data_ptr<int>(), scale_ptr, qzeros.data_ptr<int>(),
        down_ptr, up_ptr, up_qweight_ptr, up_scale_ptr, out.data_ptr<float>(),
        qweight_rows, width, qzeros_stride, vec_stride_half2, down_stride,
        out_stride, rank, lora_group);
  } else if (group_size == 64) {
    vecquant3_gptq_gemv_kernel<scalar_t, Bits, 64, LoraMode, FloatAccum,
                               KTileHalf2>
        <<<blocks, threads, 0, stream>>>(
        vec_ptr, qweight.data_ptr<int>(), scale_ptr, qzeros.data_ptr<int>(),
        down_ptr, up_ptr, up_qweight_ptr, up_scale_ptr, out.data_ptr<float>(),
        qweight_rows, width, qzeros_stride, vec_stride_half2, down_stride,
        out_stride, rank, lora_group);
  } else {
    vecquant3_gptq_gemv_kernel<scalar_t, Bits, 128, LoraMode, FloatAccum,
                               KTileHalf2>
        <<<blocks, threads, 0, stream>>>(
        vec_ptr, qweight.data_ptr<int>(), scale_ptr, qzeros.data_ptr<int>(),
        down_ptr, up_ptr, up_qweight_ptr, up_scale_ptr, out.data_ptr<float>(),
        qweight_rows, width, qzeros_stride, vec_stride_half2, down_stride,
        out_stride, rank, lora_group);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <typename scalar_t, int Bits, int LoraMode, bool FloatAccum>
torch::Tensor launch_vecquant3_gptq_gemv_typed_bits(
    torch::Tensor vec,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor down,
    torch::Tensor up,
    torch::Tensor up_qweight,
    torch::Tensor up_scales,
    int64_t group_size,
    int64_t lora_group_size,
    int64_t batch_rows,
    bool output_2d) {
  if constexpr (Bits == 3) {
    return launch_vecquant3_gptq_gemv_typed_bits_tile<
        scalar_t, Bits, LoraMode, FloatAccum, kKTileHalf2>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, batch_rows, output_2d);
  } else {
    if (batch_rows == 1 && qweight.size(1) < kGemmBatchTileMinWidth) {
      return launch_vecquant3_gptq_gemv_typed_bits_tile<
          scalar_t, Bits, LoraMode, FloatAccum, kNarrowDecodeKTileHalf2>(
          vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
          group_size, lora_group_size, batch_rows, output_2d);
    }
    if (qweight.size(1) >= kGemmBatchTileMinWidth) {
      return launch_vecquant3_gptq_gemv_typed_bits_tile<
          scalar_t, Bits, LoraMode, FloatAccum, kWideKTileHalf2>(
          vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
          group_size, lora_group_size, batch_rows, output_2d);
    }
    return launch_vecquant3_gptq_gemv_typed_bits_tile<
        scalar_t, Bits, LoraMode, FloatAccum, kKTileHalf2>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, batch_rows, output_2d);
  }
}

template <typename scalar_t, int LoraMode, bool FloatAccum>
torch::Tensor launch_vecquant3_gptq_gemv_typed(
    torch::Tensor vec,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor down,
    torch::Tensor up,
    torch::Tensor up_qweight,
    torch::Tensor up_scales,
    int64_t group_size,
    int64_t lora_group_size,
    int64_t batch_rows,
    bool output_2d,
    int64_t bits) {
  if (bits == 3) {
    return launch_vecquant3_gptq_gemv_typed_bits<scalar_t, 3, LoraMode,
                                                FloatAccum>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, batch_rows, output_2d);
  }
  if (bits == 4) {
    return launch_vecquant3_gptq_gemv_typed_bits<scalar_t, 4, LoraMode,
                                                FloatAccum>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, batch_rows, output_2d);
  }
  return launch_vecquant3_gptq_gemv_typed_bits<scalar_t, 8, LoraMode,
                                              FloatAccum>(
      vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
      group_size, lora_group_size, batch_rows, output_2d);
}

template <typename scalar_t, int Bits, int LoraMode, bool FloatAccum,
          int KTileHalf2, int BatchTileRows>
torch::Tensor launch_vecquant3_gptq_gemm_batch_typed_bits_tile(
    torch::Tensor vec,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor down,
    torch::Tensor up,
    torch::Tensor up_qweight,
    torch::Tensor up_scales,
    int64_t group_size,
    int64_t lora_group_size,
    int64_t batch_rows) {
  using scalar2_t = typename scalar_traits<scalar_t>::scalar2_t;
  using torch_t = typename scalar_traits<scalar_t>::torch_t;
  const c10::cuda::CUDAGuard device_guard(vec.device());
  if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
    const auto *props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major >= 8,
                "GrassHopper gemm bf16 requires CUDA compute capability >= 8.0");
  }

  auto out = torch::empty({batch_rows, qweight.size(1)},
                          torch::TensorOptions()
                              .device(vec.device())
                              .dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream(vec.device().index());
  C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<float>(), 0,
                                 out.numel() * sizeof(float), stream));

  const int qweight_rows = static_cast<int>(qweight.size(0));
  const int width = static_cast<int>(qweight.size(1));
  const int qzeros_stride = static_cast<int>(qzeros.size(1));
  const int rows = static_cast<int>(batch_rows);
  const int rank_values =
      LoraMode == kLoraNone ? 0 : static_cast<int>(down.numel());
  const int rank = LoraMode == kLoraNone ? 0 : rank_values / rows;
  const int vec_stride_half2 = (qweight_rows / Bits) * 16;
  const int down_stride = rank;
  const int out_stride = width;
  const int lora_group =
      LoraMode == kLoraInt8 ? static_cast<int>(lora_group_size) : 1;
  constexpr int q_rows_per_tile = (KTileHalf2 * Bits) / 16;
  dim3 blocks((qweight_rows + q_rows_per_tile - 1) / q_rows_per_tile,
              (width + kBlockWidth - 1) / kBlockWidth,
              (rows + BatchTileRows - 1) / BatchTileRows);
  dim3 threads(kThreads);

  const auto *vec_ptr =
      reinterpret_cast<const scalar2_t *>(vec.data_ptr<torch_t>());
  const auto *scale_ptr =
      reinterpret_cast<const scalar_t *>(scales.data_ptr<torch_t>());
  const auto *down_ptr = LoraMode == kLoraNone
                             ? nullptr
                             : reinterpret_cast<const scalar_t *>(
                                   down.data_ptr<torch_t>());
  const auto *up_ptr = LoraMode == kLoraDense
                           ? reinterpret_cast<const scalar_t *>(
                                 up.data_ptr<torch_t>())
                           : nullptr;
  const int8_t *up_qweight_ptr =
      LoraMode == kLoraInt8 ? up_qweight.data_ptr<int8_t>() : nullptr;
  const auto *up_scale_ptr = LoraMode == kLoraInt8
                                 ? reinterpret_cast<const scalar_t *>(
                                       up_scales.data_ptr<torch_t>())
                                 : nullptr;

  if (group_size == 32) {
    vecquant3_gptq_gemm_batch_kernel<scalar_t, Bits, 32, LoraMode, FloatAccum,
                                     KTileHalf2, BatchTileRows>
        <<<blocks, threads, 0, stream>>>(
            vec_ptr, qweight.data_ptr<int>(), scale_ptr,
            qzeros.data_ptr<int>(), down_ptr, up_ptr, up_qweight_ptr,
            up_scale_ptr, out.data_ptr<float>(), qweight_rows, width,
            qzeros_stride, vec_stride_half2, down_stride, out_stride, rows,
            rank, lora_group);
  } else if (group_size == 64) {
    vecquant3_gptq_gemm_batch_kernel<scalar_t, Bits, 64, LoraMode, FloatAccum,
                                     KTileHalf2, BatchTileRows>
        <<<blocks, threads, 0, stream>>>(
            vec_ptr, qweight.data_ptr<int>(), scale_ptr,
            qzeros.data_ptr<int>(), down_ptr, up_ptr, up_qweight_ptr,
            up_scale_ptr, out.data_ptr<float>(), qweight_rows, width,
            qzeros_stride, vec_stride_half2, down_stride, out_stride, rows,
            rank, lora_group);
  } else {
    vecquant3_gptq_gemm_batch_kernel<scalar_t, Bits, 128, LoraMode, FloatAccum,
                                     KTileHalf2, BatchTileRows>
        <<<blocks, threads, 0, stream>>>(
            vec_ptr, qweight.data_ptr<int>(), scale_ptr,
            qzeros.data_ptr<int>(), down_ptr, up_ptr, up_qweight_ptr,
            up_scale_ptr, out.data_ptr<float>(), qweight_rows, width,
            qzeros_stride, vec_stride_half2, down_stride, out_stride, rows,
            rank, lora_group);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <typename scalar_t, int Bits, int LoraMode, bool FloatAccum>
torch::Tensor launch_vecquant3_gptq_gemm_batch_typed_bits(
    torch::Tensor vec,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor down,
    torch::Tensor up,
    torch::Tensor up_qweight,
    torch::Tensor up_scales,
    int64_t group_size,
    int64_t lora_group_size,
    int64_t batch_rows) {
  if constexpr (Bits == 3) {
    const int64_t in_features = (qweight.size(0) / Bits) * 32;
    if ((group_size == 64 || group_size == 128) &&
        batch_rows >= kWideGemmBatchTileRows &&
        qweight.size(1) >= kGemmBatchTileMinWidth &&
        (qweight.size(1) >= kWideGemmBatchTileMinFeature ||
         in_features >= kWideGemmBatchTileMinFeature)) {
      return launch_vecquant3_gptq_gemm_batch_typed_bits_tile<
          scalar_t, Bits, LoraMode, FloatAccum, kKTileHalf2,
          kWideGemmBatchTileRows>(
          vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
          group_size, lora_group_size, batch_rows);
    }
    return launch_vecquant3_gptq_gemm_batch_typed_bits_tile<
        scalar_t, Bits, LoraMode, FloatAccum, kKTileHalf2,
        kGemmBatchTileRows>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, batch_rows);
  } else {
    if (qweight.size(1) >= kGemmBatchTileMinWidth) {
      const int64_t in_features = (qweight.size(0) / Bits) * 32;
      const bool use_wide_batch_tile =
          batch_rows >= kWideGemmBatchTileRows &&
          (qweight.size(1) >= kWideGemmBatchTileMinFeature ||
           in_features >= kWideGemmBatchTileMinFeature);
      if (use_wide_batch_tile) {
        if constexpr (Bits == 4 || Bits == 8) {
          if (group_size == 64) {
            return launch_vecquant3_gptq_gemm_batch_typed_bits_tile<
                scalar_t, Bits, LoraMode, FloatAccum, kNarrowDecodeKTileHalf2,
                kWideGemmBatchTileRows>(
                vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
                group_size, lora_group_size, batch_rows);
          }
        }
        return launch_vecquant3_gptq_gemm_batch_typed_bits_tile<
            scalar_t, Bits, LoraMode, FloatAccum, kWideKTileHalf2,
            kWideGemmBatchTileRows>(
            vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
            group_size, lora_group_size, batch_rows);
      }
      return launch_vecquant3_gptq_gemm_batch_typed_bits_tile<
          scalar_t, Bits, LoraMode, FloatAccum, kWideKTileHalf2,
          kGemmBatchTileRows>(
          vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
          group_size, lora_group_size, batch_rows);
    }
    return launch_vecquant3_gptq_gemm_batch_typed_bits_tile<
        scalar_t, Bits, LoraMode, FloatAccum, kKTileHalf2,
        kGemmBatchTileRows>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, batch_rows);
  }
}

template <typename scalar_t, int LoraMode, bool FloatAccum>
torch::Tensor launch_vecquant3_gptq_gemm_batch_typed(
    torch::Tensor vec,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor down,
    torch::Tensor up,
    torch::Tensor up_qweight,
    torch::Tensor up_scales,
    int64_t group_size,
    int64_t lora_group_size,
    int64_t batch_rows,
    int64_t bits) {
  if (bits == 3) {
    return launch_vecquant3_gptq_gemm_batch_typed_bits<scalar_t, 3, LoraMode,
                                                      FloatAccum>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, batch_rows);
  }
  if (bits == 4) {
    return launch_vecquant3_gptq_gemm_batch_typed_bits<scalar_t, 4, LoraMode,
                                                      FloatAccum>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, batch_rows);
  }
  return launch_vecquant3_gptq_gemm_batch_typed_bits<scalar_t, 8, LoraMode,
                                                    FloatAccum>(
      vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
      group_size, lora_group_size, batch_rows);
}

template <int LoraMode>
torch::Tensor launch_vecquant3_gptq_gemv(torch::Tensor vec,
                                         torch::Tensor qweight,
                                         torch::Tensor scales,
                                         torch::Tensor qzeros,
                                         torch::Tensor down,
                                         torch::Tensor up,
                                         torch::Tensor up_qweight,
                                         torch::Tensor up_scales,
                                         int64_t group_size,
                                         int64_t lora_group_size,
                                         int64_t accumulation_type,
                                         int64_t bits) {
  validate_common_inputs(vec, qweight, scales, qzeros, group_size,
                         accumulation_type, bits);
  if constexpr (LoraMode == kLoraDense) {
    TORCH_CHECK(down.is_cuda() && up.is_cuda(),
                "vecquant3 gemv_lora requires CUDA LoRA tensors");
    TORCH_CHECK(down.scalar_type() == vec.scalar_type() &&
                    up.scalar_type() == vec.scalar_type(),
                "vecquant3 gemv_lora LoRA dtype must match vec dtype");
    TORCH_CHECK(down.dim() == 1,
                "vecquant3 gemv_lora down must be a rank vector");
    TORCH_CHECK(up.dim() == 2 && up.size(0) == down.numel() &&
                    up.size(1) == qweight.size(1),
                "vecquant3 gemv_lora up must be [rank, out_features]");
    TORCH_CHECK(down.is_contiguous() && up.is_contiguous(),
                "vecquant3 gemv_lora LoRA tensors must be contiguous");
  } else if constexpr (LoraMode == kLoraInt8) {
    TORCH_CHECK(down.is_cuda() && up_qweight.is_cuda() && up_scales.is_cuda(),
                "vecquant3 gemv_lora_int8 requires CUDA LoRA tensors");
    TORCH_CHECK(down.scalar_type() == vec.scalar_type() &&
                    up_scales.scalar_type() == vec.scalar_type(),
                "vecquant3 gemv_lora_int8 down/up scales dtype must match vec dtype");
    TORCH_CHECK(up_qweight.scalar_type() == torch::kInt8,
                "vecquant3 gemv_lora_int8 up_qweight must be int8");
    TORCH_CHECK(down.dim() == 1,
                "vecquant3 gemv_lora_int8 down must be a rank vector");
    TORCH_CHECK(up_qweight.dim() == 1 && up_scales.dim() == 1,
                "vecquant3 gemv_lora_int8 up_qweight and up_scales must be flat tensors");
    TORCH_CHECK(lora_group_size > 0 && lora_group_size <= INT32_MAX,
                "vecquant3 gemv_lora_int8 lora_group_size must be positive");
    const int64_t expected_up_values = down.numel() * qweight.size(1);
    TORCH_CHECK(up_qweight.numel() >= expected_up_values,
                "vecquant3 gemv_lora_int8 up_qweight is too small for [rank, out_features]");
    TORCH_CHECK(
        up_scales.numel() >=
            (expected_up_values + lora_group_size - 1) / lora_group_size,
        "vecquant3 gemv_lora_int8 up_scales is too small for grouped int8 LoRA-B");
    TORCH_CHECK(down.is_contiguous() && up_qweight.is_contiguous() &&
                    up_scales.is_contiguous(),
                "vecquant3 gemv_lora_int8 LoRA tensors must be contiguous");
  }

  const bool use_float_accum = accumulation_type == kAccumulationFloat32;
  if (vec.scalar_type() == torch::kFloat16) {
    if (use_float_accum) {
      return launch_vecquant3_gptq_gemv_typed<half, LoraMode, true>(
          vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
          group_size, lora_group_size, 1, false, bits);
    }
    return launch_vecquant3_gptq_gemv_typed<half, LoraMode, false>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, 1, false, bits);
  }

  if (use_float_accum) {
    return launch_vecquant3_gptq_gemv_typed<__nv_bfloat16, LoraMode, true>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, 1, false, bits);
  }
  return launch_vecquant3_gptq_gemv_typed<__nv_bfloat16, LoraMode, false>(
      vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
      group_size, lora_group_size, 1, false, bits);
}

template <int LoraMode>
torch::Tensor launch_vecquant3_gptq_gemm(torch::Tensor vec,
                                         torch::Tensor qweight,
                                         torch::Tensor scales,
                                         torch::Tensor qzeros,
                                         torch::Tensor down,
                                         torch::Tensor up,
                                         torch::Tensor up_qweight,
                                         torch::Tensor up_scales,
                                         int64_t group_size,
                                         int64_t lora_group_size,
                                         int64_t accumulation_type,
                                         int64_t bits) {
  validate_gemm_common_inputs(vec, qweight, scales, qzeros, group_size,
                              accumulation_type, bits);
  const int64_t batch_rows = vec.size(0);
  if constexpr (LoraMode == kLoraDense) {
    TORCH_CHECK(down.is_cuda() && up.is_cuda(),
                "vecquant3 gemm_lora requires CUDA LoRA tensors");
    TORCH_CHECK(down.scalar_type() == vec.scalar_type() &&
                    up.scalar_type() == vec.scalar_type(),
                "vecquant3 gemm_lora LoRA dtype must match vec dtype");
    TORCH_CHECK(down.dim() == 2 && down.size(0) == batch_rows,
                "vecquant3 gemm_lora down must be [batch, rank]");
    TORCH_CHECK(up.dim() == 2 && up.size(0) == down.size(1) &&
                    up.size(1) == qweight.size(1),
                "vecquant3 gemm_lora up must be [rank, out_features]");
    TORCH_CHECK(down.is_contiguous() && up.is_contiguous(),
                "vecquant3 gemm_lora LoRA tensors must be contiguous");
  } else if constexpr (LoraMode == kLoraInt8) {
    TORCH_CHECK(down.is_cuda() && up_qweight.is_cuda() && up_scales.is_cuda(),
                "vecquant3 gemm_lora_int8 requires CUDA LoRA tensors");
    TORCH_CHECK(down.scalar_type() == vec.scalar_type() &&
                    up_scales.scalar_type() == vec.scalar_type(),
                "vecquant3 gemm_lora_int8 down/up scales dtype must match vec dtype");
    TORCH_CHECK(up_qweight.scalar_type() == torch::kInt8,
                "vecquant3 gemm_lora_int8 up_qweight must be int8");
    TORCH_CHECK(down.dim() == 2 && down.size(0) == batch_rows,
                "vecquant3 gemm_lora_int8 down must be [batch, rank]");
    TORCH_CHECK(up_qweight.dim() == 1 && up_scales.dim() == 1,
                "vecquant3 gemm_lora_int8 up_qweight and up_scales must be flat tensors");
    TORCH_CHECK(lora_group_size > 0 && lora_group_size <= INT32_MAX,
                "vecquant3 gemm_lora_int8 lora_group_size must be positive");
    const int64_t expected_up_values = down.size(1) * qweight.size(1);
    TORCH_CHECK(up_qweight.numel() >= expected_up_values,
                "vecquant3 gemm_lora_int8 up_qweight is too small for [rank, out_features]");
    TORCH_CHECK(
        up_scales.numel() >=
            (expected_up_values + lora_group_size - 1) / lora_group_size,
        "vecquant3 gemm_lora_int8 up_scales is too small for grouped int8 LoRA-B");
    TORCH_CHECK(down.is_contiguous() && up_qweight.is_contiguous() &&
                    up_scales.is_contiguous(),
                "vecquant3 gemm_lora_int8 LoRA tensors must be contiguous");
  }

  const bool use_float_accum = accumulation_type == kAccumulationFloat32;
  const bool use_batch_tiled =
      batch_rows >= 2 && (qweight.size(1) >= kGemmBatchTileMinWidth ||
                          batch_rows >= kGemmBatchTileRows);
  if (vec.scalar_type() == torch::kFloat16) {
    if (use_float_accum) {
      if (use_batch_tiled) {
        return launch_vecquant3_gptq_gemm_batch_typed<half, LoraMode, true>(
            vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
            group_size, lora_group_size, batch_rows, bits);
      }
      return launch_vecquant3_gptq_gemv_typed<half, LoraMode, true>(
          vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
          group_size, lora_group_size, batch_rows, true, bits);
    }
    if (use_batch_tiled) {
      return launch_vecquant3_gptq_gemm_batch_typed<half, LoraMode, false>(
          vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
          group_size, lora_group_size, batch_rows, bits);
    }
    return launch_vecquant3_gptq_gemv_typed<half, LoraMode, false>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, batch_rows, true, bits);
  }

  if (use_float_accum) {
    if (use_batch_tiled) {
        return launch_vecquant3_gptq_gemm_batch_typed<__nv_bfloat16, LoraMode,
                                                   true>(
          vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
          group_size, lora_group_size, batch_rows, bits);
    }
    return launch_vecquant3_gptq_gemv_typed<__nv_bfloat16, LoraMode, true>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, batch_rows, true, bits);
  }
  if (use_batch_tiled) {
    return launch_vecquant3_gptq_gemm_batch_typed<__nv_bfloat16, LoraMode,
                                                 false>(
        vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
        group_size, lora_group_size, batch_rows, bits);
  }
  return launch_vecquant3_gptq_gemv_typed<__nv_bfloat16, LoraMode, false>(
      vec, qweight, scales, qzeros, down, up, up_qweight, up_scales,
      group_size, lora_group_size, batch_rows, true, bits);
}

}  // namespace

torch::Tensor vecquant3_gptq_gemv_cuda(torch::Tensor vec,
                                       torch::Tensor qweight,
                                       torch::Tensor scales,
                                       torch::Tensor qzeros,
                                       int64_t group_size,
                                       int64_t accumulation_type,
                                       int64_t bits) {
  return launch_vecquant3_gptq_gemv<kLoraNone>(
      vec.reshape({-1}), qweight, scales, qzeros, torch::Tensor(),
      torch::Tensor(), torch::Tensor(), torch::Tensor(), group_size, 0,
      accumulation_type, bits);
}

torch::Tensor vecquant3_gptq_gemv_lora_cuda(torch::Tensor vec,
                                            torch::Tensor qweight,
                                            torch::Tensor scales,
                                            torch::Tensor qzeros,
                                            torch::Tensor down,
                                            torch::Tensor up,
                                            int64_t group_size,
                                            int64_t accumulation_type,
                                            int64_t bits) {
  return launch_vecquant3_gptq_gemv<kLoraDense>(
      vec.reshape({-1}), qweight, scales, qzeros, down.reshape({-1}), up,
      torch::Tensor(), torch::Tensor(), group_size, 0, accumulation_type,
      bits);
}

torch::Tensor vecquant3_gptq_gemv_lora_int8_cuda(torch::Tensor vec,
                                                 torch::Tensor qweight,
                                                 torch::Tensor scales,
                                                 torch::Tensor qzeros,
                                                 torch::Tensor down,
                                                 torch::Tensor up_qweight,
                                                 torch::Tensor up_scales,
                                                 int64_t group_size,
                                                 int64_t lora_group_size,
                                                 int64_t accumulation_type,
                                                 int64_t bits) {
  return launch_vecquant3_gptq_gemv<kLoraInt8>(
      vec.reshape({-1}), qweight, scales, qzeros, down.reshape({-1}),
      torch::Tensor(), up_qweight.reshape({-1}), up_scales.reshape({-1}),
      group_size, lora_group_size, accumulation_type, bits);
}

torch::Tensor vecquant3_gptq_gemm_cuda(torch::Tensor vec,
                                       torch::Tensor qweight,
                                       torch::Tensor scales,
                                       torch::Tensor qzeros,
                                       int64_t group_size,
                                       int64_t accumulation_type,
                                       int64_t bits) {
  return launch_vecquant3_gptq_gemm<kLoraNone>(
      vec, qweight, scales, qzeros, torch::Tensor(), torch::Tensor(),
      torch::Tensor(), torch::Tensor(), group_size, 0, accumulation_type,
      bits);
}

torch::Tensor vecquant3_gptq_gemm_lora_cuda(torch::Tensor vec,
                                            torch::Tensor qweight,
                                            torch::Tensor scales,
                                            torch::Tensor qzeros,
                                            torch::Tensor down,
                                            torch::Tensor up,
                                            int64_t group_size,
                                            int64_t accumulation_type,
                                            int64_t bits) {
  return launch_vecquant3_gptq_gemm<kLoraDense>(
      vec, qweight, scales, qzeros, down, up, torch::Tensor(),
      torch::Tensor(), group_size, 0, accumulation_type, bits);
}

torch::Tensor vecquant3_gptq_gemm_lora_int8_cuda(torch::Tensor vec,
                                                 torch::Tensor qweight,
                                                 torch::Tensor scales,
                                                 torch::Tensor qzeros,
                                                 torch::Tensor down,
                                                 torch::Tensor up_qweight,
                                                 torch::Tensor up_scales,
                                                 int64_t group_size,
                                                 int64_t lora_group_size,
                                                 int64_t accumulation_type,
                                                 int64_t bits) {
  return launch_vecquant3_gptq_gemm<kLoraInt8>(
      vec, qweight, scales, qzeros, down, torch::Tensor(),
      up_qweight.reshape({-1}), up_scales.reshape({-1}), group_size,
      lora_group_size, accumulation_type, bits);
}
