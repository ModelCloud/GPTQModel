// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "qvq_p32_abi.h"
#include "qvq_p32_internal.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

using qvq_p32_internal::set_last_error;

template <int RankCount>
__global__ __launch_bounds__(128) void p32_rank8_project_kernel(
    const float* __restrict__ input,
    const half* __restrict__ rank8_a,
    half* __restrict__ hidden,
    int size_m,
    int size_k) {
  static_assert(RankCount == 8 || RankCount == 16 || RankCount == 24);
  constexpr int kRanksPerBlock = 8;
  const int rank_base = static_cast<int>(blockIdx.x) * kRanksPerBlock;
  constexpr int kRowsPerBlock = 4;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  // One warp reuses each input element across eight adjacent ranks while
  // preserving each rank's original FMA and shuffle-reduction order.
  for (int row = static_cast<int>(blockIdx.y) * kRowsPerBlock + warp;
       row < size_m;
       row += static_cast<int>(gridDim.y) * kRowsPerBlock) {
    float accumulators[kRanksPerBlock] = {};
    for (int k = lane; k < size_k; k += 32) {
      const float input_value = input[static_cast<int64_t>(row) * size_k + k];
      union PackedRank8 {
        uint4 vector;
        half2 pairs[4];
      } weights;
      weights.vector = *reinterpret_cast<const uint4*>(
          rank8_a + static_cast<int64_t>(k) * RankCount + rank_base);
#pragma unroll
      for (int pair_index = 0; pair_index < 4; ++pair_index) {
        const float2 pair = __half22float2(weights.pairs[pair_index]);
        accumulators[2 * pair_index] = fmaf(
            input_value, pair.x, accumulators[2 * pair_index]);
        accumulators[2 * pair_index + 1] = fmaf(
            input_value, pair.y, accumulators[2 * pair_index + 1]);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      for (int local_rank = 0; local_rank < kRanksPerBlock; ++local_rank) {
        accumulators[local_rank] += __shfl_down_sync(
            0xffffffffu, accumulators[local_rank], offset);
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int local_rank = 0; local_rank < kRanksPerBlock; ++local_rank) {
        hidden[static_cast<int64_t>(row) * RankCount + rank_base + local_rank] =
            __float2half_rn(accumulators[local_rank]);
      }
    }
  }
}

template <int RankCount>
__global__ __launch_bounds__(128) void p32_rank8_epilogue_kernel(
    const float* base_output,
    const half* __restrict__ hidden,
    const float* __restrict__ rank8_b,
    float* output,
    int size_m,
    int size_n) {
  static_assert(RankCount == 8 || RankCount == 16 || RankCount == 24);
  __shared__ half hidden_row[RankCount];
  const int column = static_cast<int>(blockIdx.x) * blockDim.x +
      static_cast<int>(threadIdx.x);
  for (int row = static_cast<int>(blockIdx.y); row < size_m;
       row += static_cast<int>(gridDim.y)) {
    for (int rank = static_cast<int>(threadIdx.x); rank < RankCount;
         rank += blockDim.x) {
      hidden_row[rank] = hidden[static_cast<int64_t>(row) * RankCount + rank];
    }
    __syncthreads();
    if (column < size_n) {
      const int64_t index = static_cast<int64_t>(row) * size_n + column;
      float correction = 0.0f;
#pragma unroll
      for (int rank = 0; rank < RankCount; ++rank) {
        correction += __half2float(hidden_row[rank]) *
            rank8_b[static_cast<int64_t>(rank) * size_n + column];
      }
      output[index] = base_output[index] + correction;
    }
    __syncthreads();
  }
}

__device__ __forceinline__ float round_to_half(float value) {
  return __half2float(__float2half_rn(value));
}

__device__ __forceinline__ half2 exact_half2_add(half2 first, half2 second) {
  const float2 a = __half22float2(first);
  const float2 b = __half22float2(second);
  return __floats2half2_rn(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ half2 exact_half2_sub(half2 first, half2 second) {
  const float2 a = __half22float2(first);
  const float2 b = __half22float2(second);
  return __floats2half2_rn(a.x - b.x, a.y - b.y);
}

template <int RankCount>
__global__ __launch_bounds__(1024) void p32_hadamard_epilogue_legacy_kernel(
    const float* base_output,
    const half* __restrict__ hidden,
    const float* __restrict__ rank8_b,
    const half* __restrict__ scale_v,
    half* output,
    int size_m,
    int size_n,
    bool normalize_first) {
  extern __shared__ half legacy_values[];
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const auto padded = [](int index) { return index + ((index >> 5) << 1); };
  const float sqrt_n = sqrtf(static_cast<float>(size_n));
  const float divisor = round_to_half(sqrt_n);
  const float reciprocal = 1.0f / sqrt_n;
  for (int column = tid; column < size_n; column += blockDim.x) {
    float correction = 0.0f;
#pragma unroll
    for (int rank = 0; rank < RankCount; ++rank) {
      correction += __half2float(
          hidden[static_cast<int64_t>(row) * RankCount + rank]) *
          rank8_b[static_cast<int64_t>(rank) * size_n + column];
    }
    float value = round_to_half(
        base_output[static_cast<int64_t>(row) * size_n + column] + correction);
    if (normalize_first) value = round_to_half(value / divisor);
    legacy_values[padded(column)] = __float2half_rn(value);
  }
  __syncthreads();
  for (int bit = 1; bit < size_n; bit <<= 1) {
    for (int index = tid; index < size_n; index += blockDim.x) {
      const int peer = index ^ bit;
      if (index < peer) {
        const float first = __half2float(legacy_values[padded(index)]);
        const float second = __half2float(legacy_values[padded(peer)]);
        legacy_values[padded(index)] = __float2half_rn(first + second);
        legacy_values[padded(peer)] = __float2half_rn(first - second);
      }
    }
    __syncthreads();
  }
  for (int column = tid; column < size_n; column += blockDim.x) {
    float value = __half2float(legacy_values[padded(column)]);
    if (!normalize_first) value = round_to_half(value * reciprocal);
    value = round_to_half(value * __half2float(scale_v[column]));
    output[static_cast<int64_t>(row) * size_n + column] = __float2half_rn(value);
  }
}

template <int RankCount>
__global__ __launch_bounds__(1024) void p32_hadamard_epilogue_kernel(
    const float* base_output,
    const half* __restrict__ hidden,
    const float* __restrict__ rank8_b,
    const half* __restrict__ scale_v,
    half* output,
    int size_m,
    int size_n,
    bool normalize_first) {
  static_assert(RankCount == 0 || RankCount == 8 || RankCount == 16 || RankCount == 24);
  extern __shared__ half2 packed_values[];
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const float sqrt_n = sqrtf(static_cast<float>(size_n));
  const float divisor = round_to_half(sqrt_n);
  const float reciprocal = 1.0f / sqrt_n;
  const int pair_count = size_n / 2;

  // Preserve the legacy one-column correction expression and its FP16
  // boundary. The packed representation begins only after each scalar value
  // has been rounded exactly as in the compatibility implementation.
  half* scalar_values = reinterpret_cast<half*>(packed_values);
  for (int column = tid; column < size_n; column += blockDim.x) {
    float correction = 0.0f;
#pragma unroll
    for (int rank = 0; rank < RankCount; ++rank) {
      const float hidden_value =
          __half2float(hidden[static_cast<int64_t>(row) * RankCount + rank]);
      correction += hidden_value *
          rank8_b[static_cast<int64_t>(rank) * size_n + column];
    }
    float value = round_to_half(
        base_output[static_cast<int64_t>(row) * size_n + column] + correction);
    if (normalize_first) value = round_to_half(value / divisor);
    scalar_values[column] = __float2half_rn(value);
  }
  __syncthreads();

  for (int pair = tid; pair < pair_count; pair += blockDim.x) {
    const float2 adjacent = __half22float2(packed_values[pair]);
    half2 packed = __floats2half2_rn(
        adjacent.x + adjacent.y, adjacent.x - adjacent.y);
    // Pair-index bits below 32 never leave a warp. Keep the applicable stages
    // in registers and exchange packed FP16 values with shuffles, preserving
    // lower-minus-upper orientation while supporting N=16/32 partial warps.
    const unsigned active_mask = __activemask();
#pragma unroll
    for (int bit = 1; bit < 32 && bit < pair_count; bit <<= 1) {
      union Half2Bits {
        half2 value;
        unsigned bits;
      } self, peer;
      self.value = packed;
      peer.bits = __shfl_xor_sync(active_mask, self.bits, bit);
      packed = (pair & bit) == 0
          ? exact_half2_add(packed, peer.value)
          : exact_half2_sub(peer.value, packed);
    }
    packed_values[pair] = packed;
  }

  for (int bit = 32; bit < pair_count; bit <<= 1) {
    __syncthreads();
    for (int pair = tid; pair < pair_count; pair += blockDim.x) {
      const int peer = pair ^ bit;
      if (pair < peer) {
        const half2 first = packed_values[pair];
        const half2 second = packed_values[peer];
        packed_values[pair] = exact_half2_add(first, second);
        packed_values[peer] = exact_half2_sub(first, second);
      }
    }
  }
  __syncthreads();

  for (int pair = tid; pair < pair_count; pair += blockDim.x) {
    const int column0 = 2 * pair;
    const int column1 = column0 + 1;
    const float2 pair_values = __half22float2(packed_values[pair]);
    float value0 = pair_values.x;
    float value1 = pair_values.y;
    if (!normalize_first) {
      value0 = round_to_half(value0 * reciprocal);
      value1 = round_to_half(value1 * reciprocal);
    }
    value0 = round_to_half(value0 * __half2float(scale_v[column0]));
    value1 = round_to_half(value1 * __half2float(scale_v[column1]));
    *reinterpret_cast<half2*>(
        output + static_cast<int64_t>(row) * size_n + column0) =
        __floats2half2_rn(value0, value1);
  }
}

// Rank-8 project plus exact output Hadamard/SV epilogue. The projector keeps
// the existing one-warp, eight-accumulator load/reduction order, while the
// CTA then reuses the existing packed-half2 epilogue. This deliberately uses
// one CTA per row (the same row geometry as the epilogue) so the FP16 hidden
// boundary stays in shared memory instead of making a global round trip.
template <int ProjectionWarps>
__global__ __launch_bounds__(1024) void p32_rank8_project_hadamard_kernel(
    const float* __restrict__ input,
    const half* __restrict__ rank8_a,
    const float* base_output,
    const float* __restrict__ rank8_b,
    const half* __restrict__ scale_v,
    half* output,
    int size_m,
    int size_k,
    int size_n,
  bool normalize_first) {
  constexpr int kRankCount = 8;
  static_assert(ProjectionWarps == 1 || ProjectionWarps == 2 ||
                ProjectionWarps == 4 || ProjectionWarps == 6 ||
                ProjectionWarps == 8);
  constexpr int kRanksPerWarp = kRankCount / ProjectionWarps;
  __shared__ half hidden_row[kRankCount];
  extern __shared__ half2 packed_values[];
  const int tid = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x);

  // Use independent warps for disjoint rank pairs in the projection phase.
  // Each rank still sees the same per-lane K sequence, FP32 FMA order, and
  // descending warp-shuffle tree as p32_rank8_project_kernel<8>. This keeps
  // the exact FP16 hidden boundary while making more of the fused CTA useful
  // before it enters the full-CTA Hadamard epilogue.
  const int warp = tid >> 5;
  if constexpr (ProjectionWarps <= 4) {
    if (warp < ProjectionWarps) {
      const int lane = tid & 31;
      const int rank_base = warp * kRanksPerWarp;
      float accumulators[kRanksPerWarp] = {};
      for (int k = lane; k < size_k; k += 32) {
        const float input_value = input[static_cast<int64_t>(row) * size_k + k];
#pragma unroll
        for (int pair_index = 0; pair_index < kRanksPerWarp / 2; ++pair_index) {
          const half2 weights = *reinterpret_cast<const half2*>(
              rank8_a + static_cast<int64_t>(k) * kRankCount + rank_base +
              pair_index * 2);
          const float2 pair = __half22float2(weights);
          accumulators[2 * pair_index] = fmaf(
              input_value, pair.x, accumulators[2 * pair_index]);
          accumulators[2 * pair_index + 1] = fmaf(
              input_value, pair.y, accumulators[2 * pair_index + 1]);
        }
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int rank = 0; rank < kRanksPerWarp; ++rank) {
          accumulators[rank] += __shfl_down_sync(
              0xffffffffu, accumulators[rank], offset);
        }
      }
      if ((tid & 31) == 0) {
#pragma unroll
        for (int rank = 0; rank < kRanksPerWarp; ++rank) {
          hidden_row[rank_base + rank] = __float2half_rn(accumulators[rank]);
        }
      }
    }
  } else {
    // Six/eight warps cannot use the contiguous equal-sized grouping above.
    // Assign ranks in a warp-strided layout (W6: {0,6},{1,7},{2},{3},{4},{5};
    // W8: one rank per warp). Every rank still follows the original per-lane
    // K order and shuffle reduction, preserving its FP32 accumulation result.
    constexpr int kMaxRanksPerWarp =
        (kRankCount + ProjectionWarps - 1) / ProjectionWarps;
    if (warp < ProjectionWarps) {
      const int lane = tid & 31;
      float accumulators[kMaxRanksPerWarp] = {};
      for (int k = lane; k < size_k; k += 32) {
        const float input_value = input[static_cast<int64_t>(row) * size_k + k];
#pragma unroll
        for (int rank_slot = 0; rank_slot < kMaxRanksPerWarp; ++rank_slot) {
          const int rank = warp + rank_slot * ProjectionWarps;
          if (rank < kRankCount) {
            const half weight = rank8_a[
                static_cast<int64_t>(k) * kRankCount + rank];
            accumulators[rank_slot] = fmaf(
                input_value, __half2float(weight), accumulators[rank_slot]);
          }
        }
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int rank_slot = 0; rank_slot < kMaxRanksPerWarp; ++rank_slot) {
          accumulators[rank_slot] += __shfl_down_sync(
              0xffffffffu, accumulators[rank_slot], offset);
        }
      }
      if ((tid & 31) == 0) {
#pragma unroll
        for (int rank_slot = 0; rank_slot < kMaxRanksPerWarp; ++rank_slot) {
          const int rank = warp + rank_slot * ProjectionWarps;
          if (rank < kRankCount) {
            hidden_row[rank] = __float2half_rn(accumulators[rank_slot]);
          }
        }
      }
    }
  }
  __syncthreads();

  const float sqrt_n = sqrtf(static_cast<float>(size_n));
  const float divisor = round_to_half(sqrt_n);
  const float reciprocal = 1.0f / sqrt_n;
  const int pair_count = size_n / 2;
  half* scalar_values = reinterpret_cast<half*>(packed_values);
  for (int column = tid; column < size_n; column += blockDim.x) {
    float correction = 0.0f;
#pragma unroll
    for (int rank = 0; rank < kRankCount; ++rank) {
      correction += __half2float(hidden_row[rank]) *
          rank8_b[static_cast<int64_t>(rank) * size_n + column];
    }
    float value = round_to_half(
        base_output[static_cast<int64_t>(row) * size_n + column] + correction);
    if (normalize_first) value = round_to_half(value / divisor);
    scalar_values[column] = __float2half_rn(value);
  }
  __syncthreads();

  for (int pair = tid; pair < pair_count; pair += blockDim.x) {
    const float2 adjacent = __half22float2(packed_values[pair]);
    half2 packed = __floats2half2_rn(
        adjacent.x + adjacent.y, adjacent.x - adjacent.y);
    const unsigned active_mask = __activemask();
#pragma unroll
    for (int bit = 1; bit < 32 && bit < pair_count; bit <<= 1) {
      union Half2Bits {
        half2 value;
        unsigned bits;
      } self, peer;
      self.value = packed;
      peer.bits = __shfl_xor_sync(active_mask, self.bits, bit);
      packed = (pair & bit) == 0
          ? exact_half2_add(packed, peer.value)
          : exact_half2_sub(peer.value, packed);
    }
    packed_values[pair] = packed;
  }

  for (int bit = 32; bit < pair_count; bit <<= 1) {
    __syncthreads();
    for (int pair = tid; pair < pair_count; pair += blockDim.x) {
      const int peer = pair ^ bit;
      if (pair < peer) {
        const half2 first = packed_values[pair];
        const half2 second = packed_values[peer];
        packed_values[pair] = exact_half2_add(first, second);
        packed_values[peer] = exact_half2_sub(first, second);
      }
    }
  }
  __syncthreads();

  for (int pair = tid; pair < pair_count; pair += blockDim.x) {
    const int column0 = 2 * pair;
    const int column1 = column0 + 1;
    const float2 pair_values = __half22float2(packed_values[pair]);
    float value0 = pair_values.x;
    float value1 = pair_values.y;
    if (!normalize_first) {
      value0 = round_to_half(value0 * reciprocal);
      value1 = round_to_half(value1 * reciprocal);
    }
    value0 = round_to_half(value0 * __half2float(scale_v[column0]));
    value1 = round_to_half(value1 * __half2float(scale_v[column1]));
    *reinterpret_cast<half2*>(
        output + static_cast<int64_t>(row) * size_n + column0) =
        __floats2half2_rn(value0, value1);
  }
}

}  // namespace

extern "C" int qvq_p32_rank8_epilogue(
    const float* base_output,
    const void* hidden,
    const void* rank8_b,
    float* output,
    int size_m,
    int size_n,
    int rank_count,
    void* stream) {
  if (base_output == nullptr || hidden == nullptr || rank8_b == nullptr ||
      output == nullptr || stream == nullptr) {
    set_last_error("QVQ P32 rank8 epilogue received a null device pointer");
    return -1;
  }
  if (size_m < 1 || size_n < 1 || (rank_count != 8 && rank_count != 16 &&
      rank_count != QVQ_P32_RANK8_MAX_COUNT)) {
    set_last_error("QVQ P32 rank8 epilogue requires M >= 1 and rank_count in {8,16,24}");
    return -1;
  }
  const dim3 grid(
      static_cast<unsigned>((size_n + 127) / 128),
      static_cast<unsigned>(std::min(size_m, 65535)), 1);
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  switch (rank_count) {
    case 8:
      p32_rank8_epilogue_kernel<8><<<grid, 128, 0, cuda_stream>>>(
          base_output, reinterpret_cast<const half*>(hidden),
          reinterpret_cast<const float*>(rank8_b), output, size_m, size_n);
      break;
    case 16:
      p32_rank8_epilogue_kernel<16><<<grid, 128, 0, cuda_stream>>>(
          base_output, reinterpret_cast<const half*>(hidden),
          reinterpret_cast<const float*>(rank8_b), output, size_m, size_n);
      break;
    case 24:
      p32_rank8_epilogue_kernel<24><<<grid, 128, 0, cuda_stream>>>(
          base_output, reinterpret_cast<const half*>(hidden),
          reinterpret_cast<const float*>(rank8_b), output, size_m, size_n);
      break;
  }
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    set_last_error(cudaGetErrorString(error));
    return static_cast<int>(error);
  }
  return 0;
}

extern "C" int qvq_p32_rank8_hadamard_epilogue(
    const float* base_output,
    const void* hidden,
    const void* rank8_b,
    const void* scale_v,
    void* output,
    int size_m,
    int size_n,
    int rank_count,
    int normalize_first,
    void* stream) {
  if (base_output == nullptr || hidden == nullptr || rank8_b == nullptr ||
      scale_v == nullptr || output == nullptr || stream == nullptr) {
    set_last_error("QVQ P32 rank8 Hadamard epilogue received a null device pointer");
    return -1;
  }
  if (size_m < 1 || size_n < 16 || size_n > 16384 ||
      (size_n & (size_n - 1)) != 0 ||
      (rank_count != 8 && rank_count != 16 &&
       rank_count != QVQ_P32_RANK8_MAX_COUNT)) {
    set_last_error(
        "QVQ P32 rank8 Hadamard epilogue requires M >= 1, power-of-two N in [16,16384], and rank_count in {8,16,24}");
    return -1;
  }
  // The packed implementation is row-independent and preserves every scalar
  // FP16 rounding boundary. Extend it through the production M960 prefill
  // bucket while retaining the compatibility path for larger/offline shapes.
  const bool packed_half2 = size_m <= 960;
  const size_t shared_bytes = packed_half2
      ? static_cast<size_t>(size_n) * sizeof(half)
      : static_cast<size_t>(size_n + 2 * (size_n / 32)) * sizeof(half);
  const int threads = packed_half2
      ? std::min(size_n / 2, 1024)
      : std::min(size_n, 1024);
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
#define QVQ_LAUNCH_RANK8_HADAMARD(RANK_COUNT)                              \
  do {                                                                     \
    const cudaError_t attribute_error = cudaFuncSetAttribute(               \
        packed_half2                                                       \
            ? p32_hadamard_epilogue_kernel<RANK_COUNT>                     \
            : p32_hadamard_epilogue_legacy_kernel<RANK_COUNT>,             \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
        static_cast<int>(shared_bytes));                                   \
    if (attribute_error != cudaSuccess) {                                  \
      set_last_error(cudaGetErrorString(attribute_error));                 \
      return static_cast<int>(attribute_error);                            \
    }                                                                      \
    if (packed_half2) {                                                     \
      p32_hadamard_epilogue_kernel<RANK_COUNT>                             \
          <<<static_cast<unsigned>(size_m), threads, shared_bytes, cuda_stream>>>( \
              base_output, reinterpret_cast<const half*>(hidden),          \
              reinterpret_cast<const float*>(rank8_b),                     \
              reinterpret_cast<const half*>(scale_v),                      \
              reinterpret_cast<half*>(output), size_m, size_n,             \
              normalize_first != 0);                                       \
    } else {                                                               \
      p32_hadamard_epilogue_legacy_kernel<RANK_COUNT>                      \
          <<<static_cast<unsigned>(size_m), threads, shared_bytes, cuda_stream>>>( \
              base_output, reinterpret_cast<const half*>(hidden),          \
              reinterpret_cast<const float*>(rank8_b),                     \
              reinterpret_cast<const half*>(scale_v),                      \
              reinterpret_cast<half*>(output), size_m, size_n,             \
              normalize_first != 0);                                       \
    }                                                                      \
  } while (false)
  switch (rank_count) {
    case 8: QVQ_LAUNCH_RANK8_HADAMARD(8); break;
    case 16: QVQ_LAUNCH_RANK8_HADAMARD(16); break;
    case 24: QVQ_LAUNCH_RANK8_HADAMARD(24); break;
  }
#undef QVQ_LAUNCH_RANK8_HADAMARD
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    set_last_error(cudaGetErrorString(error));
    return static_cast<int>(error);
  }
  return 0;
}

extern "C" int qvq_p32_hadamard_epilogue(
    const float* base_output,
    const void* scale_v,
    void* output,
    int size_m,
    int size_n,
    int normalize_first,
    void* stream) {
  if (base_output == nullptr || scale_v == nullptr || output == nullptr ||
      stream == nullptr) {
    set_last_error("QVQ P32 Hadamard epilogue received a null device pointer");
    return -1;
  }
  if (size_m < 1 || size_n < 16 || size_n > 16384 ||
      (size_n & (size_n - 1)) != 0) {
    set_last_error(
        "QVQ P32 Hadamard epilogue requires M >= 1 and power-of-two N in [16,16384]");
    return -1;
  }
  const bool packed_half2 = size_m <= 960;
  const size_t shared_bytes = packed_half2
      ? static_cast<size_t>(size_n) * sizeof(half)
      : static_cast<size_t>(size_n + 2 * (size_n / 32)) * sizeof(half);
  const int threads = packed_half2
      ? std::min(size_n / 2, 1024)
      : std::min(size_n, 1024);
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const cudaError_t attribute_error = cudaFuncSetAttribute(
      packed_half2
          ? p32_hadamard_epilogue_kernel<0>
          : p32_hadamard_epilogue_legacy_kernel<0>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_bytes));
  if (attribute_error != cudaSuccess) {
    set_last_error(cudaGetErrorString(attribute_error));
    return static_cast<int>(attribute_error);
  }
  if (packed_half2) {
    p32_hadamard_epilogue_kernel<0>
        <<<static_cast<unsigned>(size_m), threads, shared_bytes, cuda_stream>>>(
            base_output, nullptr, nullptr,
            reinterpret_cast<const half*>(scale_v),
            reinterpret_cast<half*>(output), size_m, size_n,
            normalize_first != 0);
  } else {
    p32_hadamard_epilogue_legacy_kernel<0>
        <<<static_cast<unsigned>(size_m), threads, shared_bytes, cuda_stream>>>(
            base_output, nullptr, nullptr,
            reinterpret_cast<const half*>(scale_v),
            reinterpret_cast<half*>(output), size_m, size_n,
            normalize_first != 0);
  }
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    set_last_error(cudaGetErrorString(error));
    return static_cast<int>(error);
  }
  return 0;
}

extern "C" int qvq_p32_rank8_project(
    const void* input,
    const void* rank8_a,
    void* hidden,
    int size_m,
    int size_k,
    int rank_count,
    void* stream) {
  if (input == nullptr || rank8_a == nullptr || hidden == nullptr ||
      stream == nullptr) {
    set_last_error("QVQ P32 rank8 project received a null device pointer");
    return -1;
  }
  if (size_m < 1 || size_k < 1 || (rank_count != 8 && rank_count != 16 &&
      rank_count != QVQ_P32_RANK8_MAX_COUNT)) {
    set_last_error("QVQ P32 rank8 project requires M/K >= 1 and rank_count in {8,16,24}");
    return -1;
  }
  constexpr int kRowsPerBlock = 4;
  const dim3 grid(
      static_cast<unsigned>((rank_count + 7) / 8),
      static_cast<unsigned>((std::min(size_m, 65535) + kRowsPerBlock - 1) /
                            kRowsPerBlock),
      1);
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  switch (rank_count) {
    case 8:
      p32_rank8_project_kernel<8><<<grid, 128, 0, cuda_stream>>>(
          reinterpret_cast<const float*>(input),
          reinterpret_cast<const half*>(rank8_a),
          reinterpret_cast<half*>(hidden), size_m, size_k);
      break;
    case 16:
      p32_rank8_project_kernel<16><<<grid, 128, 0, cuda_stream>>>(
          reinterpret_cast<const float*>(input),
          reinterpret_cast<const half*>(rank8_a),
          reinterpret_cast<half*>(hidden), size_m, size_k);
      break;
    case 24:
      p32_rank8_project_kernel<24><<<grid, 128, 0, cuda_stream>>>(
          reinterpret_cast<const float*>(input),
          reinterpret_cast<const half*>(rank8_a),
          reinterpret_cast<half*>(hidden), size_m, size_k);
      break;
  }
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    set_last_error(cudaGetErrorString(error));
    return static_cast<int>(error);
  }
  return 0;
}

extern "C" int qvq_p32_rank8_hadamard_project_tuned(
    const void* input,
    const void* rank8_a,
    const float* base_output,
    const void* rank8_b,
    const void* scale_v,
    void* output,
    int size_m,
    int size_k,
    int size_n,
    int rank_count,
    int normalize_first,
    const struct qvq_p32_rank8_hadamard_config* config,
    void* stream) {
  if (input == nullptr || rank8_a == nullptr || base_output == nullptr ||
      rank8_b == nullptr || scale_v == nullptr || output == nullptr ||
      config == nullptr || stream == nullptr) {
    set_last_error("QVQ P32 fused rank8 project/Hadamard received a null device pointer");
    return -1;
  }
  if (config->version != QVQ_P32_RANK8_CONFIG_VERSION ||
      config->struct_bytes < sizeof(struct qvq_p32_rank8_hadamard_config) ||
      (config->tuning_mode != QVQ_P32_TUNING_AUTO &&
       config->tuning_mode != QVQ_P32_TUNING_EXTERNAL)) {
    set_last_error("QVQ P32 rank8 fused config version, size, or tuning_mode is invalid");
    return -1;
  }
  if (size_m < 1 || size_m > 960 || size_k < 1 || size_n < 16 ||
      size_n > 16384 || (size_n & (size_n - 1)) != 0 || rank_count != 8) {
    set_last_error(
        "QVQ P32 fused rank8 project/Hadamard requires M in [1,960], K >= 1, power-of-two N in [16,16384], and rank_count=8");
    return -1;
  }
  if (config->tuning_mode == QVQ_P32_TUNING_EXTERNAL &&
      (config->projection_warps == QVQ_P32_RANK8_PROJECT_WARPS_AUTO ||
       config->threads == QVQ_P32_RANK8_THREADS_AUTO ||
       config->rows_per_cta != QVQ_P32_RANK8_ROWS_PER_CTA)) {
    set_last_error(
        "QVQ P32 external rank8 tuning requires explicit projection_warps, threads, and rows_per_cta=1");
    return -1;
  }
  if (config->rows_per_cta != 0 &&
      config->rows_per_cta != QVQ_P32_RANK8_ROWS_PER_CTA) {
    set_last_error("QVQ P32 fused rank8 currently supports rows_per_cta=1 only");
    return -1;
  }
  const size_t shared_bytes = static_cast<size_t>(size_n) * sizeof(half);
  const int threads = config->threads == QVQ_P32_RANK8_THREADS_AUTO
      ? std::max(32, std::min(size_n / 2, 1024))
      : config->threads;
  if (threads < 32 || threads > 1024 || (threads & 31) != 0) {
    set_last_error("QVQ P32 fused rank8 threads must be a warp multiple in [32,1024]");
    return -1;
  }
  const int projection_warps =
      config->projection_warps == QVQ_P32_RANK8_PROJECT_WARPS_AUTO
      ? std::min(4, threads / 32)
      : config->projection_warps;
  if ((projection_warps != 1 && projection_warps != 2 &&
       projection_warps != 4 && projection_warps != 6 &&
       projection_warps != 8) || projection_warps > threads / 32) {
    set_last_error(
        "QVQ P32 fused rank8 projection_warps must be 1, 2, 4, 6, or 8 and no greater than the CTA warp count");
    return -1;
  }
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t attribute_error = cudaSuccess;
  if (projection_warps == 8) {
    attribute_error = cudaFuncSetAttribute(
        p32_rank8_project_hadamard_kernel<8>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (attribute_error == cudaSuccess) {
      p32_rank8_project_hadamard_kernel<8>
          <<<static_cast<unsigned>(size_m), threads, shared_bytes, cuda_stream>>>(
              reinterpret_cast<const float*>(input),
              reinterpret_cast<const half*>(rank8_a), base_output,
              reinterpret_cast<const float*>(rank8_b),
              reinterpret_cast<const half*>(scale_v),
              reinterpret_cast<half*>(output), size_m, size_k, size_n,
              normalize_first != 0);
    }
  } else if (projection_warps == 6) {
    attribute_error = cudaFuncSetAttribute(
        p32_rank8_project_hadamard_kernel<6>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (attribute_error == cudaSuccess) {
      p32_rank8_project_hadamard_kernel<6>
          <<<static_cast<unsigned>(size_m), threads, shared_bytes, cuda_stream>>>(
              reinterpret_cast<const float*>(input),
              reinterpret_cast<const half*>(rank8_a), base_output,
              reinterpret_cast<const float*>(rank8_b),
              reinterpret_cast<const half*>(scale_v),
              reinterpret_cast<half*>(output), size_m, size_k, size_n,
              normalize_first != 0);
    }
  } else if (projection_warps == 4) {
    attribute_error = cudaFuncSetAttribute(
        p32_rank8_project_hadamard_kernel<4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (attribute_error == cudaSuccess) {
      p32_rank8_project_hadamard_kernel<4>
          <<<static_cast<unsigned>(size_m), threads, shared_bytes, cuda_stream>>>(
              reinterpret_cast<const float*>(input),
              reinterpret_cast<const half*>(rank8_a), base_output,
              reinterpret_cast<const float*>(rank8_b),
              reinterpret_cast<const half*>(scale_v),
              reinterpret_cast<half*>(output), size_m, size_k, size_n,
              normalize_first != 0);
    }
  } else if (projection_warps == 2) {
    attribute_error = cudaFuncSetAttribute(
        p32_rank8_project_hadamard_kernel<2>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (attribute_error == cudaSuccess) {
      p32_rank8_project_hadamard_kernel<2>
          <<<static_cast<unsigned>(size_m), threads, shared_bytes, cuda_stream>>>(
              reinterpret_cast<const float*>(input),
              reinterpret_cast<const half*>(rank8_a), base_output,
              reinterpret_cast<const float*>(rank8_b),
              reinterpret_cast<const half*>(scale_v),
              reinterpret_cast<half*>(output), size_m, size_k, size_n,
              normalize_first != 0);
    }
  } else {
    attribute_error = cudaFuncSetAttribute(
        p32_rank8_project_hadamard_kernel<1>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (attribute_error == cudaSuccess) {
      p32_rank8_project_hadamard_kernel<1>
          <<<static_cast<unsigned>(size_m), threads, shared_bytes, cuda_stream>>>(
              reinterpret_cast<const float*>(input),
              reinterpret_cast<const half*>(rank8_a), base_output,
              reinterpret_cast<const float*>(rank8_b),
              reinterpret_cast<const half*>(scale_v),
              reinterpret_cast<half*>(output), size_m, size_k, size_n,
              normalize_first != 0);
    }
  }
  if (attribute_error != cudaSuccess) {
    set_last_error(cudaGetErrorString(attribute_error));
    return static_cast<int>(attribute_error);
  }
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    set_last_error(cudaGetErrorString(error));
    return static_cast<int>(error);
  }
  return 0;
}

extern "C" int qvq_p32_rank8_hadamard_project(
    const void* input,
    const void* rank8_a,
    const float* base_output,
    const void* rank8_b,
    const void* scale_v,
    void* output,
    int size_m,
    int size_k,
    int size_n,
    int rank_count,
    int normalize_first,
    void* stream) {
  const struct qvq_p32_rank8_hadamard_config config = {
      QVQ_P32_RANK8_CONFIG_VERSION,
      sizeof(struct qvq_p32_rank8_hadamard_config),
      QVQ_P32_TUNING_AUTO,
      QVQ_P32_RANK8_PROJECT_WARPS_AUTO,
      QVQ_P32_RANK8_THREADS_AUTO,
      QVQ_P32_RANK8_ROWS_PER_CTA,
  };
  return qvq_p32_rank8_hadamard_project_tuned(
      input, rank8_a, base_output, rank8_b, scale_v, output, size_m, size_k,
      size_n, rank_count, normalize_first, &config, stream);
}
