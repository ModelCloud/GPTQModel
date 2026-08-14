// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kTileRows = 16;
constexpr int kTileColumns = 16;
constexpr int kTileValues = kTileRows * kTileColumns;
constexpr int kThreads = 256;
constexpr int kRowsPerBlock = 32;
constexpr int kPgc16LevelCount = 256;
constexpr uint32_t kPgc16Multiplier = 40503u;
constexpr uint32_t kPgc16Increment = 17011u;

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

template <>
struct ScalarTraits<float> {
  static __device__ __forceinline__ float to_float(float value) { return value; }
  static __device__ __forceinline__ float from_float(float value) { return value; }
};

template <int Remaining, int Offset = 0>
struct PlanarTransitionDecoder {
  static __device__ __forceinline__ uint32_t decode(const uint32_t* words, int base, int lane) {
    constexpr int width = Remaining >= 16 ? 16 : Remaining >= 8 ? 8 : Remaining >= 4 ? 4 : Remaining >= 2 ? 2 : 1;
    constexpr int pack_factor = 32 / width;
    const uint32_t code =
        (words[base + Offset + lane / pack_factor] >> (width * (lane % pack_factor))) & ((1u << width) - 1u);
    return (code << Offset) | PlanarTransitionDecoder<Remaining - width, Offset + width>::decode(words, base, lane);
  }
};

template <int Offset>
struct PlanarTransitionDecoder<0, Offset> {
  static __device__ __forceinline__ uint32_t decode(const uint32_t*, int, int) { return 0; }
};

template <int TransitionBits>
__device__ __forceinline__ uint32_t planar_transition(const uint32_t* words, int edge) {
  const int block = edge >> 5;
  const int lane = edge & 31;
  return PlanarTransitionDecoder<TransitionBits>::decode(words, block * TransitionBits, lane);
}

template <int TransitionBits, int VectorSize = 2>
__device__ __forceinline__ uint32_t qvq_state(const uint32_t* words, int pair) {
  constexpr int total_edges = 256 / VectorSize;
  constexpr int edge_count = (15 + TransitionBits) / TransitionBits;
  if constexpr (edge_count == 1) {
    // W4 has one 16-bit transition per state.  Keep this hot path free of
    // the generic wrapped recurrence; total_edges is a power of two for both
    // vector layouts, and pair is already in range.
    return planar_transition<TransitionBits>(words, pair & (total_edges - 1));
  }
  const int first = (pair + total_edges - edge_count + 1) & (total_edges - 1);
  uint32_t state = 0;
#pragma unroll
  for (int j = 0; j < edge_count; ++j) {
    const int edge = (first + j) & (total_edges - 1);
    state = ((state << TransitionBits) | planar_transition<TransitionBits>(words, edge)) & 0xffffu;
  }
  return state;
}

__device__ __forceinline__ uint32_t pgc16_bank_mask_runtime(int transition_bits, uint32_t bank) {
  constexpr uint32_t masks[7][4] = {
      {0xA5A5u, 0x5A5Au, 0x3C3Cu, 0xC3C3u},
      {0xA5A5u, 0x5A5Au, 0x9696u, 0x6969u},
      {0xA5A5u, 0x3C3Cu, 0x5A5Au, 0xC3C3u},
      {0xA5A5u, 0x9696u, 0x3C3Cu, 0xC3C3u},
      {0xA5A5u, 0x6969u, 0x5A5Au, 0x3C3Cu},
      {0xA5A5u, 0xC3C3u, 0x9696u, 0x5A5Au},
      {0xA5A5u, 0x3C3Cu, 0x9696u, 0x6969u},
  };
  const int row = transition_bits <= 4 ? 0 : (transition_bits >= 16 ? 6 : (transition_bits - 4) / 2);
  return masks[row][bank & 3u];
}

__device__ __noinline__ uint32_t planar_transition_runtime(
    const uint32_t* words, int edge, int transition_bits) {
  const int block = edge >> 5;
  const int lane = edge & 31;
  int remaining = transition_bits;
  int offset = 0;
  uint32_t result = 0;
  while (remaining > 0) {
    const int width = remaining >= 16 ? 16 : remaining >= 8 ? 8 : remaining >= 4 ? 4 : remaining >= 2 ? 2 : 1;
    const int pack_factor = 32 / width;
    const uint32_t code =
        (words[block * transition_bits + offset + lane / pack_factor] >> (width * (lane % pack_factor))) &
        ((1u << width) - 1u);
    result |= code << offset;
    offset += width;
    remaining -= width;
  }
  return result;
}

template <int VectorSize>
__device__ __noinline__ uint32_t qvq_state_runtime(
    const uint32_t* words, int pair, int transition_bits) {
  const int total_edges = 256 / VectorSize;
  const int edge_mask = total_edges - 1;
  const int edge_count = (15 + transition_bits) / transition_bits;
  if (edge_count == 1) {
    return planar_transition_runtime(words, pair & edge_mask, transition_bits);
  }
  const int first = (pair + total_edges - edge_count + 1) & edge_mask;
  uint32_t state = 0;
  for (int j = 0; j < edge_count; ++j) {
    const int edge = (first + j) & edge_mask;
    state = ((state << transition_bits) |
             planar_transition_runtime(words, edge, transition_bits)) & 0xffffu;
  }
  return state;
}

__device__ __forceinline__ uint32_t pgc16_mix(uint32_t state) {
  uint32_t mixed = state ^ (state >> 8);
  mixed = (mixed * kPgc16Multiplier + kPgc16Increment) & 0xffffu;
  return mixed ^ (mixed >> 7);
}

// Shared decode path for normal and split-K GEMV.  Keeping this logic in one
// device helper prevents the two accumulation kernels from drifting while
// retaining compile-time V2/V4 and rate specialization at the call site.
template <int VectorSize>
__device__ __forceinline__ float qvq_decode_weight(
    const uint32_t* packed_words,
    uint8_t bank_id,
    const half* cached_levels,
    int local,
    int transition_bits) {
  const int col = local & 15;
  const int v4_group = local >> 2;
  uint32_t mixed = 0;
  uint32_t mixed2 = 0;
  if constexpr (VectorSize == 4) {
    uint32_t mixed_pair = 0;
    if ((col & 3) == 0) {
      const uint32_t state = qvq_state_runtime<VectorSize>(packed_words, v4_group, transition_bits);
      mixed_pair = pgc16_mix(state) |
          (pgc16_mix(state ^ pgc16_bank_mask_runtime(transition_bits, bank_id)) << 16);
    }
    mixed_pair = __shfl_sync(0xffffffffu, mixed_pair, local & ~3);
    mixed = mixed_pair & 0xffffu;
    mixed2 = mixed_pair >> 16;
  } else {
    const uint32_t state = qvq_state_runtime<VectorSize>(packed_words, local >> 1, transition_bits);
    mixed = pgc16_mix(state);
    mixed2 = pgc16_mix(state ^ 0xA5A5u);
  }

  if constexpr (VectorSize == 4) {
    uint32_t level_pair0 = 0;
    uint32_t level_pair1 = 0;
    if ((col & 3) == 0) {
      level_pair0 = static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed >> 8])) |
          (static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed & 0xffu])) << 16);
      level_pair1 = static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed2 >> 8])) |
          (static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed2 & 0xffu])) << 16);
    }
    level_pair0 = __shfl_sync(0xffffffffu, level_pair0, local & ~3);
    level_pair1 = __shfl_sync(0xffffffffu, level_pair1, local & ~3);
    const int component = col & 3;
    const uint32_t level_bits = component < 2 ? (level_pair0 >> (16 * component))
                                               : (level_pair1 >> (16 * (component - 2)));
    return __half2float(__ushort_as_half(static_cast<unsigned short>(level_bits)));
  }

  const uint32_t level_index = (col & 1) == 0 ? mixed >> 8 : mixed & 0xffu;
  return __half2float(cached_levels[level_index]);
}

// Decode / small-batch GEMV, specialized on the compile-time row count ROWS.
//
// Layout: one block per (n-tile x ROWS-row stripe), 256 threads as 16 k-slots x
// 16 columns. Every thread decodes the weight it needs into a register (no
// shared decoded_weight round trip) and accumulates ROWS row dot products per
// k-tile, so all 256 threads stay busy even at M=1 (the previous layout used
// only 16 threads for the FMAs at M=1). Each k-tile needs one __syncthreads
// (after the shared loads) instead of three; ncu showed barrier stalls were
// 65% of the old kernel's stall cycles. The k-slot partials are reduced
// deterministically (kslot 0..15 order) at the end. OutputScalar retains the
// range-safe FP32 accumulator from the overflow fix (the value is narrowed
// only in the fused output transform).
//
// ROWS is one of {1, 8, 16, 32}: dispatch rounds M up to the next supported
// value and the row loop is fully unrolled at compile time, avoiding per-k-tile
// predication overhead for small M.
template <typename Scalar, typename OutputScalar, int ROWS, int VectorSize = 2>
__global__ __launch_bounds__(kThreads) void qvq_gemv_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ trellis,
    const uint8_t* __restrict__ bank_ids,
    const half* __restrict__ levels,
    OutputScalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits) {
  __shared__ half cached_levels[kPgc16LevelCount];
  __shared__ float reduced[kThreads / 32][ROWS * kTileColumns];  // [warp][row*16+col]

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int n_tile = static_cast<int>(blockIdx.x);
  const int m0 = static_cast<int>(blockIdx.y) * ROWS;
  const int block_rows = min(ROWS, size_m - m0);
  const int n0 = n_tile * kTileColumns;
  const int n_tiles = size_n / kTileColumns;
  const int k_tiles = size_k / kTileRows;

  const int col = thread & 15;
  const int k_slot = thread >> 4;

  cached_levels[thread] = levels[thread];
  __syncthreads();

  float accumulator[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    accumulator[r] = 0.0f;
  }

  // k-loop batched by kUnroll tiles per sync pair: all kUnroll tiles' trellis
  // words and input rows are loaded before one barrier, then all kUnroll tiles
  // are decoded/FMA'd before the next barrier. The batched loads overlap with
  // kUnroll x the per-tile decode work, hiding the global load latency that
  // dominated the per-tile barrier waits (ncu: 65% barrier stalls).
  // Keep the high-unroll path for the small-row kernels where it hides
  // global-load latency without inflating the register/shared-memory image
  // of the wide-row specializations.
  constexpr int kUnroll = ROWS <= 8 ? 24 : (ROWS == 16 ? 12 : 8);
  constexpr int kMaxWords = VectorSize == 2 ? 64 : 32;
  __shared__ uint32_t packed_words[kUnroll][kMaxWords];
  __shared__ uint8_t packed_bank_ids[kUnroll];
  __shared__ Scalar input_tile[kUnroll][ROWS * kTileRows];

  for (int kb = 0; kb < k_tiles; kb += kUnroll) {
    const int tiles_here = min(kUnroll, k_tiles - kb);
    const int words_per_tile = (VectorSize == 2 ? 4 : 2) * transition_bits;
    for (int index = thread; index < tiles_here * words_per_tile; index += kThreads) {
      const int u = index / words_per_tile;
      const int w = index % words_per_tile;
      const int tile_index = (kb + u) * n_tiles + n_tile;
      const int32_t* tile = trellis + static_cast<int64_t>(tile_index) * words_per_tile;
      packed_words[u][w] = static_cast<uint32_t>(tile[w]);
    }
    if (thread < tiles_here) {
      const int tile_index = (kb + thread) * n_tiles + n_tile;
      packed_bank_ids[thread] = bank_ids == nullptr ? 0 : bank_ids[tile_index];
    }
    for (int index = thread; index < tiles_here * ROWS * kTileRows; index += kThreads) {
      const int u = index / (ROWS * kTileRows);
      const int cell = index % (ROWS * kTileRows);
      const int row = cell / kTileRows;
      const int k_local = cell % kTileRows;
      input_tile[u][cell] = row < block_rows
          ? input[static_cast<int64_t>(m0 + row) * size_k + (kb + u) * kTileRows + k_local]
          : ScalarTraits<Scalar>::from_float(0.0f);
    }
    __syncthreads();

#pragma unroll 4
    for (int u = 0; u < kUnroll; ++u) {
      if (u < tiles_here) {
        // Decode the single weight this thread needs (register; no shared write).
        const int local = (k_slot << 4) | col;
        const float weight = qvq_decode_weight<VectorSize>(
            packed_words[u], bank_ids == nullptr ? 0 : packed_bank_ids[u], cached_levels, local, transition_bits);

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          accumulator[r] = fmaf(
              ScalarTraits<Scalar>::to_float(input_tile[u][r * kTileRows + k_slot]), weight, accumulator[r]);
        }
      }
    }
    __syncthreads();
  }

  // Deterministic k-slot reduction: each warp sums its 2 k-slots via shuffles,
  // then warp 0 sums the 8 warp partials in ascending warp order (kslot 0..15).
  for (int r = 0; r < block_rows; ++r) {
    float partial = accumulator[r];
    partial += __shfl_xor_sync(0xffffffffu, partial, 16);
    if ((thread & 16) == 0) {
      reduced[warp][r * kTileColumns + col] = partial;
    }
  }
  __syncthreads();

  if (warp == 0) {
    for (int r = 0; r < block_rows; ++r) {
      if (thread < kTileColumns) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < kThreads / 32; ++w) {
          total += reduced[w][r * kTileColumns + thread];
        }
        output[static_cast<int64_t>(m0 + r) * size_n + n0 + thread] =
            ScalarTraits<OutputScalar>::from_float(total);
      }
    }
  }
}

// Split-K variant of the row-specialized GEMV for projections that do not
// expose enough N tiles to occupy the GPU: each block owns a contiguous k-tile
// range and writes FP32 partials reduced by qvq_reduce_splitk_kernel in a fixed
// order.
template <typename Scalar, int ROWS, int VectorSize = 2>
__global__ __launch_bounds__(kThreads) void qvq_gemv_splitk_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ trellis,
    const uint8_t* __restrict__ bank_ids,
    const half* __restrict__ levels,
    float* __restrict__ partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    int transition_bits) {
  __shared__ half cached_levels[kPgc16LevelCount];
  __shared__ float reduced[kThreads / 32][ROWS * kTileColumns];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int n_tile = static_cast<int>(blockIdx.x);
  const int m0 = static_cast<int>(blockIdx.y) * ROWS;
  const int split = static_cast<int>(blockIdx.z);
  const int block_rows = min(ROWS, size_m - m0);
  const int n0 = n_tile * kTileColumns;
  const int n_tiles = size_n / kTileColumns;
  const int k_tiles = size_k / kTileRows;
  const int k_tile_begin = (k_tiles * split) / split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / split_count;

  const int col = thread & 15;
  const int k_slot = thread >> 4;

  cached_levels[thread] = levels[thread];
  __syncthreads();

  float accumulator[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    accumulator[r] = 0.0f;
  }

  constexpr int kUnroll = ROWS <= 8 ? 24 : (ROWS == 16 ? 12 : 8);
  constexpr int kMaxWords = VectorSize == 2 ? 64 : 32;
  __shared__ uint32_t packed_words[kUnroll][kMaxWords];
  __shared__ uint8_t packed_bank_ids[kUnroll];
  __shared__ Scalar input_tile[kUnroll][ROWS * kTileRows];

  for (int kb = k_tile_begin; kb < k_tile_end; kb += kUnroll) {
    const int tiles_here = min(kUnroll, k_tile_end - kb);
    const int words_per_tile = (VectorSize == 2 ? 4 : 2) * transition_bits;
    for (int index = thread; index < tiles_here * words_per_tile; index += kThreads) {
      const int u = index / words_per_tile;
      const int w = index % words_per_tile;
      const int tile_index = (kb + u) * n_tiles + n_tile;
      const int32_t* tile = trellis + static_cast<int64_t>(tile_index) * words_per_tile;
      packed_words[u][w] = static_cast<uint32_t>(tile[w]);
    }
    if (thread < tiles_here) {
      const int tile_index = (kb + thread) * n_tiles + n_tile;
      packed_bank_ids[thread] = bank_ids == nullptr ? 0 : bank_ids[tile_index];
    }
    for (int index = thread; index < tiles_here * ROWS * kTileRows; index += kThreads) {
      const int u = index / (ROWS * kTileRows);
      const int cell = index % (ROWS * kTileRows);
      const int row = cell / kTileRows;
      const int k_local = cell % kTileRows;
      input_tile[u][cell] = row < block_rows
          ? input[static_cast<int64_t>(m0 + row) * size_k + (kb + u) * kTileRows + k_local]
          : ScalarTraits<Scalar>::from_float(0.0f);
    }
    __syncthreads();

#pragma unroll 4
    for (int u = 0; u < kUnroll; ++u) {
      if (u < tiles_here) {
        const int local = (k_slot << 4) | col;
        const float weight = qvq_decode_weight<VectorSize>(
            packed_words[u], bank_ids == nullptr ? 0 : packed_bank_ids[u], cached_levels, local, transition_bits);

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          accumulator[r] = fmaf(
              ScalarTraits<Scalar>::to_float(input_tile[u][r * kTileRows + k_slot]), weight, accumulator[r]);
        }
      }
    }
    __syncthreads();
  }

  for (int r = 0; r < block_rows; ++r) {
    float partial = accumulator[r];
    partial += __shfl_xor_sync(0xffffffffu, partial, 16);
    if ((thread & 16) == 0) {
      reduced[warp][r * kTileColumns + col] = partial;
    }
  }
  __syncthreads();

  if (warp == 0) {
    for (int r = 0; r < block_rows; ++r) {
      if (thread < kTileColumns) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < kThreads / 32; ++w) {
          total += reduced[w][r * kTileColumns + thread];
        }
        partial_output[static_cast<int64_t>(split) * size_m * size_n +
                       static_cast<int64_t>(m0 + r) * size_n + n0 + thread] = total;
      }
    }
  }
}

template <typename OutputScalar>
__global__ void qvq_reduce_splitk_kernel(
    const float* __restrict__ partial_output,
    OutputScalar* __restrict__ output,
    int size_m,
    int size_n,
    int split_count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t output_values = static_cast<int64_t>(size_m) * size_n;
  if (index >= output_values) {
    return;
  }
  float accumulator = 0.0f;
  for (int split = 0; split < split_count; ++split) {
    accumulator += partial_output[static_cast<int64_t>(split) * output_values + index];
  }
  output[index] = ScalarTraits<OutputScalar>::from_float(accumulator);
}

// M>=32 reuses each decoded tile through Ampere tensor cores. The trellis and
// PGC16 decode is identical to the scalar kernel; only the 16x16 accumulation is
// replaced by WMMA with an FP32 accumulator. Partial M tiles are explicitly
// zero-padded in shared memory, so the path does not read beyond input storage.
template <typename Scalar, typename OutputScalar, int TransitionBits>
__global__ __launch_bounds__(kThreads) void qvq_gemm_wmma_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ trellis,
    const Scalar* __restrict__ levels,
    float* __restrict__ partial_output,
    OutputScalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int split_count) {
  __shared__ uint32_t packed_words[4 * TransitionBits];
  __shared__ Scalar decoded_weight[kTileValues];
  __shared__ Scalar input_tile[kRowsPerBlock * kTileRows];
  __shared__ Scalar cached_levels[kPgc16LevelCount];
  __shared__ float output_tile[kRowsPerBlock * kTileColumns];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int n_tile = static_cast<int>(blockIdx.x);
  const int m0 = static_cast<int>(blockIdx.y) * kRowsPerBlock;
  const int split = static_cast<int>(blockIdx.z);
  const int block_rows = min(kRowsPerBlock, size_m - m0);
  const int n0 = n_tile * kTileColumns;
  const int n_tiles = size_n / kTileColumns;
  const int k_tiles = size_k / kTileRows;
  const int k_tile_begin = (k_tiles * split) / split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / split_count;
  const int active_warps = (block_rows + 15) / 16;

  cached_levels[thread] = levels[thread];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, Scalar, wmma::row_major> input_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, Scalar, wmma::row_major> weight_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  __syncthreads();

  for (int k_tile = k_tile_begin; k_tile < k_tile_end; ++k_tile) {
    const int tile_index = k_tile * n_tiles + n_tile;
    const int32_t* tile = trellis + static_cast<int64_t>(tile_index) * 4 * TransitionBits;
    for (int index = thread; index < 4 * TransitionBits; index += kThreads) {
      packed_words[index] = static_cast<uint32_t>(tile[index]);
    }
    for (int index = thread; index < kRowsPerBlock * kTileRows; index += kThreads) {
      const int row = index / kTileRows;
      const int k_local = index % kTileRows;
      input_tile[index] = row < block_rows
          ? input[static_cast<int64_t>(m0 + row) * size_k + k_tile * kTileRows + k_local]
          : ScalarTraits<Scalar>::from_float(0.0f);
    }
    __syncthreads();

    const int local = thread;
    const uint32_t state = qvq_state<TransitionBits>(packed_words, local >> 1);
    const uint32_t mixed = pgc16_mix(state);
    const uint32_t level_index = (local & 1) == 0 ? mixed >> 8 : mixed & 0xffu;
    const float value = ScalarTraits<Scalar>::to_float(cached_levels[level_index]);
    decoded_weight[local] = ScalarTraits<Scalar>::from_float(value);
    __syncthreads();

    if (warp < active_warps) {
      wmma::load_matrix_sync(input_fragment, input_tile + warp * 16 * kTileRows, kTileRows);
      wmma::load_matrix_sync(weight_fragment, decoded_weight, kTileColumns);
      wmma::mma_sync(accumulator, input_fragment, weight_fragment, accumulator);
    }
    __syncthreads();
  }

  if (warp < active_warps) {
    wmma::store_matrix_sync(
        output_tile + warp * 16 * kTileColumns, accumulator, kTileColumns, wmma::mem_row_major);
  }
  __syncthreads();
  for (int index = thread; index < block_rows * kTileColumns; index += kThreads) {
    const int row = index / kTileColumns;
    const int n_local = index % kTileColumns;
    const int64_t output_index = static_cast<int64_t>(m0 + row) * size_n + n0 + n_local;
    if (split_count == 1) {
      output[output_index] = ScalarTraits<OutputScalar>::from_float(output_tile[index]);
    } else {
      partial_output[static_cast<int64_t>(split) * size_m * size_n + output_index] = output_tile[index];
    }
  }
}

// Round M up to the supported compile-time row specialization.
constexpr int qvq_rows_for_m(int m) {
  return m <= 1 ? 1 : (m <= 8 ? 8 : (m <= 16 ? 16 : kRowsPerBlock));
}

template <typename Scalar, typename OutputScalar, int VectorSize = 2>
void launch_qvq_gemv(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor* bank_ids,
    const at::Tensor& levels,
    at::Tensor& output,
    int transition_bits,
    cudaStream_t stream) {
  const int rows = qvq_rows_for_m(static_cast<int>(input.size(0)));
  const dim3 grid(
      static_cast<unsigned int>(output.size(1) / kTileColumns),
      static_cast<unsigned int>((input.size(0) + rows - 1) / rows));
  const Scalar* input_ptr = reinterpret_cast<const Scalar*>(input.const_data_ptr());
  const half* levels_ptr = reinterpret_cast<const half*>(levels.const_data_ptr());
  OutputScalar* output_ptr = reinterpret_cast<OutputScalar*>(output.mutable_data_ptr());
  const int32_t* trellis_ptr = trellis.const_data_ptr<int32_t>();
  const uint8_t* bank_ids_ptr = bank_ids == nullptr ? nullptr : bank_ids->const_data_ptr<uint8_t>();
#define QVQ_LAUNCH(ROWS)                                                                                           \
  qvq_gemv_kernel<Scalar, OutputScalar, ROWS, VectorSize><<<grid, kThreads, 0, stream>>>(                       \
      input_ptr, trellis_ptr, bank_ids_ptr, levels_ptr, output_ptr, static_cast<int>(input.size(0)),             \
      static_cast<int>(input.size(1)), static_cast<int>(output.size(1)), transition_bits)
  if (rows == 1) {
    QVQ_LAUNCH(1);
  } else if (rows == 8) {
    QVQ_LAUNCH(8);
  } else if (rows == 16) {
    QVQ_LAUNCH(16);
  } else {
    QVQ_LAUNCH(kRowsPerBlock);
  }
#undef QVQ_LAUNCH
}

template <typename Scalar, typename OutputScalar, int VectorSize = 2>
void launch_qvq_gemv_splitk(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor* bank_ids,
    const at::Tensor& levels,
    at::Tensor& partial_output,
    at::Tensor& output,
    int transition_bits,
    int split_count,
    cudaStream_t stream) {
  float* partial_ptr = partial_output.mutable_data_ptr<float>();
  const int rows = qvq_rows_for_m(static_cast<int>(input.size(0)));
  const dim3 grid(
      static_cast<unsigned int>(output.size(1) / kTileColumns),
      static_cast<unsigned int>((input.size(0) + rows - 1) / rows),
      static_cast<unsigned int>(split_count));
  const Scalar* input_ptr = reinterpret_cast<const Scalar*>(input.const_data_ptr());
  const half* levels_ptr = reinterpret_cast<const half*>(levels.const_data_ptr());
  OutputScalar* output_ptr = reinterpret_cast<OutputScalar*>(output.mutable_data_ptr());
  const int32_t* trellis_ptr = trellis.const_data_ptr<int32_t>();
  const uint8_t* bank_ids_ptr = bank_ids == nullptr ? nullptr : bank_ids->const_data_ptr<uint8_t>();
#define QVQ_SPLITK_LAUNCH(ROWS)                                                                                   \
  qvq_gemv_splitk_kernel<Scalar, ROWS, VectorSize><<<grid, kThreads, 0, stream>>>(                               \
      input_ptr, trellis_ptr, bank_ids_ptr, levels_ptr, partial_ptr, static_cast<int>(input.size(0)),            \
      static_cast<int>(input.size(1)), static_cast<int>(output.size(1)), split_count, transition_bits)
  if (rows == 1) {
    QVQ_SPLITK_LAUNCH(1);
  } else if (rows == 8) {
    QVQ_SPLITK_LAUNCH(8);
  } else if (rows == 16) {
    QVQ_SPLITK_LAUNCH(16);
  } else {
    QVQ_SPLITK_LAUNCH(kRowsPerBlock);
  }
#undef QVQ_SPLITK_LAUNCH
  const int64_t output_values = output.numel();
  const int reduction_blocks = static_cast<int>((output_values + kThreads - 1) / kThreads);
  qvq_reduce_splitk_kernel<OutputScalar><<<reduction_blocks, kThreads, 0, stream>>>(
      partial_ptr, output_ptr, static_cast<int>(output.size(0)), static_cast<int>(output.size(1)), split_count);
}

template <typename Scalar, typename OutputScalar>
void launch_qvq_gemm_wmma(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    at::Tensor& output,
    int transition_bits,
    cudaStream_t stream) {
  const dim3 grid(
      static_cast<unsigned int>(output.size(1) / kTileColumns),
      static_cast<unsigned int>((output.size(0) + kRowsPerBlock - 1) / kRowsPerBlock),
      1);
  const Scalar* input_ptr = reinterpret_cast<const Scalar*>(input.const_data_ptr());
  const Scalar* levels_ptr = reinterpret_cast<const Scalar*>(levels.const_data_ptr());
  OutputScalar* output_ptr = reinterpret_cast<OutputScalar*>(output.mutable_data_ptr());
  const int32_t* trellis_ptr = trellis.const_data_ptr<int32_t>();
#define QVQ_WMMA_LAUNCH(BITS)                                                                                     \
  qvq_gemm_wmma_kernel<Scalar, OutputScalar, BITS><<<grid, kThreads, 0, stream>>>(                                \
      input_ptr, trellis_ptr, levels_ptr, nullptr, output_ptr, static_cast<int>(input.size(0)),                    \
      static_cast<int>(input.size(1)), static_cast<int>(output.size(1)), 1)
  switch (transition_bits) {
    case 2: QVQ_WMMA_LAUNCH(2); break;
    case 3: QVQ_WMMA_LAUNCH(3); break;
    case 4: QVQ_WMMA_LAUNCH(4); break;
    case 5: QVQ_WMMA_LAUNCH(5); break;
    case 6: QVQ_WMMA_LAUNCH(6); break;
    case 7: QVQ_WMMA_LAUNCH(7); break;
    case 8: QVQ_WMMA_LAUNCH(8); break;
    case 9: QVQ_WMMA_LAUNCH(9); break;
    case 10: QVQ_WMMA_LAUNCH(10); break;
    case 11: QVQ_WMMA_LAUNCH(11); break;
    case 12: QVQ_WMMA_LAUNCH(12); break;
    case 13: QVQ_WMMA_LAUNCH(13); break;
    case 14: QVQ_WMMA_LAUNCH(14); break;
    case 15: QVQ_WMMA_LAUNCH(15); break;
    case 16: QVQ_WMMA_LAUNCH(16); break;
  }
#undef QVQ_WMMA_LAUNCH
}

template <typename Scalar, typename OutputScalar>
void launch_qvq_gemm_wmma_splitk(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    at::Tensor& partial_output,
    at::Tensor& output,
    int transition_bits,
    int split_count,
    cudaStream_t stream) {
  float* partial_ptr = partial_output.mutable_data_ptr<float>();
  const dim3 grid(
      static_cast<unsigned int>(output.size(1) / kTileColumns),
      static_cast<unsigned int>((output.size(0) + kRowsPerBlock - 1) / kRowsPerBlock),
      static_cast<unsigned int>(split_count));
  const Scalar* input_ptr = reinterpret_cast<const Scalar*>(input.const_data_ptr());
  const Scalar* levels_ptr = reinterpret_cast<const Scalar*>(levels.const_data_ptr());
  OutputScalar* output_ptr = reinterpret_cast<OutputScalar*>(output.mutable_data_ptr());
  const int32_t* trellis_ptr = trellis.const_data_ptr<int32_t>();
#define QVQ_WMMA_SPLITK_LAUNCH(BITS)                                                                               \
  qvq_gemm_wmma_kernel<Scalar, OutputScalar, BITS><<<grid, kThreads, 0, stream>>>(                                \
      input_ptr, trellis_ptr, levels_ptr, partial_ptr, output_ptr, static_cast<int>(input.size(0)),                \
      static_cast<int>(input.size(1)), static_cast<int>(output.size(1)), split_count)
  switch (transition_bits) {
    case 2: QVQ_WMMA_SPLITK_LAUNCH(2); break;
    case 3: QVQ_WMMA_SPLITK_LAUNCH(3); break;
    case 4: QVQ_WMMA_SPLITK_LAUNCH(4); break;
    case 5: QVQ_WMMA_SPLITK_LAUNCH(5); break;
    case 6: QVQ_WMMA_SPLITK_LAUNCH(6); break;
    case 7: QVQ_WMMA_SPLITK_LAUNCH(7); break;
    case 8: QVQ_WMMA_SPLITK_LAUNCH(8); break;
    case 9: QVQ_WMMA_SPLITK_LAUNCH(9); break;
    case 10: QVQ_WMMA_SPLITK_LAUNCH(10); break;
    case 11: QVQ_WMMA_SPLITK_LAUNCH(11); break;
    case 12: QVQ_WMMA_SPLITK_LAUNCH(12); break;
    case 13: QVQ_WMMA_SPLITK_LAUNCH(13); break;
    case 14: QVQ_WMMA_SPLITK_LAUNCH(14); break;
    case 15: QVQ_WMMA_SPLITK_LAUNCH(15); break;
    case 16: QVQ_WMMA_SPLITK_LAUNCH(16); break;
  }
#undef QVQ_WMMA_SPLITK_LAUNCH
  const int64_t output_values = output.numel();
  const int reduction_blocks = static_cast<int>((output_values + kThreads - 1) / kThreads);
  qvq_reduce_splitk_kernel<OutputScalar><<<reduction_blocks, kThreads, 0, stream>>>(
      partial_ptr, output_ptr, static_cast<int>(output.size(0)), static_cast<int>(output.size(1)), split_count);
}

at::Tensor qvq_gemv_cuda_impl(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    int64_t transition_bits,
    int64_t out_features,
    bool output_fp32,
    int64_t vector_size,
    const c10::optional<at::Tensor>& bank_ids) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(trellis.is_cuda(), "trellis must be a CUDA tensor");
  TORCH_CHECK(levels.is_cuda(), "PGC16 levels must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be rank two");
  TORCH_CHECK(trellis.dim() == 2, "trellis must be rank two");
  TORCH_CHECK(levels.dim() == 1, "PGC16 levels must be rank one");
  TORCH_CHECK(transition_bits >= 2 && transition_bits <= 16, "transition_bits must be in [2, 16]");
  if (vector_size == 4) {
    TORCH_CHECK(transition_bits >= 4 && transition_bits <= 16 && (transition_bits % 2) == 0,
                "V4 transition_bits must be one of {4, 6, 8, 10, 12, 14, 16}");
  }
  TORCH_CHECK(vector_size == 2 || vector_size == 4, "vector_size must be 2 or 4");
  TORCH_CHECK(!bank_ids.has_value(), "V2 GEMV does not support V4 bank selectors");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
              "input must have dtype float16 or bfloat16");
  TORCH_CHECK(levels.scalar_type() == at::kHalf, "PGC16 levels must preserve the canonical float16 bit patterns");
  TORCH_CHECK(trellis.scalar_type() == at::kInt, "trellis must have dtype int32");
  TORCH_CHECK(input.device() == trellis.device() && input.device() == levels.device(),
              "input, trellis, and PGC16 levels must share one CUDA device");
  TORCH_CHECK(input.is_contiguous() && trellis.is_contiguous() && levels.is_contiguous(),
              "input, trellis, and PGC16 levels must be contiguous");

  const int64_t size_m = input.size(0);
  const int64_t size_k = input.size(1);
  TORCH_CHECK(size_m >= 0, "input row count must be non-negative");
  TORCH_CHECK(size_k > 0 && size_k % kTileRows == 0, "input width must be positive and divisible by 16");
  TORCH_CHECK(out_features > 0 && out_features % kTileColumns == 0,
              "out_features must be positive and divisible by 16");
  TORCH_CHECK(size_m <= std::numeric_limits<int>::max() && size_k <= std::numeric_limits<int>::max() &&
                  out_features <= std::numeric_limits<int>::max(),
              "QVQ CUDA dimensions exceed the int32 kernel limit");
  TORCH_CHECK(levels.numel() == kPgc16LevelCount, "PGC16 levels must have shape (256)");
  const int64_t tile_count = (size_k / kTileRows) * (out_features / kTileColumns);
  TORCH_CHECK(trellis.sizes() == at::IntArrayRef({tile_count, (vector_size == 2 ? 4 : 2) * transition_bits}),
              "trellis shape must match K, N, and transition_bits");

  if (size_m == 0) {
    return at::empty(
        {0, out_features}, input.options().dtype(output_fp32 ? at::kFloat : input.scalar_type()));
  }

  const c10::cuda::CUDAGuard device_guard(input.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(properties.major >= 8, "QVQ CUDA requires compute capability >= 8.0");

  at::Tensor output = at::empty(
      {size_m, out_features}, output_fp32 ? input.options().dtype(at::kFloat) : input.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const int64_t base_blocks = (out_features / kTileColumns) * ((size_m + kRowsPerBlock - 1) / kRowsPerBlock);
  const int64_t target_blocks = static_cast<int64_t>(properties.multiProcessorCount) * 6;
  const int64_t k_tiles = size_k / kTileRows;
  const int split_count = base_blocks >= 384 ? 1 : static_cast<int>(std::min(
      std::min((target_blocks + base_blocks - 1) / base_blocks, k_tiles), static_cast<int64_t>(64)));
  if (split_count > 1) {
    at::Tensor partial_output =
        at::empty({split_count, size_m, out_features}, input.options().dtype(at::kFloat));
    if (input.scalar_type() == at::kHalf && output_fp32) {
      launch_qvq_gemv_splitk<half, float>(
          input, trellis, nullptr, levels, partial_output, output, static_cast<int>(transition_bits), split_count, stream);
    } else if (input.scalar_type() == at::kHalf) {
      launch_qvq_gemv_splitk<half, half>(
          input, trellis, nullptr, levels, partial_output, output, static_cast<int>(transition_bits), split_count, stream);
    } else if (output_fp32) {
      launch_qvq_gemv_splitk<nv_bfloat16, float>(
          input, trellis, nullptr, levels, partial_output, output, static_cast<int>(transition_bits), split_count, stream);
    } else {
      launch_qvq_gemv_splitk<nv_bfloat16, nv_bfloat16>(
          input, trellis, nullptr, levels, partial_output, output, static_cast<int>(transition_bits), split_count, stream);
    }
  } else if (input.scalar_type() == at::kHalf) {
    if (output_fp32) {
      launch_qvq_gemv<half, float>(input, trellis, nullptr, levels, output, static_cast<int>(transition_bits), stream);
    } else {
      launch_qvq_gemv<half, half>(input, trellis, nullptr, levels, output, static_cast<int>(transition_bits), stream);
    }
  } else {
    if (output_fp32) {
      launch_qvq_gemv<nv_bfloat16, float>(input, trellis, nullptr, levels, output, static_cast<int>(transition_bits), stream);
    } else {
      launch_qvq_gemv<nv_bfloat16, nv_bfloat16>(
          input, trellis, nullptr, levels, output, static_cast<int>(transition_bits), stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qvq_gemv_cuda_v4(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    int64_t transition_bits,
    int64_t out_features,
    bool output_fp32,
    int64_t vector_size,
    const c10::optional<at::Tensor>& bank_ids) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(trellis.is_cuda(), "trellis must be a CUDA tensor");
  TORCH_CHECK(levels.is_cuda(), "PGC16 levels must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be rank two");
  TORCH_CHECK(trellis.dim() == 2, "trellis must be rank two");
  TORCH_CHECK(levels.dim() == 1, "PGC16 levels must be rank one");
  TORCH_CHECK(transition_bits >= 2 && transition_bits <= 16, "transition_bits must be in [2, 16]");
  TORCH_CHECK(vector_size == 2 || vector_size == 4, "vector_size must be 2 or 4");
  if (vector_size == 4) {
    TORCH_CHECK(transition_bits >= 4 && transition_bits <= 16 && (transition_bits % 2) == 0,
                "V4 transition_bits must be one of {4, 6, 8, 10, 12, 14, 16}");
  }
  TORCH_CHECK(!bank_ids.has_value() || vector_size == 4, "V4 bank selectors require vector_size=4");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
              "input must have dtype float16 or bfloat16");
  TORCH_CHECK(levels.scalar_type() == at::kHalf, "PGC16 levels must preserve the canonical float16 bit patterns");
  TORCH_CHECK(trellis.scalar_type() == at::kInt, "trellis must have dtype int32");
  TORCH_CHECK(input.device() == trellis.device() && input.device() == levels.device(),
              "input, trellis, and PGC16 levels must share one CUDA device");
  TORCH_CHECK(input.is_contiguous() && trellis.is_contiguous() && levels.is_contiguous(),
              "input, trellis, and PGC16 levels must be contiguous");

  const int64_t size_m = input.size(0);
  const int64_t size_k = input.size(1);
  TORCH_CHECK(size_m >= 0, "input row count must be non-negative");
  TORCH_CHECK(size_k > 0 && size_k % kTileRows == 0, "input width must be positive and divisible by 16");
  TORCH_CHECK(out_features > 0 && out_features % kTileColumns == 0,
              "out_features must be positive and divisible by 16");
  TORCH_CHECK(size_m <= std::numeric_limits<int>::max() && size_k <= std::numeric_limits<int>::max() &&
                  out_features <= std::numeric_limits<int>::max(),
              "QVQ CUDA dimensions exceed the int32 kernel limit");
  TORCH_CHECK(levels.numel() == kPgc16LevelCount, "PGC16 levels must have shape (256)");
  const int64_t tile_count = (size_k / kTileRows) * (out_features / kTileColumns);
  TORCH_CHECK(trellis.sizes() == at::IntArrayRef({tile_count, (vector_size == 2 ? 4 : 2) * transition_bits}),
              "trellis shape must match K, N, and transition_bits");
  if (bank_ids.has_value()) {
    const at::Tensor& selectors = *bank_ids;
    TORCH_CHECK(selectors.is_cuda() && selectors.device() == input.device(),
                "V4 bank selectors must share the input CUDA device");
    TORCH_CHECK(selectors.scalar_type() == at::kByte, "V4 bank selectors must use uint8");
    TORCH_CHECK(selectors.dim() == 1 && selectors.numel() == tile_count,
                "V4 bank selectors must have one uint8 entry per trellis tile");
    TORCH_CHECK(selectors.is_contiguous(), "V4 bank selectors must be contiguous");
    TORCH_CHECK(selectors.max().item<int64_t>() <= 3, "V4 bank selectors must be in [0, 3]");
  }

  if (size_m == 0) {
    return at::empty(
        {0, out_features}, input.options().dtype(output_fp32 ? at::kFloat : input.scalar_type()));
  }

  const c10::cuda::CUDAGuard device_guard(input.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(properties.major >= 8, "QVQ CUDA requires compute capability >= 8.0");

  at::Tensor output = at::empty(
      {size_m, out_features}, output_fp32 ? input.options().dtype(at::kFloat) : input.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const int64_t base_blocks = (out_features / kTileColumns) * ((size_m + kRowsPerBlock - 1) / kRowsPerBlock);
  const int64_t target_blocks = static_cast<int64_t>(properties.multiProcessorCount) * 6;
  const int64_t k_tiles = size_k / kTileRows;
  const int split_count = base_blocks >= 384 ? 1 : static_cast<int>(std::min(
      std::min((target_blocks + base_blocks - 1) / base_blocks, k_tiles), static_cast<int64_t>(64)));
  if (split_count > 1) {
    at::Tensor partial_output =
        at::empty({split_count, size_m, out_features}, input.options().dtype(at::kFloat));
    if (input.scalar_type() == at::kHalf && output_fp32) {
      launch_qvq_gemv_splitk<half, float, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, stream);
    } else if (input.scalar_type() == at::kHalf) {
      launch_qvq_gemv_splitk<half, half, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, stream);
    } else if (output_fp32) {
      launch_qvq_gemv_splitk<nv_bfloat16, float, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, stream);
    } else {
      launch_qvq_gemv_splitk<nv_bfloat16, nv_bfloat16, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, stream);
    }
  } else if (input.scalar_type() == at::kHalf) {
    if (output_fp32) {
      launch_qvq_gemv<half, float, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, output,
          static_cast<int>(transition_bits), stream);
    } else {
      launch_qvq_gemv<half, half, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, output,
          static_cast<int>(transition_bits), stream);
    }
  } else {
    if (output_fp32) {
      launch_qvq_gemv<nv_bfloat16, float, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, output,
          static_cast<int>(transition_bits), stream);
    } else {
      launch_qvq_gemv<nv_bfloat16, nv_bfloat16, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, output,
          static_cast<int>(transition_bits), stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def("gemv(Tensor input, Tensor trellis, Tensor levels, int transition_bits, int out_features, bool output_fp32, Tensor? bank_ids=None) -> Tensor");
  m.def("gemv_v4(Tensor input, Tensor trellis, Tensor levels, int transition_bits, int out_features, bool output_fp32, Tensor? bank_ids=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CUDA, m) {
  m.impl("gemv", [](const at::Tensor& input, const at::Tensor& trellis, const at::Tensor& levels, int64_t transition_bits, int64_t out_features, bool output_fp32, const c10::optional<at::Tensor>& bank_ids) { return qvq_gemv_cuda_impl(input, trellis, levels, transition_bits, out_features, output_fp32, 2, bank_ids); });
  m.impl("gemv_v4", [](const at::Tensor& input, const at::Tensor& trellis, const at::Tensor& levels, int64_t transition_bits, int64_t out_features, bool output_fp32, const c10::optional<at::Tensor>& bank_ids) { return qvq_gemv_cuda_v4(input, trellis, levels, transition_bits, out_features, output_fp32, 4, bank_ids); });
}
