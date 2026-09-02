// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
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
constexpr bool kQvqDebugDisableVecStaging = false;

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

__device__ __forceinline__ uint32_t pgc16_v2_bank_mask_runtime(int transition_bits, uint32_t bank) {
  constexpr uint32_t masks[6][4] = {
      {0x0000u, 0xA5A5u, 0x5A5Au, 0x3C3Cu},
      {0x0000u, 0xA5A5u, 0x9696u, 0x6969u},
      {0x0000u, 0x5A5Au, 0x3C3Cu, 0xC3C3u},
      {0x0000u, 0x9696u, 0x3C3Cu, 0xC3C3u},
      {0x0000u, 0x6969u, 0x5A5Au, 0x3C3Cu},
      {0x0000u, 0xC3C3u, 0x9696u, 0x5A5Au},
  };
  return masks[transition_bits - 2][bank & 3u];
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

// Specialized decode for compile-time transition widths: the planar decoder
// fully unrolls, the V2 pair state/mix is computed once and shared across the
// even/odd column pair via a warp shuffle, and the unused second V2 mix from
// the generic path is never issued. Produces bit-identical weights to
// qvq_decode_weight (same extraction decomposition, same FP expression order;
// min-free path so fold order cannot matter).
template <int TransitionBits, int VectorSize>
__device__ __forceinline__ float qvq_decode_weight_fast(
    const uint32_t* packed_words,
    uint8_t packed_bank_id,
    const half* cached_levels,
    int local,
    int bank_mode,
    int bank_alt_id) {
  const int col = local & 15;
  const int v4_group = local >> 2;
  if constexpr (VectorSize == 4) {
    uint32_t mixed_pair = 0;
    if ((col & 3) == 0) {
      const uint32_t state = qvq_state<TransitionBits, VectorSize>(packed_words, v4_group);
      mixed_pair = pgc16_mix(state) |
          (pgc16_mix(state ^ pgc16_bank_mask_runtime(TransitionBits, packed_bank_id)) << 16);
    }
    mixed_pair = __shfl_sync(0xffffffffu, mixed_pair, local & ~3);
    const uint32_t mixed = mixed_pair & 0xffffu;
    const uint32_t mixed2 = mixed_pair >> 16;
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
  } else {
    const int pair = local >> 1;
    uint32_t level_pair = 0;
    if ((local & 1) == 0) {
      const uint32_t state = qvq_state<TransitionBits, VectorSize>(packed_words, pair);
      uint32_t bank = 0;
      if (bank_mode == 2) {
        bank = (static_cast<uint32_t>(packed_bank_id) >> ((pair >> 5) * 2)) & 3u;
      } else if (bank_mode == 3) {
        bank = ((static_cast<uint32_t>(packed_bank_id) >> (pair >> 4)) & 1u) *
            static_cast<uint32_t>(bank_alt_id);
      }
      const uint32_t mask = bank_mode == 0 ? 0u : pgc16_v2_bank_mask_runtime(TransitionBits, bank);
      const uint32_t mixed = pgc16_mix(state ^ mask);
      level_pair = static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed >> 8])) |
          (static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed & 0xffu])) << 16);
    }
    // Direct-indexed broadcast from the even (pair-leader) lane; an xor
    // exchange would overwrite the leader's own value with its partner's zero.
    level_pair = __shfl_sync(0xffffffffu, level_pair, local & ~1);
    const uint32_t level_bits = (local & 1) == 0 ? level_pair & 0xffffu : level_pair >> 16;
    return __half2float(__ushort_as_half(static_cast<unsigned short>(level_bits)));
  }
}

// Shared decode path for normal and split-K GEMV.  Keeping this logic in one
// device helper prevents the two accumulation kernels from drifting while
// retaining compile-time V2/V4 and rate specialization at the call site.
template <int VectorSize>
__device__ __forceinline__ float qvq_decode_weight(
    const uint32_t* packed_words,
    uint8_t packed_bank_id,
    const half* cached_levels,
    int local,
    int transition_bits,
    int bank_mode,
    int bank_alt_id) {
  const int col = local & 15;
  const int v4_group = local >> 2;
  uint32_t mixed = 0;
  uint32_t mixed2 = 0;
  if constexpr (VectorSize == 4) {
    uint32_t mixed_pair = 0;
    if ((col & 3) == 0) {
      const uint32_t state = qvq_state_runtime<VectorSize>(packed_words, v4_group, transition_bits);
      mixed_pair = pgc16_mix(state) |
          (pgc16_mix(state ^ pgc16_bank_mask_runtime(transition_bits, packed_bank_id)) << 16);
    }
    mixed_pair = __shfl_sync(0xffffffffu, mixed_pair, local & ~3);
    mixed = mixed_pair & 0xffffu;
    mixed2 = mixed_pair >> 16;
  } else {
    const int pair = local >> 1;
    const uint32_t state = qvq_state_runtime<VectorSize>(packed_words, pair, transition_bits);
    uint32_t bank = 0;
    if (bank_mode == 2) {
      bank = (static_cast<uint32_t>(packed_bank_id) >> ((pair >> 5) * 2)) & 3u;
    } else if (bank_mode == 3) {
      bank = ((static_cast<uint32_t>(packed_bank_id) >> (pair >> 4)) & 1u) *
          static_cast<uint32_t>(bank_alt_id);
    }
    const uint32_t mask = bank_mode == 0 ? 0u : pgc16_v2_bank_mask_runtime(transition_bits, bank);
    mixed = pgc16_mix(state ^ mask);
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
//
// TransitionBits == 0 selects the generic runtime-width decode; any value in
// [2, 16] (V2) or the even widths [4, 16] (V4) selects a fully specialized
// decode with identical arithmetic. Trellis and input staging use 16-byte
// vector loads when the per-tile word count allows it (it always does: both
// vector layouts emit whole 32-bit words).
template <typename Scalar, typename OutputScalar, int ROWS, int VectorSize = 2, int TransitionBits = 0>
__global__ __launch_bounds__(kThreads) void qvq_gemv_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ trellis,
    const uint8_t* __restrict__ bank_ids,
    const uint8_t* __restrict__ bank_alt_ids,
    const half* __restrict__ levels,
    OutputScalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int bank_mode,
    int bank_alt_id,
    int bank_alt_boundary0,
    int bank_alt_boundary1) {
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
  const int bank_alt_segment =
      static_cast<int>(n_tile >= bank_alt_boundary0) +
      static_cast<int>(n_tile >= bank_alt_boundary1);
  const int output_bank_alt_id = bank_alt_ids == nullptr
      ? bank_alt_id
      : static_cast<int>(bank_alt_ids[bank_alt_segment]);

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
  constexpr int kUnroll = ROWS <= 8 ? 24 : (ROWS == 16 ? 16 : 12);
  constexpr int kMaxWords = VectorSize == 2 ? 64 : 32;
  constexpr int kBatches = 2;
  __shared__ __align__(16) uint32_t packed_words[kBatches][kUnroll][kMaxWords];
  __shared__ uint8_t packed_bank_ids[kBatches][kUnroll];
  __shared__ __align__(16) Scalar input_tile[kBatches][kUnroll][ROWS * kTileRows];

  const int words_per_tile = (VectorSize == 2 ? 4 : 2) * transition_bits;
  auto stage_batch = [&](int kb, int tiles_here, int dst) {
    if constexpr (TransitionBits != 0 && !kQvqDebugDisableVecStaging) {
      const int words_per_vec = words_per_tile >> 2;
      const int vec_total = tiles_here * words_per_vec;
      uint4* words4 = reinterpret_cast<uint4*>(packed_words[dst][0]);
      const uint4* trellis4 = reinterpret_cast<const uint4*>(trellis);
      for (int index = thread; index < vec_total; index += kThreads) {
        const int u = index / words_per_vec;
        const int w4 = index - u * words_per_vec;
        if constexpr (ROWS >= 16) {
          __pipeline_memcpy_async(
              words4 + u * (kMaxWords / 4) + w4,
              trellis4 + (static_cast<int64_t>(kb + u) * n_tiles + n_tile) * words_per_vec + w4, 16);
        } else {
          words4[u * (kMaxWords / 4) + w4] =
              trellis4[(static_cast<int64_t>(kb + u) * n_tiles + n_tile) * words_per_vec + w4];
        }
      }
      static_assert(kMaxWords % 4 == 0, "vectorized trellis staging requires padded rows");
    } else {
      for (int index = thread; index < tiles_here * words_per_tile; index += kThreads) {
        const int u = index / words_per_tile;
        const int w = index % words_per_tile;
        const int tile_index = (kb + u) * n_tiles + n_tile;
        const int32_t* tile = trellis + static_cast<int64_t>(tile_index) * words_per_tile;
        packed_words[dst][u][w] = static_cast<uint32_t>(tile[w]);
      }
    }
    if (thread < tiles_here) {
      const int tile_index = (kb + thread) * n_tiles + n_tile;
      packed_bank_ids[dst][thread] = bank_ids == nullptr ? 0 : bank_ids[tile_index];
    }
    if constexpr (TransitionBits != 0 && !kQvqDebugDisableVecStaging) {
      constexpr int row_vecs = kTileRows * static_cast<int>(sizeof(Scalar)) / static_cast<int>(sizeof(uint4));
      const int vec_total = tiles_here * ROWS * row_vecs;
      uint4* input4 = reinterpret_cast<uint4*>(input_tile[dst][0]);
      const uint4* input4_base = reinterpret_cast<const uint4*>(input);
      for (int index = thread; index < vec_total; index += kThreads) {
        const int u = index / (ROWS * row_vecs);
        const int cell = index - u * (ROWS * row_vecs);
        const int row = cell / row_vecs;
        const int vec = cell - row * row_vecs;
        if (row < block_rows) {
          if constexpr (ROWS >= 16) {
            __pipeline_memcpy_async(
                input4 + index,
                input4_base +
                    ((static_cast<int64_t>(m0 + row) * size_k + (kb + u) * kTileRows) *
                         static_cast<int>(sizeof(Scalar)) / 16 +
                     vec),
                16);
          } else {
            input4[index] =
                input4_base[(static_cast<int64_t>(m0 + row) * size_k + (kb + u) * kTileRows) *
                            static_cast<int>(sizeof(Scalar)) / 16 +
                        vec];
          }
        } else {
          input4[index] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    } else {
      for (int index = thread; index < tiles_here * ROWS * kTileRows; index += kThreads) {
        const int u = index / (ROWS * kTileRows);
        const int cell = index % (ROWS * kTileRows);
        const int row = cell / kTileRows;
        const int k_local = cell % kTileRows;
        input_tile[dst][u][cell] = row < block_rows
            ? input[static_cast<int64_t>(m0 + row) * size_k + (kb + u) * kTileRows + k_local]
            : ScalarTraits<Scalar>::from_float(0.0f);
      }
    }
    if constexpr (ROWS >= 16) {
      __pipeline_commit();
    }
  };

  // Software-pipelined main loop: while one shared buffer is consumed by the
  // FMA/decode phase, the next batch streams into the other buffer, so global
  // load latency overlaps compute and one barrier per iteration suffices.
  stage_batch(0, min(kUnroll, k_tiles), 0);
  int parity = 0;
  for (int kb = 0; kb < k_tiles; kb += kUnroll) {
    const int tiles_here = min(kUnroll, k_tiles - kb);
    const bool has_next = kb + kUnroll < k_tiles;
    if (has_next) {
      stage_batch(kb + kUnroll, min(kUnroll, k_tiles - kb - kUnroll), parity ^ 1);
    }
    if constexpr (ROWS >= 16) {
      // cp.async completion is per-thread; the barrier publishes every lane's
      // copies block-wide before the compute phase reads them.
      __pipeline_wait_prior(has_next ? 1 : 0);
    }
    __syncthreads();

#pragma unroll 4
    for (int u = 0; u < kUnroll; ++u) {
      if (u < tiles_here) {
        // Decode the single weight this thread needs (register; no shared write).
        const int local = (k_slot << 4) | col;
        float weight;
        if constexpr (TransitionBits != 0) {
          weight = qvq_decode_weight_fast<TransitionBits, VectorSize>(
              packed_words[parity][u], bank_ids == nullptr ? 0 : packed_bank_ids[parity][u], cached_levels, local,
              bank_mode, output_bank_alt_id);
        } else {
          weight = qvq_decode_weight<VectorSize>(
              packed_words[parity][u], bank_ids == nullptr ? 0 : packed_bank_ids[parity][u], cached_levels, local,
              transition_bits, bank_mode, output_bank_alt_id);
        }

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          accumulator[r] = fmaf(
              ScalarTraits<Scalar>::to_float(input_tile[parity][u][r * kTileRows + k_slot]), weight, accumulator[r]);
        }
      }
    }
    __syncthreads();
    parity ^= 1;
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
template <typename Scalar, int ROWS, int VectorSize = 2, int TransitionBits = 0>
__global__ __launch_bounds__(kThreads) void qvq_gemv_splitk_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ trellis,
    const uint8_t* __restrict__ bank_ids,
    const uint8_t* __restrict__ bank_alt_ids,
    const half* __restrict__ levels,
    float* __restrict__ partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    int transition_bits,
    int bank_mode,
    int bank_alt_id,
    int bank_alt_boundary0,
    int bank_alt_boundary1,
    int grouped_split_count0,
    int grouped_split_count1,
    int grouped_split_count2) {
  __shared__ half cached_levels[kPgc16LevelCount];
  __shared__ float reduced[kThreads / 32][ROWS * kTileColumns];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int m0 = static_cast<int>(blockIdx.y) * ROWS;
  const int block_rows = min(ROWS, size_m - m0);
  const int n_tiles = size_n / kTileColumns;
  const int k_tiles = size_k / kTileRows;
  int n_tile;
  int split;
  int segment_split_count;
  if (bank_alt_ids == nullptr) {
    n_tile = static_cast<int>(blockIdx.x);
    split = static_cast<int>(blockIdx.z);
    segment_split_count = split_count;
  } else {
    // A grouped QKV launch must retain each child's original split-K
    // partition.  Selecting one split count from the concatenated N changes
    // FP32 summation order and can cross a BF16 recovery rounding boundary.
    // Compact the segment-specific (N tile, split) work into blockIdx.x so we
    // preserve child arithmetic without launching max_splits for every tile.
    const int segment0_tiles = bank_alt_boundary0;
    const int segment1_tiles = bank_alt_boundary1 - bank_alt_boundary0;
    const int segment0_work = segment0_tiles * grouped_split_count0;
    const int segment1_work = segment1_tiles * grouped_split_count1;
    const int work = static_cast<int>(blockIdx.x);
    if (work < segment0_work) {
      n_tile = work / grouped_split_count0;
      split = work - n_tile * grouped_split_count0;
      segment_split_count = grouped_split_count0;
    } else if (work < segment0_work + segment1_work) {
      const int local_work = work - segment0_work;
      const int local_tile = local_work / grouped_split_count1;
      n_tile = bank_alt_boundary0 + local_tile;
      split = local_work - local_tile * grouped_split_count1;
      segment_split_count = grouped_split_count1;
    } else {
      const int local_work = work - segment0_work - segment1_work;
      const int local_tile = local_work / grouped_split_count2;
      n_tile = bank_alt_boundary1 + local_tile;
      split = local_work - local_tile * grouped_split_count2;
      segment_split_count = grouped_split_count2;
    }
  }
  const int n0 = n_tile * kTileColumns;
  const int bank_alt_segment =
      static_cast<int>(n_tile >= bank_alt_boundary0) +
      static_cast<int>(n_tile >= bank_alt_boundary1);
  const int output_bank_alt_id = bank_alt_ids == nullptr
      ? bank_alt_id
      : static_cast<int>(bank_alt_ids[bank_alt_segment]);
  const int k_tile_begin = (k_tiles * split) / segment_split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / segment_split_count;

  const int col = thread & 15;
  const int k_slot = thread >> 4;

  cached_levels[thread] = levels[thread];
  __syncthreads();

  float accumulator[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    accumulator[r] = 0.0f;
  }

  constexpr int kUnroll = ROWS <= 8 ? 24 : (ROWS == 16 ? 16 : 12);
  constexpr int kMaxWords = VectorSize == 2 ? 64 : 32;
  constexpr int kBatches = 2;
  __shared__ __align__(16) uint32_t packed_words[kBatches][kUnroll][kMaxWords];
  __shared__ uint8_t packed_bank_ids[kBatches][kUnroll];
  __shared__ __align__(16) Scalar input_tile[kBatches][kUnroll][ROWS * kTileRows];

  const int words_per_tile = (VectorSize == 2 ? 4 : 2) * transition_bits;
  auto stage_batch = [&](int kb, int tiles_here, int dst) {
    if constexpr (TransitionBits != 0 && !kQvqDebugDisableVecStaging) {
      const int words_per_vec = words_per_tile >> 2;
      const int vec_total = tiles_here * words_per_vec;
      uint4* words4 = reinterpret_cast<uint4*>(packed_words[dst][0]);
      const uint4* trellis4 = reinterpret_cast<const uint4*>(trellis);
      for (int index = thread; index < vec_total; index += kThreads) {
        const int u = index / words_per_vec;
        const int w4 = index - u * words_per_vec;
        if constexpr (ROWS >= 16) {
          __pipeline_memcpy_async(
              words4 + u * (kMaxWords / 4) + w4,
              trellis4 + (static_cast<int64_t>(kb + u) * n_tiles + n_tile) * words_per_vec + w4, 16);
        } else {
          words4[u * (kMaxWords / 4) + w4] =
              trellis4[(static_cast<int64_t>(kb + u) * n_tiles + n_tile) * words_per_vec + w4];
        }
      }
    } else {
      for (int index = thread; index < tiles_here * words_per_tile; index += kThreads) {
        const int u = index / words_per_tile;
        const int w = index % words_per_tile;
        const int tile_index = (kb + u) * n_tiles + n_tile;
        const int32_t* tile = trellis + static_cast<int64_t>(tile_index) * words_per_tile;
        packed_words[dst][u][w] = static_cast<uint32_t>(tile[w]);
      }
    }
    if (thread < tiles_here) {
      const int tile_index = (kb + thread) * n_tiles + n_tile;
      packed_bank_ids[dst][thread] = bank_ids == nullptr ? 0 : bank_ids[tile_index];
    }
    if constexpr (TransitionBits != 0 && !kQvqDebugDisableVecStaging) {
      constexpr int row_vecs = kTileRows * static_cast<int>(sizeof(Scalar)) / static_cast<int>(sizeof(uint4));
      const int vec_total = tiles_here * ROWS * row_vecs;
      uint4* input4 = reinterpret_cast<uint4*>(input_tile[dst][0]);
      const uint4* input4_base = reinterpret_cast<const uint4*>(input);
      for (int index = thread; index < vec_total; index += kThreads) {
        const int u = index / (ROWS * row_vecs);
        const int cell = index - u * (ROWS * row_vecs);
        const int row = cell / row_vecs;
        const int vec = cell - row * row_vecs;
        if (row < block_rows) {
          if constexpr (ROWS >= 16) {
            __pipeline_memcpy_async(
                input4 + index,
                input4_base +
                    ((static_cast<int64_t>(m0 + row) * size_k + (kb + u) * kTileRows) *
                         static_cast<int>(sizeof(Scalar)) / 16 +
                     vec),
                16);
          } else {
            input4[index] =
                input4_base[(static_cast<int64_t>(m0 + row) * size_k + (kb + u) * kTileRows) *
                            static_cast<int>(sizeof(Scalar)) / 16 +
                        vec];
          }
        } else {
          input4[index] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    } else {
      for (int index = thread; index < tiles_here * ROWS * kTileRows; index += kThreads) {
        const int u = index / (ROWS * kTileRows);
        const int cell = index % (ROWS * kTileRows);
        const int row = cell / kTileRows;
        const int k_local = cell % kTileRows;
        input_tile[dst][u][cell] = row < block_rows
            ? input[static_cast<int64_t>(m0 + row) * size_k + (kb + u) * kTileRows + k_local]
            : ScalarTraits<Scalar>::from_float(0.0f);
      }
    }
    if constexpr (ROWS >= 16) {
      __pipeline_commit();
    }
  };

  // Software-pipelined split-K loop: same double-buffer scheme as the plain
  // GEMV kernel (one barrier per iteration, loads overlap the decode/FMA phase).
  stage_batch(k_tile_begin, min(kUnroll, k_tile_end - k_tile_begin), 0);
  int parity = 0;
  for (int kb = k_tile_begin; kb < k_tile_end; kb += kUnroll) {
    const int tiles_here = min(kUnroll, k_tile_end - kb);
    const bool has_next = kb + kUnroll < k_tile_end;
    if (has_next) {
      stage_batch(kb + kUnroll, min(kUnroll, k_tile_end - kb - kUnroll), parity ^ 1);
    }
    if constexpr (ROWS >= 16) {
      // cp.async completion is per-thread; the barrier publishes every lane's
      // copies block-wide before the compute phase reads them.
      __pipeline_wait_prior(has_next ? 1 : 0);
    }
    __syncthreads();

#pragma unroll 4
    for (int u = 0; u < kUnroll; ++u) {
      if (u < tiles_here) {
        const int local = (k_slot << 4) | col;
        float weight;
        if constexpr (TransitionBits != 0) {
          weight = qvq_decode_weight_fast<TransitionBits, VectorSize>(
              packed_words[parity][u], bank_ids == nullptr ? 0 : packed_bank_ids[parity][u], cached_levels, local,
              bank_mode, output_bank_alt_id);
        } else {
          weight = qvq_decode_weight<VectorSize>(
              packed_words[parity][u], bank_ids == nullptr ? 0 : packed_bank_ids[parity][u], cached_levels, local,
              transition_bits, bank_mode, output_bank_alt_id);
        }

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          accumulator[r] = fmaf(
              ScalarTraits<Scalar>::to_float(input_tile[parity][u][r * kTileRows + k_slot]), weight, accumulator[r]);
        }
      }
    }
    __syncthreads();
    parity ^= 1;
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
    int split_count,
    int bank_alt_boundary0,
    int bank_alt_boundary1,
    int grouped_split_count0,
    int grouped_split_count1,
    int grouped_split_count2) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t output_values = static_cast<int64_t>(size_m) * size_n;
  if (index >= output_values) {
    return;
  }
  int output_split_count = split_count;
  if (bank_alt_boundary0 > 0) {
    const int n_tile = static_cast<int>(index % size_n) / kTileColumns;
    const int segment = static_cast<int>(n_tile >= bank_alt_boundary0) +
        static_cast<int>(n_tile >= bank_alt_boundary1);
    output_split_count = segment == 0 ? grouped_split_count0
        : segment == 1 ? grouped_split_count1
                       : grouped_split_count2;
  }
  float accumulator = 0.0f;
  for (int split = 0; split < output_split_count; ++split) {
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
    // PGC16 canonical levels are always float16 bit patterns regardless of the
    // compute dtype; decoding through float keeps exactly one rounding step.
    const half* __restrict__ levels,
    float* __restrict__ partial_output,
    OutputScalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int split_count) {
  constexpr int kBatches = 2;
  constexpr int kWordsPerTile = 4 * TransitionBits;
  __shared__ __align__(16) uint32_t packed_words[kBatches][kWordsPerTile];
  __shared__ Scalar decoded_weight[kTileValues];
  __shared__ __align__(16) Scalar input_tile[kBatches][kRowsPerBlock * kTileRows];
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

  cached_levels[thread] =
      ScalarTraits<Scalar>::from_float(ScalarTraits<half>::to_float(levels[thread]));

  wmma::fragment<wmma::matrix_a, 16, 16, 16, Scalar, wmma::row_major> input_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, Scalar, wmma::row_major> weight_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  // Async double-buffered staging: while tile i is decoded and multiplied,
  // tile i+1 streams into the other buffer through cp.async.
  auto stage_batch = [&](int k_tile, int dst) {
    const int64_t tile_index = static_cast<int64_t>(k_tile) * n_tiles + n_tile;
    const uint4* words4_src =
        reinterpret_cast<const uint4*>(trellis + tile_index * kWordsPerTile);
    uint4* words4_dst = reinterpret_cast<uint4*>(packed_words[dst]);
    for (int index = thread; index < kWordsPerTile / 4; index += kThreads) {
      __pipeline_memcpy_async(words4_dst + index, words4_src + index, 16);
    }
    uint4* input4 = reinterpret_cast<uint4*>(input_tile[dst]);
    const uint4* input4_base = reinterpret_cast<const uint4*>(input);
    for (int index = thread; index < kRowsPerBlock * kTileRows / 8; index += kThreads) {
      const int row = index / (kTileRows / 8);
      const int vec = index - row * (kTileRows / 8);
      if (row < block_rows) {
        __pipeline_memcpy_async(
            input4 + index,
            input4_base +
                ((static_cast<int64_t>(m0 + row) * size_k + k_tile * kTileRows) *
                     static_cast<int>(sizeof(Scalar)) / 16 +
                 vec),
            16);
      } else {
        input4[index] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    __pipeline_commit();
  };

  stage_batch(k_tile_begin, 0);

  int parity = 0;
  for (int k_tile = k_tile_begin; k_tile < k_tile_end; ++k_tile) {
    const bool has_next = k_tile + 1 < k_tile_end;
    if (has_next) {
      stage_batch(k_tile + 1, parity ^ 1);
    }
    // cp.async completion is per-thread; this barrier publishes every lane's
    // copies block-wide before the decode phase reads them.
    __pipeline_wait_prior(has_next ? 1 : 0);
    __syncthreads();

    const int local = thread;
    const uint32_t state = qvq_state<TransitionBits>(packed_words[parity], local >> 1);
    const uint32_t mixed = pgc16_mix(state);
    const uint32_t level_index = (local & 1) == 0 ? mixed >> 8 : mixed & 0xffu;
    const float value = ScalarTraits<Scalar>::to_float(cached_levels[level_index]);
    decoded_weight[local] = ScalarTraits<Scalar>::from_float(value);
    __syncthreads();

    if (warp < active_warps) {
      wmma::load_matrix_sync(input_fragment, input_tile[parity] + warp * 16 * kTileRows, kTileRows);
      wmma::load_matrix_sync(weight_fragment, decoded_weight, kTileColumns);
      wmma::mma_sync(accumulator, input_fragment, weight_fragment, accumulator);
    }
    parity ^= 1;
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

// Transition widths that receive a fully specialized decode instantiation.
constexpr bool qvq_tb_specialized_v2(int tb) { return tb >= 2 && tb <= 16; }
constexpr bool qvq_tb_specialized_v4(int tb) {
  return tb >= 4 && tb <= 16 && (tb % 2) == 0;
}

// The vectorized staging paths require naturally aligned trellis/input bases;
// anything else (odd view offsets) stays on the generic scalar kernels.
constexpr bool qvq_vec_aligned(const void* trellis, const void* input) {
  return (reinterpret_cast<uintptr_t>(trellis) % sizeof(uint4)) == 0 &&
      (reinterpret_cast<uintptr_t>(input) % sizeof(uint4)) == 0;
}

template <typename Scalar, typename OutputScalar, int VectorSize = 2>
void launch_qvq_gemv(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor* bank_ids,
    const at::Tensor* bank_alt_ids,
    const at::Tensor& levels,
    at::Tensor& output,
    int transition_bits,
    int bank_mode,
    int bank_alt_id,
    int bank_alt_boundary0,
    int bank_alt_boundary1,
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
  const uint8_t* bank_alt_ids_ptr =
      bank_alt_ids == nullptr ? nullptr : bank_alt_ids->const_data_ptr<uint8_t>();
  const bool specialized = VectorSize == 2 ? qvq_tb_specialized_v2(transition_bits)
                                           : qvq_tb_specialized_v4(transition_bits);
  if (specialized && qvq_vec_aligned(trellis_ptr, input_ptr)) {
#define QVQ_LAUNCH_TB(ROWS, TB)                                                                                    \
    qvq_gemv_kernel<Scalar, OutputScalar, ROWS, VectorSize, TB>                                                    \
        <<<grid, kThreads, 0, stream>>>(                                                                           \
            input_ptr, trellis_ptr, bank_ids_ptr, bank_alt_ids_ptr, levels_ptr, output_ptr,                      \
            static_cast<int>(input.size(0)),                                                                      \
            static_cast<int>(input.size(1)), static_cast<int>(output.size(1)), transition_bits, bank_mode,        \
            bank_alt_id, bank_alt_boundary0, bank_alt_boundary1)
#define QVQ_LAUNCH_ROWS(ROWS)                                                                                      \
    switch (transition_bits) {                                                                                     \
      case 2: QVQ_LAUNCH_TB(ROWS, 2); break;                                                                       \
      case 3: QVQ_LAUNCH_TB(ROWS, 3); break;                                                                       \
      case 4: QVQ_LAUNCH_TB(ROWS, 4); break;                                                                       \
      case 5: QVQ_LAUNCH_TB(ROWS, 5); break;                                                                       \
      case 6: QVQ_LAUNCH_TB(ROWS, 6); break;                                                                       \
      case 7: QVQ_LAUNCH_TB(ROWS, 7); break;                                                                       \
      case 8: QVQ_LAUNCH_TB(ROWS, 8); break;                                                                       \
      case 9: QVQ_LAUNCH_TB(ROWS, 9); break;                                                                       \
      case 10: QVQ_LAUNCH_TB(ROWS, 10); break;                                                                     \
      case 11: QVQ_LAUNCH_TB(ROWS, 11); break;                                                                     \
      case 12: QVQ_LAUNCH_TB(ROWS, 12); break;                                                                     \
      case 13: QVQ_LAUNCH_TB(ROWS, 13); break;                                                                     \
      case 14: QVQ_LAUNCH_TB(ROWS, 14); break;                                                                     \
      case 15: QVQ_LAUNCH_TB(ROWS, 15); break;                                                                     \
      case 16: QVQ_LAUNCH_TB(ROWS, 16); break;                                                                     \
      default: QVQ_LAUNCH_TB(ROWS, 0); break;                                                                      \
    }
    if (rows == 1) {
      QVQ_LAUNCH_ROWS(1);
    } else if (rows == 8) {
      QVQ_LAUNCH_ROWS(8);
    } else if (rows == 16) {
      QVQ_LAUNCH_ROWS(16);
    } else {
      QVQ_LAUNCH_ROWS(kRowsPerBlock);
    }
#undef QVQ_LAUNCH_ROWS
#undef QVQ_LAUNCH_TB
    return;
  }
#define QVQ_LAUNCH(ROWS)                                                                                           \
  qvq_gemv_kernel<Scalar, OutputScalar, ROWS, VectorSize, 0><<<grid, kThreads, 0, stream>>>(                       \
      input_ptr, trellis_ptr, bank_ids_ptr, bank_alt_ids_ptr, levels_ptr, output_ptr,                              \
      static_cast<int>(input.size(0)),                                                                              \
      static_cast<int>(input.size(1)), static_cast<int>(output.size(1)), transition_bits, bank_mode, bank_alt_id,  \
      bank_alt_boundary0, bank_alt_boundary1)
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
    const at::Tensor* bank_alt_ids,
    const at::Tensor& levels,
    at::Tensor& partial_output,
    at::Tensor& output,
    int transition_bits,
    int split_count,
    int bank_mode,
    int bank_alt_id,
    int bank_alt_boundary0,
    int bank_alt_boundary1,
    cudaStream_t stream,
    int grouped_split_count0 = 0,
    int grouped_split_count1 = 0,
    int grouped_split_count2 = 0) {
  float* partial_ptr = partial_output.mutable_data_ptr<float>();
  const int rows = qvq_rows_for_m(static_cast<int>(input.size(0)));
  const bool grouped = bank_alt_ids != nullptr;
  const int n_tiles = static_cast<int>(output.size(1) / kTileColumns);
  const int64_t grouped_work = grouped
      ? static_cast<int64_t>(bank_alt_boundary0) * grouped_split_count0 +
          static_cast<int64_t>(bank_alt_boundary1 - bank_alt_boundary0) * grouped_split_count1 +
          static_cast<int64_t>(n_tiles - bank_alt_boundary1) * grouped_split_count2
      : n_tiles;
  TORCH_CHECK(grouped_work > 0 && grouped_work <= std::numeric_limits<int>::max(),
              "grouped P32 split-K work exceeds the CUDA grid limit");
  const dim3 grid(
      static_cast<unsigned int>(grouped_work),
      static_cast<unsigned int>((input.size(0) + rows - 1) / rows),
      static_cast<unsigned int>(grouped ? 1 : split_count));
  const Scalar* input_ptr = reinterpret_cast<const Scalar*>(input.const_data_ptr());
  const half* levels_ptr = reinterpret_cast<const half*>(levels.const_data_ptr());
  OutputScalar* output_ptr = reinterpret_cast<OutputScalar*>(output.mutable_data_ptr());
  const int32_t* trellis_ptr = trellis.const_data_ptr<int32_t>();
  const uint8_t* bank_ids_ptr = bank_ids == nullptr ? nullptr : bank_ids->const_data_ptr<uint8_t>();
  const uint8_t* bank_alt_ids_ptr =
      bank_alt_ids == nullptr ? nullptr : bank_alt_ids->const_data_ptr<uint8_t>();
  const bool specialized = VectorSize == 2 ? qvq_tb_specialized_v2(transition_bits)
                                           : qvq_tb_specialized_v4(transition_bits);
  if (specialized && qvq_vec_aligned(trellis_ptr, input_ptr)) {
#define QVQ_SPLITK_LAUNCH_TB(ROWS, TB)                                                                            \
    qvq_gemv_splitk_kernel<Scalar, ROWS, VectorSize, TB>                                                          \
        <<<grid, kThreads, 0, stream>>>(                                                                          \
            input_ptr, trellis_ptr, bank_ids_ptr, bank_alt_ids_ptr, levels_ptr, partial_ptr,                     \
            static_cast<int>(input.size(0)),                                                                      \
            static_cast<int>(input.size(1)), static_cast<int>(output.size(1)), split_count, transition_bits,      \
            bank_mode, bank_alt_id, bank_alt_boundary0, bank_alt_boundary1, grouped_split_count0,                 \
            grouped_split_count1, grouped_split_count2)
#define QVQ_SPLITK_LAUNCH_ROWS(ROWS)                                                                              \
    switch (transition_bits) {                                                                                    \
      case 2: QVQ_SPLITK_LAUNCH_TB(ROWS, 2); break;                                                               \
      case 3: QVQ_SPLITK_LAUNCH_TB(ROWS, 3); break;                                                               \
      case 4: QVQ_SPLITK_LAUNCH_TB(ROWS, 4); break;                                                               \
      case 5: QVQ_SPLITK_LAUNCH_TB(ROWS, 5); break;                                                               \
      case 6: QVQ_SPLITK_LAUNCH_TB(ROWS, 6); break;                                                               \
      case 7: QVQ_SPLITK_LAUNCH_TB(ROWS, 7); break;                                                               \
      case 8: QVQ_SPLITK_LAUNCH_TB(ROWS, 8); break;                                                               \
      case 9: QVQ_SPLITK_LAUNCH_TB(ROWS, 9); break;                                                               \
      case 10: QVQ_SPLITK_LAUNCH_TB(ROWS, 10); break;                                                             \
      case 11: QVQ_SPLITK_LAUNCH_TB(ROWS, 11); break;                                                             \
      case 12: QVQ_SPLITK_LAUNCH_TB(ROWS, 12); break;                                                             \
      case 13: QVQ_SPLITK_LAUNCH_TB(ROWS, 13); break;                                                             \
      case 14: QVQ_SPLITK_LAUNCH_TB(ROWS, 14); break;                                                             \
      case 15: QVQ_SPLITK_LAUNCH_TB(ROWS, 15); break;                                                             \
      case 16: QVQ_SPLITK_LAUNCH_TB(ROWS, 16); break;                                                             \
      default: QVQ_SPLITK_LAUNCH_TB(ROWS, 0); break;                                                              \
    }
    if (rows == 1) {
      QVQ_SPLITK_LAUNCH_ROWS(1);
    } else if (rows == 8) {
      QVQ_SPLITK_LAUNCH_ROWS(8);
    } else if (rows == 16) {
      QVQ_SPLITK_LAUNCH_ROWS(16);
    } else {
      QVQ_SPLITK_LAUNCH_ROWS(kRowsPerBlock);
    }
#undef QVQ_SPLITK_LAUNCH_ROWS
#undef QVQ_SPLITK_LAUNCH_TB
    const int64_t output_values = output.numel();
    const int reduction_blocks = static_cast<int>((output_values + kThreads - 1) / kThreads);
    qvq_reduce_splitk_kernel<OutputScalar><<<reduction_blocks, kThreads, 0, stream>>>(
        partial_ptr, output_ptr, static_cast<int>(output.size(0)), static_cast<int>(output.size(1)), split_count,
        grouped ? bank_alt_boundary0 : 0, grouped ? bank_alt_boundary1 : 0, grouped_split_count0,
        grouped_split_count1, grouped_split_count2);
    return;
  }
#define QVQ_SPLITK_LAUNCH(ROWS)                                                                                   \
  qvq_gemv_splitk_kernel<Scalar, ROWS, VectorSize, 0><<<grid, kThreads, 0, stream>>>(                             \
      input_ptr, trellis_ptr, bank_ids_ptr, bank_alt_ids_ptr, levels_ptr, partial_ptr,                            \
      static_cast<int>(input.size(0)),                                                                             \
      static_cast<int>(input.size(1)), static_cast<int>(output.size(1)), split_count, transition_bits,            \
      bank_mode, bank_alt_id, bank_alt_boundary0, bank_alt_boundary1, grouped_split_count0,                       \
      grouped_split_count1, grouped_split_count2)
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
      partial_ptr, output_ptr, static_cast<int>(output.size(0)), static_cast<int>(output.size(1)), split_count,
      grouped ? bank_alt_boundary0 : 0, grouped ? bank_alt_boundary1 : 0, grouped_split_count0,
      grouped_split_count1, grouped_split_count2);
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
  const half* levels_ptr = reinterpret_cast<const half*>(levels.const_data_ptr());
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
  const half* levels_ptr = reinterpret_cast<const half*>(levels.const_data_ptr());
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
      partial_ptr, output_ptr, static_cast<int>(output.size(0)), static_cast<int>(output.size(1)), split_count,
      0, 0, 0, 0, 0);
}

at::Tensor qvq_gemv_cuda_impl(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    int64_t transition_bits,
    int64_t out_features,
    bool output_fp32,
    int64_t vector_size,
    const c10::optional<at::Tensor>& bank_ids,
    int64_t bank_mode,
    int64_t bank_alt_id,
    const c10::optional<at::Tensor>& bank_alt_ids,
    int64_t bank_alt_boundary0,
    int64_t bank_alt_boundary1) {
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
  TORCH_CHECK(bank_mode >= 0 && bank_mode <= 3 && bank_mode != 1,
              "V2 GEMV bank_mode must be 0, 2 (V2B4-P64), or 3 (V2B2-P32)");
  TORCH_CHECK((bank_mode == 0) == !bank_ids.has_value(),
              "segmented-bank V2 GEMV requires selectors exactly when bank_mode is 2 or 3");
  TORCH_CHECK(bank_mode != 3 || (bank_alt_id >= 1 && bank_alt_id <= 3),
              "V2B2-P32 alternative-bank ID must be in [1, 3]");
  TORCH_CHECK(!bank_alt_ids.has_value() || bank_mode == 3,
              "grouped alternative-bank IDs are valid only for V2B2-P32");
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
    TORCH_CHECK(transition_bits >= 2 && transition_bits <= 7,
                "segmented-bank V2 GEMV supports transition_bits in [2, 7]");
    TORCH_CHECK(selectors.is_cuda() && selectors.device() == input.device(),
                "segmented-bank V2 selectors must share the input CUDA device");
    TORCH_CHECK(selectors.scalar_type() == at::kByte && selectors.is_contiguous(),
                "segmented-bank V2 selectors must be contiguous uint8");
    TORCH_CHECK(selectors.dim() == 1 && selectors.numel() == tile_count,
                "segmented-bank V2 selectors must contain one packed byte per trellis tile");
  }
  if (bank_alt_ids.has_value()) {
    const at::Tensor& alternatives = *bank_alt_ids;
    TORCH_CHECK(alternatives.is_cuda() && alternatives.device() == input.device(),
                "grouped V2B2-P32 alternative-bank IDs must share the input CUDA device");
    TORCH_CHECK(alternatives.scalar_type() == at::kByte && alternatives.is_contiguous(),
                "grouped V2B2-P32 alternative-bank IDs must be contiguous uint8");
    TORCH_CHECK(alternatives.dim() == 1 && (alternatives.numel() == 2 || alternatives.numel() == 3),
                "grouped V2B2-P32 alternative-bank IDs must contain two or three bytes");
    const int64_t output_tiles = out_features / kTileColumns;
    TORCH_CHECK(bank_alt_boundary0 > 0 && bank_alt_boundary0 < output_tiles,
                "grouped V2B2-P32 first output-tile boundary is out of range");
    if (alternatives.numel() == 2) {
      TORCH_CHECK(bank_alt_boundary1 == output_tiles,
                  "two-way grouped V2B2-P32 must end its second segment at N/16");
    } else {
      TORCH_CHECK(bank_alt_boundary1 > bank_alt_boundary0 && bank_alt_boundary1 < output_tiles,
                  "three-way grouped V2B2-P32 output-tile boundaries must be strictly increasing");
    }
  } else {
    TORCH_CHECK(bank_alt_boundary0 == 0 && bank_alt_boundary1 == 0,
                "grouped V2B2-P32 boundaries require compact alternative-bank IDs");
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
  const int64_t n_tiles = out_features / kTileColumns;
  const int64_t m_stripes = (size_m + kRowsPerBlock - 1) / kRowsPerBlock;
  const int64_t target_blocks = static_cast<int64_t>(properties.multiProcessorCount) * 6;
  const int64_t k_tiles = size_k / kTileRows;
  const auto automatic_split_count = [&](int64_t segment_n_tiles) {
    if (segment_n_tiles == 0) {
      return 1;
    }
    const int64_t segment_blocks = segment_n_tiles * m_stripes;
    return segment_blocks >= 384 ? 1 : static_cast<int>(std::min(
        std::min((target_blocks + segment_blocks - 1) / segment_blocks, k_tiles), static_cast<int64_t>(64)));
  };
  int grouped_split_count0 = automatic_split_count(n_tiles);
  int grouped_split_count1 = grouped_split_count0;
  int grouped_split_count2 = grouped_split_count0;
  if (bank_alt_ids.has_value()) {
    grouped_split_count0 = automatic_split_count(bank_alt_boundary0);
    grouped_split_count1 = automatic_split_count(bank_alt_boundary1 - bank_alt_boundary0);
    grouped_split_count2 = automatic_split_count(n_tiles - bank_alt_boundary1);
  }
  const int split_count = std::max(
      grouped_split_count0, std::max(grouped_split_count1, grouped_split_count2));
  // Large-batch, selector-free V2 calls reuse each decoded weight tile across
  // 32 input rows through Ampere tensor cores. The FP32 WMMA accumulator
  // keeps the scalar path's numerical contract; only the reduction order
  // within a 16-wide k step differs.
  const bool wmma_dtype_ok = (input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16);
  const bool wmma_eligible_base = wmma_dtype_ok && bank_mode == 0 &&
      !bank_ids.has_value() && size_m > 16 && transition_bits >= 2 && transition_bits <= 16 &&
      qvq_vec_aligned(trellis.const_data_ptr(), input.const_data_ptr());
  at::Tensor wmma_input;
  const bool wmma_eligible = wmma_eligible_base && [&]() {
    if (input.scalar_type() == at::kHalf) {
      wmma_input = input;
      return true;
    }
    // Async saturating cast: exact within fp16 range, clamped beyond (a
    // host-side amax check would force a device sync every call).
    wmma_input = input.clamp(-65000.0, 65000.0).to(at::kHalf);
    return true;
  }();
  if (wmma_eligible) {
#define QVQ_WMMA_DISPATCH(SCALAR_T, OUT_T)                                                                            if (split_count > 1) {                                                                                              at::Tensor partial_output =                                                                                           at::empty({split_count, size_m, out_features}, input.options().dtype(at::kFloat));                            launch_qvq_gemm_wmma_splitk<SCALAR_T, OUT_T>(                                                                         wmma_input, trellis, levels, partial_output, output, static_cast<int>(transition_bits), split_count, stream);     } else {                                                                                                            launch_qvq_gemm_wmma<SCALAR_T, OUT_T>(                                                                                wmma_input, trellis, levels, output, static_cast<int>(transition_bits), stream);                                 }
    if (output_fp32) {
      QVQ_WMMA_DISPATCH(half, float);
    } else if (input.scalar_type() == at::kBFloat16) {
      QVQ_WMMA_DISPATCH(half, nv_bfloat16);
    } else {
      QVQ_WMMA_DISPATCH(half, half);
    }
#undef QVQ_WMMA_DISPATCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }
  if (split_count > 1) {
    at::Tensor partial_output =
        at::empty({split_count, size_m, out_features}, input.options().dtype(at::kFloat));
    if (input.scalar_type() == at::kHalf && output_fp32) {
      launch_qvq_gemv_splitk<half, float>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr,
          bank_alt_ids.has_value() ? &*bank_alt_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, static_cast<int>(bank_mode), static_cast<int>(bank_alt_id),
          static_cast<int>(bank_alt_boundary0), static_cast<int>(bank_alt_boundary1), stream,
          grouped_split_count0, grouped_split_count1, grouped_split_count2);
    } else if (input.scalar_type() == at::kHalf) {
      launch_qvq_gemv_splitk<half, half>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr,
          bank_alt_ids.has_value() ? &*bank_alt_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, static_cast<int>(bank_mode), static_cast<int>(bank_alt_id),
          static_cast<int>(bank_alt_boundary0), static_cast<int>(bank_alt_boundary1), stream,
          grouped_split_count0, grouped_split_count1, grouped_split_count2);
    } else if (output_fp32) {
      launch_qvq_gemv_splitk<nv_bfloat16, float>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr,
          bank_alt_ids.has_value() ? &*bank_alt_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, static_cast<int>(bank_mode), static_cast<int>(bank_alt_id),
          static_cast<int>(bank_alt_boundary0), static_cast<int>(bank_alt_boundary1), stream,
          grouped_split_count0, grouped_split_count1, grouped_split_count2);
    } else {
      launch_qvq_gemv_splitk<nv_bfloat16, nv_bfloat16>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr,
          bank_alt_ids.has_value() ? &*bank_alt_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, static_cast<int>(bank_mode), static_cast<int>(bank_alt_id),
          static_cast<int>(bank_alt_boundary0), static_cast<int>(bank_alt_boundary1), stream,
          grouped_split_count0, grouped_split_count1, grouped_split_count2);
    }
  } else if (input.scalar_type() == at::kHalf) {
    if (output_fp32) {
      launch_qvq_gemv<half, float>(input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr,
                                   bank_alt_ids.has_value() ? &*bank_alt_ids : nullptr, levels, output,
                                   static_cast<int>(transition_bits), static_cast<int>(bank_mode),
                                   static_cast<int>(bank_alt_id), static_cast<int>(bank_alt_boundary0),
                                   static_cast<int>(bank_alt_boundary1), stream);
    } else {
      launch_qvq_gemv<half, half>(input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr,
                                  bank_alt_ids.has_value() ? &*bank_alt_ids : nullptr, levels, output,
                                  static_cast<int>(transition_bits), static_cast<int>(bank_mode),
                                  static_cast<int>(bank_alt_id), static_cast<int>(bank_alt_boundary0),
                                  static_cast<int>(bank_alt_boundary1), stream);
    }
  } else {
    if (output_fp32) {
      launch_qvq_gemv<nv_bfloat16, float>(input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr,
                                          bank_alt_ids.has_value() ? &*bank_alt_ids : nullptr, levels, output,
                                          static_cast<int>(transition_bits), static_cast<int>(bank_mode),
                                          static_cast<int>(bank_alt_id), static_cast<int>(bank_alt_boundary0),
                                          static_cast<int>(bank_alt_boundary1), stream);
    } else {
      launch_qvq_gemv<nv_bfloat16, nv_bfloat16>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr,
          bank_alt_ids.has_value() ? &*bank_alt_ids : nullptr, levels, output,
          static_cast<int>(transition_bits), static_cast<int>(bank_mode), static_cast<int>(bank_alt_id),
          static_cast<int>(bank_alt_boundary0), static_cast<int>(bank_alt_boundary1), stream);
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
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, bank_ids.has_value() ? 1 : 0, 0, 0, 0, stream);
    } else if (input.scalar_type() == at::kHalf) {
      launch_qvq_gemv_splitk<half, half, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, bank_ids.has_value() ? 1 : 0, 0, 0, 0, stream);
    } else if (output_fp32) {
      launch_qvq_gemv_splitk<nv_bfloat16, float, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, bank_ids.has_value() ? 1 : 0, 0, 0, 0, stream);
    } else {
      launch_qvq_gemv_splitk<nv_bfloat16, nv_bfloat16, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, bank_ids.has_value() ? 1 : 0, 0, 0, 0, stream);
    }
  } else if (input.scalar_type() == at::kHalf) {
    if (output_fp32) {
      launch_qvq_gemv<half, float, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, nullptr, levels, output,
          static_cast<int>(transition_bits), bank_ids.has_value() ? 1 : 0, 0, 0, 0, stream);
    } else {
      launch_qvq_gemv<half, half, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, nullptr, levels, output,
          static_cast<int>(transition_bits), bank_ids.has_value() ? 1 : 0, 0, 0, 0, stream);
    }
  } else {
    if (output_fp32) {
      launch_qvq_gemv<nv_bfloat16, float, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, nullptr, levels, output,
          static_cast<int>(transition_bits), bank_ids.has_value() ? 1 : 0, 0, 0, 0, stream);
    } else {
      launch_qvq_gemv<nv_bfloat16, nv_bfloat16, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, nullptr, levels, output,
          static_cast<int>(transition_bits), bank_ids.has_value() ? 1 : 0, 0, 0, 0, stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def("gemv(Tensor input, Tensor trellis, Tensor levels, int transition_bits, int out_features, bool output_fp32, Tensor? bank_ids=None, int bank_mode=0, int bank_alt_id=0) -> Tensor");
  m.def("gemv_grouped_p32(Tensor input, Tensor trellis, Tensor levels, int transition_bits, int out_features, bool output_fp32, Tensor bank_ids, Tensor bank_alt_ids, int bank_alt_boundary0, int bank_alt_boundary1) -> Tensor");
  m.def("gemv_v4(Tensor input, Tensor trellis, Tensor levels, int transition_bits, int out_features, bool output_fp32, Tensor? bank_ids=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CUDA, m) {
  m.impl("gemv", [](const at::Tensor& input, const at::Tensor& trellis, const at::Tensor& levels, int64_t transition_bits, int64_t out_features, bool output_fp32, const c10::optional<at::Tensor>& bank_ids, int64_t bank_mode, int64_t bank_alt_id) { return qvq_gemv_cuda_impl(input, trellis, levels, transition_bits, out_features, output_fp32, 2, bank_ids, bank_mode, bank_alt_id, c10::nullopt, 0, 0); });
  m.impl("gemv_grouped_p32", [](const at::Tensor& input, const at::Tensor& trellis, const at::Tensor& levels, int64_t transition_bits, int64_t out_features, bool output_fp32, const at::Tensor& bank_ids, const at::Tensor& bank_alt_ids, int64_t bank_alt_boundary0, int64_t bank_alt_boundary1) { return qvq_gemv_cuda_impl(input, trellis, levels, transition_bits, out_features, output_fp32, 2, bank_ids, 3, 1, bank_alt_ids, bank_alt_boundary0, bank_alt_boundary1); });
  m.impl("gemv_v4", [](const at::Tensor& input, const at::Tensor& trellis, const at::Tensor& levels, int64_t transition_bits, int64_t out_features, bool output_fp32, const c10::optional<at::Tensor>& bank_ids) { return qvq_gemv_cuda_v4(input, trellis, levels, transition_bits, out_features, output_fp32, 4, bank_ids); });
}
