// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "qvq_p32_abi.h"

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kRows = 16;
constexpr int kTileRows = 16;
constexpr int kTileColumns = 16;
constexpr int kTilesPerBlock = kWarps;
constexpr int kM1Threads = kThreads;
constexpr int kM1Warps = kM1Threads / 32;
constexpr int kM1TilesPerBlock = 4 * kM1Warps;
constexpr int kStageKTiles = 2;
constexpr int kScalarTripleStageKTiles = 3;
constexpr int kScalarLongStageKTiles = 4;
constexpr int kPairsPerTile = 128;
constexpr int kReductionThreads = 256;
constexpr uint32_t kPgc16Multiplier = 40503u;
constexpr uint32_t kPgc16Increment = 17011u;

__device__ __forceinline__ uint32_t pgc16_mix(uint32_t state) {
  uint32_t mixed = state ^ (state >> 8);
  mixed = (mixed * kPgc16Multiplier + kPgc16Increment) & 0xffffu;
  return mixed ^ (mixed >> 7);
}

template <int TransitionBits>
__device__ __forceinline__ uint32_t alternate_bank_mask(int bank_alt_id) {
  static_assert(TransitionBits >= 4 && TransitionBits <= 7);
  if (bank_alt_id == 0) return 0u;
  if constexpr (TransitionBits == 4) {
    return bank_alt_id == 1 ? 0x5a5au : bank_alt_id == 2 ? 0x3c3cu : 0xc3c3u;
  } else if constexpr (TransitionBits == 5) {
    return bank_alt_id == 1 ? 0x9696u : bank_alt_id == 2 ? 0x3c3cu : 0xc3c3u;
  } else if constexpr (TransitionBits == 6) {
    return bank_alt_id == 1 ? 0x6969u : bank_alt_id == 2 ? 0x5a5au : 0x3c3cu;
  } else {
    return bank_alt_id == 1 ? 0xc3c3u : bank_alt_id == 2 ? 0x9696u : 0x5a5au;
  }
}

// Window states pair naturally at a distance of 64 pairs: both use the same
// funnel-shift amount and their first words are exactly 2 * TransitionBits
// words apart. This is the storage-neutral paired extraction used by the
// upstream Ampere P32 implementation.
template <int TransitionBits>
__device__ __forceinline__ void window_state_pair64(
    const uint32_t* __restrict__ words,
    int pair,
    uint32_t& first,
    uint32_t& second) {
  constexpr int kWordsPerTile = 4 * TransitionBits;
  constexpr int kPairWordDistance = 2 * TransitionBits;
  const int bit_position = (kPairsPerTile - 1 - pair) * TransitionBits;
  const int first_word = bit_position >> 5;
  const int shift = bit_position & 31;
  const int first_next = first_word + 1 == kWordsPerTile ? 0 : first_word + 1;
  const int second_word = first_word - kPairWordDistance;
  // The state windows are at most 27 bits wide on W3.5. A 32-bit funnel
  // shift expresses the circular extraction without promoting each pair to
  // a 64-bit value, which costs multiple integer instructions on sm_80.
  first = __funnelshift_r(words[first_word], words[first_next], shift) & 0xffffu;
  second = __funnelshift_r(words[second_word], words[second_word + 1], shift) & 0xffffu;
}

struct MmaFragmentA {
  uint32_t values[4];
};

struct MmaFragmentB {
  uint32_t values[2];
};

template <bool UpperRowsOnly>
struct MmaAccumulator;

template <>
struct MmaAccumulator<false> {
  float values[4];
};

template <>
struct MmaAccumulator<true> {
  float values[2];
};

__device__ __forceinline__ void load_mma_fragment_a(
    MmaFragmentA& fragment,
    const void* shared_source) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(fragment.values[0]),
        "=r"(fragment.values[1]),
        "=r"(fragment.values[2]),
        "=r"(fragment.values[3])
      : "r"(shared_address));
}

__device__ __forceinline__ void load_mma_fragment_a_upper(
    MmaFragmentA& fragment,
    const void* shared_source) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(fragment.values[0]), "=r"(fragment.values[2])
      : "r"(shared_address));
  fragment.values[1] = 0;
  fragment.values[3] = 0;
}

template <bool UpperRowsOnly>
__device__ __forceinline__ void mma_m16n8k16(
    const MmaFragmentA& input,
    const MmaFragmentB& weight,
    MmaAccumulator<UpperRowsOnly>& accumulator) {
  if constexpr (UpperRowsOnly) {
    float discarded_0;
    float discarded_1;
    const float zero = 0.0f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(accumulator.values[0]),
          "=f"(accumulator.values[1]),
          "=f"(discarded_0),
          "=f"(discarded_1)
        : "r"(input.values[0]),
          "r"(input.values[1]),
          "r"(input.values[2]),
          "r"(input.values[3]),
          "r"(weight.values[0]),
          "r"(weight.values[1]),
          "f"(accumulator.values[0]),
          "f"(accumulator.values[1]),
          "f"(zero),
          "f"(zero));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(accumulator.values[0]),
          "=f"(accumulator.values[1]),
          "=f"(accumulator.values[2]),
          "=f"(accumulator.values[3])
        : "r"(input.values[0]),
          "r"(input.values[1]),
          "r"(input.values[2]),
          "r"(input.values[3]),
          "r"(weight.values[0]),
          "r"(weight.values[1]),
          "f"(accumulator.values[0]),
          "f"(accumulator.values[1]),
          "f"(accumulator.values[2]),
          "f"(accumulator.values[3]));
  }
}

__device__ __forceinline__ uint32_t selected_bank_mask(
    uint8_t packed_bank_id,
    int bank_bit,
    uint32_t alt_mask) {
  return (0u - ((static_cast<uint32_t>(packed_bank_id) >> bank_bit) & 1u)) &
      alt_mask;
}

template <int TransitionBits, bool UseSharedLevels = false>
__device__ __forceinline__ uint32_t decode_pair_bits(
    int pair,
    uint32_t state,
    uint8_t packed_bank_id,
    uint32_t alt_mask,
    const half* __restrict__ levels) {
  const uint32_t bank_mask =
      ((static_cast<uint32_t>(packed_bank_id) >> (pair >> 4)) & 1u) * alt_mask;
  const uint32_t mixed = pgc16_mix(state ^ bank_mask);
  union {
    uint32_t bits;
    half2 values;
  } decoded;
  const half first_level = UseSharedLevels
      ? levels[mixed >> 8]
      : __ldg(levels + (mixed >> 8));
  const half second_level = UseSharedLevels
      ? levels[mixed & 0xffu]
      : __ldg(levels + (mixed & 0xffu));
  decoded.values = __halves2half2(first_level, second_level);
  return decoded.bits;
}

template <bool UseSharedLevels = false>
__device__ __forceinline__ uint32_t decode_state_bits(
    uint32_t state,
    uint32_t bank_mask,
    const half* __restrict__ levels) {
  const uint32_t mixed = pgc16_mix(state ^ bank_mask);
  union {
    uint32_t bits;
    half2 values;
  } decoded;
  const half first_level = UseSharedLevels
      ? levels[mixed >> 8]
      : __ldg(levels + (mixed >> 8));
  const half second_level = UseSharedLevels
      ? levels[mixed & 0xffu]
      : __ldg(levels + (mixed & 0xffu));
  decoded.values = __halves2half2(first_level, second_level);
  return decoded.bits;
}

template <bool UseSharedLevels = false>
__device__ __forceinline__ void decode_state_pair_bits(
    uint32_t first_state,
    uint32_t second_state,
    uint32_t bank_mask,
    const half* __restrict__ levels,
    uint32_t& first_decoded,
    uint32_t& second_decoded) {
  uint32_t states = (first_state ^ bank_mask) |
      ((second_state ^ bank_mask) << 16);
  uint32_t mixed = states ^ ((states >> 8) & 0x00ff00ffu);
  const uint32_t mixed_low =
      (mixed & 0xffffu) * kPgc16Multiplier + kPgc16Increment;
  const uint32_t mixed_high =
      (mixed >> 16) * kPgc16Multiplier + kPgc16Increment;
  mixed = (mixed_low & 0xffffu) | (mixed_high << 16);
  mixed ^= (mixed >> 7) & 0x01ff01ffu;
  union {
    uint32_t bits;
    half2 values;
  } first, second;
  const half first_level_0 = UseSharedLevels
      ? levels[(mixed >> 8) & 0xffu]
      : __ldg(levels + ((mixed >> 8) & 0xffu));
  const half first_level_1 = UseSharedLevels
      ? levels[mixed & 0xffu]
      : __ldg(levels + (mixed & 0xffu));
  const half second_level_0 = UseSharedLevels
      ? levels[mixed >> 24]
      : __ldg(levels + (mixed >> 24));
  const half second_level_1 = UseSharedLevels
      ? levels[(mixed >> 16) & 0xffu]
      : __ldg(levels + ((mixed >> 16) & 0xffu));
  first.values = __halves2half2(first_level_0, first_level_1);
  second.values = __halves2half2(second_level_0, second_level_1);
  first_decoded = first.bits;
  second_decoded = second.bits;
}

__device__ __forceinline__ uint32_t pack_low_halves(
    uint32_t first,
    uint32_t second) {
  return (first & 0xffffu) | (second << 16);
}

__device__ __forceinline__ uint32_t pack_high_halves(
    uint32_t first,
    uint32_t second) {
  return (first >> 16) | (second & 0xffff0000u);
}

template <bool Vectorized>
__device__ __forceinline__ void store_output_pair(
    float* destination,
    float first,
    float second) {
  if constexpr (Vectorized) {
    *reinterpret_cast<float2*>(destination) = make_float2(first, second);
  } else {
    destination[0] = first;
    destination[1] = second;
  }
}

__device__ __forceinline__ void copy_async_cg_16(
    void* destination,
    const void* source) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :
      : "r"(shared_address), "l"(source));
}

__device__ __forceinline__ void copy_async_ca_4(
    void* destination,
    const void* source) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 4;\n"
      :
      : "r"(shared_address), "l"(source));
}

__device__ __forceinline__ void copy_async_ca_16(
    void* destination,
    const void* source) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 16;\n"
      :
      : "r"(shared_address), "l"(source));
}

template <
    int TransitionBits,
    bool FullRows,
    int ActiveRows = 0,
    int StaticN = 0,
    int Threads = kThreads,
    int TilesPerBlock = kTilesPerBlock,
    int StageKTiles = kStageKTiles,
    bool HoistBankMasks = false,
    bool UpperRowsOnly = false,
    int StaticK = 0,
    int RowGroups = 1,
    bool DynamicInputTile = false>
__device__ __forceinline__ void p32_window_ampere_kernel_body(
    const half* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const half* __restrict__ levels,
    const uint8_t* __restrict__ bank_ids,
    float* __restrict__ partial_output,
    float* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    const uint8_t* __restrict__ bank_alt_id,
    int n_block,
    int split,
    int payload_n_tiles,
    int payload_n_tile_offset,
    int64_t output_n_offset,
    int output_stride,
    int64_t partial_segment_offset,
    int64_t partial_split_stride = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  constexpr int kWordsPerTile = 4 * TransitionBits;
  constexpr int kStageColumnsForKernel = StageKTiles * kTileRows;
  constexpr int kRowsPerBlock = kRows * RowGroups;
  constexpr int kInputStageElements =
      kRowsPerBlock * kStageColumnsForKernel;
  extern __shared__ __align__(32) half dynamic_input_tile[];
  __shared__ __align__(32) half input_tile[
      DynamicInputTile ? 1 : 2][kInputStageElements];
  __shared__ __align__(16) uint32_t packed_words[
      2][StageKTiles][TilesPerBlock][kWordsPerTile];
  __shared__ __align__(4) uint8_t packed_bank_ids[2][StageKTiles][TilesPerBlock];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  constexpr int kStaticNTiles = StaticN > 0 ? StaticN / kTileColumns : 0;
  constexpr int kStaticKTiles = StaticK > 0 ? StaticK / kTileRows : 0;
  const int n_tiles = StaticN > 0 ? kStaticNTiles : size_n / kTileColumns;
  const int n_tile_base = n_block * TilesPerBlock;
  const bool active_tile = StaticN > 0 || n_tile_base + warp < n_tiles;
  const int k_tiles = StaticK > 0 ? kStaticKTiles : size_k / kTileRows;
  const int input_stride = StaticK > 0 ? StaticK : size_k;
  const int k_tile_begin = (k_tiles * split) / split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / split_count;
  const uint32_t alt_mask = alternate_bank_mask<TransitionBits>(*bank_alt_id);

  auto input_stage = [&](int destination) {
    if constexpr (DynamicInputTile) {
      return dynamic_input_tile + destination * kInputStageElements;
    } else {
      return input_tile[destination];
    }
  };

  auto stage = [&](int k_tile_base, int destination) {
    auto* input_vectors = reinterpret_cast<uint4*>(input_stage(destination));
    for (int index = thread; index < kRowsPerBlock * kStageColumnsForKernel / 8; index += Threads) {
      const int row = index / (kStageColumnsForKernel / 8);
      const int vector = index - row * (kStageColumnsForKernel / 8);
      const int source_column = k_tile_base * kTileRows + vector * 8;
      if constexpr (FullRows) {
        if (source_column < input_stride) {
          const half* source = input +
              static_cast<int64_t>(row) * input_stride + source_column;
          __pipeline_memcpy_async(input_vectors + index, reinterpret_cast<const uint4*>(source), 16);
        } else {
          input_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
        }
      } else if constexpr (ActiveRows > 0) {
        if (row < ActiveRows && source_column < input_stride) {
          const half* source = input +
              static_cast<int64_t>(row) * input_stride + source_column;
          __pipeline_memcpy_async(input_vectors + index, reinterpret_cast<const uint4*>(source), 16);
        } else {
          input_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
        }
      } else if (row < size_m && source_column < input_stride) {
        const half* source = input +
            static_cast<int64_t>(row) * input_stride + source_column;
        __pipeline_memcpy_async(input_vectors + index, reinterpret_cast<const uint4*>(source), 16);
      } else {
        input_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
      }
    }

    constexpr int kVectorsPerTile = kWordsPerTile / 4;
    constexpr int kVectorsPerKTile = TilesPerBlock * kVectorsPerTile;
    constexpr int kVectorsPerBlock = StageKTiles * kVectorsPerKTile;
    auto* destination_vectors = reinterpret_cast<uint4*>(packed_words[destination]);
    for (int index = thread; index < kVectorsPerBlock; index += Threads) {
      const int stage_k_tile = index / kVectorsPerKTile;
      const int tile_index = index - stage_k_tile * kVectorsPerKTile;
      const int tile = tile_index / kVectorsPerTile;
      const int vector = tile_index - tile * kVectorsPerTile;
      const int n_tile = n_tile_base + tile;
      const int k_tile = k_tile_base + stage_k_tile;
      if constexpr (StaticN > 0) {
        if (k_tile < k_tiles) {
          const int64_t global_tile = static_cast<int64_t>(k_tile) * payload_n_tiles +
              payload_n_tile_offset + n_tile;
          const auto* source_vectors = reinterpret_cast<const uint4*>(
              trellis + global_tile * kWordsPerTile);
          copy_async_cg_16(destination_vectors + index, source_vectors + vector);
        } else {
          destination_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
        }
      } else if (n_tile < n_tiles && k_tile < k_tiles) {
        const int64_t global_tile = static_cast<int64_t>(k_tile) * payload_n_tiles +
            payload_n_tile_offset + n_tile;
        const auto* source_vectors = reinterpret_cast<const uint4*>(
            trellis + global_tile * kWordsPerTile);
        copy_async_cg_16(destination_vectors + index, source_vectors + vector);
      } else {
        destination_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    if constexpr (
        StaticN > 0 && TilesPerBlock == 4 &&
        (FullRows || ActiveRows == 8)) {
      if (thread < StageKTiles) {
        const int k_tile = k_tile_base + thread;
        auto* destination_ids = reinterpret_cast<uint32_t*>(
            packed_bank_ids[destination][thread]);
        if (k_tile < k_tiles) {
          copy_async_ca_4(
              destination_ids,
              reinterpret_cast<const uint32_t*>(
                  bank_ids + static_cast<int64_t>(k_tile) * payload_n_tiles +
                      payload_n_tile_offset + n_tile_base));
        } else {
          *destination_ids = 0u;
        }
      }
    } else if (thread < StageKTiles * TilesPerBlock) {
      const int stage_k_tile = thread / TilesPerBlock;
      const int tile = thread - stage_k_tile * TilesPerBlock;
      const int n_tile = n_tile_base + tile;
      const int k_tile = k_tile_base + stage_k_tile;
      if constexpr (StaticN > 0) {
        packed_bank_ids[destination][stage_k_tile][tile] =
            k_tile < k_tiles ? bank_ids[static_cast<int64_t>(k_tile) * payload_n_tiles +
                payload_n_tile_offset + n_tile] : 0;
      } else {
        packed_bank_ids[destination][stage_k_tile][tile] =
            n_tile < n_tiles && k_tile < k_tiles
                ? bank_ids[static_cast<int64_t>(k_tile) * payload_n_tiles +
                    payload_n_tile_offset + n_tile]
                : 0;
      }
    }
    __pipeline_commit();
  };

  static_assert(!UpperRowsOnly || (!FullRows && ActiveRows == 8));
  constexpr bool kUpperRowsOnly = UpperRowsOnly;
  MmaAccumulator<kUpperRowsOnly> accumulator_0[RowGroups] = {};
  MmaAccumulator<kUpperRowsOnly> accumulator_1[RowGroups] = {};

  stage(k_tile_begin, 0);
  int parity = 0;
  for (int k_tile = k_tile_begin; k_tile < k_tile_end; k_tile += StageKTiles) {
    const bool has_next = k_tile + StageKTiles < k_tile_end;
    if (has_next) stage(k_tile + StageKTiles, parity ^ 1);
    __pipeline_wait_prior(has_next ? 1 : 0);
    __syncthreads();

    if (active_tile) {
#pragma unroll
      for (int stage_k_tile = 0; stage_k_tile < StageKTiles; ++stage_k_tile) {
        if (k_tile + stage_k_tile >= k_tile_end) continue;
        const uint32_t* words = packed_words[parity][stage_k_tile][warp];
        const uint8_t packed_bank_id = packed_bank_ids[parity][stage_k_tile][warp];
        const int producer_fragment = lane >> 4;
        const int producer_pair_column = producer_fragment * 4 + ((lane >> 2) & 3);
        const int producer_row_pair = lane & 3;
        const int first_pair = producer_row_pair * 16 + producer_pair_column;
        const int second_pair = first_pair + 8;
        uint32_t state_row_0, state_row_8, state_row_1, state_row_9;
        window_state_pair64<TransitionBits>(words, first_pair, state_row_0, state_row_8);
        window_state_pair64<TransitionBits>(words, second_pair, state_row_1, state_row_9);
        uint32_t decoded_row_0;
        uint32_t decoded_row_8;
        uint32_t decoded_row_1;
        uint32_t decoded_row_9;
        if constexpr (HoistBankMasks) {
          const uint32_t bank_mask_0 =
              selected_bank_mask(packed_bank_id, producer_row_pair, alt_mask);
          const uint32_t bank_mask_8 =
              selected_bank_mask(packed_bank_id, producer_row_pair + 4, alt_mask);
          decode_state_pair_bits(
              state_row_0, state_row_1, bank_mask_0, levels,
              decoded_row_0, decoded_row_1);
          decode_state_pair_bits(
              state_row_8, state_row_9, bank_mask_8, levels,
              decoded_row_8, decoded_row_9);
        } else {
          decoded_row_0 = decode_pair_bits<TransitionBits>(
              first_pair, state_row_0, packed_bank_id, alt_mask, levels);
          decoded_row_8 = decode_pair_bits<TransitionBits>(
              first_pair + 64, state_row_8, packed_bank_id, alt_mask, levels);
          decoded_row_1 = decode_pair_bits<TransitionBits>(
              second_pair, state_row_1, packed_bank_id, alt_mask, levels);
          decoded_row_9 = decode_pair_bits<TransitionBits>(
              second_pair + 64, state_row_9, packed_bank_id, alt_mask, levels);
        }

        const uint32_t low_rows_01 =
            pack_low_halves(decoded_row_0, decoded_row_1);
        const uint32_t high_rows_01 =
            pack_high_halves(decoded_row_0, decoded_row_1);
        const uint32_t low_rows_89 =
            pack_low_halves(decoded_row_8, decoded_row_9);
        const uint32_t high_rows_89 =
            pack_high_halves(decoded_row_8, decoded_row_9);
        const int target_column = lane >> 2;
        const int source_lane_0 = ((target_column >> 1) << 2) + (lane & 3);
        const int source_lane_1 = source_lane_0 + 16;
        const bool select_high = (target_column & 1) != 0;
        MmaFragmentB weight_fragment_0, weight_fragment_1;
        const uint32_t low_01_0 = __shfl_sync(0xffffffffu, low_rows_01, source_lane_0);
        const uint32_t high_01_0 = __shfl_sync(0xffffffffu, high_rows_01, source_lane_0);
        const uint32_t low_89_0 = __shfl_sync(0xffffffffu, low_rows_89, source_lane_0);
        const uint32_t high_89_0 = __shfl_sync(0xffffffffu, high_rows_89, source_lane_0);
        const uint32_t low_01_1 = __shfl_sync(0xffffffffu, low_rows_01, source_lane_1);
        const uint32_t high_01_1 = __shfl_sync(0xffffffffu, high_rows_01, source_lane_1);
        const uint32_t low_89_1 = __shfl_sync(0xffffffffu, low_rows_89, source_lane_1);
        const uint32_t high_89_1 = __shfl_sync(0xffffffffu, high_rows_89, source_lane_1);
        weight_fragment_0.values[0] = select_high ? high_01_0 : low_01_0;
        weight_fragment_0.values[1] = select_high ? high_89_0 : low_89_0;
        weight_fragment_1.values[0] = select_high ? high_01_1 : low_01_1;
        weight_fragment_1.values[1] = select_high ? high_89_1 : low_89_1;

        #pragma unroll
        for (int row_group = 0; row_group < RowGroups; ++row_group) {
          MmaFragmentA input_fragment;
          if constexpr (kUpperRowsOnly) {
            const int address_row = lane & 7;
            const int address_column = ((lane >> 3) & 1) * 8;
            load_mma_fragment_a_upper(
                input_fragment,
                input_stage(parity) + row_group * kRows * kStageColumnsForKernel +
                    address_row * kStageColumnsForKernel +
                    stage_k_tile * kTileRows + address_column);
          } else {
            const int address_row = (lane & 7) + ((lane >> 3) & 1) * 8;
            const int address_column = (lane >> 4) * 8;
            load_mma_fragment_a(
                input_fragment,
                input_stage(parity) + row_group * kRows * kStageColumnsForKernel +
                    address_row * kStageColumnsForKernel +
                    stage_k_tile * kTileRows + address_column);
          }
          mma_m16n8k16(
              input_fragment, weight_fragment_0, accumulator_0[row_group]);
          mma_m16n8k16(
              input_fragment, weight_fragment_1, accumulator_1[row_group]);
        }
      }
    }
    __syncthreads();
    parity ^= 1;
  }

  const int64_t split_stride = partial_split_stride > 0
      ? partial_split_stride
      : static_cast<int64_t>(size_m) * size_n;
  float* target = split_count == 1
      ? output + output_n_offset
      : partial_output + partial_segment_offset +
          static_cast<int64_t>(split) * split_stride;
  const int target_stride = split_count == 1 ? output_stride : size_n;
  if (active_tile) {
    const int output_row_0 = lane >> 2;
    const int output_row_1 = output_row_0 + 8;
    const int output_column = n_tile_base * kTileColumns +
        warp * kTileColumns + (lane & 3) * 2;
    #pragma unroll
    for (int row_group = 0; row_group < RowGroups; ++row_group) {
      float* row_target = target +
          static_cast<int64_t>(row_group) * kRows * target_stride;
      if constexpr (FullRows) {
        store_output_pair<StaticN != 1024 || (FullRows && StaticN > 0)>(
            row_target + static_cast<int64_t>(output_row_0) * target_stride + output_column,
            accumulator_0[row_group].values[0], accumulator_0[row_group].values[1]);
        store_output_pair<StaticN != 1024 || (FullRows && StaticN > 0)>(
            row_target + static_cast<int64_t>(output_row_1) * target_stride + output_column,
            accumulator_0[row_group].values[2], accumulator_0[row_group].values[3]);
        store_output_pair<StaticN != 1024 || (FullRows && StaticN > 0)>(
            row_target + static_cast<int64_t>(output_row_0) * target_stride + output_column + 8,
            accumulator_1[row_group].values[0], accumulator_1[row_group].values[1]);
        store_output_pair<StaticN != 1024 || (FullRows && StaticN > 0)>(
            row_target + static_cast<int64_t>(output_row_1) * target_stride + output_column + 8,
            accumulator_1[row_group].values[2], accumulator_1[row_group].values[3]);
      } else if constexpr (ActiveRows > 0) {
        if (output_row_0 < ActiveRows) {
          store_output_pair<StaticN != 1024>(
              row_target + static_cast<int64_t>(output_row_0) * target_stride + output_column,
              accumulator_0[row_group].values[0], accumulator_0[row_group].values[1]);
          store_output_pair<StaticN != 1024>(
              row_target + static_cast<int64_t>(output_row_0) * target_stride + output_column + 8,
              accumulator_1[row_group].values[0], accumulator_1[row_group].values[1]);
        }
        if constexpr (!kUpperRowsOnly) {
          if (output_row_1 < ActiveRows) {
            store_output_pair<StaticN != 1024>(
                row_target + static_cast<int64_t>(output_row_1) * target_stride + output_column,
                accumulator_0[row_group].values[2], accumulator_0[row_group].values[3]);
            store_output_pair<StaticN != 1024>(
                row_target + static_cast<int64_t>(output_row_1) * target_stride + output_column + 8,
                accumulator_1[row_group].values[2], accumulator_1[row_group].values[3]);
          }
        }
      } else {
        if (output_row_0 < size_m) {
          store_output_pair<false>(
              row_target + static_cast<int64_t>(output_row_0) * target_stride + output_column,
              accumulator_0[row_group].values[0], accumulator_0[row_group].values[1]);
          store_output_pair<false>(
              row_target + static_cast<int64_t>(output_row_0) * target_stride + output_column + 8,
              accumulator_1[row_group].values[0], accumulator_1[row_group].values[1]);
        }
        if (output_row_1 < size_m) {
          store_output_pair<false>(
              row_target + static_cast<int64_t>(output_row_1) * target_stride + output_column,
              accumulator_0[row_group].values[2], accumulator_0[row_group].values[3]);
          store_output_pair<false>(
              row_target + static_cast<int64_t>(output_row_1) * target_stride + output_column + 8,
              accumulator_1[row_group].values[2], accumulator_1[row_group].values[3]);
        }
      }
    }
  }
#endif
}

template <int TransitionBits, int Rows, int StageKTiles,
          bool UseSharedLevels = false>
__device__ __forceinline__ void accumulate_scalar_grouped_pgc(
    const uint32_t* __restrict__ words,
    uint8_t packed_bank_id,
    int pair_column,
    const half* __restrict__ input_stage,
    const half* __restrict__ levels,
    uint32_t alt_mask,
    float (&accumulator_0)[Rows],
    float (&accumulator_1)[Rows]) {
#pragma unroll
  for (int bank_group = 0; bank_group < 4; ++bank_group) {
    const uint32_t bank_mask_0 =
        selected_bank_mask(packed_bank_id, bank_group, alt_mask);
    const uint32_t bank_mask_8 =
        selected_bank_mask(packed_bank_id, bank_group + 4, alt_mask);
    uint32_t state_0[2];
    uint32_t state_8[2];
#pragma unroll
    for (int row_in_group = 0; row_in_group < 2; ++row_in_group) {
      const int row = bank_group * 2 + row_in_group;
      const int pair = row * 8 + pair_column;
      window_state_pair64<TransitionBits>(
          words, pair, state_0[row_in_group], state_8[row_in_group]);
    }
    uint32_t decoded_0[2];
    uint32_t decoded_8[2];
    decode_state_pair_bits<UseSharedLevels>(
        state_0[0], state_0[1], bank_mask_0, levels,
        decoded_0[0], decoded_0[1]);
    decode_state_pair_bits<UseSharedLevels>(
        state_8[0], state_8[1], bank_mask_8, levels,
        decoded_8[0], decoded_8[1]);
#pragma unroll
    for (int row_in_group = 0; row_in_group < 2; ++row_in_group) {
      const int row = bank_group * 2 + row_in_group;
      union {
        uint32_t bits;
        half2 values;
      } pair_0{decoded_0[row_in_group]}, pair_8{decoded_8[row_in_group]};
      const float weight_0 = __half2float(pair_0.values.x);
      const float weight_1 = __half2float(pair_0.values.y);
      const float weight_8 = __half2float(pair_8.values.x);
      const float weight_9 = __half2float(pair_8.values.y);
#pragma unroll
      for (int output_row = 0; output_row < Rows; ++output_row) {
        const int row_base = output_row * (StageKTiles * kTileRows);
        const float input_0 = __half2float(input_stage[row_base + row]);
        const float input_8 = __half2float(input_stage[row_base + row + 8]);
        accumulator_0[output_row] =
            fmaf(input_0, weight_0, accumulator_0[output_row]);
        accumulator_1[output_row] =
            fmaf(input_0, weight_1, accumulator_1[output_row]);
        accumulator_0[output_row] =
            fmaf(input_8, weight_8, accumulator_0[output_row]);
        accumulator_1[output_row] =
            fmaf(input_8, weight_9, accumulator_1[output_row]);
      }
    }
  }
}

template <int TransitionBits, int Rows, int StageKTiles,
          bool UseSharedLevels = false>
__device__ __forceinline__ void accumulate_scalar_hoisted_pgc(
    const uint32_t* __restrict__ words,
    uint8_t packed_bank_id,
    int pair_column,
    const half* __restrict__ input_stage,
    const half* __restrict__ levels,
    uint32_t alt_mask,
    float (&accumulator_0)[Rows],
    float (&accumulator_1)[Rows]) {
#pragma unroll
  for (int bank_group = 0; bank_group < 4; ++bank_group) {
    const uint32_t bank_mask_0 =
        selected_bank_mask(packed_bank_id, bank_group, alt_mask);
    const uint32_t bank_mask_8 =
        selected_bank_mask(packed_bank_id, bank_group + 4, alt_mask);
#pragma unroll
    for (int row_in_group = 0; row_in_group < 2; ++row_in_group) {
      const int row = bank_group * 2 + row_in_group;
      const int pair = row * 8 + pair_column;
      uint32_t state_0;
      uint32_t state_8;
      window_state_pair64<TransitionBits>(words, pair, state_0, state_8);
      const uint32_t decoded_0 =
          decode_state_bits<UseSharedLevels>(state_0, bank_mask_0, levels);
      const uint32_t decoded_8 =
          decode_state_bits<UseSharedLevels>(state_8, bank_mask_8, levels);
      union {
        uint32_t bits;
        half2 values;
      } pair_0{decoded_0}, pair_8{decoded_8};
      const float weight_0 = __half2float(pair_0.values.x);
      const float weight_1 = __half2float(pair_0.values.y);
      const float weight_8 = __half2float(pair_8.values.x);
      const float weight_9 = __half2float(pair_8.values.y);
#pragma unroll
      for (int output_row = 0; output_row < Rows; ++output_row) {
        const int row_base = output_row * (StageKTiles * kTileRows);
        const float input_0 = __half2float(input_stage[row_base + row]);
        const float input_8 = __half2float(input_stage[row_base + row + 8]);
        accumulator_0[output_row] =
            fmaf(input_0, weight_0, accumulator_0[output_row]);
        accumulator_1[output_row] =
            fmaf(input_0, weight_1, accumulator_1[output_row]);
        accumulator_0[output_row] =
            fmaf(input_8, weight_8, accumulator_0[output_row]);
        accumulator_1[output_row] =
            fmaf(input_8, weight_9, accumulator_1[output_row]);
      }
    }
  }
}

template <
    int TransitionBits,
    int Rows,
    int Threads = kM1Threads,
    int TilesPerBlock = kM1TilesPerBlock,
    int StageKTiles = kStageKTiles,
    int StaticN = 0,
    int StaticSplitCount = 0,
    int StaticK = 0,
    bool UseSharedLevels = false>
__device__ __forceinline__ void p32_window_ampere_m1_kernel_body(
    const half* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const half* __restrict__ levels,
    const uint8_t* __restrict__ bank_ids,
    float* __restrict__ partial_output,
    float* __restrict__ output,
    int size_k,
    int size_n,
    int split_count,
    const uint8_t* __restrict__ bank_alt_id,
    int n_block,
    int split,
    int payload_n_tiles,
    int payload_n_tile_offset,
    int64_t output_n_offset,
    int output_stride,
    int64_t partial_segment_offset) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  constexpr int kWordsPerTile = 4 * TransitionBits;
  static_assert(Rows >= 1 && Rows <= 8);
  __shared__ __align__(32) half input_tile[2][Rows * StageKTiles * kTileRows];
  __shared__ __align__(16) uint32_t packed_words[
      2][StageKTiles][TilesPerBlock][kWordsPerTile];
  __shared__ __align__(16) uint8_t packed_bank_ids[2][StageKTiles][TilesPerBlock];
  __shared__ __align__(16) half shared_levels[256];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  constexpr int kStaticNTiles = StaticN > 0 ? StaticN / kTileColumns : 0;
  const int n_tiles = StaticN > 0 ? kStaticNTiles : size_n / kTileColumns;
  constexpr int kEffectiveStaticSplitCount =
      StaticSplitCount > 0 ? StaticSplitCount : 0;
  const int effective_split_count = kEffectiveStaticSplitCount > 0
      ? kEffectiveStaticSplitCount
      : split_count;
  const int block_n_tile_base = n_block * TilesPerBlock;
  const int n_tile_base = block_n_tile_base + warp * 4;
  constexpr int kStaticKTiles = StaticK > 0 ? StaticK / kTileRows : 0;
  const int k_tiles = StaticK > 0 ? kStaticKTiles : size_k / kTileRows;
  const int input_stride = StaticK > 0 ? StaticK : size_k;
  const int k_tile_begin = (k_tiles * split) / effective_split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / effective_split_count;
  const uint32_t alt_mask = alternate_bank_mask<TransitionBits>(*bank_alt_id);
  const half* decode_levels = levels;
  if constexpr (UseSharedLevels) {
    for (int level = thread; level < 256; level += Threads) {
      shared_levels[level] = levels[level];
    }
    __syncthreads();
    decode_levels = shared_levels;
  }

  auto stage = [&](int k_tile_base, int destination) {
    auto* input_vectors = reinterpret_cast<uint4*>(input_tile[destination]);
    constexpr int kInputVectorsPerRow = StageKTiles * kTileRows / 8;
    for (int index = thread; index < Rows * kInputVectorsPerRow; index += Threads) {
      const int row = index / kInputVectorsPerRow;
      const int vector = index - row * kInputVectorsPerRow;
      const int source_column = k_tile_base * kTileRows + vector * 8;
      if (source_column < input_stride) {
        const half* source =
            input + static_cast<int64_t>(row) * input_stride + source_column;
        __pipeline_memcpy_async(input_vectors + index, reinterpret_cast<const uint4*>(source), 16);
      } else {
        input_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
      }
    }

    constexpr int kVectorsPerTile = kWordsPerTile / 4;
    constexpr int kVectorsPerKTile = TilesPerBlock * kVectorsPerTile;
    constexpr int kVectorsPerBlock = StageKTiles * kVectorsPerKTile;
    auto* destination_vectors = reinterpret_cast<uint4*>(packed_words[destination]);
    for (int index = thread; index < kVectorsPerBlock; index += Threads) {
      const int stage_k_tile = index / kVectorsPerKTile;
      const int tile_index = index - stage_k_tile * kVectorsPerKTile;
      const int tile = tile_index / kVectorsPerTile;
      const int vector = tile_index - tile * kVectorsPerTile;
      const int n_tile = block_n_tile_base + tile;
      const int k_tile = k_tile_base + stage_k_tile;
      if constexpr (StaticN > 0) {
        if (k_tile < k_tiles) {
          const int64_t global_tile = static_cast<int64_t>(k_tile) * payload_n_tiles +
              payload_n_tile_offset + n_tile;
          const auto* source_vectors = reinterpret_cast<const uint4*>(
              trellis + global_tile * kWordsPerTile);
          copy_async_cg_16(destination_vectors + index, source_vectors + vector);
        } else {
          destination_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
        }
      } else if (n_tile < n_tiles && k_tile < k_tiles) {
        const int64_t global_tile = static_cast<int64_t>(k_tile) * payload_n_tiles +
            payload_n_tile_offset + n_tile;
        const auto* source_vectors = reinterpret_cast<const uint4*>(
            trellis + global_tile * kWordsPerTile);
        copy_async_cg_16(destination_vectors + index, source_vectors + vector);
      } else {
        destination_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    if constexpr (StaticN == 1024 && TilesPerBlock == 16 && Rows == 1) {
      if (thread < StageKTiles * 4) {
        const int stage_k_tile = thread >> 2;
        const int word = thread & 3;
        const int k_tile = k_tile_base + stage_k_tile;
        auto* destination_ids = reinterpret_cast<uint32_t*>(
            packed_bank_ids[destination][stage_k_tile]);
        destination_ids[word] = k_tile < k_tiles
            ? __ldg(reinterpret_cast<const uint32_t*>(
                  bank_ids + static_cast<int64_t>(k_tile) * payload_n_tiles +
                      payload_n_tile_offset + block_n_tile_base) + word)
            : 0u;
      }
    } else if constexpr (
        StaticN > 0 && StaticN != 1024 && TilesPerBlock == 16 && Rows == 1 &&
        (TransitionBits == 5 || TransitionBits == 7 ||
         (TransitionBits == 4 &&
          (StaticN == 12288 || StaticN == 10240 || StaticN == 17408)))) {
      if (thread < StageKTiles) {
        const int k_tile = k_tile_base + thread;
        auto* destination_ids = reinterpret_cast<uint4*>(
            packed_bank_ids[destination][thread]);
        if (k_tile < k_tiles) {
          copy_async_ca_16(
              destination_ids,
              reinterpret_cast<const uint4*>(
                  bank_ids + static_cast<int64_t>(k_tile) * payload_n_tiles +
                      payload_n_tile_offset + block_n_tile_base));
        } else {
          *destination_ids = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    } else if constexpr (StaticN > 0 && TilesPerBlock == 16 && Rows == 2) {
      if (thread < StageKTiles) {
        const int k_tile = k_tile_base + thread;
        auto* destination_ids = reinterpret_cast<uint4*>(
            packed_bank_ids[destination][thread]);
        if constexpr (
            StaticN != 1024 &&
            (TransitionBits == 5 || TransitionBits == 7 ||
             (TransitionBits == 6 &&
              (StaticN == 5120 || StaticN == 17408)))) {
          if (k_tile < k_tiles) {
            copy_async_ca_16(
                destination_ids,
                reinterpret_cast<const uint4*>(
                    bank_ids + static_cast<int64_t>(k_tile) * payload_n_tiles +
                        payload_n_tile_offset + block_n_tile_base));
          } else {
            *destination_ids = make_uint4(0u, 0u, 0u, 0u);
          }
        } else {
          *destination_ids = k_tile < k_tiles
              ? __ldg(reinterpret_cast<const uint4*>(
                    bank_ids + static_cast<int64_t>(k_tile) * payload_n_tiles +
                        payload_n_tile_offset + block_n_tile_base))
              : make_uint4(0u, 0u, 0u, 0u);
        }
      }
    } else if constexpr (
        StaticN > 0 && TilesPerBlock == 16 && Rows == 4 && StaticN != 1024) {
      if (thread < StageKTiles) {
        const int k_tile = k_tile_base + thread;
        auto* destination_ids = reinterpret_cast<uint4*>(
            packed_bank_ids[destination][thread]);
        if (k_tile < k_tiles) {
          copy_async_ca_16(
              destination_ids,
              reinterpret_cast<const uint4*>(
                  bank_ids + static_cast<int64_t>(k_tile) * payload_n_tiles +
                      payload_n_tile_offset + block_n_tile_base));
        } else {
          *destination_ids = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    } else if (thread < StageKTiles * TilesPerBlock) {
      const int stage_k_tile = thread / TilesPerBlock;
      const int tile = thread - stage_k_tile * TilesPerBlock;
      const int n_tile = block_n_tile_base + tile;
      const int k_tile = k_tile_base + stage_k_tile;
      if constexpr (StaticN > 0) {
        packed_bank_ids[destination][stage_k_tile][tile] =
            k_tile < k_tiles ? bank_ids[static_cast<int64_t>(k_tile) * payload_n_tiles +
                payload_n_tile_offset + n_tile] : 0;
      } else {
        packed_bank_ids[destination][stage_k_tile][tile] =
            n_tile < n_tiles && k_tile < k_tiles
                ? bank_ids[static_cast<int64_t>(k_tile) * payload_n_tiles +
                    payload_n_tile_offset + n_tile]
                : 0;
      }
    }
    __pipeline_commit();
  };

  float accumulator_0[Rows] = {};
  float accumulator_1[Rows] = {};

  stage(k_tile_begin, 0);
  int parity = 0;
  for (int k_tile = k_tile_begin; k_tile < k_tile_end; k_tile += StageKTiles) {
    const bool has_next = k_tile + StageKTiles < k_tile_end;
    if (has_next) stage(k_tile + StageKTiles, parity ^ 1);
    __pipeline_wait_prior(has_next ? 1 : 0);
    __syncthreads();

    const int tile_in_warp = lane >> 3;
    const int pair_column = lane & 7;
    const int shared_tile = warp * 4 + tile_in_warp;
    const int n_tile = n_tile_base + tile_in_warp;
    if constexpr (StaticN > 0) {
      if (true) {
#pragma unroll
        for (int stage_k_tile = 0; stage_k_tile < StageKTiles; ++stage_k_tile) {
          if (k_tile + stage_k_tile >= k_tile_end) continue;
          const uint32_t* words = packed_words[parity][stage_k_tile][shared_tile];
          const uint8_t packed_bank_id = packed_bank_ids[parity][stage_k_tile][shared_tile];
          if constexpr (
              (Rows == 2 &&
               !(TransitionBits == 4 &&
                 (StaticN == 6144 ||
                  (StaticN == 5120 &&
                   StageKTiles != kScalarLongStageKTiles)))) ||
              (StaticN != 1024 &&
               ((Rows == 4 && TransitionBits != 7) || Rows == 1))) {
            accumulate_scalar_grouped_pgc<
                TransitionBits, Rows, StageKTiles, UseSharedLevels>(
                words,
                packed_bank_id,
                pair_column,
                input_tile[parity] + stage_k_tile * kTileRows,
                decode_levels,
                alt_mask,
                accumulator_0,
                accumulator_1);
          } else {
#pragma unroll
            for (int row = 0; row < 8; ++row) {
              const int pair = row * 8 + pair_column;
              uint32_t state_0, state_8;
              window_state_pair64<TransitionBits>(
                  words, pair, state_0, state_8);
              const uint32_t decoded_0 = decode_pair_bits<
                  TransitionBits, UseSharedLevels>(
                  pair, state_0, packed_bank_id, alt_mask, decode_levels);
              const uint32_t decoded_8 = decode_pair_bits<
                  TransitionBits, UseSharedLevels>(
                  pair + 64, state_8, packed_bank_id, alt_mask, decode_levels);
              union {
                uint32_t bits;
                half2 values;
              } pair_0{decoded_0}, pair_8{decoded_8};
              const float weight_0 = __half2float(pair_0.values.x);
              const float weight_1 = __half2float(pair_0.values.y);
              const float weight_8 = __half2float(pair_8.values.x);
              const float weight_9 = __half2float(pair_8.values.y);
#pragma unroll
              for (int output_row = 0; output_row < Rows; ++output_row) {
                const int row_base =
                    output_row * (StageKTiles * kTileRows) +
                    stage_k_tile * kTileRows;
                const float input_0 =
                    __half2float(input_tile[parity][row_base + row]);
                const float input_8 =
                    __half2float(input_tile[parity][row_base + row + 8]);
                accumulator_0[output_row] =
                    fmaf(input_0, weight_0, accumulator_0[output_row]);
                accumulator_1[output_row] =
                    fmaf(input_0, weight_1, accumulator_1[output_row]);
                accumulator_0[output_row] =
                    fmaf(input_8, weight_8, accumulator_0[output_row]);
                accumulator_1[output_row] =
                    fmaf(input_8, weight_9, accumulator_1[output_row]);
              }
            }
          }
        }
      }
    } else if (n_tile < n_tiles) {
#pragma unroll
      for (int stage_k_tile = 0; stage_k_tile < StageKTiles; ++stage_k_tile) {
        if (k_tile + stage_k_tile >= k_tile_end) continue;
        const uint32_t* words = packed_words[parity][stage_k_tile][shared_tile];
        const uint8_t packed_bank_id = packed_bank_ids[parity][stage_k_tile][shared_tile];
        if constexpr (
            Rows == 2 &&
            (TransitionBits == 5 || TransitionBits == 7)) {
          accumulate_scalar_hoisted_pgc<
              TransitionBits, Rows, StageKTiles, UseSharedLevels>(
              words,
              packed_bank_id,
              pair_column,
              input_tile[parity] + stage_k_tile * kTileRows,
              decode_levels,
              alt_mask,
              accumulator_0,
              accumulator_1);
        } else {
#pragma unroll
          for (int row = 0; row < 8; ++row) {
            const int pair = row * 8 + pair_column;
            uint32_t state_0, state_8;
            window_state_pair64<TransitionBits>(words, pair, state_0, state_8);
            const uint32_t decoded_0 = decode_pair_bits<
                TransitionBits, UseSharedLevels>(
                pair, state_0, packed_bank_id, alt_mask, decode_levels);
            const uint32_t decoded_8 = decode_pair_bits<
                TransitionBits, UseSharedLevels>(
                pair + 64, state_8, packed_bank_id, alt_mask, decode_levels);
            union {
              uint32_t bits;
              half2 values;
            } pair_0{decoded_0}, pair_8{decoded_8};
            const float weight_0 = __half2float(pair_0.values.x);
            const float weight_1 = __half2float(pair_0.values.y);
            const float weight_8 = __half2float(pair_8.values.x);
            const float weight_9 = __half2float(pair_8.values.y);
#pragma unroll
            for (int output_row = 0; output_row < Rows; ++output_row) {
              const int row_base =
                  output_row * (StageKTiles * kTileRows) +
                  stage_k_tile * kTileRows;
              const float input_0 =
                  __half2float(input_tile[parity][row_base + row]);
              const float input_8 =
                  __half2float(input_tile[parity][row_base + row + 8]);
              accumulator_0[output_row] =
                  fmaf(input_0, weight_0, accumulator_0[output_row]);
              accumulator_1[output_row] =
                  fmaf(input_0, weight_1, accumulator_1[output_row]);
              accumulator_0[output_row] =
                  fmaf(input_8, weight_8, accumulator_0[output_row]);
              accumulator_1[output_row] =
                  fmaf(input_8, weight_9, accumulator_1[output_row]);
            }
          }
        }
      }
    }
    __syncthreads();
    parity ^= 1;
  }

  float* target = effective_split_count == 1
      ? output + output_n_offset
      : partial_output + partial_segment_offset +
          static_cast<int64_t>(split) * Rows * size_n;
  const int target_stride = effective_split_count == 1 ? output_stride : size_n;
  const int n_tile = n_tile_base + (lane >> 3);
  if ((StaticN > 0 || n_tile < n_tiles) && (lane & 7) < 8) {
    const int output_column = n_tile * kTileColumns + (lane & 7) * 2;
#pragma unroll
    for (int output_row = 0; output_row < Rows; ++output_row) {
      float* row_target = target + static_cast<int64_t>(output_row) * target_stride;
      store_output_pair<(Rows == 1 && StaticN == 1024)>(
          row_target + output_column,
          accumulator_0[output_row],
          accumulator_1[output_row]);
    }
  }
#endif
}

template <
    int TransitionBits,
    int Rows,
    int Threads = kM1Threads,
    int TilesPerBlock = kM1TilesPerBlock,
    int StageKTiles = kStageKTiles,
    int StaticN = 0,
    int StaticSplitCount = 0,
    int StaticK = 0,
    bool UseSharedLevels = false>
__global__ __launch_bounds__(Threads) void p32_window_ampere_m1_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const half* __restrict__ levels,
    const uint8_t* __restrict__ bank_ids,
    float* __restrict__ partial_output,
    float* __restrict__ output,
    int size_k,
    int size_n,
    int split_count,
    const uint8_t* __restrict__ bank_alt_id) {
  p32_window_ampere_m1_kernel_body<
      TransitionBits, Rows, Threads, TilesPerBlock, StageKTiles, StaticN,
      StaticSplitCount, StaticK, UseSharedLevels>(
      input,
      trellis,
      levels,
      bank_ids,
      partial_output,
      output,
      size_k,
      size_n,
      split_count,
      bank_alt_id,
      static_cast<int>(blockIdx.x),
      static_cast<int>(blockIdx.z),
      size_n / kTileColumns,
      0,
      0,
      size_n,
      0);
}

template <
    int TransitionBits,
    bool FullRows,
    int ActiveRows = 0,
    int StaticN = 0,
    int Threads = kThreads,
    int TilesPerBlock = kTilesPerBlock,
    int StageKTiles = kStageKTiles,
    bool HoistBankMasks = false,
    bool UpperRowsOnly = false,
    int StaticK = 0>
__global__ __launch_bounds__(Threads) void p32_window_ampere_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const half* __restrict__ levels,
    const uint8_t* __restrict__ bank_ids,
    float* __restrict__ partial_output,
    float* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    const uint8_t* __restrict__ bank_alt_id) {
  p32_window_ampere_kernel_body<
      TransitionBits, FullRows, ActiveRows, StaticN, Threads, TilesPerBlock,
      StageKTiles, HoistBankMasks, UpperRowsOnly, StaticK>(
      input,
      trellis,
      levels,
      bank_ids,
      partial_output,
      output,
      size_m,
      size_k,
      size_n,
      split_count,
      bank_alt_id,
      static_cast<int>(blockIdx.x),
      static_cast<int>(blockIdx.z),
      size_n / kTileColumns,
      0,
      0,
      size_n,
      0);
}

// Large prefills share one 2-D grid across all 16-row tiles.  This keeps the
// ABI's global [split, M, N] workspace layout while avoiding one host launch
// (and one reduction) per row chunk.
template <
    int TransitionBits,
    int Threads,
    int StageKTiles,
    int StaticN = 0,
    int StaticK = 0>
__global__ __launch_bounds__(Threads) void p32_window_ampere_large_m_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const half* __restrict__ levels,
    const uint8_t* __restrict__ bank_ids,
    float* __restrict__ partial_output,
    float* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    const uint8_t* __restrict__ bank_alt_id) {
  const int row_offset = static_cast<int>(blockIdx.y) * kRows;
  const int local_m = min(kRows, size_m - row_offset);
  if (local_m <= 0) return;
  const int n_tiles = size_n / kTileColumns;
  const int n_block = static_cast<int>(blockIdx.x);
  const int split = static_cast<int>(blockIdx.z);
#define QVQ_LARGE_M_BODY(FULL_ROWS) \
  p32_window_ampere_kernel_body< \
      TransitionBits, FULL_ROWS, 0, StaticN, Threads, Threads / 32, \
      StageKTiles, false, false, StaticK>( \
      input + static_cast<int64_t>(row_offset) * size_k, \
      trellis, levels, bank_ids, partial_output, \
      output + static_cast<int64_t>(row_offset) * size_n, local_m, size_k, \
      size_n, split_count, bank_alt_id, n_block, split, n_tiles, 0, 0, \
      size_n, static_cast<int64_t>(row_offset) * size_n, \
      static_cast<int64_t>(size_m) * size_n)
  if (local_m == kRows) QVQ_LARGE_M_BODY(true);
  else QVQ_LARGE_M_BODY(false);
#undef QVQ_LARGE_M_BODY
}

// Reuse one packed N-tile set across two adjacent 16-row groups.  Each warp
// keeps two accumulator fragments, so the second row group avoids repeating
// trellis loads and state decoding while retaining the four-warp N layout.
template <int TransitionBits, int StaticN, int StaticK = 0,
          int StageKTiles = 1, int RowGroups = 2>
__global__ __launch_bounds__(128) void p32_window_ampere_large_m2_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const half* __restrict__ levels,
    const uint8_t* __restrict__ bank_ids,
    float* __restrict__ partial_output,
    float* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    const uint8_t* __restrict__ bank_alt_id) {
  const int row_offset = static_cast<int>(blockIdx.y) * (RowGroups * kRows);
  const int local_m = min(RowGroups * kRows, size_m - row_offset);
  if (local_m < RowGroups * kRows) return;
  const int n_tiles = size_n / kTileColumns;
  const int n_block = static_cast<int>(blockIdx.x);
  const int split = static_cast<int>(blockIdx.z);
  p32_window_ampere_kernel_body<
      TransitionBits, true, 0, StaticN, 128, 4, StageKTiles,
      true, false, StaticK, RowGroups,
      (RowGroups == 16 && StageKTiles >= 3)>(
      input + static_cast<int64_t>(row_offset) * size_k,
      trellis, levels, bank_ids, partial_output,
      output + static_cast<int64_t>(row_offset) * size_n, local_m, size_k,
      size_n, split_count, bank_alt_id, n_block, split, n_tiles, 0, 0,
      size_n, static_cast<int64_t>(row_offset) * size_n,
      static_cast<int64_t>(size_m) * size_n);
}

constexpr int kMaxGroupedP32Segments = 3;

struct GroupedP32LaunchParams {
  int segment_count;
  int split_count[kMaxGroupedP32Segments];
  int n_tile_start[kMaxGroupedP32Segments];
  int n_tiles[kMaxGroupedP32Segments];
  int64_t output_offset[kMaxGroupedP32Segments];
  int64_t partial_offset[kMaxGroupedP32Segments];
};

void set_last_error(const char* message);

bool build_grouped_params(
    int size_m,
    int size_n,
    int group_count,
    int split_count_0,
    int split_count_1,
    int split_count_2,
    int n_tile_end_0,
    int n_tile_end_1,
    GroupedP32LaunchParams* params) {
  const int total_n_tiles = size_n / kTileColumns;
  const int ends[kMaxGroupedP32Segments] = {
      n_tile_end_0,
      n_tile_end_1,
      total_n_tiles,
  };
  *params = {};
  params->segment_count = group_count;
  params->split_count[0] = split_count_0;
  params->split_count[1] = split_count_1;
  params->split_count[2] = split_count_2;
  int tile_start = 0;
  int64_t output_offset = 0;
  int64_t partial_offset = 0;
  for (int segment = 0; segment < group_count; ++segment) {
    const int tile_end = ends[segment];
    if (tile_end <= tile_start || tile_end > total_n_tiles) {
      set_last_error("QVQ P32 grouped N-tile boundaries are invalid");
      return false;
    }
    params->n_tile_start[segment] = tile_start;
    params->n_tiles[segment] = tile_end - tile_start;
    params->output_offset[segment] = output_offset;
    params->partial_offset[segment] = partial_offset;
    output_offset += static_cast<int64_t>(size_m) *
        params->n_tiles[segment] * kTileColumns;
    if (params->split_count[segment] > 1) {
      partial_offset += static_cast<int64_t>(params->split_count[segment]) *
          size_m * params->n_tiles[segment] * kTileColumns;
    }
    tile_start = tile_end;
  }
  if (tile_start != total_n_tiles) {
    set_last_error("QVQ P32 grouped boundaries must cover all output tiles");
    return false;
  }
  return true;
}

bool grouped_work_count(
    const GroupedP32LaunchParams& params,
    int tiles_per_block,
    int64_t* grouped_work) {
  *grouped_work = 0;
  for (int segment = 0; segment < params.segment_count; ++segment) {
    const int segment_blocks =
        (params.n_tiles[segment] + tiles_per_block - 1) / tiles_per_block;
    *grouped_work += static_cast<int64_t>(segment_blocks) *
        params.split_count[segment];
  }
  if (*grouped_work <= 0 ||
      *grouped_work > std::numeric_limits<unsigned>::max()) {
    set_last_error("QVQ P32 grouped work exceeds the CUDA grid limit");
    return false;
  }
  return true;
}

bool validate_grouped_splits(
    const GroupedP32LaunchParams& params, int size_k) {
  for (int segment = 0; segment < params.segment_count; ++segment) {
    if (params.split_count[segment] < 1 ||
        params.split_count[segment] > QVQ_P32_SPLIT_COUNT_MAX ||
        params.split_count[segment] > size_k / kTileRows) {
      set_last_error(
          "QVQ P32 grouped split count exceeds the K-tile count or 128");
      return false;
    }
  }
  return true;
}

template <int TransitionBits, int Rows, int Threads, int StageKTiles,
          bool UseSharedLevels = true>
__global__ __launch_bounds__(Threads) void p32_window_ampere_grouped_scalar_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const half* __restrict__ levels,
    const uint8_t* __restrict__ bank_ids,
    const uint8_t* __restrict__ bank_alt_ids,
    const __grid_constant__ GroupedP32LaunchParams params,
    float* __restrict__ partial_output,
    float* __restrict__ output,
    int size_k,
    int total_n_tiles) {
  constexpr int kTilesPerBlock = 4 * (Threads / 32);
  int work = static_cast<int>(blockIdx.x);
  int segment = 0;
  for (; segment < params.segment_count; ++segment) {
    const int segment_blocks =
        (params.n_tiles[segment] + kTilesPerBlock - 1) / kTilesPerBlock;
    const int segment_work = segment_blocks * params.split_count[segment];
    if (work < segment_work) break;
    work -= segment_work;
  }
  if (segment >= params.segment_count) return;
  const int split = work % params.split_count[segment];
  const int n_block = work / params.split_count[segment];
  const int n_tiles = params.n_tiles[segment];
  const uint8_t bank_alt_id = bank_alt_ids[segment];
  p32_window_ampere_m1_kernel_body<
      TransitionBits, Rows, Threads, kTilesPerBlock, StageKTiles, 0, 0, 0,
      UseSharedLevels>(
      input,
      trellis,
      levels,
      bank_ids,
      partial_output,
      output,
      size_k,
      n_tiles * kTileColumns,
      params.split_count[segment],
      &bank_alt_id,
      n_block,
      split,
      total_n_tiles,
      params.n_tile_start[segment],
      static_cast<int64_t>(params.n_tile_start[segment]) * kTileColumns,
      total_n_tiles * kTileColumns,
      params.partial_offset[segment]);
}

template <int TransitionBits, int Threads, int StageKTiles, bool FullRows, int ActiveRows>
__global__ __launch_bounds__(Threads) void p32_window_ampere_grouped_block_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const half* __restrict__ levels,
    const uint8_t* __restrict__ bank_ids,
    const uint8_t* __restrict__ bank_alt_ids,
    const __grid_constant__ GroupedP32LaunchParams params,
    float* __restrict__ partial_output,
    float* __restrict__ output,
    int size_m,
    int size_k,
    int total_n_tiles) {
  constexpr int kTilesPerBlock = Threads / 32;
  int work = static_cast<int>(blockIdx.x);
  int segment = 0;
  for (; segment < params.segment_count; ++segment) {
    const int segment_blocks =
        (params.n_tiles[segment] + kTilesPerBlock - 1) / kTilesPerBlock;
    const int segment_work = segment_blocks * params.split_count[segment];
    if (work < segment_work) break;
    work -= segment_work;
  }
  if (segment >= params.segment_count) return;
  const int split = work % params.split_count[segment];
  const int n_block = work / params.split_count[segment];
  const int n_tiles = params.n_tiles[segment];
  if constexpr (FullRows) {
    p32_window_ampere_kernel_body<
        TransitionBits, true, 0, 0, Threads, kTilesPerBlock, StageKTiles>(
        input,
        trellis,
        levels,
        bank_ids,
        partial_output,
        output,
        size_m,
        size_k,
        n_tiles * kTileColumns,
        params.split_count[segment],
        bank_alt_ids + segment,
        n_block,
        split,
        total_n_tiles,
        params.n_tile_start[segment],
        static_cast<int64_t>(params.n_tile_start[segment]) * kTileColumns,
        total_n_tiles * kTileColumns,
        params.partial_offset[segment]);
  } else if constexpr (ActiveRows == 8) {
    p32_window_ampere_kernel_body<
        TransitionBits, false, 8, 0, Threads, kTilesPerBlock, StageKTiles,
        false, false>(
        input,
        trellis,
        levels,
        bank_ids,
        partial_output,
        output,
        size_m,
        size_k,
        n_tiles * kTileColumns,
        params.split_count[segment],
        bank_alt_ids + segment,
        n_block,
        split,
        total_n_tiles,
        params.n_tile_start[segment],
        static_cast<int64_t>(params.n_tile_start[segment]) * kTileColumns,
        total_n_tiles * kTileColumns,
        params.partial_offset[segment]);
  } else {
    p32_window_ampere_kernel_body<
        TransitionBits, false, 0, 0, Threads, kTilesPerBlock, StageKTiles>(
      input,
      trellis,
      levels,
      bank_ids,
      partial_output,
      output,
      size_m,
      size_k,
      n_tiles * kTileColumns,
      params.split_count[segment],
      bank_alt_ids + segment,
      n_block,
      split,
      total_n_tiles,
      params.n_tile_start[segment],
      static_cast<int64_t>(params.n_tile_start[segment]) * kTileColumns,
      total_n_tiles * kTileColumns,
      params.partial_offset[segment]);
  }
}

thread_local char last_error[256] = {};

void set_last_error(const char* message) {
  const char* text = message == nullptr ? "unknown CUDA error" : message;
  std::snprintf(last_error, sizeof(last_error), "%s", text);
}

bool normalize_external_tuning(
    const qvq_p32_config* requested,
    int size_m,
    qvq_p32_config* resolved) {
  if (requested == nullptr || resolved == nullptr) {
    set_last_error("QVQ P32 tuning config is null");
    return false;
  }
  *resolved = *requested;
  if (resolved->tuning_mode != QVQ_P32_TUNING_AUTO &&
      resolved->tuning_mode != QVQ_P32_TUNING_EXTERNAL) {
    set_last_error("QVQ P32 tuning mode must be auto (0) or external (1)");
    return false;
  }
  int threads = resolved->threads;
  if (resolved->n_warps != QVQ_P32_WARPS_AUTO) {
    if (resolved->n_warps != 2 && resolved->n_warps != 4 &&
        resolved->n_warps != 8) {
      set_last_error("QVQ P32 n_warps must be auto, 2, 4, or 8");
      return false;
    }
    const int requested_threads = resolved->n_warps * 32;
    if (threads != 0 && threads != requested_threads) {
      set_last_error("QVQ P32 threads and n_warps disagree");
      return false;
    }
    threads = requested_threads;
  }
  if (threads == 0) threads = 128;
  if (threads != 64 && threads != 128 && threads != 256) {
    set_last_error("QVQ P32 threads must be 64, 128, or 256");
    return false;
  }
  const int warps = threads / 32;
  if (resolved->n_warps != QVQ_P32_WARPS_AUTO &&
      resolved->n_warps != warps) {
    set_last_error("QVQ P32 n_warps does not match threads");
    return false;
  }
  const int native_tiles = resolved->kernel_variant == QVQ_P32_VARIANT_SCALAR
      ? 4 * warps
      : warps;
  if (resolved->n_tiles_per_block != QVQ_P32_N_TILES_AUTO &&
      resolved->n_tiles_per_block != native_tiles) {
    set_last_error(
        "QVQ P32 n_tiles_per_block is not supported by this compiled variant");
    return false;
  }
  if (size_m < 1) {
    set_last_error("QVQ P32 tuning config received an invalid M");
    return false;
  }
  resolved->threads = threads;
  resolved->n_warps = warps;
  resolved->n_tiles_per_block = native_tiles;
  return true;
}

template <
    int TransitionBits,
    int Rows,
    int Threads,
    int TilesPerBlock,
    int StageKTiles,
    int StaticN,
    int StaticSplitCount,
    int StaticK = 0>
bool launch_fixed_n_scalar_kernel(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    float* partial_output,
    float* output,
    int size_k,
    int size_n,
    int split_count,
    const uint8_t* bank_alt_id,
    const dim3 grid,
    const cudaStream_t stream) {
  p32_window_ampere_m1_kernel<
      TransitionBits, Rows, Threads, TilesPerBlock, StageKTiles, StaticN,
      StaticSplitCount, StaticK><<<grid, Threads, 0, stream>>>(
          input, trellis, levels, bank_ids, partial_output, output, size_k,
          size_n, split_count, bank_alt_id);
  return true;
}

template <
    int TransitionBits,
    int Rows,
    int Threads = kM1Threads,
    int TilesPerBlock = kM1TilesPerBlock,
    int StageKTiles = kStageKTiles,
    int StaticSplitCount = 0,
    int StaticK = 0>
bool launch_static_n_scalar_kernel(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    float* partial_output,
    float* output,
    int size_k,
    int size_n,
    int split_count,
    const uint8_t* bank_alt_id,
    const dim3 grid,
    const cudaStream_t stream) {
#define QVQ_LAUNCH_STATIC_N(N)                                                    \
  case N:                                                                         \
    p32_window_ampere_m1_kernel<TransitionBits, Rows, Threads, TilesPerBlock,     \
                                StageKTiles, N, StaticSplitCount, StaticK>         \
        <<<grid, Threads, 0, stream>>>(                                           \
            input, trellis, levels, bank_ids, partial_output, output, size_k,     \
            size_n, split_count, bank_alt_id);                                    \
    return true
  switch (size_n) {
    QVQ_LAUNCH_STATIC_N(512);
    QVQ_LAUNCH_STATIC_N(2048);
    QVQ_LAUNCH_STATIC_N(8192);
    QVQ_LAUNCH_STATIC_N(1024);
    QVQ_LAUNCH_STATIC_N(5120);
    QVQ_LAUNCH_STATIC_N(6144);
    QVQ_LAUNCH_STATIC_N(10240);
    QVQ_LAUNCH_STATIC_N(12288);
    QVQ_LAUNCH_STATIC_N(17408);
    default:
      return false;
  }
#undef QVQ_LAUNCH_STATIC_N
}

template <int TransitionBits, int StaticN, int Threads, int StageKTiles>
void launch_static_n_block_width(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    float* partial_output,
    float* output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    const uint8_t* bank_alt_id,
    const dim3 grid,
    const cudaStream_t stream) {
  if (size_m == 8) {
    if constexpr (
        Threads == kThreads && StageKTiles == kStageKTiles &&
        (StaticN == 1024 || StaticN == 10240)) {
      if (size_k == 5120) {
        p32_window_ampere_kernel<
            TransitionBits, false, 8, StaticN, Threads, Threads / 32,
            StageKTiles, true, true, 5120>
            <<<grid, Threads, 0, stream>>>(
                input, trellis, levels, bank_ids, partial_output, output,
                size_m, size_k, size_n, split_count, bank_alt_id);
        return;
      }
    }
    p32_window_ampere_kernel<
        TransitionBits, false, 8, StaticN, Threads, Threads / 32,
        StageKTiles, (StaticN != 17408), true>
        <<<grid, Threads, 0, stream>>>(
            input, trellis, levels, bank_ids, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_id);
    return;
  }

  if constexpr (Threads == kThreads && StageKTiles == kStageKTiles) {
    if constexpr (
        StaticN == 1024 || StaticN == 6144 || StaticN == 10240 ||
        StaticN == 12288 || StaticN == 17408) {
      if (size_k == 5120) {
        p32_window_ampere_kernel<
            TransitionBits, true, 0, StaticN, Threads, Threads / 32,
            StageKTiles, (StaticN == 6144 || StaticN == 10240), false, 5120>
            <<<grid, Threads, 0, stream>>>(
                input, trellis, levels, bank_ids, partial_output, output,
                size_m, size_k, size_n, split_count, bank_alt_id);
        return;
      }
    }
    if constexpr (StaticN == 5120) {
      if (size_k == 6144) {
        p32_window_ampere_kernel<
            TransitionBits, true, 0, StaticN, Threads, Threads / 32,
            StageKTiles, false, false, 6144>
            <<<grid, Threads, 0, stream>>>(
                input, trellis, levels, bank_ids, partial_output, output,
                size_m, size_k, size_n, split_count, bank_alt_id);
        return;
      }
      if (size_k == 17408) {
        p32_window_ampere_kernel<
            TransitionBits, true, 0, StaticN, Threads, Threads / 32,
            StageKTiles, false, false, 17408>
            <<<grid, Threads, 0, stream>>>(
                input, trellis, levels, bank_ids, partial_output, output,
                size_m, size_k, size_n, split_count, bank_alt_id);
        return;
      }
    }
  }
  p32_window_ampere_kernel<
      TransitionBits, true, 0, StaticN, Threads, Threads / 32, StageKTiles,
      (StaticN == 6144 || StaticN == 10240)>
      <<<grid, Threads, 0, stream>>>(
          input, trellis, levels, bank_ids, partial_output, output,
          size_m, size_k, size_n, split_count, bank_alt_id);
}

template <int TransitionBits, int Threads, int StageKTiles>
bool launch_static_n_block_kernel(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    float* partial_output,
    float* output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    const uint8_t* bank_alt_id,
    const dim3 grid,
    const cudaStream_t stream) {
#define QVQ_LAUNCH_STATIC_BLOCK_N(N)                                      \
  case N:                                                                 \
    launch_static_n_block_width<TransitionBits, N, Threads, StageKTiles>( \
        input, trellis, levels, bank_ids, partial_output, output, size_m, \
        size_k, size_n, split_count, bank_alt_id, grid, stream);           \
    return true
  switch (size_n) {
    QVQ_LAUNCH_STATIC_BLOCK_N(512);
    QVQ_LAUNCH_STATIC_BLOCK_N(2048);
    QVQ_LAUNCH_STATIC_BLOCK_N(8192);
    QVQ_LAUNCH_STATIC_BLOCK_N(1024);
    QVQ_LAUNCH_STATIC_BLOCK_N(5120);
    QVQ_LAUNCH_STATIC_BLOCK_N(6144);
    QVQ_LAUNCH_STATIC_BLOCK_N(10240);
    QVQ_LAUNCH_STATIC_BLOCK_N(12288);
    QVQ_LAUNCH_STATIC_BLOCK_N(17408);
    default:
      return false;
  }
#undef QVQ_LAUNCH_STATIC_BLOCK_N
}

template <int StaticSplitCount = 0>
__global__ void reduce_split_kernel(
    const float* __restrict__ partial_output,
    float* __restrict__ output,
    int output_values,
    int split_count) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= output_values) return;
  float accumulator = 0.0f;
#pragma unroll
  for (int split = 0;
       split < (StaticSplitCount > 0 ? StaticSplitCount : split_count);
       ++split) {
    accumulator += partial_output[static_cast<int64_t>(split) * output_values + index];
  }
  output[index] = accumulator;
}

template <int StaticSplitCount = 0>
__global__ void reduce_split_grouped_kernel(
    const float* __restrict__ partial_output,
    float* __restrict__ output,
    int size_m,
    int group_n,
    int total_n,
    int group_n_offset,
    int split_count) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int group_values = size_m * group_n;
  if (index >= group_values) return;
  const int row = index / group_n;
  const int column = index - row * group_n;
  float accumulator = 0.0f;
#pragma unroll
  for (int split = 0;
       split < (StaticSplitCount > 0 ? StaticSplitCount : split_count);
       ++split) {
    accumulator += partial_output[static_cast<int64_t>(split) * group_values + index];
  }
  output[static_cast<int64_t>(row) * total_n + group_n_offset + column] = accumulator;
}

void launch_split_reduction_grouped(
    const float* partial_output,
    float* output,
    int size_m,
    int group_n,
    int total_n,
    int group_n_offset,
    int split_count,
    cudaStream_t stream) {
  const int group_values = size_m * group_n;
  const int blocks = (group_values + kReductionThreads - 1) / kReductionThreads;
  reduce_split_grouped_kernel<<<blocks, kReductionThreads, 0, stream>>>(
      partial_output, output, size_m, group_n, total_n, group_n_offset,
      split_count);
}

template <int StaticSplitCount>
__global__ void reduce_split_warp_kernel(
    const float* __restrict__ partial_output,
    float* __restrict__ output,
    int output_values) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  constexpr int kOutputsPerWarp = 4;
  constexpr int kSplitLanes = 32 / kOutputsPerWarp;
  const int index =
      (static_cast<int>(blockIdx.x) * warps_per_block + warp) *
          kOutputsPerWarp +
      (lane & (kOutputsPerWarp - 1));
  if (index >= output_values) return;
  float accumulator = 0.0f;
#pragma unroll
  for (int split = lane / kOutputsPerWarp;
       split < StaticSplitCount;
       split += kSplitLanes) {
    accumulator +=
        partial_output[static_cast<int64_t>(split) * output_values + index];
  }
#pragma unroll
  for (int offset = 16; offset >= kOutputsPerWarp; offset >>= 1) {
    accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
  }
  if (lane < kOutputsPerWarp) output[index] = accumulator;
}

void launch_split_reduction(
    const float* partial_output,
    float* output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    cudaStream_t stream) {
  constexpr int kReductionThreads = 256;
  const int output_values = size_m * size_n;
  if (size_m <= 4 && size_n == 1024 && split_count == 64) {
    constexpr int kReductionWarps = kReductionThreads / 32;
    const int blocks =
        (output_values + kReductionWarps * 4 - 1) / (kReductionWarps * 4);
    reduce_split_warp_kernel<64><<<blocks, kReductionThreads, 0, stream>>>(
        partial_output, output, output_values);
    return;
  }
  if (size_m == 8 && size_n == 1024 && split_count == 48) {
    constexpr int kReductionWarps = kReductionThreads / 32;
    const int blocks =
        (output_values + kReductionWarps * 4 - 1) / (kReductionWarps * 4);
    reduce_split_warp_kernel<48><<<blocks, kReductionThreads, 0, stream>>>(
        partial_output, output, output_values);
    return;
  }

  const int blocks =
      (output_values + kReductionThreads - 1) / kReductionThreads;
  if (size_m == 1 && size_k == 5120 && size_n == 1024 &&
      split_count == 56) {
    reduce_split_kernel<56><<<blocks, kReductionThreads, 0, stream>>>(
        partial_output, output, output_values, 56);
    return;
  }
  const bool use_static_reducer =
      size_n != 1024 &&
      (size_m == 2 || size_m == 4 || size_m == 8 || size_m == 16 ||
       (size_m == 1 &&
        (size_n == 12288 || size_n == 10240 || size_n == 17408 ||
         (size_k == 17408 && size_n == 5120))));
#define QVQ_LAUNCH_STATIC_REDUCER(SPLITS)                                  \
  reduce_split_kernel<SPLITS><<<blocks, kReductionThreads, 0, stream>>>(   \
      partial_output, output, output_values, SPLITS)
  if (use_static_reducer) {
    switch (split_count) {
      case 9: QVQ_LAUNCH_STATIC_REDUCER(9); return;
      case 10: QVQ_LAUNCH_STATIC_REDUCER(10); return;
      case 12: QVQ_LAUNCH_STATIC_REDUCER(12); return;
      case 14: QVQ_LAUNCH_STATIC_REDUCER(14); return;
      case 16: QVQ_LAUNCH_STATIC_REDUCER(16); return;
      case 20: QVQ_LAUNCH_STATIC_REDUCER(20); return;
      case 24: QVQ_LAUNCH_STATIC_REDUCER(24); return;
      case 32: QVQ_LAUNCH_STATIC_REDUCER(32); return;
      case 40: QVQ_LAUNCH_STATIC_REDUCER(40); return;
      case 48: QVQ_LAUNCH_STATIC_REDUCER(48); return;
      case 64: QVQ_LAUNCH_STATIC_REDUCER(64); return;
      case 96: QVQ_LAUNCH_STATIC_REDUCER(96); return;
      case 128: QVQ_LAUNCH_STATIC_REDUCER(128); return;
      default: break;
    }
  }
#undef QVQ_LAUNCH_STATIC_REDUCER
  reduce_split_kernel<<<blocks, kReductionThreads, 0, stream>>>(
      partial_output, output, output_values, split_count);
}

int finish_grouped_reduction(
    const GroupedP32LaunchParams& params, float* output, float* partial_output,
    int size_m, int total_n_tiles, int reduction_mode, cudaStream_t stream) {
  if (reduction_mode == QVQ_P32_REDUCTION_PARTIALS) return 0;
  for (int segment = 0; segment < params.segment_count; ++segment) {
    if (params.split_count[segment] > 1) {
      launch_split_reduction_grouped(
          partial_output + params.partial_offset[segment],
          output,
          size_m,
          params.n_tiles[segment] * kTileColumns,
          total_n_tiles * kTileColumns,
          params.n_tile_start[segment] * kTileColumns,
          params.split_count[segment],
          stream);
      const cudaError_t error = cudaGetLastError();
      if (error != cudaSuccess) return static_cast<int>(error);
    }
  }
  return 0;
}

template <int TransitionBits, int Rows, int Threads, int StageKTiles>
int launch_p32_grouped_scalar_stage(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    const uint8_t* bank_alt_ids,
    const GroupedP32LaunchParams& params,
    float* output,
    float* partial_output,
    int size_k,
    int total_n_tiles,
    cudaStream_t stream) {
  constexpr int kTilesPerBlock = 4 * (Threads / 32);
  int64_t grouped_work = 0;
  if (!grouped_work_count(params, kTilesPerBlock, &grouped_work)) return -1;
  const dim3 grid(static_cast<unsigned>(grouped_work));
  p32_window_ampere_grouped_scalar_kernel<TransitionBits, Rows, Threads, StageKTiles>
      <<<grid, Threads, 0, stream>>>(
          input, trellis, levels, bank_ids, bank_alt_ids, params,
          partial_output, output, size_k, total_n_tiles);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return static_cast<int>(error);

  return 0;
}

template <int TransitionBits, int Rows, int Threads>
int launch_p32_grouped_scalar_stage(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    const uint8_t* bank_alt_ids,
    const GroupedP32LaunchParams& params,
    float* output,
    float* partial_output,
    int size_k,
    int total_n_tiles,
    int stage_k_tiles,
    cudaStream_t stream) {
  switch (stage_k_tiles) {
    case 1:
      return launch_p32_grouped_scalar_stage<TransitionBits, Rows, Threads, 1>(
          input, trellis, levels, bank_ids, bank_alt_ids, params, output, partial_output,
          size_k, total_n_tiles, stream);
    case 2:
      return launch_p32_grouped_scalar_stage<TransitionBits, Rows, Threads, 2>(
          input, trellis, levels, bank_ids, bank_alt_ids, params, output, partial_output,
          size_k, total_n_tiles, stream);
    case 3:
      return launch_p32_grouped_scalar_stage<TransitionBits, Rows, Threads, 3>(
          input, trellis, levels, bank_ids, bank_alt_ids, params, output, partial_output,
          size_k, total_n_tiles, stream);
    case 4:
      return launch_p32_grouped_scalar_stage<TransitionBits, Rows, Threads, 4>(
          input, trellis, levels, bank_ids, bank_alt_ids, params, output, partial_output,
          size_k, total_n_tiles, stream);
    default:
      set_last_error("QVQ P32 grouped stage_k_tiles must be in [1, 4]");
      return -1;
  }
}

template <int TransitionBits, int Rows>
int launch_p32_grouped_scalar_threads(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    const uint8_t* bank_alt_ids,
    const GroupedP32LaunchParams& params,
    float* output,
    float* partial_output,
    int size_k,
    int total_n_tiles,
    int threads,
    int stage_k_tiles,
    cudaStream_t stream) {
  switch (threads) {
    case 64:
      return launch_p32_grouped_scalar_stage<TransitionBits, Rows, 64>(
          input, trellis, levels, bank_ids, bank_alt_ids, params, output, partial_output,
          size_k, total_n_tiles, stage_k_tiles, stream);
    case 128:
      return launch_p32_grouped_scalar_stage<TransitionBits, Rows, 128>(
          input, trellis, levels, bank_ids, bank_alt_ids, params, output, partial_output,
          size_k, total_n_tiles, stage_k_tiles, stream);
    case 256:
      return launch_p32_grouped_scalar_stage<TransitionBits, Rows, 256>(
          input, trellis, levels, bank_ids, bank_alt_ids, params, output, partial_output,
          size_k, total_n_tiles, stage_k_tiles, stream);
    default:
      set_last_error("QVQ P32 grouped threads must be one of 64, 128, or 256");
      return -1;
  }
}

template <int TransitionBits, int Threads, int StageKTiles>
int launch_p32_grouped_block_stage(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    const uint8_t* bank_alt_ids,
    const GroupedP32LaunchParams& params,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int total_n_tiles,
    cudaStream_t stream) {
  constexpr int kTilesPerBlock = Threads / 32;
  int64_t grouped_work = 0;
  if (!grouped_work_count(params, kTilesPerBlock, &grouped_work)) return -1;
#define QVQ_GROUPED_BLOCK_LAUNCH(FULL_ROWS, ACTIVE_ROWS)                    \
  p32_window_ampere_grouped_block_kernel<                                       \
      TransitionBits, Threads, StageKTiles, FULL_ROWS, ACTIVE_ROWS>             \
      <<<static_cast<unsigned>(grouped_work), Threads, 0, stream>>>(             \
          input, trellis, levels, bank_ids, bank_alt_ids, params,                \
          partial_output, output, size_m, size_k, total_n_tiles)
  if (size_m == 8) {
    QVQ_GROUPED_BLOCK_LAUNCH(false, 8);
  } else if (size_m == 16) {
    QVQ_GROUPED_BLOCK_LAUNCH(true, 0);
  } else {
    QVQ_GROUPED_BLOCK_LAUNCH(false, 0);
  }
#undef QVQ_GROUPED_BLOCK_LAUNCH
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return static_cast<int>(error);

  return 0;
}

template <int TransitionBits, int Threads>
int launch_p32_grouped_block_threads(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    const uint8_t* bank_alt_ids,
    const GroupedP32LaunchParams& params,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int total_n_tiles,
    int stage_k_tiles,
    cudaStream_t stream) {
  switch (stage_k_tiles) {
    case 1:
      return launch_p32_grouped_block_stage<TransitionBits, Threads, 1>(
          input, trellis, levels, bank_ids, bank_alt_ids, params, output,
          partial_output, size_m, size_k, total_n_tiles, stream);
    case 2:
      return launch_p32_grouped_block_stage<TransitionBits, Threads, 2>(
          input, trellis, levels, bank_ids, bank_alt_ids, params, output,
          partial_output, size_m, size_k, total_n_tiles, stream);
    case 3:
      return launch_p32_grouped_block_stage<TransitionBits, Threads, 3>(
          input, trellis, levels, bank_ids, bank_alt_ids, params, output,
          partial_output, size_m, size_k, total_n_tiles, stream);
    case 4:
      return launch_p32_grouped_block_stage<TransitionBits, Threads, 4>(
          input, trellis, levels, bank_ids, bank_alt_ids, params, output,
          partial_output, size_m, size_k, total_n_tiles, stream);
    default:
      set_last_error("QVQ P32 grouped stage_k_tiles must be in [1, 4]");
      return -1;
  }
}

template <int TransitionBits>
int launch_p32_grouped_block(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_ids,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int group_count,
    int split_count_0,
    int split_count_1,
    int split_count_2,
    int n_tile_end_0,
    int n_tile_end_1,
    const qvq_p32_config& config,
    void* stream) {
  if (config.kernel_variant != QVQ_P32_VARIANT_BLOCK || config.static_n != 0 ||
      (config.reduction_mode != QVQ_P32_REDUCTION_NATIVE &&
       config.reduction_mode != QVQ_P32_REDUCTION_PARTIALS)) {
    set_last_error("QVQ P32 grouped block ABI requires block, generic, native or partials configuration");
    return -1;
  }
  const int total_n_tiles = size_n / kTileColumns;
  GroupedP32LaunchParams params{};
  if (!build_grouped_params(
          size_m, size_n, group_count, split_count_0, split_count_1,
          split_count_2, n_tile_end_0, n_tile_end_1, &params)) {
    return -1;
  }
  if (!validate_grouped_splits(params, size_k)) return -1;

  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const auto* input_half = reinterpret_cast<const half*>(input);
  const auto* trellis_words = reinterpret_cast<const uint32_t*>(trellis);
  const auto* levels_half = reinterpret_cast<const half*>(levels);
  const auto* bank_bytes = reinterpret_cast<const uint8_t*>(bank_ids);
  int status = 0;
  switch (config.threads) {
    case 64:
      status = launch_p32_grouped_block_threads<TransitionBits, 64>(
          input_half, trellis_words, levels_half, bank_bytes,
          reinterpret_cast<const uint8_t*>(bank_alt_ids), params, output,
          partial_output, size_m, size_k, total_n_tiles, config.stage_k_tiles,
          cuda_stream);
      break;
    case 128:
      status = launch_p32_grouped_block_threads<TransitionBits, 128>(
          input_half, trellis_words, levels_half, bank_bytes,
          reinterpret_cast<const uint8_t*>(bank_alt_ids), params, output,
          partial_output, size_m, size_k, total_n_tiles, config.stage_k_tiles,
          cuda_stream);
      break;
    case 256:
      status = launch_p32_grouped_block_threads<TransitionBits, 256>(
          input_half, trellis_words, levels_half, bank_bytes,
          reinterpret_cast<const uint8_t*>(bank_alt_ids), params, output,
          partial_output, size_m, size_k, total_n_tiles, config.stage_k_tiles,
          cuda_stream);
      break;
    default:
      set_last_error("QVQ P32 grouped threads must be one of 64, 128, or 256");
      return -1;
  }
  if (status != 0) {
    if (last_error[0] == '\0') set_last_error(cudaGetErrorString(static_cast<cudaError_t>(status)));
    return status;
  }
  status = finish_grouped_reduction(
      params, output, partial_output, size_m, total_n_tiles,
      config.reduction_mode, cuda_stream);
  if (status != 0) {
    set_last_error(cudaGetErrorString(static_cast<cudaError_t>(status)));
  }
  return status;
}

template <int TransitionBits>
int launch_p32_grouped_scalar(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_ids,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int group_count,
    int split_count_0,
    int split_count_1,
    int split_count_2,
    int n_tile_end_0,
    int n_tile_end_1,
    const qvq_p32_config& config,
    void* stream) {
  if (config.kernel_variant != QVQ_P32_VARIANT_SCALAR || config.static_n != 0 ||
      (config.reduction_mode != QVQ_P32_REDUCTION_NATIVE &&
       config.reduction_mode != QVQ_P32_REDUCTION_PARTIALS)) {
    set_last_error("QVQ P32 grouped ABI requires scalar, generic, native or partials configuration");
    return -1;
  }
  if (size_m < 1 || size_m > 4 || group_count < 2 || group_count > kMaxGroupedP32Segments) {
    set_last_error("QVQ P32 grouped ABI supports M in [1,4] and two or three groups");
    return -1;
  }
  const int total_n_tiles = size_n / kTileColumns;
  GroupedP32LaunchParams params{};
  if (!build_grouped_params(
          size_m, size_n, group_count, split_count_0, split_count_1,
          split_count_2, n_tile_end_0, n_tile_end_1, &params)) {
    return -1;
  }
  if (!validate_grouped_splits(params, size_k)) return -1;
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const auto* input_half = reinterpret_cast<const half*>(input);
  const auto* trellis_words = reinterpret_cast<const uint32_t*>(trellis);
  const auto* levels_half = reinterpret_cast<const half*>(levels);
  const auto* bank_bytes = reinterpret_cast<const uint8_t*>(bank_ids);
  int status = 0;
#define QVQ_GROUPED_ROWS(ROWS)                                                \
  status = launch_p32_grouped_scalar_threads<TransitionBits, ROWS>(                \
      input_half, trellis_words, levels_half, bank_bytes,                          \
      reinterpret_cast<const uint8_t*>(bank_alt_ids), params, output,              \
      partial_output, size_k, total_n_tiles, config.threads, config.stage_k_tiles, \
      cuda_stream)
  switch (size_m) {
    case 1: QVQ_GROUPED_ROWS(1); break;
    case 2: QVQ_GROUPED_ROWS(2); break;
    case 3: QVQ_GROUPED_ROWS(3); break;
    case 4: QVQ_GROUPED_ROWS(4); break;
  }
#undef QVQ_GROUPED_ROWS
  if (status != 0) {
    if (last_error[0] == '\0') set_last_error(cudaGetErrorString(static_cast<cudaError_t>(status)));
    return status;
  }
  status = finish_grouped_reduction(
      params, output, partial_output, size_m, total_n_tiles,
      config.reduction_mode, cuda_stream);
  if (status != 0) {
    set_last_error(cudaGetErrorString(static_cast<cudaError_t>(status)));
  }
  return status;
}

template <int TransitionBits, int Rows, int Threads>
const void* grouped_scalar_kernel_symbol(int stage_k_tiles) {
  switch (stage_k_tiles) {
    case 1:
      return reinterpret_cast<const void*>(
          p32_window_ampere_grouped_scalar_kernel<TransitionBits, Rows, Threads, 1>);
    case 2:
      return reinterpret_cast<const void*>(
          p32_window_ampere_grouped_scalar_kernel<TransitionBits, Rows, Threads, 2>);
    case 3:
      return reinterpret_cast<const void*>(
          p32_window_ampere_grouped_scalar_kernel<TransitionBits, Rows, Threads, 3>);
    case 4:
      return reinterpret_cast<const void*>(
          p32_window_ampere_grouped_scalar_kernel<TransitionBits, Rows, Threads, 4>);
    default:
      return nullptr;
  }
}

template <int TransitionBits, int Rows>
const void* grouped_scalar_kernel_symbol(int threads, int stage_k_tiles) {
  switch (threads) {
    case 64:
      return grouped_scalar_kernel_symbol<TransitionBits, Rows, 64>(stage_k_tiles);
    case 128:
      return grouped_scalar_kernel_symbol<TransitionBits, Rows, 128>(stage_k_tiles);
    case 256:
      return grouped_scalar_kernel_symbol<TransitionBits, Rows, 256>(stage_k_tiles);
    default:
      return nullptr;
  }
}

template <int TransitionBits>
const void* grouped_scalar_kernel_symbol(
    int size_m, int threads, int stage_k_tiles) {
  switch (size_m) {
    case 1:
      return grouped_scalar_kernel_symbol<TransitionBits, 1>(threads, stage_k_tiles);
    case 2:
      return grouped_scalar_kernel_symbol<TransitionBits, 2>(threads, stage_k_tiles);
    case 3:
      return grouped_scalar_kernel_symbol<TransitionBits, 3>(threads, stage_k_tiles);
    case 4:
      return grouped_scalar_kernel_symbol<TransitionBits, 4>(threads, stage_k_tiles);
    default:
      return nullptr;
  }
}

template <int TransitionBits, int Threads, bool FullRows, int ActiveRows>
const void* grouped_block_kernel_symbol(int stage_k_tiles) {
  switch (stage_k_tiles) {
    case 1:
      return reinterpret_cast<const void*>(
          p32_window_ampere_grouped_block_kernel<
              TransitionBits, Threads, 1, FullRows, ActiveRows>);
    case 2:
      return reinterpret_cast<const void*>(
          p32_window_ampere_grouped_block_kernel<
              TransitionBits, Threads, 2, FullRows, ActiveRows>);
    case 3:
      return reinterpret_cast<const void*>(
          p32_window_ampere_grouped_block_kernel<
              TransitionBits, Threads, 3, FullRows, ActiveRows>);
    case 4:
      return reinterpret_cast<const void*>(
          p32_window_ampere_grouped_block_kernel<
              TransitionBits, Threads, 4, FullRows, ActiveRows>);
    default:
      return nullptr;
  }
}

template <int TransitionBits, int Threads>
const void* grouped_block_kernel_symbol(int size_m, int stage_k_tiles) {
  if (size_m == 8) {
    return grouped_block_kernel_symbol<TransitionBits, Threads, false, 8>(
        stage_k_tiles);
  }
  if (size_m == 16) {
    return grouped_block_kernel_symbol<TransitionBits, Threads, true, 0>(
        stage_k_tiles);
  }
  return grouped_block_kernel_symbol<TransitionBits, Threads, false, 0>(
      stage_k_tiles);
}

template <int TransitionBits>
const void* grouped_block_kernel_symbol(
    int size_m, int threads, int stage_k_tiles) {
  switch (threads) {
    case 64:
      return grouped_block_kernel_symbol<TransitionBits, 64>(size_m, stage_k_tiles);
    case 128:
      return grouped_block_kernel_symbol<TransitionBits, 128>(size_m, stage_k_tiles);
    case 256:
      return grouped_block_kernel_symbol<TransitionBits, 256>(size_m, stage_k_tiles);
    default:
      return nullptr;
  }
}

const void* grouped_kernel_symbol(
    int transition_bits,
    int size_m,
    int kernel_variant,
    int threads,
    int stage_k_tiles) {
#define QVQ_GROUPED_KERNEL_SYMBOL(BITS)                                      \
  return kernel_variant == QVQ_P32_VARIANT_SCALAR                           \
      ? grouped_scalar_kernel_symbol<BITS>(size_m, threads, stage_k_tiles)   \
      : grouped_block_kernel_symbol<BITS>(size_m, threads, stage_k_tiles)
  switch (transition_bits) {
    case 4: QVQ_GROUPED_KERNEL_SYMBOL(4);
    case 5: QVQ_GROUPED_KERNEL_SYMBOL(5);
    case 6: QVQ_GROUPED_KERNEL_SYMBOL(6);
    case 7: QVQ_GROUPED_KERNEL_SYMBOL(7);
    default: return nullptr;
  }
#undef QVQ_GROUPED_KERNEL_SYMBOL
}

struct GroupedPlanStorage {
  GroupedP32LaunchParams params;
  int main_size_m;
  int main_size_k;
  int main_total_n_tiles;
  struct ReductionValues {
    int size_m;
    int group_n;
    int total_n;
    int group_n_offset;
    int split_count;
  } reductions[kMaxGroupedP32Segments];
};

static_assert(
    sizeof(GroupedPlanStorage) <=
        sizeof(((qvq_p32_launch_plan*)nullptr)->host_storage),
    "QVQ P32 launch-plan host storage is too small");

void set_device_arg(
    qvq_p32_launch_descriptor* launch, int index, const void* pointer) {
  launch->args[index] = {
      pointer,
      0,
      QVQ_P32_LAUNCH_ARG_DEVICE_POINTER,
  };
}

template <typename T>
void set_host_arg(
    qvq_p32_launch_descriptor* launch, int index, const T* value) {
  launch->args[index] = {
      value,
      static_cast<long long>(sizeof(T)),
      QVQ_P32_LAUNCH_ARG_HOST_VALUE,
  };
}

template <int TransitionBits>
int launch_p32(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    void* stream) {
  const int n_tiles = size_n / kTileColumns;
  const bool use_small_m_scalar =
      (size_m <= 4 && size_k <= 6144) ||
      (size_m <= 4 && size_k == 17408 && size_n == 5120);
  const bool use_three_tile_scalar_stage = size_m == 2 && size_k <= 6144;
  const bool use_four_tile_scalar_stage =
      (size_m == 4 && size_k <= 6144) ||
      (size_m <= 4 && size_k == 17408 && size_n == 5120);
  const int tiles_per_block = use_small_m_scalar ? kM1TilesPerBlock : kTilesPerBlock;
  const dim3 grid(
      static_cast<unsigned>((n_tiles + tiles_per_block - 1) / tiles_per_block),
      1,
      static_cast<unsigned>(split_count));
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const auto* input_half = reinterpret_cast<const half*>(input);
  const auto* trellis_words = reinterpret_cast<const uint32_t*>(trellis);
  const auto* levels_half = reinterpret_cast<const half*>(levels);
  const auto* bank_bytes = reinterpret_cast<const uint8_t*>(bank_ids);
  const auto* bank_alt_byte = reinterpret_cast<const uint8_t*>(bank_alt_id);

  if (size_m == 1 && use_small_m_scalar &&
      (size_n == 512 || size_n == 2048 || size_n == 8192 || size_n == 12288 ||
       size_n == 1024 || size_n == 10240 || size_n == 17408 ||
       (size_n == 5120 && size_k == 6144)) &&
      launch_static_n_scalar_kernel<TransitionBits, 1>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
          size_k, size_n, split_count, bank_alt_byte, grid, cuda_stream)) {
  } else if (size_m == 1 && use_small_m_scalar && use_four_tile_scalar_stage) {
    p32_window_ampere_m1_kernel<TransitionBits, 1, kM1Threads, kM1TilesPerBlock,
                                kScalarLongStageKTiles>
        <<<grid, kM1Threads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 1 && use_small_m_scalar) {
    p32_window_ampere_m1_kernel<TransitionBits, 1><<<grid, kM1Threads, 0, cuda_stream>>>(
        input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
        size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 2 && use_small_m_scalar && use_four_tile_scalar_stage) {
    p32_window_ampere_m1_kernel<TransitionBits, 2, kM1Threads, kM1TilesPerBlock,
                                kScalarLongStageKTiles>
        <<<grid, kM1Threads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 2 && use_small_m_scalar && use_three_tile_scalar_stage) {
    p32_window_ampere_m1_kernel<TransitionBits, 2, kM1Threads, kM1TilesPerBlock,
                                kScalarTripleStageKTiles>
        <<<grid, kM1Threads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 2 && use_small_m_scalar) {
    p32_window_ampere_m1_kernel<TransitionBits, 2><<<grid, kM1Threads, 0, cuda_stream>>>(
        input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
        size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 3 && use_small_m_scalar) {
    p32_window_ampere_m1_kernel<TransitionBits, 3><<<grid, kM1Threads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
        size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 4 && use_small_m_scalar && use_four_tile_scalar_stage) {
    p32_window_ampere_m1_kernel<TransitionBits, 4, kM1Threads, kM1TilesPerBlock,
                                kScalarLongStageKTiles>
        <<<grid, kM1Threads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 4 && use_small_m_scalar) {
    p32_window_ampere_m1_kernel<TransitionBits, 4><<<grid, kM1Threads, 0, cuda_stream>>>(
        input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
        size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 8 && size_n == 12288) {
    p32_window_ampere_kernel<TransitionBits, false, 8, 12288>
        <<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 8 && size_n == 5120) {
    p32_window_ampere_kernel<TransitionBits, false, 8, 5120>
        <<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 8 && size_n == 10240) {
    p32_window_ampere_kernel<TransitionBits, false, 8, 10240>
        <<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 8 && size_n == 6144) {
    p32_window_ampere_kernel<TransitionBits, false, 8, 6144>
        <<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 8 && size_n == 1024) {
    p32_window_ampere_kernel<TransitionBits, false, 8, 1024>
        <<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 8) {
    p32_window_ampere_kernel<TransitionBits, false, 8>
        <<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 16 && size_n == 12288) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 12288>
        <<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 16 && size_n == 5120) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 5120>
        <<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 16 && size_n == 10240) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 10240>
        <<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 16 && size_n == 6144) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 6144>
        <<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 16 && size_n == 1024) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 1024>
        <<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_byte);
  } else if (size_m == 16) {
    p32_window_ampere_kernel<TransitionBits, true><<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
        size_m, size_k, size_n, split_count, bank_alt_byte);
  } else {
    p32_window_ampere_kernel<TransitionBits, false><<<grid, kThreads, 0, cuda_stream>>>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
        size_m, size_k, size_n, split_count, bank_alt_byte);
  }
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    set_last_error(cudaGetErrorString(error));
    return static_cast<int>(error);
  }
  if (split_count > 1) {
    launch_split_reduction(
        partial_output, output, size_m, size_k, size_n, split_count,
        cuda_stream);
    const cudaError_t reduction_error = cudaGetLastError();
    if (reduction_error != cudaSuccess) {
      set_last_error(cudaGetErrorString(reduction_error));
      return static_cast<int>(reduction_error);
    }
  }
  return 0;
}

template <int TransitionBits, int Threads, int StageKTiles>
int launch_p32_config_variant(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    int kernel_variant,
    int static_n,
    int reduction_mode,
    void* stream) {
  const int n_tiles = size_n / kTileColumns;
  const int tiles_per_block = kernel_variant == QVQ_P32_VARIANT_SCALAR
      ? 4 * (Threads / 32)
      : Threads / 32;
  const dim3 grid(
      static_cast<unsigned>((n_tiles + tiles_per_block - 1) / tiles_per_block),
      1,
      static_cast<unsigned>(split_count));
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const auto* input_half = reinterpret_cast<const half*>(input);
  const auto* trellis_words = reinterpret_cast<const uint32_t*>(trellis);
  const auto* levels_half = reinterpret_cast<const half*>(levels);
  const auto* bank_bytes = reinterpret_cast<const uint8_t*>(bank_ids);
  const auto* bank_alt_byte = reinterpret_cast<const uint8_t*>(bank_alt_id);

  if (kernel_variant == QVQ_P32_VARIANT_SCALAR) {
    bool launched_static = false;
    if constexpr (Threads == kM1Threads) {
      if (static_n) {
        if constexpr (StageKTiles == kScalarTripleStageKTiles) {
          if (size_m == 1 && size_k == 5120 && size_n == 12288 &&
              split_count == 40 &&
              launch_fixed_n_scalar_kernel<
                  TransitionBits, 1, Threads, 4 * (Threads / 32),
                  kScalarTripleStageKTiles, 12288, 40, 5120>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
          if (!launched_static && size_m == 1 && size_k == 17408 &&
              size_n == 5120 && split_count == 128 &&
              launch_fixed_n_scalar_kernel<
                  TransitionBits, 1, Threads, 4 * (Threads / 32),
                  kScalarTripleStageKTiles, 5120, 128>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
          if (!launched_static && size_m == 2 && size_k == 5120 &&
              size_n == 12288 &&
              ((TransitionBits <= 5 && split_count == 64) ||
               (TransitionBits >= 6 && split_count == 40)) &&
              launch_fixed_n_scalar_kernel<
                  TransitionBits, 2, Threads, 4 * (Threads / 32),
                  kScalarTripleStageKTiles, 12288,
                  TransitionBits <= 5 ? 64 : 40>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
          if (!launched_static && size_m == 2 && size_k == 5120 &&
              size_n == 10240 && split_count == 40 &&
              launch_fixed_n_scalar_kernel<
                  TransitionBits, 2, Threads, 4 * (Threads / 32),
                  kScalarTripleStageKTiles, 10240, 40>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
          if (!launched_static && size_m == 2 && size_k == 5120 &&
              size_n == 17408 && split_count == 40 &&
              launch_fixed_n_scalar_kernel<
                  TransitionBits, 2, Threads, 4 * (Threads / 32),
                  kScalarTripleStageKTiles, 17408, 40>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
          if (!launched_static && size_m == 1 && size_k == 5120 &&
              size_n == 12288 &&
              launch_static_n_scalar_kernel<
                  TransitionBits, 1, Threads, 4 * (Threads / 32),
                  kScalarTripleStageKTiles>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
        }
        if constexpr (StageKTiles == kStageKTiles) {
          if (size_m == 1 && size_k == 2048 && size_n == 512 &&
              split_count == 32 &&
              launch_fixed_n_scalar_kernel<
                  TransitionBits, 1, Threads, 4 * (Threads / 32),
                  kStageKTiles, 512, 32, 2048>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
          if (!launched_static && size_m == 1 && size_k == 2048 &&
              size_n == 2048 && split_count == 32 &&
              launch_fixed_n_scalar_kernel<
                  TransitionBits, 1, Threads, 4 * (Threads / 32),
                  kStageKTiles, 2048, 32, 2048>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
          if (!launched_static && size_m == 1 && size_k == 2048 &&
              size_n == 8192 && split_count == 32 &&
              launch_fixed_n_scalar_kernel<
                  TransitionBits, 1, Threads, 4 * (Threads / 32),
                  kStageKTiles, 8192, 32, 2048>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
          if (size_m == 1 && size_k == 5120 && size_n == 1024 &&
              split_count == 56 &&
              launch_fixed_n_scalar_kernel<
                  TransitionBits, 1, Threads, 4 * (Threads / 32),
                  kStageKTiles, 1024, 56>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
          if (!launched_static && size_m == 1 && size_k == 5120 &&
              size_n == 17408 && split_count == 40 &&
              launch_fixed_n_scalar_kernel<
                  TransitionBits, 1, Threads, 4 * (Threads / 32),
                  kStageKTiles, 17408, 40>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
          if (!launched_static && size_m == 1 && launch_static_n_scalar_kernel<
                                  TransitionBits, 1, Threads, 4 * (Threads / 32), StageKTiles>(
                                  input_half, trellis_words, levels_half, bank_bytes,
                                  partial_output, output, size_k, size_n, split_count,
                                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
        }
        if constexpr (
            StageKTiles == kScalarTripleStageKTiles ||
            StageKTiles == kScalarLongStageKTiles) {
          if (!launched_static && size_m == 2 && launch_static_n_scalar_kernel<
                                  TransitionBits, 2, Threads, 4 * (Threads / 32), StageKTiles>(
                                  input_half, trellis_words, levels_half, bank_bytes,
                                  partial_output, output, size_k, size_n, split_count,
                                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
        }
        if constexpr (
            (TransitionBits == 4 && StageKTiles == kScalarLongStageKTiles) ||
            (TransitionBits == 6 && StageKTiles == kScalarTripleStageKTiles) ||
            ((TransitionBits == 5 || TransitionBits == 7) &&
             StageKTiles == kStageKTiles)) {
          if (size_m == 4 && size_k == 5120 && size_n == 12288 &&
              split_count == 40 && launch_fixed_n_scalar_kernel<
                  TransitionBits, 4, Threads, 4 * (Threads / 32),
                  StageKTiles, 12288, 40>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
          if (!launched_static && size_m == 4 && size_k == 5120 &&
              size_n == 10240 && split_count == 40 &&
              launch_fixed_n_scalar_kernel<
                  TransitionBits, 4, Threads, 4 * (Threads / 32),
                  StageKTiles, 10240, 40>(
                  input_half, trellis_words, levels_half, bank_bytes,
                  partial_output, output, size_k, size_n, split_count,
                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
          if (!launched_static && size_m == 4 && launch_static_n_scalar_kernel<
                                  TransitionBits, 4, Threads, 4 * (Threads / 32), StageKTiles>(
                                  input_half, trellis_words, levels_half, bank_bytes,
                                  partial_output, output, size_k, size_n, split_count,
                                  bank_alt_byte, grid, cuda_stream)) {
            launched_static = true;
          }
        }
      }
    }
    if (!launched_static) {
      // Llama 3.2 1B F6's standalone down projection is the one production
      // scalar route that is both long-K and outside the grouped path. Its
      // immutable 256-entry level table is small enough to stage once per
      // CTA, avoiding repeated constant/global-cache lookups in every K tile.
      const bool use_llama_f6_shared_levels =
          TransitionBits == 6 && size_k == 8192 && size_n == 2048;
      switch (size_m) {
        case 1:
          if (use_llama_f6_shared_levels) {
            p32_window_ampere_m1_kernel<
                TransitionBits, 1, Threads, 4 * (Threads / 32), StageKTiles,
                0, 0, true><<<grid, Threads, 0, cuda_stream>>>(
                input_half, trellis_words, levels_half, bank_bytes,
                partial_output, output, size_k, size_n, split_count,
                bank_alt_byte);
          } else {
            p32_window_ampere_m1_kernel<
                TransitionBits, 1, Threads, 4 * (Threads / 32), StageKTiles>
                <<<grid, Threads, 0, cuda_stream>>>(
                    input_half, trellis_words, levels_half, bank_bytes,
                    partial_output, output, size_k, size_n, split_count,
                    bank_alt_byte);
          }
          break;
        case 2:
          if (use_llama_f6_shared_levels) {
            p32_window_ampere_m1_kernel<
                TransitionBits, 2, Threads, 4 * (Threads / 32), StageKTiles,
                0, 0, true><<<grid, Threads, 0, cuda_stream>>>(
                input_half, trellis_words, levels_half, bank_bytes,
                partial_output, output, size_k, size_n, split_count,
                bank_alt_byte);
          } else {
            p32_window_ampere_m1_kernel<
                TransitionBits, 2, Threads, 4 * (Threads / 32), StageKTiles>
                <<<grid, Threads, 0, cuda_stream>>>(
                    input_half, trellis_words, levels_half, bank_bytes,
                    partial_output, output, size_k, size_n, split_count,
                    bank_alt_byte);
          }
          break;
        case 3:
          if (use_llama_f6_shared_levels) {
            p32_window_ampere_m1_kernel<
                TransitionBits, 3, Threads, 4 * (Threads / 32), StageKTiles,
                0, 0, true><<<grid, Threads, 0, cuda_stream>>>(
                input_half, trellis_words, levels_half, bank_bytes,
                partial_output, output, size_k, size_n, split_count,
                bank_alt_byte);
          } else {
            p32_window_ampere_m1_kernel<
                TransitionBits, 3, Threads, 4 * (Threads / 32), StageKTiles>
                <<<grid, Threads, 0, cuda_stream>>>(
                    input_half, trellis_words, levels_half, bank_bytes,
                    partial_output, output, size_k, size_n, split_count,
                    bank_alt_byte);
          }
          break;
        case 4:
          if (use_llama_f6_shared_levels) {
            p32_window_ampere_m1_kernel<
                TransitionBits, 4, Threads, 4 * (Threads / 32), StageKTiles,
                0, 0, true><<<grid, Threads, 0, cuda_stream>>>(
                input_half, trellis_words, levels_half, bank_bytes,
                partial_output, output, size_k, size_n, split_count,
                bank_alt_byte);
          } else {
            p32_window_ampere_m1_kernel<
                TransitionBits, 4, Threads, 4 * (Threads / 32), StageKTiles>
                <<<grid, Threads, 0, cuda_stream>>>(
                    input_half, trellis_words, levels_half, bank_bytes,
                    partial_output, output, size_k, size_n, split_count,
                    bank_alt_byte);
          }
          break;
        default:
          set_last_error("QVQ P32 scalar variant supports M in [1, 4]");
          return -1;
      }
    }
  } else if (kernel_variant == QVQ_P32_VARIANT_BLOCK) {
    if (static_n && (size_m == 8 || size_m == 16) &&
        launch_static_n_block_kernel<TransitionBits, Threads, StageKTiles>(
            input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
            size_m, size_k, size_n, split_count, bank_alt_byte, grid, cuda_stream)) {
      // The fixed-N launch above is the complete block-variant dispatch.
    } else {
      switch (size_m) {
        case 8:
          p32_window_ampere_kernel<TransitionBits, false, 8, 0, Threads, Threads / 32, StageKTiles>
              <<<grid, Threads, 0, cuda_stream>>>(
                  input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
                  size_m, size_k, size_n, split_count, bank_alt_byte);
          break;
        case 16:
          p32_window_ampere_kernel<TransitionBits, true, 0, 0, Threads, Threads / 32, StageKTiles>
              <<<grid, Threads, 0, cuda_stream>>>(
                  input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
                  size_m, size_k, size_n, split_count, bank_alt_byte);
          break;
        default:
          p32_window_ampere_kernel<TransitionBits, false, 0, 0, Threads, Threads / 32, StageKTiles>
              <<<grid, Threads, 0, cuda_stream>>>(
                  input_half, trellis_words, levels_half, bank_bytes, partial_output, output,
                  size_m, size_k, size_n, split_count, bank_alt_byte);
          break;
      }
    }
  } else {
    set_last_error("QVQ P32 received an unknown kernel variant");
    return -1;
  }

  if (split_count > 1 &&
      reduction_mode == QVQ_P32_REDUCTION_NATIVE) {
    launch_split_reduction(
        partial_output, output, size_m, size_k, size_n, split_count,
        cuda_stream);
  }
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    set_last_error(cudaGetErrorString(error));
    return static_cast<int>(error);
  }
  return 0;
}

template <int TransitionBits>
int launch_p32_config(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    const qvq_p32_config& config,
    void* stream) {
  int status = -1;
#define QVQ_DISPATCH_STAGE(THREAD_COUNT)                                      \
  switch (config.stage_k_tiles) {                                                  \
    case 1:                                                                        \
      status = launch_p32_config_variant<TransitionBits, THREAD_COUNT, 1>(         \
          input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,   \
          size_m, size_k, size_n, config.split_count, config.kernel_variant,       \
          config.static_n, config.reduction_mode, stream);                         \
      break;                                                                       \
    case 2:                                                                        \
      status = launch_p32_config_variant<TransitionBits, THREAD_COUNT, 2>(         \
          input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,   \
          size_m, size_k, size_n, config.split_count, config.kernel_variant,       \
          config.static_n, config.reduction_mode, stream);                         \
      break;                                                                       \
    case 3:                                                                        \
      status = launch_p32_config_variant<TransitionBits, THREAD_COUNT, 3>(         \
          input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,   \
          size_m, size_k, size_n, config.split_count, config.kernel_variant,       \
          config.static_n, config.reduction_mode, stream);                         \
      break;                                                                       \
    case 4:                                                                        \
      status = launch_p32_config_variant<TransitionBits, THREAD_COUNT, 4>(         \
          input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,   \
          size_m, size_k, size_n, config.split_count, config.kernel_variant,       \
          config.static_n, config.reduction_mode, stream);                         \
      break;                                                                       \
    default:                                                                       \
      set_last_error("QVQ P32 stage_k_tiles must be in [1, 4]");                  \
      return -1;                                                                   \
  }
#define QVQ_DISPATCH_THREADS(THREAD_COUNT)                                     \
  QVQ_DISPATCH_STAGE(THREAD_COUNT)

  switch (config.threads) {
    case 64:
      QVQ_DISPATCH_THREADS(64);
      break;
    case 128:
      QVQ_DISPATCH_THREADS(128);
      break;
    case 256:
      QVQ_DISPATCH_THREADS(256);
      break;
    default:
      set_last_error("QVQ P32 threads must be one of 64, 128, or 256");
      return -1;
  }
#undef QVQ_DISPATCH_THREADS
#undef QVQ_DISPATCH_STAGE
  return status;
}

// The Ampere kernel's shared-memory tile is fixed at 16 rows. Keep that
// hardware specialization, but batch large-M projections inside one FFI call
// so XLA/PJRT does not pay a launch boundary for every row chunk. Both large-M
// producers use global M*N split-plane strides, retaining canonical [S,M,N]
// partials across row chunks. The public dispatcher owns optional reduction.
template <int TransitionBits, int Threads, int StageKTiles, int StaticN,
          int StaticK = 0>
int launch_p32_large_m_grid(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    const uint8_t* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    cudaStream_t stream) {
  constexpr int tiles_per_block = Threads / 32;
  const int n_tiles = size_n / kTileColumns;
  const dim3 grid(
      static_cast<unsigned>((n_tiles + tiles_per_block - 1) / tiles_per_block),
      static_cast<unsigned>((size_m + kRows - 1) / kRows),
      static_cast<unsigned>(split_count));
  p32_window_ampere_large_m_kernel<
      TransitionBits, Threads, StageKTiles, StaticN, StaticK>
      <<<grid, Threads, 0, stream>>>(
      input, trellis, levels, bank_ids, partial_output, output, size_m, size_k,
      size_n, split_count, bank_alt_id);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    set_last_error(cudaGetErrorString(error));
    return static_cast<int>(error);
  }
  return 0;
}

template <int TransitionBits, int Threads, int StageKTiles>
int launch_p32_large_m_grid_dispatch(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    const uint8_t* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    bool static_n,
    cudaStream_t stream) {
  if (static_n && size_k == 5120) {
    switch (size_n) {
#define QVQ_LARGE_M_STATIC_N(N) \
      case N: \
        return launch_p32_large_m_grid< \
            TransitionBits, Threads, StageKTiles, N, 5120>( \
            input, trellis, levels, bank_ids, bank_alt_id, output, \
            partial_output, size_m, size_k, size_n, split_count, stream)
      QVQ_LARGE_M_STATIC_N(512);
      QVQ_LARGE_M_STATIC_N(2048);
      QVQ_LARGE_M_STATIC_N(8192);
      QVQ_LARGE_M_STATIC_N(1024);
      QVQ_LARGE_M_STATIC_N(5120);
      QVQ_LARGE_M_STATIC_N(6144);
      QVQ_LARGE_M_STATIC_N(10240);
      QVQ_LARGE_M_STATIC_N(12288);
      QVQ_LARGE_M_STATIC_N(17408);
#undef QVQ_LARGE_M_STATIC_N
      default:
        set_last_error("QVQ P32 large-M static_n does not support this N");
        return -1;
    }
  }
  if (static_n) {
    switch (size_n) {
#define QVQ_LARGE_M_STATIC_N(N) \
      case N: \
        return launch_p32_large_m_grid< \
            TransitionBits, Threads, StageKTiles, N, 0>( \
            input, trellis, levels, bank_ids, bank_alt_id, output, \
            partial_output, size_m, size_k, size_n, split_count, stream)
      QVQ_LARGE_M_STATIC_N(512);
      QVQ_LARGE_M_STATIC_N(2048);
      QVQ_LARGE_M_STATIC_N(8192);
      QVQ_LARGE_M_STATIC_N(1024);
      QVQ_LARGE_M_STATIC_N(5120);
      QVQ_LARGE_M_STATIC_N(6144);
      QVQ_LARGE_M_STATIC_N(10240);
      QVQ_LARGE_M_STATIC_N(12288);
      QVQ_LARGE_M_STATIC_N(17408);
#undef QVQ_LARGE_M_STATIC_N
      default:
        set_last_error("QVQ P32 large-M static_n does not support this N");
        return -1;
    }
  }
  return launch_p32_large_m_grid<
      TransitionBits, Threads, StageKTiles, 0>(
      input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,
      size_m, size_k, size_n, split_count, stream);
}

template <int TransitionBits, int StageKTiles, int StaticN, int StaticK = 0,
          int RowGroups = 2>
int launch_p32_large_m2_grid(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    const uint8_t* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    cudaStream_t stream) {
  constexpr int tiles_per_block = 4;
  const int n_tiles = size_n / kTileColumns;
  const dim3 grid(
      static_cast<unsigned>((n_tiles + tiles_per_block - 1) / tiles_per_block),
      static_cast<unsigned>(size_m / (RowGroups * kRows)),
      static_cast<unsigned>(split_count));
  constexpr bool kDynamicInputTile = RowGroups == 16 && StageKTiles >= 3;
  constexpr int kDynamicInputBytes =
      kDynamicInputTile
          ? 2 * RowGroups * kRows * StageKTiles * kTileRows * sizeof(half)
          : 0;
  if constexpr (kDynamicInputTile) {
    const cudaError_t attribute_error = cudaFuncSetAttribute(
        p32_window_ampere_large_m2_kernel<
            TransitionBits, StaticN, StaticK, StageKTiles, RowGroups>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicInputBytes);
    if (attribute_error != cudaSuccess) {
      set_last_error(cudaGetErrorString(attribute_error));
      return static_cast<int>(attribute_error);
    }
  }
  p32_window_ampere_large_m2_kernel<
      TransitionBits, StaticN, StaticK, StageKTiles, RowGroups>
      <<<grid, 128, kDynamicInputBytes, stream>>>(
      input, trellis, levels, bank_ids, partial_output, output, size_m, size_k,
      size_n, split_count, bank_alt_id);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    set_last_error(cudaGetErrorString(error));
    return static_cast<int>(error);
  }
  return 0;
}

template <int TransitionBits, int StageKTiles, int RowGroups = 2>
int launch_p32_large_m2_grid_dispatch(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    const uint8_t* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    bool static_n,
    cudaStream_t stream) {
  if (static_n && size_k == 5120) {
    switch (size_n) {
#define QVQ_LARGE_M2_STATIC_N(N) \
      case N: \
        return launch_p32_large_m2_grid< \
            TransitionBits, StageKTiles, N, 5120, RowGroups>( \
            input, trellis, levels, bank_ids, bank_alt_id, output, \
            partial_output, size_m, size_k, size_n, split_count, stream)
      QVQ_LARGE_M2_STATIC_N(512);
      QVQ_LARGE_M2_STATIC_N(2048);
      QVQ_LARGE_M2_STATIC_N(8192);
      QVQ_LARGE_M2_STATIC_N(1024);
      QVQ_LARGE_M2_STATIC_N(5120);
      QVQ_LARGE_M2_STATIC_N(6144);
      QVQ_LARGE_M2_STATIC_N(10240);
      QVQ_LARGE_M2_STATIC_N(12288);
      QVQ_LARGE_M2_STATIC_N(17408);
#undef QVQ_LARGE_M2_STATIC_N
      default:
        set_last_error("QVQ P32 large-M2 static_n does not support this N");
        return -1;
    }
  }
  if (static_n) {
    switch (size_n) {
#define QVQ_LARGE_M2_STATIC_N(N) \
      case N: \
        return launch_p32_large_m2_grid< \
            TransitionBits, StageKTiles, N, 0, RowGroups>( \
            input, trellis, levels, bank_ids, bank_alt_id, output, \
            partial_output, size_m, size_k, size_n, split_count, stream)
      QVQ_LARGE_M2_STATIC_N(512);
      QVQ_LARGE_M2_STATIC_N(2048);
      QVQ_LARGE_M2_STATIC_N(8192);
      QVQ_LARGE_M2_STATIC_N(1024);
      QVQ_LARGE_M2_STATIC_N(5120);
      QVQ_LARGE_M2_STATIC_N(6144);
      QVQ_LARGE_M2_STATIC_N(10240);
      QVQ_LARGE_M2_STATIC_N(12288);
      QVQ_LARGE_M2_STATIC_N(17408);
#undef QVQ_LARGE_M2_STATIC_N
      default:
        set_last_error("QVQ P32 large-M2 static_n does not support this N");
        return -1;
    }
  }
  return launch_p32_large_m2_grid<
      TransitionBits, StageKTiles, 0, 0, RowGroups>(
      input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,
      size_m, size_k, size_n, split_count, stream);
}

template <int TransitionBits>
int launch_p32_large_m(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    const qvq_p32_config& config,
    int row_groups,
    void* stream) {
  if (config.kernel_variant != QVQ_P32_VARIANT_BLOCK) {
    set_last_error("QVQ P32 large-M batching requires block variant");
    return -1;
  }
  const auto* input_half = reinterpret_cast<const half*>(input);
  const auto* trellis_words = reinterpret_cast<const uint32_t*>(trellis);
  const auto* levels_half = reinterpret_cast<const half*>(levels);
  const auto* bank_bytes = reinterpret_cast<const uint8_t*>(bank_ids);
  const auto* bank_alt_byte = reinterpret_cast<const uint8_t*>(bank_alt_id);
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  // Every supported model projection has a fixed N tile count.  Large-M
  // launches use that specialization by default; callers can still pass
  // static_n=0 for nonstandard N values, which take the generic path.
  const bool automatic_policy = row_groups == QVQ_P32_ROW_GROUPS_AUTO;
  const bool use_static_n = config.static_n != 0 ||
      (automatic_policy &&
       (size_n == 512 || size_n == 1024 || size_n == 2048 ||
        size_n == 5120 || size_n == 6144 || size_n == 8192 ||
        size_n == 10240 || size_n == 12288 || size_n == 17408));
  const bool qwen38_27b_shape =
      (size_k == 5120 &&
       (size_n == 1024 || size_n == 6144 || size_n == 10240 ||
        size_n == 12288 || size_n == 17408)) ||
      ((size_k == 6144 || size_k == 17408) && size_n == 5120);
  int status = -1;
  if (config.threads == 128 && size_m % (2 * kRows) == 0 &&
      row_groups != 1) {
#define QVQ_LARGE_M2_STAGE(ROW_GROUPS) \
    switch (config.stage_k_tiles) { \
      case 1: \
        status = launch_p32_large_m2_grid_dispatch<TransitionBits, 1, ROW_GROUPS>( \
            input_half, trellis_words, levels_half, bank_bytes, bank_alt_byte, \
            output, partial_output, size_m, size_k, size_n, config.split_count, \
            use_static_n, cuda_stream); \
        break; \
      case 2: \
        status = launch_p32_large_m2_grid_dispatch<TransitionBits, 2, ROW_GROUPS>( \
            input_half, trellis_words, levels_half, bank_bytes, bank_alt_byte, \
            output, partial_output, size_m, size_k, size_n, config.split_count, \
            use_static_n, cuda_stream); \
        break; \
      case 3: \
        status = launch_p32_large_m2_grid_dispatch<TransitionBits, 3, ROW_GROUPS>( \
            input_half, trellis_words, levels_half, bank_bytes, bank_alt_byte, \
            output, partial_output, size_m, size_k, size_n, config.split_count, \
            use_static_n, cuda_stream); \
        break; \
      case 4: \
        status = launch_p32_large_m2_grid_dispatch<TransitionBits, 4, ROW_GROUPS>( \
            input_half, trellis_words, levels_half, bank_bytes, bank_alt_byte, \
            output, partial_output, size_m, size_k, size_n, config.split_count, \
            use_static_n, cuda_stream); \
        break; \
      default: \
        set_last_error("QVQ P32 large-M2 stage_k_tiles must be in [1, 4]"); \
        return -1; \
    }
#define QVQ_LARGE_M2_STAGE2(ROW_GROUPS) \
    status = launch_p32_large_m2_grid_dispatch<TransitionBits, 2, ROW_GROUPS>( \
        input_half, trellis_words, levels_half, bank_bytes, bank_alt_byte, \
        output, partial_output, size_m, size_k, size_n, config.split_count, \
        use_static_n, cuda_stream)
#define QVQ_LARGE_M2_STAGE3(ROW_GROUPS) \
    status = launch_p32_large_m2_grid_dispatch<TransitionBits, 3, ROW_GROUPS>( \
        input_half, trellis_words, levels_half, bank_bytes, bank_alt_byte, \
        output, partial_output, size_m, size_k, size_n, config.split_count, \
        use_static_n, cuda_stream)
#define QVQ_LARGE_M2_STAGE4(ROW_GROUPS) \
    status = launch_p32_large_m2_grid_dispatch<TransitionBits, 4, ROW_GROUPS>( \
        input_half, trellis_words, levels_half, bank_bytes, bank_alt_byte, \
        output, partial_output, size_m, size_k, size_n, config.split_count, \
        use_static_n, cuda_stream)
    if (row_groups == 16) {
      const bool supported_stage2 =
          config.stage_k_tiles == 2 &&
          ((size_m == 1024 &&
            (size_n == 5120 || size_n == 10240 || size_n == 17408)) ||
           (size_m == 2048 && size_n == 5120 && TransitionBits >= 5) ||
           (size_m >= 4096 && size_n != 1024));
      const bool supported_stage3 =
          config.stage_k_tiles == 3 && qwen38_27b_shape &&
          ((size_m == 1024 &&
            ((size_n == 1024 && TransitionBits >= 5) ||
             size_n == 5120 || size_n == 10240 || size_n == 12288 ||
             size_n == 17408)) ||
           (size_m >= 2048 && size_n != 1024));
      const bool supported_stage4 =
          config.stage_k_tiles == 4 && qwen38_27b_shape &&
          TransitionBits == 4 && size_m == 4096 && size_n != 1024;
      if (size_m % (16 * kRows) != 0 ||
          (!supported_stage2 && !supported_stage3 && !supported_stage4)) {
        set_last_error(
            "QVQ P32 row_groups=16 requires aligned M and a supported Qwen stage/shape");
        return -1;
      }
      if (supported_stage2) {
        QVQ_LARGE_M2_STAGE2(16);
      } else if (supported_stage3) {
        QVQ_LARGE_M2_STAGE3(16);
      } else {
        QVQ_LARGE_M2_STAGE4(16);
      }
    } else if (row_groups == 8) {
      const bool supported_n1024_stage =
          size_n == 1024 &&
          (config.stage_k_tiles == 3 ||
           (config.stage_k_tiles != 4 && size_m >= 2048));
      const bool supported_other_stage =
          size_n != 1024 && config.stage_k_tiles != 4;
      if ((!supported_n1024_stage && !supported_other_stage) ||
          size_m % (8 * kRows) != 0) {
        set_last_error(
            "QVQ P32 row_groups=8 requires aligned M and N=1024/stage=3 or N=1024/stage<4/M>=2048 or N!=1024/stage<4");
        return -1;
      }
      QVQ_LARGE_M2_STAGE(8)
    } else if (row_groups == 4) {
      if (size_m % (4 * kRows) != 0) {
        set_last_error("QVQ P32 row_groups=4 requires M divisible by 64");
        return -1;
      }
      QVQ_LARGE_M2_STAGE(4)
    } else if (row_groups == 2) {
      QVQ_LARGE_M2_STAGE(2)
    } else if (row_groups != QVQ_P32_ROW_GROUPS_AUTO) {
      set_last_error("QVQ P32 multi-row groups require 128 threads and aligned M");
      return -1;
    } else if (config.stage_k_tiles == 2 &&
               size_m % (16 * kRows) == 0 &&
               ((size_m == 1024 &&
                 (size_n == 5120 || size_n == 10240 || size_n == 17408)) ||
                (size_m == 2048 && size_n == 5120 && TransitionBits >= 5) ||
                (size_m >= 4096 && size_n != 1024))) {
      QVQ_LARGE_M2_STAGE2(16);
    } else if (config.stage_k_tiles == 3 &&
               qwen38_27b_shape &&
               size_m % (16 * kRows) == 0 &&
               ((size_m == 1024 &&
                 ((size_n == 1024 && TransitionBits >= 5) ||
                  size_n == 5120 || size_n == 10240 || size_n == 12288 ||
                  size_n == 17408)) ||
                (size_m >= 2048 && size_n != 1024))) {
      QVQ_LARGE_M2_STAGE3(16);
    } else if (config.stage_k_tiles == 4 &&
               qwen38_27b_shape &&
               TransitionBits == 4 && size_m == 4096 && size_n != 1024) {
      QVQ_LARGE_M2_STAGE4(16);
    } else if (((size_n == 1024 &&
                 (config.stage_k_tiles == 3 ||
                  (config.stage_k_tiles != 4 && size_m >= 2048))) ||
                (size_n != 1024 && config.stage_k_tiles != 4) ||
                (TransitionBits >= 5 && size_n != 1024 && size_m >= 2048 &&
                 !(TransitionBits == 7 && size_m == 2048 && size_n == 6144) &&
                 config.stage_k_tiles == 4)) &&
               size_m % (8 * kRows) == 0) {
      QVQ_LARGE_M2_STAGE(8)
    } else if (size_m % (4 * kRows) == 0) {
      QVQ_LARGE_M2_STAGE(4)
    } else {
      QVQ_LARGE_M2_STAGE(2)
    }
    return status;
#undef QVQ_LARGE_M2_STAGE4
#undef QVQ_LARGE_M2_STAGE3
#undef QVQ_LARGE_M2_STAGE2
#undef QVQ_LARGE_M2_STAGE
  }
  if (row_groups != QVQ_P32_ROW_GROUPS_AUTO && row_groups != 1) {
    set_last_error("QVQ P32 multi-row groups require 128 threads and aligned M");
    return -1;
  }
#define QVQ_LARGE_M_STAGE(THREADS) \
  switch (config.stage_k_tiles) { \
    case 1: \
      status = launch_p32_large_m_grid_dispatch<TransitionBits, THREADS, 1>( \
          input_half, trellis_words, levels_half, bank_bytes, bank_alt_byte, \
          output, partial_output, size_m, size_k, size_n, config.split_count, \
          use_static_n, cuda_stream); \
      break; \
    case 2: \
      status = launch_p32_large_m_grid_dispatch<TransitionBits, THREADS, 2>( \
          input_half, trellis_words, levels_half, bank_bytes, bank_alt_byte, \
          output, partial_output, size_m, size_k, size_n, config.split_count, \
          use_static_n, cuda_stream); \
      break; \
    case 3: \
      status = launch_p32_large_m_grid_dispatch<TransitionBits, THREADS, 3>( \
          input_half, trellis_words, levels_half, bank_bytes, bank_alt_byte, \
          output, partial_output, size_m, size_k, size_n, config.split_count, \
          use_static_n, cuda_stream); \
      break; \
    case 4: \
      status = launch_p32_large_m_grid_dispatch<TransitionBits, THREADS, 4>( \
          input_half, trellis_words, levels_half, bank_bytes, bank_alt_byte, \
          output, partial_output, size_m, size_k, size_n, config.split_count, \
          use_static_n, cuda_stream); \
      break; \
    default: \
      set_last_error("QVQ P32 large-M stage_k_tiles must be in [1, 4]"); \
      return -1; \
  }
  switch (config.threads) {
    case 64:
      QVQ_LARGE_M_STAGE(64);
      break;
    case 128:
      QVQ_LARGE_M_STAGE(128);
      break;
    case 256:
      QVQ_LARGE_M_STAGE(256);
      break;
    default:
      set_last_error("QVQ P32 large-M threads must be one of 64, 128, or 256");
      return -1;
  }
#undef QVQ_LARGE_M_STAGE
  return status;
}

}  // namespace

extern "C" int qvq_p32_abi_version(void) {
  return QVQ_P32_ABI_VERSION;
}

extern "C" int qvq_p32_kernel_version(void) {
  return QVQ_P32_KERNEL_VERSION;
}

extern "C" int qvq_compiled_sm(void) {
  return QVQ_P32_COMPILED_SM;
}

extern "C" int qvq_device_sm(int device) {
  cudaDeviceProp properties{};
  const cudaError_t error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) return 0;
  return properties.major * 10 + properties.minor;
}

extern "C" int qvq_driver_version(void) {
  int version = 0;
  return cudaDriverGetVersion(&version) == cudaSuccess ? version : 0;
}

extern "C" int qvq_runtime_version(void) {
  int version = 0;
  return cudaRuntimeGetVersion(&version) == cudaSuccess ? version : 0;
}

extern "C" int qvq_toolkit_version(void) {
  return CUDART_VERSION;
}

extern "C" int qvq_device_is_supported(int device) {
  cudaDeviceProp properties{};
  const cudaError_t error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) return 0;
  return properties.major == 8 && properties.minor == 0 ? 1 : 0;
}

extern "C" const char* qvq_last_error(void) {
  return last_error;
}

static int qvq_p32_window_impl(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int split_count,
    int kernel_variant,
    int threads,
    int stage_k_tiles,
    int static_n,
    int reduction_mode,
    int row_groups,
    void* stream) {
  if (input == nullptr || trellis == nullptr || levels == nullptr ||
      bank_ids == nullptr || bank_alt_id == nullptr || output == nullptr ||
      partial_output == nullptr ||
      stream == nullptr) {
    set_last_error("QVQ P32 received a null device pointer");
    return -1;
  }
  if (size_m < 1 || size_k <= 0 || size_k % QVQ_P32_TILE_SIZE != 0 ||
      size_n <= 0 || size_n % QVQ_P32_TILE_SIZE != 0) {
    set_last_error("QVQ P32 requires M >= 1 and K/N divisible by 16");
    return -1;
  }
  if (split_count < 1 || split_count > QVQ_P32_SPLIT_COUNT_MAX ||
      split_count > size_k / kTileRows) {
    set_last_error("QVQ P32 split_count exceeds the K-tile count or 128");
    return -1;
  }
  if (kernel_variant != QVQ_P32_VARIANT_SCALAR &&
      kernel_variant != QVQ_P32_VARIANT_BLOCK) {
    set_last_error("QVQ P32 kernel_variant must be scalar (1) or block (2)");
    return -1;
  }
  if (kernel_variant == QVQ_P32_VARIANT_SCALAR &&
      size_m > QVQ_P32_SCALAR_M_MAX) {
    set_last_error("QVQ P32 scalar variant supports M in [1,4]");
    return -1;
  }
  if (threads != 64 && threads != 128 && threads != 256) {
    set_last_error("QVQ P32 threads must be one of 64, 128, or 256");
    return -1;
  }
  if (stage_k_tiles < QVQ_P32_STAGE_K_TILES_MIN ||
      stage_k_tiles > QVQ_P32_STAGE_K_TILES_MAX) {
    set_last_error("QVQ P32 stage_k_tiles must be in [1, 4]");
    return -1;
  }
  if (static_n != 0 && static_n != 1) {
    set_last_error("QVQ P32 static_n must be 0 or 1");
    return -1;
  }
  if (reduction_mode != QVQ_P32_REDUCTION_NATIVE &&
      reduction_mode != QVQ_P32_REDUCTION_PARTIALS) {
    set_last_error("QVQ P32 reduction mode must be native (1) or graph-visible (2)");
    return -1;
  }
  if (row_groups != QVQ_P32_ROW_GROUPS_AUTO && row_groups != 1 &&
      row_groups != 2 && row_groups != 4 && row_groups != 8 &&
      row_groups != 16) {
    set_last_error("QVQ P32 row_groups must be auto, 1, 2, 4, 8, or 16");
    return -1;
  }
  if (size_m <= QVQ_P32_GROUPED_M_MAX &&
      row_groups != QVQ_P32_ROW_GROUPS_AUTO && row_groups != 1) {
    set_last_error("QVQ P32 multi-row groups require M > 16");
    return -1;
  }
  const bool scalar_static_n =
      kernel_variant == QVQ_P32_VARIANT_SCALAR &&
      ((size_m == 1 &&
        (size_n == 512 || size_n == 1024 || size_n == 2048 ||
         size_n == 5120 || size_n == 6144 || size_n == 8192 ||
         size_n == 10240 || size_n == 12288 || size_n == 17408)) ||
       (size_m == 2 &&
        ((size_k <= 6144 &&
          (size_n == 512 || size_n == 1024 || size_n == 2048 ||
           size_n == 5120 || size_n == 6144 || size_n == 8192 ||
           size_n == 10240 || size_n == 12288 || size_n == 17408)) ||
         (size_k == 17408 && size_n == 5120))) ||
       (size_m == 4 &&
        (size_k <= 6144 || (size_k == 17408 && size_n == 5120)) &&
        (size_n == 512 || size_n == 1024 || size_n == 2048 ||
         size_n == 5120 || size_n == 6144 || size_n == 8192 ||
         size_n == 10240 || size_n == 12288 || size_n == 17408)));
  const bool block_static_n =
      kernel_variant == QVQ_P32_VARIANT_BLOCK &&
      (size_m == 8 || size_m == 16) &&
      (size_n == 512 || size_n == 1024 || size_n == 2048 ||
       size_n == 5120 || size_n == 6144 || size_n == 8192 ||
       size_n == 10240 || size_n == 12288 || size_n == 17408);
  const bool large_m_static_n =
      kernel_variant == QVQ_P32_VARIANT_BLOCK && size_m > 16 &&
      (size_n == 512 || size_n == 1024 || size_n == 2048 ||
       size_n == 5120 || size_n == 6144 || size_n == 8192 ||
       size_n == 10240 || size_n == 12288 || size_n == 17408);
  if (static_n != 0 && !scalar_static_n && !block_static_n &&
      !large_m_static_n) {
    set_last_error("QVQ P32 static_n is unsupported for this shape/config");
    return -1;
  }
  const qvq_p32_config config = {
      split_count, kernel_variant, threads, stage_k_tiles, static_n,
      reduction_mode, QVQ_P32_TUNING_AUTO, QVQ_P32_N_TILES_AUTO,
      QVQ_P32_WARPS_AUTO};
  int status = 0;
  switch (transition_bits) {
    case QVQ_P32_TRANSITION_BITS_MIN:
      status = size_m <= 16
          ? launch_p32_config<4>(input, trellis, levels, bank_ids, bank_alt_id,
                                 output, partial_output, size_m, size_k, size_n,
                                 config, stream)
          : launch_p32_large_m<4>(input, trellis, levels, bank_ids, bank_alt_id,
                                  output, partial_output, size_m, size_k, size_n,
                                  config, row_groups, stream);
      break;
    case 5:
      status = size_m <= 16
          ? launch_p32_config<5>(input, trellis, levels, bank_ids, bank_alt_id,
                                 output, partial_output, size_m, size_k, size_n,
                                 config, stream)
          : launch_p32_large_m<5>(input, trellis, levels, bank_ids, bank_alt_id,
                                  output, partial_output, size_m, size_k, size_n,
                                  config, row_groups, stream);
      break;
    case 6:
      status = size_m <= 16
          ? launch_p32_config<6>(input, trellis, levels, bank_ids, bank_alt_id,
                                 output, partial_output, size_m, size_k, size_n,
                                 config, stream)
          : launch_p32_large_m<6>(input, trellis, levels, bank_ids, bank_alt_id,
                                  output, partial_output, size_m, size_k, size_n,
                                  config, row_groups, stream);
      break;
    case QVQ_P32_TRANSITION_BITS_MAX:
      status = size_m <= 16
          ? launch_p32_config<7>(input, trellis, levels, bank_ids, bank_alt_id,
                                 output, partial_output, size_m, size_k, size_n,
                                 config, stream)
          : launch_p32_large_m<7>(input, trellis, levels, bank_ids, bank_alt_id,
                                  output, partial_output, size_m, size_k, size_n,
                                  config, row_groups, stream);
      break;
    default:
      set_last_error("QVQ P32 transition_bits must be in [4,7]");
      status = -1;
      break;
  }
  if (status == 0 && size_m > QVQ_P32_GROUPED_M_MAX &&
      reduction_mode == QVQ_P32_REDUCTION_NATIVE && split_count > 1) {
    launch_split_reduction(partial_output, output, size_m, size_k, size_n,
                           split_count, reinterpret_cast<cudaStream_t>(stream));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      set_last_error(cudaGetErrorString(error));
      return static_cast<int>(error);
    }
  }
  return status;
}

extern "C" int qvq_p32_window(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int split_count,
    int kernel_variant,
    int threads,
    int stage_k_tiles,
    int static_n,
    int reduction_mode,
    void* stream) {
  return qvq_p32_window_impl(
      input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,
      size_m, size_k, size_n, transition_bits, split_count, kernel_variant,
      threads, stage_k_tiles, static_n, reduction_mode,
      QVQ_P32_ROW_GROUPS_AUTO, stream);
}

extern "C" int qvq_p32_window_with_row_groups(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int split_count,
    int kernel_variant,
    int threads,
    int stage_k_tiles,
    int static_n,
    int reduction_mode,
    int row_groups,
    void* stream) {
  return qvq_p32_window_impl(
      input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,
      size_m, size_k, size_n, transition_bits, split_count, kernel_variant,
      threads, stage_k_tiles, static_n, reduction_mode, row_groups, stream);
}

extern "C" int qvq_p32_window_tuned(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int row_groups,
    const qvq_p32_config* requested,
    void* stream) {
  qvq_p32_config config{};
  if (!normalize_external_tuning(requested, size_m, &config)) return -1;
  return qvq_p32_window_impl(
      input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,
      size_m, size_k, size_n, transition_bits, config.split_count,
      config.kernel_variant, config.threads, config.stage_k_tiles,
      config.static_n, config.reduction_mode, row_groups, stream);
}

extern "C" int qvq_p32_grouped_window(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_ids,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int split_count,
    int split_count_0,
    int split_count_1,
    int split_count_2,
    int kernel_variant,
    int threads,
    int stage_k_tiles,
    int static_n,
    int reduction_mode,
    int group_count,
    int n_tile_end_0,
    int n_tile_end_1,
    void* stream) {
  if (input == nullptr || trellis == nullptr || levels == nullptr ||
      bank_ids == nullptr || bank_alt_ids == nullptr || output == nullptr ||
      partial_output == nullptr || stream == nullptr) {
    set_last_error("QVQ P32 grouped received a null device pointer");
    return -1;
  }
  if (size_m < 1 || size_m > QVQ_P32_GROUPED_M_MAX || size_k <= 0 ||
      size_k % QVQ_P32_TILE_SIZE != 0 || size_n <= 0 ||
      size_n % QVQ_P32_TILE_SIZE != 0) {
    set_last_error("QVQ P32 grouped requires M in [1,16] and K/N divisible by 16");
    return -1;
  }
  if (group_count < QVQ_P32_GROUP_COUNT_MIN ||
      group_count > QVQ_P32_GROUP_COUNT_MAX) {
    set_last_error("QVQ P32 grouped requires two or three groups");
    return -1;
  }
  if (split_count < 1 || split_count > QVQ_P32_SPLIT_COUNT_MAX ||
      split_count > size_k / kTileRows ||
      split_count_0 < 1 || split_count_1 < 1 || split_count_2 < 1) {
    set_last_error("QVQ P32 grouped split_count exceeds the K-tile count or 128");
    return -1;
  }
  if (kernel_variant != QVQ_P32_VARIANT_SCALAR &&
      kernel_variant != QVQ_P32_VARIANT_BLOCK) {
    set_last_error("QVQ P32 grouped kernel variant must be scalar or block");
    return -1;
  }
  if ((kernel_variant == QVQ_P32_VARIANT_SCALAR &&
       size_m > QVQ_P32_SCALAR_M_MAX) ||
      (kernel_variant == QVQ_P32_VARIANT_BLOCK &&
       size_m <= QVQ_P32_SCALAR_M_MAX) ||
      static_n != 0 ||
      (reduction_mode != QVQ_P32_REDUCTION_NATIVE &&
       reduction_mode != QVQ_P32_REDUCTION_PARTIALS)) {
    set_last_error("QVQ P32 grouped variant does not match M or reduction configuration");
    return -1;
  }
  if (threads != 64 && threads != 128 && threads != 256) {
    set_last_error("QVQ P32 grouped threads must be one of 64, 128, or 256");
    return -1;
  }
  if (stage_k_tiles < QVQ_P32_STAGE_K_TILES_MIN ||
      stage_k_tiles > QVQ_P32_STAGE_K_TILES_MAX) {
    set_last_error("QVQ P32 grouped stage_k_tiles must be in [1, 4]");
    return -1;
  }
  if (transition_bits < QVQ_P32_TRANSITION_BITS_MIN ||
      transition_bits > QVQ_P32_TRANSITION_BITS_MAX) {
    set_last_error("QVQ P32 grouped transition_bits must be in [4, 7]");
    return -1;
  }
  const qvq_p32_config config = {
      split_count, kernel_variant, threads, stage_k_tiles, static_n,
      reduction_mode, QVQ_P32_TUNING_AUTO, QVQ_P32_N_TILES_AUTO,
      QVQ_P32_WARPS_AUTO};
  int status = 0;
  switch (transition_bits) {
    case QVQ_P32_TRANSITION_BITS_MIN:
      status = kernel_variant == QVQ_P32_VARIANT_SCALAR
          ? launch_p32_grouped_scalar<4>(
          input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
          size_m, size_k, size_n, group_count, split_count_0, split_count_1,
          split_count_2, n_tile_end_0, n_tile_end_1,
          config, stream)
          : launch_p32_grouped_block<4>(
          input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
          size_m, size_k, size_n, group_count, split_count_0, split_count_1,
          split_count_2, n_tile_end_0, n_tile_end_1,
          config, stream);
      break;
    case 5:
      status = kernel_variant == QVQ_P32_VARIANT_SCALAR
          ? launch_p32_grouped_scalar<5>(
          input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
          size_m, size_k, size_n, group_count, split_count_0, split_count_1,
          split_count_2, n_tile_end_0, n_tile_end_1,
          config, stream)
          : launch_p32_grouped_block<5>(
          input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
          size_m, size_k, size_n, group_count, split_count_0, split_count_1,
          split_count_2, n_tile_end_0, n_tile_end_1,
          config, stream);
      break;
    case 6:
      status = kernel_variant == QVQ_P32_VARIANT_SCALAR
          ? launch_p32_grouped_scalar<6>(
          input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
          size_m, size_k, size_n, group_count, split_count_0, split_count_1,
          split_count_2, n_tile_end_0, n_tile_end_1,
          config, stream)
          : launch_p32_grouped_block<6>(
          input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
          size_m, size_k, size_n, group_count, split_count_0, split_count_1,
          split_count_2, n_tile_end_0, n_tile_end_1,
          config, stream);
      break;
    case QVQ_P32_TRANSITION_BITS_MAX:
      status = kernel_variant == QVQ_P32_VARIANT_SCALAR
          ? launch_p32_grouped_scalar<7>(
          input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
          size_m, size_k, size_n, group_count, split_count_0, split_count_1,
          split_count_2, n_tile_end_0, n_tile_end_1,
          config, stream)
          : launch_p32_grouped_block<7>(
          input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
          size_m, size_k, size_n, group_count, split_count_0, split_count_1,
          split_count_2, n_tile_end_0, n_tile_end_1,
          config, stream);
      break;
  }
  if (status != 0 && last_error[0] == '\0') {
    set_last_error("QVQ P32 grouped launch failed");
  }
  return status;
}

extern "C" int qvq_p32_grouped_window_tuned(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_ids,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int split_count,
    int split_count_0,
    int split_count_1,
    int split_count_2,
    int group_count,
    int n_tile_end_0,
    int n_tile_end_1,
    const qvq_p32_config* requested,
    void* stream) {
  qvq_p32_config config{};
  if (!normalize_external_tuning(requested, size_m, &config)) return -1;
  if (config.split_count != split_count) {
    set_last_error("QVQ P32 tuning split_count disagrees with the call");
    return -1;
  }
  return qvq_p32_grouped_window(
      input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
      size_m, size_k, size_n, transition_bits, split_count, split_count_0,
      split_count_1, split_count_2, config.kernel_variant, config.threads,
      config.stage_k_tiles, config.static_n, config.reduction_mode, group_count,
      n_tile_end_0, n_tile_end_1, stream);
}

extern "C" int qvq_p32_grouped_launch_plan(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_ids,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int split_count,
    int split_count_0,
    int split_count_1,
    int split_count_2,
    int kernel_variant,
    int threads,
    int stage_k_tiles,
    int static_n,
    int reduction_mode,
    int group_count,
    int n_tile_end_0,
    int n_tile_end_1,
    qvq_p32_launch_plan* plan) {
  if (input == nullptr || trellis == nullptr || levels == nullptr ||
      bank_ids == nullptr || bank_alt_ids == nullptr || output == nullptr ||
      partial_output == nullptr || plan == nullptr) {
    set_last_error("QVQ P32 grouped launch plan received a null pointer");
    return -1;
  }
  if (size_m < 1 || size_m > QVQ_P32_GROUPED_M_MAX || size_k <= 0 ||
      size_k % QVQ_P32_TILE_SIZE != 0 || size_n <= 0 ||
      size_n % QVQ_P32_TILE_SIZE != 0) {
    set_last_error("QVQ P32 grouped launch plan requires M in [1,16] and K/N divisible by 16");
    return -1;
  }
  if (group_count < QVQ_P32_GROUP_COUNT_MIN ||
      group_count > QVQ_P32_GROUP_COUNT_MAX) {
    set_last_error("QVQ P32 grouped launch plan requires two or three groups");
    return -1;
  }
  if (split_count < 1 || split_count > QVQ_P32_SPLIT_COUNT_MAX ||
      split_count > size_k / kTileRows || split_count_0 < 1 ||
      split_count_1 < 1 || split_count_2 < 1) {
    set_last_error("QVQ P32 grouped launch plan split count is invalid");
    return -1;
  }
  if (kernel_variant != QVQ_P32_VARIANT_SCALAR &&
      kernel_variant != QVQ_P32_VARIANT_BLOCK) {
    set_last_error("QVQ P32 grouped launch plan kernel variant is invalid");
    return -1;
  }
  if ((kernel_variant == QVQ_P32_VARIANT_SCALAR &&
       size_m > QVQ_P32_SCALAR_M_MAX) ||
      (kernel_variant == QVQ_P32_VARIANT_BLOCK &&
       size_m <= QVQ_P32_SCALAR_M_MAX) ||
      static_n != 0 ||
      (reduction_mode != QVQ_P32_REDUCTION_NATIVE &&
       reduction_mode != QVQ_P32_REDUCTION_PARTIALS)) {
    set_last_error("QVQ P32 grouped launch plan variant does not match M or reduction configuration");
    return -1;
  }
  if ((threads != 64 && threads != 128 && threads != 256) ||
      stage_k_tiles < QVQ_P32_STAGE_K_TILES_MIN ||
      stage_k_tiles > QVQ_P32_STAGE_K_TILES_MAX ||
      transition_bits < QVQ_P32_TRANSITION_BITS_MIN ||
      transition_bits > QVQ_P32_TRANSITION_BITS_MAX) {
    set_last_error("QVQ P32 grouped launch plan specialization is invalid");
    return -1;
  }

  std::memset(plan, 0, sizeof(*plan));
  auto* storage = reinterpret_cast<GroupedPlanStorage*>(plan->host_storage);
  if (!build_grouped_params(
          size_m, size_n, group_count, split_count_0, split_count_1,
          split_count_2, n_tile_end_0, n_tile_end_1, &storage->params)) {
    return -1;
  }
  if (!validate_grouped_splits(storage->params, size_k)) return -1;

  const int total_n_tiles = size_n / kTileColumns;
  const int tiles_per_block = kernel_variant == QVQ_P32_VARIANT_SCALAR
      ? 4 * (threads / 32)
      : threads / 32;
  int64_t grouped_work = 0;
  if (!grouped_work_count(storage->params, tiles_per_block, &grouped_work)) {
    return -1;
  }

  auto* main = &plan->launches[0];
  main->kernel_symbol = grouped_kernel_symbol(
      transition_bits, size_m, kernel_variant, threads, stage_k_tiles);
  if (main->kernel_symbol == nullptr) {
    set_last_error("QVQ P32 grouped launch-plan kernel symbol is unavailable");
    return -1;
  }
  main->kernel_name = kernel_variant == QVQ_P32_VARIANT_SCALAR
      ? "qvq_p32_grouped_scalar"
      : "qvq_p32_grouped_block";
  main->grid_x = static_cast<unsigned>(grouped_work);
  main->grid_y = 1;
  main->grid_z = 1;
  main->block_x = static_cast<unsigned>(threads);
  main->block_y = 1;
  main->block_z = 1;
  int arg = 0;
  set_device_arg(main, arg++, input);
  set_device_arg(main, arg++, trellis);
  set_device_arg(main, arg++, levels);
  set_device_arg(main, arg++, bank_ids);
  set_device_arg(main, arg++, bank_alt_ids);
  set_host_arg(main, arg++, &storage->params);
  set_device_arg(main, arg++, partial_output);
  set_device_arg(main, arg++, output);
  storage->main_size_m = size_m;
  storage->main_size_k = size_k;
  storage->main_total_n_tiles = total_n_tiles;
  if (kernel_variant == QVQ_P32_VARIANT_BLOCK) {
    set_host_arg(main, arg++, &storage->main_size_m);
  }
  set_host_arg(main, arg++, &storage->main_size_k);
  set_host_arg(main, arg++, &storage->main_total_n_tiles);
  main->arg_count = arg;
  plan->launch_count = 1;
  if (reduction_mode == QVQ_P32_REDUCTION_PARTIALS) return 0;

  for (int segment = 0; segment < group_count; ++segment) {
    if (storage->params.split_count[segment] == 1) continue;
    if (plan->launch_count >= QVQ_P32_LAUNCH_PLAN_MAX_LAUNCHES) {
      set_last_error("QVQ P32 grouped launch plan exceeds launch capacity");
      return -1;
    }
    auto* reduction = &plan->launches[plan->launch_count];
    auto* values = &storage->reductions[segment];
    values->size_m = size_m;
    values->group_n = storage->params.n_tiles[segment] * kTileColumns;
    values->total_n = size_n;
    values->group_n_offset =
        storage->params.n_tile_start[segment] * kTileColumns;
    values->split_count = storage->params.split_count[segment];
    const int output_values = values->size_m * values->group_n;
    reduction->kernel_symbol = reinterpret_cast<const void*>(
        reduce_split_grouped_kernel<>);
    reduction->kernel_name = "qvq_p32_grouped_reduce";
    reduction->grid_x = static_cast<unsigned>(
        (output_values + kReductionThreads - 1) / kReductionThreads);
    reduction->grid_y = 1;
    reduction->grid_z = 1;
    reduction->block_x = kReductionThreads;
    reduction->block_y = 1;
    reduction->block_z = 1;
    int reduction_arg = 0;
    set_device_arg(
        reduction,
        reduction_arg++,
        partial_output + storage->params.partial_offset[segment]);
    set_device_arg(reduction, reduction_arg++, output);
    set_host_arg(reduction, reduction_arg++, &values->size_m);
    set_host_arg(reduction, reduction_arg++, &values->group_n);
    set_host_arg(reduction, reduction_arg++, &values->total_n);
    set_host_arg(reduction, reduction_arg++, &values->group_n_offset);
    set_host_arg(reduction, reduction_arg++, &values->split_count);
    reduction->arg_count = reduction_arg;
    reduction->dependency_count = 1;
    reduction->dependencies[0] = 0;
    ++plan->launch_count;
  }
  return 0;
}

extern "C" int qvq_p32_grouped_launch_plan_tuned(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_ids,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int split_count,
    int split_count_0,
    int split_count_1,
    int split_count_2,
    int group_count,
    int n_tile_end_0,
    int n_tile_end_1,
    const qvq_p32_config* requested,
    qvq_p32_launch_plan* plan) {
  qvq_p32_config config{};
  if (!normalize_external_tuning(requested, size_m, &config)) return -1;
  if (config.split_count != split_count) {
    set_last_error("QVQ P32 tuning split_count disagrees with the plan call");
    return -1;
  }
  return qvq_p32_grouped_launch_plan(
      input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
      size_m, size_k, size_n, transition_bits, split_count, split_count_0,
      split_count_1, split_count_2, config.kernel_variant, config.threads,
      config.stage_k_tiles, config.static_n, config.reduction_mode, group_count,
      n_tile_end_0, n_tile_end_1, plan);
}
