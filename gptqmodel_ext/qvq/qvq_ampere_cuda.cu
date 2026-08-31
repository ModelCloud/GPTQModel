// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>

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
constexpr int kStageColumns = kStageKTiles * kTileRows;
constexpr int kPairsPerTile = 128;
constexpr int kLevels = 256;
constexpr uint32_t kPgc16Multiplier = 40503u;
constexpr uint32_t kPgc16Increment = 17011u;
constexpr int kMaxCachedCudaDevices = 64;
std::array<std::atomic<int>, kMaxCachedCudaDevices> device_capability_cache{};

int cached_device_capability(int device_index) {
  int encoded_capability = 0;
  if (device_index >= 0 && device_index < kMaxCachedCudaDevices) {
    encoded_capability =
        device_capability_cache[device_index].load(std::memory_order_relaxed);
  }
  if (encoded_capability == 0) {
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));
    encoded_capability = properties.major * 10 + properties.minor + 1;
    if (device_index >= 0 && device_index < kMaxCachedCudaDevices) {
      device_capability_cache[device_index].store(
          encoded_capability, std::memory_order_relaxed);
    }
  }
  return encoded_capability - 1;
}

__device__ __forceinline__ uint32_t pgc16_mix(uint32_t state) {
  uint32_t mixed = state ^ (state >> 8);
  mixed = (mixed * kPgc16Multiplier + kPgc16Increment) & 0xffffu;
  return mixed ^ (mixed >> 7);
}

template <int TransitionBits>
__device__ __forceinline__ uint32_t alternate_bank_mask(int bank_alt_id) {
  static_assert(TransitionBits >= 4 && TransitionBits <= 7);
  if (bank_alt_id == 0) {
    return 0u;
  }
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
// Hopper kernel, expressed here with ordinary Ampere integer instructions.
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
  // The state windows are at most 27 bits wide on W3.5.  A 32-bit funnel
  // shift expresses the same circular extraction without promoting each
  // pair to a 64-bit value (which costs multiple integer instructions on
  // sm_80).
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

template <int TransitionBits>
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
  decoded.values = __halves2half2(
      __ldg(levels + (mixed >> 8)), __ldg(levels + (mixed & 0xffu)));
  return decoded.bits;
}

__device__ __forceinline__ uint32_t decode_state_bits(
    uint32_t state,
    uint32_t bank_mask,
    const half* __restrict__ levels) {
  const uint32_t mixed = pgc16_mix(state ^ bank_mask);
  union {
    uint32_t bits;
    half2 values;
  } decoded;
  // The codebook is only 512 bytes and is read randomly by every lane.  On
  // Ampere, keeping it in the read-only path avoids a per-CTA shared-memory
  // copy and the bank conflicts that copy creates for the decode gathers.
  decoded.values = __halves2half2(
      __ldg(levels + (mixed >> 8)), __ldg(levels + (mixed & 0xffu)));
  return decoded.bits;
}

__device__ __forceinline__ uint32_t pack_low_halves(uint32_t first, uint32_t second) {
  return (first & 0xffffu) | (second << 16);
}

__device__ __forceinline__ uint32_t pack_high_halves(uint32_t first, uint32_t second) {
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

// Trellis words are consumed once by the owning CTA.  Bypass L1 for this
// streaming payload so the read-only codebook path and staged activations do
// not compete with it for the small Ampere L1/TEX pipe.
__device__ __forceinline__ void copy_async_cg_16(void* destination, const void* source) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :
      : "r"(shared_address), "l"(source));
}

template <
    int TransitionBits,
    bool FullRows,
    int ActiveRows = 0,
    int StaticN = 0,
    bool HoistBankMasks = false,
    bool UpperRowsOnly = false,
    int StaticK = 0>
__global__ __launch_bounds__(kThreads) void p32_window_ampere_kernel(
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
    int bank_alt_id) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  constexpr int kWordsPerTile = 4 * TransitionBits;
  __shared__ __align__(32) half input_tile[2][kRows * kStageColumns];
  __shared__ __align__(16) uint32_t packed_words[2][kStageKTiles][kTilesPerBlock][kWordsPerTile];
  __shared__ __align__(4) uint8_t packed_bank_ids[2][kStageKTiles][kTilesPerBlock];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  constexpr int kStaticNTiles = StaticN > 0 ? StaticN / kTileColumns : 0;
  constexpr int kStaticKTiles = StaticK > 0 ? StaticK / kTileRows : 0;
  const int n_tiles = StaticN > 0 ? kStaticNTiles : size_n / kTileColumns;
  const int n_tile_base = static_cast<int>(blockIdx.x) * kTilesPerBlock;
  const bool active_tile = StaticN > 0 || n_tile_base + warp < n_tiles;
  const int split = static_cast<int>(blockIdx.z);
  const int k_tiles = StaticK > 0 ? kStaticKTiles : size_k / kTileRows;
  const int input_stride = StaticK > 0 ? StaticK : size_k;
  const int k_tile_begin = (k_tiles * split) / split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / split_count;
  const uint32_t alt_mask = alternate_bank_mask<TransitionBits>(bank_alt_id);

  auto stage = [&](int k_tile_base, int destination) {
    auto* input_vectors = reinterpret_cast<uint4*>(input_tile[destination]);
    for (int index = thread; index < kRows * kStageColumns / 8; index += kThreads) {
      const int row = index / (kStageColumns / 8);
      const int vector = index - row * (kStageColumns / 8);
      const int source_column = k_tile_base * kTileRows + vector * 8;
      if constexpr (FullRows) {
        if (source_column < input_stride) {
          const half* source = input + static_cast<int64_t>(row) * input_stride + source_column;
          __pipeline_memcpy_async(input_vectors + index, reinterpret_cast<const uint4*>(source), 16);
        } else {
          input_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
        }
      } else if constexpr (ActiveRows > 0) {
        if (row < ActiveRows && source_column < input_stride) {
          const half* source = input + static_cast<int64_t>(row) * input_stride +
              source_column;
          __pipeline_memcpy_async(input_vectors + index, reinterpret_cast<const uint4*>(source), 16);
        } else {
          input_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
        }
      } else if (row < size_m && source_column < input_stride) {
        const half* source = input + static_cast<int64_t>(row) * input_stride +
            source_column;
        __pipeline_memcpy_async(input_vectors + index, reinterpret_cast<const uint4*>(source), 16);
      } else {
        input_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
      }
    }

    constexpr int kVectorsPerTile = kWordsPerTile / 4;
    constexpr int kVectorsPerKTile = kTilesPerBlock * kVectorsPerTile;
    constexpr int kVectorsPerBlock = kStageKTiles * kVectorsPerKTile;
    auto* destination_vectors = reinterpret_cast<uint4*>(packed_words[destination]);
    for (int index = thread; index < kVectorsPerBlock; index += kThreads) {
      const int stage_k_tile = index / kVectorsPerKTile;
      const int tile_index = index - stage_k_tile * kVectorsPerKTile;
      const int tile = tile_index / kVectorsPerTile;
      const int vector = tile_index - tile * kVectorsPerTile;
      const int n_tile = n_tile_base + tile;
      const int k_tile = k_tile_base + stage_k_tile;
      if constexpr (StaticN > 0) {
        if (k_tile < k_tiles) {
          const int64_t global_tile = static_cast<int64_t>(k_tile) * n_tiles + n_tile;
          const auto* source_vectors = reinterpret_cast<const uint4*>(
              trellis + global_tile * kWordsPerTile);
          copy_async_cg_16(destination_vectors + index, source_vectors + vector);
        } else {
          destination_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
        }
      } else if (n_tile < n_tiles && k_tile < k_tiles) {
        const int64_t global_tile = static_cast<int64_t>(k_tile) * n_tiles + n_tile;
        const auto* source_vectors = reinterpret_cast<const uint4*>(
            trellis + global_tile * kWordsPerTile);
        copy_async_cg_16(destination_vectors + index, source_vectors + vector);
      } else {
        destination_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    if constexpr (StaticN > 0 && StaticN != 1024 && FullRows) {
      if (thread < kStageKTiles) {
        const int k_tile = k_tile_base + thread;
        auto* destination_ids = reinterpret_cast<uint32_t*>(
            packed_bank_ids[destination][thread]);
        *destination_ids = k_tile < k_tiles
            ? __ldg(reinterpret_cast<const uint32_t*>(
                  bank_ids + static_cast<int64_t>(k_tile) * n_tiles + n_tile_base))
            : 0u;
      }
    } else if (thread < kStageKTiles * kTilesPerBlock) {
      const int stage_k_tile = thread / kTilesPerBlock;
      const int tile = thread - stage_k_tile * kTilesPerBlock;
      const int n_tile = n_tile_base + tile;
      const int k_tile = k_tile_base + stage_k_tile;
      packed_bank_ids[destination][stage_k_tile][tile] = n_tile < n_tiles && k_tile < k_tiles
          ? bank_ids[static_cast<int64_t>(k_tile) * n_tiles + n_tile]
          : 0;
    }
    __pipeline_commit();
  };

  static_assert(!UpperRowsOnly || (!FullRows && ActiveRows == 8));
  constexpr bool kUpperRowsOnly = UpperRowsOnly;
  MmaAccumulator<kUpperRowsOnly> accumulator_0 = {};
  MmaAccumulator<kUpperRowsOnly> accumulator_1 = {};

  stage(k_tile_begin, 0);
  int parity = 0;
  for (int k_tile = k_tile_begin; k_tile < k_tile_end; k_tile += kStageKTiles) {
    const bool has_next = k_tile + kStageKTiles < k_tile_end;
    if (has_next) {
      stage(k_tile + kStageKTiles, parity ^ 1);
    }
    __pipeline_wait_prior(has_next ? 1 : 0);
    __syncthreads();

    if (active_tile) {
#pragma unroll
      for (int stage_k_tile = 0; stage_k_tile < kStageKTiles; ++stage_k_tile) {
        if (k_tile + stage_k_tile >= k_tile_end) {
          continue;
        }
        const uint32_t* words = packed_words[parity][stage_k_tile][warp];
        const uint8_t packed_bank_id = packed_bank_ids[parity][stage_k_tile][warp];
        const int producer_fragment = lane >> 4;
        const int producer_pair_column = producer_fragment * 4 + ((lane >> 2) & 3);
        const int producer_row_pair = lane & 3;
        const int first_pair = producer_row_pair * 16 + producer_pair_column;
        const int second_pair = first_pair + 8;
        uint32_t state_row_0;
        uint32_t state_row_8;
        uint32_t state_row_1;
        uint32_t state_row_9;
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
          decoded_row_0 = decode_state_bits(state_row_0, bank_mask_0, levels);
          decoded_row_8 = decode_state_bits(state_row_8, bank_mask_8, levels);
          decoded_row_1 = decode_state_bits(state_row_1, bank_mask_0, levels);
          decoded_row_9 = decode_state_bits(state_row_9, bank_mask_8, levels);
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

        const uint32_t low_rows_01 = pack_low_halves(decoded_row_0, decoded_row_1);
        const uint32_t high_rows_01 = pack_high_halves(decoded_row_0, decoded_row_1);
        const uint32_t low_rows_89 = pack_low_halves(decoded_row_8, decoded_row_9);
        const uint32_t high_rows_89 = pack_high_halves(decoded_row_8, decoded_row_9);
        const int target_column = lane >> 2;
        const int source_lane_0 = ((target_column >> 1) << 2) + (lane & 3);
        const int source_lane_1 = source_lane_0 + 16;
        const bool select_high = (target_column & 1) != 0;
        constexpr uint32_t kFullWarpMask = 0xffffffffu;
        MmaFragmentB weight_fragment_0;
        MmaFragmentB weight_fragment_1;
        const uint32_t low_01_0 = __shfl_sync(kFullWarpMask, low_rows_01, source_lane_0);
        const uint32_t high_01_0 = __shfl_sync(kFullWarpMask, high_rows_01, source_lane_0);
        const uint32_t low_89_0 = __shfl_sync(kFullWarpMask, low_rows_89, source_lane_0);
        const uint32_t high_89_0 = __shfl_sync(kFullWarpMask, high_rows_89, source_lane_0);
        const uint32_t low_01_1 = __shfl_sync(kFullWarpMask, low_rows_01, source_lane_1);
        const uint32_t high_01_1 = __shfl_sync(kFullWarpMask, high_rows_01, source_lane_1);
        const uint32_t low_89_1 = __shfl_sync(kFullWarpMask, low_rows_89, source_lane_1);
        const uint32_t high_89_1 = __shfl_sync(kFullWarpMask, high_rows_89, source_lane_1);
        weight_fragment_0.values[0] = select_high ? high_01_0 : low_01_0;
        weight_fragment_0.values[1] = select_high ? high_89_0 : low_89_0;
        weight_fragment_1.values[0] = select_high ? high_01_1 : low_01_1;
        weight_fragment_1.values[1] = select_high ? high_89_1 : low_89_1;

        MmaFragmentA input_fragment;
        if constexpr (kUpperRowsOnly && StaticN != 1024) {
          const int address_row = lane & 7;
          const int address_column = ((lane >> 3) & 1) * 8;
          load_mma_fragment_a_upper(
              input_fragment,
              input_tile[parity] +
                  address_row * kStageColumns +
                  stage_k_tile * kTileRows +
                  address_column);
        } else {
          const int address_row = (lane & 7) + ((lane >> 3) & 1) * 8;
          const int address_column = (lane >> 4) * 8;
          load_mma_fragment_a(
              input_fragment,
              input_tile[parity] +
                  address_row * kStageColumns +
                  stage_k_tile * kTileRows +
                  address_column);
        }
        mma_m16n8k16(input_fragment, weight_fragment_0, accumulator_0);
        mma_m16n8k16(input_fragment, weight_fragment_1, accumulator_1);
      }
    }
    // Every warp must finish consuming the current stage before any lane can
    // enter the next iteration and cp.async-overwrite that buffer two stages
    // later. Independent thread scheduling makes warp-local completion alone
    // insufficient for this block-wide producer/consumer handoff.
    __syncthreads();
    parity ^= 1;
  }

  float* target = split_count == 1
      ? output
      : partial_output + static_cast<int64_t>(split) * size_m * size_n;
  if (active_tile) {
    const int output_row_0 = lane >> 2;
    const int output_row_1 = output_row_0 + 8;
    const int output_column =
        (n_tile_base + warp) * kTileColumns + (lane & 3) * 2;
    if constexpr (FullRows) {
      store_output_pair<StaticN != 1024>(
          target + static_cast<int64_t>(output_row_0) * size_n + output_column,
          accumulator_0.values[0], accumulator_0.values[1]);
      store_output_pair<StaticN != 1024>(
          target + static_cast<int64_t>(output_row_1) * size_n + output_column,
          accumulator_0.values[2], accumulator_0.values[3]);
      store_output_pair<StaticN != 1024>(
          target + static_cast<int64_t>(output_row_0) * size_n + output_column + 8,
          accumulator_1.values[0], accumulator_1.values[1]);
      store_output_pair<StaticN != 1024>(
          target + static_cast<int64_t>(output_row_1) * size_n + output_column + 8,
          accumulator_1.values[2], accumulator_1.values[3]);
    } else if constexpr (ActiveRows > 0) {
      if (output_row_0 < ActiveRows) {
        store_output_pair<StaticN != 1024>(
            target + static_cast<int64_t>(output_row_0) * size_n + output_column,
            accumulator_0.values[0], accumulator_0.values[1]);
        store_output_pair<StaticN != 1024>(
            target + static_cast<int64_t>(output_row_0) * size_n + output_column + 8,
            accumulator_1.values[0], accumulator_1.values[1]);
      }
      if constexpr (!kUpperRowsOnly) {
        if (output_row_1 < ActiveRows) {
          store_output_pair<StaticN != 1024>(
              target + static_cast<int64_t>(output_row_1) * size_n + output_column,
              accumulator_0.values[2], accumulator_0.values[3]);
          store_output_pair<StaticN != 1024>(
              target + static_cast<int64_t>(output_row_1) * size_n + output_column + 8,
              accumulator_1.values[2], accumulator_1.values[3]);
        }
      }
    } else {
      if (output_row_0 < size_m) {
        store_output_pair<false>(
            target + static_cast<int64_t>(output_row_0) * size_n + output_column,
            accumulator_0.values[0], accumulator_0.values[1]);
        store_output_pair<false>(
            target + static_cast<int64_t>(output_row_0) * size_n + output_column + 8,
            accumulator_1.values[0], accumulator_1.values[1]);
      }
      if (output_row_1 < size_m) {
        store_output_pair<false>(
            target + static_cast<int64_t>(output_row_1) * size_n + output_column,
            accumulator_0.values[2], accumulator_0.values[3]);
        store_output_pair<false>(
            target + static_cast<int64_t>(output_row_1) * size_n + output_column + 8,
            accumulator_1.values[2], accumulator_1.values[3]);
      }
    }
  }
#endif
}

// M=1 is a distinct workload on Ampere.  The WMMA path above must carry a
// 16-row A fragment and execute a full m16n8k16 instruction even though only
// its first row is live.  This Marlin-style route gives each warp four N16
// tiles and computes only the 16 dot products that are written.  The state
// windows are still decoded exactly as in the WMMA path; pairs 0..63 are
// extracted together with their distance-64 partners so the circular payload
// remains storage-neutral.
template <
    int TransitionBits,
    int Rows,
    int Threads = kM1Threads,
    int TilesPerBlock = kM1TilesPerBlock,
    int StageKTiles = kStageKTiles,
    int StaticN = 0>
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
    int bank_alt_id) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  constexpr int kWordsPerTile = 4 * TransitionBits;
  static_assert(Rows >= 1 && Rows <= 8);
  __shared__ __align__(32) half input_tile[2][Rows * StageKTiles * kTileRows];
  __shared__ __align__(16) uint32_t packed_words[
      2][StageKTiles][TilesPerBlock][kWordsPerTile];
  __shared__ __align__(16) uint8_t packed_bank_ids[2][StageKTiles][TilesPerBlock];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  constexpr int kStaticNTiles = StaticN > 0 ? StaticN / kTileColumns : 0;
  const int n_tiles = StaticN > 0 ? kStaticNTiles : size_n / kTileColumns;
  const int block_n_tile_base = static_cast<int>(blockIdx.x) * TilesPerBlock;
  const int n_tile_base = block_n_tile_base + warp * 4;
  const int split = static_cast<int>(blockIdx.z);
  const int k_tiles = size_k / kTileRows;
  const int k_tile_begin = (k_tiles * split) / split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / split_count;
  const uint32_t alt_mask = alternate_bank_mask<TransitionBits>(bank_alt_id);

  auto stage = [&](int k_tile_base, int destination) {
    auto* input_vectors = reinterpret_cast<uint4*>(input_tile[destination]);
    constexpr int kInputVectorsPerRow = StageKTiles * kTileRows / 8;
    for (int index = thread; index < Rows * kInputVectorsPerRow; index += Threads) {
      const int row = index / kInputVectorsPerRow;
      const int vector = index - row * kInputVectorsPerRow;
      const int source_column = k_tile_base * kTileRows + vector * 8;
      if (source_column < size_k) {
        const half* source = input + static_cast<int64_t>(row) * size_k + source_column;
        __pipeline_memcpy_async(
            input_vectors + index, reinterpret_cast<const uint4*>(source), 16);
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
          const int64_t global_tile = static_cast<int64_t>(k_tile) * n_tiles + n_tile;
          const auto* source_vectors = reinterpret_cast<const uint4*>(
              trellis + global_tile * kWordsPerTile);
          copy_async_cg_16(destination_vectors + index, source_vectors + vector);
        } else {
          destination_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
        }
      } else if (n_tile < n_tiles && k_tile < k_tiles) {
        const int64_t global_tile = static_cast<int64_t>(k_tile) * n_tiles + n_tile;
        const auto* source_vectors = reinterpret_cast<const uint4*>(
            trellis + global_tile * kWordsPerTile);
        copy_async_cg_16(destination_vectors + index, source_vectors + vector);
      } else {
        destination_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    if constexpr (
        StaticN == 1024 && TilesPerBlock == 16 && Rows == 1) {
      if (thread < StageKTiles * 4) {
        const int stage_k_tile = thread >> 2;
        const int word = thread & 3;
        const int k_tile = k_tile_base + stage_k_tile;
        auto* destination_ids = reinterpret_cast<uint32_t*>(
            packed_bank_ids[destination][stage_k_tile]);
        destination_ids[word] = k_tile < k_tiles
            ? __ldg(reinterpret_cast<const uint32_t*>(
                  bank_ids + static_cast<int64_t>(k_tile) * n_tiles +
                      block_n_tile_base) + word)
            : 0u;
      }
    } else if constexpr (
        StaticN > 0 && TilesPerBlock == 16 &&
        (Rows == 2 || (Rows == 4 && StaticN != 1024))) {
      if (thread < StageKTiles) {
        const int k_tile = k_tile_base + thread;
        auto* destination_ids = reinterpret_cast<uint4*>(
            packed_bank_ids[destination][thread]);
        *destination_ids = k_tile < k_tiles
            ? __ldg(reinterpret_cast<const uint4*>(
                  bank_ids + static_cast<int64_t>(k_tile) * n_tiles + block_n_tile_base))
            : make_uint4(0u, 0u, 0u, 0u);
      }
    } else if (thread < StageKTiles * TilesPerBlock) {
      const int stage_k_tile = thread / TilesPerBlock;
      const int tile = thread - stage_k_tile * TilesPerBlock;
      const int n_tile = block_n_tile_base + tile;
      const int k_tile = k_tile_base + stage_k_tile;
      if constexpr (StaticN > 0) {
        packed_bank_ids[destination][stage_k_tile][tile] =
            k_tile < k_tiles ? bank_ids[static_cast<int64_t>(k_tile) * n_tiles + n_tile] : 0;
      } else {
        packed_bank_ids[destination][stage_k_tile][tile] =
            n_tile < n_tiles && k_tile < k_tiles
                ? bank_ids[static_cast<int64_t>(k_tile) * n_tiles + n_tile]
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
    if (has_next) {
      stage(k_tile + StageKTiles, parity ^ 1);
    }
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
          if (k_tile + stage_k_tile >= k_tile_end) {
            continue;
          }
          const uint32_t* words = packed_words[parity][stage_k_tile][shared_tile];
          const uint8_t packed_bank_id =
              packed_bank_ids[parity][stage_k_tile][shared_tile];
#pragma unroll
          for (int row = 0; row < 8; ++row) {
            const int pair = row * 8 + pair_column;
            uint32_t state_0;
            uint32_t state_8;
            window_state_pair64<TransitionBits>(words, pair, state_0, state_8);
            const uint32_t decoded_0 = decode_pair_bits<TransitionBits>(
                pair, state_0, packed_bank_id, alt_mask, levels);
            const uint32_t decoded_8 = decode_pair_bits<TransitionBits>(
                pair + 64, state_8, packed_bank_id, alt_mask, levels);
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
              const int row_base = output_row * (StageKTiles * kTileRows) + stage_k_tile * kTileRows;
              const float input_0 = __half2float(input_tile[parity][row_base + row]);
              const float input_8 = __half2float(input_tile[parity][row_base + row + 8]);
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
    } else if (n_tile < n_tiles) {
#pragma unroll
      for (int stage_k_tile = 0; stage_k_tile < StageKTiles; ++stage_k_tile) {
        if (k_tile + stage_k_tile >= k_tile_end) {
          continue;
        }
        const uint32_t* words = packed_words[parity][stage_k_tile][shared_tile];
        const uint8_t packed_bank_id =
            packed_bank_ids[parity][stage_k_tile][shared_tile];
#pragma unroll
        for (int row = 0; row < 8; ++row) {
          const int pair = row * 8 + pair_column;
          uint32_t state_0;
          uint32_t state_8;
          window_state_pair64<TransitionBits>(words, pair, state_0, state_8);
          const uint32_t decoded_0 = decode_pair_bits<TransitionBits>(
              pair, state_0, packed_bank_id, alt_mask, levels);
          const uint32_t decoded_8 = decode_pair_bits<TransitionBits>(
              pair + 64, state_8, packed_bank_id, alt_mask, levels);
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
            const int row_base = output_row * (StageKTiles * kTileRows) + stage_k_tile * kTileRows;
            const float input_0 = __half2float(input_tile[parity][row_base + row]);
            const float input_8 = __half2float(input_tile[parity][row_base + row + 8]);
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
    __syncthreads();
    parity ^= 1;
  }

  float* target = split_count == 1
      ? output
      : partial_output + static_cast<int64_t>(split) * Rows * size_n;
  const int n_tile = n_tile_base + (lane >> 3);
  if ((StaticN > 0 || n_tile < n_tiles) && (lane & 7) < 8) {
    const int output_column = n_tile * kTileColumns + (lane & 7) * 2;
#pragma unroll
    for (int output_row = 0; output_row < Rows; ++output_row) {
      float* row_target = target + static_cast<int64_t>(output_row) * size_n;
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
    int StageKTiles = kStageKTiles>
inline bool launch_static_n_scalar_kernel(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    float* partial_output,
    float* output,
    int size_k,
    int size_n,
    int split_count,
    int bank_alt_id,
    const dim3 grid,
    const cudaStream_t stream) {
#define QVQ_LAUNCH_STATIC_N(N)                                                        \
  case N:                                                                             \
    p32_window_ampere_m1_kernel<TransitionBits, Rows, Threads, TilesPerBlock, StageKTiles, N> \
        <<<grid, Threads, 0, stream>>>(                                               \
            input, trellis, levels, bank_ids, partial_output, output, size_k, size_n, \
            split_count, bank_alt_id);                                                \
    return true
  switch (size_n) {
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

__global__ void reduce_split_kernel(
    const float* __restrict__ partial_output,
    float* __restrict__ output,
    int output_values,
    int split_count) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= output_values) {
    return;
  }
  float accumulator = 0.0f;
  for (int split = 0; split < split_count; ++split) {
    accumulator += partial_output[static_cast<int64_t>(split) * output_values + index];
  }
  output[index] = accumulator;
}

template <int TransitionBits>
at::Tensor p32_window_ampere_impl(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t out_features,
    int64_t bank_alt_id,
    int64_t split_count) {
  constexpr int kWordsPerTile = 4 * TransitionBits;
  TORCH_CHECK(input.is_cuda(), "QVQ P32 Ampere input must be CUDA");
  TORCH_CHECK(
      trellis.device() == input.device() && levels.device() == input.device() &&
          bank_ids.device() == input.device(),
      "QVQ P32 Ampere tensors must share one CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf && levels.scalar_type() == at::kHalf,
      "QVQ P32 Ampere input and levels must be float16");
  TORCH_CHECK(trellis.scalar_type() == at::kInt, "QVQ P32 Ampere trellis must be int32");
  TORCH_CHECK(bank_ids.scalar_type() == at::kByte, "QVQ P32 Ampere bank ids must be uint8");
  TORCH_CHECK(
      input.is_contiguous() && trellis.is_contiguous() && levels.is_contiguous() &&
          bank_ids.is_contiguous(),
      "QVQ P32 Ampere tensors must be contiguous");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= kRows,
      "QVQ P32 Ampere input must have between 1 and 16 rows");
  TORCH_CHECK(
      input.size(1) > 0 && input.size(1) % kTileRows == 0,
      "QVQ P32 Ampere K must be positive and divisible by 16");
  TORCH_CHECK(
      out_features > 0 && out_features % kTileColumns == 0,
      "QVQ P32 Ampere N must be positive and divisible by 16");
  TORCH_CHECK(split_count >= 1 && split_count <= 128, "QVQ P32 Ampere split count must be in [1, 128]");
  TORCH_CHECK(bank_alt_id >= 0 && bank_alt_id <= 3, "QVQ P32 Ampere bank ID must be in [0, 3]");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const int capability = cached_device_capability(input.get_device());
  TORCH_CHECK(
      capability == 80,
      "QVQ P32 Ampere WMMA requires compute capability 8.0, got ",
      capability / 10,
      ".",
      capability % 10);

  const int size_m = static_cast<int>(input.size(0));
  const int size_k = static_cast<int>(input.size(1));
  const int size_n = static_cast<int>(out_features);
  const int k_tiles = size_k / kTileRows;
  const int n_tiles = size_n / kTileColumns;
  TORCH_CHECK(split_count <= k_tiles, "QVQ P32 Ampere split count cannot exceed the K16 tile count");
  const int64_t expected_tiles = static_cast<int64_t>(k_tiles) * n_tiles;
  TORCH_CHECK(
      trellis.numel() == expected_tiles * kWordsPerTile,
      "QVQ P32 Ampere trellis has the wrong word count");
  TORCH_CHECK(bank_ids.numel() == expected_tiles, "QVQ P32 Ampere bank ids have the wrong length");
  TORCH_CHECK(levels.numel() == kLevels, "QVQ P32 Ampere requires 256 PGC16 levels");

  auto output = at::empty({size_m, size_n}, input.options().dtype(at::kFloat));
  auto partial_output = split_count == 1
      ? output
      : at::empty({split_count, size_m, size_n}, input.options().dtype(at::kFloat));
  // The scalar M<=4 route pays one decode loop per K16 row and reuses each
  // decoded pair across the live output rows. It wins for the common 5-6K K
  // projections; with the wider long-K wave and four-K16 scalar stage it also
  // wins for M1-M4, while larger long-K projections remain on WMMA.
  const bool use_small_m_scalar =
      (size_m <= 4 && size_k <= 6144) ||
      (size_m <= 4 && size_k == 17408 && size_n == 5120);
  const bool use_four_tile_scalar_stage =
      (size_m == 4 && size_k <= 6144) ||
      (size_m <= 4 && size_k == 17408 && size_n == 5120);
  const bool use_three_tile_scalar_stage = size_m == 2 && size_k <= 6144;
  constexpr int kM4StageKTiles = TransitionBits == 4
      ? kScalarLongStageKTiles
      : TransitionBits == 6 ? kScalarTripleStageKTiles : kStageKTiles;
  const int tiles_per_block = use_small_m_scalar ? kM1TilesPerBlock : kTilesPerBlock;
  const dim3 grid(
      static_cast<unsigned>((n_tiles + tiles_per_block - 1) / tiles_per_block),
      1,
      static_cast<unsigned>(split_count));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  const auto* input_ptr = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  const auto* trellis_ptr = reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>());
  const auto* levels_ptr = reinterpret_cast<const half*>(levels.data_ptr<at::Half>());
  const auto* bank_ids_ptr = bank_ids.data_ptr<uint8_t>();
  auto* partial_output_ptr = partial_output.data_ptr<float>();
  auto* output_ptr = output.data_ptr<float>();
  if (size_m == 1 && use_small_m_scalar &&
      (size_n == 12288 || size_n == 1024 || size_n == 10240 || size_n == 6144 ||
       size_n == 17408 ||
       (size_n == 5120 && size_k == 6144)) &&
             launch_static_n_scalar_kernel<TransitionBits, 1>(
                 input_ptr,
                 trellis_ptr,
                 levels_ptr,
                 bank_ids_ptr,
                 partial_output_ptr,
                 output_ptr,
                 size_k,
                 size_n,
                 static_cast<int>(split_count),
                 static_cast<int>(bank_alt_id),
                 grid,
                 stream)) {
  } else if (size_m == 1 && use_small_m_scalar && use_four_tile_scalar_stage) {
    p32_window_ampere_m1_kernel<
        TransitionBits, 1, kM1Threads, kM1TilesPerBlock, kScalarLongStageKTiles>
        <<<grid, kM1Threads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 1 && use_small_m_scalar) {
    p32_window_ampere_m1_kernel<TransitionBits, 1><<<grid, kM1Threads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 2 && use_small_m_scalar && use_four_tile_scalar_stage &&
             launch_static_n_scalar_kernel<
                 TransitionBits, 2, kM1Threads, kM1TilesPerBlock,
                 kScalarLongStageKTiles>(
                 input_ptr,
                 trellis_ptr,
                 levels_ptr,
                 bank_ids_ptr,
                 partial_output_ptr,
                 output_ptr,
                 size_k,
                 size_n,
                 static_cast<int>(split_count),
                 static_cast<int>(bank_alt_id),
                 grid,
                 stream)) {
  } else if (size_m == 2 && use_small_m_scalar && use_four_tile_scalar_stage) {
    p32_window_ampere_m1_kernel<
        TransitionBits, 2, kM1Threads, kM1TilesPerBlock, kScalarLongStageKTiles>
        <<<grid, kM1Threads, 0, stream>>>(
        input_ptr,
        trellis_ptr,
        levels_ptr,
        bank_ids_ptr,
        partial_output_ptr,
        output_ptr,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 2 && use_small_m_scalar && use_three_tile_scalar_stage) {
    if (!launch_static_n_scalar_kernel<
            TransitionBits, 2, kM1Threads, kM1TilesPerBlock,
            kScalarTripleStageKTiles>(
            input_ptr,
            trellis_ptr,
            levels_ptr,
            bank_ids_ptr,
            partial_output_ptr,
            output_ptr,
            size_k,
            size_n,
            static_cast<int>(split_count),
            static_cast<int>(bank_alt_id),
            grid,
            stream)) {
      p32_window_ampere_m1_kernel<
          TransitionBits, 2, kM1Threads, kM1TilesPerBlock,
          kScalarTripleStageKTiles><<<grid, kM1Threads, 0, stream>>>(
          input_ptr,
          trellis_ptr,
          levels_ptr,
          bank_ids_ptr,
          partial_output_ptr,
          output_ptr,
          size_k,
          size_n,
          static_cast<int>(split_count),
          static_cast<int>(bank_alt_id));
    }
  } else if (size_m == 2 && use_small_m_scalar) {
    p32_window_ampere_m1_kernel<TransitionBits, 2><<<grid, kM1Threads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 3 && use_small_m_scalar) {
    p32_window_ampere_m1_kernel<TransitionBits, 3><<<grid, kM1Threads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 4 && use_small_m_scalar && use_four_tile_scalar_stage &&
             launch_static_n_scalar_kernel<TransitionBits, 4, kM1Threads, kM1TilesPerBlock, kM4StageKTiles>(
                 input_ptr,
                 trellis_ptr,
                 levels_ptr,
                 bank_ids_ptr,
                 partial_output_ptr,
                 output_ptr,
                 size_k,
                 size_n,
                 static_cast<int>(split_count),
                 static_cast<int>(bank_alt_id),
                 grid,
                 stream)) {
  } else if (size_m == 4 && use_small_m_scalar && use_four_tile_scalar_stage) {
    p32_window_ampere_m1_kernel<
        TransitionBits, 4, kM1Threads, kM1TilesPerBlock, kM4StageKTiles>
        <<<grid, kM1Threads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 4 && use_small_m_scalar) {
    p32_window_ampere_m1_kernel<TransitionBits, 4><<<grid, kM1Threads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 8 && size_n == 12288) {
    p32_window_ampere_kernel<TransitionBits, false, 8, 12288, true, true><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 8 && size_n == 5120) {
    p32_window_ampere_kernel<TransitionBits, false, 8, 5120, true, true><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 8 && size_k == 5120 && size_n == 10240) {
    p32_window_ampere_kernel<TransitionBits, false, 8, 10240, true, true, 5120><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 8 && size_n == 6144) {
    p32_window_ampere_kernel<TransitionBits, false, 8, 6144, true, true><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 8 && size_k == 5120 && size_n == 1024) {
    p32_window_ampere_kernel<TransitionBits, false, 8, 1024, false, true, 5120><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 8 && size_n == 17408) {
    p32_window_ampere_kernel<TransitionBits, false, 8, 17408, false, true><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == 8) {
    p32_window_ampere_kernel<TransitionBits, false, 8><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == kRows && size_k == 5120 && size_n == 12288) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 12288, false, false, 5120><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == kRows && size_k == 6144 && size_n == 5120) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 5120, false, false, 6144>
        <<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == kRows && size_k == 17408 && size_n == 5120) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 5120, false, false, 17408>
        <<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == kRows && size_n == 5120) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 5120><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == kRows && size_k == 5120 && size_n == 10240) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 10240, false, false, 5120><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == kRows && size_k == 5120 && size_n == 6144) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 6144, false, false, 5120><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == kRows && size_k == 5120 && size_n == 1024) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 1024, false, false, 5120><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == kRows && size_n == 17408) {
    p32_window_ampere_kernel<TransitionBits, true, 0, 17408, false, false, 5120>
        <<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else if (size_m == kRows) {
    p32_window_ampere_kernel<TransitionBits, true><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else {
    p32_window_ampere_kernel<TransitionBits, false><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  }
  if (split_count == 1) {
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (split_count > 1) {
    constexpr int kReductionThreads = 256;
    const int output_values = size_m * size_n;
    const int blocks = (output_values + kReductionThreads - 1) / kReductionThreads;
    reduce_split_kernel<<<blocks, kReductionThreads, 0, stream>>>(
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        output_values,
        static_cast<int>(split_count));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

at::Tensor p32_window_ampere(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    int64_t out_features,
    int64_t bank_alt_id,
    int64_t split_count) {
  switch (transition_bits) {
    case 4:
      return p32_window_ampere_impl<4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 5:
      return p32_window_ampere_impl<5>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 6:
      return p32_window_ampere_impl<6>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 7:
      return p32_window_ampere_impl<7>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    default:
      TORCH_CHECK(false, "QVQ P32 Ampere transition bits must be in [4, 7]");
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq_ampere, m) {
  m.def("p32_window(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq_ampere, CUDA, m) {
  m.impl("p32_window", p32_window_ampere);
}
