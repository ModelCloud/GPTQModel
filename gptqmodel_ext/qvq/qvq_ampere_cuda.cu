// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ops/mm.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>

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

bool ampere_execution_capability_supported(int capability) {
  if (capability == 80) {
    return true;
  }
  // CI and development hosts may expose only an H100.  The extension embeds
  // compute_80 PTX, which the Hopper driver can JIT without changing the
  // Ampere instruction/reduction math.  Keep this opt-in so production Hopper
  // routing continues to use its dedicated SM90a kernel.
  const char* allow_sm90 = std::getenv("QVQ_AMPERE_ALLOW_SM90_VALIDATION");
  return capability == 90 && allow_sm90 != nullptr && allow_sm90[0] == '1' &&
      allow_sm90[1] == '\0';
}

__device__ __forceinline__ uint32_t pgc16_mix(uint32_t state) {
  uint32_t mixed = state ^ (state >> 8);
  mixed = (mixed * kPgc16Multiplier + kPgc16Increment) & 0xffffu;
  return mixed ^ (mixed >> 7);
}

template <int TransitionBits, bool FastBankAlt3 = false>
__device__ __forceinline__ uint32_t alternate_bank_mask(int bank_alt_id) {
  static_assert(TransitionBits >= 4 && TransitionBits <= 7);
  if constexpr (FastBankAlt3) {
    if (bank_alt_id == 3) {
      if constexpr (TransitionBits == 4 || TransitionBits == 5) {
        return 0xc3c3u;
      } else if constexpr (TransitionBits == 6) {
        return 0x3c3cu;
      } else {
        return 0x5a5au;
      }
    }
  }
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
template <
    int TransitionBits,
    bool UsePairWrapPredicate = false,
    bool UsePowerOfTwoWrap = false>
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
  constexpr int kWrappingPairs = 32 / TransitionBits;
  int first_next;
  if constexpr (UsePowerOfTwoWrap) {
    static_assert(TransitionBits == 4);
    static_assert(kWordsPerTile == 16);
    first_next = (first_word + 1) & (kWordsPerTile - 1);
  } else {
    first_next = UsePairWrapPredicate
        ? (pair < kWrappingPairs ? 0 : first_word + 1)
        : (first_word + 1 == kWordsPerTile ? 0 : first_word + 1);
  }
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
  // The default path keeps this 512-byte codebook in the read-only cache.
  // The exact grouped Flash-Next path opts into a per-CTA shared copy because
  // its repeated decode gathers otherwise generate excessive global sectors.
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

__device__ __forceinline__ void copy_async_ca_4(void* destination, const void* source) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 4;\n"
      :
      : "r"(shared_address), "l"(source));
}

__device__ __forceinline__ void copy_async_ca_8(void* destination, const void* source) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 8;\n"
      :
      : "r"(shared_address), "l"(source));
}

__device__ __forceinline__ void copy_async_ca_16(void* destination, const void* source) {
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
    bool HoistBankMasks = false,
    bool UpperRowsOnly = false,
    int StaticK = 0,
    bool UsePairWrapPredicate = false,
    bool UsePowerOfTwoWrap = false,
    bool WideNTiles = false,
    bool UseWideBankIdCopy = false,
    int StageKTiles = kStageKTiles,
    int StaticSplitCount = 0>
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
    int bank_alt_id,
    int n_block,
    int split,
    int payload_n_tiles,
    int payload_n_tile_offset,
    int output_n_offset,
    int output_stride,
    int64_t partial_segment_offset,
    int64_t partial_split_stride = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  constexpr int kWordsPerTile = 4 * TransitionBits;
  // The wide specialization follows Marlin's output-reuse principle: each
  // warp owns two adjacent N16 tiles and reuses one ldmatrix A fragment for
  // both pairs of m16n8 MMA instructions. The default remains one tile so its
  // carefully tuned decode/ldmatrix schedule and register footprint are
  // unchanged.
  constexpr int kWarpNTiles = WideNTiles ? 2 : 1;
  constexpr int kBlockNTiles = kTilesPerBlock * kWarpNTiles;
  constexpr int kKernelStageColumns = StageKTiles * kTileRows;
  __shared__ __align__(32) half input_tile[2][kRows * kKernelStageColumns];
  __shared__ __align__(16) uint32_t packed_words[2][StageKTiles][kBlockNTiles][kWordsPerTile];
  __shared__ __align__(8) uint8_t packed_bank_ids[2][StageKTiles][kBlockNTiles];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  constexpr int kStaticNTiles = StaticN > 0 ? StaticN / kTileColumns : 0;
  constexpr int kStaticKTiles = StaticK > 0 ? StaticK / kTileRows : 0;
  const int n_tiles = StaticN > 0 ? kStaticNTiles : size_n / kTileColumns;
  const int n_tile_base = n_block * kBlockNTiles;
  const bool active_tile = StaticN > 0 || n_tile_base + warp < n_tiles;
  const int k_tiles = StaticK > 0 ? kStaticKTiles : size_k / kTileRows;
  const int input_stride = StaticK > 0 ? StaticK : size_k;
  constexpr int kEffectiveStaticSplitCount =
      StaticSplitCount > 0 ? StaticSplitCount : 0;
  const int effective_split_count =
      kEffectiveStaticSplitCount > 0 ? kEffectiveStaticSplitCount : split_count;
  const int k_tile_begin = (k_tiles * split) / effective_split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / effective_split_count;
  const uint32_t alt_mask =
      alternate_bank_mask<TransitionBits, FullRows && WideNTiles>(bank_alt_id);

  auto stage = [&](int k_tile_base, int destination) {
    auto* input_vectors = reinterpret_cast<uint4*>(input_tile[destination]);
    for (int index = thread; index < kRows * kKernelStageColumns / 8; index += kThreads) {
      const int row = index / (kKernelStageColumns / 8);
      const int vector = index - row * (kKernelStageColumns / 8);
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
    constexpr int kVectorsPerKTile = kBlockNTiles * kVectorsPerTile;
    constexpr int kVectorsPerBlock = StageKTiles * kVectorsPerKTile;
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
        WideNTiles && StaticN > 0 &&
        (FullRows || (ActiveRows > 0 && UseWideBankIdCopy))) {
      if (thread < StageKTiles) {
        const int stage_k_tile = thread;
        const int k_tile = k_tile_base + stage_k_tile;
        auto* destination_ids = reinterpret_cast<uint32_t*>(
            packed_bank_ids[destination][stage_k_tile]);
        if (k_tile < k_tiles) {
          copy_async_ca_8(
              destination_ids,
              bank_ids + static_cast<int64_t>(k_tile) * payload_n_tiles +
                  payload_n_tile_offset + n_tile_base);
        } else {
          destination_ids[0] = 0u;
          destination_ids[1] = 0u;
        }
      }
    } else if constexpr (WideNTiles && StaticN > 0 && ActiveRows > 0) {
      if (thread < StageKTiles * 2) {
        const int stage_k_tile = thread >> 1;
        const int word = thread & 1;
        const int k_tile = k_tile_base + stage_k_tile;
        auto* destination_ids = reinterpret_cast<uint32_t*>(
            packed_bank_ids[destination][stage_k_tile]);
        if (k_tile < k_tiles) {
          copy_async_ca_4(
              destination_ids + word,
              reinterpret_cast<const uint32_t*>(
                  bank_ids + static_cast<int64_t>(k_tile) * payload_n_tiles +
                      payload_n_tile_offset + n_tile_base) + word);
        } else {
          destination_ids[word] = 0u;
        }
      }
    } else if constexpr (StaticN > 0 && StaticN != 1024 && FullRows) {
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
    } else if constexpr (StaticN > 0 && ActiveRows == 8) {
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
    } else if (thread < StageKTiles * kBlockNTiles) {
      const int stage_k_tile = thread / kBlockNTiles;
      const int tile = thread - stage_k_tile * kBlockNTiles;
      const int n_tile = n_tile_base + tile;
      const int k_tile = k_tile_base + stage_k_tile;
      packed_bank_ids[destination][stage_k_tile][tile] = n_tile < n_tiles && k_tile < k_tiles
          ? bank_ids[static_cast<int64_t>(k_tile) * payload_n_tiles +
              payload_n_tile_offset + n_tile]
          : 0;
    }
    __pipeline_commit();
  };

  static_assert(!UpperRowsOnly || (!FullRows && ActiveRows == 8));
  constexpr bool kUpperRowsOnly = UpperRowsOnly;
  MmaAccumulator<kUpperRowsOnly> accumulator_0[kWarpNTiles] = {};
  MmaAccumulator<kUpperRowsOnly> accumulator_1[kWarpNTiles] = {};

  stage(k_tile_begin, 0);
  int parity = 0;
  for (int k_tile = k_tile_begin; k_tile < k_tile_end; k_tile += StageKTiles) {
    const bool has_next = k_tile + StageKTiles < k_tile_end;
    if (has_next) {
      stage(k_tile + StageKTiles, parity ^ 1);
    }
    __pipeline_wait_prior(has_next ? 1 : 0);
    __syncthreads();

    if (active_tile) {
#pragma unroll
      for (int stage_k_tile = 0; stage_k_tile < StageKTiles; ++stage_k_tile) {
        if (k_tile + stage_k_tile >= k_tile_end) {
          continue;
        }
        MmaFragmentA input_fragment;
        if constexpr (WideNTiles) {
          if constexpr (kUpperRowsOnly) {
            const int address_row = lane & 7;
            const int address_column = ((lane >> 3) & 1) * 8;
            load_mma_fragment_a_upper(
                input_fragment,
                input_tile[parity] +
                    address_row * kKernelStageColumns +
                    stage_k_tile * kTileRows +
                    address_column);
          } else {
            const int address_row = (lane & 7) + ((lane >> 3) & 1) * 8;
            const int address_column = (lane >> 4) * 8;
            load_mma_fragment_a(
                input_fragment,
                input_tile[parity] +
                    address_row * kKernelStageColumns +
                    stage_k_tile * kTileRows +
                    address_column);
          }
        }
#pragma unroll
        for (int warp_n_tile = 0; warp_n_tile < kWarpNTiles; ++warp_n_tile) {
          const int shared_tile = warp + warp_n_tile * kWarps;
          const uint32_t* words = packed_words[parity][stage_k_tile][shared_tile];
          const uint8_t packed_bank_id =
              packed_bank_ids[parity][stage_k_tile][shared_tile];
          const int producer_fragment = lane >> 4;
          const int producer_pair_column = producer_fragment * 4 + ((lane >> 2) & 3);
          const int producer_row_pair = lane & 3;
          const int first_pair = producer_row_pair * 16 + producer_pair_column;
          const int second_pair = first_pair + 8;
          uint32_t state_row_0;
          uint32_t state_row_8;
          uint32_t state_row_1;
          uint32_t state_row_9;
          window_state_pair64<
              TransitionBits, UsePairWrapPredicate, UsePowerOfTwoWrap>(
              words, first_pair, state_row_0, state_row_8);
          window_state_pair64<
              TransitionBits, UsePairWrapPredicate, UsePowerOfTwoWrap>(
              words, second_pair, state_row_1, state_row_9);
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

          if constexpr (!WideNTiles) {
            if constexpr (kUpperRowsOnly) {
              const int address_row = lane & 7;
              const int address_column = ((lane >> 3) & 1) * 8;
              load_mma_fragment_a_upper(
                  input_fragment,
                  input_tile[parity] +
                      address_row * kKernelStageColumns +
                      stage_k_tile * kTileRows +
                      address_column);
            } else {
              const int address_row = (lane & 7) + ((lane >> 3) & 1) * 8;
              const int address_column = (lane >> 4) * 8;
              load_mma_fragment_a(
                  input_fragment,
                  input_tile[parity] +
                      address_row * kKernelStageColumns +
                      stage_k_tile * kTileRows +
                      address_column);
            }
          }
          mma_m16n8k16(
              input_fragment, weight_fragment_0, accumulator_0[warp_n_tile]);
          mma_m16n8k16(
              input_fragment, weight_fragment_1, accumulator_1[warp_n_tile]);
        }
      }
    }
    // Every warp must finish consuming the current stage before any lane can
    // enter the next iteration and cp.async-overwrite that buffer two stages
    // later. Independent thread scheduling makes warp-local completion alone
    // insufficient for this block-wide producer/consumer handoff.
    __syncthreads();
    parity ^= 1;
  }

  const int64_t split_stride = partial_split_stride > 0
      ? partial_split_stride
      : static_cast<int64_t>(size_m) * size_n;
  float* target = effective_split_count == 1
      ? output + output_n_offset
      : partial_output + partial_segment_offset +
          static_cast<int64_t>(split) * split_stride;
  const int target_stride = effective_split_count == 1 ? output_stride : size_n;
  if (active_tile) {
    const int output_row_0 = lane >> 2;
    const int output_row_1 = output_row_0 + 8;
#pragma unroll
    for (int warp_n_tile = 0; warp_n_tile < kWarpNTiles; ++warp_n_tile) {
      const int output_column =
          (n_tile_base + warp + warp_n_tile * kWarps) * kTileColumns +
          (lane & 3) * 2;
      if constexpr (FullRows) {
        store_output_pair<true>(
            target + static_cast<int64_t>(output_row_0) * target_stride + output_column,
            accumulator_0[warp_n_tile].values[0],
            accumulator_0[warp_n_tile].values[1]);
        store_output_pair<true>(
            target + static_cast<int64_t>(output_row_1) * target_stride + output_column,
            accumulator_0[warp_n_tile].values[2],
            accumulator_0[warp_n_tile].values[3]);
        store_output_pair<true>(
            target + static_cast<int64_t>(output_row_0) * target_stride + output_column + 8,
            accumulator_1[warp_n_tile].values[0],
            accumulator_1[warp_n_tile].values[1]);
        store_output_pair<true>(
            target + static_cast<int64_t>(output_row_1) * target_stride + output_column + 8,
            accumulator_1[warp_n_tile].values[2],
            accumulator_1[warp_n_tile].values[3]);
      } else if constexpr (ActiveRows > 0) {
        if (output_row_0 < ActiveRows) {
          store_output_pair<StaticN != 1024>(
              target + static_cast<int64_t>(output_row_0) * target_stride + output_column,
              accumulator_0[warp_n_tile].values[0],
              accumulator_0[warp_n_tile].values[1]);
          store_output_pair<StaticN != 1024>(
              target + static_cast<int64_t>(output_row_0) * target_stride + output_column + 8,
              accumulator_1[warp_n_tile].values[0],
              accumulator_1[warp_n_tile].values[1]);
        }
        if constexpr (!kUpperRowsOnly) {
          if (output_row_1 < ActiveRows) {
            store_output_pair<StaticN != 1024>(
                target + static_cast<int64_t>(output_row_1) * target_stride + output_column,
                accumulator_0[warp_n_tile].values[2],
                accumulator_0[warp_n_tile].values[3]);
            store_output_pair<StaticN != 1024>(
                target + static_cast<int64_t>(output_row_1) * target_stride + output_column + 8,
                accumulator_1[warp_n_tile].values[2],
                accumulator_1[warp_n_tile].values[3]);
          }
        }
      } else {
        if (output_row_0 < size_m) {
          store_output_pair<false>(
              target + static_cast<int64_t>(output_row_0) * target_stride + output_column,
              accumulator_0[warp_n_tile].values[0],
              accumulator_0[warp_n_tile].values[1]);
          store_output_pair<false>(
              target + static_cast<int64_t>(output_row_0) * target_stride + output_column + 8,
              accumulator_1[warp_n_tile].values[0],
              accumulator_1[warp_n_tile].values[1]);
        }
        if (output_row_1 < size_m) {
          store_output_pair<false>(
              target + static_cast<int64_t>(output_row_1) * target_stride + output_column,
              accumulator_0[warp_n_tile].values[2],
              accumulator_0[warp_n_tile].values[3]);
          store_output_pair<false>(
              target + static_cast<int64_t>(output_row_1) * target_stride + output_column + 8,
              accumulator_1[warp_n_tile].values[2],
              accumulator_1[warp_n_tile].values[3]);
        }
      }
    }
  }
#endif
}

template <
    int TransitionBits,
    bool FullRows,
    int ActiveRows = 0,
    int StaticN = 0,
    bool HoistBankMasks = false,
    bool UpperRowsOnly = false,
    int StaticK = 0,
    bool UsePairWrapPredicate = false,
    bool UsePowerOfTwoWrap = false,
    bool WideNTiles = false,
    bool UseWideBankIdCopy = false,
    int StageKTiles = kStageKTiles,
    int StaticSplitCount = 0>
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
  p32_window_ampere_kernel_body<
      TransitionBits, FullRows, ActiveRows, StaticN, HoistBankMasks,
      UpperRowsOnly, StaticK, UsePairWrapPredicate, UsePowerOfTwoWrap,
      WideNTiles, UseWideBankIdCopy, StageKTiles, StaticSplitCount>(
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

// Large-batch launches reuse the same 16-row WMMA tile across a two-dimensional
// row grid.  Keeping the row tile in the existing kernel body preserves the
// exact decode and accumulation order while avoiding one host launch per row
// chunk for prefill workloads.
template <
    int TransitionBits,
    int StaticN = 0,
    int StaticK = 0,
    int StageKTiles = kStageKTiles>
__global__ __launch_bounds__(kThreads) void p32_window_ampere_large_m_kernel(
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
  const int row_offset = static_cast<int>(blockIdx.y) * kRows;
  const int local_m = min(kRows, size_m - row_offset);
  if (local_m <= 0) {
    return;
  }
  const int n_tiles = size_n / kTileColumns;
  const int n_block = static_cast<int>(blockIdx.x);
  const int split = static_cast<int>(blockIdx.z);
#define QVQ_LARGE_M_BODY(FULL_ROWS) \
  p32_window_ampere_kernel_body< \
      TransitionBits, FULL_ROWS, 0, StaticN, false, false, StaticK, false, \
      false, false, false, StageKTiles>( \
      input + static_cast<int64_t>(row_offset) * size_k, \
      trellis, \
      levels, \
      bank_ids, \
      partial_output, \
      output + static_cast<int64_t>(row_offset) * size_n, \
      local_m, \
      size_k, \
      size_n, \
      split_count, \
      bank_alt_id, \
      n_block, \
      split, \
      n_tiles, \
      0, \
      0, \
      size_n, \
      static_cast<int64_t>(row_offset) * size_n, \
      static_cast<int64_t>(size_m) * size_n)
  if (local_m == kRows) {
    QVQ_LARGE_M_BODY(true);
  } else {
    QVQ_LARGE_M_BODY(false);
  }
#undef QVQ_LARGE_M_BODY
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
    int bank_alt_id,
    int n_block,
    int split,
    int payload_n_tiles,
    int payload_n_tile_offset,
    int output_n_offset,
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
  constexpr int kEffectiveStaticSplitCount = StaticSplitCount > 0 ? StaticSplitCount : 0;
  const int effective_split_count =
      kEffectiveStaticSplitCount > 0 ? kEffectiveStaticSplitCount : split_count;
  const int block_n_tile_base = n_block * TilesPerBlock;
  const int n_tile_base = block_n_tile_base + warp * 4;
  constexpr int kStaticKTiles = StaticK > 0 ? StaticK / kTileRows : 0;
  const int k_tiles = StaticK > 0 ? kStaticKTiles : size_k / kTileRows;
  const int input_stride = StaticK > 0 ? StaticK : size_k;
  const int k_tile_begin = (k_tiles * split) / effective_split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / effective_split_count;
  const uint32_t alt_mask = alternate_bank_mask<TransitionBits>(bank_alt_id);
  const half* decode_levels = levels;
  if constexpr (UseSharedLevels) {
    for (int level = thread; level < 256; level += Threads) {
      shared_levels[level] = levels[level];
    }
    __syncthreads();
    decode_levels = shared_levels;
  }
  constexpr bool kUsePairWrapPredicate =
      (Rows == 2 &&
       ((StaticN == 1024 && TransitionBits == 6) ||
        (StaticN == 17408 && TransitionBits == 6) ||
        (StaticN == 5120 && StageKTiles == kScalarLongStageKTiles &&
         TransitionBits >= 5))) ||
      (Rows == 4 &&
       ((StaticN == 12288 &&
         (TransitionBits == 5 || TransitionBits == 6)) ||
        (StaticN == 10240 && TransitionBits == 5) ||
        (StaticN == 6144 && TransitionBits == 6) ||
        (StaticN == 17408 && TransitionBits == 5))) ||
      (Rows == 1 &&
       ((StaticN == 1024 && TransitionBits != 7) ||
        (TransitionBits == 6 &&
         !(StaticN == 5120 && StageKTiles == kScalarLongStageKTiles)) ||
        (StaticN == 5120 && StageKTiles != kScalarLongStageKTiles))) ||
      (StaticN == 5120 && StageKTiles != kScalarLongStageKTiles &&
       ((Rows == 2 && TransitionBits == 4) ||
        (Rows == 4 && TransitionBits == 6)));
  constexpr bool kUsePowerOfTwoWrap =
      TransitionBits == 4 &&
      ((Rows == 2 && StaticN == 1024) ||
       (Rows == 4 &&
        StaticN == 5120 && StageKTiles == kScalarLongStageKTiles));

  auto stage = [&](int k_tile_base, int destination) {
    auto* input_vectors = reinterpret_cast<uint4*>(input_tile[destination]);
    constexpr int kInputVectorsPerRow = StageKTiles * kTileRows / 8;
    for (int index = thread; index < Rows * kInputVectorsPerRow; index += Threads) {
      const int row = index / kInputVectorsPerRow;
      const int vector = index - row * kInputVectorsPerRow;
      const int source_column = k_tile_base * kTileRows + vector * 8;
      if (source_column < input_stride) {
        const half* source = input + static_cast<int64_t>(row) * input_stride + source_column;
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
        StaticN == 1024 && TilesPerBlock == 16 && Rows == 1) {
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
             (TransitionBits == 6 && (StaticN == 5120 || StaticN == 17408)))) {
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
            k_tile < k_tiles
                ? bank_ids[static_cast<int64_t>(k_tile) * payload_n_tiles +
                    payload_n_tile_offset + n_tile]
                : 0;
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
          if constexpr (
              Rows == 2 &&
              !(TransitionBits == 4 &&
                (StaticN == 6144 ||
                 (StaticN == 5120 && StageKTiles != kScalarLongStageKTiles)))) {
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
                window_state_pair64<
                    TransitionBits, kUsePairWrapPredicate, kUsePowerOfTwoWrap>(
                    words, pair, state_0[row_in_group], state_8[row_in_group]);
              }
              uint32_t decoded_0[2];
              uint32_t decoded_8[2];
              decode_state_pair_bits<UseSharedLevels>(
                  state_0[0], state_0[1], bank_mask_0, decode_levels,
                  decoded_0[0], decoded_0[1]);
              decode_state_pair_bits<UseSharedLevels>(
                  state_8[0], state_8[1], bank_mask_8, decode_levels,
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
                  const int row_base = output_row * (StageKTiles * kTileRows) +
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
          } else if constexpr (
              StaticN != 1024 &&
              ((Rows == 4 && TransitionBits != 7) ||
               Rows == 1)) {
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
                window_state_pair64<
                    TransitionBits, kUsePairWrapPredicate, kUsePowerOfTwoWrap>(
                    words, pair, state_0[row_in_group], state_8[row_in_group]);
              }
              uint32_t decoded_0[2];
              uint32_t decoded_8[2];
              decode_state_pair_bits<UseSharedLevels>(
                  state_0[0], state_0[1], bank_mask_0, decode_levels,
                  decoded_0[0], decoded_0[1]);
              decode_state_pair_bits<UseSharedLevels>(
                  state_8[0], state_8[1], bank_mask_8, decode_levels,
                  decoded_8[0], decoded_8[1]);
#pragma unroll
              for (int row_in_group = 0; row_in_group < 2; ++row_in_group) {
                const int row = bank_group * 2 + row_in_group;
                union {
                  uint32_t bits;
                  half2 values;
                } pair_0{decoded_0[row_in_group]}, pair_8{decoded_8[row_in_group]};
#pragma unroll
                for (int output_row = 0; output_row < Rows; ++output_row) {
                  const int row_base = output_row * (StageKTiles * kTileRows) +
                      stage_k_tile * kTileRows;
                  const float input_0 =
                      __half2float(input_tile[parity][row_base + row]);
                  const float input_8 =
                      __half2float(input_tile[parity][row_base + row + 8]);
                  accumulator_0[output_row] = fmaf(
                      input_0, __half2float(pair_0.values.x), accumulator_0[output_row]);
                  accumulator_1[output_row] = fmaf(
                      input_0, __half2float(pair_0.values.y), accumulator_1[output_row]);
                  accumulator_0[output_row] = fmaf(
                      input_8, __half2float(pair_8.values.x), accumulator_0[output_row]);
                  accumulator_1[output_row] = fmaf(
                      input_8, __half2float(pair_8.values.y), accumulator_1[output_row]);
                }
              }
            }
          } else {
#pragma unroll
            for (int row = 0; row < 8; ++row) {
              const int pair = row * 8 + pair_column;
              uint32_t state_0;
              uint32_t state_8;
              window_state_pair64<
                  TransitionBits, kUsePairWrapPredicate, kUsePowerOfTwoWrap>(
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
                const int row_base = output_row * (StageKTiles * kTileRows) +
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
        if (k_tile + stage_k_tile >= k_tile_end) {
          continue;
        }
        const uint32_t* words = packed_words[parity][stage_k_tile][shared_tile];
        const uint8_t packed_bank_id =
            packed_bank_ids[parity][stage_k_tile][shared_tile];
        // Dynamic grouped plans still have the same M=4 row-pair structure as
        // the static M=4 routes.  Hoist the bank masks and decode two rows
        // together so Flash-Next QKV/gate-up tiles do not pay one mask
        // selection per decoded pair.
        if constexpr (Rows == 4 && TransitionBits != 7) {
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
              window_state_pair64<
                  TransitionBits, kUsePairWrapPredicate, kUsePowerOfTwoWrap>(
                  words, pair, state_0[row_in_group], state_8[row_in_group]);
            }
            uint32_t decoded_0[2];
            uint32_t decoded_8[2];
            decode_state_pair_bits<UseSharedLevels>(
                state_0[0], state_0[1], bank_mask_0, decode_levels,
                decoded_0[0], decoded_0[1]);
            decode_state_pair_bits<UseSharedLevels>(
                state_8[0], state_8[1], bank_mask_8, decode_levels,
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
                const int row_base = output_row * (StageKTiles * kTileRows) +
                    stage_k_tile * kTileRows;
                const float input_0 =
                    __half2float(input_tile[parity][row_base + row]);
                const float input_8 =
                    __half2float(input_tile[parity][row_base + row + 8]);
                accumulator_0[output_row] = fmaf(
                    input_0, weight_0, accumulator_0[output_row]);
                accumulator_1[output_row] = fmaf(
                    input_0, weight_1, accumulator_1[output_row]);
                accumulator_0[output_row] = fmaf(
                    input_8, weight_8, accumulator_0[output_row]);
                accumulator_1[output_row] = fmaf(
                    input_8, weight_9, accumulator_1[output_row]);
              }
            }
          }
        } else if constexpr (
            Rows == 2 && StaticN != 1024 &&
            (TransitionBits == 5 || TransitionBits == 7)) {
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
              window_state_pair64<
                  TransitionBits, kUsePairWrapPredicate, kUsePowerOfTwoWrap>(
                  words, pair, state_0, state_8);
              const uint32_t decoded_0 =
                  decode_state_bits<UseSharedLevels>(
                      state_0, bank_mask_0, decode_levels);
              const uint32_t decoded_8 =
                  decode_state_bits<UseSharedLevels>(
                      state_8, bank_mask_8, decode_levels);
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
                const int row_base = output_row * (StageKTiles * kTileRows) +
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
        } else {
#pragma unroll
          for (int row = 0; row < 8; ++row) {
            const int pair = row * 8 + pair_column;
            uint32_t state_0;
            uint32_t state_8;
            window_state_pair64<
                TransitionBits, kUsePairWrapPredicate, kUsePowerOfTwoWrap>(
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
              const int row_base = output_row * (StageKTiles * kTileRows) +
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
    int StaticK = 0>
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
  p32_window_ampere_m1_kernel_body<
      TransitionBits, Rows, Threads, TilesPerBlock, StageKTiles, StaticN,
      StaticSplitCount, StaticK>(
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

constexpr int kMaxGroupedP32Segments = 3;

struct GroupedP32LaunchParams {
  int segment_count;
  int n_tile_start[kMaxGroupedP32Segments];
  int n_tiles[kMaxGroupedP32Segments];
  int bank_alt_id[kMaxGroupedP32Segments];
  int split_count[kMaxGroupedP32Segments];
  int rank8_offset[kMaxGroupedP32Segments];
  float rank8_scale[kMaxGroupedP32Segments];
  int64_t output_offset[kMaxGroupedP32Segments];
  int64_t partial_offset[kMaxGroupedP32Segments];
};

template <int TilesPerBlock>
__device__ __forceinline__ bool grouped_block_coordinates(
    const GroupedP32LaunchParams& params,
    int segment_count,
    int linear_block,
    int& segment,
    int& n_block,
    int& split) {
  // The old grouped launch used a rectangular 3-D grid whose x/z extents
  // came from the widest child and largest split wave.  Flash-Next Q/K/V
  // groups have one wide child and two 512-column children, so most of that
  // rectangle is an empty CTA.  Flatten only the valid (segment, split,
  // n-block) tuples; this does not change partial-output ownership or the
  // increasing-split reduction order.
  for (int candidate = 0; candidate < segment_count; ++candidate) {
    const int n_blocks =
        (params.n_tiles[candidate] + TilesPerBlock - 1) / TilesPerBlock;
    const int segment_blocks = n_blocks * params.split_count[candidate];
    if (linear_block < segment_blocks) {
      segment = candidate;
      split = linear_block / n_blocks;
      n_block = linear_block - split * n_blocks;
      return true;
    }
    linear_block -= segment_blocks;
  }
  return false;
}

template <
    int TransitionBits,
    int Rows,
    int StageKTiles = kStageKTiles,
    bool UseSharedLevels = false>
__global__ __launch_bounds__(kM1Threads) void p32_window_ampere_grouped_scalar_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const half* __restrict__ levels,
    const uint8_t* __restrict__ bank_ids,
    const __grid_constant__ GroupedP32LaunchParams params,
    float* __restrict__ partial_output,
    float* __restrict__ output,
    int segment_count,
    int size_k,
    int total_n_tiles) {
  int segment = 0;
  int n_block = 0;
  int split = 0;
  if (!grouped_block_coordinates<kM1TilesPerBlock>(
          params, std::min(segment_count, params.segment_count),
          static_cast<int>(blockIdx.x), segment, n_block, split)) {
    return;
  }
  const int n_tile_start = params.n_tile_start[segment];
  const int n_tiles = params.n_tiles[segment];
  const int split_count = params.split_count[segment];
  p32_window_ampere_m1_kernel_body<
      TransitionBits, Rows, kM1Threads, kM1TilesPerBlock, StageKTiles, 0, 0, 0,
      UseSharedLevels>(
      input,
      trellis,
      levels,
      bank_ids,
      partial_output,
      output,
      size_k,
      n_tiles * kTileColumns,
      split_count,
      params.bank_alt_id[segment],
      n_block,
      split,
      total_n_tiles,
      n_tile_start,
      static_cast<int>(params.output_offset[segment]),
      n_tiles * kTileColumns,
      params.partial_offset[segment]);
}

template <
    int TransitionBits,
    bool FullRows,
    int ActiveRows = 0,
    int StaticN = 0,
    bool HoistBankMasks = false,
    bool UpperRowsOnly = false,
    int StaticK = 0,
    bool UsePairWrapPredicate = false,
    bool UsePowerOfTwoWrap = false,
    bool WideNTiles = false,
    bool UseWideBankIdCopy = false,
    int StageKTiles = kStageKTiles,
    int StaticSplitCount = 0>
__global__ __launch_bounds__(kThreads) void p32_window_ampere_grouped_wmma_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const half* __restrict__ levels,
    const uint8_t* __restrict__ bank_ids,
    GroupedP32LaunchParams params,
    float* __restrict__ partial_output,
    float* __restrict__ output,
    int segment_count,
    int size_m,
    int size_k,
    int total_n_tiles) {
  int segment = 0;
  int n_block = 0;
  int split = 0;
  if (!grouped_block_coordinates<WideNTiles ? 2 * kTilesPerBlock : kTilesPerBlock>(
          params, std::min(segment_count, params.segment_count),
          static_cast<int>(blockIdx.x), segment, n_block, split)) {
    return;
  }
  const int n_tile_start = params.n_tile_start[segment];
  const int n_tiles = params.n_tiles[segment];
  const int split_count = params.split_count[segment];
  p32_window_ampere_kernel_body<
      TransitionBits, FullRows, ActiveRows, StaticN, HoistBankMasks,
      UpperRowsOnly, StaticK, UsePairWrapPredicate, UsePowerOfTwoWrap,
      WideNTiles, UseWideBankIdCopy, StageKTiles, StaticSplitCount>(
      input,
      trellis,
      levels,
      bank_ids,
      partial_output,
      output,
      size_m,
      size_k,
      n_tiles * kTileColumns,
      split_count,
      params.bank_alt_id[segment],
      n_block,
      split,
      total_n_tiles,
      n_tile_start,
      static_cast<int>(params.output_offset[segment]),
      n_tiles * kTileColumns,
      params.partial_offset[segment]);
}

template <
    int TransitionBits,
    int Rows,
    int Threads = kM1Threads,
    int TilesPerBlock = kM1TilesPerBlock,
    int StageKTiles = kStageKTiles,
    int StaticSplitCount = 0,
    int StaticK = 0>
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
    p32_window_ampere_m1_kernel<TransitionBits, Rows, Threads, TilesPerBlock, StageKTiles, N, StaticSplitCount, StaticK> \
        <<<grid, Threads, 0, stream>>>(                                               \
            input, trellis, levels, bank_ids, partial_output, output, size_k, size_n, \
            split_count, bank_alt_id);                                                \
    return true
  switch (size_n) {
    QVQ_LAUNCH_STATIC_N(1024);
    QVQ_LAUNCH_STATIC_N(2560);
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

template <int StaticSplitCount = 0>
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
#pragma unroll
  for (int split = 0;
       split < (StaticSplitCount > 0 ? StaticSplitCount : split_count);
       ++split) {
    accumulator += partial_output[static_cast<int64_t>(split) * output_values + index];
  }
  output[index] = accumulator;
}

template <bool Rank8BFloat>
__global__ void reduce_split_rank8_kernel(
    const float* __restrict__ partial_output,
    float* __restrict__ output,
    const float* __restrict__ rank8_down,
    const void* __restrict__ rank8_b,
    int output_values,
    int size_n,
    int split_count,
    int rank8_down_stride,
    float rank8_scale) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= output_values) {
    return;
  }
  float accumulator = 0.0f;
#pragma unroll
  for (int split = 0; split < split_count; ++split) {
    accumulator += partial_output[static_cast<int64_t>(split) * output_values + index];
  }
  const int row = index / size_n;
  const int column = index - row * size_n;
  const float* rank8_b_float = static_cast<const float*>(rank8_b);
  const half* rank8_b_half = static_cast<const half*>(rank8_b);
  float correction = 0.0f;
#pragma unroll
  for (int rank = 0; rank < 8; ++rank) {
    const float b_value = Rank8BFloat
        ? rank8_b_float[rank * size_n + column]
        : __half2float(rank8_b_half[rank * size_n + column]);
    correction += rank8_down[row * rank8_down_stride + rank] * b_value;
  }
  output[index] = accumulator + rank8_scale * correction;
}

__global__ void reduce_grouped_split_kernel(
    const float* __restrict__ partial_output,
    float* __restrict__ output,
    GroupedP32LaunchParams params,
    int segment_count,
    int size_m,
    int total_n) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int output_values = size_m * total_n;
  if (index >= output_values) {
    return;
  }
  int segment = 0;
  for (; segment < segment_count; ++segment) {
    const int64_t start = params.output_offset[segment];
    const int64_t count = static_cast<int64_t>(size_m) *
        params.n_tiles[segment] * kTileColumns;
    if (index >= start && index < start + count) {
      break;
    }
  }
  if (segment == segment_count) {
    return;
  }
  const int split_count = params.split_count[segment];
  if (split_count == 1) {
    return;
  }
  const int segment_n = params.n_tiles[segment] * kTileColumns;
  const int64_t local_index = index - params.output_offset[segment];
  const int64_t split_stride = static_cast<int64_t>(size_m) * segment_n;
  const float* segment_partials =
      partial_output + params.partial_offset[segment];
  float accumulator = 0.0f;
  // Match reduce_split_kernel exactly: increasing split index and one FP32
  // addition per partial.  This is the native exactness contract.
  for (int split = 0; split < split_count; ++split) {
    accumulator += segment_partials[static_cast<int64_t>(split) * split_stride +
        local_index];
  }
  output[index] = accumulator;
}

template <bool Rank8BFloat, int StaticSplitCount = 0, bool SingleRow = false>
__global__ void reduce_grouped_split_rank8_kernel(
    const float* __restrict__ partial_output,
    float* __restrict__ output,
    const __grid_constant__ GroupedP32LaunchParams params,
    int segment_count,
    int size_m,
    int total_n,
    const float* __restrict__ rank8_down,
    const void* __restrict__ rank8_b) {
  const int block = static_cast<int>(blockIdx.x);
  int segment = 0;
  int block_offset = 0;
  for (; segment < segment_count; ++segment) {
    const int segment_values = size_m * params.n_tiles[segment] * kTileColumns;
    const int segment_blocks =
        (segment_values + blockDim.x - 1) / blockDim.x;
    if (block < block_offset + segment_blocks) {
      break;
    }
    block_offset += segment_blocks;
  }
  if (segment == segment_count) {
    return;
  }
  const int segment_n = params.n_tiles[segment] * kTileColumns;
  const int64_t local_index = static_cast<int64_t>(block - block_offset) *
      blockDim.x + threadIdx.x;
  if (local_index >= static_cast<int64_t>(size_m) * segment_n) {
    return;
  }
  const int row = SingleRow ? 0 : static_cast<int>(local_index / segment_n);
  const int column = static_cast<int>(local_index -
      static_cast<int64_t>(row) * segment_n);
  const int global_column = params.n_tile_start[segment] * kTileColumns + column;
  const int split_count = params.split_count[segment];
  const int64_t split_stride = static_cast<int64_t>(size_m) * segment_n;
  const int64_t output_index = params.output_offset[segment] + local_index;
  float accumulator = split_count == 1 ? output[output_index] : 0.0f;
  if (split_count > 1) {
    const float* segment_partials =
        partial_output + params.partial_offset[segment];
#pragma unroll
    for (int split = 0;
         split < (StaticSplitCount > 0 ? StaticSplitCount : split_count);
         ++split) {
      accumulator += segment_partials[
          static_cast<int64_t>(split) * split_stride + local_index];
    }
  }
  const float* rank8_b_float = static_cast<const float*>(rank8_b);
  const half* rank8_b_half = static_cast<const half*>(rank8_b);
  const int rank8_base = row * segment_count * 8 + params.rank8_offset[segment];
  float correction = 0.0f;
#pragma unroll
  for (int rank = 0; rank < 8; ++rank) {
    const float b_value = Rank8BFloat
        ? rank8_b_float[rank * total_n + global_column]
        : __half2float(rank8_b_half[rank * total_n + global_column]);
    correction += rank8_down[rank8_base + rank] * b_value;
  }
  output[output_index] = accumulator + params.rank8_scale[segment] * correction;
}

template <int StaticSplitCount, int OutputsPerWarp = 4>
__global__ void reduce_split_warp_kernel(
    const float* __restrict__ partial_output,
    float* __restrict__ output,
    int output_values) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  constexpr int kOutputsPerWarp = OutputsPerWarp;
  static_assert(kOutputsPerWarp > 0 && kOutputsPerWarp <= 16);
  static_assert((kOutputsPerWarp & (kOutputsPerWarp - 1)) == 0);
  constexpr int kSplitLanes = 32 / kOutputsPerWarp;
  const int index =
      (static_cast<int>(blockIdx.x) * warps_per_block + warp) *
          kOutputsPerWarp +
      (lane & (kOutputsPerWarp - 1));
  if (index >= output_values) {
    return;
  }
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
  if (lane < kOutputsPerWarp) {
    output[index] = accumulator;
  }
}

template <int TransitionBits>
at::Tensor p32_window_ampere_impl(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t out_features,
    int64_t bank_alt_id,
    int64_t split_count,
    const c10::optional<at::Tensor>& rank8_a = c10::nullopt,
    const c10::optional<at::Tensor>& rank8_b = c10::nullopt,
    double rank8_scale = 1.0,
    const c10::optional<at::Tensor>& rank8_down = c10::nullopt) {
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
      input.dim() == 2 && input.size(0) >= 1,
      "QVQ P32 Ampere input must have at least one row");
  TORCH_CHECK(
      input.size(1) > 0 && input.size(1) % kTileRows == 0,
      "QVQ P32 Ampere K must be positive and divisible by 16");
  TORCH_CHECK(
      out_features > 0 && out_features % kTileColumns == 0,
      "QVQ P32 Ampere N must be positive and divisible by 16");
  TORCH_CHECK(split_count >= 1 && split_count <= 128, "QVQ P32 Ampere split count must be in [1, 128]");
  TORCH_CHECK(bank_alt_id >= 0 && bank_alt_id <= 3, "QVQ P32 Ampere bank ID must be in [0, 3]");
  TORCH_CHECK(std::isfinite(rank8_scale), "QVQ P32 Ampere rank8 scale must be finite");
  TORCH_CHECK(
      !rank8_down.has_value() || rank8_b.has_value(),
      "rank8_down requires rank8_b");
  TORCH_CHECK(
      rank8_down.has_value() || rank8_a.has_value() == rank8_b.has_value(),
      "rank8_a and rank8_b must be provided together");
  TORCH_CHECK(
      !(rank8_a.has_value() && rank8_down.has_value()),
      "rank8_a and rank8_down are mutually exclusive");
  if (rank8_a.has_value()) {
    const at::Tensor& projection_a = *rank8_a;
    const at::Tensor& projection_b = *rank8_b;
    TORCH_CHECK(
        projection_a.device() == input.device() &&
            projection_b.device() == input.device(),
        "QVQ P32 Ampere rank8 tensors must share the input CUDA device");
    TORCH_CHECK(
        projection_a.scalar_type() == at::kHalf &&
            (projection_b.scalar_type() == at::kHalf ||
             projection_b.scalar_type() == at::kFloat),
        "QVQ P32 Ampere rank8_a must be float16 and rank8_b must be float16 or float32");
    TORCH_CHECK(
        projection_a.is_contiguous() && projection_b.is_contiguous(),
        "QVQ P32 Ampere rank8 tensors must be contiguous");
    TORCH_CHECK(
        projection_a.dim() == 2 && projection_a.size(0) == input.size(1) &&
            projection_a.size(1) == 8,
        "QVQ P32 Ampere rank8_a must have shape [K, 8]");
    TORCH_CHECK(
        projection_b.dim() == 2 && projection_b.size(0) == 8 &&
            projection_b.size(1) == out_features,
        "QVQ P32 Ampere rank8_b must have shape [8, N]");
  }
  if (rank8_down.has_value()) {
    TORCH_CHECK(
        rank8_down->device() == input.device() &&
            rank8_down->scalar_type() == at::kFloat &&
            rank8_down->dim() == 2 && rank8_down->size(0) == input.size(0) &&
            rank8_down->size(1) == 8 && rank8_down->stride(1) == 1 &&
            rank8_down->stride(0) >= 8,
        "QVQ P32 Ampere rank8_down must be FP32 [M, 8] with unit column stride");
  }

  const c10::cuda::CUDAGuard device_guard(input.device());
  const int capability = cached_device_capability(input.get_device());
  TORCH_CHECK(
      ampere_execution_capability_supported(capability),
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
  if (size_m > kRows) {
    const dim3 large_grid(
        static_cast<unsigned>((n_tiles + kTilesPerBlock - 1) / kTilesPerBlock),
        static_cast<unsigned>((size_m + kRows - 1) / kRows),
        static_cast<unsigned>(split_count));
    if (size_k == 5120 && size_n == 1024) {
      p32_window_ampere_large_m_kernel<TransitionBits, 1024, 5120, 2>
          <<<large_grid, kThreads, 0, stream>>>(
          input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, partial_output_ptr,
          output_ptr, size_m, size_k, size_n, static_cast<int>(split_count),
          static_cast<int>(bank_alt_id));
    } else if (size_k == 5120 && size_n == 12288) {
      p32_window_ampere_large_m_kernel<TransitionBits, 12288, 5120, 3>
          <<<large_grid, kThreads, 0, stream>>>(
          input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, partial_output_ptr,
          output_ptr, size_m, size_k, size_n, static_cast<int>(split_count),
          static_cast<int>(bank_alt_id));
    } else if (size_k == 6144 && size_n == 5120) {
      p32_window_ampere_large_m_kernel<TransitionBits, 5120, 6144, 3>
          <<<large_grid, kThreads, 0, stream>>>(
          input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, partial_output_ptr,
          output_ptr, size_m, size_k, size_n, static_cast<int>(split_count),
          static_cast<int>(bank_alt_id));
    } else if (size_k == 5120 && size_n == 10240) {
      p32_window_ampere_large_m_kernel<TransitionBits, 10240, 5120, 3>
          <<<large_grid, kThreads, 0, stream>>>(
          input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, partial_output_ptr,
          output_ptr, size_m, size_k, size_n, static_cast<int>(split_count),
          static_cast<int>(bank_alt_id));
    } else if (size_k == 5120 && size_n == 6144) {
      p32_window_ampere_large_m_kernel<TransitionBits, 6144, 5120, 3>
          <<<large_grid, kThreads, 0, stream>>>(
          input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, partial_output_ptr,
          output_ptr, size_m, size_k, size_n, static_cast<int>(split_count),
          static_cast<int>(bank_alt_id));
    } else if (size_k == 5120 && size_n == 17408) {
      p32_window_ampere_large_m_kernel<TransitionBits, 17408, 5120, 3>
          <<<large_grid, kThreads, 0, stream>>>(
          input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, partial_output_ptr,
          output_ptr, size_m, size_k, size_n, static_cast<int>(split_count),
          static_cast<int>(bank_alt_id));
    } else if (size_k == 17408 && size_n == 5120) {
      p32_window_ampere_large_m_kernel<TransitionBits, 5120, 17408, 3>
          <<<large_grid, kThreads, 0, stream>>>(
          input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, partial_output_ptr,
          output_ptr, size_m, size_k, size_n, static_cast<int>(split_count),
          static_cast<int>(bank_alt_id));
    } else {
      p32_window_ampere_large_m_kernel<TransitionBits>
          <<<large_grid, kThreads, 0, stream>>>(
          input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, partial_output_ptr,
          output_ptr, size_m, size_k, size_n, static_cast<int>(split_count),
          static_cast<int>(bank_alt_id));
    }
  } else if (size_m == 1 && size_k == 5120 && size_n == 12288 &&
      split_count == 40 &&
      launch_static_n_scalar_kernel<
          TransitionBits, 1, kM1Threads, kM1TilesPerBlock,
          kScalarTripleStageKTiles, 40, 5120>(
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
  } else if (size_m == 1 && size_k == 5120 && size_n == 12288 &&
      launch_static_n_scalar_kernel<
          TransitionBits, 1, kM1Threads, kM1TilesPerBlock,
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
  } else if (size_m == 1 && size_k == 17408 && size_n == 5120 &&
      split_count == 128 &&
      launch_static_n_scalar_kernel<
          TransitionBits, 1, kM1Threads, kM1TilesPerBlock,
          kScalarTripleStageKTiles, 128>(
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
  } else if (size_m == 1 && size_k == 5120 && size_n == 1024 &&
      split_count == 56 &&
      launch_static_n_scalar_kernel<
          TransitionBits, 1, kM1Threads, kM1TilesPerBlock,
          kStageKTiles, 56>(
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
  } else if (size_m == 1 && size_k == 5120 && size_n == 17408 &&
      split_count == 40 &&
      launch_static_n_scalar_kernel<
          TransitionBits, 1, kM1Threads, kM1TilesPerBlock,
          kStageKTiles, 40>(
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
  } else if (size_m == 1 && use_small_m_scalar &&
      (size_n == 12288 || size_n == 1024 || size_n == 10240 || size_n == 6144 ||
       size_n == 17408 || size_n == 5120) &&
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
  } else if (size_m == 1 && size_k == 640 && size_n == 2560 &&
             split_count == 24) {
    p32_window_ampere_m1_kernel<
        TransitionBits, 1, kM1Threads, kM1TilesPerBlock, kStageKTiles,
        2560, 24, 640><<<grid, kM1Threads, 0, stream>>>(
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
  } else if (size_m == 2 && size_k == 5120 && size_n == 12288 &&
             ((TransitionBits <= 5 && split_count == 64) ||
              (TransitionBits >= 6 && split_count == 40)) &&
             launch_static_n_scalar_kernel<
                 TransitionBits, 2, kM1Threads, kM1TilesPerBlock,
                 kScalarTripleStageKTiles,
                 TransitionBits <= 5 ? 64 : 40>(
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
  } else if (size_m == 2 && size_k == 5120 && size_n == 10240 &&
             split_count == 40 &&
             launch_static_n_scalar_kernel<
                 TransitionBits, 2, kM1Threads, kM1TilesPerBlock,
                 kScalarTripleStageKTiles, 40>(
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
  } else if (size_m == 2 && size_k == 5120 && size_n == 17408 &&
             split_count == 40 &&
             launch_static_n_scalar_kernel<
                 TransitionBits, 2, kM1Threads, kM1TilesPerBlock,
                 kScalarTripleStageKTiles, 40>(
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
  } else if (size_m == 2 && size_k == 640 && size_n == 2560 &&
             split_count == 40) {
    p32_window_ampere_m1_kernel<
        TransitionBits, 2, kM1Threads, kM1TilesPerBlock,
        kScalarTripleStageKTiles, 2560, 40, 640><<<grid, kM1Threads, 0, stream>>>(
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
  } else if (size_m == 4 && size_k == 5120 && size_n == 12288 &&
             split_count == 40 &&
             launch_static_n_scalar_kernel<
                 TransitionBits, 4, kM1Threads, kM1TilesPerBlock,
                 kM4StageKTiles, 40>(
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
  } else if (size_m == 4 && size_k == 5120 && size_n == 10240 &&
             split_count == 40 &&
             launch_static_n_scalar_kernel<
                 TransitionBits, 4, kM1Threads, kM1TilesPerBlock,
                 kM4StageKTiles, 40>(
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
  } else if (size_m == 4 && size_k == 6144 && size_n == 5120 &&
             split_count == 48 &&
             launch_static_n_scalar_kernel<
                 TransitionBits, 4, kM1Threads, kM1TilesPerBlock,
                 kM4StageKTiles, 48>(
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
  } else if (size_m == 4 && size_k == 640 && size_n == 2560 &&
             split_count == 40) {
    p32_window_ampere_m1_kernel<
        TransitionBits, 4, kM1Threads, kM1TilesPerBlock,
        kM4StageKTiles, 2560, 40, 640><<<grid, kM1Threads, 0, stream>>>(
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
  } else if (
      size_m == 8 && size_n == 12288 && TransitionBits >= 6) {
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 12288, true, true, 0, true, false, true,
        false, 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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
  } else if (size_m == 8 && size_n == 12288) {
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 12288, true, true, 0, false,
        TransitionBits == 4, true, false, 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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
  } else if (
      size_m == 8 && size_k == 6144 && size_n == 5120 &&
      (TransitionBits == 6 || TransitionBits == 7)) {
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 5120, true, true, 0, true, false, true,
        false, 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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
  } else if (size_m == 8 && size_k == 6144 && size_n == 5120 &&
             split_count == 24) {
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 5120, true, true, 0, true, false, true,
        false, 3, 24>
        <<<wide_grid, kThreads, 0, stream>>>(
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
  } else if (
      size_m == 8 && size_k == 6144 && size_n == 5120) {
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 5120, true, true, 0, false, false, true,
        false, 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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
  } else if (size_m == 8 && size_k == 17408 && size_n == 5120 &&
             split_count == 40) {
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 5120, true, true, 0, true, false, true,
        TransitionBits == 7, 3, 40>
        <<<wide_grid, kThreads, 0, stream>>>(
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
  } else if (size_m == 8 && size_k == 17408 && size_n == 5120) {
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 5120, true, true, 0, true, false, true,
        TransitionBits == 7, 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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
    p32_window_ampere_kernel<TransitionBits, false, 8, 5120, true, true>
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
  } else if (size_m == 8 && size_k == 5120 && size_n == 10240) {
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 10240, true, true, 5120, true,
        TransitionBits == 4, true, false, 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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
  } else if (size_m == 8 && size_k == 5120 && size_n == 6144 &&
             split_count == 40) {
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 6144, true, true, 0, true, false, true,
        false, 3, 40>
        <<<wide_grid, kThreads, 0, stream>>>(
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
  } else if (size_m == 8 && size_k == 5120 && size_n == 6144) {
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 6144, true, true, 0, true, false, true,
        false, 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 6144, true, true, 0, true>
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
  } else if (size_m == 8 && size_k == 5120 && size_n == 1024) {
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 1024, true, true, 5120, false,
        TransitionBits == 4>
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
  } else if (
      size_m == 8 && size_n == 17408 && TransitionBits == 7) {
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 17408, true, true, 0, true, false, true,
        false, 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, false, 8, 17408, true, true, 0, true, false, true,
        false, 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, true, 0, 12288, TransitionBits == 4, false, 5120,
        (TransitionBits == 5 || TransitionBits == 6 || TransitionBits == 7),
        TransitionBits == 4, true, false, 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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
  } else if (size_m == kRows && size_k == 6144 && size_n == 5120 &&
             split_count == 12) {
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, true, 0, 5120, false, false, 6144,
        (TransitionBits == 5 || TransitionBits == 6 || TransitionBits == 7),
        false, true, false, 3, 12>
        <<<wide_grid, kThreads, 0, stream>>>(
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
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, true, 0, 5120, false, false, 6144,
        (TransitionBits == 5 || TransitionBits == 6 || TransitionBits == 7),
        false, true, false, 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, true, 0, 5120, true, false, 17408,
        (TransitionBits == 5 || TransitionBits == 6 || TransitionBits == 7),
        TransitionBits == 4, true, (TransitionBits == 7), 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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
    p32_window_ampere_kernel<
        TransitionBits, true, 0, 10240, true, false, 5120,
        (TransitionBits == 5 || TransitionBits == 6 || TransitionBits == 7),
        TransitionBits == 4, false, false, 3>
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
  } else if (size_m == kRows && size_k == 5120 && size_n == 6144) {
    p32_window_ampere_kernel<
        TransitionBits, true, 0, 6144, true, false, 5120,
        (TransitionBits == 5 || TransitionBits == 6 || TransitionBits == 7),
        TransitionBits == 4, false, false, 3><<<grid, kThreads, 0, stream>>>(
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
    const dim3 wide_grid(
        static_cast<unsigned>((n_tiles + 2 * kTilesPerBlock - 1) /
                              (2 * kTilesPerBlock)),
        1,
        static_cast<unsigned>(split_count));
    p32_window_ampere_kernel<
        TransitionBits, true, 0, 17408, false, false, 5120,
        (TransitionBits == 5 || TransitionBits == 6 || TransitionBits == 7),
        false, true, false, 3>
        <<<wide_grid, kThreads, 0, stream>>>(
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

  if (rank8_a.has_value() || rank8_down.has_value()) {
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // Compute the shared rank-8 activation once, then fold its B-side
    // product into the existing output reducer. This removes the per-call
    // FP16->FP32 B conversion and a second tiny GEMV launch while preserving
    // the quantized split reduction's increasing-split FP32 order.
    auto rank8_down_tensor = rank8_down.has_value()
        ? *rank8_down
        : at::mm(input, *rank8_a, at::kFloat);
    constexpr int kReductionThreads = 256;
    const int output_values = size_m * size_n;
    const int blocks = (output_values + kReductionThreads - 1) / kReductionThreads;
    if (rank8_b->scalar_type() == at::kFloat) {
      reduce_split_rank8_kernel<true><<<blocks, kReductionThreads, 0, stream>>>(
          partial_output.data_ptr<float>(),
          output.data_ptr<float>(),
          rank8_down_tensor.data_ptr<float>(),
          rank8_b->data_ptr(),
          output_values,
          size_n,
          static_cast<int>(split_count),
          static_cast<int>(rank8_down_tensor.stride(0)),
          static_cast<float>(rank8_scale));
    } else {
      reduce_split_rank8_kernel<false><<<blocks, kReductionThreads, 0, stream>>>(
          partial_output.data_ptr<float>(),
          output.data_ptr<float>(),
          rank8_down_tensor.data_ptr<float>(),
          rank8_b->data_ptr(),
          output_values,
          size_n,
          static_cast<int>(split_count),
          static_cast<int>(rank8_down_tensor.stride(0)),
          static_cast<float>(rank8_scale));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  if (split_count > 1) {
    constexpr int kReductionThreads = 256;
    const int output_values = size_m * size_n;
    const int blocks = (output_values + kReductionThreads - 1) / kReductionThreads;
    if (size_m <= 4 && size_n == 1024 && split_count == 64) {
      constexpr int kReductionWarps = kReductionThreads / 32;
      const int warp_blocks =
          (output_values + kReductionWarps * 4 - 1) / (kReductionWarps * 4);
      reduce_split_warp_kernel<64>
          <<<warp_blocks, kReductionThreads, 0, stream>>>(
              partial_output.data_ptr<float>(),
              output.data_ptr<float>(),
              output_values);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return output;
    }
    if (size_m == 8 && size_n == 1024 && split_count == 48) {
      constexpr int kReductionWarps = kReductionThreads / 32;
      const int warp_blocks =
          (output_values + kReductionWarps * 4 - 1) / (kReductionWarps * 4);
      reduce_split_warp_kernel<48>
          <<<warp_blocks, kReductionThreads, 0, stream>>>(
              partial_output.data_ptr<float>(),
              output.data_ptr<float>(),
              output_values);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return output;
    }
    if (size_m == 16 && size_n == 1024 && split_count == 32) {
      constexpr int kReductionWarps = kReductionThreads / 32;
      constexpr int kOutputsPerWarp = 16;
      const int warp_blocks =
          (output_values + kReductionWarps * kOutputsPerWarp - 1) /
          (kReductionWarps * kOutputsPerWarp);
      reduce_split_warp_kernel<32, kOutputsPerWarp>
          <<<warp_blocks, kReductionThreads, 0, stream>>>(
              partial_output.data_ptr<float>(),
              output.data_ptr<float>(),
              output_values);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return output;
    }
    const bool use_m1_warp_reducer =
        size_m == 1 &&
        ((size_k == 6144 && size_n == 5120) ||
         (size_k == 5120 && size_n == 6144) ||
         (size_k == 17408 && size_n == 5120));
    if (use_m1_warp_reducer) {
      constexpr int kReductionWarps = kReductionThreads / 32;
      constexpr int kOutputsPerWarp = 16;
      const int warp_blocks =
          (output_values + kReductionWarps * kOutputsPerWarp - 1) /
          (kReductionWarps * kOutputsPerWarp);
#define QVQ_LAUNCH_WARP_REDUCER(SPLITS)                                      \
  reduce_split_warp_kernel<SPLITS, kOutputsPerWarp>                          \
      <<<warp_blocks, kReductionThreads, 0, stream>>>(                        \
          partial_output.data_ptr<float>(),                                  \
          output.data_ptr<float>(),                                           \
          output_values)
      switch (split_count) {
        case 24:
          QVQ_LAUNCH_WARP_REDUCER(24);
          break;
        case 40:
          QVQ_LAUNCH_WARP_REDUCER(40);
          break;
        case 48:
          QVQ_LAUNCH_WARP_REDUCER(48);
          break;
        case 96:
          QVQ_LAUNCH_WARP_REDUCER(96);
          break;
        case 128:
          QVQ_LAUNCH_WARP_REDUCER(128);
          break;
        default:
          break;
      }
#undef QVQ_LAUNCH_WARP_REDUCER
      if (split_count == 24 || split_count == 40 || split_count == 48 ||
          split_count == 96 || split_count == 128) {
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
      }
    }
    if (size_m == 2 && size_k == 17408 && size_n == 5120 &&
        (split_count == 96 || split_count == 128)) {
      constexpr int kReductionWarps = kReductionThreads / 32;
      constexpr int kOutputsPerWarp = 16;
      const int warp_blocks =
          (output_values + kReductionWarps * kOutputsPerWarp - 1) /
          (kReductionWarps * kOutputsPerWarp);
      if (split_count == 96) {
        reduce_split_warp_kernel<96, kOutputsPerWarp>
            <<<warp_blocks, kReductionThreads, 0, stream>>>(
                partial_output.data_ptr<float>(),
                output.data_ptr<float>(),
                output_values);
      } else {
        reduce_split_warp_kernel<128, kOutputsPerWarp>
            <<<warp_blocks, kReductionThreads, 0, stream>>>(
                partial_output.data_ptr<float>(),
                output.data_ptr<float>(),
                output_values);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return output;
    }
    if constexpr (TransitionBits == 6) {
      if (size_m == 2 && size_k == 6144 && size_n == 5120 &&
          split_count == 48) {
        constexpr int kReductionWarps = kReductionThreads / 32;
        constexpr int kOutputsPerWarp = 16;
        const int warp_blocks =
            (output_values + kReductionWarps * kOutputsPerWarp - 1) /
            (kReductionWarps * kOutputsPerWarp);
        reduce_split_warp_kernel<48, kOutputsPerWarp>
            <<<warp_blocks, kReductionThreads, 0, stream>>>(
                partial_output.data_ptr<float>(),
                output.data_ptr<float>(),
                output_values);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
      }
    }
    const bool use_m2_warp_reducer =
        size_m == 2 &&
        ((size_k == 6144 && size_n == 5120) ||
         (size_k == 5120 && size_n == 6144));
    if (use_m2_warp_reducer) {
      constexpr int kReductionWarps = kReductionThreads / 32;
      constexpr int kOutputsPerWarp = 8;
      const int warp_blocks =
          (output_values + kReductionWarps * kOutputsPerWarp - 1) /
          (kReductionWarps * kOutputsPerWarp);
#define QVQ_LAUNCH_M2_WARP_REDUCER(SPLITS)                                   \
  reduce_split_warp_kernel<SPLITS, kOutputsPerWarp>                           \
      <<<warp_blocks, kReductionThreads, 0, stream>>>(                        \
          partial_output.data_ptr<float>(),                                  \
          output.data_ptr<float>(),                                           \
          output_values)
      switch (split_count) {
        case 24:
          QVQ_LAUNCH_M2_WARP_REDUCER(24);
          break;
        case 40:
          QVQ_LAUNCH_M2_WARP_REDUCER(40);
          break;
        case 48:
          QVQ_LAUNCH_M2_WARP_REDUCER(48);
          break;
        case 64:
          QVQ_LAUNCH_M2_WARP_REDUCER(64);
          break;
        default:
          break;
      }
#undef QVQ_LAUNCH_M2_WARP_REDUCER
      if (split_count == 24 || split_count == 40 || split_count == 48 ||
          split_count == 64) {
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
      }
    }
    if (size_m == 1 && size_k == 5120 && size_n == 1024 &&
        split_count == 56) {
      reduce_split_kernel<56>
          <<<blocks, kReductionThreads, 0, stream>>>(
              partial_output.data_ptr<float>(),
              output.data_ptr<float>(),
              output_values,
              56);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return output;
    }
#define QVQ_LAUNCH_STATIC_REDUCER(SPLITS)                                    \
  reduce_split_kernel<SPLITS><<<blocks, kReductionThreads, 0, stream>>>(     \
      partial_output.data_ptr<float>(),                                      \
      output.data_ptr<float>(),                                              \
      output_values,                                                         \
      SPLITS)
    const bool use_static_reducer =
        size_n != 1024 &&
        (size_m == 2 || size_m == 4 || size_m == 8 || size_m == 16 ||
         (size_m == 1 &&
          (size_n == 12288 || size_n == 10240 || size_n == 17408 ||
           (size_k == 17408 && size_n == 5120))));
    if (use_static_reducer) {
      switch (split_count) {
        case 9:
          QVQ_LAUNCH_STATIC_REDUCER(9);
          break;
        case 10:
          QVQ_LAUNCH_STATIC_REDUCER(10);
          break;
        case 12:
          QVQ_LAUNCH_STATIC_REDUCER(12);
          break;
        case 14:
          QVQ_LAUNCH_STATIC_REDUCER(14);
          break;
        case 16:
          QVQ_LAUNCH_STATIC_REDUCER(16);
          break;
        case 20:
          QVQ_LAUNCH_STATIC_REDUCER(20);
          break;
        case 24:
          QVQ_LAUNCH_STATIC_REDUCER(24);
          break;
        case 32:
          QVQ_LAUNCH_STATIC_REDUCER(32);
          break;
        case 40:
          QVQ_LAUNCH_STATIC_REDUCER(40);
          break;
        case 48:
          QVQ_LAUNCH_STATIC_REDUCER(48);
          break;
        case 64:
          QVQ_LAUNCH_STATIC_REDUCER(64);
          break;
        case 96:
          QVQ_LAUNCH_STATIC_REDUCER(96);
          break;
        case 128:
          QVQ_LAUNCH_STATIC_REDUCER(128);
          break;
        default:
          reduce_split_kernel<<<blocks, kReductionThreads, 0, stream>>>(
              partial_output.data_ptr<float>(),
              output.data_ptr<float>(),
              output_values,
              static_cast<int>(split_count));
          break;
      }
    } else {
      reduce_split_kernel<<<blocks, kReductionThreads, 0, stream>>>(
          partial_output.data_ptr<float>(),
          output.data_ptr<float>(),
          output_values,
          static_cast<int>(split_count));
    }
#undef QVQ_LAUNCH_STATIC_REDUCER
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
    int64_t split_count,
    const c10::optional<at::Tensor>& rank8_a,
    const c10::optional<at::Tensor>& rank8_b,
    double rank8_scale,
    const c10::optional<at::Tensor>& rank8_down) {
  switch (transition_bits) {
    case 4:
      return p32_window_ampere_impl<4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count,
          rank8_a, rank8_b, rank8_scale, rank8_down);
    case 5:
      return p32_window_ampere_impl<5>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count,
          rank8_a, rank8_b, rank8_scale, rank8_down);
    case 6:
      return p32_window_ampere_impl<6>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count,
          rank8_a, rank8_b, rank8_scale, rank8_down);
    case 7:
      return p32_window_ampere_impl<7>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count,
          rank8_a, rank8_b, rank8_scale, rank8_down);
    default:
      TORCH_CHECK(false, "QVQ P32 Ampere transition bits must be in [4, 7]");
  }
}

at::Tensor p32_rank8_project(const at::Tensor& input, const at::Tensor& rank8_a) {
  TORCH_CHECK(input.is_cuda() && rank8_a.is_cuda(),
              "QVQ Ampere rank8 projection tensors must be CUDA tensors");
  TORCH_CHECK(input.device() == rank8_a.device(),
              "QVQ Ampere rank8 projection tensors must share a device");
  TORCH_CHECK(input.scalar_type() == at::kHalf && rank8_a.scalar_type() == at::kHalf,
              "QVQ Ampere rank8 projection requires FP16 input and A");
  TORCH_CHECK(input.dim() == 2 && rank8_a.dim() == 2 &&
                  rank8_a.size(0) == input.size(1) && rank8_a.size(1) % 8 == 0,
              "QVQ Ampere grouped rank8 projection has invalid shapes");
  TORCH_CHECK(input.is_contiguous() && rank8_a.is_contiguous(),
              "QVQ Ampere rank8 projection tensors must be contiguous");
  const c10::cuda::CUDAGuard device_guard(input.device());
  const int capability = cached_device_capability(input.get_device());
  TORCH_CHECK(
      ampere_execution_capability_supported(capability),
      "QVQ Ampere rank8 projection requires compute capability 8.0");
  return at::mm(input, rank8_a, at::kFloat);
}

template <int TransitionBits>
at::Tensor p32_window_ampere_grouped_fused_impl(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts,
    const c10::optional<at::Tensor>& rank8_a = c10::nullopt,
    const c10::optional<at::Tensor>& rank8_b = c10::nullopt,
    at::ArrayRef<double> rank8_scales = {}) {
  constexpr int kWordsPerTile = 4 * TransitionBits;
  const int64_t segment_count = static_cast<int64_t>(out_features.size());
  TORCH_CHECK(
      segment_count >= 1 && segment_count <= kMaxGroupedP32Segments,
      "QVQ P32 Ampere fused groups require one to three segments");
  TORCH_CHECK(
      static_cast<int64_t>(bank_alt_ids.size()) == segment_count &&
          static_cast<int64_t>(split_counts.size()) == segment_count,
      "QVQ P32 Ampere grouped segment metadata lengths must match");
  TORCH_CHECK(
      rank8_a.has_value() == rank8_b.has_value(),
      "rank8_a and rank8_b must be provided together for grouped P32");
  TORCH_CHECK(
      rank8_scales.empty() ||
          static_cast<int64_t>(rank8_scales.size()) == segment_count,
      "grouped P32 rank8 scales must be empty or one value per segment");
  TORCH_CHECK(input.is_cuda(), "QVQ P32 Ampere input must be CUDA");
  TORCH_CHECK(
      trellis.device() == input.device() && levels.device() == input.device() &&
          bank_ids.device() == input.device(),
      "QVQ P32 Ampere grouped tensors must share one CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf && levels.scalar_type() == at::kHalf,
      "QVQ P32 Ampere input and levels must be float16");
  TORCH_CHECK(trellis.scalar_type() == at::kInt, "QVQ P32 Ampere trellis must be int32");
  TORCH_CHECK(bank_ids.scalar_type() == at::kByte, "QVQ P32 Ampere bank ids must be uint8");
  TORCH_CHECK(
      input.is_contiguous() && trellis.is_contiguous() && levels.is_contiguous() &&
          bank_ids.is_contiguous(),
      "QVQ P32 Ampere grouped tensors must be contiguous");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= kRows,
      "QVQ P32 Ampere input must have between 1 and 16 rows");
  TORCH_CHECK(
      input.size(1) > 0 && input.size(1) % kTileRows == 0,
      "QVQ P32 Ampere K must be positive and divisible by 16");
  TORCH_CHECK(levels.numel() == kLevels, "QVQ P32 Ampere requires 256 PGC16 levels");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const int capability = cached_device_capability(input.get_device());
  TORCH_CHECK(
      ampere_execution_capability_supported(capability),
      "QVQ P32 Ampere WMMA requires compute capability 8.0, got ",
      capability / 10,
      ".",
      capability % 10);

  const int size_m = static_cast<int>(input.size(0));
  const int size_k = static_cast<int>(input.size(1));
  const int k_tiles = size_k / kTileRows;
  GroupedP32LaunchParams params{};
  params.segment_count = static_cast<int>(segment_count);
  int total_n_tiles = 0;
  int active_scalar_blocks = 0;
  int active_wmma_blocks = 0;
  int uniform_split_count = 0;
  bool uniform_splits = true;
  int64_t output_values = 0;
  int64_t partial_values = 0;
  bool needs_reduction = false;
  bool use_flash_next_gate_up_wide =
      TransitionBits == 6 && size_k == 2560 && segment_count == 2;
  for (int segment = 0; segment < segment_count; ++segment) {
    const int size_n = static_cast<int>(out_features[segment]);
    const int split_count = static_cast<int>(split_counts[segment]);
    const int bank_alt_id = static_cast<int>(bank_alt_ids[segment]);
    TORCH_CHECK(
        size_n > 0 && size_n % kTileColumns == 0,
        "QVQ P32 Ampere grouped N must be positive and divisible by 16");
    TORCH_CHECK(
        split_count >= 1 && split_count <= 128 && split_count <= k_tiles,
        "QVQ P32 Ampere grouped split count is invalid");
    TORCH_CHECK(
        bank_alt_id >= 0 && bank_alt_id <= 3,
        "QVQ P32 Ampere grouped bank ID must be in [0, 3]");
    if (segment == 0) {
      uniform_split_count = split_count;
    } else if (split_count != uniform_split_count) {
      uniform_splits = false;
    }
    const int n_tiles = size_n / kTileColumns;
    if (size_n != 640 || split_count != 40) {
      use_flash_next_gate_up_wide = false;
    }
    params.n_tile_start[segment] = total_n_tiles;
    params.n_tiles[segment] = n_tiles;
    params.bank_alt_id[segment] = bank_alt_id;
    params.split_count[segment] = split_count;
    params.rank8_offset[segment] = segment * 8;
    params.rank8_scale[segment] = rank8_scales.empty()
        ? 1.0f
        : static_cast<float>(rank8_scales[segment]);
    params.output_offset[segment] = output_values;
    params.partial_offset[segment] = partial_values;
    if (split_count > 1) {
      partial_values += static_cast<int64_t>(split_count) * size_m * size_n;
      needs_reduction = true;
    }
    output_values += static_cast<int64_t>(size_m) * size_n;
    total_n_tiles += n_tiles;
    active_scalar_blocks +=
        ((n_tiles + kM1TilesPerBlock - 1) / kM1TilesPerBlock) * split_count;
    active_wmma_blocks +=
        ((n_tiles + kTilesPerBlock - 1) / kTilesPerBlock) * split_count;
  }
  if (use_flash_next_gate_up_wide) {
    // Flash-Next gate/up has two aligned 640-column children.  Let each warp
    // consume both N16 tiles so one CTA covers eight tiles instead of four.
    active_wmma_blocks = 0;
    for (int segment = 0; segment < segment_count; ++segment) {
      active_wmma_blocks +=
          ((params.n_tiles[segment] + 2 * kTilesPerBlock - 1) /
           (2 * kTilesPerBlock)) * params.split_count[segment];
    }
  }
  const int total_n = total_n_tiles * kTileColumns;
  if (rank8_a.has_value()) {
    TORCH_CHECK(
        rank8_a->device() == input.device() && rank8_b->device() == input.device(),
        "grouped P32 rank8 tensors must share the input CUDA device");
    TORCH_CHECK(
        rank8_a->scalar_type() == at::kHalf &&
            (rank8_b->scalar_type() == at::kHalf ||
             rank8_b->scalar_type() == at::kFloat),
        "grouped P32 rank8_a must be float16 and rank8_b float16 or float32");
    TORCH_CHECK(
        rank8_a->is_contiguous() && rank8_b->is_contiguous(),
        "grouped P32 rank8 tensors must be contiguous");
    TORCH_CHECK(
        rank8_a->dim() == 2 && rank8_a->size(0) == size_k &&
            rank8_a->size(1) == segment_count * 8,
        "grouped P32 rank8_a must have shape [K, 8 * segment_count]");
    TORCH_CHECK(
        rank8_b->dim() == 2 && rank8_b->size(0) == 8 &&
            rank8_b->size(1) == total_n,
        "grouped P32 rank8_b must have shape [8, total_N]");
    for (double scale : rank8_scales) {
      TORCH_CHECK(std::isfinite(scale), "grouped P32 rank8 scales must be finite");
    }
  }
  const int64_t expected_tiles = static_cast<int64_t>(k_tiles) * total_n_tiles;
  TORCH_CHECK(
      trellis.numel() == expected_tiles * kWordsPerTile,
      "QVQ P32 Ampere grouped trellis has the wrong word count");
  TORCH_CHECK(
      bank_ids.numel() == expected_tiles,
      "QVQ P32 Ampere grouped bank ids have the wrong length");

  auto output = at::empty({output_values}, input.options().dtype(at::kFloat));
  auto partial_output = partial_values > 0
      ? at::empty({partial_values}, input.options().dtype(at::kFloat))
      : at::empty({0}, input.options().dtype(at::kFloat));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  const auto* input_ptr = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  const auto* trellis_ptr = reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>());
  const auto* levels_ptr = reinterpret_cast<const half*>(levels.data_ptr<at::Half>());
  const auto* bank_ids_ptr = bank_ids.data_ptr<uint8_t>();
  auto* partial_output_ptr = partial_output.data_ptr<float>();
  auto* output_ptr = output.data_ptr<float>();

  // Legal A41 groups are Q/K/V or gate/up siblings and therefore use the
  // common short-K scalar route for M<=4.  Long-K grouped shapes deliberately
  // remain on the ordinary exact dispatcher until a matching route exists.
  const bool use_small_m_scalar = size_m <= 4 && size_k <= 6144;
  const bool use_flash_next_qkv_shape =
      TransitionBits == 6 && segment_count == 3 &&
      params.n_tiles[0] == 768 && params.n_tiles[1] == 32 &&
      params.n_tiles[2] == 32;
  const bool use_flash_next_gate_up_shape =
      TransitionBits == 6 && segment_count == 2 &&
      params.n_tiles[0] == 40 && params.n_tiles[1] == 40;
  const bool use_flash_next_shape =
      use_flash_next_qkv_shape || use_flash_next_gate_up_shape;
  if (use_small_m_scalar) {
    const dim3 grid(static_cast<unsigned>(active_scalar_blocks), 1, 1);
    if (size_m == 1) {
      if (use_flash_next_shape) {
        p32_window_ampere_grouped_scalar_kernel<
            TransitionBits, 1, kStageKTiles, true>
            <<<grid, kM1Threads, 0, stream>>>(
                input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, params,
                partial_output_ptr, output_ptr, static_cast<int>(segment_count),
                size_k, total_n_tiles);
      } else {
        p32_window_ampere_grouped_scalar_kernel<TransitionBits, 1>
            <<<grid, kM1Threads, 0, stream>>>(
                input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, params,
                partial_output_ptr, output_ptr, static_cast<int>(segment_count),
                size_k, total_n_tiles);
      }
    } else if (size_m == 2) {
      if (use_flash_next_shape) {
        p32_window_ampere_grouped_scalar_kernel<
            TransitionBits, 2, kScalarTripleStageKTiles, true>
            <<<grid, kM1Threads, 0, stream>>>(
                input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, params,
                partial_output_ptr, output_ptr, static_cast<int>(segment_count),
                size_k, total_n_tiles);
      } else {
        p32_window_ampere_grouped_scalar_kernel<
            TransitionBits, 2, kScalarTripleStageKTiles>
            <<<grid, kM1Threads, 0, stream>>>(
                input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, params,
                partial_output_ptr, output_ptr, static_cast<int>(segment_count),
                size_k, total_n_tiles);
      }
    } else if (size_m == 3) {
      p32_window_ampere_grouped_scalar_kernel<TransitionBits, 3>
          <<<grid, kM1Threads, 0, stream>>>(
              input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, params,
              partial_output_ptr, output_ptr, static_cast<int>(segment_count),
              size_k, total_n_tiles);
    } else {
      // The Flash-Next QKV child widths are 768/32/32 N16 tiles.  Its
      // sixteen-way split wave gives exactly ten K tiles per split, so four
      // staged K tiles removes a tail iteration without increasing the
      // shared-memory footprint enough to reduce residency.  Gate/up keeps
      // the two-stage default because its smaller N does not amortize the
      // larger stage footprint.
      if (use_flash_next_qkv_shape) {
        p32_window_ampere_grouped_scalar_kernel<TransitionBits, 4, 4, true>
            <<<grid, kM1Threads, 0, stream>>>(
                input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, params,
                partial_output_ptr, output_ptr, static_cast<int>(segment_count),
                size_k, total_n_tiles);
      } else {
        constexpr int kM4StageKTiles = TransitionBits == 4
            ? kScalarLongStageKTiles
            : kStageKTiles;
        p32_window_ampere_grouped_scalar_kernel<
            TransitionBits, 4, kM4StageKTiles>
            <<<grid, kM1Threads, 0, stream>>>(
                input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, params,
                partial_output_ptr, output_ptr, static_cast<int>(segment_count),
                size_k, total_n_tiles);
      }
    }
  } else {
    const dim3 grid(static_cast<unsigned>(active_wmma_blocks), 1, 1);
    if (use_flash_next_gate_up_wide && size_m == kRows) {
      p32_window_ampere_grouped_wmma_kernel<
          TransitionBits, true, 0, 640, true, false, 2560, false, false,
          true, true, 4, 40><<<grid, kThreads, 0, stream>>>(
          input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, params,
          partial_output_ptr, output_ptr, static_cast<int>(segment_count),
          size_m, size_k, total_n_tiles);
    } else if (use_flash_next_gate_up_wide && size_m == 8) {
      p32_window_ampere_grouped_wmma_kernel<
          TransitionBits, false, 8, 640, true, false, 2560, false, false,
          true, true, 4, 40><<<grid, kThreads, 0, stream>>>(
          input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, params,
          partial_output_ptr, output_ptr, static_cast<int>(segment_count),
          size_m, size_k, total_n_tiles);
    } else if (size_m == kRows) {
      p32_window_ampere_grouped_wmma_kernel<TransitionBits, true>
          <<<grid, kThreads, 0, stream>>>(
              input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, params,
              partial_output_ptr, output_ptr, static_cast<int>(segment_count),
              size_m, size_k, total_n_tiles);
    } else if (size_m == 8) {
      p32_window_ampere_grouped_wmma_kernel<TransitionBits, false, 8>
          <<<grid, kThreads, 0, stream>>>(
              input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, params,
              partial_output_ptr, output_ptr, static_cast<int>(segment_count),
              size_m, size_k, total_n_tiles);
    } else {
      p32_window_ampere_grouped_wmma_kernel<TransitionBits, false>
          <<<grid, kThreads, 0, stream>>>(
              input_ptr, trellis_ptr, levels_ptr, bank_ids_ptr, params,
              partial_output_ptr, output_ptr, static_cast<int>(segment_count),
              size_m, size_k, total_n_tiles);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (rank8_a.has_value()) {
    auto rank8_down = at::mm(input, *rank8_a, at::kFloat);
    constexpr int kReductionThreads = 256;
    // Keep block ownership segment-local.  A block never crosses a child
    // boundary, so the reducer pays the segment lookup once per block rather
    // than once per output thread while retaining the exact split order.
    int blocks = 0;
    for (int segment = 0; segment < segment_count; ++segment) {
      const int segment_values =
          size_m * params.n_tiles[segment] * kTileColumns;
      blocks += (segment_values + kReductionThreads - 1) / kReductionThreads;
    }
    const dim3 reduction_grid(static_cast<unsigned>(blocks), 1, 1);
    if (uniform_splits && uniform_split_count == 16 && size_m == 1) {
      if (rank8_b->scalar_type() == at::kFloat) {
        reduce_grouped_split_rank8_kernel<true, 16, true><<<
            reduction_grid, kReductionThreads, 0, stream>>>(
            partial_output_ptr,
            output_ptr,
            params,
            static_cast<int>(segment_count),
            size_m,
            total_n,
            rank8_down.data_ptr<float>(),
            rank8_b->data_ptr());
      } else {
        reduce_grouped_split_rank8_kernel<false, 16, true><<<
            reduction_grid, kReductionThreads, 0, stream>>>(
            partial_output_ptr,
            output_ptr,
            params,
            static_cast<int>(segment_count),
            size_m,
            total_n,
            rank8_down.data_ptr<float>(),
            rank8_b->data_ptr());
      }
    } else if (uniform_splits && uniform_split_count == 16) {
      if (rank8_b->scalar_type() == at::kFloat) {
        reduce_grouped_split_rank8_kernel<true, 16><<<
            reduction_grid, kReductionThreads, 0, stream>>>(
            partial_output_ptr,
            output_ptr,
            params,
            static_cast<int>(segment_count),
            size_m,
            total_n,
            rank8_down.data_ptr<float>(),
            rank8_b->data_ptr());
      } else {
        reduce_grouped_split_rank8_kernel<false, 16><<<
            reduction_grid, kReductionThreads, 0, stream>>>(
            partial_output_ptr,
            output_ptr,
            params,
            static_cast<int>(segment_count),
            size_m,
            total_n,
            rank8_down.data_ptr<float>(),
            rank8_b->data_ptr());
      }
    } else if (uniform_splits && uniform_split_count == 40) {
      if (rank8_b->scalar_type() == at::kFloat) {
        reduce_grouped_split_rank8_kernel<true, 40><<<
            reduction_grid, kReductionThreads, 0, stream>>>(
            partial_output_ptr,
            output_ptr,
            params,
            static_cast<int>(segment_count),
            size_m,
            total_n,
            rank8_down.data_ptr<float>(),
            rank8_b->data_ptr());
      } else {
        reduce_grouped_split_rank8_kernel<false, 40><<<
            reduction_grid, kReductionThreads, 0, stream>>>(
            partial_output_ptr,
            output_ptr,
            params,
            static_cast<int>(segment_count),
            size_m,
            total_n,
            rank8_down.data_ptr<float>(),
            rank8_b->data_ptr());
      }
    } else {
      if (rank8_b->scalar_type() == at::kFloat) {
        reduce_grouped_split_rank8_kernel<true><<<
            reduction_grid, kReductionThreads, 0, stream>>>(
            partial_output_ptr,
            output_ptr,
            params,
            static_cast<int>(segment_count),
            size_m,
            total_n,
            rank8_down.data_ptr<float>(),
            rank8_b->data_ptr());
      } else {
        reduce_grouped_split_rank8_kernel<false><<<
            reduction_grid, kReductionThreads, 0, stream>>>(
            partial_output_ptr,
            output_ptr,
            params,
            static_cast<int>(segment_count),
            size_m,
            total_n,
            rank8_down.data_ptr<float>(),
            rank8_b->data_ptr());
      }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (needs_reduction) {
    constexpr int kReductionThreads = 256;
    const int output_value_count = size_m * total_n;
    const int blocks =
        (output_value_count + kReductionThreads - 1) / kReductionThreads;
    reduce_grouped_split_kernel<<<blocks, kReductionThreads, 0, stream>>>(
        partial_output_ptr,
        output_ptr,
        params,
        static_cast<int>(segment_count),
        size_m,
        total_n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

at::Tensor p32_window_ampere_grouped_fused(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts,
    const c10::optional<at::Tensor>& rank8_a,
    const c10::optional<at::Tensor>& rank8_b,
    at::ArrayRef<double> rank8_scales) {
  switch (transition_bits) {
    case 4:
      return p32_window_ampere_grouped_fused_impl<4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts,
          rank8_a, rank8_b, rank8_scales);
    case 5:
      return p32_window_ampere_grouped_fused_impl<5>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts,
          rank8_a, rank8_b, rank8_scales);
    case 6:
      return p32_window_ampere_grouped_fused_impl<6>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts,
          rank8_a, rank8_b, rank8_scales);
    case 7:
      return p32_window_ampere_grouped_fused_impl<7>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts,
          rank8_a, rank8_b, rank8_scales);
    default:
      TORCH_CHECK(false, "QVQ P32 Ampere transition bits must be in [4, 7]");
  }
}

template <int TransitionBits>
std::vector<at::Tensor> p32_window_ampere_grouped_impl(
    const at::Tensor& input,
    at::TensorList trellises,
    const at::Tensor& levels,
    at::TensorList bank_ids,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts) {
  const int64_t segment_count = static_cast<int64_t>(trellises.size());
  TORCH_CHECK(segment_count > 0, "QVQ P32 Ampere grouped execution requires at least one segment");
  TORCH_CHECK(
      static_cast<int64_t>(bank_ids.size()) == segment_count &&
          static_cast<int64_t>(out_features.size()) == segment_count &&
          static_cast<int64_t>(bank_alt_ids.size()) == segment_count &&
          static_cast<int64_t>(split_counts.size()) == segment_count,
      "QVQ P32 Ampere grouped segment metadata lengths must match");

  std::vector<at::Tensor> outputs;
  outputs.reserve(segment_count);
  // This correctness-first native dispatcher deliberately invokes the same
  // child implementation used by an independent projection.  Consequently
  // every segment retains its scalar/WMMA route, K partition, specialized
  // reducer, and left-to-right FP32 addition order.  The shared activation is
  // passed by reference and is never copied or transformed here.
  for (int64_t segment = 0; segment < segment_count; ++segment) {
    outputs.push_back(p32_window_ampere_impl<TransitionBits>(
        input,
        trellises[segment],
        levels,
        bank_ids[segment],
        out_features[segment],
        bank_alt_ids[segment],
        split_counts[segment]));
  }
  return outputs;
}

std::vector<at::Tensor> p32_window_ampere_grouped(
    const at::Tensor& input,
    at::TensorList trellises,
    const at::Tensor& levels,
    at::TensorList bank_ids,
    int64_t transition_bits,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts) {
  switch (transition_bits) {
    case 4:
      return p32_window_ampere_grouped_impl<4>(
          input, trellises, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    case 5:
      return p32_window_ampere_grouped_impl<5>(
          input, trellises, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    case 6:
      return p32_window_ampere_grouped_impl<6>(
          input, trellises, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    case 7:
      return p32_window_ampere_grouped_impl<7>(
          input, trellises, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    default:
      TORCH_CHECK(false, "QVQ P32 Ampere transition bits must be in [4, 7]");
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq_ampere, m) {
  m.def("p32_window(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int out_features, int bank_alt_id=3, int split_count=1, Tensor? rank8_a=None, Tensor? rank8_b=None, float rank8_scale=1.0, Tensor? rank8_down=None) -> Tensor");
  m.def("rank8_project(Tensor input, Tensor rank8_a) -> Tensor");
  m.def("p32_window_grouped(Tensor input, Tensor[] trellises, Tensor levels, Tensor[] bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts) -> Tensor[]");
  m.def("p32_window_grouped_fused(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts, Tensor? rank8_a=None, Tensor? rank8_b=None, float[] rank8_scales=[]) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq_ampere, CUDA, m) {
  m.impl("p32_window", p32_window_ampere);
  m.impl("rank8_project", p32_rank8_project);
  m.impl("p32_window_grouped", p32_window_ampere_grouped);
  m.impl("p32_window_grouped_fused", p32_window_ampere_grouped_fused);
}
