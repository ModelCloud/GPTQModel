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

#include <array>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kTileRows = 16;
constexpr int kTileColumns = 16;
constexpr int kLocalRingTileRows = 32;
constexpr int kLocalRingTileColumns = 8;
constexpr int kLocalRingCount = 8;
constexpr int kLocalRingSteps = 16;
constexpr int kTileValues = kTileRows * kTileColumns;
constexpr int kThreads = 256;
constexpr int kRowsPerBlock = 32;
constexpr bool kQvqDebugDisableVecStaging = false;

constexpr int kPgc16LevelCount = 256;
constexpr uint32_t kPgc16Multiplier = 40503u;
constexpr uint32_t kPgc16Increment = 17011u;
constexpr int kMaxCachedCudaDevices = 64;

// Hopper WMMA blocks trade activation reuse against grid parallelism. Two N8
// tiles per block keeps one staged K32 stripe shared by adjacent outputs while
// providing enough blocks to fill the H200's 132 SMs.
constexpr int kHopperWmmaOutputTiles = 4;

struct QvqCudaDeviceConfig {
  int major;
  int minor;
  int sm_count;
  int max_grid_y;
};

std::array<QvqCudaDeviceConfig, kMaxCachedCudaDevices> qvq_cuda_device_configs{};
std::array<std::once_flag, kMaxCachedCudaDevices> qvq_cuda_device_config_once;

const QvqCudaDeviceConfig& qvq_cuda_device_config(int device) {
  TORCH_CHECK(
      device >= 0 && device < kMaxCachedCudaDevices,
      "CUDA device ordinal is outside the CUDA device config cache: ",
      device);
  std::call_once(qvq_cuda_device_config_once[device], [device]() {
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    qvq_cuda_device_configs[device] = {
        properties.major,
        properties.minor,
        properties.multiProcessorCount,
        properties.maxGridSize[1]};
  });
  return qvq_cuda_device_configs[device];
}

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

template <int TransitionBits>
__device__ __forceinline__ uint32_t pgc16_v2_bank_mask(uint32_t bank) {
  constexpr uint32_t masks[6][4] = {
      {0x0000u, 0xA5A5u, 0x5A5Au, 0x3C3Cu},
      {0x0000u, 0xA5A5u, 0x9696u, 0x6969u},
      {0x0000u, 0x5A5Au, 0x3C3Cu, 0xC3C3u},
      {0x0000u, 0x9696u, 0x3C3Cu, 0xC3C3u},
      {0x0000u, 0x6969u, 0x5A5Au, 0x3C3Cu},
      {0x0000u, 0xC3C3u, 0x9696u, 0x5A5Au},
  };
  static_assert(TransitionBits >= 2 && TransitionBits <= 7);
  return masks[TransitionBits - 2][bank & 3u];
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

template <int TransitionBits>
__device__ __forceinline__ uint32_t qvq_local_ring_state(
    const uint32_t* words, int ring, int pair_in_ring) {
  constexpr int total_edges = kLocalRingSteps;
  constexpr int edge_mask = total_edges - 1;
  constexpr int edge_count = (15 + TransitionBits) / TransitionBits;
  const int first = (pair_in_ring + total_edges - edge_count + 1) & edge_mask;
  uint32_t state = 0;
#pragma unroll
  for (int j = 0; j < edge_count; ++j) {
    const int edge = (first + j) & edge_mask;
    state = ((state << TransitionBits) |
             planar_transition<TransitionBits>(words, ring * total_edges + edge)) & 0xffffu;
  }
  return state;
}

// Keep the Blackwell W2 recurrence in a separate specialization. CUDA 13
// otherwise reschedules the longer four-edge recurrence when direct
// higher-rate paths share the primary template, cutting non-split throughput
// by more than half despite identical resource counts.
template <>
__device__ __forceinline__ uint32_t qvq_local_ring_state<4>(
    const uint32_t* words, int ring, int pair_in_ring) {
  constexpr int total_edges = kLocalRingSteps;
  constexpr int edge_mask = total_edges - 1;
  constexpr int edge_count = (15 + 4) / 4;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  constexpr unsigned kPairLeaderMask = 0x55555555u;
  const uint32_t transition = planar_transition<4>(
      words, ring * total_edges + pair_in_ring);
  uint32_t state = 0;
#pragma unroll
  for (int j = 0; j < edge_count; ++j) {
    const int source_pair =
        (pair_in_ring + total_edges - edge_count + 1 + j) & edge_mask;
    const uint32_t edge = __shfl_sync(kPairLeaderMask, transition, source_pair << 1);
    state = ((state << 4) | edge) & 0xffffu;
  }
  return state;
#else
  const int first = (pair_in_ring + total_edges - edge_count + 1) & edge_mask;
  uint32_t state = 0;
#pragma unroll
  for (int j = 0; j < edge_count; ++j) {
    const int edge = (first + j) & edge_mask;
    state = ((state << 4) |
             planar_transition<4>(words, ring * total_edges + edge)) & 0xffffu;
  }
  return state;
#endif
}

__device__ __noinline__ uint32_t qvq_local_ring_state_runtime(
    const uint32_t* words, int ring, int pair_in_ring, int transition_bits) {
  constexpr int total_edges = kLocalRingSteps;
  constexpr int edge_mask = total_edges - 1;
  const int edge_count = (15 + transition_bits) / transition_bits;
  const int first = (pair_in_ring + total_edges - edge_count + 1) & edge_mask;
  uint32_t state = 0;
  for (int j = 0; j < edge_count; ++j) {
    const int edge = (first + j) & edge_mask;
    state = ((state << transition_bits) |
             planar_transition_runtime(words, ring * total_edges + edge, transition_bits)) & 0xffffu;
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

// LR32 maps one warp to one output column of its logical K32 x N8 tile.  The
// two lanes in each pair share the local-ring state and PGC16 lookup, while
// the 8-bit selector contains one bank bit for each warp/ring.
template <int TransitionBits>
__device__ __forceinline__ float qvq_decode_local_ring_weight_fast(
    const uint32_t* packed_words,
    uint8_t packed_bank_id,
    const half* cached_levels,
    int ring,
    int k_local,
    int bank_alt_id) {
  uint32_t level_pair = 0;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 1200
  // TB7 spills its unrolled mask lookup on Ada. Hopper has the same issue at
  // TB6, while lower rates retain the direct lookup that is faster on SM89.
  // Broadcasting the uniform per-ring value keeps the architecture/rate
  // exceptions local to the decode without adding another kernel variant.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
  constexpr bool kBroadcastBankMask = TransitionBits == 6 || TransitionBits == 7;
#else
  constexpr bool kBroadcastBankMask = TransitionBits == 7;
#endif
  uint32_t bank_mask = 0;
  if constexpr (kBroadcastBankMask) {
    if (k_local == 0) {
      const uint32_t bank = ((static_cast<uint32_t>(packed_bank_id) >> ring) & 1u) *
          static_cast<uint32_t>(bank_alt_id);
      bank_mask = pgc16_v2_bank_mask<TransitionBits>(bank);
    }
    bank_mask = __shfl_sync(0xffffffffu, bank_mask, 0);
  }
  if ((k_local & 1) == 0) {
    const uint32_t state = qvq_local_ring_state<TransitionBits>(packed_words, ring, k_local >> 1);
    uint32_t mixed;
    if constexpr (kBroadcastBankMask) {
      mixed = pgc16_mix(state ^ bank_mask);
    } else {
      const uint32_t bank = ((static_cast<uint32_t>(packed_bank_id) >> ring) & 1u) *
          static_cast<uint32_t>(bank_alt_id);
      mixed = pgc16_mix(state ^ pgc16_v2_bank_mask_runtime(TransitionBits, bank));
    }
    level_pair = static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed >> 8])) |
        (static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed & 0xffu])) << 16);
  }
#else
  // W2 has a dedicated broadcast specialization below. Higher Blackwell
  // rates avoid the broadcast and use their shorter direct recurrences.
  if ((k_local & 1) == 0) {
    const uint32_t state = qvq_local_ring_state<TransitionBits>(packed_words, ring, k_local >> 1);
    const uint32_t bank = ((static_cast<uint32_t>(packed_bank_id) >> ring) & 1u) *
        static_cast<uint32_t>(bank_alt_id);
    const uint32_t mixed = pgc16_mix(state ^ pgc16_v2_bank_mask_runtime(TransitionBits, bank));
    level_pair = static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed >> 8])) |
        (static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed & 0xffu])) << 16);
  }
#endif
  level_pair = __shfl_sync(0xffffffffu, level_pair, k_local & ~1);
  const uint32_t level_bits = (k_local & 1) == 0 ? level_pair & 0xffffu : level_pair >> 16;
  return __half2float(__ushort_as_half(static_cast<unsigned short>(level_bits)));
}

template <>
__device__ __forceinline__ float qvq_decode_local_ring_weight_fast<4>(
    const uint32_t* packed_words,
    uint8_t packed_bank_id,
    const half* cached_levels,
    int ring,
    int k_local,
    int bank_alt_id) {
  uint32_t level_pair = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  uint32_t bank_mask = 0;
  if (k_local == 0) {
    const uint32_t bank = ((static_cast<uint32_t>(packed_bank_id) >> ring) & 1u) *
        static_cast<uint32_t>(bank_alt_id);
    bank_mask = pgc16_v2_bank_mask<4>(bank);
  }
  bank_mask = __shfl_sync(0xffffffffu, bank_mask, 0);
#endif
  if ((k_local & 1) == 0) {
    const uint32_t state = qvq_local_ring_state<4>(packed_words, ring, k_local >> 1);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const uint32_t mixed = pgc16_mix(state ^ bank_mask);
#else
    const uint32_t bank = ((static_cast<uint32_t>(packed_bank_id) >> ring) & 1u) *
        static_cast<uint32_t>(bank_alt_id);
    const uint32_t mixed = pgc16_mix(state ^ pgc16_v2_bank_mask_runtime(4, bank));
#endif
    level_pair = static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed >> 8])) |
        (static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed & 0xffu])) << 16);
  }
  level_pair = __shfl_sync(0xffffffffu, level_pair, k_local & ~1);
  const uint32_t level_bits = (k_local & 1) == 0 ? level_pair & 0xffffu : level_pair >> 16;
  return __half2float(__ushort_as_half(static_cast<unsigned short>(level_bits)));
}

__device__ __noinline__ float qvq_decode_local_ring_weight_runtime(
    const uint32_t* packed_words,
    uint8_t packed_bank_id,
    const half* cached_levels,
    int ring,
    int k_local,
    int transition_bits,
    int bank_alt_id) {
  uint32_t level_pair = 0;
  if ((k_local & 1) == 0) {
    const uint32_t state = qvq_local_ring_state_runtime(packed_words, ring, k_local >> 1, transition_bits);
    const uint32_t bank = ((static_cast<uint32_t>(packed_bank_id) >> ring) & 1u) *
        static_cast<uint32_t>(bank_alt_id);
    const uint32_t mixed = pgc16_mix(state ^ pgc16_v2_bank_mask_runtime(transition_bits, bank));
    level_pair = static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed >> 8])) |
        (static_cast<uint32_t>(__half_as_ushort(cached_levels[mixed & 0xffu])) << 16);
  }
  level_pair = __shfl_sync(0xffffffffu, level_pair, k_local & ~1);
  const uint32_t level_bits = (k_local & 1) == 0 ? level_pair & 0xffffu : level_pair >> 16;
  return __half2float(__ushort_as_half(static_cast<unsigned short>(level_bits)));
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
    const half* __restrict__ levels,
    OutputScalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int bank_mode,
    int bank_alt_id) {
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
              bank_mode, bank_alt_id);
        } else {
          weight = qvq_decode_weight<VectorSize>(
              packed_words[parity][u], bank_ids == nullptr ? 0 : packed_bank_ids[parity][u], cached_levels, local,
              transition_bits, bank_mode, bank_alt_id);
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
    const half* __restrict__ levels,
    float* __restrict__ partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    int transition_bits,
    int bank_mode,
    int bank_alt_id) {
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
              bank_mode, bank_alt_id);
        } else {
          weight = qvq_decode_weight<VectorSize>(
              packed_words[parity][u], bank_ids == nullptr ? 0 : packed_bank_ids[parity][u], cached_levels, local,
              transition_bits, bank_mode, bank_alt_id);
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

// LR32 uses the serialized 256-weight payload as eight independent rings. A
// warp owns one logical N8 column, and its 32 lanes own the K32 values. This
// preserves the ring-local state order while making the reduction a single
// warp shuffle instead of a cross-warp shared-memory reduction.
template <
    typename Scalar,
    typename OutputScalar,
    int ROWS,
    int TransitionBits = 0,
    bool SplitK = false,
    int OutputTilesPerBlock = 1,
    bool VectorStaging = true>
__global__ __launch_bounds__(kThreads) void qvq_gemv_local_ring_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ trellis,
    const uint8_t* __restrict__ bank_ids,
    const half* __restrict__ levels,
    OutputScalar* __restrict__ output,
    float* __restrict__ partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    int transition_bits,
    int bank_alt_id) {
  // ROWS=16 has enough shared-memory headroom to amortize each synchronization
  // pair across twice as many K32 tiles without reducing register occupancy.
  // Blackwell W2 also benefits at ROWS=8; higher rates, Ada small-row paths,
  // and the launch-bound ROWS=1 path do better with the smaller shared-memory
  // image. Keep ROWS=32 at eight tiles because its wider activation stripe
  // would lower resident block count.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  constexpr int kBatchTiles =
      (TransitionBits == 4 || SplitK) && (ROWS == 8 || ROWS == 16) ? 16 : 8;
#else
  constexpr int kBatchTiles = ROWS == 16 ? 16 : 8;
#endif
  constexpr int kOutputTilesPerBlock = OutputTilesPerBlock;
  constexpr int kMaxWords = 32;
  constexpr int kWordsPerTile = 4 * TransitionBits;
  constexpr int kInputVecsPerRow = kLocalRingTileRows * static_cast<int>(sizeof(Scalar)) / sizeof(uint4);

  __shared__ half cached_levels[kPgc16LevelCount];
  __shared__ __align__(16) uint32_t packed_words[kBatchTiles][kOutputTilesPerBlock][kMaxWords];
  __shared__ uint8_t packed_bank_ids[kBatchTiles][kOutputTilesPerBlock];
  __shared__ __align__(16) Scalar input_tile[kBatchTiles][ROWS * kLocalRingTileRows];

  const int thread = static_cast<int>(threadIdx.x);
  const int ring = thread >> 5;
  const int k_local = thread & 31;
  const int n_tile_base = static_cast<int>(blockIdx.x) * kOutputTilesPerBlock;
  const int m0 = static_cast<int>(blockIdx.y) * ROWS;
  const int block_rows = min(ROWS, size_m - m0);
  const int n_tiles = size_n / kLocalRingTileColumns;
  const int k_tiles = size_k / kLocalRingTileRows;

  cached_levels[thread] = levels[thread];
  __syncthreads();

  float accumulator[ROWS][kOutputTilesPerBlock];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    for (int sub = 0; sub < kOutputTilesPerBlock; ++sub) {
      accumulator[r][sub] = 0.0f;
    }
  }

  const int words_per_tile = TransitionBits != 0 ? kWordsPerTile : 4 * transition_bits;
  auto stage_batch = [&](int kb, int tiles_here) {
    if constexpr (TransitionBits != 0 && VectorStaging && !kQvqDebugDisableVecStaging) {
      constexpr int words_per_vec = kWordsPerTile / 4;
      const int vec_total = tiles_here * kOutputTilesPerBlock * words_per_vec;
      uint4* words4 = reinterpret_cast<uint4*>(packed_words[0][0]);
      const uint4* trellis4 = reinterpret_cast<const uint4*>(trellis);
      for (int index = thread; index < vec_total; index += kThreads) {
        const int tile_slot = index / words_per_vec;
        const int w4 = index - tile_slot * words_per_vec;
        const int u = tile_slot / kOutputTilesPerBlock;
        const int sub = tile_slot % kOutputTilesPerBlock;
        const int n_tile = n_tile_base + sub;
        words4[tile_slot * (kMaxWords / 4) + w4] =
            n_tile < n_tiles
                ? trellis4[(static_cast<int64_t>(kb + u) * n_tiles + n_tile) * words_per_vec + w4]
                : make_uint4(0u, 0u, 0u, 0u);
      }
    } else {
      for (int index = thread; index < tiles_here * kOutputTilesPerBlock * words_per_tile; index += kThreads) {
        const int tile_slot = index / words_per_tile;
        const int w = index - tile_slot * words_per_tile;
        const int u = tile_slot / kOutputTilesPerBlock;
        const int sub = tile_slot % kOutputTilesPerBlock;
        const int n_tile = n_tile_base + sub;
        const int tile_index = (kb + u) * n_tiles + n_tile;
        packed_words[u][sub][w] =
            n_tile < n_tiles ? static_cast<uint32_t>(trellis[static_cast<int64_t>(tile_index) * words_per_tile + w])
                             : 0u;
      }
    }
    if (thread < tiles_here * kOutputTilesPerBlock) {
      const int u = thread / kOutputTilesPerBlock;
      const int sub = thread % kOutputTilesPerBlock;
      const int n_tile = n_tile_base + sub;
      const int tile_index = (kb + u) * n_tiles + n_tile;
      packed_bank_ids[u][sub] = n_tile < n_tiles ? bank_ids[tile_index] : 0;
    }

    if constexpr (TransitionBits != 0 && VectorStaging && !kQvqDebugDisableVecStaging) {
      const int vec_total = tiles_here * ROWS * kInputVecsPerRow;
      uint4* input4 = reinterpret_cast<uint4*>(input_tile[0]);
      const uint4* input4_base = reinterpret_cast<const uint4*>(input);
      for (int index = thread; index < vec_total; index += kThreads) {
        const int u = index / (ROWS * kInputVecsPerRow);
        const int cell = index - u * (ROWS * kInputVecsPerRow);
        const int row = cell / kInputVecsPerRow;
        const int vec = cell - row * kInputVecsPerRow;
        if (row < block_rows) {
          input4[index] = input4_base[
              ((static_cast<int64_t>(m0 + row) * size_k + (kb + u) * kLocalRingTileRows) *
                   static_cast<int>(sizeof(Scalar)) / 16) +
              vec];
        } else {
          input4[index] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    } else {
      for (int index = thread; index < tiles_here * ROWS * kLocalRingTileRows; index += kThreads) {
        const int u = index / (ROWS * kLocalRingTileRows);
        const int cell = index - u * (ROWS * kLocalRingTileRows);
        const int row = cell / kLocalRingTileRows;
        const int k_local_tile = cell - row * kLocalRingTileRows;
        input_tile[u][cell] = row < block_rows
            ? input[static_cast<int64_t>(m0 + row) * size_k + (kb + u) * kLocalRingTileRows + k_local_tile]
            : ScalarTraits<Scalar>::from_float(0.0f);
      }
    }
  };

  const int split = SplitK ? static_cast<int>(blockIdx.z) : 0;
  // The host rejects split products above INT_MAX so this latency-sensitive
  // partitioning stays in 32-bit registers.
  const int k_tile_begin = SplitK ? (k_tiles * split) / split_count : 0;
  const int k_tile_end = SplitK ? (k_tiles * (split + 1)) / split_count : k_tiles;
  for (int kb = k_tile_begin; kb < k_tile_end; kb += kBatchTiles) {
    const int tiles_here = min(kBatchTiles, k_tile_end - kb);
    stage_batch(kb, tiles_here);
    __syncthreads();

#pragma unroll
    for (int u = 0; u < kBatchTiles; ++u) {
      if (u < tiles_here) {
        for (int sub = 0; sub < kOutputTilesPerBlock; ++sub) {
          if (n_tile_base + sub >= n_tiles) {
            continue;
          }
          float weight;
          if constexpr (TransitionBits != 0) {
            weight = qvq_decode_local_ring_weight_fast<TransitionBits>(
                packed_words[u][sub], packed_bank_ids[u][sub], cached_levels, ring, k_local, bank_alt_id);
          } else {
            weight = qvq_decode_local_ring_weight_runtime(
                packed_words[u][sub], packed_bank_ids[u][sub], cached_levels, ring, k_local, transition_bits, bank_alt_id);
          }
#pragma unroll
          for (int r = 0; r < ROWS; ++r) {
            accumulator[r][sub] = fmaf(
                ScalarTraits<Scalar>::to_float(input_tile[u][r * kLocalRingTileRows + k_local]),
                weight,
                accumulator[r][sub]);
          }
        }
      }
    }
    __syncthreads();
  }

  for (int r = 0; r < block_rows; ++r) {
    for (int sub = 0; sub < kOutputTilesPerBlock; ++sub) {
      float total = accumulator[r][sub];
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(0xffffffffu, total, offset);
      }
      if (k_local == 0 && n_tile_base + sub < n_tiles) {
        const int64_t output_index = static_cast<int64_t>(m0 + r) * size_n +
            (n_tile_base + sub) * kLocalRingTileColumns + ring;
        if constexpr (SplitK) {
          partial_output[static_cast<int64_t>(split) * size_m * size_n + output_index] = total;
        } else {
          output[output_index] = ScalarTraits<OutputScalar>::from_float(total);
        }
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

// Register layouts for mma.sync.aligned.m16n8k16.row.col.  These match the
// native-N8 path used by Amplin and the PTX matrix-fragment specification.
// Keeping them explicit lets the cooperative decoder feed exactly the eight
// real output columns instead of materializing a padded N16 WMMA operand.
struct QvqMmaFragmentA {
  uint32_t values[4];
};

struct QvqMmaFragmentB {
  uint32_t values[2];
};

struct QvqMmaFragmentC {
  float values[4];
};

__device__ __forceinline__ void qvq_load_mma_fragment_a(
    QvqMmaFragmentA& fragment,
    const half* shared_source) {
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

__device__ __forceinline__ void qvq_mma_m16n8k16(
    const QvqMmaFragmentA& a,
    const QvqMmaFragmentB& b,
    QvqMmaFragmentC& c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c.values[0]),
        "=f"(c.values[1]),
        "=f"(c.values[2]),
        "=f"(c.values[3])
      : "r"(a.values[0]),
        "r"(a.values[1]),
        "r"(a.values[2]),
        "r"(a.values[3]),
        "r"(b.values[0]),
        "r"(b.values[1]),
        "f"(c.values[0]),
        "f"(c.values[1]),
        "f"(c.values[2]),
        "f"(c.values[3]));
}

// Hopper-only cooperative tensor-core path for W2/W2.5/W3/W3.5 at M<=16.
// One warp owns one N8 output tile; a small warp group therefore reuses the
// staged activation stripe across adjacent tiles. Each rate reconstructs the
// 16 unique ring transitions cooperatively, advances four adjacent states by
// recurrence, and consumes the decoded FP16 weights through native N8 MMA.
// The scalar LR kernel remains the
// fallback for other rates, row counts, dtypes, and unaligned views.
template <int TransitionBits, typename OutputScalar, bool SplitK, int OutputTiles, bool PermuteInputStaging = false>
__global__ __launch_bounds__(OutputTiles * 32) void qvq_gemv_local_ring_wmma_hopper_kernel(
    const half* __restrict__ input,
    const int32_t* __restrict__ trellis,
    const uint8_t* __restrict__ bank_ids,
    const half* __restrict__ levels,
    OutputScalar* __restrict__ output,
    float* __restrict__ partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    int bank_alt_id) {
  constexpr int kTransitionBits = TransitionBits;
  static_assert(kTransitionBits >= 4 && kTransitionBits <= 7);
  static_assert(!PermuteInputStaging || kTransitionBits == 7);
  constexpr int kRows = 16;
  constexpr int kBatchTiles = OutputTiles >= 8 ? 16 : 24;
  constexpr int kWmmaThreads = OutputTiles * 32;
  constexpr int kOutputTiles = OutputTiles;
  constexpr int kWordsPerTile = 4 * kTransitionBits;
  constexpr int kPaddedColumns = 16;
  constexpr bool kNativeN8 = true;
  constexpr bool kCompactNativeN8Storage = kNativeN8 && OutputTiles >= 8;
  constexpr int kDecodedColumns = kCompactNativeN8Storage ? kLocalRingTileColumns : kPaddedColumns;
  constexpr int kOutputColumns = kCompactNativeN8Storage ? 1 : kPaddedColumns;
  constexpr int kInputStride = kNativeN8 ? 40 : kLocalRingTileRows;
  constexpr bool kReadLevelsFromGlobal = kTransitionBits == 6 && OutputTiles >= 8;

  __shared__ half cached_levels[kPgc16LevelCount];
  __shared__ __align__(16) uint32_t packed_words[kBatchTiles][kOutputTiles][kWordsPerTile];
  __shared__ uint8_t packed_bank_ids[kBatchTiles][kOutputTiles];
  __shared__ __align__(16) half input_tile[kBatchTiles][kRows * kInputStride];
  __shared__ __align__(16) half decoded_weight[kOutputTiles][kLocalRingTileRows * kDecodedColumns];
  __shared__ __align__(16) float output_tile[kOutputTiles][kRows][kOutputColumns];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int n_tile_base = static_cast<int>(blockIdx.x) * kOutputTiles;
  const int m0 = static_cast<int>(blockIdx.y) * kRows;
  const int block_rows = min(kRows, size_m - m0);
  const int n_tiles = size_n / kLocalRingTileColumns;
  const int k_tiles = size_k / kLocalRingTileRows;
  const int split = SplitK ? static_cast<int>(blockIdx.z) : 0;
  const int k_tile_begin = SplitK ? (k_tiles * split) / split_count : 0;
  const int k_tile_end = SplitK ? (k_tiles * (split + 1)) / split_count : k_tiles;

  if constexpr (!kReadLevelsFromGlobal) {
    for (int index = thread; index < kPgc16LevelCount; index += kWmmaThreads) {
      cached_levels[index] = levels[index];
    }
    __syncthreads();
  }

  if constexpr (!kNativeN8) {
    // TB6 retains the accepted padded N16 WMMA consumer.
    for (int index = thread; index < kOutputTiles * kLocalRingTileRows * kPaddedColumns; index += kWmmaThreads) {
      const int cell = index % (kLocalRingTileRows * kPaddedColumns);
      const int col = cell % kPaddedColumns;
      if (col >= kLocalRingTileColumns) {
        reinterpret_cast<half*>(decoded_weight)[index] = __float2half(0.0f);
      }
    }
  }
  if constexpr (!kReadLevelsFromGlobal || !kNativeN8) {
    __syncthreads();
  }

  QvqMmaFragmentC native_accumulator = {};
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> input_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> weight_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  if constexpr (!kNativeN8) {
    wmma::fill_fragment(accumulator, 0.0f);
  }

  auto stage_batch = [&](int kb, int tiles_here) {
    constexpr int words_per_vec = kWordsPerTile / 4;
    const int vec_total = tiles_here * kOutputTiles * words_per_vec;
    uint4* words4 = reinterpret_cast<uint4*>(packed_words[0][0]);
    const uint4* trellis4 = reinterpret_cast<const uint4*>(trellis);
    for (int index = thread; index < vec_total; index += kWmmaThreads) {
      const int tile_slot = index / words_per_vec;
      const int w4 = index - tile_slot * words_per_vec;
      const int u = tile_slot / kOutputTiles;
      const int sub = tile_slot % kOutputTiles;
      const int n_tile = n_tile_base + sub;
      words4[tile_slot * words_per_vec + w4] =
          n_tile < n_tiles
              ? trellis4[(static_cast<int64_t>(kb + u) * n_tiles + n_tile) * words_per_vec + w4]
              : make_uint4(0u, 0u, 0u, 0u);
    }
    if (thread < tiles_here * kOutputTiles) {
      const int u = thread / kOutputTiles;
      const int sub = thread % kOutputTiles;
      const int n_tile = n_tile_base + sub;
      const int tile_index = (kb + u) * n_tiles + n_tile;
      packed_bank_ids[u][sub] = n_tile < n_tiles ? bank_ids[tile_index] : 0;
    }

    constexpr int input_vecs_per_row = kLocalRingTileRows * sizeof(half) / sizeof(uint4);
    constexpr int staged_vecs_per_row = kInputStride * sizeof(half) / sizeof(uint4);
    const int input_vec_total = tiles_here * kRows * input_vecs_per_row;
    uint4* input4 = reinterpret_cast<uint4*>(input_tile[0]);
    const uint4* input4_base = reinterpret_cast<const uint4*>(input);
    for (int index = thread; index < input_vec_total; index += kWmmaThreads) {
      const int u = index / (kRows * input_vecs_per_row);
      const int cell = index - u * (kRows * input_vecs_per_row);
      int row;
      int vec;
      if constexpr (PermuteInputStaging) {
        // Preserve the padded row-major tile while assigning each eight-lane
        // store transaction one chunk from every shared-bank group. The
        // logical source set is unchanged; only producer-lane ownership is
        // permuted, so the native matrix-load addresses need no change.
        const int lane_cell = index & 31;
        vec = lane_cell >> 3;
        row = ((cell >> 5) << 3) + ((5 * ((lane_cell & 7) - vec)) & 7);
      } else {
        row = cell / input_vecs_per_row;
        vec = cell - row * input_vecs_per_row;
      }
      input4[(u * kRows + row) * staged_vecs_per_row + vec] = row < block_rows
          ? input4_base[
                ((static_cast<int64_t>(m0 + row) * size_k + (kb + u) * kLocalRingTileRows) * sizeof(half) / 16) +
                vec]
          : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  for (int kb = k_tile_begin; kb < k_tile_end; kb += kBatchTiles) {
    const int tiles_here = min(kBatchTiles, k_tile_end - kb);
    stage_batch(kb, tiles_here);
    __syncthreads();

#pragma unroll 1
    for (int u = 0; u < tiles_here; ++u) {
      const int sub = warp;
        // Four lanes cooperate on each ring. Each lane extracts four disjoint
        // transitions once and advances a rate-specific sliding state across
        // four adjacent pairs. Every rate exchanges the preceding four-edge
        // group once; W3 retains only its rate-specific two-load planar
        // extraction before sharing the same recurrence.
        const int col = lane & 7;
        const int edge_group = lane >> 3;
        uint32_t edge_pack = 0;
        if constexpr (kTransitionBits == 6) {
          // W3 stores each planar block as a four-bit low plane followed by a
          // two-bit high plane. Four consecutive edges therefore need only
          // two packed-word loads.
          const int edge_block = col >> 1;
          const int edge_in_block = (col & 1) * 16 + edge_group * 4;
          const uint32_t low_codes =
              packed_words[u][sub][edge_block * kTransitionBits + edge_in_block / 8] >>
              (4 * (edge_in_block & 7));
          const uint32_t high_codes =
              packed_words[u][sub][edge_block * kTransitionBits + 4 + edge_in_block / 16] >>
              (2 * (edge_in_block & 15));
          // Expand four low nibbles and four high dibits into adjacent six-bit
          // slots with two mask/shift stages instead of four scalar extracts.
          uint32_t low_spread = (low_codes & 0x00ffu) | ((low_codes & 0xff00u) << 4);
          low_spread = (low_spread & 0x0000f00fu) | ((low_spread & 0x000f00f0u) << 2);
          uint32_t high_spread = (high_codes & 0x0fu) | ((high_codes & 0xf0u) << 8);
          high_spread =
              ((high_spread & 0x00003003u) | ((high_spread & 0x0000c00cu) << 4)) << 4;
          edge_pack = low_spread | high_spread;
        } else if constexpr (kTransitionBits == 7) {
          // W3.5 stores four-bit low, two-bit middle, and one-bit top
          // planes. Four adjacent edges fit in one word from every plane.
          const int edge_block = col >> 1;
          const int edge_in_block = (col & 1) * 16 + edge_group * 4;
          const int word_base = edge_block * kTransitionBits;
          const uint32_t low_pack =
              (packed_words[u][sub][word_base + edge_in_block / 8] >>
               (4 * (edge_in_block & 7))) &
              0xffffu;
          const uint32_t middle_pack =
              (packed_words[u][sub][word_base + 4 + edge_in_block / 16] >>
               (2 * (edge_in_block & 15))) &
              0xffu;
          const uint32_t top_pack =
              (packed_words[u][sub][word_base + 6] >> edge_in_block) & 0xfu;
          // Spread the packed planes directly into four adjacent seven-bit
          // slots. This is bit-equivalent to four scalar extract/shift/or
          // iterations but keeps the Hopper integer instruction stream short.
          uint32_t low_spread = (low_pack & 0x00ffu) | ((low_pack & 0xff00u) << 6);
          low_spread = (low_spread & 0x0003c00fu) | ((low_spread & 0x003c00f0u) << 3);
          uint32_t middle_spread =
              (middle_pack & 0x0fu) | ((middle_pack & 0xf0u) << 10);
          middle_spread =
              ((middle_spread & 0x0000c003u) | ((middle_spread & 0x0003000cu) << 5)) << 4;
          uint32_t top_spread = (top_pack & 0x3u) | ((top_pack & 0xcu) << 12);
          top_spread =
              ((top_spread & 0x00004001u) | ((top_spread & 0x00008002u) << 6)) << 6;
          edge_pack = low_spread | middle_spread | top_spread;
        } else {
          const int first_edge = col * kLocalRingSteps + edge_group * 4;
          const int planar_block = first_edge >> 5;
          const int planar_lane = first_edge & 31;
          const uint32_t low_word = packed_words[u][sub][
              planar_block * kTransitionBits + planar_lane / 8];
          const uint32_t low_pack = (low_word >> (4 * (planar_lane & 7))) & 0xffffu;
          edge_pack = low_pack;
          if constexpr (kTransitionBits == 5) {
            // Expand four contiguous low nibbles into five-bit slots, then
            // spread the four high-plane bits into the open slot in each code.
            edge_pack = (low_pack & 0x000fu) |
                ((low_pack & 0x00f0u) << 1) |
                ((low_pack & 0x0f00u) << 2) |
                ((low_pack & 0xf000u) << 3);
            const uint32_t high_word = packed_words[u][sub][planar_block * kTransitionBits + 4];
            const uint32_t high_nibble = (high_word >> planar_lane) & 0xfu;
            edge_pack |= (high_nibble * 0x11110u) & 0x84210u;
          }
        }
        const int previous_lane = col + ((edge_group + 3) & 3) * 8;
        const uint32_t previous_pack = __shfl_sync(0xffffffffu, edge_pack, previous_lane);
        const int pair_base = edge_group * 4;
        constexpr int kEdgeCount = (15 + kTransitionBits) / kTransitionBits;
        uint32_t state = 0;
#pragma unroll
        for (int j = 0; j < kEdgeCount; ++j) {
          const int relative_edge = 1 - kEdgeCount + j;
          const int pack_index = relative_edge < 0 ? 4 + relative_edge : relative_edge;
          const uint32_t source_pack = relative_edge < 0 ? previous_pack : edge_pack;
          const uint32_t transition =
              (source_pack >> (pack_index * kTransitionBits)) & ((1u << kTransitionBits) - 1u);
          state = ((state << kTransitionBits) | transition) & 0xffffu;
        }
        const uint32_t bank = ((static_cast<uint32_t>(packed_bank_ids[u][sub]) >> col) & 1u) *
            static_cast<uint32_t>(bank_alt_id);
        const uint32_t bank_mask = pgc16_v2_bank_mask<kTransitionBits>(bank);
        uint32_t decoded_pairs[4] = {};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int pair = pair_base + q;
          const uint32_t mixed = pgc16_mix(state ^ bank_mask);
          if constexpr (kNativeN8) {
            half high_level;
            half low_level;
            if constexpr (kReadLevelsFromGlobal) {
              high_level = __ldg(levels + (mixed >> 8));
              low_level = __ldg(levels + (mixed & 0xffu));
            } else {
              high_level = cached_levels[mixed >> 8];
              low_level = cached_levels[mixed & 0xffu];
            }
            decoded_pairs[q] = static_cast<uint32_t>(__half_as_ushort(high_level)) |
                (static_cast<uint32_t>(__half_as_ushort(low_level)) << 16);
          } else {
            decoded_weight[sub][pair * 2 * kPaddedColumns + col] = cached_levels[mixed >> 8];
            decoded_weight[sub][(pair * 2 + 1) * kPaddedColumns + col] = cached_levels[mixed & 0xffu];
          }
          if (q < 3) {
            const uint32_t transition =
                (edge_pack >> ((q + 1) * kTransitionBits)) & ((1u << kTransitionBits) - 1u);
            state = ((state << kTransitionBits) | transition) & 0xffffu;
          }
        }
        if constexpr (kNativeN8) {
          // Transpose producer lanes into the exact native B-fragment layout.
          // The four adjacent pairs form one aligned vector store; consumer
          // lanes then read four conflict-free fragment planes.
          reinterpret_cast<uint4*>(&decoded_weight[sub][0])[edge_group * 8 + col] =
              make_uint4(decoded_pairs[0], decoded_pairs[1], decoded_pairs[2], decoded_pairs[3]);
        }
      __syncwarp();

      if (n_tile_base + warp < n_tiles) {
        if constexpr (kNativeN8) {
          const int address_row = (lane & 7) + ((lane >> 3) & 1) * 8;
          const int address_column = (lane >> 4) * 8;
          const uint32_t* fragment_words =
              reinterpret_cast<const uint32_t*>(&decoded_weight[warp][0]);
          QvqMmaFragmentA native_a;
          QvqMmaFragmentB native_b;
          qvq_load_mma_fragment_a(
              native_a,
              input_tile[u] + address_row * kInputStride + address_column);
          native_b.values[0] = fragment_words[lane];
          native_b.values[1] = fragment_words[32 + lane];
          qvq_mma_m16n8k16(native_a, native_b, native_accumulator);
          qvq_load_mma_fragment_a(
              native_a,
              input_tile[u] + address_row * kInputStride + 16 + address_column);
          native_b.values[0] = fragment_words[64 + lane];
          native_b.values[1] = fragment_words[96 + lane];
          qvq_mma_m16n8k16(native_a, native_b, native_accumulator);
        } else {
          wmma::load_matrix_sync(input_fragment, input_tile[u], kLocalRingTileRows);
          wmma::load_matrix_sync(weight_fragment, &decoded_weight[warp][0], kPaddedColumns);
          wmma::mma_sync(accumulator, input_fragment, weight_fragment, accumulator);
          wmma::load_matrix_sync(
              input_fragment, input_tile[u] + 16, kLocalRingTileRows);
          wmma::load_matrix_sync(
              weight_fragment, &decoded_weight[warp][16 * kPaddedColumns], kPaddedColumns);
          wmma::mma_sync(accumulator, input_fragment, weight_fragment, accumulator);
        }
      }
    }
    __syncthreads();
  }

  if constexpr (kNativeN8) {
    if (n_tile_base + warp < n_tiles) {
      const int quad = lane >> 2;
      const int output_column = (n_tile_base + warp) * kLocalRingTileColumns + (lane & 3) * 2;
      const int output_rows[2] = {quad, quad + 8};
#pragma unroll
      for (int row_group = 0; row_group < 2; ++row_group) {
        const int row = output_rows[row_group];
        if (row < block_rows) {
#pragma unroll
          for (int column_pair = 0; column_pair < 2; ++column_pair) {
            const int64_t output_index = static_cast<int64_t>(m0 + row) * size_n +
                output_column + column_pair;
            const float value = native_accumulator.values[row_group * 2 + column_pair];
            if constexpr (SplitK) {
              partial_output[static_cast<int64_t>(split) * size_m * size_n + output_index] = value;
            } else {
              output[output_index] = ScalarTraits<OutputScalar>::from_float(value);
            }
          }
        }
      }
    }
  } else {
    if (n_tile_base + warp < n_tiles) {
      wmma::store_matrix_sync(&output_tile[warp][0][0], accumulator, kPaddedColumns, wmma::mem_row_major);
    }
    __syncthreads();
    if (n_tile_base + warp < n_tiles) {
      for (int index = lane; index < block_rows * kLocalRingTileColumns; index += 32) {
        const int row = index / kLocalRingTileColumns;
        const int col = index % kLocalRingTileColumns;
        const int64_t output_index = static_cast<int64_t>(m0 + row) * size_n +
            (n_tile_base + warp) * kLocalRingTileColumns + col;
        if constexpr (SplitK) {
          partial_output[static_cast<int64_t>(split) * size_m * size_n + output_index] =
              output_tile[warp][row][col];
        } else {
          output[output_index] = ScalarTraits<OutputScalar>::from_float(output_tile[warp][row][col]);
        }
      }
    }
  }
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

int qvq_local_ring_split_count(
    int size_m,
    int size_k,
    int size_n,
    const QvqCudaDeviceConfig& device_config) {
  const int rows = qvq_rows_for_m(size_m);
  const int64_t base_blocks =
      static_cast<int64_t>(size_n / kLocalRingTileColumns) * ((size_m + rows - 1) / rows);
  const int k_tiles = size_k / kLocalRingTileRows;
  const int64_t low_residency_blocks = std::max<int64_t>(32, device_config.sm_count / 2);
  const int64_t saturated_blocks = static_cast<int64_t>(device_config.sm_count) * 3;
  int split_count;
  if (device_config.major >= 12) {
    // Blackwell's one-row scheduler benefits from a few independent K waves
    // even when N already exposes many resident blocks. For wider row
    // specializations, keep the split count proportional to the available
    // blocks and avoid the reduction overhead once N is saturated.
    if (rows == 1) {
      split_count = base_blocks < low_residency_blocks ? 4 : base_blocks < saturated_blocks ? 16 : 4;
    } else if (base_blocks < low_residency_blocks) {
      split_count = rows <= 8 ? 32 : 8;
    } else if (base_blocks < saturated_blocks) {
      split_count = 16;
    } else {
      split_count = 1;
    }
  } else if (base_blocks >= saturated_blocks) {
    // Ada reaches full residency with one block per N8 tile. The larger
    // K-width MLP shape is the exception for M=1, where four K waves hide
    // the long local-ring recurrence.
    split_count = rows == 1 && k_tiles >= 128 ? 4 : 1;
  } else if (rows == 1) {
    split_count = base_blocks < 64 ? 16 : 8;
  } else if (base_blocks < low_residency_blocks) {
    split_count = rows <= 8 ? 8 : 8;
  } else {
    split_count = 4;
  }
  return std::max(1, std::min(split_count, std::min(k_tiles, 64)));
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
    const at::Tensor& levels,
    at::Tensor& output,
    int transition_bits,
    int bank_mode,
    int bank_alt_id,
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
  const bool specialized = VectorSize == 2 ? qvq_tb_specialized_v2(transition_bits)
                                           : qvq_tb_specialized_v4(transition_bits);
  if (specialized && qvq_vec_aligned(trellis_ptr, input_ptr)) {
#define QVQ_LAUNCH_TB(ROWS, TB)                                                                                    \
    qvq_gemv_kernel<Scalar, OutputScalar, ROWS, VectorSize, TB>                                                    \
        <<<grid, kThreads, 0, stream>>>(                                                                           \
            input_ptr, trellis_ptr, bank_ids_ptr, levels_ptr, output_ptr, static_cast<int>(input.size(0)),         \
            static_cast<int>(input.size(1)), static_cast<int>(output.size(1)), transition_bits, bank_mode,        \
            bank_alt_id)
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
      input_ptr, trellis_ptr, bank_ids_ptr, levels_ptr, output_ptr, static_cast<int>(input.size(0)),               \
      static_cast<int>(input.size(1)), static_cast<int>(output.size(1)), transition_bits, bank_mode, bank_alt_id)
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
    int bank_mode,
    int bank_alt_id,
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
  const bool specialized = VectorSize == 2 ? qvq_tb_specialized_v2(transition_bits)
                                           : qvq_tb_specialized_v4(transition_bits);
  if (specialized && qvq_vec_aligned(trellis_ptr, input_ptr)) {
#define QVQ_SPLITK_LAUNCH_TB(ROWS, TB)                                                                            \
    qvq_gemv_splitk_kernel<Scalar, ROWS, VectorSize, TB>                                                          \
        <<<grid, kThreads, 0, stream>>>(                                                                          \
            input_ptr, trellis_ptr, bank_ids_ptr, levels_ptr, partial_ptr, static_cast<int>(input.size(0)),       \
            static_cast<int>(input.size(1)), static_cast<int>(output.size(1)), split_count, transition_bits,      \
            bank_mode, bank_alt_id)
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
        partial_ptr, output_ptr, static_cast<int>(output.size(0)), static_cast<int>(output.size(1)), split_count);
    return;
  }
#define QVQ_SPLITK_LAUNCH(ROWS)                                                                                   \
  qvq_gemv_splitk_kernel<Scalar, ROWS, VectorSize, 0><<<grid, kThreads, 0, stream>>>(                             \
      input_ptr, trellis_ptr, bank_ids_ptr, levels_ptr, partial_ptr, static_cast<int>(input.size(0)),             \
      static_cast<int>(input.size(1)), static_cast<int>(output.size(1)), split_count, transition_bits,            \
      bank_mode, bank_alt_id)
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
      partial_ptr, output_ptr, static_cast<int>(output.size(0)), static_cast<int>(output.size(1)), split_count);
}

// LR32 specializes every legal rate (W1 through W3.5) so the planar edge
// extraction and local-ring recurrence are compile-time unrolled. The host
// dispatch still keeps the rate/type matrix explicit for predictable kernels.
template <typename Scalar, typename OutputScalar, bool SplitK, int OutputTilesPerBlock, bool VectorStaging>
void launch_qvq_local_ring_gemv_impl(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& bank_ids,
    const at::Tensor& levels,
    at::Tensor* partial_output,
    at::Tensor& output,
    int transition_bits,
    int split_count,
    int bank_alt_id,
    cudaStream_t stream) {
  const int rows = qvq_rows_for_m(static_cast<int>(input.size(0)));
  const dim3 grid(
      static_cast<unsigned int>(
          (output.size(1) / kLocalRingTileColumns + OutputTilesPerBlock - 1) / OutputTilesPerBlock),
      static_cast<unsigned int>((input.size(0) + rows - 1) / rows),
      static_cast<unsigned int>(SplitK ? split_count : 1));
  const Scalar* input_ptr = reinterpret_cast<const Scalar*>(input.const_data_ptr());
  const half* levels_ptr = reinterpret_cast<const half*>(levels.const_data_ptr());
  OutputScalar* output_ptr = reinterpret_cast<OutputScalar*>(output.mutable_data_ptr());
  float* partial_ptr = partial_output == nullptr ? nullptr : partial_output->mutable_data_ptr<float>();
  const int32_t* trellis_ptr = trellis.const_data_ptr<int32_t>();
  const uint8_t* bank_ids_ptr = bank_ids.const_data_ptr<uint8_t>();

#define QVQ_LR_LAUNCH_TB(ROWS, TB)                                                                                 \
  qvq_gemv_local_ring_kernel<Scalar, OutputScalar, ROWS, TB, SplitK, OutputTilesPerBlock, VectorStaging>          \
      <<<grid, kThreads, 0, stream>>>(                                                                             \
      input_ptr, trellis_ptr, bank_ids_ptr, levels_ptr, output_ptr, partial_ptr,                                  \
      static_cast<int>(input.size(0)), static_cast<int>(input.size(1)), static_cast<int>(output.size(1)),          \
      split_count, transition_bits, bank_alt_id)
#define QVQ_LR_LAUNCH(ROWS) QVQ_LR_LAUNCH_TB(ROWS, 0)
#define QVQ_LR_LAUNCH_ROWS(M)                                                                                      \
  if (rows == 1) {                                                                                                 \
    QVQ_LR_LAUNCH_TB(1, M);                                                                                        \
  } else if (rows == 8) {                                                                                          \
    QVQ_LR_LAUNCH_TB(8, M);                                                                                        \
  } else if (rows == 16) {                                                                                         \
    QVQ_LR_LAUNCH_TB(16, M);                                                                                       \
  } else {                                                                                                         \
    QVQ_LR_LAUNCH_TB(kRowsPerBlock, M);                                                                            \
  }

  switch (transition_bits) {
    case 2:
      QVQ_LR_LAUNCH_ROWS(2);
      break;
    case 3:
      QVQ_LR_LAUNCH_ROWS(3);
      break;
    case 4:
      QVQ_LR_LAUNCH_ROWS(4);
      break;
    case 5:
      QVQ_LR_LAUNCH_ROWS(5);
      break;
    case 6:
      QVQ_LR_LAUNCH_ROWS(6);
      break;
    case 7:
      QVQ_LR_LAUNCH_ROWS(7);
      break;
    default:
      TORCH_CHECK(false, "LR32 transition_bits must be in [2, 7]");
  }
#undef QVQ_LR_LAUNCH_ROWS
#undef QVQ_LR_LAUNCH
#undef QVQ_LR_LAUNCH_TB

  if constexpr (SplitK) {
    const int64_t output_values = output.numel();
    const int reduction_blocks = static_cast<int>((output_values + kThreads - 1) / kThreads);
    qvq_reduce_splitk_kernel<OutputScalar><<<reduction_blocks, kThreads, 0, stream>>>(
        partial_ptr, output_ptr, static_cast<int>(output.size(0)), static_cast<int>(output.size(1)), split_count);
  }
}

template <
    int TransitionBits,
    typename OutputScalar,
    bool SplitK,
    int OutputTiles = kHopperWmmaOutputTiles,
    bool PermuteInputStaging = false>
void launch_qvq_local_ring_wmma_hopper(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& bank_ids,
    const at::Tensor& levels,
    at::Tensor* partial_output,
    at::Tensor& output,
    int split_count,
    int bank_alt_id,
    cudaStream_t stream) {
  const dim3 grid(
      static_cast<unsigned int>((output.size(1) / kLocalRingTileColumns + OutputTiles - 1) / OutputTiles),
      static_cast<unsigned int>((input.size(0) + 15) / 16),
      static_cast<unsigned int>(SplitK ? split_count : 1));
  const half* input_ptr = reinterpret_cast<const half*>(input.const_data_ptr());
  const half* levels_ptr = reinterpret_cast<const half*>(levels.const_data_ptr());
  OutputScalar* output_ptr = reinterpret_cast<OutputScalar*>(output.mutable_data_ptr());
  float* partial_ptr = partial_output == nullptr ? nullptr : partial_output->mutable_data_ptr<float>();
  const int32_t* trellis_ptr = trellis.const_data_ptr<int32_t>();
  const uint8_t* bank_ids_ptr = bank_ids.const_data_ptr<uint8_t>();
  qvq_gemv_local_ring_wmma_hopper_kernel<
      TransitionBits, OutputScalar, SplitK, OutputTiles, PermuteInputStaging>
      <<<grid, OutputTiles * 32, 0, stream>>>(
          input_ptr,
          trellis_ptr,
          bank_ids_ptr,
          levels_ptr,
          output_ptr,
          partial_ptr,
          static_cast<int>(input.size(0)),
          static_cast<int>(input.size(1)),
          static_cast<int>(output.size(1)),
          split_count,
          bank_alt_id);
  if constexpr (SplitK) {
    const int64_t output_values = output.numel();
    const int reduction_blocks = static_cast<int>((output_values + kThreads - 1) / kThreads);
    qvq_reduce_splitk_kernel<OutputScalar><<<reduction_blocks, kThreads, 0, stream>>>(
        partial_ptr,
        output_ptr,
        static_cast<int>(output.size(0)),
        static_cast<int>(output.size(1)),
        split_count);
  }
}

template <typename OutputScalar, bool SplitK>
void launch_qvq_local_ring_wmma_hopper_dispatch(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& bank_ids,
    const at::Tensor& levels,
    at::Tensor* partial_output,
    at::Tensor& output,
    int transition_bits,
    int split_count,
    int bank_alt_id,
    cudaStream_t stream) {
#define QVQ_LR_LAUNCH_HOPPER_WMMA(BITS, TILES)                                                                     \
  launch_qvq_local_ring_wmma_hopper<BITS, OutputScalar, SplitK, TILES>(                                            \
      input, trellis, bank_ids, levels, partial_output, output, split_count, bank_alt_id, stream)
  switch (transition_bits) {
    case 4:
      QVQ_LR_LAUNCH_HOPPER_WMMA(4, 4);
      break;
    case 5:
      QVQ_LR_LAUNCH_HOPPER_WMMA(5, 4);
      break;
    case 6:
      if (input.size(1) <= 2048 && output.size(1) >= 8192) {
        QVQ_LR_LAUNCH_HOPPER_WMMA(6, 8);
      } else {
        QVQ_LR_LAUNCH_HOPPER_WMMA(6, 4);
      }
      break;
    case 7:
      if (input.size(0) <= 8) {
        launch_qvq_local_ring_wmma_hopper<7, OutputScalar, SplitK, 4, true>(
            input, trellis, bank_ids, levels, partial_output, output, split_count, bank_alt_id, stream);
      } else {
        QVQ_LR_LAUNCH_HOPPER_WMMA(7, 4);
      }
      break;
    default:
      TORCH_CHECK(false, "Hopper cooperative WMMA requires transition_bits in [4, 7]");
  }
#undef QVQ_LR_LAUNCH_HOPPER_WMMA
}

// Two adjacent N8 tiles amortize the LR32 input staging and launch overhead
// for M=1. For larger row blocks the second accumulator tile increases
// register pressure enough to regress throughput, so retain one tile/block.
template <typename Scalar, typename OutputScalar, bool SplitK>
void launch_qvq_local_ring_gemv(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& bank_ids,
    const at::Tensor& levels,
    at::Tensor* partial_output,
    at::Tensor& output,
    int transition_bits,
    int split_count,
    int bank_alt_id,
    cudaStream_t stream,
    bool prefer_tb6_vector) {
  // A 24-word TB=6 tile has a six-vector stride. On SM89/SM120 this access
  // pattern defeats the coalescer for the wider row specializations; scalar
  // staging is measurably faster and avoids the TB=6 occupancy cliff.
  const bool vector_staging =
      (transition_bits != 6 || prefer_tb6_vector) &&
      qvq_vec_aligned(trellis.const_data_ptr(), input.const_data_ptr());
  if (qvq_rows_for_m(static_cast<int>(input.size(0))) == 1 && vector_staging) {
    launch_qvq_local_ring_gemv_impl<Scalar, OutputScalar, SplitK, 2, true>(
        input,
        trellis,
        bank_ids,
        levels,
        partial_output,
        output,
        transition_bits,
        split_count,
        bank_alt_id,
        stream);
  } else if (qvq_rows_for_m(static_cast<int>(input.size(0))) == 1) {
    launch_qvq_local_ring_gemv_impl<Scalar, OutputScalar, SplitK, 2, false>(
        input,
        trellis,
        bank_ids,
        levels,
        partial_output,
        output,
        transition_bits,
        split_count,
        bank_alt_id,
        stream);
  } else if (vector_staging) {
    launch_qvq_local_ring_gemv_impl<Scalar, OutputScalar, SplitK, 1, true>(
        input,
        trellis,
        bank_ids,
        levels,
        partial_output,
        output,
        transition_bits,
        split_count,
        bank_alt_id,
        stream);
  } else {
    launch_qvq_local_ring_gemv_impl<Scalar, OutputScalar, SplitK, 1, false>(
        input,
        trellis,
        bank_ids,
        levels,
        partial_output,
        output,
        transition_bits,
        split_count,
        bank_alt_id,
        stream);
  }
}

at::Tensor qvq_gemv_cuda_local_ring_impl(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    int64_t transition_bits,
    int64_t out_features,
    bool output_fp32,
    const at::Tensor& bank_ids,
    int64_t bank_alt_id,
    int64_t split_count_override) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(trellis.is_cuda(), "trellis must be a CUDA tensor");
  TORCH_CHECK(levels.is_cuda(), "PGC16 levels must be a CUDA tensor");
  TORCH_CHECK(bank_ids.is_cuda(), "LR32 bank selectors must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2 && trellis.dim() == 2 && levels.dim() == 1 && bank_ids.dim() == 1,
              "LR32 GEMV expects rank-two input/trellis and rank-one levels/selectors");
  TORCH_CHECK(transition_bits >= 2 && transition_bits <= 7,
              "LR32 transition_bits must be in [2, 7] for W1 through W3.5");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
              "LR32 input must have dtype float16 or bfloat16");
  TORCH_CHECK(levels.scalar_type() == at::kHalf,
              "PGC16 levels must preserve the canonical float16 bit patterns");
  TORCH_CHECK(trellis.scalar_type() == at::kInt, "LR32 trellis must have dtype int32");
  TORCH_CHECK(bank_ids.scalar_type() == at::kByte, "LR32 bank selectors must use uint8");
  TORCH_CHECK(input.device() == trellis.device() && input.device() == levels.device() &&
                  input.device() == bank_ids.device(),
              "LR32 tensors must share one CUDA device");
  TORCH_CHECK(input.is_contiguous() && trellis.is_contiguous() && levels.is_contiguous() && bank_ids.is_contiguous(),
              "LR32 tensors must be contiguous");
  TORCH_CHECK(bank_alt_id >= 1 && bank_alt_id <= 3, "LR32 alternative-bank ID must be in [1, 3]");

  const int64_t size_m = input.size(0);
  const int64_t size_k = input.size(1);
  TORCH_CHECK(size_m >= 0, "input row count must be non-negative");
  TORCH_CHECK(size_k > 0 && size_k % kLocalRingTileRows == 0,
              "LR32 input width must be positive and divisible by 32");
  TORCH_CHECK(out_features > 0 && out_features % kLocalRingTileColumns == 0,
              "LR32 out_features must be positive and divisible by 8");
  TORCH_CHECK(size_m <= std::numeric_limits<int>::max() && size_k <= std::numeric_limits<int>::max() &&
                  out_features <= std::numeric_limits<int>::max(),
              "LR32 dimensions exceed the int32 kernel limit");
  TORCH_CHECK(levels.numel() == kPgc16LevelCount, "PGC16 levels must have shape (256)");
  const int64_t k_tiles = size_k / kLocalRingTileRows;
  const int64_t tile_count = (size_k / kLocalRingTileRows) * (out_features / kLocalRingTileColumns);
  TORCH_CHECK(
      tile_count <= std::numeric_limits<int>::max(),
      "LR32 tile count exceeds the int32 kernel index limit");
  TORCH_CHECK(split_count_override >= 0, "LR32 split_count must be non-negative");
  const int64_t max_split_count = std::min<int64_t>(k_tiles, 64);
  TORCH_CHECK(
      split_count_override == 0 || split_count_override <= max_split_count,
      "LR32 split_count must be in [1, min(K/32, 64)]");
  TORCH_CHECK(
      split_count_override == 0 ||
          k_tiles <= std::numeric_limits<int>::max() / split_count_override,
      "LR32 split_count overflows the int32 kernel partition limit");
  TORCH_CHECK(trellis.sizes() == at::IntArrayRef({tile_count, 4 * transition_bits}),
              "LR32 trellis shape must match K32, N8, and transition_bits");
  TORCH_CHECK(bank_ids.numel() == tile_count,
              "LR32 bank selectors must contain one packed byte per K32 x N8 trellis tile");

  if (size_m == 0) {
    return at::empty(
        {0, out_features}, input.options().dtype(output_fp32 ? at::kFloat : input.scalar_type()));
  }

  const c10::cuda::CUDAGuard device_guard(input.device());
  const QvqCudaDeviceConfig& device_config = qvq_cuda_device_config(input.get_device());
  TORCH_CHECK(device_config.major >= 8, "QVQ CUDA requires compute capability >= 8.0");
  const int rows = qvq_rows_for_m(static_cast<int>(size_m));
  const int64_t row_blocks = (size_m + rows - 1) / rows;
  TORCH_CHECK(
      row_blocks <= device_config.max_grid_y,
      "LR32 input row count requires ",
      row_blocks,
      " CUDA grid-Y blocks, exceeding the device limit of ",
      device_config.max_grid_y);

  at::Tensor output = at::empty(
      {size_m, out_features}, output_fp32 ? input.options().dtype(at::kFloat) : input.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const int automatic_split_count = qvq_local_ring_split_count(
      static_cast<int>(size_m),
      static_cast<int>(size_k),
      static_cast<int>(out_features),
      device_config);
  int split_count =
      split_count_override == 0 ? automatic_split_count : static_cast<int>(split_count_override);
  TORCH_CHECK(split_count >= 1 && split_count <= max_split_count,
              "LR32 split_count must be in [1, min(K/32, 64)]");
  TORCH_CHECK(
      k_tiles <= std::numeric_limits<int>::max() / split_count,
      "LR32 split_count overflows the int32 kernel partition limit");

  const bool use_hopper_cooperative_wmma = device_config.major == 9 &&
      transition_bits >= 4 && transition_bits <= 7 &&
      size_m <= 16 &&
      out_features >= 512 &&
      input.scalar_type() == at::kHalf && qvq_vec_aligned(input.const_data_ptr(), trellis.const_data_ptr());
  // Four-output-tile WMMA leaves only 256 blocks for the Llama gate/up
  // projection on a 132-SM H200. Two K partitions supply a second scheduling
  // wave and are consistently faster than either one or four partitions.
  if (split_count_override == 0 && use_hopper_cooperative_wmma && transition_bits == 6 &&
      size_k <= 2048 && out_features >= 8192) {
    split_count = 2;
  }
  // N512 exposes only 16 cooperative N32 blocks before K partitioning. On
  // Hopper, 16 partitions provide two complete CTA waves and consistently
  // beat the previous eight-partition policy across M1-M16.
  if (split_count_override == 0 && use_hopper_cooperative_wmma && transition_bits == 6 &&
      size_k <= 2048 && out_features == 512) {
    split_count = 16;
  }
  at::Tensor partial_output;

  if (use_hopper_cooperative_wmma && split_count > 1 && output_fp32) {
    partial_output = at::empty({split_count, size_m, out_features}, input.options().dtype(at::kFloat));
    launch_qvq_local_ring_wmma_hopper_dispatch<float, true>(
        input, trellis, bank_ids, levels, &partial_output, output, static_cast<int>(transition_bits), split_count,
        static_cast<int>(bank_alt_id), stream);
  } else if (use_hopper_cooperative_wmma && split_count > 1) {
    partial_output = at::empty({split_count, size_m, out_features}, input.options().dtype(at::kFloat));
    launch_qvq_local_ring_wmma_hopper_dispatch<half, true>(
        input, trellis, bank_ids, levels, &partial_output, output, static_cast<int>(transition_bits), split_count,
        static_cast<int>(bank_alt_id), stream);
  } else if (use_hopper_cooperative_wmma && output_fp32) {
    launch_qvq_local_ring_wmma_hopper_dispatch<float, false>(
        input, trellis, bank_ids, levels, nullptr, output, static_cast<int>(transition_bits), 1,
        static_cast<int>(bank_alt_id), stream);
  } else if (use_hopper_cooperative_wmma) {
    launch_qvq_local_ring_wmma_hopper_dispatch<half, false>(
        input, trellis, bank_ids, levels, nullptr, output, static_cast<int>(transition_bits), 1,
        static_cast<int>(bank_alt_id), stream);
  } else if (split_count > 1) {
    partial_output = at::empty({split_count, size_m, out_features}, input.options().dtype(at::kFloat));
    if (input.scalar_type() == at::kHalf && output_fp32) {
      launch_qvq_local_ring_gemv<half, float, true>(
          input, trellis, bank_ids, levels, &partial_output, output, static_cast<int>(transition_bits),
          split_count, static_cast<int>(bank_alt_id), stream, device_config.major == 9);
    } else if (input.scalar_type() == at::kHalf) {
      launch_qvq_local_ring_gemv<half, half, true>(
          input, trellis, bank_ids, levels, &partial_output, output, static_cast<int>(transition_bits),
          split_count, static_cast<int>(bank_alt_id), stream, device_config.major == 9);
    } else if (output_fp32) {
      launch_qvq_local_ring_gemv<nv_bfloat16, float, true>(
          input, trellis, bank_ids, levels, &partial_output, output, static_cast<int>(transition_bits),
          split_count, static_cast<int>(bank_alt_id), stream, device_config.major == 9);
    } else {
      launch_qvq_local_ring_gemv<nv_bfloat16, nv_bfloat16, true>(
          input, trellis, bank_ids, levels, &partial_output, output, static_cast<int>(transition_bits),
          split_count, static_cast<int>(bank_alt_id), stream, device_config.major == 9);
    }
  } else if (input.scalar_type() == at::kHalf && output_fp32) {
    launch_qvq_local_ring_gemv<half, float, false>(
        input, trellis, bank_ids, levels, nullptr, output, static_cast<int>(transition_bits), 1,
        static_cast<int>(bank_alt_id), stream, device_config.major == 9);
  } else if (input.scalar_type() == at::kHalf) {
    launch_qvq_local_ring_gemv<half, half, false>(
        input, trellis, bank_ids, levels, nullptr, output, static_cast<int>(transition_bits), 1,
        static_cast<int>(bank_alt_id), stream, device_config.major == 9);
  } else if (output_fp32) {
    launch_qvq_local_ring_gemv<nv_bfloat16, float, false>(
        input, trellis, bank_ids, levels, nullptr, output, static_cast<int>(transition_bits), 1,
        static_cast<int>(bank_alt_id), stream, device_config.major == 9);
  } else {
    launch_qvq_local_ring_gemv<nv_bfloat16, nv_bfloat16, false>(
        input, trellis, bank_ids, levels, nullptr, output, static_cast<int>(transition_bits), 1,
        static_cast<int>(bank_alt_id), stream, device_config.major == 9);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
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
    int64_t bank_alt_id) {
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

  if (size_m == 0) {
    return at::empty(
        {0, out_features}, input.options().dtype(output_fp32 ? at::kFloat : input.scalar_type()));
  }

  const c10::cuda::CUDAGuard device_guard(input.device());
  const QvqCudaDeviceConfig& device_config = qvq_cuda_device_config(input.get_device());
  TORCH_CHECK(device_config.major >= 8, "QVQ CUDA requires compute capability >= 8.0");

  at::Tensor output = at::empty(
      {size_m, out_features}, output_fp32 ? input.options().dtype(at::kFloat) : input.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const int64_t n_tiles = out_features / kTileColumns;
  const int64_t m_stripes = (size_m + kRowsPerBlock - 1) / kRowsPerBlock;
  const int64_t base_blocks = n_tiles * m_stripes;
  const int64_t target_blocks = static_cast<int64_t>(device_config.sm_count) * 6;
  const int64_t k_tiles = size_k / kTileRows;
  const int split_count = base_blocks >= 384 ? 1 : static_cast<int>(std::min(
      std::min((target_blocks + base_blocks - 1) / base_blocks, k_tiles), static_cast<int64_t>(64)));
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
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, static_cast<int>(bank_mode), static_cast<int>(bank_alt_id), stream);
    } else if (input.scalar_type() == at::kHalf) {
      launch_qvq_gemv_splitk<half, half>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, static_cast<int>(bank_mode), static_cast<int>(bank_alt_id), stream);
    } else if (output_fp32) {
      launch_qvq_gemv_splitk<nv_bfloat16, float>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, static_cast<int>(bank_mode), static_cast<int>(bank_alt_id), stream);
    } else {
      launch_qvq_gemv_splitk<nv_bfloat16, nv_bfloat16>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, static_cast<int>(bank_mode), static_cast<int>(bank_alt_id), stream);
    }
  } else if (input.scalar_type() == at::kHalf) {
    if (output_fp32) {
      launch_qvq_gemv<half, float>(input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, output,
                                   static_cast<int>(transition_bits), static_cast<int>(bank_mode),
                                   static_cast<int>(bank_alt_id), stream);
    } else {
      launch_qvq_gemv<half, half>(input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, output,
                                  static_cast<int>(transition_bits), static_cast<int>(bank_mode),
                                  static_cast<int>(bank_alt_id), stream);
    }
  } else {
    if (output_fp32) {
      launch_qvq_gemv<nv_bfloat16, float>(input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, output,
                                          static_cast<int>(transition_bits), static_cast<int>(bank_mode),
                                          static_cast<int>(bank_alt_id), stream);
    } else {
      launch_qvq_gemv<nv_bfloat16, nv_bfloat16>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, output,
          static_cast<int>(transition_bits), static_cast<int>(bank_mode), static_cast<int>(bank_alt_id), stream);
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
  const QvqCudaDeviceConfig& device_config = qvq_cuda_device_config(input.get_device());
  TORCH_CHECK(device_config.major >= 8, "QVQ CUDA requires compute capability >= 8.0");

  at::Tensor output = at::empty(
      {size_m, out_features}, output_fp32 ? input.options().dtype(at::kFloat) : input.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const int64_t base_blocks = (out_features / kTileColumns) * ((size_m + kRowsPerBlock - 1) / kRowsPerBlock);
  const int64_t target_blocks = static_cast<int64_t>(device_config.sm_count) * 6;
  const int64_t k_tiles = size_k / kTileRows;
  const int split_count = base_blocks >= 384 ? 1 : static_cast<int>(std::min(
      std::min((target_blocks + base_blocks - 1) / base_blocks, k_tiles), static_cast<int64_t>(64)));
  if (split_count > 1) {
    at::Tensor partial_output =
        at::empty({split_count, size_m, out_features}, input.options().dtype(at::kFloat));
    if (input.scalar_type() == at::kHalf && output_fp32) {
      launch_qvq_gemv_splitk<half, float, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, bank_ids.has_value() ? 1 : 0, 0, stream);
    } else if (input.scalar_type() == at::kHalf) {
      launch_qvq_gemv_splitk<half, half, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, bank_ids.has_value() ? 1 : 0, 0, stream);
    } else if (output_fp32) {
      launch_qvq_gemv_splitk<nv_bfloat16, float, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, bank_ids.has_value() ? 1 : 0, 0, stream);
    } else {
      launch_qvq_gemv_splitk<nv_bfloat16, nv_bfloat16, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, partial_output, output,
          static_cast<int>(transition_bits), split_count, bank_ids.has_value() ? 1 : 0, 0, stream);
    }
  } else if (input.scalar_type() == at::kHalf) {
    if (output_fp32) {
      launch_qvq_gemv<half, float, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, output,
          static_cast<int>(transition_bits), bank_ids.has_value() ? 1 : 0, 0, stream);
    } else {
      launch_qvq_gemv<half, half, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, output,
          static_cast<int>(transition_bits), bank_ids.has_value() ? 1 : 0, 0, stream);
    }
  } else {
    if (output_fp32) {
      launch_qvq_gemv<nv_bfloat16, float, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, output,
          static_cast<int>(transition_bits), bank_ids.has_value() ? 1 : 0, 0, stream);
    } else {
      launch_qvq_gemv<nv_bfloat16, nv_bfloat16, 4>(
          input, trellis, bank_ids.has_value() ? &*bank_ids : nullptr, levels, output,
          static_cast<int>(transition_bits), bank_ids.has_value() ? 1 : 0, 0, stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def("gemv(Tensor input, Tensor trellis, Tensor levels, int transition_bits, int out_features, bool output_fp32, Tensor? bank_ids=None, int bank_mode=0, int bank_alt_id=0) -> Tensor");
  m.def("gemv_lr(Tensor input, Tensor trellis, Tensor levels, int transition_bits, int out_features, bool output_fp32, Tensor bank_ids, int bank_alt_id, int split_count=0) -> Tensor");
  m.def("gemv_v4(Tensor input, Tensor trellis, Tensor levels, int transition_bits, int out_features, bool output_fp32, Tensor? bank_ids=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CUDA, m) {
  m.impl("gemv", [](const at::Tensor& input, const at::Tensor& trellis, const at::Tensor& levels, int64_t transition_bits, int64_t out_features, bool output_fp32, const c10::optional<at::Tensor>& bank_ids, int64_t bank_mode, int64_t bank_alt_id) { return qvq_gemv_cuda_impl(input, trellis, levels, transition_bits, out_features, output_fp32, 2, bank_ids, bank_mode, bank_alt_id); });
  m.impl("gemv_lr", [](const at::Tensor& input, const at::Tensor& trellis, const at::Tensor& levels, int64_t transition_bits, int64_t out_features, bool output_fp32, const at::Tensor& bank_ids, int64_t bank_alt_id, int64_t split_count) { return qvq_gemv_cuda_local_ring_impl(input, trellis, levels, transition_bits, out_features, output_fp32, bank_ids, bank_alt_id, split_count); });
  m.impl("gemv_v4", [](const at::Tensor& input, const at::Tensor& trellis, const at::Tensor& levels, int64_t transition_bits, int64_t out_features, bool output_fp32, const c10::optional<at::Tensor>& bank_ids) { return qvq_gemv_cuda_v4(input, trellis, levels, transition_bits, out_features, output_fp32, 4, bank_ids); });
}
