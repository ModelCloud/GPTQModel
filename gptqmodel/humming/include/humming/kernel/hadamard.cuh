// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <humming/utils/all.cuh>


// ---- Vectorized load/store ----

template <class T, uint32_t E>
CUDA_INLINE void vec_load_to_float(float (&reg)[E], const T *gptr) {
  if constexpr (std::is_same<T, float>::value) {
    if constexpr (E % 4 == 0) {
      PRAGMA_UNROLL
      for (uint32_t v = 0; v < E / 4; v++) {
        float4 b = reinterpret_cast<const float4 *>(gptr)[v];
        reg[v * 4 + 0] = b.x;
        reg[v * 4 + 1] = b.y;
        reg[v * 4 + 2] = b.z;
        reg[v * 4 + 3] = b.w;
      }
    } else if constexpr (E % 2 == 0) {
      PRAGMA_UNROLL
      for (uint32_t v = 0; v < E / 2; v++) {
        float2 b = reinterpret_cast<const float2 *>(gptr)[v];
        reg[v * 2 + 0] = b.x;
        reg[v * 2 + 1] = b.y;
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E; i++)
        reg[i] = gptr[i];
    }
  } else if constexpr (std::is_same<T, __half>::value) {
    if constexpr (E % 8 == 0) {
      PRAGMA_UNROLL
      for (uint32_t v = 0; v < E / 8; v++) {
        uint4 b = reinterpret_cast<const uint4 *>(gptr)[v];
        float2 f0 = __half22float2(*reinterpret_cast<__half2 *>(&b.x));
        float2 f1 = __half22float2(*reinterpret_cast<__half2 *>(&b.y));
        float2 f2 = __half22float2(*reinterpret_cast<__half2 *>(&b.z));
        float2 f3 = __half22float2(*reinterpret_cast<__half2 *>(&b.w));
        reg[v * 8 + 0] = f0.x;
        reg[v * 8 + 1] = f0.y;
        reg[v * 8 + 2] = f1.x;
        reg[v * 8 + 3] = f1.y;
        reg[v * 8 + 4] = f2.x;
        reg[v * 8 + 5] = f2.y;
        reg[v * 8 + 6] = f3.x;
        reg[v * 8 + 7] = f3.y;
      }
    } else if constexpr (E % 2 == 0) {
      PRAGMA_UNROLL
      for (uint32_t v = 0; v < E / 2; v++) {
        __half2 h = reinterpret_cast<const __half2 *>(gptr)[v];
        float2 f = __half22float2(h);
        reg[v * 2 + 0] = f.x;
        reg[v * 2 + 1] = f.y;
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E; i++)
        reg[i] = __half2float(gptr[i]);
    }
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    if constexpr (E % 8 == 0) {
      PRAGMA_UNROLL
      for (uint32_t v = 0; v < E / 8; v++) {
        uint4 b = reinterpret_cast<const uint4 *>(gptr)[v];
        float2 f0 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&b.x));
        float2 f1 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&b.y));
        float2 f2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&b.z));
        float2 f3 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&b.w));
        reg[v * 8 + 0] = f0.x;
        reg[v * 8 + 1] = f0.y;
        reg[v * 8 + 2] = f1.x;
        reg[v * 8 + 3] = f1.y;
        reg[v * 8 + 4] = f2.x;
        reg[v * 8 + 5] = f2.y;
        reg[v * 8 + 6] = f3.x;
        reg[v * 8 + 7] = f3.y;
      }
    } else if constexpr (E % 2 == 0) {
      PRAGMA_UNROLL
      for (uint32_t v = 0; v < E / 2; v++) {
        __nv_bfloat162 h = reinterpret_cast<const __nv_bfloat162 *>(gptr)[v];
        float2 f = __bfloat1622float2(h);
        reg[v * 2 + 0] = f.x;
        reg[v * 2 + 1] = f.y;
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E; i++)
        reg[i] = __bfloat162float(gptr[i]);
    }
  } else {
    static_assert(sizeof(T) == 0, "unsupported dtype");
  }
}


template <class T, uint32_t E>
CUDA_INLINE void vec_store_from_float(T *gptr, const float (&reg)[E], float norm) {
  if constexpr (std::is_same<T, float>::value) {
    if constexpr (E % 4 == 0) {
      PRAGMA_UNROLL
      for (uint32_t v = 0; v < E / 4; v++) {
        float4 b;
        b.x = reg[v * 4 + 0] * norm;
        b.y = reg[v * 4 + 1] * norm;
        b.z = reg[v * 4 + 2] * norm;
        b.w = reg[v * 4 + 3] * norm;
        reinterpret_cast<float4 *>(gptr)[v] = b;
      }
    } else if constexpr (E % 2 == 0) {
      PRAGMA_UNROLL
      for (uint32_t v = 0; v < E / 2; v++) {
        float2 b;
        b.x = reg[v * 2 + 0] * norm;
        b.y = reg[v * 2 + 1] * norm;
        reinterpret_cast<float2 *>(gptr)[v] = b;
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E; i++)
        gptr[i] = reg[i] * norm;
    }
  } else if constexpr (std::is_same<T, __half>::value) {
    if constexpr (E % 8 == 0) {
      PRAGMA_UNROLL
      for (uint32_t v = 0; v < E / 8; v++) {
        uint4 b;
        __half2 h0 = __floats2half2_rn(reg[v * 8 + 0] * norm, reg[v * 8 + 1] * norm);
        __half2 h1 = __floats2half2_rn(reg[v * 8 + 2] * norm, reg[v * 8 + 3] * norm);
        __half2 h2 = __floats2half2_rn(reg[v * 8 + 4] * norm, reg[v * 8 + 5] * norm);
        __half2 h3 = __floats2half2_rn(reg[v * 8 + 6] * norm, reg[v * 8 + 7] * norm);
        b.x = *reinterpret_cast<uint32_t *>(&h0);
        b.y = *reinterpret_cast<uint32_t *>(&h1);
        b.z = *reinterpret_cast<uint32_t *>(&h2);
        b.w = *reinterpret_cast<uint32_t *>(&h3);
        reinterpret_cast<uint4 *>(gptr)[v] = b;
      }
    } else if constexpr (E % 2 == 0) {
      PRAGMA_UNROLL
      for (uint32_t v = 0; v < E / 2; v++) {
        __half2 h = __floats2half2_rn(reg[v * 2 + 0] * norm, reg[v * 2 + 1] * norm);
        reinterpret_cast<__half2 *>(gptr)[v] = h;
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E; i++)
        gptr[i] = __float2half_rn(reg[i] * norm);
    }
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    if constexpr (E % 8 == 0) {
      PRAGMA_UNROLL
      for (uint32_t v = 0; v < E / 8; v++) {
        uint4 b;
        __nv_bfloat162 h0 = __floats2bfloat162_rn(reg[v * 8 + 0] * norm, reg[v * 8 + 1] * norm);
        __nv_bfloat162 h1 = __floats2bfloat162_rn(reg[v * 8 + 2] * norm, reg[v * 8 + 3] * norm);
        __nv_bfloat162 h2 = __floats2bfloat162_rn(reg[v * 8 + 4] * norm, reg[v * 8 + 5] * norm);
        __nv_bfloat162 h3 = __floats2bfloat162_rn(reg[v * 8 + 6] * norm, reg[v * 8 + 7] * norm);
        b.x = *reinterpret_cast<uint32_t *>(&h0);
        b.y = *reinterpret_cast<uint32_t *>(&h1);
        b.z = *reinterpret_cast<uint32_t *>(&h2);
        b.w = *reinterpret_cast<uint32_t *>(&h3);
        reinterpret_cast<uint4 *>(gptr)[v] = b;
      }
    } else if constexpr (E % 2 == 0) {
      PRAGMA_UNROLL
      for (uint32_t v = 0; v < E / 2; v++) {
        __nv_bfloat162 h = __floats2bfloat162_rn(reg[v * 2 + 0] * norm, reg[v * 2 + 1] * norm);
        reinterpret_cast<__nv_bfloat162 *>(gptr)[v] = h;
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E; i++)
        gptr[i] = __float2bfloat16_rn(reg[i] * norm);
    }
  } else {
    static_assert(sizeof(T) == 0, "unsupported dtype");
  }
}


// ---- Register-blocked FHT kernel ----
//
// Layout:
//   - Each thread block processes kTilesPerBlock tiles of length kBlockSize.
//   - Within each tile, kThreadsPerTile threads cooperate; each holds
//     kElemsPerThread = kBlockSize / kThreadsPerTile floats in registers.
//   - Stages 0..log2(E)-1 are pure register butterflies (no sync).
//   - Stages log2(E)..min(log2(N), log2(E)+5)-1 use __shfl_xor_sync.
//   - Remaining stages (only when kThreadsPerTile > 32) use shared memory.
template <
    class T,
    uint32_t kBlockSize,
    uint32_t kThreadsPerTile,
    uint32_t kTilesPerBlock,
    bool kHasScale>
__global__ void hadamard_transform(
    const T *__restrict__ in_ptr,
    T *__restrict__ out_ptr,
    float extra_scale,
    uint32_t num_tiles) {

  static_assert((kBlockSize & (kBlockSize - 1)) == 0);
  static_assert((kThreadsPerTile & (kThreadsPerTile - 1)) == 0);
  static_assert(kBlockSize % kThreadsPerTile == 0);

  constexpr uint32_t E = kBlockSize / kThreadsPerTile;
  static_assert((E & (E - 1)) == 0);

  // log2 via constexpr loop (NVRTC doesn't have __builtin_ctz at host scope).
  auto constexpr_log2 = [](uint32_t v) constexpr {
    uint32_t r = 0;
    while ((1u << r) < v)
      r++;
    return r;
  };
  constexpr uint32_t kLog2E = constexpr_log2(E);
  constexpr uint32_t kLog2T = constexpr_log2(kThreadsPerTile);
  constexpr uint32_t kLog2Warp = 5;
  constexpr uint32_t kShflStages = kLog2T < kLog2Warp ? kLog2T : kLog2Warp;
  constexpr uint32_t kSmemStages = kLog2T > kLog2Warp ? kLog2T - kLog2Warp : 0;
  constexpr uint32_t kThreadsPerBlock = kThreadsPerTile * kTilesPerBlock;
  constexpr uint32_t kSmemElems = kSmemStages > 0 ? kBlockSize * kTilesPerBlock : 1;

  __shared__ float smem[kSmemElems];

  uint32_t tid = threadIdx.x;
  uint32_t lane_in_tile = tid % kThreadsPerTile;
  uint32_t tile_in_block = tid / kThreadsPerTile;
  uint32_t tile_idx = blockIdx.x * kTilesPerBlock + tile_in_block;
  bool valid = tile_idx < num_tiles;

  float reg[E];

  // ---- Load ----
  if (valid) {
    const T *gptr = in_ptr + tile_idx * kBlockSize + lane_in_tile * E;
    vec_load_to_float<T, E>(reg, gptr);
  } else {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < E; i++)
      reg[i] = 0.f;
  }

  // ---- Register stages ----
  PRAGMA_UNROLL
  for (uint32_t s = 0; s < kLog2E; s++) {
    uint32_t stride = 1u << s;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < E; i++) {
      uint32_t j = i ^ stride;
      if (i < j) {
        float a = reg[i];
        float b = reg[j];
        reg[i] = a + b;
        reg[j] = a - b;
      }
    }
  }

  // ---- Warp shuffle stages ----
  PRAGMA_UNROLL
  for (uint32_t s = 0; s < kShflStages; s++) {
    uint32_t partner_xor = 1u << s;
    bool is_low = (lane_in_tile & partner_xor) == 0;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < E; i++) {
      // width = min(kThreadsPerTile, 32) keeps shuffle confined to a tile.
      constexpr uint32_t kShflWidth = kThreadsPerTile < 32 ? kThreadsPerTile : 32;
      float other = __shfl_xor_sync(0xffffffff, reg[i], partner_xor, kShflWidth);
      reg[i] = is_low ? (reg[i] + other) : (other - reg[i]);
    }
  }

  // ---- Shared-memory stages ----
  if constexpr (kSmemStages > 0) {
    PRAGMA_UNROLL
    for (uint32_t s = 0; s < kSmemStages; s++) {
      uint32_t partner_lane_xor = 1u << (kLog2Warp + s);
      uint32_t partner_lane = lane_in_tile ^ partner_lane_xor;
      bool is_low = (lane_in_tile & partner_lane_xor) == 0;

      uint32_t my_base = tile_in_block * kBlockSize + lane_in_tile * E;
      uint32_t partner_base = tile_in_block * kBlockSize + partner_lane * E;

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E; i++)
        smem[my_base + i] = reg[i];
      __syncthreads();

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E; i++) {
        float other = smem[partner_base + i];
        reg[i] = is_low ? (reg[i] + other) : (other - reg[i]);
      }
      __syncthreads();
    }
  }

  // ---- Normalize and store ----
  float norm = rsqrtf((float)kBlockSize);
  if constexpr (kHasScale) norm *= extra_scale;

  if (valid) {
    T *gptr = out_ptr + tile_idx * kBlockSize + lane_in_tile * E;
    vec_store_from_float<T, E>(gptr, reg, norm);
  }
}
