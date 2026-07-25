// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <humming/datatype/dtypes.cuh>
#include <humming/kernel/hadamard.cuh>
#include <humming/kernel/hadamard_quant.cuh>
#include <humming/utils/all.cuh>


// Fused FHT + per-group symmetric quant for the case kGroupSize > kBlockSize
// (e.g. channelwise quant with a small rotation block).
//
// Layout:
//   - One CTA per quant group (G elements = G/N independent FHT tiles).
//   - threads_per_block = (G/N) * T_tile / M
//   - Each thread owns M = kTilesPerThread FHT tiles (M * E_lane floats).
//   - FHT stages are all warp-shuffle (requires T_tile <= 32, i.e. N <= 256).
//
// Constraints (asserted at compile time):
//   - kGroupSize > kBlockSize and kBlockSize divides kGroupSize.
//   - kThreadsPerTile <= 32  (shuffle-only FHT).
//   - kTilesPerThread divides kGroupSize / kBlockSize.
//   - threads_per_block in [32, 1024], multiple of 32.
template <
    class SourceType,
    class TargetType,
    uint32_t kBlockSize,
    uint32_t kGroupSize,
    uint32_t kThreadsPerTile,
    uint32_t kTilesPerThread,
    bool kHasExtraScale,
    bool kMMajor = false,
    uint32_t kScaleStore = kScaleStoreF32,
    bool kHasGlobalScale = false>
__global__ void hadamard_quant_input_wide(
    const SourceType *__restrict__ in_ptr,
    void *__restrict__ out_ptr,
    void *__restrict__ scales_ptr,
    float extra_scale,
    uint32_t num_groups,
    uint32_t shape_m = 0,
    uint32_t groups_per_row = 0,
    const float *__restrict__ global_scale_ptr = nullptr) {

  static_assert((kBlockSize & (kBlockSize - 1)) == 0);
  static_assert(kGroupSize > kBlockSize);
  static_assert(kGroupSize % kBlockSize == 0);
  static_assert(kBlockSize % kThreadsPerTile == 0);
  static_assert(kThreadsPerTile <= 32, "T_tile must be <= 32 for shuffle-only FHT");

  constexpr uint32_t E_lane = kBlockSize / kThreadsPerTile;
  static_assert((E_lane & (E_lane - 1)) == 0);

  constexpr uint32_t kTilesPerGroup = kGroupSize / kBlockSize;
  static_assert(kTilesPerGroup % kTilesPerThread == 0);
  constexpr uint32_t kTileSlotsPerBlock = kTilesPerGroup / kTilesPerThread;
  constexpr uint32_t kThreadsPerBlock = kTileSlotsPerBlock * kThreadsPerTile;
  static_assert(kThreadsPerBlock >= 32 && kThreadsPerBlock <= 1024);
  static_assert(kThreadsPerBlock % 32 == 0);

  constexpr uint32_t kElemsPerThread = kTilesPerThread * E_lane;

  auto constexpr_log2 = [](uint32_t v) constexpr {
    uint32_t r = 0;
    while ((1u << r) < v)
      r++;
    return r;
  };
  constexpr uint32_t kLog2E = constexpr_log2(E_lane);
  constexpr uint32_t kLog2T = constexpr_log2(kThreadsPerTile);
  constexpr uint32_t kNumWarps = kThreadsPerBlock / 32;
  constexpr bool kIsInt = TargetType::kIsIntegerType;
  constexpr uint32_t kReduceSlots = kIsInt ? 2 * kNumWarps : kNumWarps;

  __shared__ float reduce_smem[kReduceSlots > 0 ? kReduceSlots : 1];

  uint32_t group_idx = blockIdx.x;
  if (group_idx >= num_groups) return;

  uint32_t tid = threadIdx.x;
  uint32_t lane_in_tile = tid % kThreadsPerTile;
  uint32_t tile_slot = tid / kThreadsPerTile;

  float reg[kElemsPerThread];

  // ---- Load M tiles ----
  // When T_tile == 1, all M tiles owned by this thread are contiguous in
  // gmem; do one big vec load. Otherwise per-tile load.
  if constexpr (kThreadsPerTile == 1) {
    const SourceType *gptr = in_ptr + group_idx * kGroupSize + tile_slot * kElemsPerThread;
    vec_load_to_float<SourceType, kElemsPerThread>(reg, gptr);
  } else {
    PRAGMA_UNROLL
    for (uint32_t m = 0; m < kTilesPerThread; m++) {
      uint32_t tile_in_group = tile_slot * kTilesPerThread + m;
      const SourceType *gptr = in_ptr + group_idx * kGroupSize + tile_in_group * kBlockSize + lane_in_tile * E_lane;
      float tile_reg[E_lane];
      vec_load_to_float<SourceType, E_lane>(tile_reg, gptr);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E_lane; i++)
        reg[m * E_lane + i] = tile_reg[i];
    }
  }

  // ---- FHT each tile (independent, shuffle-only) ----
  PRAGMA_UNROLL
  for (uint32_t m = 0; m < kTilesPerThread; m++) {
    // Register stages.
    PRAGMA_UNROLL
    for (uint32_t s = 0; s < kLog2E; s++) {
      uint32_t stride = 1u << s;
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E_lane; i++) {
        uint32_t j = i ^ stride;
        if (i < j) {
          float a = reg[m * E_lane + i];
          float b = reg[m * E_lane + j];
          reg[m * E_lane + i] = a + b;
          reg[m * E_lane + j] = a - b;
        }
      }
    }
    // Warp shuffle stages within tile (kThreadsPerTile <= 32).
    PRAGMA_UNROLL
    for (uint32_t s = 0; s < kLog2T; s++) {
      uint32_t partner_xor = 1u << s;
      bool is_low = (lane_in_tile & partner_xor) == 0;
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E_lane; i++) {
        float other = __shfl_xor_sync(
            0xffffffff, reg[m * E_lane + i], partner_xor, kThreadsPerTile);
        reg[m * E_lane + i] = is_low ? (reg[m * E_lane + i] + other)
                                     : (other - reg[m * E_lane + i]);
      }
    }
  }

  // ---- Normalize (combined with extra_scale) ----
  float norm = rsqrtf((float)kBlockSize);
  if constexpr (kHasExtraScale) norm *= extra_scale;
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kElemsPerThread; i++)
    reg[i] *= norm;

  // ---- Channel-wide reduction ----
  float local_max, local_min, local_absmax;
  if constexpr (kIsInt) {
    local_max = -1e30f;
    local_min = 1e30f;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kElemsPerThread; i++) {
      local_max = fmaxf(local_max, reg[i]);
      local_min = fminf(local_min, reg[i]);
    }
  } else {
    local_absmax = 0.f;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kElemsPerThread; i++) {
      local_absmax = fmaxf(local_absmax, fabsf(reg[i]));
    }
  }

  // Intra-warp.
  PRAGMA_UNROLL
  for (uint32_t s = 1; s < 32; s <<= 1) {
    if constexpr (kIsInt) {
      local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, s, 32));
      local_min = fminf(local_min, __shfl_xor_sync(0xffffffff, local_min, s, 32));
    } else {
      local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, s, 32));
    }
  }

  // Cross-warp via smem.
  if constexpr (kNumWarps > 1) {
    uint32_t warp_id = tid / 32;
    uint32_t lane_id_in_warp = tid & 31;
    if (lane_id_in_warp == 0) {
      if constexpr (kIsInt) {
        reduce_smem[warp_id] = local_max;
        reduce_smem[kNumWarps + warp_id] = -local_min;
      } else {
        reduce_smem[warp_id] = local_absmax;
      }
    }
    __syncthreads();
    if (warp_id == 0) {
      float v = (lane_id_in_warp < kNumWarps) ? reduce_smem[lane_id_in_warp] : -1e30f;
      PRAGMA_UNROLL
      for (uint32_t s = 1; s < kNumWarps; s <<= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, s, 32));
      }
      if (lane_id_in_warp == 0) reduce_smem[0] = v;

      if constexpr (kIsInt) {
        float w = (lane_id_in_warp < kNumWarps) ? reduce_smem[kNumWarps + lane_id_in_warp] : -1e30f;
        PRAGMA_UNROLL
        for (uint32_t s = 1; s < kNumWarps; s <<= 1) {
          w = fmaxf(w, __shfl_xor_sync(0xffffffff, w, s, 32));
        }
        if (lane_id_in_warp == 0) reduce_smem[1] = w;
      }
    }
    __syncthreads();
    if constexpr (kIsInt) {
      local_max = reduce_smem[0];
      local_min = -reduce_smem[1];
    } else {
      local_absmax = reduce_smem[0];
    }
  }

  // ---- Compute scale ----
  float scale;
  if constexpr (kIsInt) {
    constexpr float pos_lim = (float)((1 << (TargetType::kBits - 1)) - 1) + 0.0f;
    constexpr float neg_lim = (float)(1 << (TargetType::kBits - 1)) + 0.0f;
    float s1 = local_max / pos_lim;
    float s2 = -local_min / neg_lim;
    scale = fmaxf(fmaxf(s1, s2), 1e-30f);
  } else {
    float dtype_max = fp_target_max_value<TargetType>();
    scale = fmaxf(local_absmax / dtype_max, 1e-30f);
  }

  uint32_t scale_idx;
  if constexpr (kMMajor) {
    uint32_t row = group_idx / groups_per_row;
    uint32_t group_in_row = group_idx - row * groups_per_row;
    if constexpr (kScaleStore == kScaleStoreE8M0) {
      scale_idx = (group_in_row / 4) * (shape_m * 4) + row * 4 + (group_in_row % 4);
    } else {
      scale_idx = group_in_row * shape_m + row;
    }
  } else {
    scale_idx = group_idx;
  }

  float inv_scale;
  if constexpr (kScaleStore == kScaleStoreF32 && !kHasGlobalScale) {
    // Default path: bit-identical to the original f32-scale kernel.
    inv_scale = 1.f / scale;
    if (tid == 0) reinterpret_cast<float *>(scales_ptr)[scale_idx] = scale;
  } else {
    float gs = 1.f, inv_gs = 1.f;
    if constexpr (kHasGlobalScale) {
      gs = __ldg(global_scale_ptr);
      inv_gs = 1.f / gs;
    }
    // Reg is already normalized in this kernel, so the effective stored scale
    // is the full quantization divisor.
    float eff_scale = finalize_scale<kScaleStore>(
        scale, gs, inv_gs, scales_ptr, scale_idx, tid == 0);
    inv_scale = 1.f / eff_scale;
  }

  // ---- Quantize + store ----
  // Thread output bytes are contiguous in the group's output stream:
  //   bytes_in_group[t*kElemsPerThread .. (t+1)*kElemsPerThread) for 8-bit
  //   bytes_in_group[t*kElemsPerThread/2 .. ] for 4-bit
  // because thread t handles consecutive tiles `tile_slot*M .. tile_slot*M+M-1`
  // and within each tile lane 0..T_tile-1 occupies E_lane consecutive elements.
  constexpr uint32_t kBits = TargetType::kBits;
  if constexpr (kBits == 8) {
    uint8_t bytes[kElemsPerThread];
    constexpr bool kIsFp8 =
        std::is_same<TargetType, Float8E4M3>::value ||
        std::is_same<TargetType, Float8E3M4>::value ||
        std::is_same<TargetType, Float8E5M2>::value;
    if constexpr (kIsFp8) {
      static_assert(kElemsPerThread % 2 == 0,
                    "fp8 requires even kElemsPerThread (paired HW cvt)");
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kElemsPerThread; i += 2) {
        uint16_t pair = quant_pair_fp8<TargetType>(
            reg[i] * inv_scale, reg[i + 1] * inv_scale);
        bytes[i] = pair & 0xFFu;
        bytes[i + 1] = (pair >> 8) & 0xFFu;
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kElemsPerThread; i++) {
        bytes[i] = static_cast<uint8_t>(
            quant_one_value<TargetType>(reg[i], inv_scale));
      }
    }
    // Output offset for this thread:
    //   group_idx * kGroupSize + (tile_slot * kTilesPerThread) * kBlockSize
    //     + lane_in_tile * E_lane
    uint8_t *gptr = reinterpret_cast<uint8_t *>(out_ptr) + group_idx * kGroupSize + tile_slot * kTilesPerThread * kBlockSize + lane_in_tile * E_lane;

    // When T_tile == 1, all kElemsPerThread bytes are contiguous → big vec
    // store. Otherwise per-tile store.
    if constexpr (kThreadsPerTile == 1) {
      if constexpr (kElemsPerThread == 16) {
        uint4 packed;
        packed.x = *reinterpret_cast<uint32_t *>(&bytes[0]);
        packed.y = *reinterpret_cast<uint32_t *>(&bytes[4]);
        packed.z = *reinterpret_cast<uint32_t *>(&bytes[8]);
        packed.w = *reinterpret_cast<uint32_t *>(&bytes[12]);
        *reinterpret_cast<uint4 *>(gptr) = packed;
      } else if constexpr (kElemsPerThread == 8) {
        uint2 packed;
        packed.x = *reinterpret_cast<uint32_t *>(&bytes[0]);
        packed.y = *reinterpret_cast<uint32_t *>(&bytes[4]);
        *reinterpret_cast<uint2 *>(gptr) = packed;
      } else if constexpr (kElemsPerThread == 4) {
        *reinterpret_cast<uint32_t *>(gptr) =
            *reinterpret_cast<uint32_t *>(&bytes[0]);
      } else if constexpr (kElemsPerThread == 2) {
        *reinterpret_cast<uint16_t *>(gptr) =
            *reinterpret_cast<uint16_t *>(&bytes[0]);
      } else if constexpr (kElemsPerThread > 16 && kElemsPerThread % 16 == 0) {
        PRAGMA_UNROLL
        for (uint32_t v = 0; v < kElemsPerThread / 16; v++) {
          uint4 packed;
          packed.x = *reinterpret_cast<uint32_t *>(&bytes[v * 16 + 0]);
          packed.y = *reinterpret_cast<uint32_t *>(&bytes[v * 16 + 4]);
          packed.z = *reinterpret_cast<uint32_t *>(&bytes[v * 16 + 8]);
          packed.w = *reinterpret_cast<uint32_t *>(&bytes[v * 16 + 12]);
          *reinterpret_cast<uint4 *>(gptr + v * 16) = packed;
        }
      } else {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kElemsPerThread; i++)
          gptr[i] = bytes[i];
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t m = 0; m < kTilesPerThread; m++) {
        uint8_t *p = gptr + m * kBlockSize;
        if constexpr (E_lane == 16) {
          uint4 packed;
          packed.x = *reinterpret_cast<uint32_t *>(&bytes[m * E_lane + 0]);
          packed.y = *reinterpret_cast<uint32_t *>(&bytes[m * E_lane + 4]);
          packed.z = *reinterpret_cast<uint32_t *>(&bytes[m * E_lane + 8]);
          packed.w = *reinterpret_cast<uint32_t *>(&bytes[m * E_lane + 12]);
          *reinterpret_cast<uint4 *>(p) = packed;
        } else if constexpr (E_lane == 8) {
          uint2 packed;
          packed.x = *reinterpret_cast<uint32_t *>(&bytes[m * E_lane]);
          packed.y = *reinterpret_cast<uint32_t *>(&bytes[m * E_lane + 4]);
          *reinterpret_cast<uint2 *>(p) = packed;
        } else if constexpr (E_lane == 4) {
          *reinterpret_cast<uint32_t *>(p) =
              *reinterpret_cast<uint32_t *>(&bytes[m * E_lane]);
        } else if constexpr (E_lane == 2) {
          *reinterpret_cast<uint16_t *>(p) =
              *reinterpret_cast<uint16_t *>(&bytes[m * E_lane]);
        } else {
          PRAGMA_UNROLL
          for (uint32_t i = 0; i < E_lane; i++)
            p[i] = bytes[m * E_lane + i];
        }
      }
    }
  } else if constexpr (kBits == 4) {
    // Pair adjacent elements within each tile, then write per-tile.
    static_assert(E_lane >= 2 && E_lane % 2 == 0);
    constexpr bool kIsFp4 = std::is_same<TargetType, Float4E2M1>::value ||
                            std::is_same<TargetType, Float4E0M3>::value;
    uint8_t bytes[kElemsPerThread / 2];
    PRAGMA_UNROLL
    for (uint32_t m = 0; m < kTilesPerThread; m++) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E_lane / 2; i++) {
        if constexpr (kIsFp4) {
          bytes[m * (E_lane / 2) + i] = quant_pair_fp4<TargetType>(
              reg[m * E_lane + 2 * i] * inv_scale,
              reg[m * E_lane + 2 * i + 1] * inv_scale);
        } else {
          uint32_t a = quant_one_value<TargetType>(reg[m * E_lane + 2 * i], inv_scale) & 0xFu;
          uint32_t b = quant_one_value<TargetType>(reg[m * E_lane + 2 * i + 1], inv_scale) & 0xFu;
          bytes[m * (E_lane / 2) + i] = static_cast<uint8_t>(a | (b << 4));
        }
      }
    }
    uint8_t *gptr = reinterpret_cast<uint8_t *>(out_ptr) + (group_idx * kGroupSize + tile_slot * kTilesPerThread * kBlockSize) / 2 + lane_in_tile * (E_lane / 2);
    PRAGMA_UNROLL
    for (uint32_t m = 0; m < kTilesPerThread; m++) {
      uint8_t *p = gptr + m * (kBlockSize / 2);
      if constexpr (E_lane / 2 == 8) {
        uint2 packed;
        packed.x = *reinterpret_cast<uint32_t *>(&bytes[m * (E_lane / 2) + 0]);
        packed.y = *reinterpret_cast<uint32_t *>(&bytes[m * (E_lane / 2) + 4]);
        *reinterpret_cast<uint2 *>(p) = packed;
      } else if constexpr (E_lane / 2 == 4) {
        *reinterpret_cast<uint32_t *>(p) =
            *reinterpret_cast<uint32_t *>(&bytes[m * (E_lane / 2)]);
      } else if constexpr (E_lane / 2 == 2) {
        *reinterpret_cast<uint16_t *>(p) =
            *reinterpret_cast<uint16_t *>(&bytes[m * (E_lane / 2)]);
      } else {
        static_assert(E_lane / 2 == 1);
        p[0] = bytes[m];
      }
    }
  } else {
    static_assert(kBits == 8 || kBits == 4);
  }
}
