// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <humming/datatype/dtypes.cuh>
#include <humming/kernel/hadamard.cuh>
#include <humming/utils/all.cuh>


// Hardcoded representable max for supported FP target types.
template <class DataType>
CUDA_INLINE float fp_target_max_value() {
  if constexpr (std::is_same<DataType, Float8E4M3>::value) {
    return 448.f;
  } else if constexpr (std::is_same<DataType, Float8E3M4>::value) {
    return 30.f;
  } else if constexpr (std::is_same<DataType, Float8E5M2>::value) {
    return 57344.f;
  } else if constexpr (std::is_same<DataType, Float4E2M1>::value) {
    return 6.f;
  } else if constexpr (std::is_same<DataType, Float4E0M3>::value) {
    return 7.f;
  } else {
    static_assert(sizeof(DataType) == 0, "unsupported FP target type");
    return 0.f;
  }
}


// Paired fp8 conversion via hardware cvt instruction. Requires SM89+.
// Returns two fp8 bytes packed into a uint16_t.
template <class TargetType>
CUDA_INLINE uint16_t quant_pair_fp8(float a, float b) {
  uint16_t out;
#if __CUDA_ARCH__ >= 890
  if constexpr (std::is_same<TargetType, Float8E4M3>::value) {
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(out) : "f"(b), "f"(a));
  } else if constexpr (std::is_same<TargetType, Float8E5M2>::value ||
                       std::is_same<TargetType, Float8E3M4>::value) {
    // Float8E3M4 has no PTX type; emit the e5m2 cvt and patch the cubin's SASS
    // format bits (mode "cvt_e3m4") afterwards.
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(out) : "f"(b), "f"(a));
  } else {
    static_assert(sizeof(TargetType) == 0, "quant_pair_fp8: unsupported type");
  }
#else
  asm("trap;");
  out = 0;
#endif
  return out;
}


// Paired fp4 (e2m1) conversion via hardware cvt instruction. Requires SM100+.
// Returns two fp4 nibbles packed into one byte (low nibble = a, high = b).
template <class TargetType>
CUDA_INLINE uint8_t quant_pair_fp4(float a, float b) {
  uint16_t out;
#if __CUDA_ARCH__ >= 1000
  // Float4E0M3 has no PTX type; emit the e2m1 cvt and patch the cubin's SASS
  // format bits (mode "cvt_e0m3") afterwards.
  if constexpr (std::is_same<TargetType, Float4E2M1>::value ||
                std::is_same<TargetType, Float4E0M3>::value) {
    // e2m1x2 packs two fp4 into a .b8 register (low nibble = a, high = b);
    // widen to u16 so the byte can be returned. %1 -> high, %2 -> low.
    asm("{\n\t"
        ".reg .b8 t;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n\t"
        "cvt.u16.u8 %0, t;\n\t"
        "}" : "=h"(out) : "f"(b), "f"(a));
  } else {
    static_assert(sizeof(TargetType) == 0, "quant_pair_fp4: unsupported type");
  }
#else
  asm("trap;");
  out = 0;
#endif
  return static_cast<uint8_t>(out & 0xFFu);
}


// Quantize one integer-typed element (symmetric, two's complement).
template <class TargetType>
CUDA_INLINE uint32_t quant_one_value(float val, float inv_scale) {
  static_assert(TargetType::kIsIntegerType && TargetType::kIsSigned);
  int32_t v = __float2int_rn(val * inv_scale);
  constexpr int32_t lo = -(1 << (TargetType::kBits - 1));
  constexpr int32_t hi = (1 << (TargetType::kBits - 1)) - 1;
  v = max(lo, min(hi, v));
  return static_cast<uint32_t>(v) & ((1u << TargetType::kBits) - 1);
}


// Scale storage dtype codes (mirror humming.ops scale_dtype strings).
static constexpr uint32_t kScaleStoreF32 = 0;
static constexpr uint32_t kScaleStoreE4M3 = 1;
static constexpr uint32_t kScaleStoreE8M0 = 2;


// Quantize the per-group scale for storage, optionally folding a precomputed
// per-tensor global scale. `s` is the full-precision scale to store (already
// including the rsqrt(N) * extra_scale normalization). Returns the *effective*
// stored scale (== quantized_value * gs) so the caller can derive the matching
// inverse for data quantization; the encoded scale is written to
// `store_ptr[idx]` only when `do_write` is true.
template <uint32_t kScaleStore>
CUDA_INLINE float finalize_scale(
    float s, float gs, float inv_gs, void *store_ptr, uint32_t idx, bool do_write) {
  if constexpr (kScaleStore == kScaleStoreE8M0) {
    s *= inv_gs;
    uint32_t u = __float_as_uint(s);
    u = (u + 0x007FFFFFu) & 0x7F800000u; // round up to a power of two
    if (do_write) reinterpret_cast<uint8_t *>(store_ptr)[idx] = static_cast<uint8_t>(u >> 23);
    return __uint_as_float(u) * gs;
  } else if constexpr (kScaleStore == kScaleStoreE4M3) {
    s *= inv_gs;
    __nv_fp8_e4m3 q = static_cast<__nv_fp8_e4m3>(s);
    if (do_write) reinterpret_cast<uint8_t *>(store_ptr)[idx] = q.__x;
    return static_cast<float>(q) * gs;
  } else {
    if (do_write) reinterpret_cast<float *>(store_ptr)[idx] = s;
    return s;
  }
}


// Per-group cross-warp max reduction via smem. Each group's `kWarpsPerGroup`
// warps reduce independently using their own slice of `smem`. The whole block
// syncs together (we cannot sync per-group).
template <uint32_t kWarpsPerGroup>
CUDA_INLINE float cross_warp_max(float val, float *smem, uint32_t warp_id, uint32_t lane_id) {
  uint32_t warp_in_group = warp_id % kWarpsPerGroup;
  uint32_t group_base = (warp_id / kWarpsPerGroup) * kWarpsPerGroup;
  if (lane_id == 0) smem[group_base + warp_in_group] = val;
  __syncthreads();
  if (warp_in_group == 0) {
    float v = (lane_id < kWarpsPerGroup) ? smem[group_base + lane_id] : -1e30f;
    PRAGMA_UNROLL
    for (uint32_t s = 1; s < kWarpsPerGroup; s <<= 1) {
      v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, s, 32));
    }
    if (lane_id == 0) smem[group_base] = v;
  }
  __syncthreads();
  return smem[group_base];
}


// Fused FHT + per-group symmetric quantization.
//
// Constraints (enforced by static_asserts):
//   - kBlockSize, kGroupSize, kThreadsPerTile, kTilesPerBlock all powers of 2.
//   - kGroupSize divides kBlockSize, kGroupSize >= kElemsPerThread.
//   - The case kGroupSize > kBlockSize is NOT supported.
//
// scale convention: returned `scales` already absorb rsqrt(N) * extra_scale,
// so `dequant(q, scale) == FHT(input) * extra_scale`.
template <
    class SourceType,
    class TargetType,
    uint32_t kBlockSize,
    uint32_t kGroupSize,
    uint32_t kThreadsPerTile,
    uint32_t kTilesPerBlock,
    bool kHasExtraScale,
    bool kMMajor = false,
    uint32_t kScaleStore = kScaleStoreF32,
    bool kHasGlobalScale = false>
__global__ void hadamard_quant_input(
    const SourceType *__restrict__ in_ptr,
    void *__restrict__ out_ptr,
    void *__restrict__ scales_ptr,
    float extra_scale,
    uint32_t num_tiles,
    uint32_t shape_m = 0,
    uint32_t groups_per_row = 0,
    const float *__restrict__ global_scale_ptr = nullptr) {

  static_assert((kBlockSize & (kBlockSize - 1)) == 0);
  static_assert(kBlockSize % kGroupSize == 0);
  static_assert((kThreadsPerTile & (kThreadsPerTile - 1)) == 0);
  static_assert(kBlockSize % kThreadsPerTile == 0);

  constexpr uint32_t E = kBlockSize / kThreadsPerTile;
  static_assert(kGroupSize >= E && kGroupSize % E == 0,
                "kGroupSize must be >= kElemsPerThread and a multiple of it");

  constexpr uint32_t kGroupsPerTile = kBlockSize / kGroupSize;
  constexpr uint32_t kLanesPerGroup = kGroupSize / E;
  constexpr uint32_t kThreadsPerBlock = kThreadsPerTile * kTilesPerBlock;

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
  constexpr uint32_t kFhtSmemElems = kSmemStages > 0 ? kBlockSize * kTilesPerBlock : 1;

  constexpr bool kIsInt = TargetType::kIsIntegerType;
  // For int targets we track raw max/min (asymmetric formula).
  // For FP targets we track abs-max only.

  // Cross-warp reduce smem: one float per warp; only used when one group
  // spans multiple warps within the block. That implies kTilesPerBlock == 1
  // and kLanesPerGroup > 32.
  constexpr uint32_t kWarpsPerGroup = kLanesPerGroup > 32 ? kLanesPerGroup / 32 : 1;
  constexpr bool kNeedsCrossWarpReduce = kLanesPerGroup > 32;
  // Per-block smem must hold one slot per warp; with multiple groups per
  // block, each group occupies kWarpsPerGroup contiguous slots.
  constexpr uint32_t kWarpsPerBlock = kThreadsPerBlock >= 32 ? kThreadsPerBlock / 32 : 1;
  constexpr uint32_t kReduceSmemElems = kNeedsCrossWarpReduce
                                            ? (kIsInt ? 2 * kWarpsPerBlock : kWarpsPerBlock)
                                            : 1;

  __shared__ float fht_smem[kFhtSmemElems];
  __shared__ float reduce_smem[kReduceSmemElems];

  uint32_t tid = threadIdx.x;
  uint32_t lane_in_tile = tid % kThreadsPerTile;
  uint32_t tile_in_block = tid / kThreadsPerTile;
  uint32_t tile_idx = blockIdx.x * kTilesPerBlock + tile_in_block;
  bool valid = tile_idx < num_tiles;

  float reg[E];

  // ---- Load ----
  if (valid) {
    const SourceType *gptr = in_ptr + tile_idx * kBlockSize + lane_in_tile * E;
    vec_load_to_float<SourceType, E>(reg, gptr);
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
  constexpr uint32_t kShflWidth = kThreadsPerTile < 32 ? kThreadsPerTile : 32;
  PRAGMA_UNROLL
  for (uint32_t s = 0; s < kShflStages; s++) {
    uint32_t partner_xor = 1u << s;
    bool is_low = (lane_in_tile & partner_xor) == 0;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < E; i++) {
      float other = __shfl_xor_sync(0xffffffff, reg[i], partner_xor, kShflWidth);
      reg[i] = is_low ? (reg[i] + other) : (other - reg[i]);
    }
  }

  // ---- Shared-memory stages (only when kThreadsPerTile > 32) ----
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
        fht_smem[my_base + i] = reg[i];
      __syncthreads();

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E; i++) {
        float other = fht_smem[partner_base + i];
        reg[i] = is_low ? (reg[i] + other) : (other - reg[i]);
      }
      __syncthreads();
    }
  }

  // ---- Per-group reduction ----
  // After FHT, lane `l` owns positions [l*E, l*E+E-1] in its tile.
  // With kGroupSize >= E and a multiple of E, all E regs of a lane fall in
  // the same group. group_in_tile = lane_in_tile / kLanesPerGroup.
  uint32_t group_in_tile = lane_in_tile / kLanesPerGroup;
  uint32_t lane_in_group = lane_in_tile % kLanesPerGroup;

  float local_max;
  float local_min;
  float local_absmax;

  if constexpr (kIsInt) {
    local_max = -1e30f;
    local_min = 1e30f;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < E; i++) {
      local_max = fmaxf(local_max, reg[i]);
      local_min = fminf(local_min, reg[i]);
    }
  } else {
    local_absmax = 0.f;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < E; i++) {
      local_absmax = fmaxf(local_absmax, fabsf(reg[i]));
    }
  }

  // Intra-warp cross-lane reduction within group.
  constexpr uint32_t kIntraWarpLanes =
      kLanesPerGroup < 32 ? kLanesPerGroup : 32;
  PRAGMA_UNROLL
  for (uint32_t step = 1; step < kIntraWarpLanes; step <<= 1) {
    if constexpr (kIsInt) {
      local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, step, kIntraWarpLanes));
      local_min = fminf(local_min, __shfl_xor_sync(0xffffffff, local_min, step, kIntraWarpLanes));
    } else {
      local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, step, kIntraWarpLanes));
    }
  }

  // Cross-warp reduction (only when one group spans multiple warps).
  if constexpr (kNeedsCrossWarpReduce) {
    uint32_t warp_id = tid / 32;
    uint32_t lane_id_in_warp = tid & 31;
    if constexpr (kIsInt) {
      local_max = cross_warp_max<kWarpsPerGroup>(local_max, reduce_smem, warp_id, lane_id_in_warp);
      local_min = -cross_warp_max<kWarpsPerGroup>(-local_min,
                                                  reduce_smem + kWarpsPerBlock,
                                                  warp_id, lane_id_in_warp);
    } else {
      local_absmax = cross_warp_max<kWarpsPerGroup>(local_absmax, reduce_smem, warp_id, lane_id_in_warp);
    }
  }

  // ---- Compute scale ----
  float scale_raw;
  if constexpr (kIsInt) {
    constexpr float pos_lim = (float)((1 << (TargetType::kBits - 1)) - 1) + 0.0f;
    constexpr float neg_lim = (float)(1 << (TargetType::kBits - 1)) + 0.0f;
    float s1 = local_max / pos_lim;
    float s2 = -local_min / neg_lim;
    scale_raw = fmaxf(fmaxf(s1, s2), 1e-30f);
  } else {
    float dtype_max = fp_target_max_value<TargetType>();
    scale_raw = fmaxf(local_absmax / dtype_max, 1e-30f);
  }

  // ---- Finalize + write scale (only one lane per group) ----
  float norm = rsqrtf((float)kBlockSize);
  if constexpr (kHasExtraScale) norm *= extra_scale;
  float scale_stored = scale_raw * norm;

  uint32_t scale_idx;
  if constexpr (kMMajor) {
    uint32_t num_blocks_per_row = groups_per_row / kGroupsPerTile;
    uint32_t row = tile_idx / num_blocks_per_row;
    uint32_t block_in_row = tile_idx - row * num_blocks_per_row;
    uint32_t group_global = block_in_row * kGroupsPerTile + group_in_tile;
    if constexpr (kScaleStore == kScaleStoreE8M0) {
      scale_idx = (group_global / 4) * (shape_m * 4) + row * 4 + (group_global % 4);
    } else {
      scale_idx = group_global * shape_m + row;
    }
  } else {
    scale_idx = tile_idx * kGroupsPerTile + group_in_tile;
  }

  float inv_scale;
  if constexpr (kScaleStore == kScaleStoreF32 && !kHasGlobalScale) {
    // Default path: bit-identical to the original f32-scale kernel.
    inv_scale = 1.f / scale_raw;
    if (valid && lane_in_group == 0)
      reinterpret_cast<float *>(scales_ptr)[scale_idx] = scale_stored;
  } else {
    float gs = 1.f, inv_gs = 1.f;
    if constexpr (kHasGlobalScale) {
      gs = __ldg(global_scale_ptr);
      inv_gs = 1.f / gs;
    }
    // Effective stored scale (quantized + global folded back in); data is
    // un-normalized here, so the per-element multiplier carries the norm factor.
    float eff_scale = finalize_scale<kScaleStore>(
        scale_stored, gs, inv_gs, scales_ptr, scale_idx, valid && lane_in_group == 0);
    inv_scale = norm / eff_scale;
  }

  // ---- Quantize + store ----
  if (valid) {
    constexpr uint32_t kBits = TargetType::kBits;
    if constexpr (kBits == 8) {
      // 1 byte per element. Pack E bytes (E=8 → uint2).
      uint8_t local_bytes[E];
      constexpr bool kIsFp8 =
          std::is_same<TargetType, Float8E4M3>::value ||
          std::is_same<TargetType, Float8E3M4>::value ||
          std::is_same<TargetType, Float8E5M2>::value;
      if constexpr (kIsFp8) {
        static_assert(E % 2 == 0, "fp8 requires even E (paired HW cvt)");
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < E; i += 2) {
          uint16_t pair = quant_pair_fp8<TargetType>(
              reg[i] * inv_scale, reg[i + 1] * inv_scale);
          local_bytes[i] = pair & 0xFFu;
          local_bytes[i + 1] = (pair >> 8) & 0xFFu;
        }
      } else {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < E; i++) {
          local_bytes[i] = static_cast<uint8_t>(quant_one_value<TargetType>(reg[i], inv_scale));
        }
      }
      uint8_t *gptr = reinterpret_cast<uint8_t *>(out_ptr) + tile_idx * kBlockSize + lane_in_tile * E;
      if constexpr (E == 8) {
        uint2 packed;
        packed.x = *reinterpret_cast<uint32_t *>(&local_bytes[0]);
        packed.y = *reinterpret_cast<uint32_t *>(&local_bytes[4]);
        *reinterpret_cast<uint2 *>(gptr) = packed;
      } else if constexpr (E == 4) {
        uint32_t packed = *reinterpret_cast<uint32_t *>(&local_bytes[0]);
        *reinterpret_cast<uint32_t *>(gptr) = packed;
      } else {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < E; i++)
          gptr[i] = local_bytes[i];
      }
    } else if constexpr (kBits == 4) {
      // Two elements per byte. Out tile has kBlockSize/2 bytes.
      static_assert(E % 2 == 0);
      constexpr bool kIsFp4 = std::is_same<TargetType, Float4E2M1>::value ||
                              std::is_same<TargetType, Float4E0M3>::value;
      uint8_t local_bytes[E / 2];
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < E / 2; i++) {
        if constexpr (kIsFp4) {
          local_bytes[i] = quant_pair_fp4<TargetType>(
              reg[2 * i] * inv_scale, reg[2 * i + 1] * inv_scale);
        } else {
          uint32_t a = quant_one_value<TargetType>(reg[2 * i], inv_scale) & 0xFu;
          uint32_t b = quant_one_value<TargetType>(reg[2 * i + 1], inv_scale) & 0xFu;
          local_bytes[i] = static_cast<uint8_t>(a | (b << 4));
        }
      }
      uint8_t *gptr = reinterpret_cast<uint8_t *>(out_ptr) + tile_idx * (kBlockSize / 2) + lane_in_tile * (E / 2);
      if constexpr (E / 2 == 4) {
        *reinterpret_cast<uint32_t *>(gptr) = *reinterpret_cast<uint32_t *>(&local_bytes[0]);
      } else if constexpr (E / 2 == 2) {
        *reinterpret_cast<uint16_t *>(gptr) = *reinterpret_cast<uint16_t *>(&local_bytes[0]);
      } else {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < E / 2; i++)
          gptr[i] = local_bytes[i];
      }
    } else {
      static_assert(kBits == 8 || kBits == 4, "only 4-bit or 8-bit targets supported");
    }
  }
}
