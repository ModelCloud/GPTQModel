// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class S2RMemoryLoaderBS {
private:
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementBS = typename Ctx::ElementBS;

  static constexpr bool kUseWgmma = Ctx::kUseWgmma;
  static constexpr bool kUseMxmma = Ctx::kUseMxmma;
  static constexpr bool kIsChannel = Ctx::kIsChannelWeightScale;
  static constexpr bool kIsBlock = Ctx::kIsBlockWeightScale;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;
  static constexpr uint32_t kGroupSize = Ctx::kWeightScaleGroupSize > 0 ? Ctx::kWeightScaleGroupSize : BlockShape::K;
  static constexpr uint32_t kGroupSizeN = Ctx::kWeightScaleGroupSizeN;
  static constexpr uint32_t kPartMmaShapeK = Ctx::kPartMmaShapeK;
  static constexpr uint32_t M_WARPS = Ctx::M_WARPS;
  static constexpr uint32_t N_WARPS = Ctx::N_WARPS;
  static constexpr uint32_t kSmemStride = BlockShape::N * ElementBS::kBits / 32 / 4;
  static constexpr uint32_t kScaleVec = kPartMmaShapeK / kGroupSize;

  Ctx &ctx;

public:
  CUDA_INLINE S2RMemoryLoaderBS(Ctx &ctx) : ctx(ctx) {}

  CUDA_INLINE
  void load_sf(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    const uint32_t *smem_ptr_load = reinterpret_cast<const uint32_t *>(smem_ptr);
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;

    uint32_t col_in_tile = lane_id / 4;
    uint32_t k_warp_base = (warp_id / (M_WARPS * N_WARPS)) * WarpShape::K + iter_id * kPartMmaShapeK;
    uint32_t group_base = k_warp_base / kGroupSize / (kScaleVec == 1 ? 2 : 4);
    uint32_t n_warp_base = (warp_id % N_WARPS) * WarpShape::N;

    if constexpr (kScaleVec == 1 && WarpShape::N == 16) {
      uint32_t s_sh_rd = lane_id / 4;
      regs_ptr[0] = smem_ptr_load[group_base * (BlockShape::N / 2) + (warp_id % N_WARPS) * (WarpShape::N / 2) + s_sh_rd];
    } else if constexpr (WarpShape::N == 32 && kScaleVec == 1) {
      uint32_t s_sh_rd = (lane_id % 2) * 8 + lane_id / 4;
      regs_ptr[0] = smem_ptr_load[group_base * (BlockShape::N / 2) + (warp_id % N_WARPS) * (WarpShape::N / 2) + s_sh_rd];
    } else if constexpr (WarpShape::N == 16) {
      uint32_t s_sh_rd = (lane_id / 2) % 2 * 8 + lane_id / 4;
      regs_ptr[0] = smem_ptr_load[group_base * WarpShape::N + n_warp_base + s_sh_rd];
    } else {
      uint32_t s_sh_rd = lane_id % 4 * 8 + lane_id / 4;
      constexpr uint32_t kRowStride = kScaleVec == 1 ? BlockShape::N / 2 : BlockShape::N;
      uint32_t nb = (warp_id % N_WARPS) * (kScaleVec == 1 ? WarpShape::N / 2 : WarpShape::N);

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < WarpShape::N / (kScaleVec == 1 ? 64 : 32); i++) {
        regs_ptr[i] = smem_ptr_load[group_base * kRowStride + i * 32 + nb + s_sh_rd];
      }
    }
  }

  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    if constexpr (kIsBlock) {
      load_block(smem_ptr, regs_ptr, iter_id);
    } else if constexpr (!kUseFusedE8m0Scale && (kIsChannel || (!kUseWgmma && ElementA::kBits != 16))) {
      load_layout2(smem_ptr, regs_ptr, iter_id);
    } else {
      load_layout1(smem_ptr, regs_ptr, iter_id);
    }
  }

  CUDA_INLINE
  void load_block(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    static_assert(kGroupSizeN >= 64);

    uint32_t index = ctx.n_warp_offset() / kGroupSizeN;
    if constexpr (BlockShape::K >= kGroupSize) {
      uint32_t k_index = ctx.k_warp_offset() + iter_id * kPartMmaShapeK;
      uint32_t group_index = k_index / kGroupSize;
      index += group_index * CEIL_DIV(BlockShape::N, kGroupSizeN);
    }
    regs_ptr[0] = reinterpret_cast<const uint32_t *>(smem_ptr)[index];
  };

  CUDA_INLINE
  void load_layout1(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    uint32_t lane_id = ctx.lane_id();
    uint32_t n_warp_id = ctx.n_warp_id();

    using LoadType = typename LoadTypeChooser<WarpShape::N / 8 * ElementBS::kBits / 8>::Type;
    constexpr uint32_t kNumWarpsPerPackBlock = 64 / WarpShape::N;
    constexpr uint32_t kPackBlockSize = 64 * ElementBS::kBits / 8 / sizeof(LoadType);
    uint32_t pack_block_id = n_warp_id / kNumWarpsPerPackBlock;

    uint32_t s_sh_rd = (lane_id / 4) * kNumWarpsPerPackBlock + n_warp_id % kNumWarpsPerPackBlock + pack_block_id * kPackBlockSize;

    if constexpr (kGroupSize < BlockShape::K) {
      uint32_t k_index = ctx.k_warp_offset() + iter_id * kPartMmaShapeK;
      uint32_t group_index = k_index / kGroupSize;
      s_sh_rd += group_index * (kSmemStride * 16 / sizeof(LoadType));
    };

    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);
    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);
    reg_ptr_load[0] = smem_ptr_load[s_sh_rd];
  }

  CUDA_INLINE
  void load_layout2(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    uint32_t lane_id = ctx.lane_id();
    uint32_t n_warp_id = ctx.n_warp_id();

    using LoadType = typename LoadTypeChooser<MIN(WarpShape::N, 32) / 4 * ElementBS::kBits / 8>::Type;
    constexpr uint32_t kNumWarpsPerPackBlock = 32 / MIN(WarpShape::N, 32);
    constexpr uint32_t kPackBlockSize = 32 * ElementBS::kBits / 8 / sizeof(LoadType);

    uint32_t pack_block_id = n_warp_id / kNumWarpsPerPackBlock;
    uint32_t s_sh_rd = (kIsChannel && kUseWgmma) ? (lane_id / 8) : (lane_id % 4);
    s_sh_rd = s_sh_rd * kNumWarpsPerPackBlock + n_warp_id % kNumWarpsPerPackBlock + pack_block_id * kPackBlockSize * CEIL_DIV(WarpShape::N, 32);

    if constexpr (kGroupSize < BlockShape::K) {
      uint32_t k_index = ctx.k_warp_offset() + iter_id * kPartMmaShapeK;
      uint32_t group_index = k_index / kGroupSize;
      s_sh_rd += group_index * (kSmemStride * 16 / sizeof(LoadType));
    };

    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);
    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);

    PRAGMA_UNROLL
    for (uint32_t j = 0; j < CEIL_DIV(WarpShape::N, 32); j++) {
      reg_ptr_load[j] = smem_ptr_load[s_sh_rd + kPackBlockSize * j];
    }
  };
};
