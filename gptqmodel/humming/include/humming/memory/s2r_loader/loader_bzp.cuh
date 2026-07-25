// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class S2RMemoryLoaderBZP {
private:
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;

  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kIsChannelScale = Ctx::kIsChannelWeightScale;
  static constexpr bool kIsGroupScale = Ctx::kIsGroupWeightScale;
  static constexpr uint32_t kGroupSize = kIsChannelScale ? BlockShape::K : Ctx::kWeightScaleGroupSize;

  static constexpr uint32_t kPartMmaShapeK = Ctx::kPartMmaShapeK;
  static constexpr uint32_t M_WARPS = Ctx::M_WARPS;
  static constexpr uint32_t N_WARPS = Ctx::N_WARPS;
  static constexpr uint32_t K_WARPS = Ctx::K_WARPS;

  static constexpr uint32_t kNumZPBits = kIsFpZeroPoint ? 16 : MAX(4, static_next_power_of_2(ElementB::kBits));
  static constexpr uint32_t kLoadBytes = WarpShape::N / 8 * kNumZPBits / 8;
  using LoadType = typename LoadTypeChooser<kLoadBytes>::Type;
  static constexpr uint32_t kSmemStride = BlockShape::N * kNumZPBits / 32 / 4;
  static constexpr uint32_t kSmemStrideLoadType = kSmemStride * 16 / sizeof(LoadType);

  Ctx &ctx;

public:
  CUDA_INLINE S2RMemoryLoaderBZP(Ctx &ctx) : ctx(ctx) {}

  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    constexpr uint32_t kNumLoadBlockEvery64Rows = (64 * kNumZPBits) / kLoadBytes;
    constexpr uint32_t kNumWarpsEvery64Rows = 64 / WarpShape::N;
    uint32_t warp_id = ctx.warp_id();
    uint32_t lane_id = ctx.lane_id();

    if constexpr (!kIsFpZeroPoint) {
      uint32_t zp_sh_rd = ctx.n_warp_id() / kNumWarpsEvery64Rows * (8 * kNumWarpsEvery64Rows);
      zp_sh_rd += lane_id / 4 * kNumWarpsEvery64Rows + warp_id % kNumWarpsEvery64Rows;

      if constexpr (kGroupSize < BlockShape::K) {
        uint32_t k_index = ctx.k_warp_offset() + iter_id * kPartMmaShapeK;
        uint32_t group_index = k_index / kGroupSize;
        zp_sh_rd += group_index * kSmemStrideLoadType;
      };

      LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);
      const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);

      reg_ptr_load[0] = smem_ptr_load[zp_sh_rd];
    } else {
      static_assert(ElementA::kBits == 16);
      uint32_t zp_sh_rd = lane_id / 4 + (ctx.n_warp_id() / (64 / WarpShape::N)) * 8;
      if constexpr (kGroupSize < BlockShape::K) {
        uint32_t k_index = ctx.k_warp_offset() + iter_id * kPartMmaShapeK;
        uint32_t group_index = k_index / kGroupSize;
        zp_sh_rd += group_index * kSmemStrideLoadType;
      };
      LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);
      const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);
      if (WarpShape::N == 32) zp_sh_rd = zp_sh_rd * 2 + warp_id % 2;
      reg_ptr_load[0] = smem_ptr_load[zp_sh_rd];
    }
  }
};
