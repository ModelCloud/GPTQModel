// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>

template <class Ctx, class MMA>
class EpilogueSmemReducer {
private:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementC = typename Ctx::ElementC;

  static constexpr bool kHasGroupScale = Ctx::kWeightScaleGroupSize > 0 || Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;
  static constexpr bool kForceFloatAccum = std::is_same<typename MmaOpClass::ValTypeC, int32_t>::value && kHasGroupScale && !Ctx::kUseIntWeightScale && !kUseFusedE8m0Scale;
  using MmaTypeC = std::conditional_t<kForceFloatAccum, float, typename MmaOpClass::ValTypeC>;
  using MmaShape = typename MmaOpClass::MmaShape;
  using OutputType32 = std::conditional_t<std::is_same<ElementC, Float16>::value, half2, nv_bfloat162>;
  using ValTypeC32 = std::conditional_t<sizeof(MmaTypeC) == 2, OutputType32, MmaTypeC>;

  using CRegistersArrayType = typename MMA::CRegistersArrayType;

public:
  Ctx &ctx;

  CUDA_INLINE
  EpilogueSmemReducer(Ctx &ctx) : ctx(ctx) {
  }

  CUDA_INLINE
  void reduce(uint32_t *regs_ptr) {
    constexpr uint32_t num_int4s = sizeof(CRegistersArrayType) / 16;
    constexpr uint32_t num_reduce_iters = MAX(CEIL_DIV(WarpShape::M, 16), 1);
    constexpr uint32_t num_int4s_per_time = CEIL_DIV(num_int4s, num_reduce_iters);
    constexpr uint32_t group_num_warps = BlockShape::K / WarpShape::K;
    constexpr uint32_t num_groups = Ctx::kNumMathThreads / 32 / group_num_warps;
    uint32_t group_id = ctx.warp_id() % num_groups;
    uint32_t group_warp_id = ctx.warp_id() / num_groups;
    uint32_t laneid = ctx.lane_id();

    using ReductionSmemType = int4[group_num_warps / 2][num_groups][num_int4s_per_time][32];
    auto &smem_arr = *reinterpret_cast<ReductionSmemType *>(ctx.smem.reduce);

    auto write_to_smem = [&](uint32_t buffer_id, uint32_t m) {
      int4 *regs_int4_ptr = reinterpret_cast<int4 *>(regs_ptr) + m * num_int4s_per_time;

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < num_int4s_per_time; i++) {
        if (m * num_int4s_per_time + i < num_int4s) {
          smem_arr[buffer_id][group_id][i][laneid] = regs_int4_ptr[i];
        }
      };
    };

    auto read_from_smem_and_reduce = [&](uint32_t buffer_id, uint32_t m) {
      int4 *regs_int4_ptr = reinterpret_cast<int4 *>(regs_ptr) + m * num_int4s_per_time;

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < num_int4s_per_time; i++) {
        if (m * num_int4s_per_time + i >= num_int4s) continue;
        int4 val = smem_arr[buffer_id][group_id][i][laneid];

        ValTypeC32 *sval_scalar_ptr = reinterpret_cast<ValTypeC32 *>(&val);
        ValTypeC32 *regs_scalar_ptr = reinterpret_cast<ValTypeC32 *>(regs_int4_ptr + i);

        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 4; j++) {
          if constexpr (sizeof(MmaTypeC) == 2) {
            regs_scalar_ptr[j] = __hadd2(regs_scalar_ptr[j], sval_scalar_ptr[j]);
          } else {
            regs_scalar_ptr[j] += sval_scalar_ptr[j];
          }
        }
      };
    };

    PRAGMA_UNROLL
    for (uint32_t m = 0; m < num_reduce_iters; m++) {
      PRAGMA_UNROLL
      for (uint32_t i = 1; i < group_num_warps; i *= 2) {
        uint32_t buffer_id = group_warp_id % (group_num_warps / (2 * i));
        if (group_warp_id >= group_num_warps / i) {
          ctx.sync_math_threads();
        } else if (group_warp_id >= group_num_warps / (2 * i)) {
          write_to_smem(buffer_id, m);
          ctx.sync_math_threads();
        } else {
          ctx.sync_math_threads();
          read_from_smem_and_reduce(buffer_id, m);
        };

        ctx.sync_math_threads();
      };
    };
  };
};
