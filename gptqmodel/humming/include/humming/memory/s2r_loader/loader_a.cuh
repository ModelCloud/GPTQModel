// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class S2RMemoryLoaderA {
private:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using MmaShape = typename Ctx::MmaShape;
  using SharedStorage = typename Ctx::SharedStorage;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;

  Ctx &ctx;

public:
  CUDA_INLINE S2RMemoryLoaderA(Ctx &ctx) : ctx(ctx) {}

  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id, uint32_t stage_id = 0) {
    const uint32_t lane_id = ctx.lane_id();
    const uint32_t m_iter_id = ctx.m_warp_id();
    const uint32_t k_warp_id = ctx.k_warp_id();
    uint32_t smem_uint = offsetof(SharedStorage, stages) + stage_id * sizeof(typename SharedStorage::StageStorage);
    uint32_t smem_base = smem_uint / 128 % (BlockShape::K * ElementA::kBits == 512 ? 4 : 8);

    PRAGMA_UNROLL
    for (uint32_t load_iter_id = 0; load_iter_id < CEIL_DIV(WarpShape::M, 16); load_iter_id++) {

      uint32_t row = ctx.m_warp_offset() + load_iter_id * 16;
      uint32_t col = iter_id * 2 + k_warp_id * (Ctx::kWarpIters * 2);

      if constexpr (MmaShape::M == 8) {
        row += (lane_id / 16) * 8 + lane_id % 8;
        col += (lane_id / 8) % 2;
      } else {
        row += lane_id % 16;
        col += lane_id / 16;
      }

      if constexpr (BlockShape::K * ElementA::kBits > 1024) {
        row = BlockShape::M * (col / 8) + row;
        col = (col % 8) ^ ((row + smem_base) % 8);
      } else if constexpr (BlockShape::K * ElementA::kBits == 1024) {
        col = col ^ ((row + smem_base) % 8);
      } else if constexpr (BlockShape::K * ElementA::kBits == 512) {
        col = row % 2 * 4 + col;
        row = row / 2;
        col = col ^ ((row + smem_base) % 4);
      }

      uint32_t a_sh_rd = row * 8 + col;

      if ((load_iter_id == CEIL_DIV(WarpShape::M, 16) - 1) && WarpShape::M % 16 == 8) {
        ld_shared<2>(smem_ptr + a_sh_rd, reinterpret_cast<int4 *>(regs_ptr) + load_iter_id);
      } else {
        ld_shared<4>(smem_ptr + a_sh_rd, reinterpret_cast<int4 *>(regs_ptr) + load_iter_id);
      }
    };
  };
};
