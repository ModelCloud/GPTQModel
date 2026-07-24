// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class S2RMemoryLoaderBias {
private:
  using WarpShape = typename Ctx::WarpShape;

  Ctx &ctx;

public:
  CUDA_INLINE S2RMemoryLoaderBias(Ctx &ctx) : ctx(ctx) {}

  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t pred) {
    uint32_t warp_id = ctx.warp_id();
    uint32_t n_warp_id = ctx.n_warp_id();
    uint32_t lane_id = ctx.lane_id();
    uint32_t bias_sh_rd = Ctx::kUseWgmma ? (lane_id / 8) : (lane_id % 4);

    if constexpr (WarpShape::N == 16) {
      bias_sh_rd = (n_warp_id / 2) * 8 + bias_sh_rd * 2 + warp_id % 2;
      const int2 *smem_ptr_load = reinterpret_cast<const int2 *>(smem_ptr);
      int2 *reg_ptr_load = reinterpret_cast<int2 *>(regs_ptr);
      reg_ptr_load[0] = pred ? smem_ptr_load[bias_sh_rd] : int2();
    } else {
      bias_sh_rd += n_warp_id * (WarpShape::N / 16 * 2);

      const int4 *smem_ptr_load = smem_ptr + bias_sh_rd;
      int4 *reg_ptr_load = reinterpret_cast<int4 *>(regs_ptr);
      constexpr uint32_t kLoadIters = WarpShape::N / 4 * 16 / 8 / 16;

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kLoadIters; i++) {
        reg_ptr_load[i] = pred ? smem_ptr_load[i * 4] : int4();
      };
    }
  }
};
