// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class S2RMemoryLoaderB {
private:
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;

  static constexpr uint32_t N_WARPS = Ctx::N_WARPS;
  static constexpr uint32_t K_WARPS = Ctx::K_WARPS;

  static constexpr bool kIsWarpHalfGroup = WarpShape::N == ElementA::kBits * 2;
  static constexpr bool kLoadHalfGroup = ElementB::kBits % 2 == 0 && kIsWarpHalfGroup;
  static constexpr uint32_t TRUE_N_WARPS = kIsWarpHalfGroup ? N_WARPS / 2 : N_WARPS;
  static constexpr uint32_t kSmemStride = BlockShape::N * Ctx::kPartMmaShapeK * ElementB::kBits / 32 / 4 * (Ctx::kUsePackedKLayout ? 2 : 1);
  static constexpr uint32_t kNumIntsPerThread = ElementB::kBits / (kLoadHalfGroup ? 2 : 1);
  using LoadType = typename LoadTypeChooser<kNumIntsPerThread * 4>::Type;
  static constexpr uint32_t kLoadIters = kNumIntsPerThread / (sizeof(LoadType) / 4);

  Ctx &ctx;

public:
  CUDA_INLINE S2RMemoryLoaderB(Ctx &ctx) : ctx(ctx) {}

  CUDA_INLINE
  void load_packed_k(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    static_assert(Ctx::kPartMmaShapeK == 32);
    static_assert(ElementA::kBits == 8);
    static_assert(Ctx::kUseWgmma);
    static_assert(Ctx::kWarpIters == WarpShape::N / 16);
    static_assert(WarpShape::N / 16 >= 2);

    uint32_t n_warp_id = ctx.n_warp_id();
    uint32_t lane_id = ctx.lane_id();

    constexpr uint32_t kKChunks = WarpShape::K / 64;
    constexpr uint32_t kPlane = N_WARPS * (WarpShape::N / 16) * 32;

    uint32_t idx = n_warp_id * (WarpShape::N / 16) + iter_id;
    idx = idx * 32 + lane_id;

    if constexpr (K_WARPS > 1) {
      idx += kPlane * kKChunks * ctx.k_warp_id();
    }

    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);
    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);

    PRAGMA_UNROLL
    for (uint32_t kc = 0; kc < kKChunks; kc++) {
      uint32_t smem_start_idx = (idx + kc * kPlane) * kLoadIters;
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kLoadIters; j++) {
        reg_ptr_load[kc * kLoadIters + j] = smem_ptr_load[smem_start_idx + j];
      }
    }
  }

  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    if constexpr (Ctx::kUsePackedKLayout) return load_packed_k(smem_ptr, regs_ptr, iter_id);
    uint32_t warp_id = ctx.warp_id();
    uint32_t n_warp_id = ctx.n_warp_id();
    if (kIsWarpHalfGroup) n_warp_id = n_warp_id / 2;
    uint32_t lane_id = ctx.lane_id();
    constexpr uint32_t warp_weight_blocks = MAX(WarpShape::N / (ElementA::kBits * 4), 1);
    uint32_t idx = warp_weight_blocks * 32 * n_warp_id + lane_id;

    if constexpr (K_WARPS > 1) {
      idx = TRUE_N_WARPS * 32 * warp_weight_blocks * Ctx::kWarpIters * ctx.k_warp_id() + idx;
    }

    uint32_t smem_start_idx = idx * kLoadIters;
    smem_ptr = smem_ptr + kSmemStride * iter_id;
    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);
    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < warp_weight_blocks; i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kLoadIters; j++) {
        if constexpr (kLoadHalfGroup) {
          reg_ptr_load[i * kLoadIters + j] = smem_ptr_load[(smem_start_idx + 32 * kLoadIters * i) * 2 + warp_id % 2 * kLoadIters + j];
        } else {
          reg_ptr_load[i * kLoadIters + j] = smem_ptr_load[smem_start_idx + 32 * kLoadIters * i + j];
        }
      }
    }
  };
};
