// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class S2RMemoryLoaderAS {
private:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;

  static constexpr bool kUseWgmma = Ctx::kUseWgmma;
  static constexpr bool kUseMxmma = Ctx::kUseMxmma;
  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelScale = kHasInputScale && Ctx::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupScale = kHasInputScale && Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kMMajorInputScale = Ctx::kUseMMajorInputScale && kIsGroupScale;

  static constexpr uint32_t kGroupSize = kIsGroupScale ? Ctx::kInputScaleGroupSize : BlockShape::K;
  static constexpr uint32_t kPartMmaShapeK = Ctx::kPartMmaShapeK;
  static constexpr uint32_t M_WARPS = Ctx::M_WARPS;
  static constexpr uint32_t N_WARPS = Ctx::N_WARPS;
  static constexpr uint32_t kNumLinesPerBlock = kUseWgmma && kIsGroupScale ? 2 : 1;
  static constexpr uint32_t kSmemStride = CEIL_DIV(BlockShape::K / (kUseMxmma ? 4 : 1), kGroupSize);

  Ctx &ctx;

public:
  CUDA_INLINE S2RMemoryLoaderAS(Ctx &ctx) : ctx(ctx) {}

  CUDA_INLINE
  void load_sf(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    const uint32_t *smem_ptr_load = reinterpret_cast<const uint32_t *>(smem_ptr);
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;

    uint32_t m_warp_base = (warp_id / N_WARPS % M_WARPS) * WarpShape::M;
    uint32_t k_warp_base = (warp_id / (M_WARPS * N_WARPS)) * WarpShape::K + iter_id * kPartMmaShapeK;
    uint32_t word_base = k_warp_base / kGroupSize / 4;
    uint32_t base = word_base * BlockShape::M + m_warp_base;

    constexpr uint32_t kNumLoadsPart2 = WarpShape::M % 32 == 0 ? 0 : 1;
    constexpr uint32_t kNumLoadsPart1 = (WarpShape::M - kNumLoadsPart2 * 16) / 32;

    PRAGMA_UNROLL
    for (uint32_t m = 0; m < kNumLoadsPart1; m++) {
      uint32_t m_index = lane_id / 4 + lane_id % 4 * 8 + 32 * m;
      regs_ptr[m] = smem_ptr_load[base + m_index];
    }

    if constexpr (kNumLoadsPart2 == 1) {
      uint32_t m_index = lane_id / 4 + lane_id % 2 * 8 + kNumLoadsPart1 * 32;
      regs_ptr[kNumLoadsPart1] = smem_ptr_load[base + m_index];
    }
  }

  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    uint32_t lane_id = ctx.lane_id();
    uint32_t sub_row;

    if constexpr (kUseWgmma && kIsChannelScale) {
      sub_row = (lane_id % 4) * 2 + (lane_id % 8) / 4;
    } else if constexpr (kUseWgmma && kIsGroupScale) {
      sub_row = (lane_id % 4) * 2;
    } else if constexpr (!kUseWgmma) {
      sub_row = lane_id / 4;
    }

    uint32_t group_index = 0;
    if constexpr (kGroupSize < BlockShape::K) {
      uint32_t k_index = ctx.k_warp_offset() + iter_id * kPartMmaShapeK;
      group_index = k_index / kGroupSize;
    };

    uint32_t *reg_ptr_load = reinterpret_cast<uint32_t *>(regs_ptr);
    const uint32_t *smem_ptr_load = reinterpret_cast<const uint32_t *>(smem_ptr);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < WarpShape::M / 8; i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumLinesPerBlock; j++) {
        uint32_t m_index = ctx.m_warp_offset() + i * 8 + sub_row + j;
        uint32_t smem_idx;
        if constexpr (kMMajorInputScale) {
          smem_idx = group_index * BlockShape::M + m_index;
        } else {
          smem_idx = m_index * kSmemStride + group_index;
        }
        uint32_t reg_idx = i * kNumLinesPerBlock + j;
        reg_ptr_load[reg_idx] = smem_ptr_load[smem_idx];
      }
    }
  }
};
