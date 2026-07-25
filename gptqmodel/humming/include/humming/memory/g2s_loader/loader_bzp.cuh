// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class G2SMemoryLoaderBZP {
private:
  using ProblemShape = typename Ctx::ProblemShape;
  using BlockShape = typename Ctx::BlockShape;
  using ElementB = typename Ctx::ElementB;

  static constexpr bool kUseWarpSpec = Ctx::kUseWarpSpec;
  static constexpr bool kUseTma = Ctx::kUseTmaBZP;
  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;
  static constexpr uint32_t kNumLoadThreads = Ctx::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = Ctx::kNumThreads - kNumLoadThreads;

  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kIsChannel = Ctx::kIsChannelWeightScale;
  static constexpr bool kIsGroup = Ctx::kIsGroupWeightScale;
  static constexpr uint32_t kGroupSize = kIsGroup ? Ctx::kWeightScaleGroupSize : ProblemShape::K;

  static constexpr uint32_t kNumZPBits = kIsFpZeroPoint ? 16 : MAX(4, static_next_power_of_2(ElementB::kBits));
  static constexpr uint32_t kSmemStride = BlockShape::N * kNumZPBits / 32 / 4;
  static constexpr uint32_t kGmemStride = ProblemShape::N * kNumZPBits / 32 / 4;
  static constexpr uint32_t kProblemNumGroups = CEIL_DIV(ProblemShape::K, kGroupSize);
  static constexpr uint32_t kGmemExpertStride = kGmemStride * kProblemNumGroups;
  static constexpr uint32_t kNumGroups = CEIL_DIV(BlockShape::K, kGroupSize);
  static constexpr uint32_t kNumInt4s = kSmemStride * kNumGroups;
  static constexpr uint32_t kLoadsPerGroup = kIsChannel ? 1 : CEIL_DIV(kGroupSize, BlockShape::K);

public:
  Ctx &ctx;
  const CUtensorMap *tensor_map_ptr;
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  uint32_t row_offset;
  uint32_t col_offset;
  uint32_t counter = 0;

  CUDA_INLINE
  G2SMemoryLoaderBZP(Ctx &ctx) : ctx(ctx) {
    const void *ptr = ctx.params.bzp;
    if constexpr (kUseTma) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<const int4 *>(ptr);
    }
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(int4 *smem_ptr, void *mbar_ptr) {
    counter = kLoadsPerGroup != 1 ? (counter + 1) % kLoadsPerGroup : 0;
    if constexpr (kUseTma) load_tma(smem_ptr, mbar_ptr);
    else load_legacy(smem_ptr);
    if constexpr (kShouldAdvance) advance();
  }

  CUDA_INLINE
  void load_tma(int4 *smem_ptr, void *mbar_ptr) {
    if (ctx.load_thread_id() == 0) tma_load_2d(tensor_map_ptr, smem_ptr, mbar_ptr, col_offset, row_offset);
  }

  CUDA_INLINE void load_legacy(int4 *smem_ptr) {
    legacy_load_2d<kUseCpAsync, kNumInt4s, kNumLoadThreads, kGmemStride, kSmemStride, kLoadThreadOffset>(gmem_ptr, smem_ptr);
  }

  CUDA_INLINE
  void advance() {
    if (kIsGroup && (kLoadsPerGroup == 1 || counter == 0)) {
      row_offset += kNumGroups;
      gmem_ptr += kGmemStride * kNumGroups;
    }
  };

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t n_block_id, uint32_t k_block_id) {
    row_offset = kProblemNumGroups * expert_id;
    col_offset = n_block_id * (BlockShape::N * kNumZPBits / 32);

    if constexpr (kIsGroup) {
      if constexpr (BlockShape::K >= kGroupSize) {
        row_offset += k_block_id * kNumGroups;
      } else {
        row_offset += (k_block_id * BlockShape::K) / kGroupSize;
      }
    }

    uint32_t gmem_offset = row_offset * kGmemStride + n_block_id * kSmemStride;
    gmem_ptr = gmem_ptr_raw + gmem_offset;
  };
};
