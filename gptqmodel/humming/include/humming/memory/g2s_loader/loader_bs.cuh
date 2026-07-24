// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class G2SMemoryLoaderBS {
private:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using ProblemShape = typename Ctx::ProblemShape;
  using BlockShape = typename Ctx::BlockShape;
  using ElementBS = typename Ctx::ElementBS;

  static constexpr bool kUseMxmma = Ctx::kUseMxmma;
  static constexpr bool kUseWarpSpec = Ctx::kUseWarpSpec;
  static constexpr bool kUseTma = Ctx::kUseTmaBS;
  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;
  static constexpr uint32_t kNumLoadThreads = Ctx::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = Ctx::kNumThreads - kNumLoadThreads;

  static constexpr bool kIsChannel = Ctx::kIsChannelWeightScale;
  static constexpr bool kIsGroup = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsBlock = Ctx::kIsBlockWeightScale;
  static constexpr bool kIsGroupOrBlock = kIsGroup || kIsBlock;
  static constexpr uint32_t kGroupSize = Ctx::kWeightScaleGroupSize > 0 ? Ctx::kWeightScaleGroupSize : ProblemShape::K;
  static constexpr uint32_t kGroupSizeN = kIsBlock ? Ctx::kWeightScaleGroupSizeN : 1;

  static constexpr uint32_t kSmemStride = CEIL_DIV(BlockShape::N, kGroupSizeN) * ElementBS::kBits / 32 / 4;
  static constexpr uint32_t kGmemStride = ProblemShape::N * ElementBS::kBits / 32 / 4;
  static constexpr uint32_t kProblemNumGroups = CEIL_DIV(ProblemShape::K, kGroupSize);
  static constexpr uint32_t kGmemExpertStride = kGmemStride * kProblemNumGroups;
  static constexpr uint32_t kNumGroups = CEIL_DIV(BlockShape::K, kGroupSize);
  static constexpr uint32_t kNumInt4s = kSmemStride * kNumGroups;
  static constexpr uint32_t kLoadsPerGroup = kIsChannel ? 1 : CEIL_DIV(kGroupSize, BlockShape::K);

  static constexpr uint32_t kPartMmaShapeK = 256 / MmaOpClass::kATypeBits;
  static constexpr uint32_t kMxScaleVec = kPartMmaShapeK / kGroupSize;
  static constexpr uint32_t kMxSmemStride = BlockShape::N / (kMxScaleVec == 1 ? 8 : 4);
  static constexpr uint32_t kMxGmemStride = ProblemShape::N / (kMxScaleVec == 1 ? 8 : 4);
  static constexpr uint32_t kMxGmemExpertStride = kMxGmemStride * kProblemNumGroups / (kMxScaleVec == 1 ? 2 : 4);
  static constexpr uint32_t kMxNumInt4s = kMxSmemStride * kNumGroups / (kMxScaleVec == 1 ? 2 : 4);

public:
  Ctx &ctx;
  const CUtensorMap *tensor_map_ptr;
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  uint32_t row_offset = 0;
  uint32_t col_offset;
  uint32_t counter = 0;

  CUDA_INLINE
  G2SMemoryLoaderBS(Ctx &ctx) : ctx(ctx) {
    const void *ptr = ctx.params.bs;
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
  };

  CUDA_INLINE
  void load_tma(int4 *smem_ptr, void *mbar_ptr) {
    if (ctx.load_thread_id() == 0) {
      if constexpr (!kUseMxmma) tma_load_3d(tensor_map_ptr, smem_ptr, mbar_ptr, 0, col_offset, row_offset);
      else tma_load_2d(tensor_map_ptr, smem_ptr, mbar_ptr, col_offset, row_offset);
    }
  }

  CUDA_INLINE
  void load_legacy(int4 *smem_ptr) {
    if constexpr (kIsBlock) {
      constexpr uint32_t kLoadStride = ProblemShape::N / kGroupSizeN;
      constexpr uint32_t kNW = CEIL_DIV(BlockShape::N, kGroupSizeN);
      constexpr uint32_t kNumScales = CEIL_DIV(BlockShape::K, kGroupSize) * kNW;
      static_assert(kNumScales <= kNumLoadThreads);
      const uint32_t thread_id = ctx.load_thread_id();
      if (thread_id < kNumScales) {
        const uint32_t *gmem_ptr_load = reinterpret_cast<const uint32_t *>(gmem_ptr_raw);
        uint32_t *smem_ptr_load = reinterpret_cast<uint32_t *>(smem_ptr);
        uint32_t gmem_row = row_offset + thread_id / kNW;
        uint32_t gmem_col = col_offset + thread_id % kNW;
        legacy_load<Ctx::kUseCpAsync>(&gmem_ptr_load[gmem_row * kLoadStride + gmem_col], &smem_ptr_load[thread_id]);
      }
    } else if constexpr (kUseMxmma) {
      legacy_load_2d<
          kUseCpAsync, kMxNumInt4s, kNumLoadThreads,
          kMxGmemStride, kMxSmemStride, kLoadThreadOffset>(gmem_ptr, smem_ptr);
    } else {
      legacy_load_2d<
          kUseCpAsync, kNumInt4s, kNumLoadThreads,
          kGmemStride, kSmemStride, kLoadThreadOffset>(gmem_ptr, smem_ptr);
    }
  }

  CUDA_INLINE
  void advance() {
    if constexpr (kUseMxmma) {
      row_offset += kNumGroups / (kMxScaleVec == 1 ? 2 : 4);
      gmem_ptr += kMxGmemStride * kNumGroups / (kMxScaleVec == 1 ? 2 : 4);
    } else if (kIsGroupOrBlock && (kLoadsPerGroup == 1 || counter == 0)) {
      row_offset += kNumGroups;
      gmem_ptr += kGmemStride * kNumGroups;
    }
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t n_block_id, uint32_t k_block_id) {
    row_offset = kProblemNumGroups * expert_id;

    if constexpr (kIsGroupOrBlock) {
      if constexpr (kUseMxmma) {
        row_offset += k_block_id * (kNumGroups / (kMxScaleVec == 1 ? 2 : 4));
      } else if constexpr (BlockShape::K >= kGroupSize) {
        row_offset += k_block_id * kNumGroups;
      } else {
        row_offset += (k_block_id * BlockShape::K) / kGroupSize;
      }
    }

    if constexpr (kUseMxmma) {
      col_offset = n_block_id * (BlockShape::N / (kMxScaleVec == 1 ? 2 : 1));
    } else if constexpr (kIsBlock) {
      col_offset = (n_block_id * BlockShape::N) / kGroupSizeN;
    } else {
      col_offset = n_block_id * (BlockShape::N / 16);
    }

    uint32_t gmem_offset;
    if constexpr (kUseMxmma) {
      gmem_offset = row_offset * kMxGmemStride + n_block_id * kMxSmemStride;
    } else {
      gmem_offset = row_offset * kGmemStride + n_block_id * kSmemStride;
    }
    gmem_ptr = gmem_ptr_raw + gmem_offset;
  }
};
