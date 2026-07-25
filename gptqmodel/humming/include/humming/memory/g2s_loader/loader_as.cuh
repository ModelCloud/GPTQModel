// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class G2SMemoryLoaderAS {
private:
  using ProblemShape = typename Ctx::ProblemShape;
  using BlockShape = typename Ctx::BlockShape;
  using PadShape = typename Ctx::PadShape;
  using ElementA = typename Ctx::ElementA;

  static constexpr bool kUseMxmma = Ctx::kUseMxmma;
  static constexpr bool kUseWarpSpec = Ctx::kUseWarpSpec;
  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;
  static constexpr bool kIsIndexedGemm = Ctx::kIsIndexedGemm;
  static constexpr bool kIsGroupedGemm = Ctx::kIsGroupedGemm;

  static constexpr uint32_t kNumLoadThreads = Ctx::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = Ctx::kNumThreads - kNumLoadThreads;

  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelScale = kHasInputScale && Ctx::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupScale = kHasInputScale && Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kMMajorInputScale = Ctx::kUseMMajorInputScale && kIsGroupScale;
  static_assert(!kMMajorInputScale || !kIsIndexedGemm);
  static constexpr bool kUseTma = Ctx::kUseTmaAS && kHasInputScale && !kIsIndexedGemm;
  static_assert(!kUseTma || kMMajorInputScale || kIsChannelScale || kUseMxmma);
  static constexpr uint32_t kGroupSize = kIsGroupScale ? Ctx::kInputScaleGroupSize : ProblemShape::K;

  static_assert(ProblemShape::K == kGroupSize || (ProblemShape::K - PadShape::K) % kGroupSize == 0);
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kProblemNumGroups = CEIL_DIV(ProblemShape::K - PadShape::K, kGroupSize);
  static constexpr uint32_t kNumGroups = CEIL_DIV(BlockShape::K, kGroupSize);
  static constexpr uint32_t kMxScaleVec = kPartMmaShapeK / kGroupSize;
  static constexpr uint32_t kLoadsPerGroup = kUseMxmma ? MAX(1u, 4 / kNumGroups) : CEIL_DIV(kGroupSize, BlockShape::K);

  using LoadType = typename LoadTypeChooser<kNumGroups * 4>::Type;

public:
  Ctx &ctx;
  const CUtensorMap *tensor_map_ptr;
  const uint32_t *gmem_ptr_raw;
  const uint32_t *gmem_ptr;

  uint32_t shape_m;
  uint32_t total_shape_m;
  uint32_t block_shape_m;
  uint32_t row_offset;
  uint32_t load_row_index;
  uint32_t col_offset = 0;
  uint32_t counter = 0;

  CUDA_INLINE
  G2SMemoryLoaderAS(Ctx &ctx)
      : ctx(ctx), shape_m(ctx.params.shape_m), total_shape_m((ctx.params.shape_m + 3u) & ~3u) {
    const void *ptr = ctx.params.as;
    if constexpr (kUseTma) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<const uint32_t *>(ptr);
    }
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(void *smem_ptr, void *mbar_ptr) {
    counter = kLoadsPerGroup != 1 ? (counter + 1) % kLoadsPerGroup : 0;
    if constexpr (kUseMxmma) {
      if constexpr (kUseTma) load_mx_tma(smem_ptr, mbar_ptr);
      else if constexpr (kMMajorInputScale) load_mx_legacy_m_major(smem_ptr);
      else load_mx_legacy(smem_ptr);
    } else if constexpr (kUseTma) load_tma(smem_ptr, mbar_ptr);
    else if constexpr (kMMajorInputScale) load_legacy_m_major(smem_ptr);
    else load_legacy(smem_ptr);
    if constexpr (kShouldAdvance) advance();
  }

  CUDA_INLINE void load_mx_legacy(void *smem_ptr) {
    uint32_t thread_id = ctx.load_thread_id();
    uint32_t *smem_ptr_load = reinterpret_cast<uint32_t *>(smem_ptr);
    const uint32_t *gmem_ptr_load = reinterpret_cast<const uint32_t *>(gmem_ptr);

    constexpr uint32_t kNumRows = CEIL_DIV(BlockShape::K / kPartMmaShapeK * kMxScaleVec, 4);
    constexpr uint32_t kMxGmemStride = ProblemShape::K / kPartMmaShapeK * kMxScaleVec / 4;
    constexpr uint32_t kNumInts = BlockShape::M * kNumRows;

    if constexpr (kNumInts <= kNumLoadThreads) {
      uint32_t smem_offset = thread_id;
      uint32_t smem_row = smem_offset / BlockShape::M;
      uint32_t smem_col = smem_offset % BlockShape::M;

      uint32_t gmem_row = smem_col;
      uint32_t gmem_col = smem_row;
      uint32_t gmem_offset = gmem_row * kMxGmemStride + gmem_col;
      uint32_t pred = thread_id < kNumInts;

      legacy_load_pred<kUseCpAsync>(gmem_ptr_load + gmem_offset, smem_ptr_load + smem_offset, pred);
    } else {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kNumRows; i++) {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < CEIL_DIV(BlockShape::M, kNumLoadThreads); j++) {
          uint32_t m_index = j * kNumLoadThreads + thread_id;
          uint32_t gmem_offset = m_index * kMxGmemStride + i;
          uint32_t smem_offset = i * BlockShape::M + m_index;
          uint32_t pred = m_index < block_shape_m;

          legacy_load_pred<kUseCpAsync>(gmem_ptr_load + gmem_offset, smem_ptr_load + smem_offset, pred);
        }
      }
    }
  }

  CUDA_INLINE void load_mx_legacy_m_major(void *smem_ptr) {
    uint32_t thread_id = ctx.load_thread_id();
    const int4 *gmem4 = reinterpret_cast<const int4 *>(gmem_ptr);
    int4 *smem4 = reinterpret_cast<int4 *>(smem_ptr);
    constexpr uint32_t kNumRows = CEIL_DIV(BlockShape::K / kPartMmaShapeK * kMxScaleVec, 4);
    uint32_t block_m_aligned = MIN(total_shape_m - row_offset, BlockShape::M);
    PRAGMA_UNROLL
    for (uint32_t r = 0; r < kNumRows; r++) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M / 4, kNumLoadThreads); i++) {
        uint32_t m4 = i * kNumLoadThreads + thread_id;
        uint32_t gmem_offset = (r * total_shape_m) / 4 + m4;
        uint32_t smem_offset = (r * BlockShape::M) / 4 + m4;
        legacy_load_pred<kUseCpAsync>(gmem4 + gmem_offset, smem4 + smem_offset, m4 * 4 < block_m_aligned);
      }
    }
  }

  CUDA_INLINE void load_legacy_m_major(void *smem_ptr) {
    uint32_t thread_id = ctx.load_thread_id();
    const int4 *gmem4 = reinterpret_cast<const int4 *>(gmem_ptr);
    int4 *smem4 = reinterpret_cast<int4 *>(smem_ptr);
    uint32_t block_m_aligned = MIN(total_shape_m - row_offset, BlockShape::M);
    PRAGMA_UNROLL
    for (uint32_t g = 0; g < kNumGroups; g++) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M / 4, kNumLoadThreads); i++) {
        uint32_t m4 = i * kNumLoadThreads + thread_id;
        uint32_t smem_offset = (g * BlockShape::M) / 4 + m4;
        uint32_t gmem_offset = (g * total_shape_m) / 4 + m4;
        legacy_load_pred<kUseCpAsync>(gmem4 + gmem_offset, smem4 + smem_offset, m4 * 4 < block_m_aligned);
      }
    }
  }

  CUDA_INLINE void load_tma(void *smem_ptr, void *mbar_ptr) {
    static_assert(kMMajorInputScale && !kIsIndexedGemm);
    if (ctx.load_thread_id() == 0) tma_load_2d(tensor_map_ptr, smem_ptr, mbar_ptr, row_offset, col_offset);
  }

  CUDA_INLINE void load_mx_tma(void *smem_ptr, void *mbar_ptr) {
    static_assert(kMMajorInputScale && !kIsIndexedGemm);
    if (ctx.load_thread_id() == 0) tma_load_2d(tensor_map_ptr, smem_ptr, mbar_ptr, row_offset, col_offset / 4);
  }

  CUDA_INLINE void load_legacy(void *smem_ptr) {
    uint32_t thread_id = ctx.load_thread_id();
    if constexpr (!kIsIndexedGemm && kIsChannelScale) {
      uint32_t *smem_ptr_load = reinterpret_cast<uint32_t *>(smem_ptr);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
        uint32_t idx = i * kNumLoadThreads + thread_id;
        legacy_load_pred<kUseCpAsync>(gmem_ptr + idx, smem_ptr_load + idx, idx < block_shape_m);
      }
    } else {
      constexpr uint32_t kSmemStride = kNumGroups / (sizeof(LoadType) / 4);
      constexpr uint32_t kGmemStride = kProblemNumGroups / (sizeof(LoadType) / 4);

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < kSmemStride; j++) {
          uint32_t smem_offset = (i * kNumLoadThreads + thread_id) * kSmemStride + j;
          uint32_t smem_row = smem_offset / kSmemStride;
          uint32_t smem_col = smem_offset % kSmemStride;

          uint32_t gmem_row = kIsIndexedGemm ? load_row_index : smem_row;
          uint32_t gmem_offset = gmem_row * kGmemStride + smem_col;

          const LoadType *gmem_ptr_load = reinterpret_cast<const LoadType *>(gmem_ptr);
          LoadType *smem_ptr_load = reinterpret_cast<LoadType *>(smem_ptr);
          bool pred = kIsIndexedGemm ? (gmem_row < shape_m) : (smem_row < block_shape_m);
          legacy_load_pred<kUseCpAsync>(gmem_ptr_load + gmem_offset, smem_ptr_load + smem_offset, pred);
        }
      }
    }
  }

  CUDA_INLINE
  void advance() {
    if (kIsGroupScale && (kLoadsPerGroup == 1 || counter == 0)) {
      col_offset += kUseMxmma ? CEIL_DIV(kNumGroups, 4) * 4 : kNumGroups;
      if constexpr (!kUseTma) {
        if constexpr (kUseMxmma) {
          if constexpr (kMMajorInputScale) gmem_ptr += CEIL_DIV(kNumGroups, 4) * total_shape_m;
          else gmem_ptr += CEIL_DIV(kNumGroups, 4);
        } else if constexpr (kMMajorInputScale) {
          gmem_ptr += kNumGroups * total_shape_m;
        } else {
          gmem_ptr += kNumGroups;
        }
      }
    }
  }

  CUDA_INLINE
  void seek(uint32_t m_block_id, uint32_t k_block_id, uint32_t current_shape_m, uint32_t m_offset) {
    if constexpr (kIsGroupScale) {
      if constexpr (BlockShape::K >= kGroupSize) {
        col_offset = k_block_id * kNumGroups;
      } else {
        col_offset = (k_block_id * BlockShape::K) / kGroupSize;
      }
    } else {
      col_offset = 0;
    }

    if constexpr (kIsGroupedGemm) {
      shape_m = current_shape_m;
      row_offset = m_offset;
    } else {
      row_offset = m_block_id * BlockShape::M;
    }
    block_shape_m = MIN((shape_m - row_offset), BlockShape::M);

    if constexpr (!kIsIndexedGemm) {
      if constexpr (kUseMxmma) {
        if constexpr (!kUseTma) {
          if constexpr (kMMajorInputScale)
            gmem_ptr = gmem_ptr_raw + ((col_offset / 4) * total_shape_m + row_offset);
          else
            gmem_ptr = gmem_ptr_raw + (row_offset * (kProblemNumGroups / 4) + col_offset / 4);
        }
      } else if constexpr (kUseTma) {
        // tma loads via tensor map; gmem_ptr unused
      } else if constexpr (kMMajorInputScale) {
        gmem_ptr = gmem_ptr_raw + (col_offset * total_shape_m + row_offset);
      } else {
        gmem_ptr = gmem_ptr_raw + ((row_offset * kProblemNumGroups) + col_offset);
      }
    } else {
      gmem_ptr = gmem_ptr_raw + col_offset;

      constexpr uint32_t kSmemStride = kNumGroups / (sizeof(LoadType) / 4);
      uint32_t smem_row = ctx.load_thread_id() / kSmemStride;

      if (smem_row < BlockShape::M) {
        load_row_index = ctx.smem.rd_row_index[smem_row];
      } else {
        load_row_index = shape_m;
      }
    }
  }
};
