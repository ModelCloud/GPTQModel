// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class G2SMemoryLoaderA {
private:
  using SharedStorage = typename Ctx::SharedStorage;
  using ProblemShape = typename Ctx::ProblemShape;
  using BlockShape = typename Ctx::BlockShape;
  using PadShape = typename Ctx::PadShape;
  using ElementA = typename Ctx::ElementA;

  static constexpr bool kUseWarpSpec = Ctx::kUseWarpSpec;
  static constexpr bool kUseTma = Ctx::kUseTmaA;
  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;
  static constexpr bool kIsIndexedGemm = Ctx::kIsIndexedGemm;

  static constexpr uint32_t kNumLoadThreads = Ctx::kNumLoadThreads;
  static constexpr uint32_t kMultiCastSizeA = Ctx::kMultiCastSizeA;

  static constexpr uint32_t kSmemStride = BlockShape::K * ElementA::kBits / 32 / 4;
  static constexpr uint32_t kGmemStride = (ProblemShape::K - PadShape::K) * ElementA::kBits / 32 / 4;
  static constexpr uint32_t kNumInt4s = kSmemStride * BlockShape::M;
  static constexpr uint32_t kColOffsetToElem = MAX(ElementA::kBits, 8) / ElementA::kBits;

  static_assert(BlockShape::K * ElementA::kBits >= 512);
  static constexpr uint32_t kSwizzleBytes = BlockShape::K * ElementA::kBits == 512 ? 64 : 128;
  static constexpr uint32_t kNumTmaLoadsPerLine = CEIL_DIV(BlockShape::K * ElementA::kBits, kSwizzleBytes * 8);
  static constexpr uint32_t kLoadIters = CEIL_DIV(kNumInt4s, kNumLoadThreads);

public:
  Ctx &ctx;
  const CUtensorMap *tensor_map_ptr;
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  uint32_t shape_m;
  uint32_t block_shape_m;
  uint32_t load_row_index[kLoadIters];
  uint32_t row_offset;
  uint32_t col_offset;

  CUDA_INLINE
  G2SMemoryLoaderA(Ctx &ctx)
      : ctx(ctx), shape_m(ctx.params.shape_m) {
    const void *ptr = ctx.params.a;
    if constexpr (kUseTma) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<const int4 *>(ptr);
    }
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(int4 *smem_ptr, void *mbar_ptr, uint32_t stage_id = 0) {
    if constexpr (kUseTma) load_tma(smem_ptr, mbar_ptr);
    else load_legacy(smem_ptr, stage_id);
    if constexpr (kShouldAdvance) advance();
  }

  CUDA_INLINE
  void load_tma(int4 *smem_ptr, void *mbar_ptr) {
    uint32_t thread_id = ctx.load_thread_id();
    if (thread_id < kNumTmaLoadsPerLine) {
      const uint32_t block_idx = thread_id;
      const uint32_t smem_offset = BlockShape::M * 8 * block_idx;
      const uint32_t col_offset2 = col_offset + (1024 / ElementA::kBits) * block_idx;
      if constexpr (kMultiCastSizeA == 1) {
        tma_load_2d(tensor_map_ptr, smem_ptr + smem_offset, mbar_ptr, col_offset2, row_offset);
      } else if (blockIdx.x % kMultiCastSizeA == 0) {
        tma_load_2d<kMultiCastSizeA>(tensor_map_ptr, smem_ptr + smem_offset, mbar_ptr, col_offset2, row_offset);
      }
    }
  }

  CUDA_INLINE
  void load_legacy(int4 *smem_ptr, uint32_t stage_id) {
    if constexpr (kSwizzleBytes == 128) {
      load_legacy_swizzled_128B(smem_ptr, stage_id);
    } else if constexpr (kSwizzleBytes == 64) {
      load_legacy_swizzled_64B(smem_ptr, stage_id);
    }
  }

  CUDA_INLINE
  void load_legacy_swizzled_128B(int4 *smem_ptr, uint32_t stage_id) {
    static_assert(BlockShape::K * ElementA::kBits >= 1024);
    uint32_t thread_id = ctx.load_thread_id();
    uint32_t smem_uint = offsetof(SharedStorage, stages) + stage_id * sizeof(typename SharedStorage::StageStorage);
    uint32_t smem_base = smem_uint / 128 % 8;
    uint32_t smem_swizzled_col = (thread_id % 8) ^ (((thread_id % 64) / 8 + smem_base)) % 8;

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kLoadIters; i++) {
      uint32_t smem_offset = i * kNumLoadThreads + thread_id;
      uint32_t smem_row = smem_offset / 8;
      uint32_t smem_col = smem_offset % 8;
      uint32_t smem_swizzled_offset = smem_row * 8 + smem_swizzled_col;

      uint32_t gmem_col = smem_row / BlockShape::M * 8 + smem_col;
      uint32_t gmem_row = kIsIndexedGemm ? load_row_index[i] : (smem_row % BlockShape::M);
      uint32_t gmem_offset = gmem_row * kGmemStride + gmem_col;

      bool pred0 = (gmem_col * (128 / ElementA::kBits) + col_offset * kColOffsetToElem) < (ProblemShape::K - PadShape::K);
      bool pred1 = kNumInt4s % kNumLoadThreads == 0 || i != kLoadIters - 1 || smem_offset < kNumInt4s;
      bool pred2 = gmem_row < (kIsIndexedGemm ? shape_m : block_shape_m);

      if constexpr (PadShape::K == 0) {
        legacy_load_pred<kUseCpAsync>(gmem_ptr + gmem_offset, smem_ptr + smem_swizzled_offset, pred1 && pred2);
      } else {
        legacy_load_zfill_pred<kUseCpAsync>(gmem_ptr + gmem_offset, smem_ptr + smem_swizzled_offset, pred0, pred1 && pred2);
      }
    }
  }

  CUDA_INLINE
  void load_legacy_swizzled_64B(int4 *smem_ptr, uint32_t stage_id) {
    static_assert(BlockShape::K * ElementA::kBits == 512);
    uint32_t thread_id = ctx.load_thread_id();
    uint32_t smem_uint = offsetof(SharedStorage, stages) + stage_id * sizeof(typename SharedStorage::StageStorage);
    uint32_t smem_base = smem_uint / 128 % 4;
    uint32_t smem_swizzled_col = (thread_id % 8) ^ (((thread_id % 32) / 8 + smem_base) % 4);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kLoadIters; i++) {
      uint32_t smem_offset = i * kNumLoadThreads + thread_id;
      uint32_t smem_row = smem_offset / 8;
      uint32_t smem_col = smem_offset % 8;
      uint32_t smem_swizzled_offset = smem_row * 8 + smem_swizzled_col;

      uint32_t gmem_row = smem_row % (BlockShape::M / 2) * 2 + smem_col / 4;
      gmem_row = kIsIndexedGemm ? load_row_index[i] : gmem_row;
      uint32_t gmem_col = smem_col % 4;
      uint32_t gmem_offset = gmem_row * kGmemStride + gmem_col;

      bool pred0 = (gmem_col * (128 / ElementA::kBits) + col_offset * kColOffsetToElem) < (ProblemShape::K - PadShape::K);
      bool pred1 = kNumInt4s % kNumLoadThreads == 0 || i != kLoadIters - 1 || smem_offset < kNumInt4s;
      bool pred2 = gmem_row < (kIsIndexedGemm ? shape_m : block_shape_m);
      if constexpr (PadShape::K == 0) {
        legacy_load_pred<kUseCpAsync>(gmem_ptr + gmem_offset, smem_ptr + smem_swizzled_offset, pred1 && pred2);
      } else {
        legacy_load_zfill_pred<kUseCpAsync>(gmem_ptr + gmem_offset, smem_ptr + smem_swizzled_offset, pred0, pred1 && pred2);
      }
    }
  }

  CUDA_INLINE
  void advance() {
    col_offset += BlockShape::K * ElementA::kBits / MAX(ElementA::kBits, 8);
    gmem_ptr += kSmemStride;
  }

  CUDA_INLINE
  void seek(uint32_t m_block_id, uint32_t k_block_id, uint32_t current_shape_m, uint32_t m_offset) {
    if constexpr (Ctx::kIsGroupedGemm) {
      shape_m = current_shape_m;
      row_offset = m_offset;
    } else {
      row_offset = m_block_id * BlockShape::M;
    }
    col_offset = k_block_id * (BlockShape::K * ElementA::kBits / MAX(ElementA::kBits, 8));
    block_shape_m = MIN(shape_m - row_offset, BlockShape::M);

    uint32_t gmem_offset = k_block_id * kSmemStride;
    gmem_offset += kIsIndexedGemm ? 0 : (row_offset * kGmemStride);
    gmem_ptr = gmem_ptr_raw + gmem_offset;

    if constexpr (kIsIndexedGemm) {
      static_assert(!kUseTma);
      uint32_t thread_id = ctx.load_thread_id();

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kLoadIters; i++) {
        uint32_t smem_offset = i * kNumLoadThreads + thread_id;
        uint32_t smem_row = smem_offset / 8;
        uint32_t smem_col = smem_offset % 8;
        uint32_t gmem_row;

        if constexpr (BlockShape::K * ElementA::kBits >= 1024) {
          gmem_row = smem_row % BlockShape::M;
        } else {
          gmem_row = smem_row % (BlockShape::M / 2) * 2 + smem_col / 4;
        }
        load_row_index[i] = ctx.smem.rd_row_index[gmem_row];
      }
    }
  }
};
