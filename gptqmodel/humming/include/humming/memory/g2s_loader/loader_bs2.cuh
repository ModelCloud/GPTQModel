// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class G2SMemoryLoaderBS2 {
private:
  using ProblemShape = typename Ctx::ProblemShape;
  using BlockShape = typename Ctx::BlockShape;

  static constexpr bool kUseTma = Ctx::kUseTmaBS2;
  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;
  static constexpr uint32_t kNumLoadThreads = Ctx::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = Ctx::kNumThreads - kNumLoadThreads;

  static constexpr uint32_t kSmemStride = BlockShape::N * 16 / 32 / 4;
  static constexpr uint32_t kGmemStride = ProblemShape::N * 16 / 32 / 4;
  static constexpr uint32_t kNumInt4s = kSmemStride;

public:
  Ctx &ctx;
  const CUtensorMap *tensor_map_ptr;
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  uint32_t offset;

  CUDA_INLINE
  G2SMemoryLoaderBS2(Ctx &ctx) : ctx(ctx) {
    const void *ptr = ctx.params.bs2;
    if constexpr (kUseTma) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<const int4 *>(ptr);
    }
  }

  CUDA_INLINE
  void load(int4 *smem_ptr, void *mbar_ptr) {
    if constexpr (kUseTma) load_tma(smem_ptr, mbar_ptr);
    else load_legacy(smem_ptr);
  }

  CUDA_INLINE
  void load_tma(int4 *smem_ptr, void *mbar_ptr) {
    if (ctx.load_thread_id() == 0) tma_load_2d(tensor_map_ptr, smem_ptr, mbar_ptr, 0, offset);
  }

  CUDA_INLINE
  void load_legacy(int4 *smem_ptr) {
    legacy_load_1d<kUseCpAsync, kNumInt4s, kNumLoadThreads, kLoadThreadOffset>(gmem_ptr, smem_ptr);
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t n_block_id) {
    offset = expert_id * (ProblemShape::N / 64) + n_block_id * (BlockShape::N / 64);
    uint32_t gmem_offset = expert_id * kGmemStride + n_block_id * kSmemStride;
    gmem_ptr = gmem_ptr_raw + gmem_offset;
  }
};
