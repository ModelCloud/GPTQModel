// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>

#include <humming/epilogue/gmem_writer.cuh>
#include <humming/epilogue/smem_reducer.cuh>
#include <humming/epilogue/smem_writer.cuh>


template <class Ctx, class MMA, class ArithClass>
class EpiloguePipeline {
private:
  using SharedStorage = typename Ctx::SharedStorage;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;

  using SmemReducer = EpilogueSmemReducer<Ctx, MMA>;
  using SmemWriter = EpilogueSmemWriter<Ctx, MMA, ArithClass>;
  using GmemWriter = EpilogueGmemWriter<Ctx, ArithClass>;

  static constexpr bool kIsGroupedGemm = Ctx::kIsGroupedGemm;
  static constexpr uint32_t kNumWriteSplits = Ctx::kNumWriteSplits;

public:
  Ctx &ctx;
  SmemReducer smem_reducer;
  SmemWriter smem_writer;
  GmemWriter gmem_writer;
  ArithClass &arith;
  int32_t *locks;

  uint32_t slice_count;
  uint32_t slice_id;
  uint32_t locks_offset;

  CUDA_INLINE
  EpiloguePipeline(Ctx &ctx, ArithClass &arith)
      : ctx(ctx), locks(ctx.params.locks), arith(arith),
        smem_reducer(ctx), smem_writer(ctx, arith), gmem_writer(ctx, arith) {
    if constexpr (Ctx::kUseTmaC) {
      if constexpr (kIsGroupedGemm) gmem_writer.update_tensor_map_ptr(ctx.params.tensor_map_buffer + blockIdx.x);
      else if (threadIdx.x == 0) prefetch_tensor_map(ctx.params.c);
    }
    ctx.sync_math_threads();
  }

  CUDA_INLINE
  void call(uint32_t *regs_c_ptr) {
    ctx.sync_math_threads();
    if constexpr (BlockShape::K > WarpShape::K) smem_reducer.reduce(regs_c_ptr);
    static_assert(kNumWriteSplits == 1 || kNumWriteSplits == 2);
    if constexpr (kNumWriteSplits > 1) {
      static_assert(BlockShape::M == WarpShape::M);
      static_assert(BlockShape::M % 32 == 0);
      static_assert(!Ctx::kUseTmaC);
    }

    if (slice_count > 1) acquire_gmem_barrier();
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kNumWriteSplits; i++) {
      smem_writer.write(regs_c_ptr, slice_count, i);
      if constexpr (Ctx::kUseTmaC) {
        if (ctx.is_math_thread()) tma_fence_async_shared();
      }
      ctx.sync_math_threads();
      gmem_writer.write(slice_id, slice_count, i);
      ctx.sync_math_threads();
    }
    if (slice_count > 1) release_gmem_barrier();
  }

  CUDA_INLINE
  void acquire_gmem_barrier() {
    if (Ctx::kUseTmaC || slice_count > 3) {
      int32_t val = slice_id == 0 ? 0 : -1;
      barrier_acquire2<Ctx::kNumMathThreads, Ctx::kNumThreads>(&locks[locks_offset], val);
    } else {
      barrier_acquire<Ctx::kNumMathThreads, Ctx::kNumThreads>(&locks[locks_offset], slice_id);
    }
  }

  CUDA_INLINE
  void release_gmem_barrier() {
    if (Ctx::kUseTmaC || slice_count > 3) {
      int32_t val = slice_id == 0 ? 1 - static_cast<int32_t>(slice_count) : 0;
      barrier_release2<Ctx::kNumMathThreads, Ctx::kNumThreads>(&locks[locks_offset], val);
    } else {
      barrier_release<Ctx::kNumMathThreads, Ctx::kNumThreads>(&locks[locks_offset], slice_id == slice_count - 1);
    }
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t m_block_id, uint32_t n_block_id, uint32_t current_shape_m, uint32_t m_offset) {
    gmem_writer.seek(m_block_id, n_block_id, current_shape_m, m_offset);
    if constexpr (Ctx::kHasTensorWeightScale) {
      const uint32_t *gs_ptr = reinterpret_cast<const uint32_t *>(Ctx::kIsTensorWeightScale2 ? ctx.params.bs2 : ctx.params.bs);
      arith.gs = gs_ptr[Ctx::kIsDenseGemm ? 0 : expert_id];
    }
  };

  CUDA_INLINE
  void set_streamk_state(uint32_t slice_count_, uint32_t slice_id_, uint32_t locks_offset_) {
    slice_count = slice_count_;
    slice_id = slice_id_;
    locks_offset = locks_offset_;
  };
};
