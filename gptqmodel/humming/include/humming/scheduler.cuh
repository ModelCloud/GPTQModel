// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>

template <class Ctx>
class Scheduler {
private:
  using ProblemShape = typename Ctx::ProblemShape;
  using BlockShape = typename Ctx::BlockShape;

  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;
  static constexpr bool kUseWarpSpec = Ctx::kUseWarpSpec;
  static constexpr bool kUseStreamK = Ctx::kUseStreamK;
  static constexpr bool kIsDenseGemm = Ctx::kIsDenseGemm;
  static constexpr bool kIsIndexedGemm = Ctx::kIsIndexedGemm;
  static constexpr bool kIsGroupedGemm = Ctx::kIsGroupedGemm;
  static constexpr bool kIsGroupedContiguousGemm = Ctx::kIsGroupedContiguousGemm;
  static constexpr uint32_t kNumExperts = Ctx::kNumExperts;
  static constexpr uint32_t kNumThreads = Ctx::kNumThreads;
  static constexpr uint32_t kNumMathThreads = Ctx::kNumMathThreads;
  static constexpr uint32_t kNumLoadThreads = Ctx::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = kNumThreads - kNumLoadThreads;
  static constexpr uint32_t kMultiCastSizeA = Ctx::kMultiCastSizeA;
  static constexpr uint32_t kMultiCastSizeB = Ctx::kMultiCastSizeB;
  static constexpr uint32_t kMultiCastSize = kMultiCastSizeA * kMultiCastSizeB;

  static constexpr uint32_t kInputScaleGroupSize = Ctx::kInputScaleGroupSize > 0 ? Ctx::kInputScaleGroupSize : 1;
  static constexpr uint32_t kWeightScaleGroupSize = Ctx::kWeightScaleGroupSize > 0 ? Ctx::kWeightScaleGroupSize : 1;
  static constexpr uint32_t kMaxGroupSize = MAX(kInputScaleGroupSize, kWeightScaleGroupSize);

  static constexpr bool kUseMxmma = Ctx::kUseMxmma;
  static constexpr uint32_t kAsBlocksPerWord = kUseMxmma ? MAX(1u, 4 * kInputScaleGroupSize / BlockShape::K) : 1;

  static constexpr uint32_t N_BLOCKS = ProblemShape::N / BlockShape::N / kMultiCastSizeA;
  static constexpr uint32_t K_BLOCKS = ProblemShape::K / BlockShape::K;

  static constexpr uint32_t kRasterGroupM = Ctx::kRasterGroupM;
  static constexpr uint32_t kNumStages = Ctx::TuningConfig::kNumStages;

  static constexpr int32_t ct_gcd(int32_t a, int32_t b) { return b == 0 ? a : ct_gcd(b, a % b); }

  uint32_t m_blocks;
  uint32_t mn_blocks;
  uint32_t mnk_blocks;

  uint32_t streamk_mnk_total_iters;
  uint32_t streamk_mnk_iters;
  uint32_t streamk_mnk_next_index;
  uint32_t dp_mn_total_iters;
  uint32_t dp_mn_iters;
  uint32_t dp_mn_next_index;

public:
  Ctx &ctx;

  uint32_t m_block_id;
  uint32_t n_block_id;
  uint32_t k_block_id;
  uint32_t old_m_block_id = 0;

  // for stream-k
  uint32_t slice_iters;
  uint32_t slice_count;
  uint32_t slice_id;
  uint32_t locks_offset;

  // for tma multi-cast
  uint32_t cluster_rank = blockIdx.x % kMultiCastSize;

  // for moe gemm (indexed gemm or grouped gemm)
  uint32_t old_expert_id = (1 << 30);
  uint32_t expert_id = 0;

  // for grouped gemm
  uint32_t current_shape_m;
  uint32_t expert_max_num_tokens;
  uint32_t offset_in_expert = 0;
  uint32_t m_offset = 0;

  CUDA_INLINE
  Scheduler(Ctx &ctx) : ctx(ctx) {

    if constexpr (kIsGroupedGemm && Ctx::kUseTmaC) {
      if (threadIdx.x == 0) ctx.smem.tensor_map_buffer[0] = reinterpret_cast<const CUtensorMap *>(ctx.params.c)[0];
      __syncwarp();
    }

    current_shape_m = ctx.params.shape_m;
    expert_max_num_tokens = ctx.params.shape_m / Ctx::kNumExperts;
    calc_m_blocks();
    mn_blocks = m_blocks * N_BLOCKS;
    mnk_blocks = mn_blocks * K_BLOCKS;
    uint32_t kNumCtaGroups = gridDim.x / kMultiCastSize;

    if constexpr (kUseStreamK) {
      uint32_t streamk_mn_blocks = mn_blocks;
      if (mn_blocks > kNumCtaGroups) {
        streamk_mn_blocks = mn_blocks % kNumCtaGroups;
        if (streamk_mn_blocks && streamk_mn_blocks * 10 <= kNumCtaGroups) streamk_mn_blocks += kNumCtaGroups;
      }

      dp_mn_iters = (mn_blocks - streamk_mn_blocks) / kNumCtaGroups;

      uint32_t streamk_mnk_blocks = streamk_mn_blocks * K_BLOCKS;

      streamk_mnk_total_iters = CEIL_DIV(streamk_mnk_blocks, kNumCtaGroups);

      constexpr int32_t blocks_per_group = MAX(kMaxGroupSize / BlockShape::K, kAsBlocksPerWord);
      constexpr int32_t bpg = blocks_per_group > 1 ? blocks_per_group : 1;
      constexpr int32_t align_iters = bpg / ct_gcd(bpg, (int32_t)kNumStages) * (int32_t)kNumStages;
      if constexpr (align_iters > 1) {
        streamk_mnk_total_iters = align_iters * CEIL_DIV(streamk_mnk_total_iters, align_iters);
      };

      streamk_mnk_next_index = kNumCtaGroups * dp_mn_iters * K_BLOCKS + streamk_mnk_total_iters * (blockIdx.x / kMultiCastSize);

      if (streamk_mnk_next_index >= mnk_blocks) {
        streamk_mnk_iters = 0;
      } else {
        streamk_mnk_iters = mnk_blocks - streamk_mnk_next_index;
        if (streamk_mnk_iters > streamk_mnk_total_iters) streamk_mnk_iters = streamk_mnk_total_iters;
      };
    } else {
      dp_mn_iters = mn_blocks / kNumCtaGroups;
      if (blockIdx.x / kMultiCastSize < mn_blocks % kNumCtaGroups) { dp_mn_iters += 1; };
    }

    dp_mn_total_iters = dp_mn_iters;
    slice_count = 1;
    slice_id = 0;

    if (dp_mn_iters) { dp_mn_next_index = blockIdx.x / kMultiCastSize; };
  };

  CUDA_INLINE
  void calc_m_blocks() {
    if constexpr (kIsDenseGemm) {
      m_blocks = CEIL_DIV(ctx.params.shape_m, BlockShape::M * kMultiCastSizeB);
    } else if constexpr (kIsIndexedGemm) {
      uint32_t padded_shape_m = ctx.params.num_tokens_padded_ptr[0];
      m_blocks = CEIL_DIV(padded_shape_m, BlockShape::M * kMultiCastSizeB);
    } else if constexpr (kIsGroupedGemm) {
      auto &expert_layout = ctx.params.expert_layout_ptr;
      if constexpr (kIsGroupedContiguousGemm) {
        if (ctx.params.use_int64_expert_layout)
          legacy_load_2d<kUseCpAsync, kNumExperts + 1, kNumThreads, 2, 1>(expert_layout, ctx.smem.expert_offset);
        else
          legacy_load_2d<kUseCpAsync, kNumExperts + 1, kNumThreads, 1, 1>(expert_layout, ctx.smem.expert_offset);
      } else {
        if (ctx.params.use_int64_expert_layout)
          legacy_load_2d<kUseCpAsync, kNumExperts, kNumThreads, 2, 1>(expert_layout, ctx.smem.expert_tokens);
        else
          legacy_load_2d<kUseCpAsync, kNumExperts, kNumThreads, 1, 1>(expert_layout, ctx.smem.expert_tokens);
      }
      if constexpr (kUseCpAsync) cp_async_commit_group();
      if constexpr (kUseCpAsync) cp_async_wait_group<0>();
      __syncthreads();

      if constexpr (kIsGroupedContiguousGemm) {
        ctx.smem.expert_tokens[kNumExperts - 1] = ctx.params.shape_m - ctx.smem.expert_offset[kNumExperts - 1];
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kNumExperts - 1, kNumThreads); i++) {
          uint32_t index = kNumThreads * i + threadIdx.x;
          if (index < kNumExperts - 1) {
            ctx.smem.expert_tokens[index] = ctx.smem.expert_offset[index + 1] - ctx.smem.expert_offset[index];
          }
        }

        __syncthreads();
      }

      if (ctx.warp_id() == 0) {
        uint32_t tmp_m_blocks = 0;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kNumExperts, 32); i++) {
          uint32_t index = 32 * i + threadIdx.x;
          if (index < kNumExperts) {
            tmp_m_blocks += CEIL_DIV(ctx.smem.expert_tokens[index], BlockShape::M);
          }
        }

        __syncwarp();
        m_blocks = warp_reduce_add(tmp_m_blocks);
        if (threadIdx.x == 0) ctx.smem.total_m_blocks[0] = m_blocks;
      }

      __syncthreads();
      m_blocks = ctx.smem.total_m_blocks[0];
    }
  }

  CUDA_INLINE
  void map_mn_block(uint32_t mn_index, uint32_t &m_id, uint32_t &n_id) {
    if constexpr (kRasterGroupM <= 1 || !kIsDenseGemm) {
      m_id = mn_index / N_BLOCKS;
      n_id = mn_index % N_BLOCKS;
    } else {
      const uint32_t blocks_per_group = kRasterGroupM * N_BLOCKS;
      const uint32_t group_id = mn_index / blocks_per_group;
      const uint32_t first_m = group_id * kRasterGroupM;
      const uint32_t group_m = MIN(m_blocks - first_m, kRasterGroupM);
      const uint32_t idx_in_group = mn_index - group_id * blocks_per_group;
      m_id = first_m + idx_in_group % group_m;
      n_id = idx_in_group / group_m;
    }
  }

  CUDA_INLINE
  bool get_next_block() {
    bool has_next_block = false;
    if (dp_mn_iters) {
      slice_iters = K_BLOCKS;

      map_mn_block(dp_mn_next_index, m_block_id, n_block_id);

      if constexpr (kMultiCastSizeB > 1) {
        m_block_id = m_block_id * kMultiCastSizeB + cluster_rank;
      } else if constexpr (kMultiCastSizeA > 1) {
        n_block_id = n_block_id * kMultiCastSizeA + cluster_rank;
      }
      k_block_id = 0;
      dp_mn_next_index += gridDim.x / kMultiCastSize;
      dp_mn_iters--;
      has_next_block = true;
    } else if constexpr (kUseStreamK) {
      has_next_block = get_streamk_next_block();
    }

    if constexpr (kIsIndexedGemm) {
      if (has_next_block) fetch_moe_index_block();
    }
    if constexpr (kIsGroupedGemm) {
      if (has_next_block) fetch_moe_group_block();
    }

    return has_next_block;
  };

  CUDA_INLINE
  bool get_streamk_next_block() {
    if (!streamk_mnk_iters) return false;
    uint32_t streamk_mn_index = streamk_mnk_next_index / K_BLOCKS;

    map_mn_block(streamk_mn_index, m_block_id, n_block_id);
    if constexpr (kMultiCastSizeB > 1) {
      m_block_id = m_block_id * kMultiCastSizeB + cluster_rank;
    } else if constexpr (kMultiCastSizeA > 1) {
      n_block_id = n_block_id * kMultiCastSizeA + cluster_rank;
    }
    k_block_id = streamk_mnk_next_index - streamk_mn_index * K_BLOCKS;

    slice_iters = K_BLOCKS - k_block_id;
    slice_iters = slice_iters > streamk_mnk_iters ? streamk_mnk_iters : slice_iters;

    streamk_mnk_iters -= slice_iters;
    streamk_mnk_next_index += slice_iters;

    if (k_block_id == 0) {
      slice_id = 0;
      slice_count = CEIL_DIV(K_BLOCKS - slice_iters, streamk_mnk_total_iters) + 1;
    } else {
      slice_id = k_block_id / streamk_mnk_total_iters;
      uint32_t slice_first_block_iters = k_block_id - slice_id * streamk_mnk_total_iters;
      slice_count = CEIL_DIV(K_BLOCKS - slice_first_block_iters, streamk_mnk_total_iters);
      if (slice_first_block_iters) {
        slice_id++;
        slice_count++;
      }
    }

    slice_id = slice_count - 1 - slice_id;

    locks_offset = streamk_mn_index - dp_mn_total_iters * gridDim.x / kMultiCastSize;
    locks_offset = locks_offset * kMultiCastSize + cluster_rank;

    return true;
  };

  CUDA_INLINE
  void fetch_moe_group_block() {
    uint32_t delta_m_block_id = m_block_id - old_m_block_id;
    offset_in_expert += delta_m_block_id * BlockShape::M;

    while (offset_in_expert >= ctx.smem.expert_tokens[expert_id]) {
      offset_in_expert -= CEIL_DIV(ctx.smem.expert_tokens[expert_id], BlockShape::M) * BlockShape::M;
      expert_id++;
    }

    old_m_block_id = m_block_id;
    if constexpr (Ctx::kIsGroupedMaskedGemm) {
      m_offset = expert_id * expert_max_num_tokens + offset_in_expert;
      current_shape_m = expert_id * expert_max_num_tokens + ctx.smem.expert_tokens[expert_id];
    } else if constexpr (Ctx::kIsGroupedContiguousGemm) {
      m_offset = ctx.smem.expert_offset[expert_id] + offset_in_expert;
      current_shape_m = ctx.smem.expert_offset[expert_id] + ctx.smem.expert_tokens[expert_id];
    }

    if (old_expert_id != expert_id) update_tensor_map_c();
    old_expert_id = expert_id;
  };

  CUDA_INLINE
  void fetch_moe_index_block() {
    if (kUseWarpSpec && ctx.is_math_thread()) return;

    expert_id = ctx.params.expert_ids_ptr[m_block_id];

    const uint32_t *gmem_ptr = ctx.params.sorted_ids_ptr + m_block_id * BlockShape::M;
    const int4 *gmem_ptr_load = reinterpret_cast<const int4 *>(gmem_ptr);
    int4 *smem_ptr_load = reinterpret_cast<int4 *>(ctx.smem.wr_row_index);

    legacy_load_1d<kUseCpAsync, BlockShape::M / 4, kNumLoadThreads>(gmem_ptr_load, smem_ptr_load);
    if constexpr (kUseCpAsync) cp_async_commit_group();
    if constexpr (kUseCpAsync) cp_async_wait_group<0>();

    ctx.sync_load_threads();

    uint32_t thread_id = threadIdx.x;
    if constexpr (kUseWarpSpec) thread_id = thread_id - kNumMathThreads;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
      uint32_t index = kNumLoadThreads * i + thread_id;
      if (index < BlockShape::M) {
        uint32_t idx = ctx.smem.wr_row_index[index];
        ctx.smem.rd_row_index[index] = idx / ctx.params.top_k;
      };
    }

    ctx.sync_load_threads();
  };

  CUDA_INLINE
  void update_tensor_map_c() {
    if constexpr (kIsGroupedGemm && Ctx::kUseTmaC) {
      if (threadIdx.x < 32) {
        __syncwarp();
        if (threadIdx.x == 0) {
          tensor_map_replace_global_dim<1>(ctx.smem.tensor_map_buffer, current_shape_m);
          ctx.params.tensor_map_buffer[blockIdx.x] = ctx.smem.tensor_map_buffer[0];
          tensor_map_release_cta();
          tensor_map_acquire_cta(ctx.params.tensor_map_buffer + blockIdx.x);
        }
        __syncwarp();
      }
    }
  }
};
