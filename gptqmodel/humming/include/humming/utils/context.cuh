// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/base.cuh>
#include <humming/utils/storage.cuh>

struct KernelParams {
  uint32_t shape_m;
  uint32_t top_k;
  bool use_int64_expert_layout;
  const void *a;
  const void *b;
  const void *as;
  const void *bs;
  const void *bzp;
  const void *bias;
  const void *c;
  const void *bs2;
  const uint32_t *sorted_ids_ptr;
  const uint32_t *expert_ids_ptr;
  const uint32_t *num_tokens_padded_ptr;
  const uint32_t *expert_layout_ptr;
  CUtensorMap *tensor_map_buffer;
  int32_t *locks;
};

template <
    class MmaOpClass_, class ProblemShape_, class BlockShape_, class WarpShape_, class PadShape_,
    class ElementA_, class ElementB_, class ElementC_, class ElementBS_,
    class LayerConfig_, class ComputeConfig_, class TuningConfig_>
struct KernelContext : LayerConfig_, ComputeConfig_, TuningConfig_ {
  using MmaOpClass = MmaOpClass_;
  using ProblemShape = ProblemShape_;
  using BlockShape = BlockShape_;
  using WarpShape = WarpShape_;
  using PadShape = PadShape_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
  using ElementBS = ElementBS_;
  using LayerConfig = LayerConfig_;
  using ComputeConfig = ComputeConfig_;
  using TuningConfig = TuningConfig_;
  using MmaShape = typename MmaOpClass::MmaShape;

  using SharedStorage = ::SharedStorage<
      MmaOpClass, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;

  static constexpr bool kIsDenseGemm = ComputeConfig::kGemmType == GemmType::DENSE;
  static constexpr bool kIsIndexedGemm = ComputeConfig::kGemmType == GemmType::INDEXED;
  static constexpr bool kIsGroupedContiguousGemm = ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS;
  static constexpr bool kIsGroupedMaskedGemm = ComputeConfig::kGemmType == GemmType::GROUPED_MASKED;
  static constexpr bool kIsGroupedGemm = kIsGroupedContiguousGemm || kIsGroupedMaskedGemm;

  static constexpr bool kUseWmma = LayerConfig::kMmaType == MmaType::MMA;
  static constexpr bool kUseWgmma = LayerConfig::kMmaType == MmaType::WGMMA;
  static constexpr bool kUseMxmma = LayerConfig::kMmaType == MmaType::MXMMA;

  static constexpr bool kUsePackedKLayout = LayerConfig::kUsePackedKLayout;
  static constexpr uint32_t kPackedKFactor = kUsePackedKLayout ? 2 : 1;

  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kWarpIters = kUsePackedKLayout ? (WarpShape::N / 16) : (WarpShape::K / kPartMmaShapeK);

  static constexpr uint32_t kUseWarpSpec = TuningConfig_::kUseWarpSpec;
  static constexpr uint32_t kNumThreads = TuningConfig_::kNumThreads;
  static constexpr uint32_t kNumMathThreads = TuningConfig_::kNumMathThreads;
  static constexpr uint32_t kNumLoadThreads = TuningConfig_::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = kNumThreads - kNumLoadThreads;
  static constexpr uint32_t kMultiCastSize = TuningConfig_::kMultiCastSizeA * TuningConfig_::kMultiCastSizeB;
  static constexpr uint32_t kRasterGroupM = TuningConfig_::kRasterGroupM;

  SharedStorage &smem;
  const KernelParams &params;

  CUDA_INLINE KernelContext(SharedStorage &smem, const KernelParams &params)
      : smem(smem), params(params) {}

  CUDA_INLINE uint32_t warp_id() { return threadIdx.x / 32; }
  CUDA_INLINE uint32_t lane_id() { return threadIdx.x % 32; }
  CUDA_INLINE uint32_t load_thread_id() { return threadIdx.x - kLoadThreadOffset; }
  CUDA_INLINE uint32_t cluster_rank() { return blockIdx.x % kMultiCastSize; }

  CUDA_INLINE uint32_t m_warp_id() { return M_WARPS == 1 ? 0 : (warp_id() / N_WARPS % M_WARPS); }
  CUDA_INLINE uint32_t n_warp_id() { return N_WARPS == 1 ? 0 : (warp_id() % N_WARPS); }
  CUDA_INLINE uint32_t k_warp_id() { return K_WARPS == 1 ? 0 : (warp_id() / (M_WARPS * N_WARPS)); }

  CUDA_INLINE uint32_t m_warp_offset() { return m_warp_id() * WarpShape::M; }
  CUDA_INLINE uint32_t n_warp_offset() { return n_warp_id() * WarpShape::N; }
  CUDA_INLINE uint32_t k_warp_offset() { return k_warp_id() * WarpShape::K; }

  CUDA_INLINE bool is_math_thread() { return threadIdx.x < kNumMathThreads; }
  CUDA_INLINE bool is_load_thread() { return threadIdx.x >= kLoadThreadOffset; }

  CUDA_INLINE static void sync_math_threads() { sync_part_threads<kNumMathThreads, kNumThreads, kUseWarpSpec ? 1 : 0>(); }
  CUDA_INLINE static void sync_load_threads() { sync_part_threads<kNumLoadThreads, kNumThreads, kUseWarpSpec ? 2 : 0>(); }
};
