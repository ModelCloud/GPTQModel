// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/base.cuh>

// Conditional member macros: when the condition is false, the member is completely eliminated.

#if HUMMING_HAS_INPUT_SCALE && HUMMING_INPUT_SCALE_GROUP_SIZE > 0
#define IF_HAS_STAGE_INPUT_SCALE(x) x
#else
#define IF_HAS_STAGE_INPUT_SCALE(x)
#endif

#if HUMMING_WEIGHT_SCALE_GROUP_SIZE > 0
#define IF_HAS_STAGE_WEIGHT_SCALE(x) x
#else
#define IF_HAS_STAGE_WEIGHT_SCALE(x)
#endif

#if HUMMING_HAS_ZERO_POINT && !HUMMING_IS_CHANNEL_WEIGHT_SCALE
#define IF_HAS_STAGE_ZERO_POINT(x) x
#else
#define IF_HAS_STAGE_ZERO_POINT(x)
#endif

#if HUMMING_HAS_ZERO_POINT && HUMMING_IS_CHANNEL_WEIGHT_SCALE
#define IF_HAS_CHANNEL_ZERO_POINT(x) x
#else
#define IF_HAS_CHANNEL_ZERO_POINT(x)
#endif

#if HUMMING_IS_CHANNEL_WEIGHT_SCALE
#define IF_HAS_CHANNEL_WEIGHT_SCALE(x) x
#else
#define IF_HAS_CHANNEL_WEIGHT_SCALE(x)
#endif

#if HUMMING_IS_CHANNEL_WEIGHT_SCALE_2
#define IF_HAS_CHANNEL_WEIGHT_SCALE_2(x) x
#else
#define IF_HAS_CHANNEL_WEIGHT_SCALE_2(x)
#endif

#if HUMMING_HAS_BIAS
#define IF_HAS_BIAS(x) x
#else
#define IF_HAS_BIAS(x)
#endif

#if HUMMING_HAS_INPUT_SCALE && HUMMING_INPUT_SCALE_GROUP_SIZE == 0
#define IF_HAS_CHANNEL_INPUT_SCALE(x) x
#else
#define IF_HAS_CHANNEL_INPUT_SCALE(x)
#endif

#if HUMMING_IS_INDEXED_GEMM
#define IF_IS_INDEXED_GEMM(x) x
#else
#define IF_IS_INDEXED_GEMM(x)
#endif

#if HUMMING_IS_GROUPED_GEMM
#define IF_IS_GROUPED_GEMM(x) x
#else
#define IF_IS_GROUPED_GEMM(x)
#endif

#if HUMMING_IS_GROUPED_CONTIGUOUS_GEMM
#define IF_IS_GROUPED_CONTIGUOUS_GEMM(x) x
#else
#define IF_IS_GROUPED_CONTIGUOUS_GEMM(x)
#endif

#if HUMMING_USE_MBARRIER
#define IF_USE_MBARRIER(x) x
#else
#define IF_USE_MBARRIER(x)
#endif

#if HUMMING_USE_WARP_SPEC
#define IF_USE_WARP_SPEC(x) x
#else
#define IF_USE_WARP_SPEC(x)
#endif

#if HUMMING_REDUCE_OVERLAP_LAST_STAGE_ONLY
#define IF_REDUCE_LAST_STAGE_ONLY(x) x
#else
#define IF_REDUCE_LAST_STAGE_ONLY(x)
#endif


template <
    class MmaOpClass,
    class BlockShape, class WarpShape,
    class ElementA, class ElementB, class ElementBS,
    class LayerConfig, class ComputeConfig, class TuningConfig>
struct SharedStorage {
private:
  static constexpr bool kUseMxmma = MmaOpClass::kMmaType == MmaType::MXMMA;
  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelInputScale = kHasInputScale && LayerConfig::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupInputScale = kHasInputScale && LayerConfig::kInputScaleGroupSize > 0;
  static constexpr bool kIsChannelWeightScale = LayerConfig::kIsChannelWeightScale;
  static constexpr bool kIsChannelWeightScale2 = LayerConfig::kIsChannelWeightScale2;
  static constexpr bool kIsGroupWeightScale = LayerConfig::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = LayerConfig::kIsBlockWeightScale;
  static constexpr bool kIsGroupOrBlockWeightScale = kIsGroupWeightScale || kIsBlockWeightScale;
  static constexpr bool kHasZeroPoint = LayerConfig::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = LayerConfig::kIsFpZeroPoint;

public:
  static constexpr uint32_t kNumExperts = LayerConfig::kNumExperts;
  static constexpr uint32_t kNumStages = TuningConfig::kNumStages;
  static constexpr uint32_t kNumWriteSplits = TuningConfig::kNumWriteSplits;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kNumWarpsDimK = BlockShape::K / WarpShape::K;
  static constexpr uint32_t kMmaCTypeBits = MmaOpClass::kCTypeBits;
  static constexpr uint32_t M_WARPS = (BlockShape::M / WarpShape::M);
  static constexpr uint32_t kWarpReduceSize = M_WARPS * 16 * BlockShape::N * kMmaCTypeBits / 128 * (kNumWarpsDimK / 2);
  static constexpr uint32_t kBlockOutputSize = BlockShape::M * BlockShape::N / 2 / 4 / kNumWriteSplits;
  static constexpr uint32_t kNumZPBits = kIsFpZeroPoint ? 16 : MAX(4, static_next_power_of_2(ElementB::kBits));

  static constexpr uint32_t kSmemStrideA = BlockShape::K * ElementA::kBits / 32 / 4;
  static constexpr uint32_t kSmemStrideB = BlockShape::N * kPartMmaShapeK * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kSmemStrideBS = BlockShape::N * ElementBS::kBits / 32 / 4;
  static constexpr uint32_t kSmemStrideBZP = BlockShape::N * kNumZPBits / 32 / 4;
  static constexpr uint32_t kSmemStrideBias = BlockShape::N * 16 / 32 / 4;

  static constexpr uint32_t kGroupSizeA = LayerConfig::kInputScaleGroupSize;
  static constexpr uint32_t kGroupSizeB = LayerConfig::kWeightScaleGroupSize;
  static constexpr uint32_t kNumGroupsA = kIsGroupInputScale ? CEIL_DIV(BlockShape::K, kGroupSizeA) : 0;
  static constexpr uint32_t kNumGroupsB = kIsGroupOrBlockWeightScale ? CEIL_DIV(BlockShape::K, kGroupSizeB) : 0;

  static constexpr uint32_t kStageSizeA = BlockShape::M * kSmemStrideA;
  static constexpr uint32_t kStageSizeB = BlockShape::K / kPartMmaShapeK * kSmemStrideB;
  static constexpr uint32_t kNumGroupsAStorage = CEIL_DIV(kNumGroupsA, 4) * 4;
  static constexpr uint32_t kStageSizeAS = kUseMxmma
                                               ? CEIL_DIV(kNumGroupsAStorage * BlockShape::M * ElementBS::kBits / 8, sizeof(int4))
                                               : kNumGroupsA * BlockShape::M / 4;
  static constexpr uint32_t kStageSizeBS = kNumGroupsB * kSmemStrideBS;
  static constexpr uint32_t kStageSizeBZP = kNumGroupsB * kSmemStrideBZP;

  static constexpr uint32_t kChannelSizeAS = kIsChannelInputScale ? BlockShape::M / 4 : 0;
  static constexpr uint32_t kChannelSizeBS = kIsChannelWeightScale ? kSmemStrideBS : 0;
  static constexpr uint32_t kChannelSizeBS2 = kIsChannelWeightScale2 ? kSmemStrideBias : 0;
  static constexpr uint32_t kChannelSizeBZP = (kIsChannelWeightScale && kHasZeroPoint) ? kSmemStrideBZP : 0;
  static constexpr uint32_t kBiasSize = LayerConfig::kHasBias ? kSmemStrideBias : 0;

  static constexpr uint32_t kStageBytesA = kStageSizeA * sizeof(int4);
  static constexpr uint32_t kStageBytesB = kStageSizeB * sizeof(int4);
  static constexpr uint32_t kStageBytesAS = kStageSizeAS * sizeof(int4);
  static constexpr uint32_t kStageBytesBS = kStageSizeBS * sizeof(int4);
  static constexpr uint32_t kStageBytesBZP = kStageSizeBZP * sizeof(int4);
  static constexpr uint32_t kChannelBytesAS = kChannelSizeAS * sizeof(int4);
  static constexpr uint32_t kChannelBytesBS = kChannelSizeBS * sizeof(int4);
  static constexpr uint32_t kChannelBytesBS2 = kChannelSizeBS2 * sizeof(int4);
  static constexpr uint32_t kBiasBytes = kBiasSize * sizeof(int4);

  static constexpr bool kUseWarpSpec = TuningConfig::kUseWarpSpec;
  static constexpr bool kUseMBarrier = TuningConfig::kUseMBarrier;
  static constexpr bool kIsIndexedGemm = ComputeConfig::kGemmType == GemmType::INDEXED;
  static constexpr bool kIsGroupedGemm = ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS || ComputeConfig::kGemmType == GemmType::GROUPED_MASKED;

  struct StageStorage {
    alignas(1024) int4 a[kStageSizeA];
    alignas(128) int4 b[kStageSizeB];
    IF_HAS_STAGE_INPUT_SCALE(alignas(128) int4 as[kStageSizeAS];)
    IF_HAS_STAGE_WEIGHT_SCALE(alignas(128) int4 bs[kStageSizeBS];)
    IF_HAS_STAGE_ZERO_POINT(alignas(128) int4 bzp[kStageSizeBZP];)
  };

  union alignas(1024) {
    struct {
      IF_HAS_CHANNEL_ZERO_POINT(alignas(128) int4 bzp_c[kChannelSizeBZP];)
      StageStorage stages[kNumStages];
      IF_HAS_CHANNEL_WEIGHT_SCALE(alignas(128) int4 bs_c[kChannelSizeBS];)
      IF_HAS_CHANNEL_WEIGHT_SCALE_2(alignas(128) int4 bs2_c[kChannelSizeBS2];)
      IF_HAS_BIAS(alignas(128) int4 bias[kBiasSize];)
      IF_HAS_CHANNEL_INPUT_SCALE(alignas(128) int4 as_c[kChannelSizeAS];)
    };
    struct {
      IF_REDUCE_LAST_STAGE_ONLY(IF_HAS_CHANNEL_ZERO_POINT(alignas(128) int4 reduce_skip_bzp_c[kChannelSizeBZP];))
      IF_REDUCE_LAST_STAGE_ONLY(StageStorage reduce_skip[kNumStages - 1];)
      alignas(128) int4 reduce[MAX(kWarpReduceSize, kBlockOutputSize)];
    };
  };

  IF_IS_INDEXED_GEMM(uint32_t rd_row_index[BlockShape::M];)
  IF_IS_INDEXED_GEMM(uint32_t wr_row_index[BlockShape::M];)

  IF_IS_GROUPED_GEMM(CUtensorMap tensor_map_buffer[1];)
  IF_IS_GROUPED_GEMM(uint32_t expert_tokens[kNumExperts];)
  IF_IS_GROUPED_GEMM(uint32_t total_m_blocks[1];)
  IF_IS_GROUPED_CONTIGUOUS_GEMM(uint32_t expert_offset[kNumExperts + 1];)

  IF_USE_MBARRIER(alignas(128) uint64_t load_mbar[kNumStages + 2];)
  IF_USE_WARP_SPEC(uint64_t math_mbar[kNumStages + 1];)
};
