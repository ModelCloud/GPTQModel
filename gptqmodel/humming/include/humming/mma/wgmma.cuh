// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <uint32_t swizzle_bytes = 128>
CUDA_INLINE uint64_t make_wgmma_smem_desc(uint32_t addr) {
  static_assert(swizzle_bytes == 128 || swizzle_bytes == 64);

  constexpr uint64_t swizzle_type = swizzle_bytes == 128 ? 1 : 2;
  constexpr uint64_t stride = (swizzle_bytes * 8) >> 4;
  constexpr uint64_t desc_base = (swizzle_type << 62) | (stride << 32);

  uint64_t desc = desc_base;
  reinterpret_cast<uint32_t *>(&desc)[0] = (addr >> 4);

  return desc;
};


template <class Ctx, class ArithClass>
struct WGMMA {
public:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using MmaShape = typename Ctx::MmaShape;
  using SharedStorage = typename Ctx::SharedStorage;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;
  using CRegistersType = typename MmaOpClass::CRegisters;
  using CRegistersArrayType = CRegistersType[WarpShape::N * 4 / MmaShape::N][WarpShape::M / MmaShape::M];

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kSwizzleBytes = ElementA::kBits * BlockShape::K >= 1024 ? 128 : 64;
  static constexpr uint32_t kNumWarpShapeNSplits = WarpShape::N == ElementA::kBits * 2 ? 2 : 1;

  static constexpr bool kUsePackedKLayout = Ctx::kUsePackedKLayout;
  static constexpr uint32_t kPackedKFactor = Ctx::kPackedKFactor;
  static constexpr uint32_t kWarpIters = Ctx::kWarpIters;
  static constexpr uint32_t kNumKSlabs = WarpShape::K / kPartMmaShapeK;
  static constexpr uint32_t kRegsBKDim = kUsePackedKLayout ? kNumKSlabs : (kPartMmaShapeK / MmaShape::K);

  Ctx &ctx;
  ArithClass &arith;
  uint32_t regs_qb[2][ElementB::kBits * (16 / ElementA::kBits)];
  typename MmaOpClass::BRegisters regs_b[2][kUsePackedKLayout ? 1 : (WarpShape::N * 4 / MmaShape::N / kPackedKFactor)][kRegsBKDim];
  CRegistersArrayType regs_c[2];
  uint32_t smem_offset = 0;

  CUDA_INLINE
  WGMMA(Ctx &ctx, ArithClass &arith)
      : ctx(ctx), arith(arith) {
    constexpr uint32_t kSwizzleSizeK = kSwizzleBytes * 8 / ElementA::kBits;
    static_assert(kSwizzleSizeK >= WarpShape::K);

    const uint32_t row_offset = ctx.m_warp_offset();
    const uint32_t col_offset = ctx.k_warp_offset();

    smem_offset = row_offset * (kSwizzleBytes / 16);
    smem_offset += (col_offset % kSwizzleSizeK) * ElementA::kBits / 128;
    smem_offset += (col_offset / kSwizzleSizeK) * (BlockShape::M * kSwizzleBytes / 16);
    smem_offset = smem_offset * sizeof(int4);
  }

  CUDA_INLINE
  void zero_accum() {
    uint32_t *regs_c_ptr = regs_c_as_ptr();
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs_c) / 4; i++) {
      regs_c_ptr[i] = 0;
    };
  };

  CUDA_INLINE
  void transform_b(uint32_t buffer_id) {
    if constexpr (std::is_same<ElementA, ElementB>::value) return;

    if constexpr (kUseFusedE8m0Scale) {
      uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id]);
      fused_dequant_for_mxfp4<ElementA, WarpShape::N / 16, true>(regs_qb[buffer_id], regs_b_ptr, arith.bs[buffer_id]);
    } else {
      if constexpr (ElementB::kBits == 1 && kNumWarpShapeNSplits == 2) {
        regs_qb[buffer_id][0] = regs_qb[buffer_id][0] >> (ctx.warp_id() % 2 * 8);
      }

      constexpr uint32_t kTransformIters = kUsePackedKLayout ? kNumKSlabs : (WarpShape::N / (MmaShape::N / 4));
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kTransformIters; i++) {
        uint32_t *regs_b_ptr;
        if constexpr (kUsePackedKLayout) {
          regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][0][i]);
        } else {
          regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][i / kPackedKFactor][i % kPackedKFactor]);
        }
        uint4 zp_vals = arith.prepare_zp_for_dequant(buffer_id, i);
        uint32_t *zp_vals_ptr = reinterpret_cast<uint32_t *>(&zp_vals);
        dequant<ElementB, ElementA, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(regs_qb[buffer_id], regs_b_ptr, i, zp_vals_ptr);
        arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
      };
    }
  };

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    static_assert(WarpShape::M == MmaShape::M);
    static_assert(kPartMmaShapeK == MmaShape::K);
    uint32_t buffer_id = iter_id % 2;

    const uint32_t smem_base = cast_smem_ptr_to_uint(&ctx.smem);
    constexpr uint32_t kItersPerHalf = kWarpIters / kPackedKFactor;
    constexpr uint32_t kNumIters = kUsePackedKLayout ? 1 : (WarpShape::N / (MmaShape::N / 4) / kPackedKFactor);
    constexpr uint32_t kRunKLoop = kUsePackedKLayout ? kNumKSlabs : kPackedKFactor;

    uint32_t delta_m = kUsePackedKLayout ? iter_id : 0;
    uint32_t delta_j = final_regs_c_index() == 0 ? delta_m : 0;

    wgmma_fence();
    may_fence_regs(delta_j);

    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kRunKLoop; k++) {
      uint32_t k_slab = kUsePackedKLayout ? k : ((iter_id % kItersPerHalf) * kPackedKFactor + k);
      uint32_t smem_addr = smem_base + offsetof(SharedStorage, stages) + stage_id * sizeof(typename SharedStorage::StageStorage);
      smem_addr += k_slab * 2 * sizeof(int4) + smem_offset;
      uint64_t desc = make_wgmma_smem_desc<kSwizzleBytes>(smem_addr);

      bool scale_d = true;
      if constexpr (ElementA::kBits != 16 && Ctx::kInputScaleGroupSize > 0) {
        scale_d = (k_slab * kPartMmaShapeK) % Ctx::kInputScaleGroupSize > 0;
      }
      if constexpr (!kUseFusedE8m0Scale && ElementA::kBits != 16 && Ctx::kWeightScaleGroupSize > 0) {
        scale_d = scale_d && (k_slab * kPartMmaShapeK) % Ctx::kWeightScaleGroupSize > 0;
      }

      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++) {
        MmaOpClass::fma(desc, regs_b[buffer_id][j][k], regs_c[0][delta_j + j][0], scale_d);
      }
    }

    wgmma_commit();
    wgmma_wait<0>();
    may_fence_regs(delta_j);

    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kRunKLoop; k++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++) {
        arith.may_apply_as_and_bs_on_wgmma_c(regs_c_as_ptr(), j, k, iter_id, delta_m);
      }
    }
  };

  CUDA_INLINE void may_fence_regs(uint32_t delta_j) {
    if constexpr (final_regs_c_index() != 0) {
      constexpr uint32_t kNumIters = kUsePackedKLayout ? 1 : (WarpShape::N / (MmaShape::N / 4) / kPackedKFactor);
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++)
        fence_regs(regs_c[0][delta_j + j][0]);
    }
  }

  template <class T>
  CUDA_INLINE void fence_regs(T &regs) {
    PRAGMA_UNROLL
    for (uint32_t r = 0; r < sizeof(T) / 4; r++) {
      warpgroup_fence_operand(reinterpret_cast<uint32_t *>(regs)[r]);
    }
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_qb_as_ptr(uint32_t buffer_id) {
    if constexpr (std::is_same<ElementA, ElementB>::value) {
      return reinterpret_cast<T *>(regs_b[buffer_id]);
    } else {
      return reinterpret_cast<T *>(regs_qb[buffer_id]);
    };
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr(uint32_t buffer_id = 0) {
    return reinterpret_cast<T *>(regs_c[buffer_id]);
  };

  static constexpr uint32_t final_regs_c_index() {
    if constexpr (ElementA::kBits < 16 && Ctx::kInputScaleGroupSize > 0) return 1;
    if constexpr (ElementA::kBits < 16 && !kUseFusedE8m0Scale && (Ctx::kIsGroupWeightScale || Ctx::kIsBlockWeightScale)) return 1;
    return 0;
  };

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    return regs_c_as_ptr<T>(final_regs_c_index());
  };
};
