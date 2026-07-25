// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx, class ArithClass>
struct WMMA {
public:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using MmaShape = typename Ctx::MmaShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;
  using CRegistersType = typename MmaOpClass::CRegisters;
  using CRegistersArrayType = CRegistersType[WarpShape::M / MmaShape::M][WarpShape::N / MmaShape::N];

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;
  static constexpr bool kNativeMixed = MmaOpClass::kNativeMixed;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kNumWarpShapeNSplits = WarpShape::N == ElementA::kBits * 2 ? 2 : 1;

  Ctx &ctx;
  ArithClass &arith;
  typename MmaOpClass::ARegisters regs_a[2][WarpShape::M / MmaShape::M][kPartMmaShapeK / MmaShape::K];
  uint32_t regs_qb[2][ElementB::kBits * (16 / ElementA::kBits)];
  typename MmaOpClass::BRegisters regs_b[2][WarpShape::N / MmaShape::N][kPartMmaShapeK / MmaShape::K];
  CRegistersArrayType regs_c[2];

  CUDA_INLINE
  WMMA(Ctx &ctx, ArithClass &arith)
      : ctx(ctx), arith(arith) {
  }

  CUDA_INLINE
  void zero_accum() {
    uint32_t *regs_c_ptr = regs_c_as_ptr(0);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs_c) / 4; i++) {
      regs_c_ptr[i] = 0;
    };
  };

  CUDA_INLINE
  void transform_b(uint32_t buffer_id) {
    if constexpr (std::is_same<ElementA, ElementB>::value) return;

    if constexpr (kNativeMixed) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < WarpShape::N / 16; i++) {
        uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][i * 16 / MmaShape::N]);
        repack_native_mxf8f6f4<ElementB>(regs_qb[buffer_id], regs_b_ptr, i);
      }
      return;
    }

    if constexpr (kUseFusedE8m0Scale) {
      uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id]);
      fused_dequant_for_mxfp4<ElementA, WarpShape::N / 16, false>(regs_qb[buffer_id], regs_b_ptr, arith.bs[buffer_id]);
    } else {
      if constexpr (ElementB::kBits == 1 && kNumWarpShapeNSplits == 2) {
        regs_qb[buffer_id][0] = regs_qb[buffer_id][0] >> (ctx.warp_id() % 2 * 8);
      }

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < WarpShape::N / 16; i++) {
        uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][i * 16 / MmaShape::N]);
        uint4 zp_vals = arith.prepare_zp_for_dequant(buffer_id, i);
        uint32_t *zp_vals_ptr = reinterpret_cast<uint32_t *>(&zp_vals);
        dequant<ElementB, ElementA, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(regs_qb[buffer_id], regs_b_ptr, i, zp_vals_ptr);
        arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
      };
    }
  };

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    uint32_t buffer_id = iter_id % 2;
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kPartMmaShapeK / MmaShape::K; k++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < WarpShape::N / MmaShape::N; j++) {
        PRAGMA_UNROLL
        for (uint32_t m = 0; m < WarpShape::M / MmaShape::M; m++) {
          MmaOpClass::fma(regs_a[buffer_id][m][k], regs_b[buffer_id][j][k], regs_c[0][m][j], regs_c[0][m][j]);
          arith.may_apply_as_and_bs_on_mma_c(regs_c_as_ptr(), m, j, k, iter_id);
        }
      }
    }
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_a_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(regs_a[buffer_id]);
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
  CUDA_INLINE T *regs_b_as_ptr() {
    return reinterpret_cast<T *>(regs_b);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr(uint32_t buffer_id = 0) {
    return reinterpret_cast<T *>(regs_c[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    uint32_t index = 0;
    constexpr bool kIsGroupInputScale = Ctx::kInputScaleGroupSize > 0;
    constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
    constexpr bool kIsBlockWeightScale = Ctx::kIsBlockWeightScale;

    if constexpr (ElementA::kBits < 16 && kIsGroupInputScale) {
      index = 1;
    }

    if constexpr (ElementA::kBits < 16 && !kUseFusedE8m0Scale && (kIsGroupWeightScale || kIsBlockWeightScale)) {
      index = 1;
    }

    return regs_c_as_ptr<T>(index);
  };
};
