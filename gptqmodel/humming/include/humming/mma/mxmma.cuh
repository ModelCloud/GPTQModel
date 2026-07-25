// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <class Ctx, class ArithClass>
struct MXMMA {
public:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using MmaShape = typename MmaOpClass::MmaShape;
  using WarpShape = typename Ctx::WarpShape;
  using BlockShape = typename Ctx::BlockShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;
  using CRegistersType = typename MmaOpClass::CRegisters;
  using CRegistersArrayType = CRegistersType[WarpShape::M / MmaShape::M][WarpShape::N / MmaShape::N];

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kNumWarpShapeNSplits = WarpShape::N == ElementA::kBits * 2 ? 2 : 1;
  static constexpr uint32_t kScaleVec = MmaOpClass::kScaleVec;
  static constexpr bool kNativeMixed = MmaOpClass::kNativeMixed;

  static constexpr uint32_t kAsGroup = Ctx::kInputScaleGroupSize;
  static constexpr uint32_t kAsBlockGroups = kAsGroup > 0 ? BlockShape::K / kAsGroup : 1;
  static constexpr uint32_t kAsBlocksPerWord = kAsBlockGroups >= 4 ? 1 : 4 / kAsBlockGroups;
  static constexpr uint32_t kWarpKIters = WarpShape::K / kPartMmaShapeK;
  static constexpr uint32_t M_WARPS = Ctx::M_WARPS;
  static constexpr uint32_t N_WARPS = Ctx::N_WARPS;

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;

  Ctx &ctx;
  ArithClass &arith;
  typename MmaOpClass::ARegisters regs_a[2][WarpShape::M / MmaShape::M][kPartMmaShapeK / MmaShape::K];
  uint32_t regs_qb[2][ElementB::kBits * (16 / ElementA::kBits)];
  typename MmaOpClass::BRegisters regs_b[2][WarpShape::N / MmaShape::N][kPartMmaShapeK / MmaShape::K];
  CRegistersArrayType regs_c;

  uint32_t regs_sfa[2][WarpShape::M / MmaShape::M][kPartMmaShapeK / MmaShape::K];
  uint32_t regs_sfb[2][WarpShape::N / MmaShape::N][kPartMmaShapeK / MmaShape::K];
  uint32_t as_byte_phase = 0;
  uint32_t k_block_idx = 0;
  uint32_t k_warp_byte_base = 0;

  CUDA_INLINE
  MXMMA(Ctx &ctx, ArithClass &arith)
      : ctx(ctx), arith(arith) {
    if constexpr (kAsGroup > 0) {
      uint32_t k_warp_id = (threadIdx.x / 32) / (M_WARPS * N_WARPS);
      k_warp_byte_base = (k_warp_id * kWarpKIters * kScaleVec) % 4;
    }
  }

  CUDA_INLINE
  void zero_accum() {
    if constexpr (kAsBlocksPerWord > 1) k_block_idx = 0;
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
    } else {
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
  uint32_t get_thread_id_a(uint32_t stage_id, uint32_t iter_id, uint32_t m_id) {
    return m_id % 2;
  }

  CUDA_INLINE
  uint32_t get_thread_id_b(uint32_t stage_id, uint32_t iter_id, uint32_t n_id) {
    if constexpr (kScaleVec == 1) {
      return n_id / 2;
    } else {
      return n_id;
    }
  }

  CUDA_INLINE
  uint32_t get_byte_id_a(uint32_t stage_id, uint32_t iter_id, uint32_t m_id) {
    return (as_byte_phase + k_warp_byte_base + iter_id * kScaleVec) % 4;
  }

  CUDA_INLINE
  uint32_t get_byte_id_b(uint32_t stage_id, uint32_t iter_id, uint32_t n_id) {
    if constexpr (kScaleVec == 4) {
      return 0;
    } else if constexpr (kScaleVec == 2) {
      return (iter_id % 2) * 2;
    } else if constexpr (kScaleVec == 1) {
      return (n_id % 2) * 2 + (iter_id % 2);
    }

    return 0;
  }

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    uint32_t buffer_id = iter_id % 2;
    if constexpr (kAsBlocksPerWord > 1) {
      as_byte_phase = (k_block_idx % kAsBlocksPerWord) * kAsBlockGroups;
    }
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kPartMmaShapeK / MmaShape::K; k++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < WarpShape::N / MmaShape::N; j++) {
        PRAGMA_UNROLL
        for (uint32_t m = 0; m < WarpShape::M / MmaShape::M; m++) {
          uint32_t reg_m = m / 2;
          uint32_t reg_n = kScaleVec == 1 ? 0 : j / 4;
          MmaOpClass::fma(
              regs_a[buffer_id][m][k], regs_b[buffer_id][j][k],
              regs_sfa[buffer_id][reg_m][k], regs_sfb[buffer_id][reg_n][k],
              regs_c[m][j], regs_c[m][j],
              get_byte_id_a(stage_id, iter_id, m),
              get_thread_id_a(stage_id, iter_id, m),
              get_byte_id_b(stage_id, iter_id, j),
              get_thread_id_b(stage_id, iter_id, j));
        }
      }
    }
    if constexpr (kAsBlocksPerWord > 1) {
      if (iter_id == kWarpKIters - 1) k_block_idx++;
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
  CUDA_INLINE T *regs_sfa_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(regs_sfa[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_sfb_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(regs_sfb[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr(uint32_t buffer_id = 0) {
    return reinterpret_cast<T *>(regs_c);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    return regs_c_as_ptr<T>(0);
  };
};
