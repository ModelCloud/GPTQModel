// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/arith/exp_offset.cuh>
#include <humming/datatype/base_conversion.cuh>
#include <humming/datatype/dequant.cuh>
#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>

template <class Ctx>
class EpilogueArithmetic : F16Conversion<typename Ctx::ElementC> {
private:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;
  using ElementC = typename Ctx::ElementC;
  using ElementBS = typename Ctx::ElementBS;

  using scalar_t = typename F16Conversion<ElementC>::scalar_t;
  using scalar_t2 = typename F16Conversion<ElementC>::scalar_t2;

  static constexpr bool kUseStreamK = Ctx::kUseStreamK;
  static constexpr bool kIsF16Accum = MmaOpClass::kCTypeBits == 16;
  static constexpr bool kHasBias = Ctx::kHasBias;
  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsGroupInputScale = kHasInputScale && Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kIsChannelInputScale = kHasInputScale && Ctx::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = Ctx::kIsBlockWeightScale;
  static constexpr bool kIsChannelWeightScale = Ctx::kIsChannelWeightScale;
  static constexpr bool kIsChannelWeightScale2 = Ctx::kIsChannelWeightScale2;
  static constexpr bool kIsGroupOrBlockWeightScale = kIsGroupWeightScale || kIsBlockWeightScale;
  static constexpr bool kHasChannelWeightScale = Ctx::kHasChannelWeightScale;
  static constexpr bool kHasTensorWeightScale = Ctx::kHasTensorWeightScale;
  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;

  static constexpr uint2 kExpOffset = get_epilogue_exp_offset<
      ElementA, ElementB, ElementC, ElementBS, kHasZeroPoint,
      kIsF16Accum, kIsGroupInputScale, kIsGroupOrBlockWeightScale, MmaOpClass::kNativeMixed>();

  static constexpr uint32_t kSizeAS = WarpShape::M / 8;
  static constexpr uint32_t kSizeBS = WarpShape::N / 4 * ElementBS::kBits / 32;
  static constexpr uint32_t kSizeDequantBS = kSizeBS * 16 / ElementBS::kBits;
  static constexpr uint32_t kSizeBias = WarpShape::N / 4 * 16 / 32;

public:
  uint32_t as[kSizeAS];
  uint32_t bs[MAX(kSizeBS, 2)];
  uint32_t dq_bs[MAX(kSizeDequantBS, 4)];
  uint32_t bias[kSizeBias];
  uint32_t bs2[kSizeBias];
  uint32_t gs = 0;
  uint32_t _dummy;

  CUDA_INLINE
  void may_process_f32_on_smem_write(uint32_t row, uint32_t col) {
    if (kHasTensorWeightScale && row == 0 && col == 0) {
      float &gs_f32 = *reinterpret_cast<float *>(&gs);
      if constexpr (kExpOffset.x) gs_f32 *= prepare_exp_scale_factor<float, kExpOffset.x>();
      float *as_f32_ptr = reinterpret_cast<float *>(as);

      if constexpr (kIsChannelInputScale) {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kSizeAS; i++) {
          as_f32_ptr[i] = as_f32_ptr[i] * gs_f32;
        }
      }
    }
  }

  CUDA_INLINE
  void may_apply_f32_on_smem_write(float2 &regs, uint32_t row, uint32_t col) {
    may_process_f32_on_smem_write(row, col);
    if constexpr (kIsChannelInputScale && !kIsF16Accum) {
      float *as_f32_ptr = reinterpret_cast<float *>(as);
      regs.x = regs.x * as_f32_ptr[row];
      regs.y = regs.y * as_f32_ptr[row];
    } else if constexpr (kHasTensorWeightScale && !kIsF16Accum) {
      float &gs_f32 = *reinterpret_cast<float *>(&gs);
      regs.x = regs.x * gs_f32;
      regs.y = regs.y * gs_f32;
    }
  };

  CUDA_INLINE
  void may_process_on_smem_write(uint32_t row, uint32_t col) {
    if (kHasTensorWeightScale && kIsF16Accum && row == 0 && col == 0) {
      scalar_t2 &gs_scalar2 = *reinterpret_cast<scalar_t2 *>(&gs);
      gs_scalar2 = this->float2num2(*reinterpret_cast<float *>(&gs));

      if constexpr (kExpOffset.x) {
        gs_scalar2 = gs_scalar2 * prepare_exp_scale_factor<scalar_t2, kExpOffset.x>();
      }
    }
    if constexpr (kIsChannelInputScale && kIsF16Accum) {
      if (row == 0 && col == 0) {
        scalar_t2 &gs_scalar2 = *reinterpret_cast<scalar_t2 *>(&gs);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kSizeAS; i++) {
          reinterpret_cast<scalar_t2 *>(as)[i] = this->float2num2(reinterpret_cast<float *>(as)[i]);
          if constexpr (kHasTensorWeightScale)
            reinterpret_cast<scalar_t2 *>(as)[i] = __hmul2(reinterpret_cast<scalar_t2 *>(as)[i], gs_scalar2);
        };
      };
    }

    if constexpr (kIsChannelWeightScale && ElementBS::kBits == 8) {
      if (row == 0 && col == 0) {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(WarpShape::N, 32); i++) {
          dequant<ElementBS, ElementC>(bs, dq_bs + i * 4, i);
          scalar_t *dq_bs_scalar_ptr = reinterpret_cast<scalar_t *>(dq_bs + i * 4);

          PRAGMA_UNROLL
          for (uint32_t j = 0; j < 2; j++) {
            scalar_t tmp = dq_bs_scalar_ptr[2 + 4 * j];
            dq_bs_scalar_ptr[2 + 4 * j] = dq_bs_scalar_ptr[1 + 4 * j];
            dq_bs_scalar_ptr[1 + 4 * j] = tmp;
          }
        }

        const scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, kExpOffset.y>();
        scalar_t2 *dq_bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(dq_bs);

        PRAGMA_UNROLL
        for (uint32_t j = 0; j < CEIL_DIV(WarpShape::N, 8); j++) {
          dq_bs_scalar2_ptr[j] = __hmul2(dq_bs_scalar2_ptr[j], scale_factor);
        }
      };
    };
  };

  CUDA_INLINE
  void may_apply_on_smem_write(uint32_t &regs, uint32_t row, uint32_t col) {
    may_process_on_smem_write(row, col);

    auto apply_exp_offset = [&]() {
      if constexpr (kExpOffset.x && !kHasTensorWeightScale) {
        const scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, kExpOffset.x>();
        scalar_t2 *b_f16_ptr = reinterpret_cast<scalar_t2 *>(&regs);
        b_f16_ptr[0] = __hmul2(b_f16_ptr[0], scale_factor);
      }
    };

    if constexpr (!kIsF16Accum) apply_exp_offset();

    scalar_t2 *as_half2 = reinterpret_cast<scalar_t2 *>(as);
    scalar_t2 *gs_half2 = reinterpret_cast<scalar_t2 *>(&gs);
    scalar_t2 *bs_half2 = reinterpret_cast<scalar_t2 *>(ElementBS::kBits == 8 ? dq_bs : bs);
    scalar_t2 *bias_half2 = reinterpret_cast<scalar_t2 *>(bias);
    scalar_t2 *regs_half2 = reinterpret_cast<scalar_t2 *>(&regs);
    scalar_t2 *cs_half2 = kIsChannelWeightScale2 ? reinterpret_cast<scalar_t2 *>(bs2) : bs_half2;

    if constexpr (kIsChannelInputScale && kIsF16Accum) {
      regs_half2[0] = __hmul2(regs_half2[0], as_half2[row]);
    } else if constexpr (kHasTensorWeightScale && kIsF16Accum) {
      regs_half2[0] = __hmul2(regs_half2[0], gs_half2[0]);
    }

    if constexpr (kHasChannelWeightScale && kHasBias && !kIsF16Accum) {
      regs_half2[0] = __hfma2(regs_half2[0], cs_half2[col], bias_half2[col]);
    } else if constexpr (kHasChannelWeightScale) {
      regs_half2[0] = __hmul2(regs_half2[0], cs_half2[col]);
    } else if constexpr (kHasBias) {
      regs_half2[0] = __hadd2(regs_half2[0], bias_half2[col]);
    };

    if constexpr (kIsF16Accum) apply_exp_offset();

    if constexpr (kIsF16Accum && kHasBias) {
      regs_half2[0] = __hadd2(regs_half2[0], bias_half2[col]);
    }
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_bs_as_ptr() {
    return reinterpret_cast<T *>(bs);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_as_as_ptr() {
    return reinterpret_cast<T *>(as);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_bias_as_ptr() {
    return reinterpret_cast<T *>(bias);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_bs2_as_ptr() {
    return reinterpret_cast<T *>(bs2);
  };
};
