// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


CUDA_INLINE void shlf_trans_mma_c_32b(void *vals_ptr) {
  uint32_t *vals_uint_ptr = reinterpret_cast<uint32_t *>(vals_ptr);

  uint32_t val;
  uint32_t idx = (threadIdx.x / 4) % 2;
  switch (idx) {
    case 0: {
      val = vals_uint_ptr[1];
      break;
    };
    case 1: {
      val = vals_uint_ptr[0];
      break;
    }
  }

  uint32_t swapped_val = __shfl_xor_sync(0xffffffff, val, 4);

  switch (idx) {
    case 0: {
      vals_uint_ptr[1] = swapped_val;
      break;
    };
    case 1: {
      vals_uint_ptr[0] = swapped_val;
      break;
    }
  }
}


CUDA_INLINE void shlf_trans_mma_c_16b(void *vals_ptr) {
  uint32_t *vals_uint_ptr = reinterpret_cast<uint32_t *>(vals_ptr);

  uint32_t &val = vals_uint_ptr[0];
  uint32_t swapped_val = __shfl_xor_sync(0xffffffff, val, 4);
  uint32_t idx = (threadIdx.x / 4) % 2;

  uint16_t *vals_ushort_ptr = reinterpret_cast<uint16_t *>(&val);
  uint16_t *swapped_vals_ushort_ptr = reinterpret_cast<uint16_t *>(&swapped_val);

  switch (idx) {
    case 0: {
      vals_ushort_ptr[1] = swapped_vals_ushort_ptr[0];
      break;
    };
    case 1: {
      vals_ushort_ptr[0] = swapped_vals_ushort_ptr[1];
      break;
    }
  }
}


template <typename T>
CUDA_INLINE void shlf_trans_mma_c(T &vals) {
  static_assert(sizeof(T) == 8 || sizeof(T) == 4);
  if constexpr (sizeof(T) == 8) {
    shlf_trans_mma_c_32b(&vals);
  } else {
    shlf_trans_mma_c_16b(&vals);
  }
}


template <class Ctx, class MMA, class ArithClass>
class EpilogueSmemWriter : F16Conversion<typename Ctx::ElementC> {
private:
  using SharedStorage = typename Ctx::SharedStorage;
  using MmaOpClass = typename Ctx::MmaOpClass;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementC = typename Ctx::ElementC;

  static constexpr bool kUseWgmma = Ctx::kUseWgmma;

  using scalar_t = typename F16Conversion<ElementC>::scalar_t;
  using scalar_t2 = typename F16Conversion<ElementC>::scalar_t2;
  using MmaShape = typename MmaOpClass::MmaShape;
  using ValTypeC = typename MmaOpClass::ValTypeC;
  using CRegistersArrayType = typename MMA::CRegistersArrayType;

  static constexpr uint32_t kNumWriteSplits = Ctx::kNumWriteSplits;
  static constexpr uint32_t kNumMathThreads = Ctx::kNumMathThreads;
  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsGroupInputScale = kHasInputScale && Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = Ctx::kIsBlockWeightScale;
  static constexpr bool kUseIntWeightScale = Ctx::kUseIntWeightScale;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;
  static constexpr bool kHasGroupScale = kIsGroupInputScale || kIsGroupWeightScale || kIsBlockWeightScale;
  static constexpr bool kIsIntAccum = std::is_same<ValTypeC, int32_t>::value && (!kHasGroupScale || kUseIntWeightScale || kUseFusedE8m0Scale);

  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

public:
  Ctx &ctx;
  ArithClass &arith;

  CUDA_INLINE
  EpilogueSmemWriter(Ctx &ctx, ArithClass &arith)
      : ctx(ctx), arith(arith) {
  }

  CUDA_INLINE
  void write(uint32_t *regs_ptr, uint32_t slice_count, uint32_t split_idx) {
    if (threadIdx.x >= kNumMathThreads / K_WARPS) return;

    auto &regs = *reinterpret_cast<CRegistersArrayType *>(regs_ptr);
    scalar_t2 *smem_half2_ptr = reinterpret_cast<scalar_t2 *>(ctx.smem.reduce);
    uint32_t smem = offsetof(SharedStorage, reduce) / 128 % 8;
    using PackTypeC = std::conditional_t<
        sizeof(ValTypeC) == 2, scalar_t2,
        std::conditional_t<kIsIntAccum, int2, float2>>;

    uint32_t laneid = ctx.lane_id();
    uint32_t warpid = ctx.warp_id();
    uint32_t warp_delta_row = ctx.m_warp_offset();
    uint32_t n_warp_id = ctx.n_warp_id();
    uint32_t group_warp_id = warpid % 4;
    auto write_to_smem = [&](PackTypeC val, uint32_t row_8x8block, uint32_t col_8x8block) {
      scalar_t2 val_half2;

      static_assert(kNumWriteSplits == 1 || kNumWriteSplits == 2);
      if constexpr (kNumWriteSplits == 2) {
        static_assert(M_WARPS == 1);
        uint32_t m_8x8block = kUseWgmma ? col_8x8block : row_8x8block;
        if (split_idx == 0 && m_8x8block >= BlockShape::M / 8 / 2) return;
        if (split_idx == 1 && m_8x8block < BlockShape::M / 8 / 2) return;
      }

      if constexpr (kUseWgmma) shlf_trans_mma_c(val);
      if constexpr (sizeof(ValTypeC) != 4) {
        val_half2 = val;
      } else if constexpr (kIsIntAccum) {
        float2 val_float2 = {__int2float_rn(val.x), __int2float_rn(val.y)};
        if constexpr (kUseWgmma) {
          arith.may_apply_f32_on_smem_write(val_float2, col_8x8block, row_8x8block);
        } else {
          arith.may_apply_f32_on_smem_write(val_float2, row_8x8block, col_8x8block);
        }
        val_half2 = this->float22num2(val_float2);
      } else {
        if constexpr (kUseWgmma) {
          arith.may_apply_f32_on_smem_write(val, col_8x8block, row_8x8block);
        } else {
          arith.may_apply_f32_on_smem_write(val, row_8x8block, col_8x8block);
        }
        val_half2 = this->float22num2(val);
      };

      uint32_t &val_uint = *reinterpret_cast<uint32_t *>(&val_half2);
      if constexpr (kUseWgmma) {
        arith.may_apply_on_smem_write(val_uint, col_8x8block, row_8x8block);
        col_8x8block = col_8x8block - BlockShape::M / 8 / 2 * split_idx;
      } else {
        arith.may_apply_on_smem_write(val_uint, row_8x8block, col_8x8block);
        row_8x8block = row_8x8block - BlockShape::M / 8 / 2 * split_idx;
      }

      if constexpr (!kUseWgmma) {
        uint32_t sub_row = laneid / 4;
        uint32_t row = warp_delta_row + 8 * row_8x8block + sub_row;
        uint32_t col = col_8x8block * 4 + WarpShape::N / 2 * n_warp_id;

        row = row + (BlockShape::M / kNumWriteSplits) * (col / 32);
        col = ((col % 32 / 4) ^ ((sub_row + smem) % 8)) * 4 + laneid % 4;

        uint32_t idx = row * 32 + col;
        smem_half2_ptr[idx] = val_half2;
      } else {
        uint32_t sub_row = (laneid % 4) * 2 + (laneid % 8) / 4;
        uint32_t row = warp_delta_row + 8 * col_8x8block + sub_row;

        uint32_t count = (64 / WarpShape::N);
        uint32_t col1 = ((n_warp_id % count * (8 / count) + row_8x8block) ^ ((sub_row + smem) % 8)) * 4 + laneid / 8;
        uint32_t col2 = (n_warp_id / count) * (BlockShape::M / kNumWriteSplits * 64 / 2);
        uint32_t idx = row * 32 + col1 + col2;
        smem_half2_ptr[idx] = val_half2;
      }
    };

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs) / sizeof(regs[0]); i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < sizeof(regs[0]) / sizeof(regs[0][0]); j++) {
        auto part_regs = reinterpret_cast<PackTypeC *>(&regs[i][j]);
        constexpr uint32_t inner_m = (kUseWgmma ? (MmaShape::N / 4) : MmaShape::M) / 8;
        constexpr uint32_t inner_n = sizeof(regs[0][0]) / sizeof(PackTypeC) / inner_m;

        PRAGMA_UNROLL
        for (uint32_t m = 0; m < inner_m; m++) {
          PRAGMA_UNROLL
          for (uint32_t n = 0; n < inner_n; n++) {
            uint32_t row_index = i * inner_m + m;
            uint32_t col_index = j * inner_n + n;
            write_to_smem(part_regs[n * inner_m + m], row_index, col_index);
          }
        }
      }
    }
  }
};
