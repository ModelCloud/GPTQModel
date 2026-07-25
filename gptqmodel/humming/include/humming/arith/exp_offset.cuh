// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/datatype/base_conversion.cuh>
#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>


template <class ElementA, class ElementB, bool kHasZeroPoint = false>
CUDA_INLINE constexpr uint32_t get_dtype_dequant_exp_offset() {
  if constexpr (ElementA::kIsFloatingPointType) {
    if constexpr (ElementA::kExponentBits >= 1 && ElementA::kBits > ElementB::kBits) {
      if constexpr (ElementB::kIsFloatingPointType) {
        constexpr uint32_t source_exp_offset = 1 << (ElementB::kExponentBits - 1);
        constexpr uint32_t target_exp_offset = 1 << (ElementA::kExponentBits - 1);
        return target_exp_offset - source_exp_offset;
      }

      if constexpr (ElementA::kBits != 16 && ElementB::kIsIntegerType) {
        constexpr uint32_t target_exp_offset = 1 << (ElementA::kExponentBits - 1);
        return target_exp_offset - 2 + ElementA::kMantissaBits;
      }

      if constexpr (std::is_same<ElementA, BFloat16>::value && ElementB::kIsIntegerType) {
        if constexpr (kHasZeroPoint && ElementB::kBits > 6) {
          constexpr uint32_t target_exp_offset = 1 << (ElementA::kExponentBits - 1);
          return target_exp_offset - 2 + ElementA::kMantissaBits;
        }
        if constexpr (!kHasZeroPoint && ElementB::kBits > 7) {
          constexpr uint32_t target_exp_offset = 1 << (ElementA::kExponentBits - 1);
          return target_exp_offset - 2 + ElementA::kMantissaBits;
        }
      }
    }
  }

  return 0;
}


template <typename T, uint32_t kExpOffset>
CUDA_INLINE constexpr T prepare_exp_scale_factor() {
  static_assert(std::is_same<T, half2>::value || std::is_same<T, nv_bfloat162>::value || std::is_same<T, float>::value);

  T val;
  uint32_t &val_uint = *reinterpret_cast<uint32_t *>(&val);
  if constexpr (std::is_same<T, half2>::value) {
    val_uint = (kExpOffset * 0x00010001 + 0x000F000F) << 10;
  } else if constexpr (std::is_same<T, nv_bfloat162>::value) {
    val_uint = (kExpOffset * 0x00010001 + 0x007F007F) << 7;
  } else if constexpr (std::is_same<T, float>::value) {
    val_uint = (kExpOffset + 0x007F) << 23;
  }

  return val;
}


template <class ElementA, class ElementB, class ElementBS, bool kHasZeroPoint, bool kNativeMixed = false>
CUDA_INLINE constexpr uint32_t get_total_exp_offset() {
  uint32_t offset = 0;

  if constexpr (ElementA::kBits == 16)
    offset += get_dtype_dequant_exp_offset<ElementA, ElementBS>();
  if constexpr (!kNativeMixed)
    offset += get_dtype_dequant_exp_offset<ElementA, ElementB, kHasZeroPoint>();

  return offset;
};


template <
    class ElementA, class ElementB, class ElementBS, bool kHasZeroPoint,
    bool kIsF16Accum, bool kIsGroupInputScale, bool kIsGroupWeightScale, bool kNativeMixed = false>
CUDA_INLINE constexpr uint2 get_mainloop_exp_offset() {
  uint32_t total_offset = get_total_exp_offset<ElementA, ElementB, ElementBS, kHasZeroPoint, kNativeMixed>();

  // channelwise float8 scales must be applied on epilogue pipeline
  if constexpr (ElementA::kBits == 16 && ElementBS::kBits == 8 && !kIsGroupWeightScale) {
    total_offset -= get_dtype_dequant_exp_offset<ElementA, ElementBS>();
  }

  uint2 offset = {0, 0};

  if constexpr (ElementA::kBits == 16) {
    uint32_t max_allowed_offset = (1 << (ElementA::kExponentBits - 1)) - 1;
    if constexpr (std::is_same<ElementA, Float16>::value) {
      max_allowed_offset = max_allowed_offset - ElementB::kBits + 1;
    } else if constexpr (kHasZeroPoint && ElementB::kBits <= 6) {
      max_allowed_offset = max_allowed_offset - ElementB::kBits + 1;
    } else if constexpr (!kHasZeroPoint && ElementB::kBits <= 7) {
      max_allowed_offset = max_allowed_offset - ElementB::kBits + 1;
    }

    offset.x = MIN(max_allowed_offset, total_offset);

    if constexpr (std::is_same<ElementA, BFloat16>::value && ElementBS::kBits == 8 && kIsGroupWeightScale) {
      constexpr uint32_t scale_offset = get_dtype_dequant_exp_offset<ElementA, ElementBS>();
      offset.y = MIN(total_offset - offset.x, scale_offset);
    }
  }

  if constexpr (ElementA::kBits != 16 && kIsF16Accum && kIsGroupWeightScale) {
    offset.y = MIN(total_offset, 2);
  }

  return offset;
};


template <
    class ElementA, class ElementB, class ElementC, class ElementBS, bool kHasZeroPoint,
    bool kIsF16Accum, bool kIsGroupInputScale, bool kIsGroupWeightScale, bool kNativeMixed = false>
CUDA_INLINE constexpr uint2 get_epilogue_exp_offset() {
  uint32_t total_offset = get_total_exp_offset<ElementA, ElementB, ElementBS, kHasZeroPoint, kNativeMixed>();

  uint2 offset = {0, 0};
  if constexpr (ElementBS::kBits == 8 && !kIsGroupWeightScale) {
    offset.y = get_dtype_dequant_exp_offset<ElementC, ElementBS>();
  }

  uint2 mainloop_offset = get_mainloop_exp_offset<
      ElementA, ElementB, ElementBS, kHasZeroPoint,
      kIsF16Accum, kIsGroupInputScale, kIsGroupWeightScale, kNativeMixed>();

  offset.x = total_offset - mainloop_offset.x - mainloop_offset.y;
  if constexpr (ElementA::kBits == 16) offset.x -= offset.y;

  return offset;
};
