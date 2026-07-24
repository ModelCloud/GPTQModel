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


template <class SourceType, class TargetType, bool kHasZeroPoint>
CUDA_INLINE uint32_t uint_to_int(uint32_t val, const uint32_t &zp_val) {
  static_assert(SourceType::kBits < TargetType::kBits);
  static_assert(SourceType::kIsIntegerType);
  static_assert(!SourceType::kIsSigned);
  static_assert(std::is_same<TargetType, Int8>::value || std::is_same<TargetType, Int4>::value);

  constexpr uint32_t one = std::is_same<TargetType, Int8>::value ? 0x01010101 : 0x11111111;
  constexpr uint32_t mask1 = (one << SourceType::kBits) - one;
  constexpr uint32_t mask2 = std::is_same<TargetType, Int8>::value ? 0x80808080 : 0x88888888;
  constexpr uint32_t fixed_zp_val = one << (SourceType::kBits - 1);

  uint32_t final_zp_val = kHasZeroPoint ? zp_val : fixed_zp_val;
  return (lop3_and_or(val, mask1, mask2) - final_zp_val) ^ mask2;
}


template <class SourceType, class TargetType, bool kHasZeroPoint, bool kIsFpZeroPoint>
CUDA_INLINE uint32_t uint_to_f16(uint32_t val, const uint32_t &zp_val) {
  using scalar_t2 = typename F16Conversion<TargetType>::scalar_t2;
  static_assert(SourceType::kIsIntegerType);
  static_assert(!SourceType::kIsSigned);
  static_assert(TargetType::kIsFloatingPointType);
  static_assert(TargetType::kBits == 16);
  static_assert(SourceType::kBits <= TargetType::kMantissaBits);

  constexpr uint32_t base = std::is_same<TargetType, Float16>::value ? 0x64006400 : 0x43004300;
  constexpr uint32_t fixed_zp_val = (0x00010001 << (SourceType::kBits - 1)) + base;
  constexpr uint32_t mask = ((0x00010001 << SourceType::kBits) - 0x00010001);

  uint32_t extracted_val = lop3_and_or(val, mask, base);
  scalar_t2 extracted_val_half2 = *reinterpret_cast<scalar_t2 *>(&extracted_val);

  uint32_t final_zp_val = kHasZeroPoint ? zp_val : fixed_zp_val;
  if constexpr (kIsFpZeroPoint) final_zp_val = base;
  const scalar_t2 zp_half2 = *reinterpret_cast<const scalar_t2 *>(&final_zp_val);
  extracted_val_half2 = __hsub2(extracted_val_half2, zp_half2);

  return *reinterpret_cast<uint32_t *>(&extracted_val_half2);
};


template <class SourceType, class TargetType, bool kHasZeroPoint, bool kIsFpZeroPoint>
CUDA_INLINE uint32_t normalized_uint_to_fp(uint32_t val, const uint32_t &zp_val) {
  static_assert(SourceType::kIsIntegerType);
  static_assert(!SourceType::kIsSigned);
  static_assert(TargetType::kIsFloatingPointType);
  static_assert(TargetType::kIsSigned);
  static_assert(SourceType::kBits <= TargetType::kMantissaBits + 2);
  static_assert(SourceType::kBits < TargetType::kBits);
  static_assert(TargetType::kBits == 16 || TargetType::kBits == 8 || TargetType::kBits == 4);

  constexpr uint32_t one = TargetType::kBits == 16 ? 0x00010001 : (TargetType::kBits == 8 ? 0x01010101 : 0x11111111);

  if constexpr (!kHasZeroPoint) {
    constexpr uint32_t mask1 = one << (SourceType::kBits - 1);
    constexpr uint32_t mask2 = mask1 - one;
    constexpr uint32_t delta_bits1 = TargetType::kBits - SourceType::kBits;
    constexpr uint32_t delta_bits2 = SourceType::kBits - 1;

    uint32_t sign_bit = val & mask1;
    return lop3_and_or(val, mask2, sign_bit << delta_bits1) + (sign_bit >> delta_bits2);
  } else {
    constexpr uint32_t mask1 = (one << SourceType::kBits) - one;
    constexpr uint32_t mask2 = ((one << (TargetType::kBits - 1)) - one) ^ mask1;
    constexpr uint32_t sign_mask = one << (TargetType::kBits - 1);

    uint32_t zp_val1 = kIsFpZeroPoint ? 0 : zp_val;
    uint32_t zp_val2 = zp_val1 + 1;
    uint32_t val2 = lop3_and_or(val, mask1, mask2);
    uint32_t sign_bit = (val2 + (zp_val1 * one)) & sign_mask;
    uint32_t val3 = (sign_bit >> (TargetType::kBits - 1)) * zp_val2 + val2;
    return lop3_and_or(val3, mask1, sign_bit);
  }
};


template <class SourceType, class TargetType>
CUDA_INLINE uint32_t fp_to_fp(uint32_t val) {
  static_assert(SourceType::kIsFloatingPointType);
  static_assert(TargetType::kIsFloatingPointType);
  static_assert(SourceType::kExponentBits <= TargetType::kExponentBits);
  static_assert(SourceType::kMantissaBits <= TargetType::kMantissaBits);
  static_assert(SourceType::kBits < TargetType::kBits);
  static_assert(!SourceType::kIsSigned || TargetType::kIsSigned);
  static_assert(TargetType::kBits == 16 || TargetType::kBits == 8 || TargetType::kBits == 4);

  constexpr uint32_t repeated_one = TargetType::kBits == 16 ? 0x00010001 : (TargetType::kBits == 8 ? 0x01010101 : 0x11111111);
  constexpr uint32_t signbit_mask = TargetType::kBits == 16 ? 0x80008000 : (TargetType::kBits == 8 ? 0x80808080 : 0x88888888);
  constexpr uint32_t nonsign_bits = SourceType::kExponentBits + SourceType::kMantissaBits;
  constexpr uint32_t shifted_mask = (repeated_one << nonsign_bits) - repeated_one;
  constexpr uint32_t mask = shifted_mask << (TargetType::kBits - SourceType::kBits);

  constexpr uint32_t diff_exp_bits =
      SourceType::kExponentBits == 0
          ? (TargetType::kMantissaBits - SourceType::kMantissaBits)
          : (TargetType::kExponentBits - SourceType::kExponentBits);

  if constexpr (std::is_same<SourceType, Float8E8M0>::value && std::is_same<TargetType, BFloat16>::value) {
    return lop3_and_or(val, 0, (val & 0xFF00FF00) >> 1);
  } else {
    return lop3_and_or(val, signbit_mask, (val & mask) >> diff_exp_bits);
  }
};


template <class SourceType, class TargetType, bool kHasZeroPoint, bool kIsFpZeroPoint>
CUDA_INLINE uint32_t dequant_single(uint32_t val, const uint32_t &zp_val) {
  if constexpr (std::is_same<SourceType, TargetType>::value) {
    static_assert(!kHasZeroPoint);
    return val;
  } else if constexpr (SourceType::kIsFloatingPointType && TargetType::kIsFloatingPointType) {
    static_assert(!kHasZeroPoint);
    return fp_to_fp<SourceType, TargetType>(val);
  } else if constexpr (TargetType::kIsFloatingPointType) {
    static_assert(SourceType::kIsIntegerType && !SourceType::kIsSigned);
    if constexpr (TargetType::kBits == 16 && !kHasZeroPoint && SourceType::kBits <= TargetType::kMantissaBits) {
      return uint_to_f16<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(val, zp_val);
    } else if constexpr (TargetType::kBits == 16 && kHasZeroPoint && SourceType::kBits <= TargetType::kMantissaBits - 1) {
      return uint_to_f16<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(val, zp_val);
    } else {
      return normalized_uint_to_fp<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(val, zp_val);
    }
  } else if constexpr (TargetType::kIsIntegerType) {
    static_assert(SourceType::kIsIntegerType);
    return uint_to_int<SourceType, TargetType, kHasZeroPoint>(val, zp_val);
  };
};


template <class SourceType, class TargetType>
CUDA_INLINE uint32_t dequant_single_zero_point(uint32_t &qzp_val) {
  constexpr uint32_t one = TargetType::kBits == 16 ? 0x00010001 : (TargetType::kBits == 8 ? 0x01010101 : 0x11111111);
  constexpr uint32_t mask = (1 << SourceType::kBits) - 1;

  if constexpr (SourceType::kIsIntegerType && !SourceType::kIsSigned) {
    if constexpr (std::is_same<TargetType, Float16>::value) {
      return (qzp_val & mask) * one + 0x64006400;
    } else if constexpr (std::is_same<TargetType, BFloat16>::value && SourceType::kBits <= 6) {
      return (qzp_val & mask) * one + 0x43004300;
    } else if constexpr (TargetType::kIsIntegerType) {
      return (qzp_val & mask) * one;
    } else {
      return (qzp_val & mask);
    }
  }

  return 0;
};
