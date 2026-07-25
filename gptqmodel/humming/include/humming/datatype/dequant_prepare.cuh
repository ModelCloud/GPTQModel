// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


template <uint32_t kLeftBits, uint32_t kRightBits, bool kIsRTL>
CUDA_INLINE uint32_t construct_quanted_value(const uint32_t &val1, const uint32_t &val2) {
  constexpr uint32_t kTotalBits = kLeftBits + kRightBits;
  if constexpr (kRightBits <= 31) {
    static_assert(kTotalBits == 3 || kTotalBits == 5 || kTotalBits == 6 || kTotalBits == 7);
    constexpr uint32_t kPaddedTotalBits = static_next_power_of_2(kTotalBits);
    constexpr uint32_t kPadBits = kPaddedTotalBits - kTotalBits;
    static_assert(kPaddedTotalBits == 4 || kPaddedTotalBits == 8);

    constexpr uint32_t one = kPaddedTotalBits == 8 ? 0x01010101 : 0x11111111;
    constexpr uint32_t left_mask = ((one << (kPaddedTotalBits - kLeftBits)) - one) ^ 0xFFFFFFFF;
    constexpr uint32_t right_mask = (one << kRightBits) - one;

    if constexpr (kLeftBits == 0 && kIsRTL) {
      return val1;
    } else if constexpr (kLeftBits == 0 && !kIsRTL) {
      return val1 << kPadBits;
    } else if constexpr (kRightBits == 0 && kIsRTL) {
      return val2 >> kPadBits;
    } else if constexpr (kRightBits == 0 && !kIsRTL) {
      return val2;
    } else if constexpr (kIsRTL) {
      return lop3_and_or(val2, right_mask, (val1 & left_mask) >> kPadBits);
    } else if constexpr (!kIsRTL) {
      return lop3_and_or(val1, left_mask, (val2 & right_mask) << kPadBits);
    }
  };
}


template <uint32_t kNumBits, bool kIsRTL>
CUDA_INLINE uint32_t get_quanted_value_group(const uint32_t *qb, uint32_t i) {
  static_assert(kNumBits == 3 || kNumBits == 5 || kNumBits == 6 || kNumBits == 7);
  constexpr uint32_t kPaddedTotalBits = static_next_power_of_2(kNumBits);

  auto construct = [&](const uint32_t left_bits, const uint32_t &left_val, const uint32_t &right_val) {
    switch (left_bits) {
      case 0: return construct_quanted_value<0, kNumBits - 0, kIsRTL>(left_val, right_val);
      case 1: return construct_quanted_value<1, kNumBits - 1, kIsRTL>(left_val, right_val);
      case 2: return construct_quanted_value<2, kNumBits - 2, kIsRTL>(left_val, right_val);
      case 3: return construct_quanted_value<3, kNumBits - 3, kIsRTL>(left_val, right_val);
      case 4: return construct_quanted_value<4, kNumBits - 4, kIsRTL>(left_val, right_val);
      case 5: return construct_quanted_value<5, kNumBits - 5, kIsRTL>(left_val, right_val);
      case 6: return construct_quanted_value<6, kNumBits - 6, kIsRTL>(left_val, right_val);
      case 7: return construct_quanted_value<7, kNumBits - 7, kIsRTL>(left_val, right_val);
    };
  };

  uint32_t start_bits = i * kNumBits;
  uint32_t end_bits = (i + 1) * kNumBits - 1;
  uint32_t left_bits = CEIL_DIV(start_bits, kPaddedTotalBits) * kPaddedTotalBits - start_bits;
  uint32_t right_bits = kNumBits - left_bits;

  const uint32_t &left_val = qb[start_bits / kPaddedTotalBits];
  const uint32_t &right_val = qb[end_bits / kPaddedTotalBits];
  if (left_bits <= kNumBits) {
    return construct(left_bits, left_val, right_val);
  } else if (kIsRTL) {
    return left_val >> (kPaddedTotalBits - left_bits);
  } else {
    return left_val << (left_bits - kNumBits);
  }
};
