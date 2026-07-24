// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/all.cuh>


constexpr uint32_t get_dtype_id(
    uint32_t base_type_id, bool is_signed,
    uint32_t num_bits, uint32_t exponent_bits, uint32_t mantissa_bits) {

  uint32_t dtype_id = base_type_id * 10000000 + is_signed * 1000000;
  dtype_id += num_bits * 10000 + exponent_bits * 100 + mantissa_bits;
  return dtype_id;
}


template <bool kIsSigned_, uint32_t kNumBits_>
class IntegerType {
public:
  static constexpr bool kIsIntegerType = true;
  static constexpr bool kIsFloatingPointType = false;
  static constexpr bool kIsSigned = kIsSigned_;
  static constexpr uint32_t kBits = kNumBits_;
  static constexpr uint32_t kNumBits = kNumBits_;
  static constexpr uint32_t kId = get_dtype_id(1, kIsSigned, kNumBits_, 0, 0);
};

template <uint32_t kNumBits_, uint32_t kExponentBits_, uint32_t kMantissaBits_>
class FloatingPointType {
public:
  static constexpr bool kIsIntegerType = false;
  static constexpr bool kIsFloatingPointType = true;
  static constexpr uint32_t kBits = kNumBits_;
  static constexpr uint32_t kNumBits = kNumBits_;
  static constexpr uint32_t kSignBits = kNumBits_ - kExponentBits_ - kMantissaBits_;
  static_assert(kSignBits == 0 || kSignBits == 1);
  static constexpr bool kIsSigned = kSignBits != 0;
  static constexpr uint32_t kExponentBits = kExponentBits_;
  static constexpr uint32_t kMantissaBits = kMantissaBits_;
  static constexpr uint32_t kId = get_dtype_id(2, kIsSigned, kNumBits_, kExponentBits, kMantissaBits);
};

using UInt1 = IntegerType<false, 1>;
using UInt2 = IntegerType<false, 2>;
using UInt3 = IntegerType<false, 3>;
using UInt4 = IntegerType<false, 4>;
using UInt5 = IntegerType<false, 5>;
using UInt6 = IntegerType<false, 6>;
using UInt7 = IntegerType<false, 7>;
using UInt8 = IntegerType<false, 8>;

using Int2 = IntegerType<true, 2>;
using Int3 = IntegerType<true, 3>;
using Int4 = IntegerType<true, 4>;
using Int5 = IntegerType<true, 5>;
using Int6 = IntegerType<true, 6>;
using Int7 = IntegerType<true, 7>;
using Int8 = IntegerType<true, 8>;
using Int32 = IntegerType<true, 32>;

using Sign = FloatingPointType<1, 0, 0>;
using Float4E0M3 = FloatingPointType<4, 0, 3>;
using Float4E2M1 = FloatingPointType<4, 2, 1>;
using Float6E3M2 = FloatingPointType<6, 3, 2>;
using Float6E2M3 = FloatingPointType<6, 2, 3>;
using Float8E3M4 = FloatingPointType<8, 3, 4>;
using Float8E4M3 = FloatingPointType<8, 4, 3>;
using Float8E5M2 = FloatingPointType<8, 5, 2>;
using Float16 = FloatingPointType<16, 5, 10>;
using BFloat16 = FloatingPointType<16, 8, 7>;
using Float32 = FloatingPointType<32, 8, 23>;

using Float8E8M0 = FloatingPointType<8, 8, 0>;
