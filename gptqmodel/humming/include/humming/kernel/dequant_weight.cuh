// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/base.cuh>


__global__ void dequant_unpacked_fp_type(
    uint32_t *in_ptr, uint32_t *out_ptr,
    uint32_t total_size,
    uint32_t exponent_bits,
    uint32_t mantissa_bits,
    bool is_signed) {

  constexpr uint64_t COUNT = 32L;
  uint64_t offset = COUNT * (blockIdx.x * blockDim.x + threadIdx.x);

  const uint32_t mask = (1 << (exponent_bits + mantissa_bits)) - 1;
  const uint32_t sign_mask = is_signed ? mask + 1 : 0;

  uint32_t exp_offset = 128 - (1 << (exponent_bits - 1));
  uint32_t scale_factor = (exp_offset << 23) + 0x3F800000;
  float scale_factor_float = *reinterpret_cast<float *>(&scale_factor);

  uint32_t vals[COUNT];
  float *vals_float = reinterpret_cast<float *>(vals);

#pragma unroll
  for (uint64_t i = 0; i < COUNT; i++) {
    if (offset + i >= total_size) break;
    vals[i] = in_ptr[offset + i];
  }

#pragma unroll
  for (uint64_t i = 0; i < COUNT; i++) {
    uint32_t val = vals[i];
    if (exponent_bits == 0) {
      // Fixed-point sign-magnitude format (e0mX): value = (-1)^sign * magnitude.
      float f = (float)(val & mask);
      vals_float[i] = (is_signed && (val & sign_mask)) ? -f : f;
    } else {
      uint32_t part1 = (val & sign_mask) << (31 - (exponent_bits + mantissa_bits));
      uint32_t part2 = (val & mask) << (23 - mantissa_bits);
      vals[i] = part1 | part2;
      vals_float[i] = vals_float[i] * scale_factor_float;
    }
  }

#pragma unroll
  for (uint64_t i = 0; i < COUNT; i++) {
    if (offset + i >= total_size) break;
    out_ptr[offset + i] = vals[i];
  }
}
