// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/base.cuh>


CUDA_INLINE float dequant_fp4_val(uint32_t val) {
  uint32_t sign = (val & 0x8) << 28;
  uint32_t other = (val & 0x7) << 22;
  uint32_t res = sign | other;

  return *reinterpret_cast<float *>(&res);
};


CUDA_INLINE uint32_t quant_to_fp4_val(float val) {
  uint32_t val_uint = *reinterpret_cast<uint32_t *>(&val);

  constexpr uint32_t mask = 0x81C00000;

  uint32_t val_uint_rz = val_uint & mask;
  uint32_t val_uint_ru = (val_uint + 0x00200000) & mask;

  float val_rz = *reinterpret_cast<float *>(&val_uint_rz);
  float val_ru = *reinterpret_cast<float *>(&val_uint_ru);

  float val_rn;
  float delta_rz = fabsf(val - val_rz);
  float delta_ru = fabsf(val - val_ru);

  if (delta_rz >= delta_ru) {
    val_rn = val_ru;
  } else {
    val_rn = val_rz;
  }

  uint32_t fp_val_uint = *reinterpret_cast<uint32_t *>(&val_rn);
  return ((fp_val_uint & 0x80000000) >> 28) | ((fp_val_uint & 0x01C00000) >> 22);
}

__global__ void process_mxfp4_w4a8(uint4 *in_ptr, uint4 *out_ptr, uint8_t *delta_scale_offset_ptr, uint32_t num_groups) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_groups) return;

  uint32_t delta_scale_offset = delta_scale_offset_ptr[index];
  uint4 int4_val = in_ptr[index];

  uint32_t *ints = reinterpret_cast<uint32_t *>(&int4_val);
  uint32_t res[4] = {0, 0, 0, 0};

  PRAGMA_UNROLL
  for (uint32_t j = 0; j < 4; j++) {
    uint32_t val = ints[j];
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < 8; k++) {
      uint32_t tmp_val = (val >> (k * 4)) & 0xF;
      tmp_val = tmp_val == 8 ? 0 : tmp_val; // 0b1000 is negative zero
      if (delta_scale_offset) {
        float float_val = dequant_fp4_val(tmp_val);
        uint32_t scale_factor_uint = 0x3F800000 - (delta_scale_offset << 23);
        float scale_factor = *reinterpret_cast<float *>(&scale_factor_uint);
        float_val = float_val * scale_factor;
        tmp_val = quant_to_fp4_val(float_val);
      }
      res[j] = res[j] | (tmp_val << (k * 4));
    }
  }

  out_ptr[index] = *reinterpret_cast<uint4 *>(res);
}
