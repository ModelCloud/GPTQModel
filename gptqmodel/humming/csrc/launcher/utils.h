// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

inline void check_curesult(const CUresult res, const char *func_name) {
  if (res != CUDA_SUCCESS) {
    const char *errName;
    const char *errStr;
    cuGetErrorName(res, &errName);
    cuGetErrorString(res, &errStr);
    ASSERT_CHECK(false, func_name, " failed with error: ", errName, " (", errStr, ")");
  }
}

uint32_t manual_crc32(const std::string &data) {
  static uint32_t table[256];
  static bool table_computed = false;

  if (!table_computed) {
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (int j = 0; j < 8; j++) {
        if (c & 1) c = 0xEDB88320L ^ (c >> 1);
        else c = c >> 1;
      }
      table[i] = c;
    }
    table_computed = true;
  }

  uint32_t crc = 0xFFFFFFFFL;
  for (unsigned char b : data) {
    crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFL;
}

uint32_t get_dtype_num_bits(uint32_t dtype_id) {
  return (dtype_id / 10000) % 100;
};

ScalarType dtype_id_to_tensor_dtype(uint32_t dtype_id) {
  switch (dtype_id) {
    case 10080000: return ScalarType::Byte;
    case 11080000: return ScalarType::Char;
    case 21080403: return ScalarType::Float8_e4m3fn;
    case 21080502: return ScalarType::Float8_e5m2;
    case 20080800: return ScalarType::Float8_e8m0fnu;
    case 21160510: return ScalarType::Half;
    case 21160807: return ScalarType::BFloat16;
    case 21320823: return ScalarType::Float;
    default: {
      uint32_t num_bits = get_dtype_num_bits(dtype_id);
      if (num_bits == 4 || num_bits == 8) return ScalarType::Byte;
      ASSERT_CHECK(false, "invalid dtype_id: ", dtype_id)
    };
  };
};

struct KernelData {
  CUlibrary library;
  CUfunction func;

  uint32_t smem_size;
  uint32_t num_threads;
  uint32_t a_dtype_id;
  uint32_t b_dtype_id;
  uint32_t c_dtype_id;
  uint32_t bs_dtype_id;
  uint32_t problem_shape_n;
  uint32_t problem_shape_k;
  uint32_t block_shape_m;
  uint32_t block_shape_n;
  uint32_t block_shape_k;
  uint32_t pad_shape_n;
  uint32_t pad_shape_k;
  uint32_t num_experts;
  uint32_t input_scale_group_size;
  uint32_t weight_scale_group_size;
  uint32_t weight_scale_group_size_n;
  uint32_t num_ctas_per_sm;
  uint32_t multi_cast_size_a;
  uint32_t multi_cast_size_b;
  uint32_t gemm_type_id;
  uint32_t mma_type_id;

  bool use_stream_k;
  bool is_fp_zero_point;
  bool is_channel_weight_scale;
  bool is_group_weight_scale;
  bool is_block_weight_scale;
  bool is_tensor_weight_scale;
  bool is_channel_weight_scale_2;
  bool is_tensor_weight_scale_2;
  bool has_zero_point;
  bool has_bias;
  bool use_m_major_input_scale;
  bool use_tma_a;
  bool use_tma_as;
  bool use_tma_b;
  bool use_tma_c;
  bool use_tma_bs;
  bool use_tma_bs2;
  bool use_tma_bzp;
  bool use_tma_bias;
  bool use_packed_k_layout;
};

struct KernelLaunchData {
  KernelData kernel_data;
  int64_t num_sms;
};
