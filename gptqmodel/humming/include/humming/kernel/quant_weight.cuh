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


template <class DataType>
CUDA_INLINE float get_data_type_max_num() {
  if constexpr (DataType::kIsIntegerType) {
    uint32_t val = (1 << (DataType::kBits - 1)) - 1;
    return (float)val;
  } else if constexpr (DataType::kExponentBits == 0) {
    // Fixed-point sign-magnitude format (e0mX): {0, +/-1 .. +/-(2^M - 1)}.
    return (float)((1 << DataType::kMantissaBits) - 1);
  } else if constexpr (DataType::kIsFloatingPointType) {
    uint32_t max_val = (1 << (DataType::kBits - DataType::kIsSigned)) - 1;

    if constexpr (std::is_same<DataType, Float8E4M3>::value || std::is_same<DataType, Float8E3M4>::value) {
      // FN format
      max_val = max_val - 1;
    } else if constexpr (std::is_same<DataType, Float8E5M2>::value) {
      // IEEE754 format
      max_val = max_val - (1 << 2);
    }

    constexpr uint32_t mask = (1 << (DataType::kExponentBits + DataType::kMantissaBits)) - 1;
    constexpr uint32_t sign_mask = DataType::kIsSigned ? mask + 1 : 0;

    uint32_t part1 = (max_val & sign_mask) << (31 - (DataType::kExponentBits + DataType::kMantissaBits));
    uint32_t part2 = (max_val & mask) << (23 - DataType::kMantissaBits);
    max_val = part1 | part2;

    constexpr uint32_t exp_offset = 128 - (1 << (DataType::kExponentBits - 1));
    constexpr uint32_t scale_factor = (exp_offset << 23) + 0x3F800000;
    const float scale_factor_float = *reinterpret_cast<const float *>(&scale_factor);

    return *reinterpret_cast<float *>(&max_val) * scale_factor_float;
  }
}


template <class SourceType>
CUDA_INLINE void calculate_buffer_range_value(float &max_val, float &min_val, uint4 buffer) {

  if constexpr (std::is_same<SourceType, Float32>::value) {
    float *buffer_ptr = reinterpret_cast<float *>(&buffer);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      max_val = fmaxf(max_val, buffer_ptr[i]);
      min_val = fminf(min_val, buffer_ptr[i]);
    }
  } else {
    using scalar_t2 = typename F16Conversion<SourceType>::scalar_t2;
    scalar_t2 *buffer_ptr = reinterpret_cast<scalar_t2 *>(&buffer);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      float2 buffer_val = F16Conversion<SourceType>::num22float2(buffer_ptr[i]);
      max_val = fmaxf(max_val, buffer_val.x);
      max_val = fmaxf(max_val, buffer_val.y);

      min_val = fminf(min_val, buffer_val.x);
      min_val = fminf(min_val, buffer_val.y);
    }
  }
};


template <class SourceType, class TargetType, bool kHasZeroPoint>
CUDA_INLINE void quant_buffer(
    float &inv_scale_val, uint4 buffer, uint4 *out_buffer_ptr,
    float max_abs_val, float min_val) {

  constexpr uint32_t num_elements = 128 / SourceType::kBits;
  float vals[num_elements];

  if constexpr (SourceType::kBits == 16) {
    using scalar_t2 = typename F16Conversion<SourceType>::scalar_t2;
    scalar_t2 *buffer_ptr = reinterpret_cast<scalar_t2 *>(&buffer);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      float2 buffer_val = F16Conversion<SourceType>::num22float2(buffer_ptr[i]);
      vals[i * 2] = buffer_val.x;
      vals[i * 2 + 1] = buffer_val.y;
    }
  } else {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      vals[i] = reinterpret_cast<float *>(&buffer)[i];
    }
  }

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < num_elements; i++) {
    if constexpr (TargetType::kIsIntegerType && kHasZeroPoint) {
      int32_t *out_vals = reinterpret_cast<int32_t *>(out_buffer_ptr);
      out_vals[i] = __float2int_rn((vals[i] - min_val) * inv_scale_val);
      out_vals[i] = fmaxf(out_vals[i], 0);
      out_vals[i] = fminf(out_vals[i], (1 << TargetType::kBits) - 1);
    } else if constexpr (TargetType::kIsIntegerType && !kHasZeroPoint) {
      int32_t *out_vals = reinterpret_cast<int32_t *>(out_buffer_ptr);
      int32_t code = __float2int_rn(vals[i] * inv_scale_val);
      if constexpr (!TargetType::kIsSigned) {
        code += 1 << (TargetType::kBits - 1);
        code = min(max(code, 0), (1 << TargetType::kBits) - 1);
      } else {
        code = min(max(code, -(1 << (TargetType::kBits - 1))), (1 << (TargetType::kBits - 1)) - 1);
      }
      out_vals[i] = code;
    } else if constexpr (TargetType::kExponentBits == 0) {
      // Fixed-point sign-magnitude format (e0mX): sign bit at position M, the
      // magnitude (0 .. 2^M - 1) in the low M bits.
      static_assert(TargetType::kIsSigned);
      int32_t code = __float2int_rn(vals[i] * inv_scale_val);
      int32_t mag = min(abs(code), (1 << TargetType::kMantissaBits) - 1);
      uint32_t sign = code < 0 ? 1u : 0u;
      uint32_t *out_vals = reinterpret_cast<uint32_t *>(out_buffer_ptr);
      out_vals[i] = (sign << TargetType::kMantissaBits) | (uint32_t)mag;
    } else {
      static_assert(TargetType::kIsSigned);

      constexpr uint32_t exp_offset = 128 - (1 << (TargetType::kExponentBits - 1));
      constexpr uint32_t scale_factor = (exp_offset << 23) + 0x3F800000;
      const float scale_factor_float = *reinterpret_cast<const float *>(&scale_factor);

      float val = vals[i] * inv_scale_val / scale_factor_float;
      uint32_t val_uint = *reinterpret_cast<uint32_t *>(&val);

      constexpr uint32_t mask = ((1 << (TargetType::kBits - 1)) - 1) << (23 - TargetType::kMantissaBits);
      constexpr uint32_t sign_mask = 0x80000000;
      constexpr uint32_t mask2 = mask | sign_mask;

      uint32_t val_uint_rz = val_uint & mask2 | (val_uint & sign_mask);
      uint32_t val_uint_ru = (val_uint + (1 << (23 - TargetType::kMantissaBits - 1))) & mask2;

      float val_rz = *reinterpret_cast<float *>(&val_uint_rz);
      float val_ru = *reinterpret_cast<float *>(&val_uint_ru);

      float val_rn;
      if (fabsf(val - val_rz) > fabsf(val_ru - val)) {
        val_rn = val_ru;
      } else {
        val_rn = val_rz;
      }

      uint32_t val_uint_rn = *reinterpret_cast<uint32_t *>(&val_rn);

      uint32_t part1 = (val_uint_rn & sign_mask) >> (32 - TargetType::kBits);
      uint32_t part2 = (val_uint_rn & mask) >> (23 - TargetType::kMantissaBits);

      uint32_t *out_vals = reinterpret_cast<uint32_t *>(out_buffer_ptr);
      out_vals[i] = part1 | part2;
    }
  };
};


CUDA_INLINE float warp_reduce_max(float val) {
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
};

CUDA_INLINE float warp_reduce_min(float val) {
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
};


template <
    class SourceType, class TargetType,
    uint32_t kQuantGroupSize, bool kHasScale,
    bool kUseUE8M0Scale, bool kHasZeroPoint, bool kIsFpZeroPoint,
    bool kAllowNegativeScale = true>
__global__ void quant_weight(uint4 *in_ptr, uint4 *out_ptr, uint32_t *out_scale_ptr, uint32_t *zero_point_ptr) {

  static_assert(std::is_same<SourceType, Float16>::value ||
                std::is_same<SourceType, BFloat16>::value ||
                std::is_same<SourceType, Float32>::value);

  __shared__ float smem[2];
  static_assert(kQuantGroupSize % 8 == 0);
  uint32_t num_groups = blockDim.x;

  uint32_t buffer_num_elem = 128 / SourceType::kBits;

  uint4 buffer = {0, 0, 0, 0};
  float max_val = 1e-30;
  float min_val = -1e-30;
  float scale_val;
  float max_abs_val;
  int32_t zero_point;
  float zero_point_float;

  if constexpr (kHasScale) {
    PRAGMA_UNROLL_COUNT(4)
    for (uint32_t i = 0; i < CEIL_DIV(kQuantGroupSize, 32 * buffer_num_elem); i++) {
      uint32_t offset = (32 * i + threadIdx.x) * buffer_num_elem;
      uint32_t gmem_offset = offset + blockIdx.x * kQuantGroupSize;

      if (offset < kQuantGroupSize) {
        buffer = in_ptr[gmem_offset / buffer_num_elem];
      }

      calculate_buffer_range_value<SourceType>(max_val, min_val, buffer);
    }

    max_val = warp_reduce_max(max_val);
    min_val = warp_reduce_min(min_val);

    if (threadIdx.x == 0) {
      smem[0] = max_val;
      smem[1] = min_val;
    }

    __syncthreads();

    max_val = smem[0];
    min_val = smem[1];
    max_abs_val = fmaxf(max_val, fabsf(min_val));

    if constexpr (TargetType::kIsFloatingPointType) {
      scale_val = max_abs_val / get_data_type_max_num<TargetType>();
    } else if constexpr (!kHasZeroPoint) {
      float min_abs_val = fminf(max_val, fabsf(min_val));
      float dtype_max_val = get_data_type_max_num<TargetType>();
      float scale_val1, scale_val2;
      if constexpr (TargetType::kBits == 1) {
        // 1-bit has no positive code (dtype_max_val == 0) and
        // quant_buffer does not clamp: pick a scale that rounds the
        // positive side to code 0 and the negative side to code -1.
        scale_val1 = max_abs_val / (dtype_max_val + 1.499);
        scale_val2 = min_abs_val / (dtype_max_val + 0.499);
      } else {
        scale_val1 = max_abs_val / (dtype_max_val + 1.0);
        scale_val2 = min_abs_val / (dtype_max_val + 0.0);
      }

      scale_val = max(scale_val1, scale_val2);
      if constexpr (kAllowNegativeScale) {
        if (max_val > fabsf(min_val)) scale_val = -scale_val;
      }
    } else {
      scale_val = (max_val - min_val) / (get_data_type_max_num<TargetType>() * 2 + 1);
      zero_point_float = (-min_val) / scale_val;
      if constexpr (!kIsFpZeroPoint) {
        zero_point = __float2int_rn(zero_point_float);
        min_val = (float)zero_point * -scale_val;
      }
    }

    if constexpr (kUseUE8M0Scale) {
      uint32_t scale_val_uint = __float_as_uint(scale_val);
      scale_val_uint = (scale_val_uint + 0x007FFFFF) & 0x7F800000;
      scale_val = __uint_as_float(scale_val_uint);
    }

    if (threadIdx.x == 0) {
      if constexpr (kUseUE8M0Scale) {
        uint32_t scale_val_uint = *reinterpret_cast<uint32_t *>(&scale_val);
        scale_val_uint = (scale_val_uint & 0x7F800000) >> 23;
        uint8_t *out_scale_ptr_uint8 = reinterpret_cast<uint8_t *>(out_scale_ptr);
        out_scale_ptr_uint8[blockIdx.x] = reinterpret_cast<uint8_t *>(&scale_val_uint)[0];
      } else {
        float *out_scale_ptr_float = reinterpret_cast<float *>(out_scale_ptr);
        out_scale_ptr_float[blockIdx.x] = scale_val;
      }

      if constexpr (kHasZeroPoint && !kIsFpZeroPoint) {
        zero_point_ptr[blockIdx.x] = static_cast<uint32_t>(zero_point);
      } else if constexpr (kHasZeroPoint && kIsFpZeroPoint) {
        zero_point_ptr[blockIdx.x] = __float_as_uint(zero_point_float);
      }
    }
  } else {
    scale_val = 1;
  }

  float inv_scale_val = 1 / scale_val;

  PRAGMA_UNROLL_COUNT(4)
  for (uint32_t i = 0; i < CEIL_DIV(kQuantGroupSize, 32 * buffer_num_elem); i++) {
    uint32_t offset = (32 * i + threadIdx.x) * buffer_num_elem;
    uint32_t gmem_offset = offset + blockIdx.x * kQuantGroupSize;

    if (offset < kQuantGroupSize) {
      buffer = in_ptr[gmem_offset / buffer_num_elem];
    }

    uint4 out_buffer[32 / SourceType::kBits];

    quant_buffer<SourceType, TargetType, kHasZeroPoint>(
        inv_scale_val, buffer, out_buffer, max_abs_val, min_val);

    if (offset < kQuantGroupSize) {
      if constexpr (SourceType::kBits == 16) {
        out_ptr[(gmem_offset / buffer_num_elem) * 2] = out_buffer[0];
        out_ptr[(gmem_offset / buffer_num_elem) * 2 + 1] = out_buffer[1];
      } else {
        out_ptr[gmem_offset / buffer_num_elem] = out_buffer[0];
      }
    }
  }
}
