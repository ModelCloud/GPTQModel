// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>


template <class ElementC>
class F16Conversion {};

template <>
class F16Conversion<Float16> {
public:
  using scalar_t = half;
  using scalar_t2 = half2;

  CUDA_INLINE
  static half2 num2num2(half x) {
    return __half2half2(x);
  };

  CUDA_INLINE
  static half2 float2num2(float x) {
    return __float2half2_rn(x);
  };

  CUDA_INLINE
  static half2 float22num2(float2 x) {
    return __float22half2_rn(x);
  };

  CUDA_INLINE
  static half2 floats2num2(float x, float y) {
    return __floats2half2_rn(x, y);
  };

  CUDA_INLINE
  static float2 num22float2(half2 x) {
    return __half22float2(x);
  };
};

template <>
class F16Conversion<BFloat16> {
public:
  using scalar_t = nv_bfloat16;
  using scalar_t2 = nv_bfloat162;

  CUDA_INLINE
  static nv_bfloat162 num2num2(nv_bfloat16 x) {
    return __bfloat162bfloat162(x);
  };

  CUDA_INLINE
  static nv_bfloat162 float2num2(float x) {
    return __float2bfloat162_rn(x);
  };

  CUDA_INLINE
  static nv_bfloat162 float22num2(float2 x) {
    return __float22bfloat162_rn(x);
  };

  CUDA_INLINE
  static nv_bfloat162 floats2num2(float x, float y) {
    return __floats2bfloat162_rn(x, y);
  };

  CUDA_INLINE
  static float2 num22float2(nv_bfloat162 x) {
    return __bfloat1622float2(x);
  };
};

template <class ElementC>
class F8Conversion {};

template <>
class F8Conversion<Float8E4M3> {
public:
  using scalar_t = __nv_fp8_e4m3;
  using scalar_t2 = __nv_fp8x2_e4m3;
  using scalar_t4 = __nv_fp8x4_e4m3;

  CUDA_INLINE
  static float4 num42float4(__nv_fp8x4_e4m3 x) {
    return (float4)x;
  };
};

template <>
class F8Conversion<Float8E5M2> {
public:
  using scalar_t = __nv_fp8_e5m2;
  using scalar_t2 = __nv_fp8x2_e5m2;
  using scalar_t4 = __nv_fp8x4_e5m2;

  CUDA_INLINE
  static float4 num42float4(__nv_fp8x4_e5m2 x) {
    return (float4)x;
  };
};


template <>
class F8Conversion<Float8E8M0> {
public:
  using scalar_t = __nv_fp8_e8m0;
  using scalar_t2 = __nv_fp8x2_e8m0;
  using scalar_t4 = __nv_fp8x4_e8m0;

  CUDA_INLINE
  static float4 num42float4(__nv_fp8x4_e8m0 x) {
    int32_t &x_uint = *reinterpret_cast<int32_t *>(&x);
    int4 res;
    res.x = (x_uint & 0x000000FF) << 23;
    res.y = (x_uint & 0x0000FF00) << 15;
    res.z = (x_uint & 0x00FF0000) << 7;
    res.w = (x_uint & 0xFF000000) >> 1;

    return *reinterpret_cast<float4 *>(&res);
  };
};
