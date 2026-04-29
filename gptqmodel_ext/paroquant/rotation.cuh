// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
// SPDX-License-Identifier: Apache-2.0

/******************************************************************************
 * Adapted from https://github.com/z-lab/paroquant
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename scalar_t> struct RotateAccess;

template <> struct RotateAccess<float> {
  template <int CTA_M, int ROW_STRIDE, int GROUP_SIZE, bool USE_SCALE>
  __device__ static void load_group(float *__restrict__ x_grp, const float *__restrict__ x,
                                    const float *__restrict__ scales, const int s, const int h,
                                    const int j, const int g, const int t) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE / 2;
    float scale0 = USE_SCALE ? scales[base0] : float(1);
    float scale1 = USE_SCALE ? scales[base1] : float(1);
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        x_grp[t * ROW_STRIDE + i] = x[row * h + base0] * scale0;
        x_grp[(t + GROUP_SIZE / 2) * ROW_STRIDE + i] = x[row * h + base1] * scale1;
      }
    }
  }

  template <int KROT, int GROUP_SIZE>
  __device__ static void
  load_coeffs(float reg_theta[KROT], int reg_idx[KROT], const int16_t *__restrict__ idx_ij,
              const float *__restrict__ theta, const int h, const int g, const int t) {
#pragma unroll
    for (int r = 0; r < KROT; r++) {
      reg_theta[r] = theta[r * h / 2 + g * GROUP_SIZE / 2 + t];
      reg_idx[r] = *reinterpret_cast<const int *>(idx_ij + r * h + g * GROUP_SIZE + 2 * t);
    }
  }

  template <int CTA_M, int ROW_STRIDE>
  __device__ static void apply_one(float *__restrict__ x_grp, const int ij, const float theta) {
    int16_t i = ij & 0xFFFF, j = ij >> 16;
    float s_, c_;
    __sincosf(theta, &s_, &c_);
#pragma unroll
    for (int m = 0; m < CTA_M; m++) {
      float xi = x_grp[i * ROW_STRIDE + m];
      float xj = x_grp[j * ROW_STRIDE + m];
      x_grp[i * ROW_STRIDE + m] = xi * c_ + xj * s_;
      x_grp[j * ROW_STRIDE + m] = xi * (-s_) + xj * c_;
    }
  }

  template <int CTA_M, int ROW_STRIDE, int GROUP_SIZE>
  __device__ static void store_group(float *__restrict__ out, const float *__restrict__ x_grp,
                                     const int s, const int h, const int j, const int g,
                                     const int t) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE / 2;
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        out[row * h + base0] = x_grp[t * ROW_STRIDE + i];
        out[row * h + base1] = x_grp[(t + GROUP_SIZE / 2) * ROW_STRIDE + i];
      }
    }
  }
};

template <typename HalfT, typename Half2T, typename Traits> struct RotateAccessHalf {
  template <int CTA_M, int ROW_STRIDE, int GROUP_SIZE, bool USE_SCALE>
  __device__ static void load_group(HalfT *__restrict__ x_grp, const HalfT *__restrict__ x,
                                    const HalfT *__restrict__ scales, const int s, const int h,
                                    const int j, const int g, const int t) {
    static_assert((ROW_STRIDE % 2) == 0, "ROW_STRIDE must be even for vectorized half access");
    const int offset = GROUP_SIZE * g + 2 * t;
    HalfT scale_i, scale_j;
    if constexpr (USE_SCALE) {
      Half2T scale_pair = *reinterpret_cast<const Half2T *>(scales + offset);
      scale_i = Traits::low(scale_pair);
      scale_j = Traits::high(scale_pair);
    } else {
      scale_i = Traits::from_float(1.0f);
      scale_j = Traits::from_float(1.0f);
    }

#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        Half2T x2 = *reinterpret_cast<const Half2T *>(x + row * h + offset);
        HalfT lo = Traits::hmul(Traits::low(x2), scale_i);
        HalfT hi = Traits::hmul(Traits::high(x2), scale_j);
        x_grp[(2 * t) * ROW_STRIDE + i] = lo;
        x_grp[(2 * t + 1) * ROW_STRIDE + i] = hi;
      }
    }
  }

  template <int KROT, int GROUP_SIZE>
  __device__ static void
  load_coeffs(float reg_theta[KROT], int reg_idx[KROT], const int16_t *__restrict__ idx_ij,
              const HalfT *__restrict__ theta, const int h, const int g, const int t) {
#pragma unroll
    for (int r = 0; r < KROT; r++) {
      reg_theta[r] = Traits::to_float(theta[r * h / 2 + g * GROUP_SIZE / 2 + t]);
      reg_idx[r] = *reinterpret_cast<const int *>(idx_ij + r * h + g * GROUP_SIZE + 2 * t);
    }
  }

  template <int CTA_M, int ROW_STRIDE>
  __device__ static void apply_one(HalfT *__restrict__ x_grp, const int ij, const float theta) {
    static_assert((ROW_STRIDE % 2) == 0, "ROW_STRIDE must be even for vectorized half access");
    int16_t i = ij & 0xFFFF, j = ij >> 16;
    float s_, c_;
    __sincosf(theta, &s_, &c_);

#pragma unroll
    for (int m = 0; m < CTA_M / 2; ++m) {
      Half2T *pi2 = reinterpret_cast<Half2T *>(x_grp + i * ROW_STRIDE + m * 2);
      Half2T *pj2 = reinterpret_cast<Half2T *>(x_grp + j * ROW_STRIDE + m * 2);

      float2 xi = Traits::to_float2(*pi2);
      float2 xj = Traits::to_float2(*pj2);

      float2 yi, yj;
      yi.x = fmaf(c_, xi.x, s_ * xj.x);
      yi.y = fmaf(c_, xi.y, s_ * xj.y);
      yj.x = fmaf(c_, xj.x, -s_ * xi.x);
      yj.y = fmaf(c_, xj.y, -s_ * xi.y);

      *pi2 = Traits::from_floats(yi.x, yi.y);
      *pj2 = Traits::from_floats(yj.x, yj.y);
    }
  }

  template <int CTA_M, int ROW_STRIDE, int GROUP_SIZE>
  __device__ static void store_group(HalfT *__restrict__ out, const HalfT *__restrict__ x_grp,
                                     const int s, const int h, const int j, const int g,
                                     const int t) {
    static_assert((ROW_STRIDE % 2) == 0, "ROW_STRIDE must be even for vectorized half access");
    const int base = GROUP_SIZE * g + 2 * t;
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        Half2T out2;
        out2.x = x_grp[(2 * t) * ROW_STRIDE + i];
        out2.y = x_grp[(2 * t + 1) * ROW_STRIDE + i];
        *reinterpret_cast<Half2T *>(out + row * h + base) = out2;
      }
    }
  }
};

struct HalfTraits {
  __device__ static float2 to_float2(__half2 v) { return __half22float2(v); }
  __device__ static __half2 from_floats(float a, float b) { return __floats2half2_rn(a, b); }
  __device__ static float to_float(__half v) { return __half2float(v); }
  __device__ static __half from_float(float v) { return __float2half_rn(v); }
  __device__ static __half low(__half2 v) { return __low2half(v); }
  __device__ static __half high(__half2 v) { return __high2half(v); }
  __device__ static __half hmul(__half a, __half b) { return __hmul(a, b); }
};

template <> struct RotateAccess<__half> : RotateAccessHalf<__half, __half2, HalfTraits> {};

struct BFloat16Traits {
  __device__ static float2 to_float2(__nv_bfloat162 v) { return __bfloat1622float2(v); }
  __device__ static __nv_bfloat162 from_floats(float a, float b) {
    return __floats2bfloat162_rn(a, b);
  }
  __device__ static __half2 to_half2(__nv_bfloat162 v) {
    float2 pair = __bfloat1622float2(v);
    return __floats2half2_rn(pair.x, pair.y);
  }
  __device__ static __nv_bfloat162 from_half2(__half2 v) {
    float2 pair = __half22float2(v);
    return __floats2bfloat162_rn(pair.x, pair.y);
  }
  __device__ static float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
  __device__ static __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
  __device__ static __nv_bfloat16 low(__nv_bfloat162 v) { return __low2bfloat16(v); }
  __device__ static __nv_bfloat16 high(__nv_bfloat162 v) { return __high2bfloat16(v); }
  __device__ static __nv_bfloat16 hmul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
};

// Keep BF16 inputs/outputs on the fused path while using the FP16 workspace
// update pattern internally to reduce BF16 round-trip loss across k-rot stages.
struct RotateAccessBFloat16HalfWorkspace {
  template <int CTA_M, int ROW_STRIDE, int GROUP_SIZE, bool USE_SCALE>
  __device__ static void load_group(__half *__restrict__ x_grp, const __nv_bfloat16 *__restrict__ x,
                                    const __nv_bfloat16 *__restrict__ scales, const int s,
                                    const int h, const int j, const int g, const int t) {
    static_assert((ROW_STRIDE % 2) == 0, "ROW_STRIDE must be even for vectorized half access");
    const int offset = GROUP_SIZE * g + 2 * t;
    __half2 scale_pair;
    if constexpr (USE_SCALE) {
      scale_pair = BFloat16Traits::to_half2(*reinterpret_cast<const __nv_bfloat162 *>(scales + offset));
    } else {
      scale_pair = __floats2half2_rn(1.0f, 1.0f);
    }

#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        __half2 x_pair =
            BFloat16Traits::to_half2(*reinterpret_cast<const __nv_bfloat162 *>(x + row * h + offset));
        __half2 prod = __hmul2(x_pair, scale_pair);
        x_grp[(2 * t) * ROW_STRIDE + i] = __low2half(prod);
        x_grp[(2 * t + 1) * ROW_STRIDE + i] = __high2half(prod);
      }
    }
  }

  template <int KROT, int GROUP_SIZE>
  __device__ static void load_coeffs(float reg_theta[KROT], int reg_idx[KROT],
                                     const int16_t *__restrict__ idx_ij,
                                     const __nv_bfloat16 *__restrict__ theta, const int h,
                                     const int g, const int t) {
#pragma unroll
    for (int r = 0; r < KROT; r++) {
      reg_theta[r] = BFloat16Traits::to_float(theta[r * h / 2 + g * GROUP_SIZE / 2 + t]);
      reg_idx[r] = *reinterpret_cast<const int *>(idx_ij + r * h + g * GROUP_SIZE + 2 * t);
    }
  }

  template <int CTA_M, int ROW_STRIDE, int GROUP_SIZE>
  __device__ static void store_group(__nv_bfloat16 *__restrict__ out, const __half *__restrict__ x_grp,
                                     const int s, const int h, const int j, const int g,
                                     const int t) {
    static_assert((ROW_STRIDE % 2) == 0, "ROW_STRIDE must be even for vectorized half access");
    const int base = GROUP_SIZE * g + 2 * t;
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        __half2 out_pair = __halves2half2(x_grp[(2 * t) * ROW_STRIDE + i], x_grp[(2 * t + 1) * ROW_STRIDE + i]);
        *reinterpret_cast<__nv_bfloat162 *>(out + row * h + base) = BFloat16Traits::from_half2(out_pair);
      }
    }
  }
};

template <>
struct RotateAccess<__nv_bfloat16>
    : RotateAccessHalf<__nv_bfloat16, __nv_bfloat162, BFloat16Traits> {};
