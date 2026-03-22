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
  template <int CTA_M, int GROUP_SIZE, bool USE_SCALE>
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
        x_grp[t * CTA_M + i] = x[row * h + base0] * scale0;
        x_grp[(t + GROUP_SIZE / 2) * CTA_M + i] = x[row * h + base1] * scale1;
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

  template <int CTA_M>
  __device__ static void apply_one(float *__restrict__ x_grp, const int ij, const float theta) {
    int16_t i = ij & 0xFFFF, j = ij >> 16;
    float s_, c_;
    __sincosf(theta, &s_, &c_);
#pragma unroll
    for (int m = 0; m < CTA_M; m++) {
      float xi = x_grp[i * CTA_M + m];
      float xj = x_grp[j * CTA_M + m];
      x_grp[i * CTA_M + m] = xi * c_ + xj * s_;
      x_grp[j * CTA_M + m] = xi * (-s_) + xj * c_;
    }
  }

  template <int CTA_M, int GROUP_SIZE>
  __device__ static void store_group(float *__restrict__ out, const float *__restrict__ x_grp,
                                     const int s, const int h, const int j, const int g,
                                     const int t) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE / 2;
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        out[row * h + base0] = x_grp[t * CTA_M + i];
        out[row * h + base1] = x_grp[(t + GROUP_SIZE / 2) * CTA_M + i];
      }
    }
  }
};

template <typename HalfT, typename Half2T, typename Traits> struct RotateAccessHalf {
  template <int CTA_M, int GROUP_SIZE, bool USE_SCALE>
  __device__ static void load_group(HalfT *__restrict__ x_grp, const HalfT *__restrict__ x,
                                    const HalfT *__restrict__ scales, const int s, const int h,
                                    const int j, const int g, const int t) {
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
        x_grp[(2 * t) * CTA_M + i] = lo;
        x_grp[(2 * t + 1) * CTA_M + i] = hi;
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

  template <int CTA_M>
  __device__ static void apply_one(HalfT *__restrict__ x_grp, const int ij, const float theta) {
    int16_t i = ij & 0xFFFF, j = ij >> 16;
    float s_, c_;
    __sincosf(theta, &s_, &c_);

#pragma unroll
    for (int m = 0; m < CTA_M / 2; ++m) {
      Half2T *pi2 = reinterpret_cast<Half2T *>(x_grp + i * CTA_M + m * 2);
      Half2T *pj2 = reinterpret_cast<Half2T *>(x_grp + j * CTA_M + m * 2);

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

  template <int CTA_M, int GROUP_SIZE>
  __device__ static void store_group(HalfT *__restrict__ out, const HalfT *__restrict__ x_grp,
                                     const int s, const int h, const int j, const int g,
                                     const int t) {
    const int base = GROUP_SIZE * g + 2 * t;
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        Half2T out2;
        out2.x = x_grp[(2 * t) * CTA_M + i];
        out2.y = x_grp[(2 * t + 1) * CTA_M + i];
        *reinterpret_cast<Half2T *>(out + row * h + base) = out2;
      }
    }
  }
};

struct HalfTraits {
  __device__ static float2 to_float2(__half2 v) { return __half22float2(v); }
  __device__ static __half2 from_floats(float a, float b) { return __floats2half2_rn(a, b); }
  __device__ static float to_float(__half v) { return static_cast<float>(v); }
  __device__ static __half from_float(float v) { return __half(v); }
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
  __device__ static float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
  __device__ static __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
  __device__ static __nv_bfloat16 low(__nv_bfloat162 v) { return __low2bfloat16(v); }
  __device__ static __nv_bfloat16 high(__nv_bfloat162 v) { return __high2bfloat16(v); }
  __device__ static __nv_bfloat16 hmul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
};

template <>
struct RotateAccess<__nv_bfloat16>
    : RotateAccessHalf<__nv_bfloat16, __nv_bfloat162, BFloat16Traits> {};
