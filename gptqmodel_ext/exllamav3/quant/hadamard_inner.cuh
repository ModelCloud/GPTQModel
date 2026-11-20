#pragma once

// Hadamard transform 128-element vector across one warp, with optional pre and post scales

// The warp shuffles below use __activemask() rather than a hard-coded
// 0xffffffff: HIP requires a 64-bit shuffle mask (a 32-bit literal fails a
// static_assert), while CUDA wants the 32-bit active-lane mask. __activemask()
// is exactly the active-lane set and is the right width on both platforms.

__device__ inline half hreduce(half2 x)
{
    return __hadd(__low2half(x), __high2half(x));
}

__device__ inline void shuffle_had_f4x32(float& h0, float& h1, float& h2, float& h3, const int lane_id)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        uint32_t i0 = __float_as_uint(h0);
        uint32_t i1 = __float_as_uint(h1);
        uint32_t i2 = __float_as_uint(h2);
        uint32_t i3 = __float_as_uint(h3);
        uint64_t h01 =  (uint64_t) i0 | (((uint64_t) i1) << 32);
        uint64_t h23 =  (uint64_t) i2 | (((uint64_t) i3) << 32);
        uint64_t ph01 = __shfl_xor_sync(__activemask(), h01, i);
        uint64_t ph23 = __shfl_xor_sync(__activemask(), h23, i);
        float ph0 = __uint_as_float((uint32_t) (ph01 & 0xffffffff));
        float ph1 = __uint_as_float((uint32_t) (ph01 >> 32));
        float ph2 = __uint_as_float((uint32_t) (ph23 & 0xffffffff));
        float ph3 = __uint_as_float((uint32_t) (ph23 >> 32));
        int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
        i0 ^= sfm & 0x80000000;
        i1 ^= sfm & 0x80000000;
        i2 ^= sfm & 0x80000000;
        i3 ^= sfm & 0x80000000;
        h0 = __uint_as_float(i0) + ph0;
        h1 = __uint_as_float(i1) + ph1;
        h2 = __uint_as_float(i2) + ph2;
        h3 = __uint_as_float(i3) + ph3;
    }
}

__device__ inline void shuffle_had_f2x32(float& v, float& w, const int lane_id)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        uint64_t vw = ((uint64_t) __float_as_uint(v)) | (((uint64_t) __float_as_uint(w)) << 32);
        uint64_t pvw = __shfl_xor_sync(__activemask(), vw, i);
        float pv = __uint_as_float((uint32_t) (pvw & 0xffffffff));
        float pw = __uint_as_float((uint32_t) (pvw >> 32));
        uint32_t vi = __float_as_uint(v);
        uint32_t wi = __float_as_uint(w);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        vi ^= (sfm & 0x80000000);
        wi ^= (sfm & 0x80000000);
        v = __uint_as_float(vi) + pv;
        w = __uint_as_float(wi) + pw;
    }
}

__device__ inline float shuffle_had_fx32(float v, const int lane_id)
{
    for (int i = 1; i < 32; i <<= 1)
    {
        float pv = __shfl_xor_sync(__activemask(), v, i);
        uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        *vi ^= (sfm & 0x80000000);
        v = v + pv;
    }
    return v;
}

__device__ inline half2 shuffle_had_h2x32(half2 v, int lane_id)
{
    for (int i = 1; i < 32; i <<= 1)
    {
        half2 pv = __shfl_xor_sync(__activemask(), v, i);
        uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        *vi ^= (sfm & 0x80008000);
        v = __hadd2(v, pv);
    }
    return v;
}

// Half vector, half scales

inline __device__
void had_hf_r_128_inner
(
    const half* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ pre_scale,
    const half* __restrict__ post_scale,
    const float r_scale
)
{
    int t = threadIdx.x & 31;

    // Load
    half4 v = ((half4*) input_ptr)[t];

    // Pre scale
    if (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) pre_scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }

    // 4 element had
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    // 32 element had, warp shuffle
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    // Post scale
    if (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) post_scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }

    // Store
    ((half4*) output_ptr)[t] = v;
}

// Float vector, half scales

inline __device__
void had_ff_r_128_inner
(
    const float* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    const half* __restrict__ pre_scale,
    const half* __restrict__ post_scale,
    const float r_scale
)
{
    int t = threadIdx.x & 31;

    // Load
    float4 v = ((float4*) input_ptr)[t];

    // Pre scale
    if (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) pre_scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    // 4 element had
    float v0 = v.x;
    float v1 = v.y;
    float v2 = v.z;
    float v3 = v.w;
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;

    // 32 element had, warp shuffle
    shuffle_had_f2x32(v.x, v.y, t);
    shuffle_had_f2x32(v.z, v.w, t);
    v.x *= r_scale;
    v.y *= r_scale;
    v.z *= r_scale;
    v.w *= r_scale;

    // Post scale
    if (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) post_scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    // Store
    ((float4*) output_ptr)[t] = v;
}

// Adapted from ExLlamaV3 v1.6.7 (8c593e4), copyright TurboDerp and contributors, MIT.
template <bool pre_scale, bool post_scale>
inline __device__
void had_hf_r_128_inner
(
    const half* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ scale,
    const float r_scale
)
{
    int t = threadIdx.x & 31;

    // Load
    half4 v = ((half4*) input_ptr)[t];

    // Pre scale
    if constexpr (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }

    // 4 element had
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    // 32 element had, warp shuffle
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    // Post scale
    if constexpr (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }

    // Store
    ((half4*) output_ptr)[t] = v;
}

// Float vector, half scales

template <bool pre_scale, bool post_scale>
inline __device__
void had_ff_r_128_inner
(
    const float* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    const half* __restrict__ scale,
    const float r_scale
)
{
    int t = threadIdx.x & 31;

    // Load
    float4 v = ((float4*) input_ptr)[t];

    // Pre scale
    if constexpr (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    // 4 element had
    float v0 = v.x;
    float v1 = v.y;
    float v2 = v.z;
    float v3 = v.w;
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;

    // 32 element had, warp shuffle
    shuffle_had_f2x32(v.x, v.y, t);
    shuffle_had_f2x32(v.z, v.w, t);
    v.x *= r_scale;
    v.y *= r_scale;
    v.z *= r_scale;
    v.w *= r_scale;

    // Post scale
    if constexpr (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    // Store
    ((float4*) output_ptr)[t] = v;
}

// Float vector, half scales, half output

template <bool pre_scale, bool post_scale>
inline __device__
void had_fh_r_128_inner
(
    const float* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ scale,
    const float r_scale
)
{
    int t = threadIdx.x & 31;

    // Load
    float4 v = ((float4*) input_ptr)[t];

    // Pre scale
    if constexpr (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    // 4 element had
    float v0 = v.x;
    float v1 = v.y;
    float v2 = v.z;
    float v3 = v.w;
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;

    // 32 element had, warp shuffle
    shuffle_had_f2x32(v.x, v.y, t);
    shuffle_had_f2x32(v.z, v.w, t);
    v.x *= r_scale;
    v.y *= r_scale;
    v.z *= r_scale;
    v.w *= r_scale;

    half4 o;
    o.x = __floats2half2_rn(v.x, v.y);
    o.y = __floats2half2_rn(v.z, v.w);

    // Post scale
    if constexpr (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        o.x = __hmul2(o.x, scales.x);
        o.y = __hmul2(o.y, scales.y);
    }

    // Store
    ((half4*) output_ptr)[t] = o;
}

#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
// ---------------------------------------------------------------------------
// Per-row FP16/BF16 input-range guard, folded into the input Hadamard and the
// GEMM epilogue (#5942).
//
// The guard used to be two launches around the GEMM (`bf16_prescale` and
// `bf16_rescale`) plus an FP32 output buffer. The GEMM already reads the
// activation and the checkpoint scales for its input Hadamard, so the bound
// is reduced in that same pass, the rotation applies the row's power-of-two
// scale, and the epilogue applies it back with the bias before its single
// cast. The common case - no row out of range - pays one extra reduction read
// instead of two launches and a wider output.
//
// Every step reproduces the reference operation for operation, so the result
// is bit-identical to `bf16_prescale` + GEMM + `bf16_rescale`:
//   * the bound is max(|x|, |x * suh|), reduced in FP32, or in FP64 for BF16
//     with checkpoint scales whose product can leave FP32's range;
//   * a row is scaled only when its bound is finite and above the FP16-safe
//     amax, by 2**exponent with `exponent = frexp(amax / bound)`; NaN rows
//     keep a scale of 1.0, like the reference;
//   * the input is multiplied by 2**-exponent (exact) and rounded once to FP16;
//   * the epilogue computes `out * 2**exponent + bias` at the scale's
//     precision and rounds once to the output dtype.
// ---------------------------------------------------------------------------

#include <cuda_bf16.h>

__device__ __forceinline__ float exl3_guard_to_float(__half value)
{
    return __half2float(value);
}

__device__ __forceinline__ float exl3_guard_to_float(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

// One 128-element block of one row. `suh` is null when the checkpoint carries
// no input scale, in which case only |x| is bounded (as the reference does).
// Returns a quiet NaN when the row holds one, so the scale decision below keeps
// 1.0 and the NaN reaches the output exactly as before.
template <typename InputT, bool Wide>
__device__ __forceinline__ double exl3_guard_block_max
(
    const InputT* __restrict__ x,
    const half* __restrict__ suh
)
{
    const int lane = threadIdx.x & 31;
    int nan = 0;

    if constexpr (Wide)
    {
        double amax = 0.0;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const double value = static_cast<double>(exl3_guard_to_float(x[lane * 4 + j]));
            nan |= static_cast<int>(isnan(value));
            amax = fmax(amax, fabs(value));
            if (suh)
            {
                const double scaled = __dmul_rn(value, static_cast<double>(__half2float(suh[lane * 4 + j])));
                nan |= static_cast<int>(isnan(scaled));
                amax = fmax(amax, fabs(scaled));
            }
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1)
        {
            amax = fmax(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            nan |= __shfl_xor_sync(0xffffffffu, nan, off);
        }
        return nan ? static_cast<double>(__int_as_float(0x7fc00000)) : amax;
    }
    else
    {
        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const float value = exl3_guard_to_float(x[lane * 4 + j]);
            nan |= static_cast<int>(isnan(value));
            amax = fmaxf(amax, fabsf(value));
            if (suh)
            {
                const float scaled = __fmul_rn(value, __half2float(suh[lane * 4 + j]));
                nan |= static_cast<int>(isnan(scaled));
                amax = fmaxf(amax, fabsf(scaled));
            }
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1)
        {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            nan |= __shfl_xor_sync(0xffffffffu, nan, off);
        }
        return nan ? static_cast<double>(__int_as_float(0x7fc00000)) : static_cast<double>(amax);
    }
}

// Non-negative doubles order the same as their IEEE bit patterns, so the
// hardware 64-bit atomic max over the unsigned image is a double atomic max
// (CUDA has no native double atomicMax). A quiet NaN has the largest pattern,
// which is exactly the "NaN wins the row" rule above. The hardware atomic
// matters: a CAS retry loop serializes badly when many warps of one row race
// on the same slot, which showed up as tens of microseconds on a 32x4096
// prefill.
__device__ __forceinline__ void exl3_guard_atomic_max(double* address, double value)
{
    union
    {
        double d;
        unsigned long long u;
    } bits;
    bits.d = value;
    atomicMax(reinterpret_cast<unsigned long long*>(address), bits.u);
}

// The reference exponent: zero unless the row is finite and above the bound.
template <bool Wide>
__device__ __forceinline__ int exl3_guard_exponent(double amax, double bound)
{
    if (!isfinite(amax) || amax <= bound) return 0;
    int exponent = 0;
    if constexpr (Wide)
        frexp(__ddiv_rn(amax, bound), &exponent);
    else
        frexp(__fdiv_rn(static_cast<float>(amax), static_cast<float>(bound)), &exponent);
    return exponent;
}

// Input Hadamard with the guard folded in: convert the original activation to
// FP16, divide by the row's power of two (exact), apply the checkpoint scales
// in FP16 and rotate. Mirrors `bf16_prescale` followed by
// `had_hf_r_128_inner<true, false>`. `Rotate` is false when the checkpoint has
// no input scales: the kernel then only narrows the activation, as the
// unguarded prologue skips the rotation in that case.
template <typename InputT, bool Wide, bool Rotate>
__device__ __forceinline__ void had_guard_r_128_inner
(
    const InputT* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ pre_scale,
    const double amax,
    const double bound
)
{
    const int t = threadIdx.x & 31;
    const int exponent = exl3_guard_exponent<Wide>(amax, bound);

    half4 v;
    if constexpr (Wide)
    {
        const double inv_scale = ldexp(1.0, -exponent);
        v.x = __floats2half2_rn
        (
            static_cast<float>(__dmul_rn(static_cast<double>(exl3_guard_to_float(input_ptr[t * 4 + 0])), inv_scale)),
            static_cast<float>(__dmul_rn(static_cast<double>(exl3_guard_to_float(input_ptr[t * 4 + 1])), inv_scale))
        );
        v.y = __floats2half2_rn
        (
            static_cast<float>(__dmul_rn(static_cast<double>(exl3_guard_to_float(input_ptr[t * 4 + 2])), inv_scale)),
            static_cast<float>(__dmul_rn(static_cast<double>(exl3_guard_to_float(input_ptr[t * 4 + 3])), inv_scale))
        );
    }
    else
    {
        const float inv_scale = ldexpf(1.0f, -exponent);
        v.x = __floats2half2_rn
        (
            __fmul_rn(exl3_guard_to_float(input_ptr[t * 4 + 0]), inv_scale),
            __fmul_rn(exl3_guard_to_float(input_ptr[t * 4 + 1]), inv_scale)
        );
        v.y = __floats2half2_rn
        (
            __fmul_rn(exl3_guard_to_float(input_ptr[t * 4 + 2]), inv_scale),
            __fmul_rn(exl3_guard_to_float(input_ptr[t * 4 + 3]), inv_scale)
        );
    }

    if constexpr (Rotate)
    {
        if (pre_scale)
        {
            const int i = blockIdx.y * 32 + t;
            const half4 scales = ((half4*) pre_scale)[i];
            v.x = __hmul2(v.x, scales.x);
            v.y = __hmul2(v.y, scales.y);
        }

        float v0 = __half2float(__low2half(v.x));
        float v1 = __half2float(__high2half(v.x));
        float v2 = __half2float(__low2half(v.y));
        float v3 = __half2float(__high2half(v.y));
        float s0 = v0 + v1;
        float d0 = v0 - v1;
        float s1 = v2 + v3;
        float d1 = v2 - v3;
        float h0 = s0 + s1;
        float h1 = d0 + d1;
        float h2 = s0 - s1;
        float h3 = d0 - d1;

        // Same 1/sqrt(128) the unguarded prologue passes to
        // `had_hf_r_128_inner<true, false>`.
        constexpr float had_scale = 0.088388347648f;
        shuffle_had_f4x32(h0, h1, h2, h3, t);
        v.x = __floats2half2_rn(h0 * had_scale, h1 * had_scale);
        v.y = __floats2half2_rn(h2 * had_scale, h3 * had_scale);
    }

    ((half4*) output_ptr)[t] = v;
}

// FP32 Hadamard with the output checkpoint scale, returning the four values
// instead of storing them: the guarded epilogue adds the row scale and the bias
// before its single cast. Same arithmetic as `had_ff_r_128_inner<false, true>`.
template <bool Rotate>
__device__ __forceinline__ void had_ff_r_128_values
(
    const float* __restrict__ input_ptr,
    float (&out)[4],
    const half* __restrict__ post_scale,
    const float r_scale
)
{
    const int t = threadIdx.x & 31;
    float4 v = ((float4*) input_ptr)[t];

    if constexpr (!Rotate)
    {
        out[0] = v.x;
        out[1] = v.y;
        out[2] = v.z;
        out[3] = v.w;
        return;
    }

    float v0 = v.x;
    float v1 = v.y;
    float v2 = v.z;
    float v3 = v.w;
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;

    shuffle_had_f2x32(v.x, v.y, t);
    shuffle_had_f2x32(v.z, v.w, t);
    v.x *= r_scale;
    v.y *= r_scale;
    v.z *= r_scale;
    v.w *= r_scale;

    if (post_scale)
    {
        const int i = blockIdx.y * 32 + t;
        const half4 scales = ((half4*) post_scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.w;
}


// Checkpoint bias element in the epilogue's precision. The branch is uniform
// across the warp and the compiler hoists it out of the four-value store.
__device__ __forceinline__ double exl3_guard_bias_at(const void* bias, int mode, int i)
{
    switch (mode)
    {
        case 1: return static_cast<double>(__half2float(static_cast<const __half*>(bias)[i]));
        case 2: return static_cast<double>(__bfloat162float(static_cast<const __nv_bfloat16*>(bias)[i]));
        case 3: return static_cast<double>(static_cast<const float*>(bias)[i]);
        default: return 0.0;
    }
}

// One RTNE rounding from the epilogue's working precision to the output dtype.
__device__ __forceinline__ __half exl3_guard_cast(double value)
{
    return __float2half_rn(static_cast<float>(value));
}

__device__ __forceinline__ __half exl3_guard_cast(float value)
{
    return __float2half_rn(value);
}

__device__ __forceinline__ __nv_bfloat16 exl3_guard_cast_bf16(double value)
{
    return __float2bfloat16_rn(static_cast<float>(value));
}

__device__ __forceinline__ __nv_bfloat16 exl3_guard_cast_bf16(float value)
{
    return __float2bfloat16_rn(value);
}

// Epilogue store: the row's power of two and the checkpoint bias are applied to
// the FP32 GEMM result and the value is rounded once to the output dtype.
// `Wide` is the FP64 path BF16 takes when checkpoint scales can push the
// product past FP32.
template <bool Wide, bool Bf16>
__device__ __forceinline__ void exl3_guard_store
(
    const float (&values)[4],
    void* __restrict__ out_ptr,
    const double amax,
    const double bound,
    const void* __restrict__ bias,
    const int bias_mode,
    const int bias_index
)
{
    const int exponent = exl3_guard_exponent<Wide>(amax, bound);
    if constexpr (Wide)
    {
        const double scale = ldexp(1.0, exponent);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            double value = __dmul_rn(static_cast<double>(values[j]), scale);
            if (bias)
                value = __dadd_rn(value, exl3_guard_bias_at(bias, bias_mode, bias_index + j));
            if constexpr (Bf16)
                static_cast<__nv_bfloat16*>(out_ptr)[j] = exl3_guard_cast_bf16(value);
            else
                static_cast<__half*>(out_ptr)[j] = exl3_guard_cast(value);
        }
    }
    else
    {
        const float scale = ldexpf(1.0f, exponent);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float value = __fmul_rn(values[j], scale);
            if (bias)
                value = __fadd_rn(value, static_cast<float>(exl3_guard_bias_at(bias, bias_mode, bias_index + j)));
            if constexpr (Bf16)
                static_cast<__nv_bfloat16*>(out_ptr)[j] = exl3_guard_cast_bf16(value);
            else
                static_cast<__half*>(out_ptr)[j] = exl3_guard_cast(value);
        }
    }
}
// Folded input-range guard (#5942): zero the per-row bound, reduce it over the
// activation, then narrow and rotate with each row's power-of-two scale.
// `Wide` selects the FP64 bound BF16 needs when checkpoint scales can push
// `x * suh` past FP32; `Rotate` is false for a checkpoint without input scales,
// where the unguarded prologue also leaves the activation unrotated.
template <bool Wide, typename InputT, bool Rotate, typename Grid>
__device__ __forceinline__ void exl3_guard_input
(
    const InputT* __restrict__ A,
    half* __restrict__ A_had,
    const half* __restrict__ suh,
    double* __restrict__ row_amax,
    const int size_m,
    const int size_k,
    const double bound,
    Grid grid
)
{
    // The bound buffer is allocated per call and zeroed here rather than by a
    // separate fill launch, so the guard adds no launch of its own.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size_m; i += gridDim.x * blockDim.x)
        row_amax[i] = 0.0;
    grid.sync();

    const int total_warps = size_m * size_k / 128;
    const int warps_grid = gridDim.x * blockDim.x / 32;
    const int first_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

    for (int warp = first_warp; warp < total_warps; warp += warps_grid)
    {
        const int row = (warp * 128) / size_k;
        const half* block_suh = suh ? suh + (warp * 128) % size_k : nullptr;
        const double block_max = exl3_guard_block_max<InputT, Wide>(A + warp * 128, block_suh);
        // `block_max` is warp-uniform after the shuffle reduction; one lane
        // publishes it so the row slot sees one atomic per 128-element block,
        // not 32.
        if ((threadIdx.x & 31) == 0)
            exl3_guard_atomic_max(row_amax + row, block_max);
    }
    grid.sync();

    for (int warp = first_warp; warp < total_warps; warp += warps_grid)
    {
        const int row = (warp * 128) / size_k;
        const half* block_suh = suh ? suh + (warp * 128) % size_k : nullptr;
        had_guard_r_128_inner<InputT, Wide, Rotate>
        (
            A + warp * 128,
            A_had + warp * 128,
            block_suh,
            row_amax[row],
            bound
        );
    }
    grid.sync();
}

#endif  // !__HIP_PLATFORM_AMD__ && !USE_ROCM
