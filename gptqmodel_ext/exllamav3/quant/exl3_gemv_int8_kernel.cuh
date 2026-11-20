// Adapted from ExLlamaV3 v1.6.7 (8c593e4), copyright TurboDerp and contributors, MIT.
#pragma once

#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)

// Device side of the fused int8-activation GEMV (see comment below). Kernels are instantiated in
// comp_units/exl3_gemv_int8_inst_*.cu; the host code in exl3_gemv_int8.cu includes this header only
// for the launch-geometry constants and __host__ __device__ helpers

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cooperative_groups.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"

namespace cg_gemv = cooperative_groups;

// Fused int8-activation GEMV for mul1 (cb 2) tensors, single cooperative launch.
//
// The mul1 codebook value is affine in the byte sum of x * 0x83DCD12D, so with activations quantized
// to int8, dp4a(x * M, splat(a_int8), acc) evaluates codebook(x) * a in a single instruction with
// exact int32 accumulation, and
//
//   y[n] = k_inv * (q * acc1[n] + q2 * acc2[n]) + (1024 * k_inv + k_bias) * (q * sum1 + q2 * sum2)
//
// recovers the result. In residual mode (EXL3_INT8_GEMV=1) the int8 rounding error r = a - q*i, which
// has range +/-q/2 by construction, is quantized with the fixed scale q2 = q/254 and accumulated by a
// second dp4a sharing the decoded products, for ~15-16 bit effective activation precision (KL at
// parity with the fp16 kernel or better). EXL3_INT8_GEMV=2 is the plain int8 mode.
//
// One cooperative launch with the same argument list as exl3_gemm_kernel; `locks` doubles as
// workspace: [2*size_n int32 accumulators][{float q, int s1, int s2, pad} per row][partial max per
// block]. Phases (persistent 256-thread blocks):
//   1a. input Hadamard A -> A_had (as in exl3_gemm_kernel) + accumulator zeroing; for m == 1 also a
//       per-block partial max of the transformed activations
//   1b. (m > 1 only) per-row scale and exact int sums
//   per row:
//     2. GEMV over (256-column x k-slice) units; activation splats are quantized INLINE while
//        staging to shared memory (for m == 1, every block re-derives q from the phase-1a partial
//        maxes - fmax is order-independent so this is deterministic - and block 0 computes the exact
//        epilogue sums concurrently); grid.sync
//     3. epilogue: affine correction + output Hadamard + svh + accumulator reset, warp-striped
//
// K == 4 uses the wide unit: a warp owns an ADJACENT BLOCK PAIR; lane l's uint2 = words {2m, 2m+1} of
// the contiguous 64-word region are exactly the primary words of runs t = 8*(2m), 8*(2m+1), which
// scatter to the SAME two n values (accumulator count unchanged, splats merge into two uint4 loads,
// single-stage butterfly, boundary word via one shuffle). 256 B contiguous per warp per block row
// with two-row prefetch. Other K use a generic narrow unit (two blocks per warp, pointer-based
// extraction). Natural register allocation (2-3 resident blocks/SM) measures faster than forcing
// higher occupancy: with wide loads the per-warp ILP is worth more than the extra warps.

#define NUM_THREADS 256

// ---------------------------------------------------------------------------------------------------------
// Extract 8 packed 16-bit trellis windows for positions t0..t0+7 as masked uint32s. Index math copied
// from the corresponding paths in exl3_dq.cuh (dq8_aligned_*, dq8, dq4, dq2x2).

__device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c)
{
#if defined(USE_ROCM)
    return exl3_dp4a_us(a, b, c);
#else
    int d;
    asm ("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#endif
}

// i0/i2 land in [0, 2*words); a compare+subtract replaces the modulo (words is not a power of two for
// odd bit widths)
template <int bits, bool HALF = false>
__device__ __forceinline__ int wrap_idx(int i)
{
    constexpr int words = HALF ? 4 * (2 * bits + 1) : bits * 256 / 32;
    return i >= words ? i - words : i;
}

// Half-integer rate bits + 0.5: same two-group extraction as dq8_half (exl3_dq.cuh), 18 + 3 * bits bits per group
template <int KA>
__device__ __forceinline__ void ext8w_half
(
    const uint32_t* ptr, int t0,
    uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3,
    uint32_t& w4, uint32_t& w5, uint32_t& w6, uint32_t& w7
)
{
    constexpr int bits2 = 2 * KA + 1;
    constexpr int gspan = 18 + 3 * KA;
    const int e7 = ((t0 >> 1) + 4) * bits2 + 128 * bits2;
    const int e3 = e7 - 2 * bits2;
    {
        int hi = (e7 - 1) / 32, lo = (e7 - gspan) / 32, s = (hi + 1) * 32 - e7;
        uint32_t a = ptr[wrap_idx<KA, true>(lo)], b = ptr[wrap_idx<KA, true>(hi)];
        w7 = fshift(b, a, s); w6 = w7 >> (KA + 1); w5 = w6 >> KA; w4 = w5 >> (KA + 1);
    }
    {
        int hi = (e3 - 1) / 32, lo = (e3 - gspan) / 32, s = (hi + 1) * 32 - e3;
        uint32_t a = ptr[wrap_idx<KA, true>(lo)], b = ptr[wrap_idx<KA, true>(hi)];
        w3 = fshift(b, a, s); w2 = w3 >> (KA + 1); w1 = w2 >> KA; w0 = w1 >> (KA + 1);
    }
    w7 &= 0xffff; w6 &= 0xffff; w5 &= 0xffff; w4 &= 0xffff;
    w3 &= 0xffff; w2 &= 0xffff; w1 &= 0xffff; w0 &= 0xffff;
}

template <int bits>
__device__ __forceinline__ void ext4w(const uint32_t* ptr, int t0, uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3)
{
    int b0 = (t0 + 257) * bits - 16;
    int b2 = b0 + 3 * bits + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[wrap_idx<bits>(i0)];
    uint32_t b = ptr[wrap_idx<bits>(i2)];
    w3 = fshift(b, a, s2) & 0xffff;
    w2 = fshift(b, a, s2 + bits) & 0xffff;
    w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
    w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void ext2w(const uint32_t* ptr, int t0, uint32_t& w0, uint32_t& w1)
{
    int b0 = (t0 + 257) * bits - 16;
    int b2 = b0 + bits + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[wrap_idx<bits>(i0)];
    uint32_t b = ptr[wrap_idx<bits>(i2)];
    w1 = fshift(b, a, s2) & 0xffff;
    w0 = fshift(b, a, s2 + bits) & 0xffff;
}

template <int bits, bool HALF = false>
__device__ __forceinline__ void ext8w
(
    const uint32_t* ptr, int t0,
    uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3,
    uint32_t& w4, uint32_t& w5, uint32_t& w6, uint32_t& w7
)
{
    if constexpr (HALF)
    {
        ext8w_half<bits>(ptr, t0, w0, w1, w2, w3, w4, w5, w6, w7);
    }
    else if constexpr (bits == 1)
    {
        uint32_t i1 = t0 >> 5;
        uint32_t i0 = (i1 + 7) & 7;
        uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = fshift(b, a, ((~t0) & 24));
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 1);
        BFE16_IMM(w5, b, 2);
        BFE16_IMM(w4, b, 3);
        BFE16_IMM(w3, b, 4);
        BFE16_IMM(w2, b, 5);
        BFE16_IMM(w1, b, 6);
        BFE16_IMM(w0, b, 7);
    }
    else if constexpr (bits == 2)
    {
        uint32_t i1 = t0 >> 4;
        uint32_t i0 = (i1 + 15) & 15;
        uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = fshift(b, a, ((~t0) & 8) << 1);
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 2);
        BFE16_IMM(w5, b, 4);
        BFE16_IMM(w4, b, 6);
        BFE16_IMM(w3, b, 8);
        BFE16_IMM(w2, b, 10);
        BFE16_IMM(w1, b, 12);
        BFE16_IMM(w0, b, 14);
    }
    else if constexpr (bits == 3)
    {
        int b1 = (t0 + 257) * bits;
        int b0 = b1 - 16;
        int b2 = b1 + bits * 7;
        int i0 = b0 / 32;
        int i2 = (b2 - 1) / 32;
        int s2 = (i2 + 1) * 32 - b2;
        uint32_t a = ptr[wrap_idx<bits>(i0)];
        uint32_t b = ptr[wrap_idx<bits>(i2)];
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
        w7 &= 0xffff; w6 &= 0xffff; w5 &= 0xffff; w4 &= 0xffff;
        w3 &= 0xffff; w2 &= 0xffff; w1 &= 0xffff; w0 &= 0xffff;
    }
    else if constexpr (bits == 4)
    {
        uint32_t i1 = t0 >> 3;
        uint32_t i0 = (i1 + 31) & 31;
        uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        uint32_t s;
        FSHF_IMM(s, b, a, 20);
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 4);
        BFE16_IMM(w5, b, 8);
        BFE16_IMM(w4, b, 12);
        BFE16_IMM(w3, b, 16);
        w2 = s & 0xffff;
        BFE16_IMM(w1, s, 4);
        BFE16_IMM(w0, s, 8);
    }
    else if constexpr (bits == 7)
    {
        ext2w<bits>(ptr, t0,     w0, w1);
        ext2w<bits>(ptr, t0 + 2, w2, w3);
        ext2w<bits>(ptr, t0 + 4, w4, w5);
        ext2w<bits>(ptr, t0 + 6, w6, w7);
    }
    else  // 5, 6, 8
    {
        ext4w<bits>(ptr, t0,     w0, w1, w2, w3);
        ext4w<bits>(ptr, t0 + 4, w4, w5, w6, w7);
    }
}

// 4bpw extraction from two already-loaded words (same window order)
__device__ __forceinline__ void extract8_4bits_words(uint32_t a, uint32_t b,
    uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3,
    uint32_t& w4, uint32_t& w5, uint32_t& w6, uint32_t& w7)
{
    uint32_t s;
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
}

// ---------------------------------------------------------------------------------------------------------
// Device building blocks (also reusable from batched mgemm/MoE-style kernels)

// Quantize one k-slice of the (Hadamard-transformed) activation row into int8 byte-splats in shared
// memory, plus optional error-feedback residual splats
template <bool residual>
__device__ __forceinline__ void gemv_int8_stage_splats
(
    const half* __restrict__ A_had_row,
    float q,
    uint32_t* __restrict__ sh_as,
    uint32_t* __restrict__ sh_as2,
    int kb0,
    int nrows
)
{
    __syncthreads();
    float rq = 1.0f / q;
    float rq2 = rq * 254.0f;
    for (int i = threadIdx.x; i < nrows * 16; i += NUM_THREADS)
    {
        float a = __half2float(A_had_row[(kb0 << 4) + i]);
        int v = __float2int_rn(a * rq);
        v = max(-127, min(127, v));
        sh_as[i] = ((uint32_t)(uint8_t)(int8_t) v) * 0x01010101u;
        if constexpr (residual)
        {
            float r = a - q * (float) v;
            int v2 = __float2int_rn(r * rq2);
            v2 = max(-127, min(127, v2));
            sh_as2[i] = ((uint32_t)(uint8_t)(int8_t) v2) * 0x01010101u;
        }
    }
    __syncthreads();
}

// Block-wide: exact int8(+residual) sums of one activation row, writes {q, s1, s2} to qsums_row
template <bool residual>
__device__ __forceinline__ void gemv_int8_row_sums
(
    const half* __restrict__ Ar,
    float q,
    int size_k,
    float* __restrict__ qsums_row
)
{
    __shared__ int sh_s1[32];
    __shared__ int sh_s2[32];
    int t = threadIdx.x;
    float rq = 1.0f / q;
    float rq2 = rq * 254.0f;
    int s1 = 0, s2 = 0;
    for (int i = t; i < size_k; i += NUM_THREADS)
    {
        float a = __half2float(Ar[i]);
        int v = __float2int_rn(a * rq);
        v = max(-127, min(127, v));
        s1 += v;
        if constexpr (residual)
        {
            float r = a - q * (float) v;
            int v2 = __float2int_rn(r * rq2);
            s2 += max(-127, min(127, v2));
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        s1 += __shfl_xor_sync(0xffffffff, s1, o);
        s2 += __shfl_xor_sync(0xffffffff, s2, o);
    }
    if ((t & 31) == 0) { sh_s1[t >> 5] = s1; sh_s2[t >> 5] = s2; }
    __syncthreads();
    if (t < 32)
    {
        int v1 = t < (NUM_THREADS >> 5) ? sh_s1[t] : 0;
        int v2 = t < (NUM_THREADS >> 5) ? sh_s2[t] : 0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            v1 += __shfl_xor_sync(0xffffffff, v1, o);
            v2 += __shfl_xor_sync(0xffffffff, v2, o);
        }
        if (t == 0)
        {
            qsums_row[0] = q;
            ((int*) qsums_row)[1] = v1;
            ((int*) qsums_row)[2] = v2;
        }
    }
    __syncthreads();
}

// The K cap (fall back to the regular kernel in the DRAM-bound regime) is per-arch; see
// exl3_gemv_int8_max_k in exl3_gemv_int8.cu
#define GEMV_STAGE_D 4      // cp.async pipeline depth (rows) for the smem-staged unit
#define GEMV_STAGE_MAX_BYTES (8 * GEMV_STAGE_D * 16 * 8 * 4)   // 8 warps, K = 8

// Per-slice-scale ("sq") kernel parameters
#define SQ_KSPLIT_CAP 64
#define SQ_MINROWS 16       // minimum slice height (staging overhead amortization)
#define SQ_ROWS_MAX 512     // shared memory cap
#define SQ_COUNTERS_CAP 4096                                    // fixed counter region: size_n up to 1M
#define SQ_WS_RESERVED (SQ_COUNTERS_CAP + 4 * SQ_KSPLIT_CAP * 4)    // sq-private region at workspace start (counters + per-slice-per-row scales, m <= 4)

// Wide unit (4 bpw): one (256-column x k-slice) unit, warp per adjacent block pair, uint2 loads.
// M activation rows share the decoded weights: extraction and the B stream are amortized, each row
// only adds its dp4a chains against its own splat region (sh_as + r * slice_stride; residual splats
// at (M + r) * slice_stride - at M = 1 this is exactly the coop kernel's sh_as/sh_as2 layout). Row
// accumulators go to accs + r * acc_stride, atomically or with plain stores (exclusive per-slice
// partials of the sq kernel).
template <int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_unit_wide
(
    const uint16_t* __restrict__ B,
    int* __restrict__ accs,
    size_t acc_stride,
    const uint32_t* __restrict__ sh_as,
    int slice_stride,
    int nb256,
    int kb0,
    int nrows,
    int size_n
)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n * 2;                              // u32 per block row at 4 bpw
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * 64 + 2 * lane;
    int c2 = (lane & 1) ? 4 : 0;
    int shfl_src = (lane & 16) | ((lane + 15) & 15);

    int iacc0[M] = {}, iacc1[M] = {}, jacc0[M] = {}, jacc1[M] = {};
    uint2 r0 = *(const uint2*) bp;
    uint2 r1 = {};
    if (nrows > 1) r1 = *(const uint2*) (bp + row_stride);

    for (int kb = 0; kb < nrows; ++kb)
    {
        uint2 r2 = {};
        if (kb + 2 < nrows) r2 = *(const uint2*) (bp + (size_t) (kb + 2) * row_stride);
        uint32_t prev = __shfl_sync(0xffffffff, r0.y, shfl_src);

        uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
        uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
        extract8_4bits_words(prev, r0.x, w0, w1, w2, w3, w4, w5, w6, w7);   // run t = 8*(2m)
        extract8_4bits_words(r0.x, r0.y, v0, v1, v2, v3, v4, v5, v6, v7);   // run t = 8*(2m+1)
        w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
        w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
        v0 *= 0x83DCD12Du; v1 *= 0x83DCD12Du; v2 *= 0x83DCD12Du; v3 *= 0x83DCD12Du;
        v4 *= 0x83DCD12Du; v5 *= 0x83DCD12Du; v6 *= 0x83DCD12Du; v7 *= 0x83DCD12Du;

        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            const uint32_t* as = sh_as + r * slice_stride + (kb << 4);
            uint4 as0 = *(const uint4*) (as + c2);
            uint4 as8 = *(const uint4*) (as + c2 + 8);
            iacc0[r] = dp4a_us(w0, as0.x, iacc0[r]);
            iacc0[r] = dp4a_us(w1, as0.y, iacc0[r]);
            iacc0[r] = dp4a_us(w2, as8.x, iacc0[r]);
            iacc0[r] = dp4a_us(w3, as8.y, iacc0[r]);
            iacc1[r] = dp4a_us(w4, as0.x, iacc1[r]);
            iacc1[r] = dp4a_us(w5, as0.y, iacc1[r]);
            iacc1[r] = dp4a_us(w6, as8.x, iacc1[r]);
            iacc1[r] = dp4a_us(w7, as8.y, iacc1[r]);
            iacc0[r] = dp4a_us(v0, as0.z, iacc0[r]);
            iacc0[r] = dp4a_us(v1, as0.w, iacc0[r]);
            iacc0[r] = dp4a_us(v2, as8.z, iacc0[r]);
            iacc0[r] = dp4a_us(v3, as8.w, iacc0[r]);
            iacc1[r] = dp4a_us(v4, as0.z, iacc1[r]);
            iacc1[r] = dp4a_us(v5, as0.w, iacc1[r]);
            iacc1[r] = dp4a_us(v6, as8.z, iacc1[r]);
            iacc1[r] = dp4a_us(v7, as8.w, iacc1[r]);
            if constexpr (residual)
            {
                const uint32_t* as2 = sh_as + (M + r) * slice_stride + (kb << 4);
                uint4 bs0 = *(const uint4*) (as2 + c2);
                uint4 bs8 = *(const uint4*) (as2 + c2 + 8);
                jacc0[r] = dp4a_us(w0, bs0.x, jacc0[r]);
                jacc0[r] = dp4a_us(w1, bs0.y, jacc0[r]);
                jacc0[r] = dp4a_us(w2, bs8.x, jacc0[r]);
                jacc0[r] = dp4a_us(w3, bs8.y, jacc0[r]);
                jacc1[r] = dp4a_us(w4, bs0.x, jacc1[r]);
                jacc1[r] = dp4a_us(w5, bs0.y, jacc1[r]);
                jacc1[r] = dp4a_us(w6, bs8.x, jacc1[r]);
                jacc1[r] = dp4a_us(w7, bs8.y, jacc1[r]);
                jacc0[r] = dp4a_us(v0, bs0.z, jacc0[r]);
                jacc0[r] = dp4a_us(v1, bs0.w, jacc0[r]);
                jacc0[r] = dp4a_us(v2, bs8.z, jacc0[r]);
                jacc0[r] = dp4a_us(v3, bs8.w, jacc0[r]);
                jacc1[r] = dp4a_us(v4, bs0.z, jacc1[r]);
                jacc1[r] = dp4a_us(v5, bs0.w, jacc1[r]);
                jacc1[r] = dp4a_us(v6, bs8.z, jacc1[r]);
                jacc1[r] = dp4a_us(v7, bs8.w, jacc1[r]);
            }
        }

        r0 = r1;
        r1 = r2;
    }

    // Lanes l, l^1 share both n values
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        iacc0[r] += __shfl_xor_sync(0xffffffff, iacc0[r], 1);
        iacc1[r] += __shfl_xor_sync(0xffffffff, iacc1[r], 1);
        if constexpr (residual)
        {
            jacc0[r] += __shfl_xor_sync(0xffffffff, jacc0[r], 1);
            jacc1[r] += __shfl_xor_sync(0xffffffff, jacc1[r], 1);
        }
    }
    if (!(lane & 1))
    {
        int nb = nbp * 2 + (lane >> 4);
        int n0 = nb * 16 + ((lane & 15) >> 1);
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            int* acc = accs + r * acc_stride;
            if constexpr (atomic)
            {
                atomicAdd(acc + n0, iacc0[r]);
                atomicAdd(acc + n0 + 8, iacc1[r]);
                if constexpr (residual)
                {
                    atomicAdd(acc + size_n + n0, jacc0[r]);
                    atomicAdd(acc + size_n + n0 + 8, jacc1[r]);
                }
            }
            else
            {
                acc[n0] = iacc0[r];
                acc[n0 + 8] = iacc1[r];
                if constexpr (residual)
                {
                    acc[size_n + n0] = jacc0[r];
                    acc[size_n + n0 + 8] = jacc1[r];
                }
            }
        }
    }
}

// One k-row of an adjacent block pair, generic K: extract + dp4a for both blocks from block pointers
// (global or shared memory), M rows sharing the extraction
template <int bits, int M, bool residual, bool HALF = false>
__device__ __forceinline__ void gemv_int8_pair_row
(
    const uint32_t* blockA, const uint32_t* blockB,
    const uint32_t* as_kb,             // sh_as + (kb << 4)
    int slice_stride, int c2, int t0,
    int* ia0, int* ia1, int* ib0, int* ib1,
    int* ja0, int* ja1, int* jb0, int* jb1
)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    ext8w<bits, HALF>(blockA, t0, w0, w1, w2, w3, w4, w5, w6, w7);
    w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
    w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        const uint32_t* as = as_kb + r * slice_stride;
        uint2 as01 = *(const uint2*) (as + c2);
        uint2 as89 = *(const uint2*) (as + c2 + 8);
        ia0[r] = dp4a_us(w0, as01.x, ia0[r]);
        ia0[r] = dp4a_us(w1, as01.y, ia0[r]);
        ia0[r] = dp4a_us(w2, as89.x, ia0[r]);
        ia0[r] = dp4a_us(w3, as89.y, ia0[r]);
        ia1[r] = dp4a_us(w4, as01.x, ia1[r]);
        ia1[r] = dp4a_us(w5, as01.y, ia1[r]);
        ia1[r] = dp4a_us(w6, as89.x, ia1[r]);
        ia1[r] = dp4a_us(w7, as89.y, ia1[r]);
        if constexpr (residual)
        {
            const uint32_t* as2 = as_kb + (M + r) * slice_stride;
            uint2 bs01 = *(const uint2*) (as2 + c2);
            uint2 bs89 = *(const uint2*) (as2 + c2 + 8);
            ja0[r] = dp4a_us(w0, bs01.x, ja0[r]);
            ja0[r] = dp4a_us(w1, bs01.y, ja0[r]);
            ja0[r] = dp4a_us(w2, bs89.x, ja0[r]);
            ja0[r] = dp4a_us(w3, bs89.y, ja0[r]);
            ja1[r] = dp4a_us(w4, bs01.x, ja1[r]);
            ja1[r] = dp4a_us(w5, bs01.y, ja1[r]);
            ja1[r] = dp4a_us(w6, bs89.x, ja1[r]);
            ja1[r] = dp4a_us(w7, bs89.y, ja1[r]);
        }
    }

    ext8w<bits, HALF>(blockB, t0, w0, w1, w2, w3, w4, w5, w6, w7);
    w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
    w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        const uint32_t* as = as_kb + r * slice_stride;
        uint2 as01 = *(const uint2*) (as + c2);
        uint2 as89 = *(const uint2*) (as + c2 + 8);
        ib0[r] = dp4a_us(w0, as01.x, ib0[r]);
        ib0[r] = dp4a_us(w1, as01.y, ib0[r]);
        ib0[r] = dp4a_us(w2, as89.x, ib0[r]);
        ib0[r] = dp4a_us(w3, as89.y, ib0[r]);
        ib1[r] = dp4a_us(w4, as01.x, ib1[r]);
        ib1[r] = dp4a_us(w5, as01.y, ib1[r]);
        ib1[r] = dp4a_us(w6, as89.x, ib1[r]);
        ib1[r] = dp4a_us(w7, as89.y, ib1[r]);
        if constexpr (residual)
        {
            const uint32_t* as2 = as_kb + (M + r) * slice_stride;
            uint2 bs01 = *(const uint2*) (as2 + c2);
            uint2 bs89 = *(const uint2*) (as2 + c2 + 8);
            jb0[r] = dp4a_us(w0, bs01.x, jb0[r]);
            jb0[r] = dp4a_us(w1, bs01.y, jb0[r]);
            jb0[r] = dp4a_us(w2, bs89.x, jb0[r]);
            jb0[r] = dp4a_us(w3, bs89.y, jb0[r]);
            jb1[r] = dp4a_us(w4, bs01.x, jb1[r]);
            jb1[r] = dp4a_us(w5, bs01.y, jb1[r]);
            jb1[r] = dp4a_us(w6, bs89.x, jb1[r]);
            jb1[r] = dp4a_us(w7, bs89.y, jb1[r]);
        }
    }
}

// Shared reduction tail for the pair-per-warp generic units (atomic, or exclusive plain stores for
// the per-slice partials of the sq kernel): four lanes share each n
template <int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_pair_tail
(
    int* __restrict__ accs, size_t acc_stride, int nbp, int lane, int size_n,
    int* ia0, int* ia1, int* ib0, int* ib1,
    int* ja0, int* ja1, int* jb0, int* jb1
)
{
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        #pragma unroll
        for (int o = 1; o < 4; o <<= 1)
        {
            ia0[r] += __shfl_xor_sync(0xffffffff, ia0[r], o);
            ia1[r] += __shfl_xor_sync(0xffffffff, ia1[r], o);
            ib0[r] += __shfl_xor_sync(0xffffffff, ib0[r], o);
            ib1[r] += __shfl_xor_sync(0xffffffff, ib1[r], o);
            if constexpr (residual)
            {
                ja0[r] += __shfl_xor_sync(0xffffffff, ja0[r], o);
                ja1[r] += __shfl_xor_sync(0xffffffff, ja1[r], o);
                jb0[r] += __shfl_xor_sync(0xffffffff, jb0[r], o);
                jb1[r] += __shfl_xor_sync(0xffffffff, jb1[r], o);
            }
        }
    }
    if ((lane & 3) == 0)
    {
        int nA = (nbp * 2) * 16 + (lane >> 2);
        int nB = nA + 16;
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            int* acc = accs + r * acc_stride;
            if constexpr (atomic)
            {
                atomicAdd(acc + nA, ia0[r]);
                atomicAdd(acc + nA + 8, ia1[r]);
                atomicAdd(acc + nB, ib0[r]);
                atomicAdd(acc + nB + 8, ib1[r]);
                if constexpr (residual)
                {
                    atomicAdd(acc + size_n + nA, ja0[r]);
                    atomicAdd(acc + size_n + nA + 8, ja1[r]);
                    atomicAdd(acc + size_n + nB, jb0[r]);
                    atomicAdd(acc + size_n + nB + 8, jb1[r]);
                }
            }
            else
            {
                acc[nA] = ia0[r];
                acc[nA + 8] = ia1[r];
                acc[nB] = ib0[r];
                acc[nB + 8] = ib1[r];
                if constexpr (residual)
                {
                    acc[size_n + nA] = ja0[r];
                    acc[size_n + nA + 8] = ja1[r];
                    acc[size_n + nB] = jb0[r];
                    acc[size_n + nB + 8] = jb1[r];
                }
            }
        }
    }
}

// Narrow generic unit (any K): one (256-column x k-slice) unit, warp per adjacent block pair
// processed sequentially with pointer-based extraction straight from global memory
// Words per 16x16 tile: 8 * bits, or 4 * (2 * bits + 1) at the half-integer rate bits + 0.5
template <int bits, bool HALF>
__host__ __device__ constexpr int gemv_int8_twords() { return HALF ? 4 * (2 * bits + 1) : 8 * bits; }

template <int bits, int M, bool residual, bool atomic = true, bool HALF = false>
__device__ __forceinline__ void gemv_int8_unit_narrow
(
    const uint16_t* __restrict__ B,
    int* __restrict__ accs,
    size_t acc_stride,
    const uint32_t* __restrict__ sh_as,
    int slice_stride,
    int nb256,
    int kb0,
    int nrows,
    int size_n
)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    constexpr int TWORDS = gemv_int8_twords<bits, HALF>();
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n / 16 * TWORDS;
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * (2 * TWORDS);
    int c2 = 2 * (lane & 3);
    int ia0[M] = {}, ia1[M] = {}, ib0[M] = {}, ib1[M] = {};
    int ja0[M] = {}, ja1[M] = {}, jb0[M] = {}, jb1[M] = {};

    for (int kb = 0; kb < nrows; ++kb)
    {
        const uint32_t* blockA = bp + (size_t) kb * row_stride;
        gemv_int8_pair_row<bits, M, residual, HALF>(blockA, blockA + TWORDS,
            sh_as + (kb << 4), slice_stride, c2, lane << 3,
            ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
    }
    gemv_int8_pair_tail<M, residual, atomic>(accs, acc_stride, nbp, lane, size_n, ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
}

// Smem-staged generic unit: the warp stages its block pair's rows into a warp-private shared memory
// slice with cp.async (coalesced 16 B chunks, one commit group per row, GEMV_STAGE_D rows deep) and
// extracts from shared memory. Used for the K where scattered pointer extraction leaves the most
// load latency exposed (3, 5, 7); warp-private slices need no block-level synchronization.
template <int bits, int M, bool residual, bool atomic = true, bool HALF = false>
__device__ __forceinline__ void gemv_int8_unit_smem
(
    const uint16_t* __restrict__ B,
    int* __restrict__ accs,
    size_t acc_stride,
    const uint32_t* __restrict__ sh_as,
    int slice_stride,
    uint32_t* __restrict__ sh_b,
    int nb256,
    int kb0,
    int nrows,
    int size_n
)
{
    constexpr int D = GEMV_STAGE_D;
    constexpr int TWORDS = gemv_int8_twords<bits, HALF>();
    constexpr int pairwords = 2 * TWORDS;
    constexpr int chunks = pairwords / 4;      // 16 B cp.async chunks per pair row
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n / 16 * TWORDS;
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * pairwords;
    uint32_t* sb = sh_b + warp * (D * pairwords);

    auto stage_row = [&] (int kb)
    {
        if (kb < nrows && lane < chunks)
            cp_async(sb + (kb % D) * pairwords + lane * 4, bp + (size_t) kb * row_stride + lane * 4);
        cp_async_fence();
    };
    #pragma unroll
    for (int r = 0; r < D - 1; ++r) stage_row(r);

    int c2 = 2 * (lane & 3);
    int ia0[M] = {}, ia1[M] = {}, ib0[M] = {}, ib1[M] = {};
    int ja0[M] = {}, ja1[M] = {}, jb0[M] = {}, jb1[M] = {};

    for (int kb = 0; kb < nrows; ++kb)
    {
        cp_async_wait<D - 2>();
        // Also orders the previous iteration's smem reads (all lanes) before the overwrite below
        __syncwarp();
        stage_row(kb + D - 1);

        const uint32_t* blockA = sb + (kb % D) * pairwords;
        gemv_int8_pair_row<bits, M, residual, HALF>(blockA, blockA + TWORDS,
            sh_as + (kb << 4), slice_stride, c2, lane << 3,
            ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
    }
    gemv_int8_pair_tail<M, residual, atomic>(accs, acc_stride, nbp, lane, size_n, ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
}

// K values routed to the smem-staged unit (measured on 3090: K=3 -13%; narrow wins for 2/6/8 which
// are at the ALU floor / DRAM-bound, wide covers 4)
__host__ __device__ constexpr bool gemv_int8_stage_smem(int bits, bool half = false)
{
    return half || bits == 3 || bits == 5 || bits == 7;   // half-integer rates: staged (unmeasured, the generic choice)
}

// Stage bytes per block for the staged unit: 8 warps x GEMV_STAGE_D pair rows
__host__ __device__ constexpr int gemv_int8_stage_bytes(int bits, bool half = false)
{
    return gemv_int8_stage_smem(bits, half) ? 8 * GEMV_STAGE_D * (half ? 8 * (2 * bits + 1) : 16 * bits) * 4 : 0;
}

// Epilogue for one row: affine correction + output Hadamard + svh scale + accumulator reset,
// striped over all warps of the grid. sh_tmp: 128 floats per warp.
template <bool c_fp32, bool residual>
__device__ __forceinline__ void gemv_int8_epilogue
(
    int* __restrict__ accs,
    void* __restrict__ C_row,
    const float* __restrict__ qrow,
    const half* __restrict__ svh,
    float* __restrict__ sh_tmp,
    int size_n,
    float output_scale
)
{
    int lane = threadIdx.x & 31;
    int warps_grid = gridDim.x * NUM_THREADS / 32;
    int this_warp = threadIdx.x / 32 + NUM_THREADS / 32 * blockIdx.x;
    float* tmp = sh_tmp + (threadIdx.x / 32) * 128;

    float k_inv  = __half2float(__ushort_as_half(0x1eee));
    float k_bias = __half2float(__ushort_as_half(0xc931));
    float q = qrow[0];
    float q2 = q * (1.0f / 254.0f);
    float suma = q * (float) ((const int*) qrow)[1];
    if constexpr (residual) suma += q2 * (float) ((const int*) qrow)[2];
    float corr = (1024.0f * k_inv + k_bias) * suma;

    int total_chunks = size_n / 128;
    for (; this_warp < total_chunks; this_warp += warps_grid)
    {
        int base = this_warp * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int j = lane * 4 + i;
            float s = q * (float) accs[base + j];
            accs[base + j] = 0;
            if constexpr (residual)
            {
                s += q2 * (float) accs[size_n + base + j];
                accs[size_n + base + j] = 0;
            }
            tmp[j] = k_inv * s + corr;
        }
        __syncwarp();
        // gridDim.y == 1, so the inner's blockIdx.y-based scale offset is zero; pre-offset svh
        if constexpr (c_fp32)
            had_ff_r_128_inner<false, true>(tmp, ((float*) C_row) + base, svh + base, 0.088388347648f * output_scale);
        else
            had_fh_r_128_inner<false, true>(tmp, ((half*) C_row) + base, svh + base, 0.088388347648f * output_scale);
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------------------------------------
// Per-slice-scale ("sq") kernel for m == 1: kills ALL global pre-GEMV coordination. Each k-slice is
// Hadamard-transformed, maxed and quantized block-locally during staging with its own
// q_s = slicemax/127 (deterministic, so duplicating blocks agree bit-exactly - and strictly finer
// quantization than a global q), units write exact per-slice int32 partials with plain stores
// (exclusive writers - no atomics, no accumulator zeroing), and a per-256-column completion counter
// gates an inline epilogue that combines sum_s q_s*acc_s in fixed slice order (deterministic) plus
// the per-slice affine corrections. One REGULAR launch: no cooperative machinery, no grid.sync, no
// A_had traffic. Same 10 kernel arguments (A_had unused). Workspace layout (fixed offsets so mixed
// sq/coop calls of any shape can share the buffer):
//   [counters: SQ_COUNTERS_CAP][{q_s, s1_s, s2_s, pad} x SQ_KSPLIT_CAP][partials: ksplit x pstride]
// The coop kernels use the region beyond SQ_WS_RESERVED. Counters are zeroed when the workspace is
// (re)allocated and reset by the finishing block, so they stay zero between calls.

// Stage one slice for M activation rows: Hadamard from A into sh_ah, slice max -> q_s, splats +
// exact sums per row. Bit-identical in every block that stages the same slice. Rows >= size_m
// (padding at M = 4, m = 3) get zero splats and dummy scales.
template <int M, bool residual>
__device__ __forceinline__ void gemv_int8_stage_slice
(
    const half* __restrict__ A,
    int size_m,
    int size_k,
    const half* __restrict__ suh,
    float* __restrict__ qs,            // qsums + 4 * slice * M, rows consecutive, published (idempotent)
    half* __restrict__ sh_ah,
    uint32_t* __restrict__ sh_as,      // M regions of slice_stride words (+ M more for residual)
    int slice_stride,
    float* __restrict__ sh_red,        // 33 floats
    int kb0,
    int nrows
)
{
    int t = threadIdx.x;
    int nel = nrows * 16;
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        uint32_t* as = sh_as + r * slice_stride;
        uint32_t* as2 = sh_as + (M + r) * slice_stride;
        float* qsr = qs + 4 * r;
        __syncthreads();
        if (r >= size_m)
        {
            for (int i = t; i < nel; i += NUM_THREADS)
            {
                as[i] = 0;
                if constexpr (residual) as2[i] = 0;
            }
            if (t == 0)
            {
                qsr[0] = 1.0f;
                ((int*) qsr)[1] = 0;
                ((int*) qsr)[2] = 0;
            }
            continue;
        }
        const half* Ar = A + (size_t) r * size_k;
        for (int sp = t >> 5; sp < (nel >> 7); sp += NUM_THREADS >> 5)
            had_hf_r_128_inner<true, false>(Ar + (kb0 << 4) + (sp << 7), sh_ah + (sp << 7), suh + (kb0 << 4) + (sp << 7), 0.088388347648f);
        __syncthreads();

        float mx = 0.0f;
        for (int i = t; i < nel; i += NUM_THREADS) mx = fmaxf(mx, fabsf(__half2float(sh_ah[i])));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        if ((t & 31) == 0) sh_red[t >> 5] = mx;
        __syncthreads();
        if (t < 32)
        {
            float v = t < (NUM_THREADS >> 5) ? sh_red[t] : 0.0f;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
            if (t == 0) sh_red[32] = fmaxf(v, 1e-8f) / 127.0f;
        }
        __syncthreads();
        float q_s = sh_red[32];

        float rq = 1.0f / q_s;
        float rq2 = rq * 254.0f;
        int l1 = 0, l2 = 0;
        for (int i = t; i < nel; i += NUM_THREADS)
        {
            float a = __half2float(sh_ah[i]);
            int v = __float2int_rn(a * rq);
            v = max(-127, min(127, v));
            as[i] = ((uint32_t)(uint8_t)(int8_t) v) * 0x01010101u;
            l1 += v;
            if constexpr (residual)
            {
                float rr = a - q_s * (float) v;
                int v2 = __float2int_rn(rr * rq2);
                v2 = max(-127, min(127, v2));
                as2[i] = ((uint32_t)(uint8_t)(int8_t) v2) * 0x01010101u;
                l2 += v2;
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            l1 += __shfl_xor_sync(0xffffffff, l1, o);
            l2 += __shfl_xor_sync(0xffffffff, l2, o);
        }
        if ((t & 31) == 0) { ((int*) sh_red)[t >> 5] = l1; sh_red[16 + (t >> 5)] = __int_as_float(l2); }
        __syncthreads();
        if (t < 32)
        {
            int v1 = t < (NUM_THREADS >> 5) ? ((int*) sh_red)[t] : 0;
            int v2 = t < (NUM_THREADS >> 5) ? __float_as_int(sh_red[16 + t]) : 0;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                v1 += __shfl_xor_sync(0xffffffff, v1, o);
                v2 += __shfl_xor_sync(0xffffffff, v2, o);
            }
            if (t == 0)
            {
                qsr[0] = sh_red[32];
                ((int*) qsr)[1] = v1;
                ((int*) qsr)[2] = v2;
            }
        }
    }
    __syncthreads();
}

// Epilogue for one 256-column group: deterministic fixed-order combine over the per-slice partials,
// warp per (row, 128-span). Reads bypass L1 (__ldcg): the contributions arrived from other blocks
// with no grid-wide barrier.
template <int M, bool c_fp32, bool residual>
__device__ __forceinline__ void gemv_int8_epilogue_group_sq
(
    const int* __restrict__ partials,
    const float* __restrict__ qsums,
    int pstride,
    int ksplit,
    int size_m,
    void* __restrict__ C,
    const half* __restrict__ svh,
    float* __restrict__ sh_tmp,
    int nb256,
    int size_n
)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = warp >> 1;
    if (warp >= 2 * M || row >= size_m) return;
    float k_inv  = __half2float(__ushort_as_half(0x1eee));
    float k_bias = __half2float(__ushort_as_half(0xc931));
    float aff = 1024.0f * k_inv + k_bias;

    int base = nb256 * 256 + (warp & 1) * 128;
    float* tmp = sh_tmp + warp * 128;
    float acc[4] = {};
    float corr = 0.0f;
    for (int sl = 0; sl < ksplit; ++sl)
    {
        int idx = sl * M + row;
        float q_s = qsums[4 * idx];
        float suma = q_s * (float) ((const int*) qsums)[4 * idx + 1];
        const int* p = partials + (size_t) idx * pstride + base;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            acc[i] += q_s * (float) __ldcg(p + lane * 4 + i);
        if constexpr (residual)
        {
            float q2_s = q_s * (1.0f / 254.0f);
            suma += q2_s * (float) ((const int*) qsums)[4 * idx + 2];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                acc[i] += q2_s * (float) __ldcg(p + size_n + lane * 4 + i);
        }
        corr += aff * suma;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        tmp[lane * 4 + i] = k_inv * acc[i] + corr;
    __syncwarp();
    if constexpr (c_fp32)
        had_ff_r_128_inner<false, true>(tmp, ((float*) C) + (size_t) row * size_n + base, svh + base, 0.088388347648f);
    else
        had_fh_r_128_inner<false, true>(tmp, ((half*) C) + (size_t) row * size_n + base, svh + base, 0.088388347648f);
}

// Shared-memory cap for the sq decomposition: row halfs (32 B/row) + M splat regions (64 B/row each,
// x2 residual), within ~80 KB so the stage region and epilogue staging still fit under the opt-in max
__host__ __device__ constexpr int gemv_int8_sq_rows_max(int M, bool residual)
{
    int cap = (80 * 1024) / (32 + 64 * M * (residual ? 2 : 1));
    cap &= ~7;
    return cap < SQ_ROWS_MAX ? cap : SQ_ROWS_MAX;
}

template <int bits, int M, bool c_fp32, bool residual, bool HALF = false>
__global__ __launch_bounds__(NUM_THREADS)
void exl3_gemv_int8_sq_kernel
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,       // 1 <= size_m <= M
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* __restrict__ suh,
    half* __restrict__ A_had,
    const half* __restrict__ svh
)
{
    extern __shared__ uint32_t shmem[];

    // Work decomposition: rows_per multiple of 8 (whole 128-spans for the local Hadamard); the host
    // mirrors this for the shared memory size and workspace bound. Swept on 3090: the wall time is
    // ~one unit's duration as long as all (slice x 256-column) units fit in one wave of the grid, so
    // take the smallest slice height with units <= gridDim.x - but at least half-wave occupancy up
    // to 32 rows, since very short slices inflate ksplit and the epilogue's serial per-slice combine
    int rows_total = size_k >> 4;
    int nb256_total = size_n / 256;
    int r = CEIL_DIVIDE(rows_total * nb256_total, (int) gridDim.x);
    int rows_per = (MAX(r, MIN(2 * r, 32)) + 7) & ~7;
    rows_per = MAX(rows_per, SQ_MINROWS);
    rows_per = MIN(rows_per, gemv_int8_sq_rows_max(M, residual));
    rows_per = MIN(rows_per, (rows_total + 7) & ~7);
    int ksplit = CEIL_DIVIDE(rows_total, rows_per);
    int units = nb256_total * ksplit;
    int slice_stride = rows_per * 16;
    int pstride = size_n * (residual ? 2 : 1);

    int* counters = locks;
    float* qsums = (float*) (locks + SQ_COUNTERS_CAP);
    int* partials = locks + SQ_WS_RESERVED;

    // Shared layout: [slice halfs][M x splats (+ M x residual splats)][B stage][epilogue tmp]
    half* sh_ah = (half*) shmem;
    uint32_t* sh_as = shmem + rows_per * 8;
    uint32_t* sh_b = sh_as + slice_stride * M * (residual ? 2 : 1);
    float* sh_tmp = (float*) (sh_b + gemv_int8_stage_bytes(bits, HALF) / 4);
    __shared__ float sh_red[33];
    __shared__ int sh_last;

    int t = threadIdx.x;
    int prev_slice = -1;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x)
    {
        int slice = unit / nb256_total;
        int nb256 = unit % nb256_total;
        int kb0 = slice * rows_per;
        int nrows = MIN(rows_per, rows_total - kb0);
        if (slice != prev_slice)
        {
            gemv_int8_stage_slice<M, residual>(A, size_m, size_k, suh, qsums + 4 * slice * M,
                                               sh_ah, sh_as, slice_stride, sh_red, kb0, nrows);
            prev_slice = slice;
        }
        int* pacc = partials + (size_t) slice * M * pstride;
        if constexpr (bits == 4 && !HALF)
            gemv_int8_unit_wide<M, residual, false>(B, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);
        else if constexpr (gemv_int8_stage_smem(bits, HALF))
            gemv_int8_unit_smem<bits, M, residual, false, HALF>(B, pacc, pstride, sh_as, slice_stride, sh_b, nb256, kb0, nrows, size_n);
        else
            gemv_int8_unit_narrow<bits, M, residual, false, HALF>(B, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);

        // Completion counter: the ksplit-th contributor runs the epilogue for this 256-column group.
        // (sh_last reuse across iterations is ordered by the next iteration's __syncthreads.)
        __threadfence();
        __syncthreads();
        if (t == 0) sh_last = (atomicAdd(&counters[nb256], 1) == ksplit - 1) ? 1 : 0;
        __syncthreads();
        if (sh_last)
        {
            gemv_int8_epilogue_group_sq<M, c_fp32, residual>(partials, qsums, pstride, ksplit, size_m,
                                                             C, svh, sh_tmp, nb256, size_n);
            if (t == 0) counters[nb256] = 0;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Cooperative kernel: same argument list as exl3_gemm_kernel

template <int bits, bool c_fp32, bool residual, bool HALF = false>
__global__ __launch_bounds__(NUM_THREADS)
void exl3_gemv_int8_coop_kernel
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* __restrict__ suh,
    half* __restrict__ A_had,
    const half* __restrict__ svh
)
{
    auto grid = cg_gemv::this_grid();
    extern __shared__ uint32_t shmem[];

    int* accs = locks;
    float* qsums = (float*) (locks + 2 * size_n);
    float* partial_max = qsums + 4 * size_m;   // [gridDim.x], m == 1 fast path

    // Phase 1a: input Hadamard (as in exl3_gemm_kernel) + zero the accumulators. For m == 1, also
    // track a per-block partial max of the transformed activations.
    {
        __shared__ float sh_m[32];
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * NUM_THREADS / 32;
        int this_warp = threadIdx.x / 32 + NUM_THREADS / 32 * blockIdx.x;
        int lane = threadIdx.x & 31;
        float mx = 0.0f;
        for (; this_warp < total_warps; this_warp += warps_grid)
        {
            had_hf_r_128_inner<true, false>
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f
            );
            if (size_m == 1)
            {
                __syncwarp();
                const half2* h2 = (const half2*) (A_had + this_warp * 128);
                #pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    half2 v = h2[lane * 2 + i];
                    mx = fmaxf(mx, fabsf(__half2float(__low2half(v))));
                    mx = fmaxf(mx, fabsf(__half2float(__high2half(v))));
                }
            }
        }
        if (size_m == 1)
        {
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
            if (lane == 0) sh_m[threadIdx.x >> 5] = mx;
            __syncthreads();
            if (threadIdx.x < 32)
            {
                float v = threadIdx.x < (NUM_THREADS >> 5) ? sh_m[threadIdx.x] : 0.0f;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
                if (threadIdx.x == 0) partial_max[blockIdx.x] = v;
            }
        }
        int tg = blockIdx.x * NUM_THREADS + threadIdx.x;
        for (int i = tg; i < 2 * size_n; i += gridDim.x * NUM_THREADS) accs[i] = 0;
    }
    grid.sync();

    // Phase 1b (m > 1 only): per-row activation scale and exact sums
    if (size_m > 1)
    {
        __shared__ float sh_r[33];
        for (int row = blockIdx.x; row < size_m; row += gridDim.x)
        {
            const half* Ar = A_had + (size_t) row * size_k;
            int t = threadIdx.x;
            float mx = 0.0f;
            for (int i = t; i < size_k; i += NUM_THREADS) mx = fmaxf(mx, fabsf(__half2float(Ar[i])));
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
            if ((t & 31) == 0) sh_r[t >> 5] = mx;
            __syncthreads();
            if (t < 32)
            {
                float v = t < (NUM_THREADS >> 5) ? sh_r[t] : 0.0f;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
                if (t == 0) sh_r[32] = fmaxf(v, 1e-8f) / 127.0f;
            }
            __syncthreads();
            gemv_int8_row_sums<residual>(Ar, sh_r[32], size_k, qsums + 4 * row);
        }
        grid.sync();
    }

    // Work decomposition for phase 2 (host must mirror this for the shared memory size).
    // Units are 256 columns (warp per adjacent block pair) x k-slice; fine-grained (~4 per block)
    // because with grid.sync at the end of the phase, tail imbalance stalls the whole grid.
    int rows_total = size_k >> 4;
    int nb256_total = size_n / 256;
    int smem_rows_max = residual ? 384 : 768;
    int ksplit = CEIL_DIVIDE(4 * gridDim.x, nb256_total);
    ksplit = MAX(ksplit, CEIL_DIVIDE(rows_total, smem_rows_max));
    ksplit = MIN(ksplit, rows_total);
    int rows_per = CEIL_DIVIDE(rows_total, ksplit);
    int units = nb256_total * ksplit;

    uint32_t* sh_as = shmem;
    uint32_t* sh_as2 = shmem + rows_per * 16;
    uint32_t* sh_b = shmem + rows_per * 16 * (residual ? 2 : 1);   // smem-staged unit region

    __shared__ float sh_q[33];

    for (int row = 0; row < size_m; ++row)
    {
        const half* Ar = A_had + (size_t) row * size_k;
        const float* qrow = qsums + 4 * row;

        // Row scale: for m == 1, derived from the phase-1a partial maxes (deterministic: fmax is
        // order-independent). Block 0 also computes the exact sums for the epilogue, concurrently
        // with the other blocks' GEMV work.
        float q_row;
        if (size_m == 1)
        {
            float v = 0.0f;
            for (int i = threadIdx.x; i < gridDim.x; i += NUM_THREADS) v = fmaxf(v, partial_max[i]);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
            if ((threadIdx.x & 31) == 0) sh_q[threadIdx.x >> 5] = v;
            __syncthreads();
            if (threadIdx.x < 32)
            {
                float w = threadIdx.x < (NUM_THREADS >> 5) ? sh_q[threadIdx.x] : 0.0f;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) w = fmaxf(w, __shfl_xor_sync(0xffffffff, w, o));
                if (threadIdx.x == 0) sh_q[32] = fmaxf(w, 1e-8f) / 127.0f;
            }
            __syncthreads();
            q_row = sh_q[32];
            if (blockIdx.x == 0)
                gemv_int8_row_sums<residual>(Ar, q_row, size_k, qsums);
        }
        else q_row = qrow[0];

        // Phase 2: GEMV over (nb256, k-slice) units, splats quantized inline during smem staging
        int prev_slice = -1;
        for (int unit = blockIdx.x; unit < units; unit += gridDim.x)
        {
            int slice = unit / nb256_total;
            int nb256 = unit % nb256_total;
            int kb0 = slice * rows_per;
            int nrows = MIN(rows_per, rows_total - kb0);
            if (slice != prev_slice)
            {
                gemv_int8_stage_splats<residual>(Ar, q_row, sh_as, sh_as2, kb0, nrows);
                prev_slice = slice;
            }
            if constexpr (bits == 4 && !HALF)
                gemv_int8_unit_wide<1, residual>(B, accs, 0, sh_as, rows_per * 16, nb256, kb0, nrows, size_n);
            else if constexpr (gemv_int8_stage_smem(bits, HALF))
                gemv_int8_unit_smem<bits, 1, residual, true, HALF>(B, accs, 0, sh_as, rows_per * 16, sh_b, nb256, kb0, nrows, size_n);
            else
                gemv_int8_unit_narrow<bits, 1, residual, true, HALF>(B, accs, 0, sh_as, rows_per * 16, nb256, kb0, nrows, size_n);
        }
        grid.sync();

        // Phase 3: epilogue + output Hadamard (shared memory reused as per-warp staging)
        void* C_row = c_fp32 ? (void*) (((float*) C) + (size_t) row * size_n)
                             : (void*) (((half*)  C) + (size_t) row * size_n);
        gemv_int8_epilogue<c_fp32, residual>(accs, C_row, qrow, svh, (float*) shmem, size_n, 1.0f);

        if (row + 1 < size_m) grid.sync();
    }
}
#else  // __HIP_PLATFORM_AMD__ || USE_ROCM

// ROCm INT8-activation GEMV for mul1 (cb 2) trellises on RDNA4.
//
// Same arithmetic as the CUDA int8 path above: the cb 2 codebook is affine in the byte sum of
// state * 0x83DCD12D, so
//
//   value * a = k_inv * byte_sum(state * 0x83DCD12D) * a + k_bias * a
//
// and a dp4a against a splat of an int8-quantized activation evaluates the byte sum times that
// activation in one instruction, without ever decoding the trellis. The quantized activation's
// per-row scale and exact int sums are the CUDA cooperative kernel's global scale
// (q = max|A_had| / 127, v = clamp(rint(a / q), -127, 127), residual mode re-quantizes the
// rounding error with q2 = q / 254).
//
// RDNA has no cooperative launch, so the HIP entry point runs the CUDA kernel's phases as
// separate launches on the caller's stream, exactly like the fp16 HIP GEMV:
//
//   1. input Hadamard A -> shared, int8 + row max + sums (exl3_gemv_int8_hip_quant_kernel)
//   2. GEMV + affine epilogue                             (exl3_gemv_int8_hip_kernel)
//   3. output Hadamard + svh                              (exl3_gemv_hip_had_out_kernel)
//
// RDNA4's v_dot4_i32_i8 is signed x signed where the CUDA dp4a is u32 x s32, so the kernel flips
// the sign bit of every byte of the trellis product before the dot. That turns the unsigned byte
// sum into sum_i (b_i - 128) * v = byte_sum * v - 512 * v, and the -512 * sum(v) it introduces
// is folded into the affine correction (1536 * k_inv + k_bias instead of 1024 * k_inv + k_bias).

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include "exl3_gemv.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_devctx.cuh"

namespace exl3_gemv_int8_hip_ns {

// Signed byte dot product. Full rate on the RDNA3/RDNA4 targets that implement it; the
// portable form keeps every other target assembling (the host declines them at run time).
__device__ __forceinline__ int dp4a(uint32_t a, uint32_t b, int c)
{
#if defined(__GFX11__) || defined(__GFX12__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1202__)
    int d;
    asm("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
#else
    int d = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        d += (int) (int8_t) ((a >> (8 * i)) & 0xff) * (int) (int8_t) ((b >> (8 * i)) & 0xff);
    return d;
#endif
}

// Byte J of x replicated into all four bytes (one v_perm_b32 on RDNA).
template <int J>
__device__ __forceinline__ uint32_t splat(uint32_t x)
{
    return __builtin_amdgcn_perm(x, x, 0x01010101u * J);
}

// The quantize kernel writes the four int8 activations a lane needs (k = 2q, 2q + 1, 2q + 8,
// 2q + 9 of every 16-k row, q = lane & 3) as the four bytes of one word, so a lane loads them
// with a single broadcast load. Grouping k by 16 makes that a byte permutation of one 16-byte
// vector: source words 0..3 hold k 0..7 and source words 4..7 hold k 8..15, so output word w
// takes the low half of source word w and the low half of source word w + 4.
__device__ __forceinline__ float lo_half(uint32_t u)
{
    return __half2float(__ushort_as_half((unsigned short) (u & 0xffffu)));
}

__device__ __forceinline__ float hi_half(uint32_t u)
{
    return __half2float(__ushort_as_half((unsigned short) (u >> 16)));
}

// v = clamp(rint(a / q), -127, 127), the CUDA kernel's per-element quantization.
__device__ __forceinline__ int q8(float a, float rq)
{
    return max(-127, min(127, __float2int_rn(a * rq)));
}

// Pack four int8 lanes into one word (byte 0 = lowest k).
__device__ __forceinline__ uint32_t pack4(int v0, int v1, int v2, int v3)
{
    return (uint32_t) (v0 & 0xff) | ((uint32_t) (v1 & 0xff) << 8)
         | ((uint32_t) (v2 & 0xff) << 16) | ((uint32_t) (v3 & 0xff) << 24);
}

}  // namespace exl3_gemv_int8_hip_ns

// Quantize one activation row: input Hadamard into shared memory, per-row max, int8
// splat-ready layout, exact int sums. One block per row; the row is at most a few tens of KiB,
// so keeping it in shared memory costs less than a separate Hadamard launch plus its global
// round trip, and the single-block max stays deterministic.
template <bool RESIDUAL>
__global__ __launch_bounds__(512)
void exl3_gemv_int8_hip_quant_kernel
(
    const half* __restrict__ A,
    const half* __restrict__ suh,
    int8_t* __restrict__ qa,
    int8_t* __restrict__ qb,
    float* __restrict__ qscale,
    int* __restrict__ qsum,
    int* __restrict__ qsum2,
    int size_k
)
{
    using namespace exl3_gemv_int8_hip_ns;
    constexpr int THREADS = 512;
    constexpr int WARPS = THREADS / 32;
    const int row = blockIdx.x;
    const int t = threadIdx.x;

    // Stage the row's input Hadamard in shared memory: the quantization needs the transformed
    // row, so doing it here saves a launch and the global round trip through A_had that the
    // CUDA phase split implies. One warp per 128 k, exactly as the standalone had_in kernel.
    extern __shared__ uint4 sh_vec[];
    half* sh_ah = reinterpret_cast<half*>(sh_vec);
    const half* Ar = A + (size_t) row * size_k;
    for (int off = (t >> 5) * 128; off < size_k; off += WARPS * 128)
        had_hf_r_128_inner<true, false>(Ar + off, sh_ah + off, suh + off, 0.088388347648f);
    __syncthreads();

    const uint4* Ar4 = reinterpret_cast<const uint4*>(sh_ah);       // 8 halves (16 B) per vector
    const int ngroups = size_k >> 4;                                // 16 k per int8 group
    uint4* qa4 = reinterpret_cast<uint4*>(qa + (size_t) row * size_k);
    uint4* qb4 = reinterpret_cast<uint4*>(qb + (size_t) row * size_k);

    __shared__ float sh_max[THREADS / 32];
    __shared__ float sh_q;
    __shared__ int sh_s1[THREADS / 32];
    __shared__ int sh_s2[THREADS / 32];

    // Row max. One thread per 16-k group: two 16-byte vector loads instead of 32 scalar ones,
    // which is what keeps this pass off the critical path on a single block.
    float mx = 0.0f;
    for (int g = t; g < ngroups; g += THREADS)
    {
        const uint4 a0 = Ar4[2 * g], a1 = Ar4[2 * g + 1];
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(lo_half(a0.x)), fabsf(hi_half(a0.x))),
                             fmaxf(fabsf(lo_half(a0.y)), fabsf(hi_half(a0.y)))));
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(lo_half(a0.z)), fabsf(hi_half(a0.z))),
                             fmaxf(fabsf(lo_half(a0.w)), fabsf(hi_half(a0.w)))));
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(lo_half(a1.x)), fabsf(hi_half(a1.x))),
                             fmaxf(fabsf(lo_half(a1.y)), fabsf(hi_half(a1.y)))));
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(lo_half(a1.z)), fabsf(hi_half(a1.z))),
                             fmaxf(fabsf(lo_half(a1.w)), fabsf(hi_half(a1.w)))));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffull, mx, o));
    if ((t & 31) == 0) sh_max[t >> 5] = mx;
    __syncthreads();
    if (t == 0)
    {
        float v = sh_max[0];
        #pragma unroll
        for (int i = 1; i < THREADS / 32; ++i) v = fmaxf(v, sh_max[i]);
        // Same floor and scale as the CUDA kernel's per-row phase
        sh_q = fmaxf(v, 1e-8f) / 127.0f;
        qscale[row] = sh_q;
    }
    __syncthreads();

    const float q = sh_q;
    const float rq = 1.0f / q;
    const float rq2 = rq * 254.0f;
    int l1 = 0, l2 = 0;
    for (int g = t; g < ngroups; g += THREADS)
    {
        const uint4 a0 = Ar4[2 * g], a1 = Ar4[2 * g + 1];
        const uint32_t src[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        uint32_t s[4], r[4];
        #pragma unroll
        for (int w = 0; w < 4; ++w)
        {
            const float x0 = lo_half(src[w]), x1 = hi_half(src[w]);
            const float x2 = lo_half(src[w + 4]), x3 = hi_half(src[w + 4]);
            const int v0 = q8(x0, rq), v1 = q8(x1, rq), v2 = q8(x2, rq), v3 = q8(x3, rq);
            s[w] = pack4(v0, v1, v2, v3);
            if constexpr (RESIDUAL)
            {
                const int u0 = q8(x0 - q * (float) v0, rq2);
                const int u1 = q8(x1 - q * (float) v1, rq2);
                const int u2 = q8(x2 - q * (float) v2, rq2);
                const int u3 = q8(x3 - q * (float) v3, rq2);
                r[w] = pack4(u0, u1, u2, u3);
            }
        }
        qa4[g] = make_uint4(s[0], s[1], s[2], s[3]);
        if constexpr (RESIDUAL) qb4[g] = make_uint4(r[0], r[1], r[2], r[3]);
        // Exact int sums of the quantized bytes, one v_dot4_i32_i8 per word
        const uint32_t ones = 0x01010101u;
        l1 = dp4a(s[3], ones, dp4a(s[2], ones, dp4a(s[1], ones, dp4a(s[0], ones, l1))));
        if constexpr (RESIDUAL)
            l2 = dp4a(r[3], ones, dp4a(r[2], ones, dp4a(r[1], ones, dp4a(r[0], ones, l2))));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        l1 += __shfl_xor_sync(0xffffffffull, l1, o);
        l2 += __shfl_xor_sync(0xffffffffull, l2, o);
    }
    if ((t & 31) == 0) { sh_s1[t >> 5] = l1; sh_s2[t >> 5] = l2; }
    __syncthreads();
    if (t == 0)
    {
        int a1 = 0, a2 = 0;
        #pragma unroll
        for (int i = 0; i < THREADS / 32; ++i) { a1 += sh_s1[i]; a2 += sh_s2[i]; }
        qsum[row] = a1;
        qsum2[row] = a2;
    }
}

// One warp per 16-column tile and k split, as in the fp16 HIP kernel: lane l covers columns
// l / 4 and l / 4 + 8 at k rows 2 * (l % 4) + {0, 1, 8, 9} of the 16x16 tile, so its eight
// trellis windows pair with exactly four activation values. ROWS is the row count the caller
// compiled for (1, 2, 4 or 8), WK warps split k and a block covers 16 * WT columns.
template <int bits, int ROWS, int WK, int WT, bool RESIDUAL>
__global__ __launch_bounds__(32 * WK * WT)
void exl3_gemv_int8_hip_kernel
(
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const uint32_t* __restrict__ qa32,
    const uint32_t* __restrict__ qb32,
    const float* __restrict__ qscale,
    const int* __restrict__ qsum,
    const int* __restrict__ qsum2,
    int size_m,
    int size_k,
    int size_n
)
{
    constexpr int TWORDS = 8 * bits;
    constexpr int COLS = 16 * WT;
    constexpr int THREADS = 32 * WK * WT;
    constexpr int PF = 2;
    using namespace exl3_gemv_int8_hip_ns;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int ksplit = warp % WK;
    const int tile = warp / WK;

    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int chunk = (kslices + WK - 1) / WK;
    const int ks0 = ksplit * chunk;
    const int ks1 = min(ks0 + chunk, kslices);
    const bool full = size_m >= ROWS;
    const int rows_out = full ? ROWS : size_m;
    const int k4 = size_k >> 2;                                    // uint32 per activation row

    const int q = lane & 3;
    const int c0 = lane >> 2;

    const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const int ntile = blockIdx.x * WT + tile;
    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + (size_t) ntile * TWORDS;

    // Per-lane extraction constants, same derivation as the fp16 kernel
    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2)
    {
        const int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    else if constexpr (bits == 3)
    {
        const int t_offset = lane << 3;
        const int b1 = (t_offset + 257) * 3;
        const int b2 = b1 + 21;
        const int i0 = (b1 - 16) / 32;
        const int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    auto load = [&] (int ks) -> uint32_t
    {
        if (ks >= ks1) return 0u;
        const uint32_t* tp = bp + (size_t) (ks - ks0) * slice_stride;
        return lane < TWORDS ? tp[lane] : 0u;
    };

    uint32_t pf[PF];
    #pragma unroll
    for (int d = 0; d < PF; ++d) pf[d] = load(ks0 + d);

    const uint32_t* ap[ROWS];
    const uint32_t* aq[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r)
    {
        ap[r] = qa32 + (size_t) r * k4 + (size_t) ks0 * 4 + q;
        if constexpr (RESIDUAL) aq[r] = qb32 + (size_t) r * k4 + (size_t) ks0 * 4 + q;
    }

    int acc[ROWS][2] = {};
    int acc2[ROWS][2] = {};

    for (int ks = ks0; ks < ks1; ++ks)
    {
        const uint32_t bw = pf[0];
        pf[0] = pf[1];
        pf[1] = load(ks + PF);

        uint32_t w[8];
        if constexpr (bits == 4)
            exl3_gemv_hip_ns::ext8_regs_4bits(exl3_gemv_hip_ns::shfl32(bw, (lane + 31) & 31), bw, w);
        else if constexpr (bits == 2)
            exl3_gemv_hip_ns::ext8_regs_2bits(exl3_gemv_hip_ns::shfl32(bw, x_src_a),
                                              exl3_gemv_hip_ns::shfl32(bw, x_src_b), lane << 3, w);
        else
            exl3_gemv_hip_ns::ext8_regs_3bits(exl3_gemv_hip_ns::shfl32(bw, x_src_a),
                                              exl3_gemv_hip_ns::shfl32(bw, x_src_b), x_s2, w);

        // state * 0x83DCD12D with the byte sign bits flipped (see the header comment)
        uint32_t t[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) t[i] = (w[i] * 0x83DCD12Du) ^ 0x80808080u;

        #pragma unroll
        for (int r = 0; r < ROWS; ++r)
        {
            if (!full && r >= rows_out) break;
            const uint32_t av = ap[r][0];
            ap[r] += 4;
            const uint32_t s0 = splat<0>(av), s1 = splat<1>(av), s2 = splat<2>(av), s3 = splat<3>(av);
            acc[r][0] = dp4a(t[0], s0, acc[r][0]);
            acc[r][0] = dp4a(t[1], s1, acc[r][0]);
            acc[r][0] = dp4a(t[2], s2, acc[r][0]);
            acc[r][0] = dp4a(t[3], s3, acc[r][0]);
            acc[r][1] = dp4a(t[4], s0, acc[r][1]);
            acc[r][1] = dp4a(t[5], s1, acc[r][1]);
            acc[r][1] = dp4a(t[6], s2, acc[r][1]);
            acc[r][1] = dp4a(t[7], s3, acc[r][1]);
            if constexpr (RESIDUAL)
            {
                const uint32_t bv = aq[r][0];
                aq[r] += 4;
                const uint32_t u0 = splat<0>(bv), u1 = splat<1>(bv), u2 = splat<2>(bv), u3 = splat<3>(bv);
                acc2[r][0] = dp4a(t[0], u0, acc2[r][0]);
                acc2[r][0] = dp4a(t[1], u1, acc2[r][0]);
                acc2[r][0] = dp4a(t[2], u2, acc2[r][0]);
                acc2[r][0] = dp4a(t[3], u3, acc2[r][0]);
                acc2[r][1] = dp4a(t[4], u0, acc2[r][1]);
                acc2[r][1] = dp4a(t[5], u1, acc2[r][1]);
                acc2[r][1] = dp4a(t[6], u2, acc2[r][1]);
                acc2[r][1] = dp4a(t[7], u3, acc2[r][1]);
            }
        }
    }

    // Fold the four k splits of a tile: lanes 4c + q are contiguous.
    #pragma unroll
    for (int r = 0; r < ROWS; ++r)
    {
        if (!full && r >= rows_out) break;
        acc[r][0] += __shfl_xor_sync(0xffffffffull, acc[r][0], 1);
        acc[r][0] += __shfl_xor_sync(0xffffffffull, acc[r][0], 2);
        acc[r][1] += __shfl_xor_sync(0xffffffffull, acc[r][1], 1);
        acc[r][1] += __shfl_xor_sync(0xffffffffull, acc[r][1], 2);
        if constexpr (RESIDUAL)
        {
            acc2[r][0] += __shfl_xor_sync(0xffffffffull, acc2[r][0], 1);
            acc2[r][0] += __shfl_xor_sync(0xffffffffull, acc2[r][0], 2);
            acc2[r][1] += __shfl_xor_sync(0xffffffffull, acc2[r][1], 1);
            acc2[r][1] += __shfl_xor_sync(0xffffffffull, acc2[r][1], 2);
        }
    }

    const float k_inv = __half2float(__ushort_as_half(0x1eee));
    const float k_bias = __half2float(__ushort_as_half(0xc931));
    // 1024 * k_inv + k_bias is the CUDA affine term; the extra 512 * k_inv is the byte sign flip
    const float aff = 1536.0f * k_inv + k_bias;

    __shared__ float sh_red[WK][ROWS][COLS];
    if (q == 0)
    {
        const int c = tile * 16 + c0;
        #pragma unroll
        for (int r = 0; r < ROWS; ++r)
        {
            if (!full && r >= rows_out) break;
            const float qf = qscale[r];
            const float q2f = qf * (1.0f / 254.0f);
            const float s0 = qf * (float) acc[r][0] + (RESIDUAL ? q2f * (float) acc2[r][0] : 0.0f);
            const float s1 = qf * (float) acc[r][1] + (RESIDUAL ? q2f * (float) acc2[r][1] : 0.0f);
            sh_red[ksplit][r][c] = k_inv * s0;
            sh_red[ksplit][r][c + 8] = k_inv * s1;
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
    {
        const int r = idx / COLS;
        const int c = idx - r * COLS;
        float sum = sh_red[0][r][c];
        #pragma unroll
        for (int j = 1; j < WK; ++j) sum += sh_red[j][r][c];
        const float qf = qscale[r];
        const float q2f = qf * (1.0f / 254.0f);
        const float suma = qf * (float) qsum[r] + (RESIDUAL ? q2f * (float) qsum2[r] : 0.0f);
        const int col = blockIdx.x * COLS + c;
        ((float*) C)[(size_t) r * size_n + col] = sum + aff * suma;
    }
}

// ---------------------------------------------------------------------------------------------
// Host side: launch selection and the four-stage pipeline.

namespace exl3_gemv_int8_hip_host {

// Per-device arch gate: gfx12xx (RDNA4) only. The kernel is tuned and validated there; RDNA3
// has dp4a as well but is unmeasured, and everything older lacks it. 0 = unknown, 1 = yes,
// 2 = no (idempotent, so the race between two first calls is benign).
inline bool arch_ok(int device)
{
    static int8_t cached[MAX_DEVICES] = {};
    if (device < 0 || device >= MAX_DEVICES) return false;
    int8_t& slot = cached[device];
    if (slot == 0)
    {
        cudaDeviceProp prop;
        const bool ok = cudaGetDeviceProperties(&prop, device) == cudaSuccess
                     && prop.major == 12 && prop.warpSize == 32;
        if (!ok) (void) cudaGetLastError();
        slot = ok ? 1 : 2;
    }
    return slot == 1;
}

struct Cfg
{
    int wk;
    int wt;
};

template <int bits, int ROWS, bool RESIDUAL>
inline void* select_cfg(int cfg)
{
    if (cfg == 0) return (void*) exl3_gemv_int8_hip_kernel<bits, ROWS, 8, 2, RESIDUAL>;
    return (void*) exl3_gemv_int8_hip_kernel<bits, ROWS, 4, 4, RESIDUAL>;
}

template <int bits, bool RESIDUAL>
inline void* select_rows(int rows, int cfg)
{
    if (rows <= 1) return select_cfg<bits, 1, RESIDUAL>(cfg);
    if (rows == 2) return select_cfg<bits, 2, RESIDUAL>(cfg);
    if (rows <= 4) return select_cfg<bits, 4, RESIDUAL>(cfg);
    return select_cfg<bits, 8, RESIDUAL>(cfg);
}

inline void* select(int bits, int rows, bool residual, int cfg)
{
    if (residual)
    {
        switch (bits)
        {
            case 2: return select_rows<2, true>(rows, cfg);
            case 3: return select_rows<3, true>(rows, cfg);
            case 4: return select_rows<4, true>(rows, cfg);
        }
        return nullptr;
    }
    switch (bits)
    {
        case 2: return select_rows<2, false>(rows, cfg);
        case 3: return select_rows<3, false>(rows, cfg);
        case 4: return select_rows<4, false>(rows, cfg);
    }
    return nullptr;
}

// Benchmark-only override of the launch geometry.
inline int cfg_override()
{
    static const int value = [] () -> int
    {
        const char* env = std::getenv("GPTQMODEL_EXL3_HIP_GEMV_INT8_CFG");
        if (!env || !*env) return -1;
        const int parsed = std::atoi(env);
        return parsed == 0 || parsed == 1 ? parsed : -1;
    }();
    return value;
}

// Largest dynamic shared-memory request the device admits. The fused quantize launch keeps one
// activation row (size_k halves) in shared memory, so longer rows decline to the fp16 GEMV.
inline int shared_limit(int device)
{
    static int cached[MAX_DEVICES] = {};
    if (device < 0 || device >= MAX_DEVICES) return 0;
    int& slot = cached[device];
    if (slot == 0)
    {
        cudaDeviceProp prop;
        slot = cudaGetDeviceProperties(&prop, device) == cudaSuccess
             ? (int) prop.sharedMemPerBlock : -1;
        if (slot <= 0) { (void) cudaGetLastError(); slot = -1; }
    }
    return slot > 0 ? slot : 0;
}

typedef void (*gemv_int8_fn)
(
    const uint16_t*, void*, const uint32_t*, const uint32_t*, const float*, const int*, const int*,
    int, int, int
);

}  // namespace exl3_gemv_int8_hip_host

// Returns 1 when the three launches were submitted, 0 when the shape/rate/device is not
// eligible (the caller keeps the fp16 HIP GEMV or the reconstruct path), -1 on a launch error
// (the error is left for the caller to report).
inline int exl3_gemv_int8_hip_launch
(
    const half* A,
    const uint16_t* B,
    void* C,
    const half* suh,
    const half* svh,
    int8_t* qa,
    int8_t* qb,
    float* qscale,
    int* qsum,
    int* qsum2,
    int size_m,
    int size_k,
    int size_n,
    int bits,
    bool residual,
    int device,
    cudaStream_t stream
)
{
    using namespace exl3_gemv_int8_hip_host;
    if (size_m < 1 || size_m > EXL3_GEMV_MAX_M) return 0;
    if (size_k % 128 || size_n % 128) return 0;
    if (bits < 2 || bits > 4) return 0;
    if (!arch_ok(device)) return 0;
    // 4 KiB of slack for the kernel's static shared state and the device's reserved LDS
    if ((int64_t) size_k * (int64_t) sizeof(half) + 4096 > (int64_t) shared_limit(device)) return 0;
    if (!qa || !qscale || !qsum || (residual && !qb)) return 0;

    const int cfg = cfg_override() >= 0 ? cfg_override() : (size_n >= 8192 ? 1 : 0);
    const Cfg geometry = cfg == 0 ? Cfg{8, 2} : Cfg{4, 4};
    const int grid = (size_n / 16) / geometry.wt;
    if (grid < 1) return 0;
    void* kernel = select(bits, size_m, residual, cfg);
    if (!kernel) return 0;

    const int warps_out = size_m * size_n / 128;
    constexpr float R_SCALE = 0.088388347648f;   // 1 / sqrt(128)
    const size_t sh_bytes = (size_t) size_k * sizeof(half);

    // 1. input Hadamard + suh (staged in shared memory), quantize, row max and exact int sums
    if (residual)
        exl3_gemv_int8_hip_quant_kernel<true><<<size_m, 512, sh_bytes, stream>>>
        (
            A, suh, qa, qb, qscale, qsum, qsum2, size_k
        );
    else
        exl3_gemv_int8_hip_quant_kernel<false><<<size_m, 512, sh_bytes, stream>>>
        (
            A, suh, qa, qb, qscale, qsum, qsum2, size_k
        );
    if (cudaPeekAtLastError() != cudaSuccess) return -1;

    // 2. GEMV + affine epilogue. The kernel is selected at run time, so it goes through the
    // launch API with an argument array instead of a triple-chevron call (hipify rewrites a
    // triple-chevron launch of a function pointer into an invalid hipLaunchKernelGGL).
    const uint16_t* b_ptr = B;
    void* c_ptr = C;
    const uint32_t* qa_ptr = reinterpret_cast<const uint32_t*>(qa);
    const uint32_t* qb_ptr = reinterpret_cast<const uint32_t*>(qb);
    const float* qs_ptr = qscale;
    const int* q1_ptr = qsum;
    const int* q2_ptr = qsum2;
    void* kernel_args[] =
    {
        (void*) &b_ptr, (void*) &c_ptr, (void*) &qa_ptr, (void*) &qb_ptr,
        (void*) &qs_ptr, (void*) &q1_ptr, (void*) &q2_ptr,
        (void*) &size_m, (void*) &size_k, (void*) &size_n
    };
    if (cudaLaunchKernel
    (
        kernel,
        dim3(grid),
        dim3(32 * geometry.wk * geometry.wt),
        kernel_args,
        0,
        stream
    ) != cudaSuccess) return -1;

    // 3. output Hadamard + svh (the CUDA kernel's epilogue)
    exl3_gemv_hip_had_out_kernel<true><<<(warps_out + 7) / 8, 256, 0, stream>>>
    (
        C, C, svh, size_n, warps_out * 128, R_SCALE
    );
    if (cudaPeekAtLastError() != cudaSuccess) return -1;
    return 1;
}

#endif
