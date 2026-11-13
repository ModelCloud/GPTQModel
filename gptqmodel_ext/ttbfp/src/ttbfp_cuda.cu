// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
/*
 * ttbfp_cuda - CUDA kernels for Tenstorrent bf8_b / bf4_b / bf2_b.
 *
 * Numerics match src/ttbfp_cpu.cpp and the tt-llk reference packers:
 *   bf8_b = Bfp8_b (E8M6), bf4_b = Bfp4_b (E8M2), bf2_b = Bfp2_b (E8M0)
 *   16-element blocks, one shared 8-bit exponent (bias 127), sign-magnitude.
 *
 * Layouts are identical to the CPU API:
 *   exps[rows * nblk], data[rows * nblk * 16, 8 or 4].
 *
 * This file is optional; it needs nvcc and a CUDA-capable GPU.  The CPU
 * implementation is the tested reference.
 */

#include "ttbfp_cuda.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

__device__ __forceinline__ int magnitude_bits_dev(ttbfp_fmt fmt) {
    return fmt == TTBFP_BF8_B ? 7 : (fmt == TTBFP_BF4_B ? 3 : 1);
}

__device__ __forceinline__ int datum_bytes_dev(ttbfp_fmt fmt) {
    return fmt == TTBFP_BF8_B ? 16 : (fmt == TTBFP_BF4_B ? 8 : 4);
}

__device__ __forceinline__ float decode_datum(ttbfp_fmt fmt, const uint8_t *data,
                                              int64_t block_index, int lane,
                                              uint8_t e) {
    if (e == 0) return 0.0f;
    const int mb = magnitude_bits_dev(fmt);
    const int mask = (1 << mb) - 1;
    const int sbit = 1 << mb;
    const int db = datum_bytes_dev(fmt);
    uint8_t v;
    if (fmt == TTBFP_BF8_B) {
        v = data[block_index * db + lane];
    } else if (fmt == TTBFP_BF4_B) {
        uint8_t b = data[block_index * db + (lane >> 1)];
        v = (lane & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0F);
    } else {
        uint8_t b = data[block_index * db + (lane >> 2)];
        v = (uint8_t)((b >> ((lane & 3) * 2)) & 0x03);
    }
    const int sh = (int)e - (126 + mb);
    float f;
    if (sh < -126) {
        // The direct scale is subnormal; split off 2^-6 so the first
        // multiply stays normal even with FTZ/DAZ enabled.
        f = (float)(v & mask) * ldexpf(1.0f, sh + 6);
        f *= 0x1p-6f;
    } else {
        f = (float)(v & mask) * ldexpf(1.0f, sh);
    }
    return (v & sbit) ? -f : f;
}

__device__ __forceinline__ void quantize_block_dev(const float *x, int n, int mb,
                                                   uint8_t *exp_out,
                                                   uint8_t *dat_out) {
    int32_t exps[TTBFP_BLOCK];
    int32_t mants[TTBFP_BLOCK];
    int32_t signs[TTBFP_BLOCK];
    int maxe = 0;

    for (int i = 0; i < n; ++i) {
        uint32_t b = __float_as_uint(x[i]);
        uint32_t bf = b >> 16;
        if (((bf >> 7) & 0xFF) == 0xFF && (bf & 0x7F) != 0) bf |= 0x0040u;
        int e = (int)((bf >> 7) & 0xFF);
        exps[i] = e;
        signs[i] = (int)(bf >> 15);
        mants[i] = (int)(((bf & 0x7F) >> 1) | 0x40);
        if (e > maxe) maxe = e;
    }

    *exp_out = (uint8_t)maxe;
    uint8_t dat[TTBFP_BLOCK];
    for (int i = 0; i < TTBFP_BLOCK; ++i) dat[i] = 0;

    for (int i = 0; i < n; ++i) {
        int d = maxe - exps[i];
        int shift = d > 8 ? 8 : d;
        int guard = 0;
        if (d > 0) {
            int gs = d - 1;
            if (gs > 8) gs = 8;
            guard = (mants[i] >> gs) & 1;
        }
        int q = ((mants[i] >> shift) + guard) & 0x7F;
        q >>= (7 - mb);
        dat[i] = q ? (uint8_t)((signs[i] << mb) | q) : (uint8_t)0;
    }

    if (mb == 7) {
        for (int i = 0; i < n; ++i) dat_out[i] = dat[i];
    } else if (mb == 3) {
        int nbytes = (n + 1) / 2;
        for (int i = 0; i < nbytes; ++i) {
            uint8_t lo = dat[2 * i];
            uint8_t hi = (2 * i + 1 < n) ? dat[2 * i + 1] : 0;
            dat_out[i] = (uint8_t)(lo | (hi << 4));
        }
    } else {
        int nbytes = (n + 3) / 4;
        for (int i = 0; i < nbytes; ++i) {
            uint8_t v = 0;
            for (int k = 0; k < 4; ++k) {
                int idx = 4 * i + k;
                if (idx < n) v = (uint8_t)(v | (dat[idx] << (2 * k)));
            }
            dat_out[i] = v;
        }
    }
}

/* ------------------------------------------------------------------ */
/* quantize: one thread per 16-element block                           */
/* ------------------------------------------------------------------ */

__global__ void quantize_kernel(ttbfp_fmt fmt, const float *src, int64_t rows,
                                int64_t cols, uint8_t *exps, uint8_t *data) {
    int64_t nblk = (cols + TTBFP_BLOCK - 1) / TTBFP_BLOCK;
    int64_t total = rows * nblk;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t r = idx / nblk;
    int64_t j = idx - r * nblk;
    const float *x = src + r * cols + j * TTBFP_BLOCK;
    int n = (int)min((int64_t)TTBFP_BLOCK, cols - j * TTBFP_BLOCK);
    int db = datum_bytes_dev(fmt);
    uint8_t *d = data + (r * nblk + j) * db;
    for (int i = 0; i < db; ++i) d[i] = 0;
    quantize_block_dev(x, n, magnitude_bits_dev(fmt), exps + (r * nblk + j), d);
}

/* ------------------------------------------------------------------ */
/* dequantize: one thread per 16-element block                         */
/* ------------------------------------------------------------------ */

__global__ void dequantize_kernel(ttbfp_fmt fmt, const uint8_t *exps,
                                  const uint8_t *data, int64_t rows, int64_t cols,
                                  float *dst, int ftz) {
    int64_t nblk = (cols + TTBFP_BLOCK - 1) / TTBFP_BLOCK;
    int64_t total = rows * nblk;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t r = idx / nblk;
    int64_t j = idx - r * nblk;
    uint8_t e = exps[r * nblk + j];
    int64_t block_index = r * nblk + j;
    int n = (int)min((int64_t)TTBFP_BLOCK, cols - j * TTBFP_BLOCK);
    float *out = dst + r * cols + j * TTBFP_BLOCK;
    for (int i = 0; i < n; ++i) {
        float v = decode_datum(fmt, data, block_index, i, e);
        if (ftz && e < 7 && fabsf(v) < FLT_MIN) v = 0.0f;
        out[i] = v;
    }
}

/* ------------------------------------------------------------------ */
/* matmul: one 16x16 output tile per block, 16x16 threads              */
/* ------------------------------------------------------------------ */

__global__ void matmul_kernel(ttbfp_fmt fmt, const float *A, int64_t lda,
                              const uint8_t *exps, const uint8_t *data,
                              int64_t M, int64_t K, int64_t N,
                              float *C, int64_t ldc, int ftz) {
    __shared__ float sA[TTBFP_BLOCK][TTBFP_BLOCK];
    __shared__ float sB[TTBFP_BLOCK][TTBFP_BLOCK];

    const int tx = threadIdx.x;  // n lane
    const int ty = threadIdx.y;  // m lane
    const int64_t m0 = (int64_t)blockIdx.y * TTBFP_BLOCK;
    const int64_t n0 = (int64_t)blockIdx.x * TTBFP_BLOCK;
    const int64_t nblk = (N + TTBFP_BLOCK - 1) / TTBFP_BLOCK;

    float acc = 0.0f;
    for (int64_t k0 = 0; k0 < K; k0 += TTBFP_BLOCK) {
        // A tile: sA[m][k]
        {
            int64_t m = m0 + ty;
            int64_t k = k0 + tx;
            sA[ty][tx] = (m < M && k < K) ? A[m * lda + k] : 0.0f;
        }
        // B tile: sB[k][n], decoded from the packed representation.
        {
            int64_t k = k0 + ty;
            int64_t n = n0 + tx;
            float v = 0.0f;
            if (k < K && n < N) {
                int64_t j = n / TTBFP_BLOCK;
                int lane = (int)(n - j * TTBFP_BLOCK);
                uint8_t e = exps[k * nblk + j];
                v = decode_datum(fmt, data, k * nblk + j, lane, e);
                if (ftz && e < 7 && fabsf(v) < FLT_MIN) v = 0.0f;
            }
            sB[ty][tx] = v;
        }
        __syncthreads();

        for (int kk = 0; kk < TTBFP_BLOCK; ++kk) {
            acc = fmaf(sA[ty][kk], sB[kk][tx], acc);
        }
        __syncthreads();
    }

    int64_t m = m0 + ty;
    int64_t n = n0 + tx;
    if (m < M && n < N) {
        C[m * ldc + n] = acc;
    }
}

inline int check(cudaError_t e) {
    return (int)e;
}

}  // namespace

extern "C" {

int ttbfp_cuda_quantize(ttbfp_fmt fmt,
                        const float *d_src, int64_t rows, int64_t cols,
                        uint8_t *d_exps, uint8_t *d_data,
                        cudaStream_t stream) {
    if (!d_src || !d_exps || !d_data || rows <= 0 || cols <= 0) return -1;
    int64_t nblk = (cols + TTBFP_BLOCK - 1) / TTBFP_BLOCK;
    int64_t total = rows * nblk;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    quantize_kernel<<<blocks, threads, 0, stream>>>(fmt, d_src, rows, cols,
                                                    d_exps, d_data);
    return check(cudaGetLastError());
}

int ttbfp_cuda_dequantize(ttbfp_fmt fmt,
                          const uint8_t *d_exps, const uint8_t *d_data,
                          int64_t rows, int64_t cols, float *d_dst,
                          int flush_denormals, cudaStream_t stream) {
    if (!d_exps || !d_data || !d_dst || rows <= 0 || cols <= 0) return -1;
    int64_t nblk = (cols + TTBFP_BLOCK - 1) / TTBFP_BLOCK;
    int64_t total = rows * nblk;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    dequantize_kernel<<<blocks, threads, 0, stream>>>(fmt, d_exps, d_data, rows,
                                                      cols, d_dst,
                                                      flush_denormals);
    return check(cudaGetLastError());
}

int ttbfp_cuda_matmul(ttbfp_fmt fmt,
                      const float *d_A, int64_t lda,
                      const uint8_t *d_exps, const uint8_t *d_data,
                      int64_t M, int64_t K, int64_t N,
                      float *d_C, int64_t ldc,
                      int flush_denormals, cudaStream_t stream) {
    if (!d_A || !d_exps || !d_data || !d_C || M <= 0 || K <= 0 || N <= 0) return -1;
    dim3 block(16, 16);
    dim3 grid((unsigned)((N + 15) / 16), (unsigned)((M + 15) / 16));
    matmul_kernel<<<grid, block, 0, stream>>>(fmt, d_A, lda, d_exps, d_data,
                                              M, K, N, d_C, ldc,
                                              flush_denormals);
    return check(cudaGetLastError());
}

const char *ttbfp_cuda_version(void) {
    return "0.1.0";
}

}  // extern "C"
