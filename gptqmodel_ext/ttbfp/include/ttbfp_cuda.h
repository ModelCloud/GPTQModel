// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
/*
 * ttbfp_cuda - optional CUDA kernels for bf8_b / bf4_b / bf2_b.
 *
 * These kernels reproduce exactly the same numerics as src/ttbfp_cpu.cpp.
 * They are intended for non-Tenstorrent NVIDIA GPUs; the CPU library remains
 * the reference implementation.
 *
 * Build (requires nvcc):
 *   make cuda
 */
#ifndef TTBFP_CUDA_H
#define TTBFP_CUDA_H

#include <cuda_runtime.h>
#include <stdint.h>

#include "ttbfp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* All pointers are device pointers.  Return 0 on success, else cudaError_t. */
int ttbfp_cuda_quantize(ttbfp_fmt fmt,
                        const float *d_src, int64_t rows, int64_t cols,
                        uint8_t *d_exps, uint8_t *d_data,
                        cudaStream_t stream);

int ttbfp_cuda_dequantize(ttbfp_fmt fmt,
                          const uint8_t *d_exps, const uint8_t *d_data,
                          int64_t rows, int64_t cols, float *d_dst,
                          int flush_denormals, cudaStream_t stream);

/*
 * d_C[M,N] = d_A[M,K] @ dequant(d_B)[K,N]
 * d_A is row-major fp32 with leading dimension lda.
 * d_B is the packed [K,N] matrix (exps: K*nblk, data: K*nblk*datum_bytes).
 */
int ttbfp_cuda_matmul(ttbfp_fmt fmt,
                      const float *d_A, int64_t lda,
                      const uint8_t *d_exps, const uint8_t *d_data,
                      int64_t M, int64_t K, int64_t N,
                      float *d_C, int64_t ldc,
                      int flush_denormals, cudaStream_t stream);

const char *ttbfp_cuda_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TTBFP_CUDA_H */
