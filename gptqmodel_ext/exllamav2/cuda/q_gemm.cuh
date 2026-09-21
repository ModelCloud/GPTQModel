#ifndef _q_gemm_cuh
#define _q_gemm_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "q_matrix.cuh"

void gemm_half_q_half_cuda
(
    cublasHandle_t cublas_handle,
    const half* a,
    QMatrix* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    bool clear = false,
    half* reconstruct = NULL,
    bool force_cuda = false
);

// GPTQ-only dispatch entry point.  Keep the historical declaration above
// unchanged because q_gemm_awq.cu includes this shared header.
void gemm_half_q_half_cuda_path
(
    cublasHandle_t cublas_handle,
    const half* a,
    QMatrix* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    bool clear = false,
    half* reconstruct = NULL,
    bool force_cuda = false,
    int path = -1
);

void clear_tensor_cuda
(
    half* c,
    int size_m,
    int size_n
);

#endif
