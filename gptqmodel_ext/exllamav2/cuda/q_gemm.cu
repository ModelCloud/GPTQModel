#include <ATen/ATen.h>
#include "q_gemm.cuh"
#include "util.cuh"
#include "matrix_view.cuh"
#include "../config.h"

#include "quant/qdq_2.cuh"
#include "quant/qdq_3.cuh"
#include "quant/qdq_4.cuh"
#include "quant/qdq_5.cuh"
#include "quant/qdq_6.cuh"
#include "quant/qdq_8.cuh"

#define BLOCK_KN_SIZE 128
#ifndef EXLLAMAV2_SPLIT_REDUCE_MAX_BYTES
// 32 MiB: large enough for every per-layer shape in a 70B model at the row
// counts that use split-K, small enough to stay off the critical VRAM path.
#define EXLLAMAV2_SPLIT_REDUCE_MAX_BYTES (32LL * 1024 * 1024)
#endif
#define BLOCK_M_SIZE_MAX 8
#define MAX_GROUPS_IN_BLOCK (BLOCK_KN_SIZE / 32)
#define CLEAR_N_SIZE 256

#include "q_gemm_kernel.cuh"
#include "q_gemm_kernel_gptq.cuh"

#include "compat_gemm.cuh"


// Fused split reduction + FP32->FP16 conversion. Each K split wrote a private
// partial-sum slice, so this single launch replaces both the atomic contention
// and the separate conversion pass.
__global__ void c_reduce_splits_to_half_kernel
(
    const float* __restrict__ partials,
    half* __restrict__ out,
    const int64_t split_stride,
    const int splits,
    const int64_t count
)
{
    int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float sum = partials[i];
    for (int s = 1; s < splits; ++s)
        sum += partials[(int64_t) s * split_stride + i];
    out[i] = __float2half_rn(sum);
}

static void c_reduce_splits_to_half(const float* partials, half* out, int64_t split_stride, int splits, int64_t count)
{
    const int threads = 256;
    const int64_t blocks = DIVIDE(count, threads);
    c_reduce_splits_to_half_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>
        (partials, out, split_stride, splits, count);
}

void gemm_half_q_half_cuda_part
(
    const half* a,
    QMatrix* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    int m_count,
    bool clear,
    float* c_float = nullptr,
    int splits = 1,
    int64_t c_split_stride = 0
)
{
    if (!b->is_gptq)
    {
        dim3 blockDim, gridDim;
        blockDim.x = BLOCK_KN_SIZE;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE * 4);
        gridDim.y = DIVIDE(size_m, m_count);
        gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);

        fp_gemm_half_q_half_kernel kernel = pick_gemm_half_q_half_kernel(true, m_count);

        kernel<<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>
        (
            a,
            b->cuda_q_weight,
            b->cuda_q_scale,
            b->cuda_q_scale_max,
            c,
            size_m,
            size_n,
            size_k,
            b->groups,
            b->groupsize,
            b->cuda_q_perm,
            b->rows_8,
            b->rows_6,
            b->rows_5,
            b->rows_4,
            b->rows_3,
            b->rows_2,
            clear
        );
    }
    else
    {
        dim3 blockDim, gridDim;
        blockDim.x = BLOCK_KN_SIZE;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE * 4);
        gridDim.y = DIVIDE(size_m, m_count);
        gridDim.z = splits;
        // A zero stride selects the shared-buffer atomic fallback, where every
        // K block accumulates into the same slice.

        fp_gemm_half_q_half_gptq_kernel kernel = pick_gemm_half_q_half_gptq_kernel(true, m_count);

//         DBGX((uint64_t) b->cuda_q_perm);
//         DBGI(b->rows_4);
//         DBGI(b->height);

        kernel<<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>
        (
            a,
            b->cuda_q_weight,
            b->cuda_gptq_qzeros,
            b->cuda_gptq_scales,
            c_float,
            size_m,
            size_n,
            size_k,
            b->groups,
            b->groupsize,
            b->cuda_q_perm,
            b->rows_4,
            c_split_stride,
            clear
        );
    }
}


void gemm_half_half_cuda
(
    cublasHandle_t cublas_handle,
    const half* a,
    const half* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    bool clear
)
{
    const float alpha = 1.0f;
    const float beta = clear ? 0.0f : 1.0f;
    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 size_n, size_m, size_k,
                 &alpha, b, CUDA_R_16F, size_n,
                         a, CUDA_R_16F, size_k,
                 &beta,  c, CUDA_R_16F, size_n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
}

void gemm_half_q_half_cuda
(
    cublasHandle_t cublas_handle,
    const half* a,
    QMatrix* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    bool clear
)
{
    // Quantized matmul. This entry point is called only for activations and row
    // counts that the ext_*.cpp dispatcher already mapped to the quantized
    // kernel: it never reconstructs a dense weight, so it needs no scratch
    // buffer. The dense fallback reconstructs into a transient per-call tensor
    // in ext_*.cpp instead of model-owned storage.

    //if (clear) clear_tensor_cuda(c, size_m, size_n);

    const int split_count = DIVIDE(size_k, BLOCK_KN_SIZE);
    const int64_t split_element_stride = (int64_t) size_m * size_n;
    // Private per-split slices remove atomic contention and the pre-zeroing
    // pass, but cost split_count times the old FP32 accumulator. Fall back
    // to the single zeroed buffer with atomics when that would exceed the
    // transient budget, so wide layers keep a bounded peak allocation.
    const int64_t atomic_bytes = split_element_stride * (int64_t) sizeof(float);
    const int64_t split_bytes = atomic_bytes * split_count;
    const bool private_splits = split_bytes <= EXLLAMAV2_SPLIT_REDUCE_MAX_BYTES;
    // The kernel always launches split_count K blocks; a zero element
    // stride means "one shared buffer, accumulate with atomics".
    const int64_t c_split_stride = private_splits ? split_element_stride : 0;

    auto float_options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    at::Tensor accum;
    if (b->is_gptq) {
        accum = private_splits
            ? at::empty({split_count * split_element_stride}, float_options)
            : at::zeros({split_element_stride}, float_options);
    }
    float* c_float = b->is_gptq ? accum.data_ptr<float>() : nullptr;

    int max_chunks = size_m / BLOCK_M_SIZE_MAX;
    int last_chunk = max_chunks * BLOCK_M_SIZE_MAX;
    int last_chunk_size = size_m - last_chunk;

    if (max_chunks)
    {
        gemm_half_q_half_cuda_part(a, b, c, last_chunk, size_n, size_k, BLOCK_M_SIZE_MAX, clear, c_float, split_count, c_split_stride);
    }

    if (last_chunk_size)
    {
        // The tail block is dispatched with its own row origin: A and C are
        // advanced by `last_chunk` rows and the partial buffer by
        // `last_chunk * size_n` elements. That offset is smaller than one
        // split slice, so split 0 still begins at the buffer base while the
        // kernel's `blockIdx.z * c_split_stride` advances by whole slices;
        // the two offsets compose without overlapping.
        gemm_half_q_half_cuda_part(a + last_chunk * size_k, b, c + last_chunk * size_n, last_chunk_size, size_n, size_k, last_chunk_size, clear, c_float ? c_float + last_chunk * size_n : nullptr, split_count, c_split_stride);
    }
    if (b->is_gptq) {
        // Private slices are summed here; the shared fallback is one slice,
        // which makes the same kernel a plain FP32->FP16 conversion.
        c_reduce_splits_to_half(
            c_float, c, c_split_stride,
            private_splits ? split_count : 1,
            (int64_t) size_m * size_n);
    }

}

__global__ void clear_kernel
(
    half* __restrict__ c,
    const int size_m,
    const int size_n
)
{
    int m = blockIdx.y;
    int n = (blockIdx.x * CLEAR_N_SIZE + threadIdx.x) * 8;
    if (n >= size_n) return;
    int4* c_ptr = (int4*)(c + m * size_n + n);
    *c_ptr = {};
}

void clear_tensor_cuda
(
    half* c,
    int size_m,
    int size_n
)
{
    return;
    dim3 blockDim, gridDim;
    blockDim.x = CLEAR_N_SIZE;
    blockDim.y = 1;
    gridDim.x = DIVIDE(size_n / 8, CLEAR_N_SIZE);
    gridDim.y = size_m;
    clear_kernel<<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(c, size_m, size_n);
}
