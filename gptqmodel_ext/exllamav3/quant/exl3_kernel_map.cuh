#pragma once

#include "arch.cuh"
#include "q4nx.cuh"
#include "r4nx.cuh"

#define EXL3_GEMM_T_ARGS \
    const int bits, \
    const bool half_k, \
    const bool c_fp32, \
    const int cb, \
    const int TILESIZE_M, \
    const int TILESIZE_K, \
    const int TILESIZE_N, \
    const int SH_STAGES, \
    const int FRAG_STAGES

#define EXL3_GEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t* __restrict__ B, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half* __restrict__ suh, \
    half* __restrict__ A_had, \
    const half* __restrict__ svh

typedef void (*fp_exl3_gemm_kernel) (EXL3_GEMM_ARGS);

#define EXL3_GEMM_BASE_THREADS 256
#define SMEM_MAX EXL3_SMEM_MAX_DEFAULT
#define EXL3_GEMM_TILESIZE_M  16

#define EXL3_GEMM_SHAPE_1     EXL3_GEMM_TILESIZE_M,     16,    128,     6,     5
#define EXL3_GEMM_SHAPE_2     EXL3_GEMM_TILESIZE_M,     32,    128,     4,     3
#define EXL3_GEMM_SHAPE_3     EXL3_GEMM_TILESIZE_M,     32,    256,     4,     3
#define EXL3_GEMM_SHAPE_4     EXL3_GEMM_TILESIZE_M,     16,    512,     4,     3
#define EXL3_GEMM_SHAPE_5     32,                      32,    128,     3,     3
#define EXL3_GEMM_SHAPE_6     48,                      32,    128,     3,     3
#define EXL3_GEMM_SHAPE_7     64,                      32,    128,     3,     3

#define EXL3_GEMM_TILESIZE_K  0, 16, 32, 32, 16, 32, 32, 32
#define EXL3_GEMM_TILESIZE_N  0, 128, 128, 256, 512, 128, 128, 128
#define EXL3_GEMM_BLOCKDIM    0, 256, 512, 512, 256, 512, 512, 512

#define EXL3_GEMM_NUM_SHAPES 7
#define EXL3_LEGACY_GEMM_NUM_SHAPES 4
#define EXL3_GEMM_FOREACH_SHAPE(X, ...) \
    X(1, __VA_ARGS__) X(2, __VA_ARGS__) X(3, __VA_ARGS__) X(4, __VA_ARGS__) \
    X(5, __VA_ARGS__) X(6, __VA_ARGS__) X(7, __VA_ARGS__)

__host__ __device__ constexpr int exl3_gemm_smem_bytes(
    int tilesize_m, int tilesize_k, int tilesize_n, int stages, int,
    int bits, bool half_k, bool output_hadamard)
{
    const int sh_a = tilesize_m * tilesize_k;
    const int sh_b = (tilesize_k / 16) * (tilesize_n / 16) * (16 * bits + 8 * half_k);
    const int fragments_per_warp = 2 * (tilesize_n / 16) / (EXL3_GEMM_BASE_THREADS / 32);
    const int reduce = 4 * EXL3_GEMM_BASE_THREADS * fragments_per_warp * (tilesize_m / 16);
    const int hadamard = output_hadamard ? tilesize_m * tilesize_n : 0;
    return stages * 2 * (sh_a + sh_b) + 4 * (reduce > hadamard ? reduce : hadamard);
}

#define EXL3_GEMM_SMEM_SHAPE(_shape, _bits, _half) \
    exl3_gemm_smem_bytes(_shape, _bits, _half, true)
#define EXL3_GEMM_SMEM_CASE(S, _bits, _half) \
    case S: return EXL3_GEMM_SMEM_SHAPE(EXL3_GEMM_SHAPE_##S, _bits, _half);
inline int exl3_gemm_shape_smem(int shape, int bits, bool half_k)
{
    switch (shape)
    {
        EXL3_GEMM_FOREACH_SHAPE(EXL3_GEMM_SMEM_CASE, bits, half_k)
        default: return 0;
    }
}

// The tensor-core GEMM kernels are written against NVIDIA inline PTX
// (ldmatrix / mma.sync / cp.async) and have no ROCm implementation yet. On ROCm
// we still emit the exported kernel tables so the extension links, but every
// entry is null and `exl3_gemm` refuses to launch them with a clear error. The
// EXL3 quantization lifecycle does not use these kernels (it goes through
// `hgemm`/hipBLAS), so quantization runs at full speed on MI350X.
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

#define EXL3_GEMM_INST(S, _bits, _half, _c_fp32, cb) nullptr,
#define EXL3_LEGACY_GEMM_INST(S, _bits, _c_fp32, cb) nullptr,

#else

#define EXL3_GEMM_INST(S, _bits, _half, _c_fp32, cb) \
    exl3_gemm_kernel<_bits, _half, _c_fp32, cb, EXL3_GEMM_SHAPE_##S>,
#define EXL3_LEGACY_GEMM_INST(S, _bits, _c_fp32, cb) \
    gptq_exl3_legacy::exl3_gemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_##S>,

#endif

#define EXL3_GEMM_KERNEL_INSTANCES(_bits, _half, _c_fp32, cb) \
    nullptr, EXL3_GEMM_FOREACH_SHAPE(EXL3_GEMM_INST, _bits, _half, _c_fp32, cb)

#define EXL3_LEGACY_GEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, EXL3_LEGACY_GEMM_INST(1, _bits, _c_fp32, cb) EXL3_LEGACY_GEMM_INST(2, _bits, _c_fp32, cb) \
    EXL3_LEGACY_GEMM_INST(3, _bits, _c_fp32, cb) EXL3_LEGACY_GEMM_INST(4, _bits, _c_fp32, cb)

#define ALL_EXL3_KERNEL_EXTERNS(K) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_legacy_fp32_b##K[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_legacy_fp16_b##K[];

#define ALL_EXL3_KERNEL_INSTANCES(K) \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, true, 0) \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, true, 1) \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, true, 2) \
    }; \
    \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, false, 0) \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, false, 1) \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, false, 2) \
    };

#define ALL_EXL3_LEGACY_KERNEL_INSTANCES(K) \
    fp_exl3_gemm_kernel tfp_exl3_legacy_fp32_b##K[] = { \
        EXL3_LEGACY_GEMM_KERNEL_INSTANCES(K, true, 0) \
        EXL3_LEGACY_GEMM_KERNEL_INSTANCES(K, true, 1) \
        EXL3_LEGACY_GEMM_KERNEL_INSTANCES(K, true, 2) \
    }; \
    fp_exl3_gemm_kernel tfp_exl3_legacy_fp16_b##K[] = { \
        EXL3_LEGACY_GEMM_KERNEL_INSTANCES(K, false, 0) \
        EXL3_LEGACY_GEMM_KERNEL_INSTANCES(K, false, 1) \
        EXL3_LEGACY_GEMM_KERNEL_INSTANCES(K, false, 2) \
    };

#define EXL3_FRACTIONAL_KERNEL_EXTERNS(K) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_h##K[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_h##K[];

#define EXL3_FRACTIONAL_KERNEL_INSTANCES(K) \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_h##K[] = { EXL3_GEMM_KERNEL_INSTANCES(K, true, true, 2) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_h##K[] = { EXL3_GEMM_KERNEL_INSTANCES(K, true, false, 2) };

EXL3_FRACTIONAL_KERNEL_EXTERNS(1)
EXL3_FRACTIONAL_KERNEL_EXTERNS(2)
EXL3_FRACTIONAL_KERNEL_EXTERNS(3)

#define EXL3_Q4NX_KERNEL_EXTERNS(RATE) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_q4nx_##RATE[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_q4nx_##RATE[];

#define EXL3_Q4NX_KERNEL_INSTANCES(RATE, K, HALF) \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_q4nx_##RATE[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, HALF, true, EXL3_Q4NX_CODEBOOK) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_q4nx_##RATE[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, HALF, false, EXL3_Q4NX_CODEBOOK) };

EXL3_Q4NX_KERNEL_EXTERNS(w25)
EXL3_Q4NX_KERNEL_EXTERNS(w30)
EXL3_Q4NX_KERNEL_EXTERNS(w35)

#define EXL3_R4NX_KERNEL_EXTERNS(RATE) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_r4nx_##RATE[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_r4nx_##RATE[];

#define EXL3_R4NX_KERNEL_INSTANCES(RATE, K, HALF) \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_r4nx_##RATE[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, HALF, true, EXL3_R4NX_CODEBOOK) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_r4nx_##RATE[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, HALF, false, EXL3_R4NX_CODEBOOK) };

EXL3_R4NX_KERNEL_EXTERNS(w25)
EXL3_R4NX_KERNEL_EXTERNS(w30)
EXL3_R4NX_KERNEL_EXTERNS(w35)
EXL3_R4NX_KERNEL_EXTERNS(w40)

struct TResult
{
    fp_exl3_gemm_kernel kernel;
    int shape_idx;
    int num_sms;
    int block_dim;
};

TResult* select_exl3_gemm_kernel_tuned
(
    const int cc,
    const int size_k,
    const int size_n,
    const int K,
    const bool c_fp32,
    const int force_shape_idx,
    const int force_num_sms,
    const int cb,
    const bool half_k,
    const bool fused_gemm,
    const int size_m,
    const int smem_max
);
