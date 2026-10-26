#pragma once

#define EXL3_GEMM_T_ARGS \
    const int bits, \
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

#define EXL3_GEMM_SHAPE_1     16,     16,    128,     6,     5
#define EXL3_GEMM_SHAPE_2     16,     32,    128,     4,     3
#define EXL3_GEMM_SHAPE_3     16,     32,    256,     4,     3
#define EXL3_GEMM_SHAPE_4     16,     16,    512,     4,     3

#define EXL3_GEMM_TILESIZE_K  0, 16, 32, 32, 16
#define EXL3_GEMM_TILESIZE_N  0, 128, 128, 256, 512
#define EXL3_GEMM_BLOCKDIM  0, 256, 512, 512, 256

#define EXL3_GEMM_NUM_SHAPES 4

// The tensor-core GEMM kernels are written against NVIDIA inline PTX
// (ldmatrix / mma.sync / cp.async) and have no ROCm implementation yet. On ROCm
// we still emit the exported kernel tables so the extension links, but every
// entry is null and `exl3_gemm` refuses to launch them with a clear error. The
// EXL3 quantization lifecycle does not use these kernels (it goes through
// `hgemm`/hipBLAS), so quantization runs at full speed on MI350X.
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

#define EXL3_GEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, nullptr, nullptr, nullptr, nullptr

#else

#define EXL3_GEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, \
    exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
    exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_2>, \
    exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_3>, \
    exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_4>

#endif

#define ALL_EXL3_KERNEL_EXTERNS(K) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K[]; \

#define ALL_EXL3_KERNEL_INSTANCES(K) \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, true, 0), \
        EXL3_GEMM_KERNEL_INSTANCES(K, true, 1), \
        EXL3_GEMM_KERNEL_INSTANCES(K, true, 2) \
    }; \
    \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, 0), \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, 1), \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, 2) \
    };

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
    const int cb
);
