// Adapted from ExLlamaV3 v1.6.7 (8c593e4), copyright TurboDerp and contributors, MIT.
#pragma once
#include "exl3_kernel_map.cuh"
#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"

template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_gemm_kernel(EXL3_GEMM_ARGS)
{
    auto grid = cg::this_grid();

    if (guard_mode)
    {
        // The guarded prologue always writes A_had, including the
        // no-checkpoint-scale case where it only narrows the activation.
        if (guard_mode == 2)
        {
            const __nv_bfloat16* A_bf16 = reinterpret_cast<const __nv_bfloat16*>(A);
            exl3_guard_input<false, __nv_bfloat16, false>
                (A_bf16, A_had, nullptr, row_amax, size_m, size_k, guard_bound, grid);
        }
        else if (guard_mode == 3)
        {
            const __nv_bfloat16* A_bf16 = reinterpret_cast<const __nv_bfloat16*>(A);
            exl3_guard_input<true, __nv_bfloat16, true>
                (A_bf16, A_had, suh, row_amax, size_m, size_k, guard_bound, grid);
        }
        else if (suh)
        {
            exl3_guard_input<false, __half, true>
                (A, A_had, suh, row_amax, size_m, size_k, guard_bound, grid);
        }
        else
        {
            exl3_guard_input<false, __half, false>
                (A, A_had, nullptr, row_amax, size_m, size_k, guard_bound, grid);
        }
        A = A_had;
    }
    else if (suh)
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<true, false>
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f  // 1/sqrt(128)
            );

        grid.sync();
        A = A_had;
    }

    int size_m_ = size_m;
    const half* A_ = A;
    void* C_ = C;
    int row_base = 0;

    while (size_m_ > 0)
    {
        exl3_gemm_kernel_inner
        <bits, half_k, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, true>
        (A_, B, C_, MIN(size_m_, TILESIZE_M), size_k, size_n, locks, svh, 0,
         row_amax, guard_out, guard_bias, guard_mode, guard_bias_mode, guard_bound, row_base);

        A_ += TILESIZE_M * size_k;
        if constexpr (c_fp32) C_ = (void*) (((float*) C_) + TILESIZE_M * size_n);
        else                  C_ = (void*) (((half*) C_) + TILESIZE_M * size_n);
        size_m_ -= TILESIZE_M;
        row_base += TILESIZE_M;

        if (size_m_ > 0 || svh)
            grid.sync();
    }
}
#endif
