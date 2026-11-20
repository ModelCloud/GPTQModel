// Adapted from ExLlamaV3 v1.6.7 (8c593e4), copyright TurboDerp and contributors, MIT.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
#include "exl3_gemv_int8_instances.cuh"
#include "../exl3_gemv_int8_kernel.cuh"

void* exl3_gemv_int8_sq_sel_k5(int M, bool c_fp32, bool residual)
{
    #define SELM_(M_) \
        if (c_fp32)  return residual ? (void*) exl3_gemv_int8_sq_kernel<5, M_, true, true> \
                                     : (void*) exl3_gemv_int8_sq_kernel<5, M_, true, false>; \
        else         return residual ? (void*) exl3_gemv_int8_sq_kernel<5, M_, false, true> \
                                     : (void*) exl3_gemv_int8_sq_kernel<5, M_, false, false>;
    switch (M)
    {
        case 1: { SELM_(1) }
        case 2: { SELM_(2) }
    }
    #undef SELM_
    return nullptr;
}
#endif
