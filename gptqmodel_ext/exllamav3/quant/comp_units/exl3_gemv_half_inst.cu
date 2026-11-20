// Adapted from ExLlamaV3 v1.6.7 (8c593e4), copyright TurboDerp and contributors, MIT.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../exl3_gemv_kernel.cuh"

// Half-integer bitrate instances of the small-m GEMV kernel: 1.5 / 2.5 / 3.5 bpw, mul1 codebook
void* exl3_gemv_select_kernel_half(int bits, bool c_fp32, int mmode, int cfg, bool smem)
{
    #define SEL(bits_, fp32_, mm_, cfg_, sm_) \
        if (bits == bits_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_) \
            return (void*) exl3_gemv_kernel<bits_, fp32_, 2, mm_, cfg_, sm_, true>;
    #define SEL_GRID(bits_, sm_) \
        SEL(bits_, false, 0, 0, sm_) SEL(bits_, false, 0, 1, sm_) \
        SEL(bits_, false, 1, 0, sm_) SEL(bits_, false, 1, 1, sm_) \
        SEL(bits_, true,  0, 0, sm_) SEL(bits_, true,  0, 1, sm_) \
        SEL(bits_, true,  1, 0, sm_) SEL(bits_, true,  1, 1, sm_)
    SEL_GRID(1, false) SEL_GRID(1, true)
    SEL_GRID(2, false) SEL_GRID(2, true)
    SEL_GRID(3, false) SEL_GRID(3, true)
    #undef SEL_GRID
    #undef SEL
    return nullptr;
}

// Q4NX uses AlexBrandes's zero-prefixed windows, including Qubitium's
// 3.5 bpw extension. Every instance writes a normalized FP32 result.
void* exl3_gemv_select_kernel_q4nx(int bits, bool half_k, int mmode, int cfg)
{
    #define SEL(bits_, half_, rows_, cfg_) \
        if (bits == bits_ && half_k == half_ && mmode == rows_ && cfg == cfg_) \
            return (void*) exl3_gemv_kernel<bits_, true, EXL3_Q4NX_CODEBOOK, rows_, cfg_, false, half_, true>;
    #define SEL_RATE(bits_, half_) \
        SEL(bits_, half_, 0, 0) SEL(bits_, half_, 0, 1) SEL(bits_, half_, 1, 0) SEL(bits_, half_, 1, 1)
    SEL_RATE(2, true)
    SEL_RATE(3, false)
    SEL_RATE(3, true)
    #undef SEL_RATE
    #undef SEL
    return nullptr;
}
#endif
