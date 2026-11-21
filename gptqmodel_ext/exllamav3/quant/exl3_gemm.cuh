#pragma once

#include <ATen/Tensor.h>

// Folded FP16/BF16 input-range guard (#5942). When set, the GEMM reduces each
// row's bound in its input-Hadamard pass, applies the row's power-of-two scale
// while narrowing the activation, and writes the final-dtype result with the
// scale and the bias folded into its epilogue. `C` stays the FP32 accumulation
// buffer, so the split-K partial sums keep their precision.
struct Exl3Guard
{
    // Per-row bound; filled by the kernel, so it is only allocated here.
    at::Tensor row_amax;
    // Final-dtype result, same shape as C.
    at::Tensor out;
    // 1 FP16, 2 BF16 with an FP32 bound, 3 BF16 with an FP64 bound.
    int mode = 0;
    // Bias element type: 0 none, 1 FP16, 2 BF16, 3 FP32.
    int bias_mode = 0;
    const void* bias = nullptr;
    // FP16-safe per-row amax: 65504 / sqrt(128).
    double bound = 0.0;
};

int exl3_gemm(
    const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
    const at::Tensor& suh, const at::Tensor& A_had, const at::Tensor& svh,
    bool mcg, bool mul1, bool fp16_gemv, int64_t int8_gemv, bool fused_gemm, bool q4nx,
    bool r4nx = false, const Exl3Guard* guard = nullptr);
