#pragma once
#include "../../torch_stable_compat.h"


int exl3_gemm(
    const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
    const at::Tensor& suh, const at::Tensor& A_had, const at::Tensor& svh,
    bool mcg, bool mul1, bool fp16_gemv, int64_t int8_gemv, bool fused_gemm, bool q4nx);
