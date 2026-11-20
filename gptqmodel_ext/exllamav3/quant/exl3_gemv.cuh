#pragma once

#include "../../torch_stable_compat.h"
#include <cuda_runtime.h>

// Returns false without submitting work when the shape or device declines.
bool exl3_gemv_try_launch(
    void** args, int m, int k, int n, int bits, bool half_k, int cb,
    bool output_fp32, int device, cudaStream_t stream);

bool exl3_gemv_int8(
    const at::Tensor& input, const at::Tensor& trellis, at::Tensor& output,
    const at::Tensor& su, const at::Tensor& xh, const at::Tensor& sv,
    int mode, cudaStream_t stream);
