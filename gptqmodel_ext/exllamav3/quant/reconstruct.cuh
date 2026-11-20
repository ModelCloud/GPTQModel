#pragma once

#include <ATen/Tensor.h>

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
);

void reconstruct_frac(at::Tensor unpacked, at::Tensor packed, int K);

#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
// Writes the exact unnormalized Q4NX integers as FP16 values.
void reconstruct_q4nx(at::Tensor unpacked, at::Tensor packed, int K, bool half_k);
#endif
