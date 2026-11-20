#pragma once
#include "../../torch_stable_compat.h"


void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
);

void reconstruct_frac(at::Tensor unpacked, at::Tensor packed, int K);

// Decode an R4NX trellis (2.5, 3.0, 3.5 or 4.0 bpw) into row-major FP16 tiles.
// Available on both CUDA and ROCm; ROCm decodes the codebook natively.
void reconstruct_r4nx(at::Tensor unpacked, at::Tensor packed, double rate);

#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
// Writes the exact unnormalized Q4NX integers as FP16 values.
void reconstruct_q4nx(at::Tensor unpacked, at::Tensor packed, int K, bool half_k);
#endif
