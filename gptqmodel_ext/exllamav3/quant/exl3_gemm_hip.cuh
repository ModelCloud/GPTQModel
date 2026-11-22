#pragma once

#include <ATen/Tensor.h>

// Native EXL3 tensor-core GEMM for RDNA4 (gfx120x) on ROCm/HIP.
//
// `exl3_gemm.cu` keeps the CUDA kernel dispatch and calls this on HIP builds.
// The function owns the whole ROCm decision: it runs the fused kernel when the
// shape and device are covered and otherwise reconstructs the weight for
// hipBLAS itself, so it always produces `C` and returns one of the codes below.
//
// `dispatch` values (kept disjoint from the CUDA table, the GEMV paths and the
// Q4NX reconstruct path):
//   EXL3_HIP_DISPATCH_NATIVE    the fused trellis -> WMMA kernel ran
//   EXL3_HIP_DISPATCH_FALLBACK  reconstruct + hipBLAS HGEMM ran
#define EXL3_HIP_DISPATCH_NATIVE 93
#define EXL3_HIP_DISPATCH_FALLBACK 94

int exl3_gemm_hip(
    const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
    const at::Tensor& suh, const at::Tensor& A_had, const at::Tensor& svh,
    int size_m, int size_k, int size_n, int K, bool half_k, int cb, bool c_fp32);

// Whether the current build/device pair can execute the native gfx12 WMMA
// kernel. Exposed for tests and for the Python-side capability probe.
bool exl3_gemm_hip_available(int device);
