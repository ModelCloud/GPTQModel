#pragma once

// Tensor-core fragment types shared by the GEMM kernels and by the
// quantize/reconstruct lifecycle kernels.
//
// These are deliberately kept free of any CUDA-only intrinsic (unlike
// `ptx.cuh`, which holds NVIDIA inline PTX). `reconstruct.cu` needs only the
// fragment types, so it includes this header instead of `ptx.cuh`, which lets
// the EXL3 quantization lifecycle build on ROCm.

template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;
