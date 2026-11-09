// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12 / NF12-AD kernel computation.
//
// Algorithm: "NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit
// Weight Compression", Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).
// Together AI: https://www.together.ai/
//
// Original implementation: https://github.com/dphnAI/nf12
// Copyright (c) 2026 AlpinDale and the nf12 contributors.
// Licensed under the Apache License, Version 2.0.
//
// Vendored into GPT-QModel (https://github.com/ModelCloud/GPTQModel) and
// modified for the NF12-AD (AlpinDale) method/format. See
// gptqmodel_ext/nf12/README.md for the list of local modifications.
// ---------------------------------------------------------------------------
// Sum topk rows in increasing j using FP32, then round once to BF16 (SPEC.md 9.2).
// SM90+ uses the programmatic-dependency contract in linear/dispatch.cu.
#include <cuda_bf16.h>

#include <cstdint>

#include "internal.h"

namespace nf12::detail {
namespace {

constexpr int kThreads = 256;
constexpr int kBatch = 4;  // loads in flight per thread

struct CombineParams {
  const __nv_bfloat16* y;
  int64_t ldy;
  int64_t num_tokens;
  int32_t topk;
  int64_t vecs_per_row;  // n / 8
  __nv_bfloat16* out;
  int64_t ldo;
};

__device__ __forceinline__ void accumulate(float (&acc)[8], const uint4& v) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    acc[2 * i] += f.x;
    acc[2 * i + 1] += f.y;
  }
}

__global__ void __launch_bounds__(kThreads) combine_kernel(CombineParams p) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  const int64_t t = idx / p.vecs_per_row;
  const int64_t c = idx - t * p.vecs_per_row;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  if (t >= p.num_tokens) return;
  const __nv_bfloat16* src = p.y + t * p.topk * p.ldy + c * 8;
  // Starting from -0.0 preserves a -0.0 first term.
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = -0.0f;
  // j0 + kBatch would overflow int32 for topk near INT32_MAX.
  for (int64_t j0 = 0; j0 < p.topk; j0 += kBatch) {
    uint4 v[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      if (j0 + b < p.topk) v[b] = __ldg(reinterpret_cast<const uint4*>(src + (j0 + b) * p.ldy));
    }
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      if (j0 + b < p.topk) accumulate(acc, v[b]);
    }
  }
  uint4 o;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
  *reinterpret_cast<uint4*>(p.out + t * p.ldo + c * 8) = o;
}

}  // namespace

cudaError_t moe_combine(const __nv_bfloat16* y, int64_t ldy, int64_t num_tokens, int64_t topk, int64_t n,
                        __nv_bfloat16* out, int64_t ldo, cudaStream_t stream) {
  if (num_tokens < 0 || topk < 1 || topk > INT32_MAX || n <= 0 || n % 8 != 0) return cudaErrorInvalidValue;
  if (num_tokens == 0) return cudaSuccess;
  const int64_t vecs = num_tokens * (n / 8);
  const int64_t blocks = (vecs + kThreads - 1) / kThreads;
  if (blocks > INT32_MAX) return cudaErrorInvalidValue;
  int device = 0;
  if (cudaError_t err = cudaGetDevice(&device); err != cudaSuccess) return err;

  CombineParams p{y, ldy, num_tokens, static_cast<int32_t>(topk), n / 8, out, ldo};
  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(static_cast<unsigned>(blocks), 1, 1);
  lc.blockDim = dim3(kThreads, 1, 1);
  lc.stream = stream;
  cudaLaunchAttribute attr[1];
  if (device_info(device).sm_arch >= 90) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = attr;
    lc.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&lc, combine_kernel, p);
}

}  // namespace nf12::detail
