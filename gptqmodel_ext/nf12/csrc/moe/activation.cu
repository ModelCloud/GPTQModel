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
// SiLU and multiplication use FP32 with one BF16 rounding (SPEC.md 9.2).
// SM90+ launches use the moe_combine programmatic-dependency contract.
#include <cuda_bf16.h>

#include <cstdint>

#include "internal.h"

namespace nf12::detail {
namespace {

constexpr int kThreads = 256;

struct SiluMulParams {
  const __nv_bfloat16* h;
  int64_t ldh;
  int64_t rows;
  int64_t inter;
  int64_t vecs_per_row;  // inter / 8
  __nv_bfloat16* out;
  int64_t ldo;
};

__global__ void __launch_bounds__(kThreads) silu_mul_kernel(SiluMulParams p) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  const int64_t r = idx / p.vecs_per_row;
  const int64_t c = idx - r * p.vecs_per_row;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  if (r >= p.rows) return;
  const __nv_bfloat16* src = p.h + r * p.ldh + c * 8;
  const uint4 gv = __ldg(reinterpret_cast<const uint4*>(src));
  const uint4 uv = __ldg(reinterpret_cast<const uint4*>(src + p.inter));
  const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gv);
  const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&uv);
  uint4 o;
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 g = __bfloat1622float2(g2[i]);
    const float2 u = __bfloat1622float2(u2[i]);
    // expf and explicit rounding intrinsics prevent fast-math approximation
    // and contraction.
    const float sx = __fdiv_rn(g.x, __fadd_rn(1.0f, expf(-g.x)));
    const float sy = __fdiv_rn(g.y, __fadd_rn(1.0f, expf(-g.y)));
    o2[i] = __floats2bfloat162_rn(__fmul_rn(sx, u.x), __fmul_rn(sy, u.y));
  }
  *reinterpret_cast<uint4*>(p.out + r * p.ldo + c * 8) = o;
}

}  // namespace

cudaError_t moe_silu_mul(const __nv_bfloat16* h, int64_t ldh, int64_t rows, int64_t inter, __nv_bfloat16* out,
                         int64_t ldo, cudaStream_t stream) {
  if (rows < 0 || inter <= 0 || inter % 8 != 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  const int64_t vecs_per_row = inter / 8;
  if (rows > INT64_MAX / vecs_per_row) return cudaErrorInvalidValue;
  const int64_t vecs = rows * vecs_per_row;
  const int64_t blocks = vecs / kThreads + (vecs % kThreads != 0);
  if (blocks > INT32_MAX) return cudaErrorInvalidValue;
  int device = 0;
  if (cudaError_t err = cudaGetDevice(&device); err != cudaSuccess) return err;

  SiluMulParams p{h, ldh, rows, inter, vecs_per_row, out, ldo};
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
  return cudaLaunchKernelEx(&lc, silu_mul_kernel, p);
}

}  // namespace nf12::detail
