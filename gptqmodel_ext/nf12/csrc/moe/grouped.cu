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
// The external grouped GEMM and TMA-fed families need sorted, contiguous x rows.
// Arguments are validated by the C API layer or moe_linear().
#include <algorithm>

#include "internal.h"
#include "moe/grouped.h"
#include "moe/schedule.cuh"

namespace nf12::detail {
namespace {

constexpr int kEndsThreads = 1024;
constexpr int kGatherThreads = 256;
constexpr int kGatherWarps = kGatherThreads / 32;
constexpr int kGatherUnroll = 4;  // 16-byte vectors in flight per lane
constexpr int64_t kGatherMaxBlocks = 65535;

__global__ void __launch_bounds__(kEndsThreads)
    group_ends_kernel(const int64_t* __restrict__ expert_offsets, int num_experts, int64_t max_assignments,
                      int32_t* __restrict__ ends) {
  extern __shared__ int smem[];
  moe_mtile_prefix(expert_offsets, num_experts, 1, max_assignments, smem);
  const int* start = smem + num_experts + 1;
  for (int e = static_cast<int>(threadIdx.x); e < num_experts; e += kEndsThreads) ends[e] = start[e + 1];
}

// Only [lo, hi) is written. hi is the maximum clamped expert offset.
// External grouped GEMM starts at lo = 0. With skip_leading, lo is the clamped
// off[0], matching moe_linear's valid positions.
// No global reads may precede griddepcontrol.wait under PDL.
__global__ void __launch_bounds__(kGatherThreads)
    gather_rows_kernel(const uint4* __restrict__ x, int64_t ldx_vecs, const int32_t* __restrict__ gather,
                       int64_t gather_div, const int64_t* __restrict__ expert_offsets, int num_experts,
                       int64_t max_assignments, bool skip_leading, int row_vecs, uint4* __restrict__ out,
                       int64_t ldo_vecs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  __shared__ int64_t warp_hi[kGatherWarps];
  const auto clamped = [&](int i) { return min(max(__ldg(expert_offsets + i), int64_t{0}), max_assignments); };
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int warp = static_cast<int>(threadIdx.x >> 5);
  int64_t hi = 0;
  for (int i = static_cast<int>(threadIdx.x); i <= num_experts; i += kGatherThreads) hi = max(hi, clamped(i));
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, d));
  if (lane == 0) warp_hi[warp] = hi;
  __syncthreads();
#pragma unroll
  for (int w = 0; w < kGatherWarps; ++w) hi = max(hi, warp_hi[w]);
  const int64_t stride = int64_t(gridDim.x) * kGatherWarps;
  const int64_t lo = skip_leading ? clamped(0) : 0;
  for (int64_t p = lo + int64_t(blockIdx.x) * kGatherWarps + warp; p < hi; p += stride) {
    const uint4* src = x + int64_t(__ldg(gather + p)) / gather_div * ldx_vecs;
    uint4* dst = out + p * ldo_vecs;
    for (int v0 = lane; v0 < row_vecs; v0 += 32 * kGatherUnroll) {
      uint4 r[kGatherUnroll];
#pragma unroll
      for (int u = 0; u < kGatherUnroll; ++u)
        if (v0 + 32 * u < row_vecs) r[u] = __ldg(src + v0 + 32 * u);
#pragma unroll
      for (int u = 0; u < kGatherUnroll; ++u)
        if (v0 + 32 * u < row_vecs) dst[v0 + 32 * u] = r[u];
    }
  }
}

cudaError_t launch_gather(const __nv_bfloat16* x, int64_t ldx, const int32_t* gather, int64_t gather_div,
                          const int64_t* expert_offsets, int64_t max_assignments, int num_experts, int64_t k,
                          __nv_bfloat16* out, int64_t ldo, bool skip_leading, bool pdl, cudaStream_t stream) {
  const int64_t blocks = std::min((max_assignments + kGatherWarps - 1) / kGatherWarps, kGatherMaxBlocks);
  if (blocks == 0) return cudaSuccess;
  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(static_cast<unsigned>(blocks), 1, 1);
  lc.blockDim = dim3(kGatherThreads, 1, 1);
  lc.stream = stream;
  cudaLaunchAttribute attr[1];
  if (pdl) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = attr;
    lc.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&lc, gather_rows_kernel, reinterpret_cast<const uint4*>(x), ldx / 8, gather, gather_div,
                            expert_offsets, num_experts, max_assignments, skip_leading, static_cast<int>(k / 8),
                            reinterpret_cast<uint4*>(out), ldo / 8);
}

}  // namespace

cudaError_t moe_grouped_operands(const __nv_bfloat16* x, int64_t ldx, const int32_t* gather, int64_t gather_div,
                                 const int64_t* expert_offsets, int64_t max_assignments, int64_t num_experts,
                                 int64_t k, __nv_bfloat16* x_out, int64_t ldo, int32_t* group_ends,
                                 cudaStream_t stream) {
  if (num_experts < 1 || num_experts > kMoeMaxExperts) return cudaErrorNotSupported;
  const int e = static_cast<int>(num_experts);
  const size_t smem = static_cast<size_t>(moe_schedule_smem_ints(e)) * sizeof(int);
  group_ends_kernel<<<1, kEndsThreads, smem, stream>>>(expert_offsets, e, max_assignments, group_ends);
  if (cudaError_t err = cudaGetLastError(); err != cudaSuccess || gather == nullptr) return err;
  return launch_gather(x, ldx, gather, gather_div, expert_offsets, max_assignments, e, k, x_out, ldo, false, false,
                       stream);
}

cudaError_t moe_gather_rows(const MoeLinearArgs& a, __nv_bfloat16* out) {
  return launch_gather(a.x, a.ldx, a.gather, a.gather_div, a.expert_offsets, a.max_assignments,
                       static_cast<int>(a.num_experts), a.k, out, a.k, true, a.dev->sm_arch >= 90, a.stream);
}

}  // namespace nf12::detail
