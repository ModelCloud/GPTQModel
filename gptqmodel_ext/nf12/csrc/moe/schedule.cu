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
#include "moe/schedule.cuh"

#include "internal.h"

namespace nf12::detail {
namespace {

constexpr int kThreads = 1024;

__global__ void __launch_bounds__(kThreads)
    moe_schedule_kernel(const int64_t* __restrict__ expert_offsets, int num_experts, int64_t max_assignments,
                        int bm, uint8_t* __restrict__ workspace) {
  extern __shared__ int smem[];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  const int total = moe_mtile_prefix(expert_offsets, num_experts, bm, max_assignments, smem);
  auto* tiles = reinterpret_cast<int4*>(workspace + kMoeScheduleHeaderBytes);
  for (int i = static_cast<int>(threadIdx.x); i < total; i += kThreads) {
    int e, row0, rows;
    moe_mtile_find(smem, num_experts, bm, i, e, row0, rows);
    tiles[i] = make_int4(e, row0, rows, i - smem[e]);
  }
  auto* row_end = reinterpret_cast<int32_t*>(
      workspace + moe_schedule_row_end_offset(max_assignments, num_experts, bm));
  const int* start = smem + num_experts + 1;
  for (int e = static_cast<int>(threadIdx.x); e < num_experts; e += kThreads) row_end[e] = start[e + 1];
  if (threadIdx.x == 0) *reinterpret_cast<int32_t*>(workspace) = total;
}

}  // namespace

size_t moe_schedule_bytes(int64_t max_assignments, int64_t num_experts, int bm) {
  return moe_schedule_row_end_offset(max_assignments, num_experts, bm) +
         static_cast<size_t>(num_experts) * sizeof(int32_t);
}

cudaError_t moe_schedule(const int64_t* expert_offsets, int64_t num_experts, int64_t max_assignments, int bm,
                         void* workspace, cudaStream_t stream) {
  if (num_experts < 1 || num_experts > kMoeMaxExperts || max_assignments < 0 || max_assignments > INT32_MAX ||
      bm < 1 || workspace == nullptr || expert_offsets == nullptr)
    return cudaErrorInvalidValue;
  int device = 0;
  cudaError_t e = cudaGetDevice(&device);
  if (e != cudaSuccess) return e;
  const DeviceInfo& dev = device_info(device);
  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(1, 1, 1);
  lc.blockDim = dim3(kThreads, 1, 1);
  lc.dynamicSmemBytes = static_cast<size_t>(moe_schedule_smem_ints(static_cast<int>(num_experts))) * sizeof(int);
  lc.stream = stream;
  cudaLaunchAttribute attr[1];
  if (dev.sm_arch >= 90) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = attr;
    lc.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&lc, moe_schedule_kernel, expert_offsets, static_cast<int>(num_experts),
                            max_assignments, bm, static_cast<uint8_t*>(workspace));
}

}  // namespace nf12::detail
