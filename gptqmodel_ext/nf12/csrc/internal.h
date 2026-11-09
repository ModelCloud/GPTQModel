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
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace nf12::detail {

struct DeviceInfo {
  int device;
  int sm_arch;   // major * 10 + minor, e.g. 110, 120
  int num_sms;
  size_t smem_optin_bytes;
  size_t l2_bytes;
};
const DeviceInfo& device_info(int device);  // cached per device

enum class WeightSource { kNF12, kBF16T };

struct LinearArgs {
  const __nv_bfloat16* x;
  int64_t m, ldx;
  WeightSource source;
  const uint8_t* weight;    // NF12 packed or bf16t bytes
  const uint4* escapes;     // NF12 only
  uint32_t num_escapes;     // bounds NF12 escape-table prefetches
  uint32_t exp_base;        // NF12 only
  int64_t n, k;
  const __nv_bfloat16* bias;  // may be null
  __nv_bfloat16* y;
  int64_t ldy;
  void* workspace;
  size_t workspace_bytes;
  cudaStream_t stream;
  const DeviceInfo* dev;
};

size_t linear_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);
cudaError_t linear(const LinearArgs& args);
// Recommended N-slice height for decode + external BF16 GEMM (SPEC.md 6).
// Zero selects the fused route.
int64_t linear_decode_gemm_rows(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);

size_t encode_workspace_bytes(int64_t n, int64_t k);
// Both encoding operations synchronize stream. Analysis outputs are host pointers.
cudaError_t encode_analyze(const __nv_bfloat16* w, int64_t n, int64_t k, int64_t ldw, void* workspace,
                           size_t workspace_bytes, uint8_t* exp_base_out, uint32_t* num_escapes_out,
                           cudaStream_t stream);
cudaError_t encode(const __nv_bfloat16* w, int64_t n, int64_t k, int64_t ldw, uint8_t exp_base, uint8_t* packed,
                   uint4* escapes, uint32_t escapes_capacity, void* workspace, size_t workspace_bytes,
                   cudaStream_t stream);
cudaError_t decode(const uint8_t* packed, const uint4* escapes, uint32_t exp_base, int64_t n, int64_t k,
                   __nv_bfloat16* out, int64_t ldo, cudaStream_t stream);
cudaError_t pack_bf16t(const __nv_bfloat16* w, int64_t n, int64_t k, int64_t ldw, uint8_t* out,
                       cudaStream_t stream);
// out holds n * k * 2 bytes in the bf16t layout (SPEC.md 4.4).
cudaError_t decode_bf16t(const uint8_t* packed, const uint4* escapes, uint32_t exp_base, int64_t n, int64_t k,
                         uint8_t* out, cudaStream_t stream);
// Decode expert rows [row0, row0 + rows) to out + (e * rows + r) * ldo.
// Non-null expert_offsets skip inactive experts and leave their output untouched.
// Activity follows moe_linear's clamped, monotone offsets (SPEC.md 9.1).
cudaError_t moe_decode(const uint8_t* packed, const uint4* escapes, const int64_t* escape_offsets,
                       const uint8_t* exp_base, int64_t num_experts, int64_t n, int64_t k,
                       const int64_t* expert_offsets, int64_t max_assignments, int64_t row0, int64_t rows,
                       __nv_bfloat16* out, int64_t ldo, cudaStream_t stream);

struct MoeLinearArgs {
  const __nv_bfloat16* x;
  int64_t ldx;
  const int32_t* gather;   // null selects in(p) = p
  int64_t gather_div;      // >= 1
  const int32_t* scatter;  // null selects out(p) = p
  const int64_t* expert_offsets;  // device [E + 1]
  int64_t max_assignments;
  WeightSource source;
  const uint8_t* weight;          // [E, n, k*3/2] NF12 or [E, n, 2k] bf16t
  const uint4* escapes;           // NF12 only, may be null
  const int64_t* escape_offsets;  // NF12 only, device [E + 1]
  const uint8_t* exp_base;        // NF12 only, device [E]
  int64_t num_experts, n, k;
  const float* row_scale;  // optional, indexed by out(p)
  __nv_bfloat16* y;
  int64_t ldy;
  void* workspace;
  size_t workspace_bytes;
  cudaStream_t stream;
  const DeviceInfo* dev;
};

size_t moe_align_workspace_bytes(int64_t num_tokens, int64_t topk, int64_t num_experts);
cudaError_t moe_align(const int32_t* topk_ids, int64_t num_tokens, int64_t topk, int64_t num_experts,
                      int32_t* sorted, int64_t* expert_offsets, void* workspace, size_t workspace_bytes,
                      cudaStream_t stream);
size_t moe_linear_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                  const DeviceInfo& dev);
cudaError_t moe_linear(const MoeLinearArgs& args);
cudaError_t moe_combine(const __nv_bfloat16* y, int64_t ldy, int64_t num_tokens, int64_t topk, int64_t n,
                        __nv_bfloat16* out, int64_t ldo, cudaStream_t stream);
cudaError_t moe_silu_mul(const __nv_bfloat16* h, int64_t ldh, int64_t rows, int64_t inter, __nv_bfloat16* out,
                         int64_t ldo, cudaStream_t stream);
// Recommended N-slice height for decode + external grouped GEMM (SPEC.md 9.4).
// Zero selects the fused route.
int64_t moe_linear_decode_gemm_rows(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                    const DeviceInfo& dev);
// group_ends contains clamped, monotone ends in [0, max_assignments].
// With gather, copy x[gather[p] / gather_div] to x_out[p] for p < group_ends[E-1].
// Output rows have stride ldo.
cudaError_t moe_grouped_operands(const __nv_bfloat16* x, int64_t ldx, const int32_t* gather, int64_t gather_div,
                                 const int64_t* expert_offsets, int64_t max_assignments, int64_t num_experts,
                                 int64_t k, __nv_bfloat16* x_out, int64_t ldo, int32_t* group_ends,
                                 cudaStream_t stream);

// Exactly one of ids32/ids64 is non-null (SPEC.md 10).
cudaError_t embedding(const uint8_t* packed, const uint4* escapes, uint32_t num_escapes, uint32_t exp_base, int64_t n,
                      int64_t k, const int32_t* ids32, const int64_t* ids64, int64_t num_ids, __nv_bfloat16* out,
                      int64_t ldo, cudaStream_t stream);

}
