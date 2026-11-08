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
// Kernels assume 16-byte bases, strides divisible by 8 and SPEC.md shapes.
// The C API validates these before launching.
#include <cstdint>
#include <limits>

#include "internal.h"
#include "nf12/nf12.h"

namespace {

using nf12::detail::DeviceInfo;

thread_local cudaError_t g_last_cuda_error = cudaSuccess;

constexpr int kMinSmArch = 80;
constexpr uint32_t kMaxExpBase = 120;

nf12_status_t from_cuda(cudaError_t err) {
  if (err == cudaSuccess) return NF12_OK;
  g_last_cuda_error = err;
  if (err == cudaErrorNotSupported || err == cudaErrorNoKernelImageForDevice) return NF12_ERROR_UNSUPPORTED;
  return NF12_ERROR_CUDA;
}

// Codec cudaErrorInvalidValue means escape capacity/index or grid overflow
// after argument validation. Packed and escape outputs remain untouched.
nf12_status_t from_codec(cudaError_t err) {
  return err == cudaErrorInvalidValue ? NF12_ERROR_INVALID_ARGUMENT : from_cuda(err);
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// Bound intermediate byte counts to int64.
bool valid_shape(int64_t n, int64_t k) {
  return n > 0 && k > 0 && n % 16 == 0 && k % 64 == 0 && n <= std::numeric_limits<int64_t>::max() / 4 / k;
}

// A row-major BF16 matrix the kernels read or write with 16-byte vectors.
bool valid_bf16_matrix(const void* p, int64_t cols, int64_t ld) {
  return p != nullptr && aligned16(p) && ld >= cols && ld % 8 == 0;
}

bool valid_weight(const nf12_weight_t* w) {
  return w != nullptr && valid_shape(w->n, w->k) && w->packed != nullptr && aligned16(w->packed) &&
         w->exp_base <= kMaxExpBase && aligned16(w->escapes) && (w->num_escapes == 0 || w->escapes != nullptr);
}

bool valid_workspace(const void* workspace, size_t bytes) {
  return bytes == 0 || (workspace != nullptr && aligned16(workspace));
}

nf12_status_t current_device(const DeviceInfo** dev) {
  int device = 0;
  if (cudaError_t err = cudaGetDevice(&device); err != cudaSuccess) return from_cuda(err);
  *dev = &nf12::detail::device_info(device);
  return (*dev)->sm_arch >= kMinSmArch ? NF12_OK : NF12_ERROR_UNSUPPORTED;
}

nf12_status_t run_linear(const void* x, int64_t m, int64_t ldx, nf12::detail::WeightSource source, const void* weight,
                         const void* escapes, uint32_t num_escapes, uint32_t exp_base, int64_t n, int64_t k,
                         const void* bias, void* y, int64_t ldy, void* workspace, size_t workspace_bytes,
                         cudaStream_t stream) {
  if (m < 0 || (bias != nullptr && !aligned16(bias)) || !valid_workspace(workspace, workspace_bytes)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (m == 0) return NF12_OK;
  if (!valid_bf16_matrix(x, k, ldx) || !valid_bf16_matrix(y, n, ldy)) return NF12_ERROR_INVALID_ARGUMENT;
  const DeviceInfo* dev = nullptr;
  if (nf12_status_t s = current_device(&dev); s != NF12_OK) return s;
  if (workspace_bytes < nf12::detail::linear_workspace_bytes(m, n, k, *dev)) return NF12_ERROR_INSUFFICIENT_WORKSPACE;

  nf12::detail::LinearArgs args{};
  args.x = static_cast<const __nv_bfloat16*>(x);
  args.m = m;
  args.ldx = ldx;
  args.source = source;
  args.weight = static_cast<const uint8_t*>(weight);
  args.escapes = static_cast<const uint4*>(escapes);
  args.num_escapes = num_escapes;
  args.exp_base = exp_base;
  args.n = n;
  args.k = k;
  args.bias = static_cast<const __nv_bfloat16*>(bias);
  args.y = static_cast<__nv_bfloat16*>(y);
  args.ldy = ldy;
  args.workspace = workspace;
  args.workspace_bytes = workspace_bytes;
  args.stream = stream;
  args.dev = dev;
  return from_cuda(nf12::detail::linear(args));
}

bool aligned(const void* p, uintptr_t bytes) { return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0; }

// The larger bf16t layout must fit in int64 across all experts.
bool valid_moe_shape(int64_t num_experts, int64_t n, int64_t k) {
  return num_experts >= 1 && valid_shape(n, k) && num_experts <= std::numeric_limits<int64_t>::max() / 2 / n / k;
}

bool valid_moe_weight(const nf12_moe_weight_t* w) {
  return w != nullptr && valid_moe_shape(w->num_experts, w->n, w->k) && w->packed != nullptr && aligned16(w->packed) &&
         aligned16(w->escapes) && w->escape_offsets != nullptr && aligned(w->escape_offsets, 8) &&
         w->exp_base != nullptr;
}

// Gather/scatter indices and sorted positions limit assignments to int32.
nf12_status_t run_moe_linear(const void* x, int64_t ldx, const int32_t* gather, int64_t gather_div,
                             const int32_t* scatter, const int64_t* expert_offsets, int64_t max_assignments,
                             nf12::detail::MoeLinearArgs args, const float* row_scale, void* y, int64_t ldy,
                             void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  if (max_assignments < 0 || max_assignments > std::numeric_limits<int32_t>::max() || gather_div < 1 ||
      !aligned(gather, 4) || !aligned(scatter, 4) || !aligned(row_scale, 4) || !aligned(expert_offsets, 8) ||
      !valid_workspace(workspace, workspace_bytes)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (max_assignments == 0) return NF12_OK;
  if (expert_offsets == nullptr || !valid_bf16_matrix(x, args.k, ldx) || !valid_bf16_matrix(y, args.n, ldy)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  const DeviceInfo* dev = nullptr;
  if (nf12_status_t s = current_device(&dev); s != NF12_OK) return s;
  if (workspace_bytes < nf12::detail::moe_linear_workspace_bytes(max_assignments, args.num_experts, args.n, args.k,
                                                                 *dev)) {
    return NF12_ERROR_INSUFFICIENT_WORKSPACE;
  }
  args.x = static_cast<const __nv_bfloat16*>(x);
  args.ldx = ldx;
  args.gather = gather;
  args.gather_div = gather_div;
  args.scatter = scatter;
  args.expert_offsets = expert_offsets;
  args.max_assignments = max_assignments;
  args.row_scale = row_scale;
  args.y = static_cast<__nv_bfloat16*>(y);
  args.ldy = ldy;
  args.workspace = workspace;
  args.workspace_bytes = workspace_bytes;
  args.stream = stream;
  args.dev = dev;
  return from_cuda(nf12::detail::moe_linear(args));
}

template <typename Id>
nf12_status_t run_embedding(const nf12_weight_t* w, const Id* ids, int64_t num_ids, void* out, int64_t ldo,
                            cudaStream_t stream) {
  if (!valid_weight(w) || num_ids < 0 || !aligned(ids, sizeof(Id))) return NF12_ERROR_INVALID_ARGUMENT;
  if (num_ids == 0) return NF12_OK;
  // Bound BF16 byte offsets to int64.
  if (ids == nullptr || !valid_bf16_matrix(out, w->k, ldo) || num_ids > std::numeric_limits<int64_t>::max() / 2 / ldo) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  const DeviceInfo* dev = nullptr;
  if (nf12_status_t s = current_device(&dev); s != NF12_OK) return s;
  const int32_t* ids32 = nullptr;
  const int64_t* ids64 = nullptr;
  if constexpr (sizeof(Id) == sizeof(int32_t)) {
    ids32 = ids;
  } else {
    ids64 = ids;
  }
  return from_codec(nf12::detail::embedding(static_cast<const uint8_t*>(w->packed),
                                            static_cast<const uint4*>(w->escapes), w->num_escapes, w->exp_base, w->n,
                                            w->k, ids32, ids64, num_ids, static_cast<__nv_bfloat16*>(out), ldo,
                                            stream));
}

const DeviceInfo* query_device(int device) {
  int count = 0;
  if (device < 0) return nullptr;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();
    return nullptr;
  }
  if (device >= count) return nullptr;
  const DeviceInfo& dev = nf12::detail::device_info(device);
  return dev.sm_arch >= kMinSmArch ? &dev : nullptr;
}

}

extern "C" {

const char* nf12_status_string(nf12_status_t status) {
  switch (status) {
    case NF12_OK:
      return "NF12_OK";
    case NF12_ERROR_INVALID_ARGUMENT:
      return "NF12_ERROR_INVALID_ARGUMENT: shape, alignment or pointer violates SPEC.md";
    case NF12_ERROR_UNSUPPORTED:
      return "NF12_ERROR_UNSUPPORTED: device architecture or configuration not supported";
    case NF12_ERROR_INSUFFICIENT_WORKSPACE:
      return "NF12_ERROR_INSUFFICIENT_WORKSPACE";
    case NF12_ERROR_CUDA:
      return "NF12_ERROR_CUDA: see nf12_last_cuda_error()";
  }
  return "NF12: unknown status";
}

cudaError_t nf12_last_cuda_error(void) { return g_last_cuda_error; }

size_t nf12_packed_bytes(int64_t n, int64_t k) {
  return valid_shape(n, k) ? static_cast<size_t>(n) * static_cast<size_t>(k) * 3 / 2 : 0;
}

size_t nf12_bf16t_bytes(int64_t n, int64_t k) {
  return valid_shape(n, k) ? static_cast<size_t>(n) * static_cast<size_t>(k) * 2 : 0;
}

size_t nf12_encode_workspace_bytes(int64_t n, int64_t k) {
  return valid_shape(n, k) ? nf12::detail::encode_workspace_bytes(n, k) : 0;
}

nf12_status_t nf12_encode_analyze(const void* w_bf16, int64_t n, int64_t k, int64_t ldw, void* workspace,
                                  size_t workspace_bytes, uint8_t* exp_base_out, uint32_t* num_escapes_out,
                                  cudaStream_t stream) {
  if (!valid_shape(n, k) || !valid_bf16_matrix(w_bf16, k, ldw) || exp_base_out == nullptr ||
      num_escapes_out == nullptr || !valid_workspace(workspace, workspace_bytes)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (workspace_bytes < nf12::detail::encode_workspace_bytes(n, k)) return NF12_ERROR_INSUFFICIENT_WORKSPACE;
  const DeviceInfo* dev = nullptr;
  if (nf12_status_t s = current_device(&dev); s != NF12_OK) return s;
  return from_codec(nf12::detail::encode_analyze(static_cast<const __nv_bfloat16*>(w_bf16), n, k, ldw, workspace,
                                                 workspace_bytes, exp_base_out, num_escapes_out, stream));
}

nf12_status_t nf12_encode(const void* w_bf16, int64_t n, int64_t k, int64_t ldw, uint8_t exp_base, void* packed_out,
                          void* escapes_out, uint32_t escapes_capacity, void* workspace, size_t workspace_bytes,
                          cudaStream_t stream) {
  if (!valid_shape(n, k) || !valid_bf16_matrix(w_bf16, k, ldw) || exp_base > kMaxExpBase || packed_out == nullptr ||
      !aligned16(packed_out) || !aligned16(escapes_out) || (escapes_capacity > 0 && escapes_out == nullptr) ||
      !valid_workspace(workspace, workspace_bytes)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (workspace_bytes < nf12::detail::encode_workspace_bytes(n, k)) return NF12_ERROR_INSUFFICIENT_WORKSPACE;
  const DeviceInfo* dev = nullptr;
  if (nf12_status_t s = current_device(&dev); s != NF12_OK) return s;
  return from_codec(nf12::detail::encode(static_cast<const __nv_bfloat16*>(w_bf16), n, k, ldw, exp_base,
                                         static_cast<uint8_t*>(packed_out), static_cast<uint4*>(escapes_out),
                                         escapes_capacity, workspace, workspace_bytes, stream));
}

nf12_status_t nf12_decode(const nf12_weight_t* w, void* out_bf16, int64_t ldo, cudaStream_t stream) {
  if (!valid_weight(w) || !valid_bf16_matrix(out_bf16, w->k, ldo)) return NF12_ERROR_INVALID_ARGUMENT;
  const DeviceInfo* dev = nullptr;
  if (nf12_status_t s = current_device(&dev); s != NF12_OK) return s;
  return from_codec(nf12::detail::decode(static_cast<const uint8_t*>(w->packed), static_cast<const uint4*>(w->escapes),
                                         w->exp_base, w->n, w->k, static_cast<__nv_bfloat16*>(out_bf16), ldo, stream));
}

size_t nf12_linear_workspace_bytes(int64_t m, int64_t n, int64_t k, int device) {
  if (m <= 0 || !valid_shape(n, k)) return 0;
  const DeviceInfo* dev = query_device(device);
  return dev != nullptr ? nf12::detail::linear_workspace_bytes(m, n, k, *dev) : 0;
}

int64_t nf12_linear_decode_gemm_rows(int64_t m, int64_t n, int64_t k, int device) {
  if (m <= 0 || m > std::numeric_limits<int32_t>::max() || !valid_shape(n, k)) return 0;
  const DeviceInfo* dev = query_device(device);
  return dev != nullptr ? nf12::detail::linear_decode_gemm_rows(m, n, k, *dev) : 0;
}

nf12_status_t nf12_linear(const void* x_bf16, int64_t m, int64_t ldx, const nf12_weight_t* w, const void* bias_bf16,
                          void* y_bf16, int64_t ldy, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  if (!valid_weight(w)) return NF12_ERROR_INVALID_ARGUMENT;
  return run_linear(x_bf16, m, ldx, nf12::detail::WeightSource::kNF12, w->packed, w->escapes, w->num_escapes,
                    w->exp_base, w->n, w->k, bias_bf16, y_bf16, ldy, workspace, workspace_bytes, stream);
}

nf12_status_t nf12_pack_bf16t(const void* w_bf16, int64_t n, int64_t k, int64_t ldw, void* out, cudaStream_t stream) {
  if (!valid_shape(n, k) || !valid_bf16_matrix(w_bf16, k, ldw) || out == nullptr || !aligned16(out)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  const DeviceInfo* dev = nullptr;
  if (nf12_status_t s = current_device(&dev); s != NF12_OK) return s;
  return from_codec(nf12::detail::pack_bf16t(static_cast<const __nv_bfloat16*>(w_bf16), n, k, ldw,
                                             static_cast<uint8_t*>(out), stream));
}

nf12_status_t nf12_linear_bf16t(const void* x_bf16, int64_t m, int64_t ldx, const void* w_bf16t, int64_t n, int64_t k,
                                const void* bias_bf16, void* y_bf16, int64_t ldy, void* workspace,
                                size_t workspace_bytes, cudaStream_t stream) {
  if (!valid_shape(n, k) || w_bf16t == nullptr || !aligned16(w_bf16t)) return NF12_ERROR_INVALID_ARGUMENT;
  return run_linear(x_bf16, m, ldx, nf12::detail::WeightSource::kBF16T, w_bf16t, nullptr, 0, 0, n, k, bias_bf16,
                    y_bf16, ldy, workspace, workspace_bytes, stream);
}

size_t nf12_moe_align_workspace_bytes(int64_t num_tokens, int64_t topk, int64_t num_experts) {
  if (num_tokens < 0 || topk < 1 || num_experts < 1 || num_experts > std::numeric_limits<int32_t>::max() ||
      num_tokens > std::numeric_limits<int32_t>::max() / topk) {
    return 0;
  }
  return nf12::detail::moe_align_workspace_bytes(num_tokens, topk, num_experts);
}

nf12_status_t nf12_moe_align(const int32_t* topk_ids, int64_t num_tokens, int64_t topk, int64_t num_experts,
                             int32_t* sorted_out, int64_t* expert_offsets_out, void* workspace,
                             size_t workspace_bytes, cudaStream_t stream) {
  // Assignment positions and expert IDs must fit in int32.
  if (num_tokens < 0 || topk < 1 || num_experts < 1 || num_experts > std::numeric_limits<int32_t>::max() ||
      num_tokens > std::numeric_limits<int32_t>::max() / topk || expert_offsets_out == nullptr ||
      !aligned(expert_offsets_out, 8) || !aligned(topk_ids, 4) || !aligned(sorted_out, 4) ||
      !valid_workspace(workspace, workspace_bytes)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  // Empty input still writes zero offsets, but needs no ID/sorted buffers.
  if (num_tokens > 0 && (topk_ids == nullptr || sorted_out == nullptr)) return NF12_ERROR_INVALID_ARGUMENT;
  if (workspace_bytes < nf12::detail::moe_align_workspace_bytes(num_tokens, topk, num_experts)) {
    return NF12_ERROR_INSUFFICIENT_WORKSPACE;
  }
  const DeviceInfo* dev = nullptr;
  if (nf12_status_t s = current_device(&dev); s != NF12_OK) return s;
  return from_cuda(nf12::detail::moe_align(topk_ids, num_tokens, topk, num_experts, sorted_out, expert_offsets_out,
                                           workspace, workspace_bytes, stream));
}

size_t nf12_moe_linear_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                       int device) {
  if (max_assignments <= 0 || max_assignments > std::numeric_limits<int32_t>::max() ||
      !valid_moe_shape(num_experts, n, k)) {
    return 0;
  }
  const DeviceInfo* dev = query_device(device);
  return dev != nullptr ? nf12::detail::moe_linear_workspace_bytes(max_assignments, num_experts, n, k, *dev) : 0;
}

nf12_status_t nf12_moe_linear(const void* x_bf16, int64_t ldx, const int32_t* gather, int64_t gather_div,
                              const int32_t* scatter, const int64_t* expert_offsets, int64_t max_assignments,
                              const nf12_moe_weight_t* w, const float* row_scale, void* y_bf16, int64_t ldy,
                              void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  if (!valid_moe_weight(w)) return NF12_ERROR_INVALID_ARGUMENT;
  nf12::detail::MoeLinearArgs args{};
  args.source = nf12::detail::WeightSource::kNF12;
  args.weight = static_cast<const uint8_t*>(w->packed);
  args.escapes = static_cast<const uint4*>(w->escapes);
  args.escape_offsets = w->escape_offsets;
  args.exp_base = w->exp_base;
  args.num_experts = w->num_experts;
  args.n = w->n;
  args.k = w->k;
  return run_moe_linear(x_bf16, ldx, gather, gather_div, scatter, expert_offsets, max_assignments, args, row_scale,
                        y_bf16, ldy, workspace, workspace_bytes, stream);
}

nf12_status_t nf12_moe_linear_bf16t(const void* x_bf16, int64_t ldx, const int32_t* gather, int64_t gather_div,
                                    const int32_t* scatter, const int64_t* expert_offsets, int64_t max_assignments,
                                    const void* w_bf16t, int64_t num_experts, int64_t n, int64_t k,
                                    const float* row_scale, void* y_bf16, int64_t ldy, void* workspace,
                                    size_t workspace_bytes, cudaStream_t stream) {
  if (!valid_moe_shape(num_experts, n, k) || w_bf16t == nullptr || !aligned16(w_bf16t)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  nf12::detail::MoeLinearArgs args{};
  args.source = nf12::detail::WeightSource::kBF16T;
  args.weight = static_cast<const uint8_t*>(w_bf16t);
  args.num_experts = num_experts;
  args.n = n;
  args.k = k;
  return run_moe_linear(x_bf16, ldx, gather, gather_div, scatter, expert_offsets, max_assignments, args, row_scale,
                        y_bf16, ldy, workspace, workspace_bytes, stream);
}

nf12_status_t nf12_moe_combine(const void* y_bf16, int64_t ldy, int64_t num_tokens, int64_t topk, int64_t n,
                               void* out_bf16, int64_t ldo, cudaStream_t stream) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (num_tokens < 0 || topk < 1 || topk > std::numeric_limits<int32_t>::max() || n <= 0 || n % 8 != 0) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (num_tokens == 0) return NF12_OK;
  // Bound BF16 byte offsets in both buffers to int64.
  if (!valid_bf16_matrix(y_bf16, n, ldy) || !valid_bf16_matrix(out_bf16, n, ldo) ||
      num_tokens > kMax / 2 / topk / ldy || num_tokens > kMax / 2 / ldo) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  const DeviceInfo* dev = nullptr;
  if (nf12_status_t s = current_device(&dev); s != NF12_OK) return s;
  return from_cuda(nf12::detail::moe_combine(static_cast<const __nv_bfloat16*>(y_bf16), ldy, num_tokens, topk, n,
                                             static_cast<__nv_bfloat16*>(out_bf16), ldo, stream));
}

nf12_status_t nf12_moe_silu_mul(const void* h_bf16, int64_t ldh, int64_t rows, int64_t inter, void* out_bf16,
                                int64_t ldo, cudaStream_t stream) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (rows < 0 || inter <= 0 || inter % 8 != 0 || inter > kMax / 2) return NF12_ERROR_INVALID_ARGUMENT;
  if (rows == 0) return NF12_OK;
  // Bound BF16 byte offsets in both buffers to int64.
  if (!valid_bf16_matrix(h_bf16, 2 * inter, ldh) || !valid_bf16_matrix(out_bf16, inter, ldo) ||
      rows > kMax / 2 / ldh || rows > kMax / 2 / ldo) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  const DeviceInfo* dev = nullptr;
  if (nf12_status_t s = current_device(&dev); s != NF12_OK) return s;
  return from_cuda(nf12::detail::moe_silu_mul(static_cast<const __nv_bfloat16*>(h_bf16), ldh, rows, inter,
                                              static_cast<__nv_bfloat16*>(out_bf16), ldo, stream));
}

nf12_status_t nf12_moe_decode(const nf12_moe_weight_t* w, const int64_t* expert_offsets, int64_t max_assignments,
                              int64_t row0, int64_t rows, void* out_bf16, int64_t ldo, cudaStream_t stream) {
  if (!valid_moe_weight(w) || row0 < 0 || rows < 0 || row0 % 16 != 0 || rows % 16 != 0 || rows > w->n - row0 ||
      !aligned(expert_offsets, 8) || max_assignments < 0 || max_assignments > std::numeric_limits<int32_t>::max()) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (rows == 0) return NF12_OK;
  // Bound stacked BF16 output byte offsets to int64.
  if (!valid_bf16_matrix(out_bf16, w->k, ldo) || w->num_experts > std::numeric_limits<int64_t>::max() / 2 / rows / ldo) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  const DeviceInfo* dev = nullptr;
  if (nf12_status_t s = current_device(&dev); s != NF12_OK) return s;
  return from_codec(nf12::detail::moe_decode(static_cast<const uint8_t*>(w->packed),
                                             static_cast<const uint4*>(w->escapes), w->escape_offsets, w->exp_base,
                                             w->num_experts, w->n, w->k, expert_offsets, max_assignments, row0, rows,
                                             static_cast<__nv_bfloat16*>(out_bf16), ldo, stream));
}

int64_t nf12_moe_linear_decode_gemm_rows(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                         int device) {
  if (max_assignments <= 0 || max_assignments > std::numeric_limits<int32_t>::max() ||
      !valid_moe_shape(num_experts, n, k)) {
    return 0;
  }
  const DeviceInfo* dev = query_device(device);
  return dev != nullptr ? nf12::detail::moe_linear_decode_gemm_rows(max_assignments, num_experts, n, k, *dev) : 0;
}

nf12_status_t nf12_moe_grouped_operands(const void* x_bf16, int64_t ldx, const int32_t* gather, int64_t gather_div,
                                        const int64_t* expert_offsets, int64_t max_assignments, int64_t num_experts,
                                        int64_t k, void* x_out_bf16, int64_t ldo, int32_t* group_ends,
                                        cudaStream_t stream) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (max_assignments < 0 || max_assignments > std::numeric_limits<int32_t>::max() || gather_div < 1 ||
      num_experts < 1 || k <= 0 || k % 8 != 0 || expert_offsets == nullptr || !aligned(expert_offsets, 8) ||
      group_ends == nullptr || !aligned(group_ends, 4) || !aligned(gather, 4)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  // Gather can address any int32 row, regardless of max_assignments.
  if (gather != nullptr && (!valid_bf16_matrix(x_bf16, k, ldx) || !valid_bf16_matrix(x_out_bf16, k, ldo) ||
                            ldx > kMax / 2 / std::numeric_limits<int32_t>::max() ||
                            ldo > kMax / 2 / std::numeric_limits<int32_t>::max())) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  const DeviceInfo* dev = nullptr;
  if (nf12_status_t s = current_device(&dev); s != NF12_OK) return s;
  return from_cuda(nf12::detail::moe_grouped_operands(
      static_cast<const __nv_bfloat16*>(x_bf16), ldx, gather, gather_div, expert_offsets, max_assignments,
      num_experts, k, static_cast<__nv_bfloat16*>(x_out_bf16), ldo, group_ends, stream));
}

nf12_status_t nf12_embedding(const nf12_weight_t* w, const int32_t* ids, int64_t num_ids, void* out_bf16, int64_t ldo,
                             cudaStream_t stream) {
  return run_embedding(w, ids, num_ids, out_bf16, ldo, stream);
}

nf12_status_t nf12_embedding_i64(const nf12_weight_t* w, const int64_t* ids, int64_t num_ids, void* out_bf16,
                                 int64_t ldo, cudaStream_t stream) {
  return run_embedding(w, ids, num_ids, out_bf16, ldo, stream);
}

}
