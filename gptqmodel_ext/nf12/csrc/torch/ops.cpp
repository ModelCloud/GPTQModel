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
// Stable-ABI C shims support torch >= 2.10, including the cuBLAS routes.
// Ops use torch's current stream and caching allocator for graph-safe workspace.
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "nf12/nf12.h"
#include "torch/fused_decode.h"

namespace {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

constexpr int64_t kMaxExpBase = 120;

void check_status(nf12_status_t status, const char* op) {
  if (status == NF12_OK) return;
  if (status == NF12_ERROR_CUDA || status == NF12_ERROR_UNSUPPORTED) {
    STD_TORCH_CHECK(false, "nf12.", op, " failed: ", nf12_status_string(status), " (",
                    cudaGetErrorString(nf12_last_cuda_error()), ")");
  }
  STD_TORCH_CHECK(false, "nf12.", op, " failed: ", nf12_status_string(status));
}

cudaStream_t current_stream(int32_t device) {
  void* stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device, &stream));
  return static_cast<cudaStream_t>(stream);
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

Tensor byte_buffer(const Tensor& like, size_t bytes) {
  return torch::stable::new_empty(like, {static_cast<int64_t>(bytes)}, ScalarType::Byte);
}

// Ensure 16-byte vector access, copying only if strides or alignment require it.
// A single row does not constrain the row stride.
Tensor vector_rows(const Tensor& t) {
  auto ok = [](const Tensor& v) {
    return v.stride(1) == 1 && (v.size(0) <= 1 || (v.stride(0) % 8 == 0 && v.stride(0) >= v.size(1))) &&
           aligned16(v.data_ptr());
  };
  if (ok(t)) return t;
  Tensor c = torch::stable::contiguous(t);
  return ok(c) ? c : torch::stable::clone(c);
}

void check_cuda_tensor(const Tensor& t, const char* name, int32_t device) {
  STD_TORCH_CHECK(t.is_cuda(), "nf12: ", name, " must be a CUDA tensor");
  STD_TORCH_CHECK(t.get_device_index() == device, "nf12: ", name, " must be on cuda:", device, ", got cuda:",
                  t.get_device_index());
}

void check_shape(int64_t n, int64_t k, const char* what) {
  STD_TORCH_CHECK(n > 0 && n % 16 == 0, "nf12: ", what, " needs N (out_features) to be a positive multiple of 16, got ",
                  n);
  STD_TORCH_CHECK(k > 0 && k % 64 == 0, "nf12: ", what, " needs K (in_features) to be a positive multiple of 64, got ",
                  k);
}

nf12_weight_t weight_view(const Tensor& packed, const Tensor& escapes, int64_t exp_base, int32_t device) {
  check_cuda_tensor(packed, "packed", device);
  check_cuda_tensor(escapes, "escapes", device);
  STD_TORCH_CHECK(packed.scalar_type() == ScalarType::Byte && packed.dim() == 2,
                  "nf12: packed must be a uint8 [N, K*3/2] tensor");
  STD_TORCH_CHECK(packed.size(1) % 3 == 0, "nf12: packed row length must be K*3/2, got ", packed.size(1));
  STD_TORCH_CHECK(packed.is_contiguous() && aligned16(packed.data_ptr()),
                  "nf12: packed must be contiguous and 16-byte aligned");
  STD_TORCH_CHECK(escapes.scalar_type() == ScalarType::Short && escapes.dim() == 2 && escapes.size(1) == 8,
                  "nf12: escapes must be an int16 [E, 8] tensor");
  STD_TORCH_CHECK(escapes.is_contiguous() && aligned16(escapes.data_ptr()),
                  "nf12: escapes must be contiguous and 16-byte aligned");
  STD_TORCH_CHECK(exp_base >= 0 && exp_base <= kMaxExpBase, "nf12: exp_base must be in [0, 120], got ", exp_base);
  STD_TORCH_CHECK(escapes.size(0) <= INT64_C(0xffffffff), "nf12: too many escapes");
  const int64_t n = packed.size(0);
  const int64_t k = packed.size(1) / 3 * 2;
  check_shape(n, k, "packed weight");

  nf12_weight_t w{};
  w.packed = packed.data_ptr();
  w.escapes = escapes.size(0) > 0 ? escapes.data_ptr() : nullptr;
  w.num_escapes = static_cast<uint32_t>(escapes.size(0));
  w.n = n;
  w.k = k;
  w.exp_base = static_cast<uint8_t>(exp_base);
  return w;
}

Tensor check_weight_matrix(const Tensor& w, const char* op) {
  STD_TORCH_CHECK(w.is_cuda(), "nf12.", op, ": weight must be a CUDA tensor");
  STD_TORCH_CHECK(w.scalar_type() == ScalarType::BFloat16 && w.dim() == 2, "nf12.", op,
                  ": weight must be a bfloat16 [N, K] tensor");
  check_shape(w.size(0), w.size(1), op);
  return vector_rows(w);
}

// Current-stream ordering permits reuse of one decoded slice buffer.
// addmm adds bias before the single BF16 rounding, matching nf12_linear.
void decode_gemm(const Tensor& x2, const nf12_weight_t& w, std::optional<Tensor>& bias, Tensor y2, int64_t rows,
                 cudaStream_t stream) {
  Tensor buf = torch::stable::new_empty(x2, {rows, w.k});
  for (int64_t n0 = 0; n0 < w.n; n0 += rows) {
    const int64_t r = std::min(rows, w.n - n0);
    nf12_weight_t slice = w;
    slice.packed = static_cast<const uint8_t*>(w.packed) + n0 * w.k * 3 / 2;
    slice.n = r;
    Tensor wd = r == rows ? buf : torch::stable::narrow(buf, 0, 0, r);
    check_status(nf12_decode(&slice, wd.data_ptr(), w.k, stream), "linear");
    const Tensor wt = torch::stable::transpose(wd, 0, 1);
    const Tensor ys = torch::stable::narrow(y2, 1, n0, r);
    if (bias.has_value()) {
      const Tensor bs = torch::stable::narrow(*bias, 0, n0, r);
      TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_addmm_out(ys.get(), bs.get(), x2.get(), wt.get(), 1.0, 1.0));
    } else {
      TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(ys.get(), x2.get(), wt.get()));
    }
  }
}

template <typename Launch>
Tensor run_linear(const Tensor& x, int64_t n, int64_t k, const std::optional<Tensor>& bias, int32_t device,
                  const char* op, const nf12_weight_t* w, Launch&& launch) {
  check_cuda_tensor(x, "x", device);
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::BFloat16, "nf12.", op, ": x must be bfloat16");
  STD_TORCH_CHECK(x.dim() >= 1 && x.size(x.dim() - 1) == k, "nf12.", op, ": x must have shape [..., ", k, "]");

  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n;
  const int64_t m = x.numel() / k;
  Tensor x2 = x.dim() == 2 ? x : torch::stable::reshape(x, {m, k});
  Tensor y = torch::stable::new_empty(x, out_sizes);
  if (m == 0) return y;
  x2 = vector_rows(x2);

  const void* bias_ptr = nullptr;
  std::optional<Tensor> bias_buf;
  if (bias.has_value()) {
    check_cuda_tensor(*bias, "bias", device);
    STD_TORCH_CHECK(bias->scalar_type() == ScalarType::BFloat16 && bias->dim() == 1 && bias->size(0) == n, "nf12.", op,
                    ": bias must be a bfloat16 [", n, "] tensor");
    bias_buf = bias->is_contiguous() ? *bias : torch::stable::contiguous(*bias);
    if (!aligned16(bias_buf->data_ptr())) bias_buf = torch::stable::clone(*bias_buf);
    bias_ptr = bias_buf->data_ptr();
  }
  const int64_t rows = w != nullptr ? nf12_linear_decode_gemm_rows(m, n, k, device) : 0;
  if (rows > 0) {
    decode_gemm(x2, *w, bias_buf, y.dim() == 2 ? y : torch::stable::reshape(y, {m, n}), rows, current_stream(device));
    return y;
  }

  const size_t ws_bytes = nf12_linear_workspace_bytes(m, n, k, device);
  std::optional<Tensor> ws;
  if (ws_bytes > 0) ws = byte_buffer(x, ws_bytes);
  const int64_t ldx = m == 1 ? k : x2.stride(0);
  check_status(launch(x2.data_ptr(), m, ldx, bias_ptr, y.data_ptr(), n, ws ? ws->data_ptr() : nullptr, ws_bytes,
                      current_stream(device)),
               op);
  return y;
}

std::tuple<Tensor, Tensor, int64_t> encode(const Tensor& w_in) {
  const int32_t device = w_in.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  const Tensor w = check_weight_matrix(w_in, "encode");
  const int64_t n = w.size(0);
  const int64_t k = w.size(1);
  const cudaStream_t stream = current_stream(device);

  const size_t ws_bytes = nf12_encode_workspace_bytes(n, k);
  Tensor ws = byte_buffer(w, ws_bytes);
  uint8_t exp_base = 0;
  uint32_t num_escapes = 0;
  check_status(nf12_encode_analyze(w.data_ptr(), n, k, w.stride(0), ws.data_ptr(), ws_bytes, &exp_base, &num_escapes,
                                   stream),
               "encode");

  Tensor packed = torch::stable::new_empty(w, {n, k * 3 / 2}, ScalarType::Byte);
  Tensor escapes = torch::stable::new_empty(w, {static_cast<int64_t>(num_escapes), 8}, ScalarType::Short);
  check_status(nf12_encode(w.data_ptr(), n, k, w.stride(0), exp_base, packed.data_ptr(),
                           num_escapes > 0 ? escapes.data_ptr() : nullptr, num_escapes, ws.data_ptr(), ws_bytes,
                           stream),
               "encode");
  return {packed, escapes, static_cast<int64_t>(exp_base)};
}

Tensor decode(const Tensor& packed, const Tensor& escapes, int64_t exp_base) {
  const int32_t device = packed.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  const nf12_weight_t w = weight_view(packed, escapes, exp_base, device);
  Tensor out = torch::stable::new_empty(packed, {w.n, w.k}, ScalarType::BFloat16);
  check_status(nf12_decode(&w, out.data_ptr(), w.k, current_stream(device)), "decode");
  return out;
}

Tensor linear(const Tensor& x, const Tensor& packed, const Tensor& escapes, int64_t exp_base,
              const std::optional<Tensor>& bias) {
  const int32_t device = packed.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  const nf12_weight_t w = weight_view(packed, escapes, exp_base, device);
  return run_linear(x, w.n, w.k, bias, device, "linear", &w,
                    [&](const void* xp, int64_t m, int64_t ldx, const void* bp, void* yp, int64_t ldy, void* ws,
                        size_t ws_bytes, cudaStream_t stream) {
                      return nf12_linear(xp, m, ldx, &w, bp, yp, ldy, ws, ws_bytes, stream);
                    });
}

Tensor pack_bf16t(const Tensor& w_in) {
  const int32_t device = w_in.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  const Tensor w = check_weight_matrix(w_in, "pack_bf16t");
  const int64_t n = w.size(0);
  const int64_t k = w.size(1);
  Tensor out = torch::stable::new_empty(w, {n, 2 * k}, ScalarType::Byte);
  check_status(nf12_pack_bf16t(w.data_ptr(), n, k, w.stride(0), out.data_ptr(), current_stream(device)), "pack_bf16t");
  return out;
}

Tensor linear_bf16t(const Tensor& x, const Tensor& w_bf16t, const std::optional<Tensor>& bias) {
  const int32_t device = w_bf16t.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  check_cuda_tensor(w_bf16t, "w_bf16t", device);
  STD_TORCH_CHECK(w_bf16t.scalar_type() == ScalarType::Byte && w_bf16t.dim() == 2 && w_bf16t.size(1) % 2 == 0,
                  "nf12.linear_bf16t: w_bf16t must be a uint8 [N, 2*K] tensor from pack_bf16t");
  STD_TORCH_CHECK(w_bf16t.is_contiguous() && aligned16(w_bf16t.data_ptr()),
                  "nf12.linear_bf16t: w_bf16t must be contiguous and 16-byte aligned");
  const int64_t n = w_bf16t.size(0);
  const int64_t k = w_bf16t.size(1) / 2;
  check_shape(n, k, "linear_bf16t");
  const void* wp = w_bf16t.data_ptr();
  return run_linear(x, n, k, bias, device, "linear_bf16t", nullptr,
                    [&](const void* xp, int64_t m, int64_t ldx, const void* bp, void* yp, int64_t ldy, void* ws,
                        size_t ws_bytes, cudaStream_t stream) {
                      return nf12_linear_bf16t(xp, m, ldx, wp, n, k, bp, yp, ldy, ws, ws_bytes, stream);
                    });
}

Tensor embedding(const Tensor& ids_in, const Tensor& packed, const Tensor& escapes, int64_t exp_base) {
  const int32_t device = packed.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  const nf12_weight_t w = weight_view(packed, escapes, exp_base, device);
  check_cuda_tensor(ids_in, "ids", device);
  const ScalarType id_type = ids_in.scalar_type();
  STD_TORCH_CHECK(id_type == ScalarType::Int || id_type == ScalarType::Long,
                  "nf12.embedding: ids must be an int32 or int64 tensor");
  std::vector<int64_t> out_sizes(ids_in.sizes().begin(), ids_in.sizes().end());
  out_sizes.push_back(w.k);
  Tensor out = torch::stable::new_empty(packed, out_sizes, ScalarType::BFloat16);
  const int64_t num_ids = ids_in.numel();
  if (num_ids == 0) return out;
  const Tensor ids = ids_in.is_contiguous() ? ids_in : torch::stable::contiguous(ids_in);
  const cudaStream_t stream = current_stream(device);
  check_status(id_type == ScalarType::Int
                   ? nf12_embedding(&w, static_cast<const int32_t*>(ids.data_ptr()), num_ids, out.data_ptr(), w.k,
                                    stream)
                   : nf12_embedding_i64(&w, static_cast<const int64_t*>(ids.data_ptr()), num_ids, out.data_ptr(), w.k,
                                        stream),
               "embedding");
  return out;
}

// A contiguous 1-D tensor of `dtype` on `device` (copied if strided).
Tensor index_vector(const Tensor& t, ScalarType dtype, const char* op, const char* name, const char* type,
                    int32_t device) {
  check_cuda_tensor(t, name, device);
  STD_TORCH_CHECK(t.scalar_type() == dtype && t.dim() == 1, "nf12.", op, ": ", name, " must be a 1-D ", type,
                  " tensor");
  return t.is_contiguous() ? t : torch::stable::contiguous(t);
}

nf12_moe_weight_t moe_weight_view(const Tensor& packed, const Tensor& escapes, const Tensor& escape_offsets,
                                  const Tensor& exp_base, int32_t device) {
  check_cuda_tensor(packed, "packed", device);
  check_cuda_tensor(escapes, "escapes", device);
  check_cuda_tensor(escape_offsets, "escape_offsets", device);
  check_cuda_tensor(exp_base, "exp_base", device);
  STD_TORCH_CHECK(packed.scalar_type() == ScalarType::Byte && packed.dim() == 3 && packed.size(0) > 0,
                  "nf12: MoE packed must be a uint8 [E, N, K*3/2] tensor with E >= 1");
  STD_TORCH_CHECK(packed.size(2) % 3 == 0, "nf12: packed row length must be K*3/2, got ", packed.size(2));
  STD_TORCH_CHECK(packed.is_contiguous() && aligned16(packed.data_ptr()),
                  "nf12: packed must be contiguous and 16-byte aligned");
  STD_TORCH_CHECK(escapes.scalar_type() == ScalarType::Short && escapes.dim() == 2 && escapes.size(1) == 8,
                  "nf12: escapes must be an int16 [S, 8] tensor");
  STD_TORCH_CHECK(escapes.is_contiguous() && aligned16(escapes.data_ptr()),
                  "nf12: escapes must be contiguous and 16-byte aligned");
  const int64_t num_experts = packed.size(0);
  STD_TORCH_CHECK(escape_offsets.scalar_type() == ScalarType::Long && escape_offsets.dim() == 1 &&
                      escape_offsets.size(0) == num_experts + 1 && escape_offsets.is_contiguous(),
                  "nf12: escape_offsets must be a contiguous int64 [E + 1] tensor");
  STD_TORCH_CHECK(exp_base.scalar_type() == ScalarType::Byte && exp_base.dim() == 1 &&
                      exp_base.size(0) == num_experts && exp_base.is_contiguous(),
                  "nf12: exp_base must be a contiguous uint8 [E] tensor");
  const int64_t n = packed.size(1);
  const int64_t k = packed.size(2) / 3 * 2;
  check_shape(n, k, "MoE packed weight");

  nf12_moe_weight_t w{};
  w.packed = packed.data_ptr();
  w.escapes = escapes.size(0) > 0 ? escapes.data_ptr() : nullptr;
  w.escape_offsets = static_cast<const int64_t*>(escape_offsets.data_ptr());
  w.exp_base = static_cast<const uint8_t*>(exp_base.data_ptr());
  w.num_experts = num_experts;
  w.n = n;
  w.k = k;
  return w;
}

nf12::torch_detail::FusedDecodeWeight moe_bf16t_weight_view(const Tensor& packed, int32_t device) {
  check_cuda_tensor(packed, "w_bf16t", device);
  STD_TORCH_CHECK(packed.scalar_type() == ScalarType::Byte && packed.dim() == 3 && packed.size(0) > 0 &&
                      packed.size(2) % 2 == 0,
                  "nf12.moe_linear_bf16t: w_bf16t must be a uint8 [E, N, 2*K] tensor from pack_bf16t");
  STD_TORCH_CHECK(packed.is_contiguous() && aligned16(packed.data_ptr()),
                  "nf12.moe_linear_bf16t: w_bf16t must be contiguous and 16-byte aligned");
  const int64_t n = packed.size(1);
  const int64_t k = packed.size(2) / 2;
  check_shape(n, k, "moe_linear_bf16t");
  return {packed.data_ptr(), nullptr, nullptr, nullptr, packed.size(0), n, k, true};
}

nf12::torch_detail::FusedDecodeWeight fused_weight_view(
    const Tensor& packed, const std::optional<Tensor>& escapes, const std::optional<Tensor>& escape_offsets,
    const std::optional<Tensor>& exp_base, int32_t device) {
  STD_TORCH_CHECK(escapes.has_value() == escape_offsets.has_value() && escapes.has_value() == exp_base.has_value(),
                  "nf12._fused_moe: escapes, escape_offsets and exp_base must all be present or all be None");
  if (!escapes) return moe_bf16t_weight_view(packed, device);
  const auto w = moe_weight_view(packed, *escapes, *escape_offsets, *exp_base, device);
  return {w.packed, w.escapes, w.escape_offsets, w.exp_base, w.num_experts, w.n, w.k, false};
}

// Gathered calls without scatter, scaling or zero fill can use grouped GEMM.
// Reuse one decoded slice buffer on the current stream. Rows beyond the valid
// positions remain unspecified, as in moe_linear without zero fill.
Tensor moe_decode_gemm(const Tensor& x2, int64_t ldx, const Tensor& gather, int64_t gather_div,
                       const Tensor& offsets, int64_t a, const nf12_moe_weight_t& w, int64_t rows,
                       cudaStream_t stream) {
  const int64_t e = w.num_experts;
  const int64_t k = w.k;
  const Tensor ends = torch::stable::new_empty(offsets, {e}, ScalarType::Int);
  const Tensor xg = torch::stable::new_empty(x2, {a, k});
  check_status(nf12_moe_grouped_operands(x2.data_ptr(), ldx, static_cast<const int32_t*>(gather.data_ptr()),
                                         gather_div, static_cast<const int64_t*>(offsets.data_ptr()), a, e, k,
                                         xg.data_ptr(), k, static_cast<int32_t*>(ends.data_ptr()), stream),
               "moe_linear");
  Tensor buf = torch::stable::new_empty(x2, {e * rows * k});
  std::optional<Tensor> y;
  if (rows < w.n) y = torch::stable::new_empty(x2, {a, w.n});
  for (int64_t n0 = 0; n0 < w.n; n0 += rows) {
    const int64_t r = std::min(rows, w.n - n0);
    Tensor wd = torch::stable::view(torch::stable::narrow(buf, 0, 0, e * r * k), {e, r, k});
    check_status(nf12_moe_decode(&w, static_cast<const int64_t*>(offsets.data_ptr()), a, n0, r, wd.data_ptr(), k,
                                 stream),
                 "moe_linear");
    std::array<StableIValue, 5> stack{
        torch::stable::detail::from(xg), torch::stable::detail::from(torch::stable::transpose(wd, 1, 2)),
        torch::stable::detail::from(std::optional<Tensor>(ends)), torch::stable::detail::from(std::nullopt),
        torch::stable::detail::from(std::nullopt)};
    TORCH_ERROR_CODE_CHECK(torch_call_dispatcher("aten::_grouped_mm", "", stack.data(), TORCH_ABI_VERSION));
    Tensor out = torch::stable::detail::to<Tensor>(stack[0]);
    if (!y) return out;
    Tensor ys = torch::stable::narrow(*y, 1, n0, r);
    torch::stable::copy_(ys, out);
  }
  return *y;
}

// Dropped assignments are never written, so zero_fill must initialize y.
// Assignment count comes from gather, then scatter, then x.
template <typename Launch>
Tensor run_moe_linear(const Tensor& x, int64_t num_experts, int64_t n, int64_t k, const Tensor& expert_offsets,
                      const std::optional<Tensor>& gather, int64_t gather_div, const std::optional<Tensor>& scatter,
                      const std::optional<Tensor>& row_scale, bool zero_fill, int32_t device, const char* op,
                      const nf12_moe_weight_t* w, Launch&& launch) {
  check_cuda_tensor(x, "x", device);
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::BFloat16 && x.dim() == 2 && x.size(1) == k, "nf12.", op,
                  ": x must be a bfloat16 [rows, ", k, "] tensor");
  const Tensor offsets = index_vector(expert_offsets, ScalarType::Long, op, "expert_offsets", "int64", device);
  STD_TORCH_CHECK(offsets.size(0) == num_experts + 1, "nf12.", op, ": expert_offsets must have E + 1 = ",
                  num_experts + 1, " entries, got ", offsets.size(0));
  std::optional<Tensor> g, s, r;
  if (gather.has_value()) g = index_vector(*gather, ScalarType::Int, op, "gather", "int32", device);
  if (scatter.has_value()) s = index_vector(*scatter, ScalarType::Int, op, "scatter", "int32", device);
  STD_TORCH_CHECK(!g || !s || g->size(0) == s->size(0), "nf12.", op, ": gather and scatter must have the same length");
  const int64_t num_assignments = g ? g->size(0) : s ? s->size(0) : x.size(0);
  STD_TORCH_CHECK(num_assignments <= INT32_MAX, "nf12.", op, ": more than 2^31 - 1 assignments");
  if (row_scale.has_value()) {
    r = index_vector(*row_scale, ScalarType::Float, op, "row_scale", "float32", device);
    STD_TORCH_CHECK(r->size(0) == num_assignments, "nf12.", op, ": row_scale must have ", num_assignments,
                    " entries, got ", r->size(0));
  }

  if (num_assignments == 0) {
    return zero_fill ? torch::stable::new_zeros(x, {num_assignments, n})
                     : torch::stable::new_empty(x, {num_assignments, n});
  }
  STD_TORCH_CHECK(x.size(0) > 0, "nf12.", op, ": x has no rows");
  const Tensor x2 = vector_rows(x);
  const int64_t ldx = x2.size(0) == 1 ? k : x2.stride(0);
  const int64_t rows = w != nullptr && g && !s && !r && !zero_fill
                           ? nf12_moe_linear_decode_gemm_rows(num_assignments, num_experts, n, k, device)
                           : 0;
  if (rows > 0) {
    return moe_decode_gemm(x2, ldx, *g, gather_div, offsets, num_assignments, *w, rows, current_stream(device));
  }

  Tensor y = zero_fill ? torch::stable::new_zeros(x, {num_assignments, n})
                       : torch::stable::new_empty(x, {num_assignments, n});
  const size_t ws_bytes = nf12_moe_linear_workspace_bytes(num_assignments, num_experts, n, k, device);
  std::optional<Tensor> ws;
  if (ws_bytes > 0) ws = byte_buffer(x, ws_bytes);
  auto ptr = [](const std::optional<Tensor>& t) { return t ? t->data_ptr() : nullptr; };
  check_status(launch(x2.data_ptr(), ldx, static_cast<const int32_t*>(ptr(g)), static_cast<const int32_t*>(ptr(s)),
                      static_cast<const int64_t*>(offsets.data_ptr()), num_assignments,
                      static_cast<const float*>(ptr(r)), y.data_ptr(), ws ? ws->data_ptr() : nullptr, ws_bytes,
                      current_stream(device)),
               op);
  return y;
}

std::tuple<Tensor, Tensor> moe_align(const Tensor& topk_ids_in, int64_t num_experts) {
  const int32_t device = topk_ids_in.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  check_cuda_tensor(topk_ids_in, "topk_ids", device);
  STD_TORCH_CHECK(topk_ids_in.scalar_type() == ScalarType::Int && topk_ids_in.dim() == 2 && topk_ids_in.size(1) > 0,
                  "nf12.moe_align: topk_ids must be an int32 [T, topk] tensor with topk >= 1");
  STD_TORCH_CHECK(num_experts >= 1 && num_experts <= INT32_MAX, "nf12.moe_align: num_experts must be in [1, 2^31 - 1]");
  const Tensor ids = topk_ids_in.is_contiguous() ? topk_ids_in : torch::stable::contiguous(topk_ids_in);
  const int64_t t = ids.size(0);
  const int64_t topk = ids.size(1);
  STD_TORCH_CHECK(t <= INT32_MAX / topk, "nf12.moe_align: more than 2^31 - 1 assignments");
  Tensor sorted = torch::stable::new_empty(ids, {t * topk});
  Tensor offsets = torch::stable::new_empty(ids, {num_experts + 1}, ScalarType::Long);
  const size_t ws_bytes = nf12_moe_align_workspace_bytes(t, topk, num_experts);
  std::optional<Tensor> ws;
  if (ws_bytes > 0) ws = byte_buffer(ids, ws_bytes);
  check_status(nf12_moe_align(static_cast<const int32_t*>(ids.data_ptr()), t, topk, num_experts,
                              static_cast<int32_t*>(sorted.data_ptr()), static_cast<int64_t*>(offsets.data_ptr()),
                              ws ? ws->data_ptr() : nullptr, ws_bytes, current_stream(device)),
               "moe_align");
  return {sorted, offsets};
}

Tensor moe_linear(const Tensor& x, const Tensor& packed, const Tensor& escapes, const Tensor& escape_offsets,
                  const Tensor& exp_base, const Tensor& expert_offsets, const std::optional<Tensor>& gather,
                  int64_t gather_div, const std::optional<Tensor>& scatter, const std::optional<Tensor>& row_scale,
                  bool zero_fill) {
  const int32_t device = packed.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  const nf12_moe_weight_t w = moe_weight_view(packed, escapes, escape_offsets, exp_base, device);
  STD_TORCH_CHECK(gather_div >= 1, "nf12.moe_linear: gather_div must be >= 1, got ", gather_div);
  return run_moe_linear(x, w.num_experts, w.n, w.k, expert_offsets, gather, gather_div, scatter, row_scale, zero_fill,
                        device, "moe_linear", &w,
                        [&](const void* xp, int64_t ldx, const int32_t* gp, const int32_t* sp, const int64_t* op,
                            int64_t a, const float* rp, void* yp, void* ws, size_t ws_bytes, cudaStream_t stream) {
                          return nf12_moe_linear(xp, ldx, gp, gather_div, sp, op, a, &w, rp, yp, w.n, ws, ws_bytes,
                                                 stream);
                        });
}

Tensor moe_linear_bf16t(const Tensor& x, const Tensor& w_bf16t, const Tensor& expert_offsets,
                        const std::optional<Tensor>& gather, int64_t gather_div, const std::optional<Tensor>& scatter,
                        const std::optional<Tensor>& row_scale, bool zero_fill) {
  const int32_t device = w_bf16t.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  const auto w = moe_bf16t_weight_view(w_bf16t, device);
  STD_TORCH_CHECK(gather_div >= 1, "nf12.moe_linear_bf16t: gather_div must be >= 1, got ", gather_div);
  const int64_t e = w.experts;
  const int64_t n = w.n;
  const int64_t k = w.k;
  const void* wp = w.packed;
  return run_moe_linear(x, e, n, k, expert_offsets, gather, gather_div, scatter, row_scale, zero_fill, device,
                        "moe_linear_bf16t", nullptr,
                        [&](const void* xp, int64_t ldx, const int32_t* gp, const int32_t* sp, const int64_t* op,
                            int64_t a, const float* rp, void* yp, void* ws, size_t ws_bytes, cudaStream_t stream) {
                          return nf12_moe_linear_bf16t(xp, ldx, gp, gather_div, sp, op, a, wp, e, n, k, rp, yp, n, ws,
                                                       ws_bytes, stream);
                        });
}

// Recovers the encoder input without host synchronization.
Tensor moe_decode(const Tensor& packed, const Tensor& escapes, const Tensor& escape_offsets, const Tensor& exp_base) {
  const int32_t device = packed.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  const nf12_moe_weight_t w = moe_weight_view(packed, escapes, escape_offsets, exp_base, device);
  Tensor out = torch::stable::new_empty(packed, {w.num_experts, w.n, w.k}, ScalarType::BFloat16);
  check_status(nf12_moe_decode(&w, nullptr, 0, 0, w.n, out.data_ptr(), w.k, current_stream(device)), "moe_decode");
  return out;
}

Tensor moe_combine(const Tensor& y_in, int64_t topk) {
  const int32_t device = y_in.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  check_cuda_tensor(y_in, "y", device);
  STD_TORCH_CHECK(y_in.scalar_type() == ScalarType::BFloat16 && y_in.dim() == 2, "nf12.moe_combine: y must be a "
                  "bfloat16 [T * topk, N] tensor");
  STD_TORCH_CHECK(topk >= 1 && y_in.size(0) % topk == 0, "nf12.moe_combine: y rows (", y_in.size(0),
                  ") must be a multiple of topk (", topk, ")");
  const int64_t n = y_in.size(1);
  STD_TORCH_CHECK(n > 0 && n % 8 == 0, "nf12.moe_combine: N must be a positive multiple of 8, got ", n);
  const int64_t t = y_in.size(0) / topk;
  Tensor out = torch::stable::new_empty(y_in, {t, n});
  if (t == 0) return out;
  const Tensor y = vector_rows(y_in);
  const int64_t ldy = y.size(0) == 1 ? n : y.stride(0);
  check_status(nf12_moe_combine(y.data_ptr(), ldy, t, topk, n, out.data_ptr(), n, current_stream(device)),
               "moe_combine");
  return out;
}

Tensor moe_silu_mul(const Tensor& h_in) {
  const int32_t device = h_in.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  check_cuda_tensor(h_in, "h", device);
  STD_TORCH_CHECK(h_in.scalar_type() == ScalarType::BFloat16 && h_in.dim() == 2, "nf12.moe_silu_mul: h must be a "
                  "bfloat16 [rows, 2 * I] tensor");
  const int64_t inter = h_in.size(1) / 2;
  STD_TORCH_CHECK(h_in.size(1) % 2 == 0 && inter > 0 && inter % 8 == 0, "nf12.moe_silu_mul: h columns (",
                  h_in.size(1), ") must be 2 * I with I a positive multiple of 8");
  const int64_t rows = h_in.size(0);
  Tensor out = torch::stable::new_empty(h_in, {rows, inter});
  if (rows == 0) return out;
  const Tensor h = vector_rows(h_in);
  const int64_t ldh = h.size(0) == 1 ? 2 * inter : h.stride(0);
  check_status(nf12_moe_silu_mul(h.data_ptr(), ldh, rows, inter, out.data_ptr(), inter, current_stream(device)),
               "moe_silu_mul");
  return out;
}

Tensor fused_moe(const Tensor& x, const Tensor& packed13, const std::optional<Tensor>& escapes13,
                 const std::optional<Tensor>& escape_offsets13, const std::optional<Tensor>& exp_base13,
                 const Tensor& packed2, const std::optional<Tensor>& escapes2,
                 const std::optional<Tensor>& escape_offsets2, const std::optional<Tensor>& exp_base2,
                 const Tensor& topk_weights, const Tensor& topk_ids, bool zero_fill) {
  const int32_t device = packed13.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  const auto w13 = fused_weight_view(packed13, escapes13, escape_offsets13, exp_base13, device);
  const auto w2 = fused_weight_view(packed2, escapes2, escape_offsets2, exp_base2, device);
  STD_TORCH_CHECK(w13.experts == w2.experts && w13.n == 2 * w2.k,
                  "nf12._fused_moe: weights need the same E and N13 == 2 * K2");
  STD_TORCH_CHECK(w13.experts <= INT32_MAX, "nf12._fused_moe: num_experts must be in [1, 2^31 - 1]");
  check_cuda_tensor(x, "x", device);
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::BFloat16 && x.dim() == 2 && x.size(1) == w13.k,
                  "nf12._fused_moe: x must be a bfloat16 [T, ", w13.k, "] tensor");
  check_cuda_tensor(topk_ids, "topk_ids", device);
  STD_TORCH_CHECK(topk_ids.scalar_type() == ScalarType::Int && topk_ids.dim() == 2 && topk_ids.size(1) > 0 &&
                      topk_ids.size(0) == x.size(0),
                  "nf12._fused_moe: topk_ids must be an int32 [T, topk] tensor with topk >= 1");
  check_cuda_tensor(topk_weights, "topk_weights", device);
  STD_TORCH_CHECK(topk_weights.scalar_type() == ScalarType::Float && topk_weights.dim() == 2 &&
                      topk_weights.size(0) == topk_ids.size(0) && topk_weights.size(1) == topk_ids.size(1),
                  "nf12._fused_moe: topk_weights must be a float32 [T, topk] tensor");
  const int64_t t = x.size(0);
  const int64_t topk = topk_ids.size(1);
  STD_TORCH_CHECK(t <= INT32_MAX / topk, "nf12._fused_moe: more than 2^31 - 1 assignments");
  if (t == 0) return torch::stable::new_empty(x, {0, w2.n});
  const Tensor ids = topk_ids.is_contiguous() ? topk_ids : torch::stable::contiguous(topk_ids);
  const Tensor scales = topk_weights.is_contiguous() ? topk_weights : torch::stable::contiguous(topk_weights);
  if (t == 1) {
    nf12::torch_detail::FusedDecodeArgs args{
        w13, w2, x.data_ptr(), static_cast<const int32_t*>(ids.data_ptr()),
        static_cast<const float*>(scales.data_ptr()), nullptr, nullptr,
        topk, w13.k, topk, topk, w2.k, w2.n};
    if (nf12::torch_detail::fused_decode_supported(device, args)) {
      const Tensor x2 = vector_rows(x);
      args.x = x2.data_ptr();
      Tensor activation = torch::stable::new_empty(x, {topk, w2.k});
      Tensor out = torch::stable::new_empty(x, {1, w2.n});
      args.activation = activation.data_ptr();
      args.out = out.data_ptr();
      const cudaError_t err = nf12::torch_detail::fused_decode(args, current_stream(device));
      STD_TORCH_CHECK(err == cudaSuccess, "nf12._fused_moe failed: ", cudaGetErrorString(err));
      return out;
    }
  }

  // Keep the existing kernels and large-T grouped-GEMM policy as the fallback.
  auto [sorted, offsets] = moe_align(ids, w13.experts);
  Tensor h = w13.bf16t
                 ? moe_linear_bf16t(x, packed13, offsets, sorted, topk, std::nullopt, std::nullopt, false)
                 : moe_linear(x, packed13, *escapes13, *escape_offsets13, *exp_base13, offsets, sorted, topk,
                              std::nullopt, std::nullopt, false);
  h = moe_silu_mul(h);
  const Tensor row_scale = torch::stable::view(scales, {t * topk});
  const Tensor y = w2.bf16t
                       ? moe_linear_bf16t(h, packed2, offsets, std::nullopt, 1, sorted, row_scale, zero_fill)
                       : moe_linear(h, packed2, *escapes2, *escape_offsets2, *exp_base2, offsets, std::nullopt, 1,
                                    sorted, row_scale, zero_fill);
  return moe_combine(y, topk);
}

// Per-expert analysis synchronizes before allocating the stacked escape table.
std::tuple<Tensor, Tensor, Tensor, Tensor> encode_moe(const Tensor& w_in) {
  const int32_t device = w_in.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  STD_TORCH_CHECK(w_in.is_cuda(), "nf12.encode_moe: weight must be a CUDA tensor");
  STD_TORCH_CHECK(w_in.scalar_type() == ScalarType::BFloat16 && w_in.dim() == 3 && w_in.size(0) > 0,
                  "nf12.encode_moe: weight must be a bfloat16 [E, N, K] tensor with E >= 1");
  const int64_t num_experts = w_in.size(0);
  const int64_t n = w_in.size(1);
  const int64_t k = w_in.size(2);
  check_shape(n, k, "encode_moe");
  const cudaStream_t stream = current_stream(device);

  std::vector<Tensor> experts;
  experts.reserve(num_experts);
  for (int64_t e = 0; e < num_experts; ++e) experts.push_back(vector_rows(torch::stable::select(w_in, 0, e)));

  const size_t ws_bytes = nf12_encode_workspace_bytes(n, k);
  Tensor ws = byte_buffer(w_in, ws_bytes);
  std::vector<uint8_t> exp_base(num_experts);
  std::vector<uint32_t> counts(num_experts);
  std::vector<int64_t> offsets(num_experts + 1, 0);
  for (int64_t e = 0; e < num_experts; ++e) {
    const Tensor& we = experts[e];
    check_status(nf12_encode_analyze(we.data_ptr(), n, k, we.stride(0), ws.data_ptr(), ws_bytes, &exp_base[e],
                                     &counts[e], stream),
                 "encode_moe");
    offsets[e + 1] = offsets[e] + counts[e];
  }

  const int64_t expert_bytes = n * k * 3 / 2;
  Tensor packed = torch::stable::new_empty(w_in, {num_experts, n, k * 3 / 2}, ScalarType::Byte);
  Tensor escapes = torch::stable::new_empty(w_in, {offsets[num_experts], 8}, ScalarType::Short);
  auto* packed_bytes = static_cast<uint8_t*>(packed.data_ptr());
  auto* escape_bytes = static_cast<uint8_t*>(escapes.data_ptr());
  for (int64_t e = 0; e < num_experts; ++e) {
    const Tensor& we = experts[e];
    check_status(nf12_encode(we.data_ptr(), n, k, we.stride(0), exp_base[e], packed_bytes + e * expert_bytes,
                             counts[e] > 0 ? escape_bytes + offsets[e] * 16 : nullptr, counts[e], ws.data_ptr(),
                             ws_bytes, stream),
                 "encode_moe");
  }

  Tensor escape_offsets = torch::stable::new_empty(w_in, {num_experts + 1}, ScalarType::Long);
  Tensor exp_base_t = torch::stable::new_empty(w_in, {num_experts}, ScalarType::Byte);
  cudaError_t err = cudaMemcpyAsync(escape_offsets.data_ptr(), offsets.data(), offsets.size() * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(exp_base_t.data_ptr(), exp_base.data(), exp_base.size(), cudaMemcpyHostToDevice, stream);
  }
  // Complete copies before the host vectors are destroyed.
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(err == cudaSuccess, "nf12.encode_moe failed: ", cudaGetErrorString(err));
  return {packed, escapes, escape_offsets, exp_base_t};
}

}

STABLE_TORCH_LIBRARY(nf12, m) {
  m.def("encode(Tensor w) -> (Tensor packed, Tensor escapes, int exp_base)");
  m.def("decode(Tensor packed, Tensor escapes, int exp_base) -> Tensor");
  m.def("linear(Tensor x, Tensor packed, Tensor escapes, int exp_base, Tensor? bias=None) -> Tensor");
  m.def("pack_bf16t(Tensor w) -> Tensor");
  m.def("linear_bf16t(Tensor x, Tensor w_bf16t, Tensor? bias=None) -> Tensor");
  m.def("embedding(Tensor ids, Tensor packed, Tensor escapes, int exp_base) -> Tensor");
  m.def("encode_moe(Tensor w) -> (Tensor packed, Tensor escapes, Tensor escape_offsets, Tensor exp_base)");
  m.def("moe_align(Tensor topk_ids, int num_experts) -> (Tensor sorted, Tensor expert_offsets)");
  m.def("moe_decode(Tensor packed, Tensor escapes, Tensor escape_offsets, Tensor exp_base) -> Tensor");
  m.def(
      "moe_linear(Tensor x, Tensor packed, Tensor escapes, Tensor escape_offsets, Tensor exp_base, "
      "Tensor expert_offsets, Tensor? gather=None, int gather_div=1, Tensor? scatter=None, Tensor? row_scale=None, "
      "bool zero_fill=False) -> Tensor");
  m.def(
      "moe_linear_bf16t(Tensor x, Tensor w_bf16t, Tensor expert_offsets, Tensor? gather=None, int gather_div=1, "
      "Tensor? scatter=None, Tensor? row_scale=None, bool zero_fill=False) -> Tensor");
  m.def("moe_combine(Tensor y, int topk) -> Tensor");
  m.def("moe_silu_mul(Tensor h) -> Tensor");
  m.def(
      "_fused_moe(Tensor x, Tensor packed13, Tensor? escapes13, Tensor? escape_offsets13, Tensor? exp_base13, "
      "Tensor packed2, Tensor? escapes2, Tensor? escape_offsets2, Tensor? exp_base2, "
      "Tensor topk_weights, Tensor topk_ids, bool zero_fill=False) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(nf12, CUDA, m) {
  m.impl("encode", TORCH_BOX(&encode));
  m.impl("decode", TORCH_BOX(&decode));
  m.impl("linear", TORCH_BOX(&linear));
  m.impl("pack_bf16t", TORCH_BOX(&pack_bf16t));
  m.impl("linear_bf16t", TORCH_BOX(&linear_bf16t));
  m.impl("embedding", TORCH_BOX(&embedding));
  m.impl("encode_moe", TORCH_BOX(&encode_moe));
  m.impl("moe_align", TORCH_BOX(&moe_align));
  m.impl("moe_decode", TORCH_BOX(&moe_decode));
  m.impl("moe_linear", TORCH_BOX(&moe_linear));
  m.impl("moe_linear_bf16t", TORCH_BOX(&moe_linear_bf16t));
  m.impl("moe_combine", TORCH_BOX(&moe_combine));
  m.impl("moe_silu_mul", TORCH_BOX(&moe_silu_mul));
  m.impl("_fused_moe", TORCH_BOX(&fused_moe));
}
