// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Torch bindings for the Tenstorrent block-float CUDA emulation kernels.
//
// The device code lives in ``tt_blockfloat_cuda.cu`` and the host contract in
// ``tt_blockfloat_cuda.h``; this file only validates and marshals tensors.  It
// deliberately carries no layout knowledge beyond the buffer sizes, so the
// CUDA and HIP siblings expose an identical Python surface.

#include <torch/extension.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <optional>

#include "tt_blockfloat_cuda.h"

namespace {

void ok(cudaError_t status, const char* what) {
  TORCH_CHECK(status == cudaSuccess, what, " failed: ", cudaGetErrorString(status));
}

int dtype_code(const torch::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kFloat:
      return BLOCKFLOAT_DTYPE_F32;
    case at::kHalf:
      return BLOCKFLOAT_DTYPE_F16;
    case at::kBFloat16:
      return BLOCKFLOAT_DTYPE_BF16;
    default:
      TORCH_CHECK(false, "unsupported block-float operand dtype: ", t.scalar_type());
  }
}

// Every operand must live on the reference tensor's device and be contiguous:
// the kernels index raw pointers with no stride arithmetic.
void check_like(const torch::Tensor& t, const torch::Tensor& ref, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.device() == ref.device(), name, " must be on the same device as x");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// One exponent per 16 values along K, so ``n = numel / (k / 16)``.
int64_t out_features(const torch::Tensor& exponents, int64_t k) {
  const int64_t per_row = k / 16;
  TORCH_CHECK(exponents.numel() % per_row == 0,
              "exponents must hold a whole number of rows of K/16 values");
  return exponents.numel() / per_row;
}

// bf4 packs two codes per byte and rounds an odd count up, matching
// ``gptqmodel.utils.tt_blockfloat_quant._pack_codes``.
void check_codes(const torch::Tensor& codes, int64_t bits, int64_t n, int64_t k) {
  TORCH_CHECK(codes.scalar_type() == at::kByte, "codes must be uint8");
  const int64_t needed = bits == 4 ? (n * k + 1) / 2 : n * k;
  TORCH_CHECK(codes.numel() >= needed,
              "codes holds ", codes.numel(), " bytes but ", needed, " are required");
}

void check_exponents(const torch::Tensor& exponents, const torch::Tensor& ref,
                     int64_t rows, int64_t k) {
  check_like(exponents, ref, "exponents");
  TORCH_CHECK(exponents.scalar_type() == at::kChar, "exponents must be int8");
  TORCH_CHECK(exponents.numel() == rows * (k / 16),
              "exponents must be [", rows, ", ", k / 16, "]");
}

// ---------------------------------------------------------------------------
// linear
// ---------------------------------------------------------------------------
void blockfloat_linear_out(const torch::Tensor& x, const torch::Tensor& codes,
                           const torch::Tensor& exponents, int64_t bits,
                           const std::optional<torch::Tensor>& bias,
                           torch::Tensor& out) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "x must be a 2D CUDA tensor");
  TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4 or 8");
  // A sliced or transposed activation is materialised rather than rejected: the
  // dense path this replaces accepted any layout, and callers pass views such as
  // ``x[:, ::2]``.  ``contiguous()`` is a no-op when the tensor is already packed.
  const torch::Tensor xc = x.contiguous();
  const int64_t m = xc.size(0);
  const int64_t k = xc.size(1);
  TORCH_CHECK(m > 0 && k > 0, "x must not be empty");
  TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16");
  check_like(codes, xc, "codes");
  check_like(exponents, xc, "exponents");
  TORCH_CHECK(exponents.scalar_type() == at::kChar, "exponents must be int8");
  const int64_t n = out_features(exponents, k);
  check_codes(codes, bits, n, k);

  check_like(out, xc, "out");
  TORCH_CHECK(out.scalar_type() == xc.scalar_type(), "out dtype must match x");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == m && out.size(1) == n,
              "out must be [", m, ", ", n, "]");

  const void* bias_ptr = nullptr;
  if (bias.has_value()) {
    const torch::Tensor& b = *bias;
    check_like(b, xc, "bias");
    TORCH_CHECK(b.dim() == 1 && b.size(0) == n, "bias must be [n]");
    TORCH_CHECK(b.scalar_type() == xc.scalar_type(), "bias dtype must match x");
    bias_ptr = b.data_ptr();
  }

  const c10::cuda::CUDAGuard guard(xc.device());
  ok(blockfloat_linear(xc.data_ptr(), dtype_code(xc), codes.data_ptr<uint8_t>(),
                       exponents.data_ptr<int8_t>(), static_cast<int>(bits), m, n, k,
                       bias_ptr, out.data_ptr(), at::cuda::getCurrentCUDAStream()),
     "blockfloat_linear");
}

// ---------------------------------------------------------------------------
// dequant
// ---------------------------------------------------------------------------
void blockfloat_dequant_out(const torch::Tensor& codes, const torch::Tensor& exponents,
                            int64_t bits, int64_t rows, int64_t k, torch::Tensor& out) {
  TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4 or 8");
  TORCH_CHECK(rows > 0 && k > 0 && k % 16 == 0,
              "rows must be > 0 and K a multiple of 16");
  check_exponents(exponents, exponents, rows, k);
  check_like(codes, exponents, "codes");
  check_codes(codes, bits, rows, k);
  check_like(out, exponents, "out");
  TORCH_CHECK(out.numel() == rows * k, "out must hold rows * k elements");

  const c10::cuda::CUDAGuard guard(exponents.device());
  ok(blockfloat_dequant(codes.data_ptr<uint8_t>(), exponents.data_ptr<int8_t>(),
                        static_cast<int>(bits), rows, k, out.data_ptr(), dtype_code(out),
                        at::cuda::getCurrentCUDAStream()),
     "blockfloat_dequant");
}

// ---------------------------------------------------------------------------
// quantize / decompose
// ---------------------------------------------------------------------------
void blockfloat_quantize_out(const torch::Tensor& x, int64_t bits, torch::Tensor& codes,
                             torch::Tensor& exponents) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "x must be a 2D CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4 or 8");
  const torch::Tensor xc = x.contiguous();
  const int64_t rows = xc.size(0);
  const int64_t k = xc.size(1);
  TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16");
  check_like(codes, xc, "codes");
  check_codes(codes, bits, rows, k);
  check_exponents(exponents, xc, rows, k);

  const c10::cuda::CUDAGuard guard(xc.device());
  ok(blockfloat_quantize(xc.data_ptr<float>(), static_cast<int>(bits), rows, k,
                         codes.data_ptr<uint8_t>(), exponents.data_ptr<int8_t>(),
                         at::cuda::getCurrentCUDAStream()),
     "blockfloat_quantize");
}

void blockfloat_decompose_out(const torch::Tensor& x, int64_t bits, torch::Tensor& codes,
                              torch::Tensor& exponents, torch::Tensor& failed) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "x must be a 2D CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4 or 8");
  const torch::Tensor xc = x.contiguous();
  const int64_t rows = xc.size(0);
  const int64_t k = xc.size(1);
  TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16");
  check_like(codes, xc, "codes");
  check_codes(codes, bits, rows, k);
  check_exponents(exponents, xc, rows, k);
  check_like(failed, xc, "failed");
  TORCH_CHECK(failed.scalar_type() == at::kInt && failed.numel() == 1,
              "failed must be a one-element int32 tensor");

  const c10::cuda::CUDAGuard guard(xc.device());
  ok(blockfloat_decompose(xc.data_ptr<float>(), static_cast<int>(bits), rows, k,
                          codes.data_ptr<uint8_t>(), exponents.data_ptr<int8_t>(),
                          failed.data_ptr<int32_t>(), at::cuda::getCurrentCUDAStream()),
     "blockfloat_decompose");
}

// ---------------------------------------------------------------------------
// embedding
// ---------------------------------------------------------------------------
void blockfloat_embedding_out(const torch::Tensor& ids, const torch::Tensor& codes,
                              const torch::Tensor& exponents, int64_t bits, int64_t rows,
                              int64_t k, torch::Tensor& out) {
  TORCH_CHECK(ids.is_cuda(), "ids must be a CUDA tensor");
  TORCH_CHECK(ids.scalar_type() == at::kLong, "ids must be int64");
  const torch::Tensor idx = ids.contiguous();
  TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4 or 8");
  TORCH_CHECK(rows > 0 && k > 0 && k % 16 == 0,
              "rows must be > 0 and K a multiple of 16");
  check_like(codes, idx, "codes");
  check_like(exponents, idx, "exponents");
  TORCH_CHECK(exponents.scalar_type() == at::kChar, "exponents must be int8");
  check_exponents(exponents, idx, rows, k);
  check_codes(codes, bits, rows, k);
  check_like(out, idx, "out");
  const int64_t tokens = idx.numel();
  TORCH_CHECK(out.numel() == tokens * k, "out must hold tokens * k elements");

  const c10::cuda::CUDAGuard guard(idx.device());
  ok(blockfloat_embedding(idx.data_ptr<int64_t>(), codes.data_ptr<uint8_t>(),
                          exponents.data_ptr<int8_t>(), static_cast<int>(bits), tokens,
                          rows, k, out.data_ptr(), dtype_code(out),
                          at::cuda::getCurrentCUDAStream()),
     "blockfloat_embedding");
}

// ---------------------------------------------------------------------------
// selftest
// ---------------------------------------------------------------------------
void blockfloat_selftest_out(torch::Tensor& out) {
  TORCH_CHECK(out.is_cuda() && out.is_contiguous(), "out must be a contiguous CUDA tensor");
  TORCH_CHECK(out.scalar_type() == at::kInt && out.numel() == 1,
              "out must be a one-element int32 tensor");
  const c10::cuda::CUDAGuard guard(out.device());
  ok(blockfloat_selftest(out.data_ptr<int32_t>(), at::cuda::getCurrentCUDAStream()),
     "blockfloat_selftest");
}

}  // namespace

TORCH_LIBRARY(gptqmodel_blockfloat_cuda, m) {
  m.def(
      "blockfloat_linear(Tensor x, Tensor codes, Tensor exponents, int bits, "
      "Tensor? bias, Tensor(a!) out) -> ()");
  m.def(
      "blockfloat_dequant(Tensor codes, Tensor exponents, int bits, int rows, int k, "
      "Tensor(a!) out) -> ()");
  m.def(
      "blockfloat_quantize(Tensor x, int bits, Tensor(a!) codes, Tensor(b!) exponents) -> ()");
  m.def(
      "blockfloat_decompose(Tensor x, int bits, Tensor(a!) codes, Tensor(b!) exponents, "
      "Tensor(c!) failed) -> ()");
  m.def(
      "blockfloat_embedding(Tensor ids, Tensor codes, Tensor exponents, int bits, "
      "int rows, int k, Tensor(a!) out) -> ()");
  m.def("blockfloat_selftest(Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(gptqmodel_blockfloat_cuda, CUDA, m) {
  m.impl("blockfloat_linear", &blockfloat_linear_out);
  m.impl("blockfloat_dequant", &blockfloat_dequant_out);
  m.impl("blockfloat_quantize", &blockfloat_quantize_out);
  m.impl("blockfloat_decompose", &blockfloat_decompose_out);
  m.impl("blockfloat_embedding", &blockfloat_embedding_out);
  m.impl("blockfloat_selftest", &blockfloat_selftest_out);
}
