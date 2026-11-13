// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// Torch bindings for the Tenstorrent block-float HIP kernels.
//
// Host-only translation unit: no device code lives here, so the .hip file can
// be compiled without torch/ATen headers (mirrors allspark_torch_hip.cpp and
// nf12_ad_torch.cpp).
//
// Every op takes the packed ``codes`` buffer and the int8 ``exponents`` buffer
// exactly as the checkpoint stores them.  Nothing here expands a weight to a
// dense tensor except the explicit ``dequant`` entry point, and ``linear``
// never does.
// ---------------------------------------------------------------------------

#include <torch/extension.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <optional>
#include <tuple>

#include "tt_blockfloat_hip.h"

namespace {

void check_hip(hipError_t status, const char* what) {
    TORCH_CHECK(status == hipSuccess, what, ": ", hipGetErrorString(status));
}

int dtype_code(at::ScalarType type) {
    switch (type) {
        case at::kFloat:
            return TTBF_DTYPE_F32;
        case at::kHalf:
            return TTBF_DTYPE_F16;
        case at::kBFloat16:
            return TTBF_DTYPE_BF16;
        default:
            TORCH_CHECK(false, "tt_blockfloat_hip: unsupported dtype ", type);
    }
}

at::ScalarType scalar_type_from_code(int64_t code) {
    switch (code) {
        case TTBF_DTYPE_F32:
            return at::kFloat;
        case TTBF_DTYPE_F16:
            return at::kHalf;
        case TTBF_DTYPE_BF16:
            return at::kBFloat16;
        default:
            TORCH_CHECK(false, "tt_blockfloat_hip: unknown dtype code ", code);
    }
}

void check_bits(int64_t bits) {
    TORCH_CHECK(bits == 4 || bits == 8,
                "tt_blockfloat_hip: bits must be 4 or 8, got ", bits);
}

void check_weight(const torch::Tensor& codes, const torch::Tensor& exponents,
                  int64_t bits, int64_t rows, int64_t cols) {
    TORCH_CHECK(rows > 0 && cols > 0, "tt_blockfloat_hip: empty weight");
    TORCH_CHECK(cols % 16 == 0,
                "tt_blockfloat_hip: in_features must be a multiple of 16, got ",
                cols);
    TORCH_CHECK(codes.scalar_type() == at::kByte,
                "tt_blockfloat_hip: codes must be uint8");
    TORCH_CHECK(exponents.scalar_type() == at::kChar,
                "tt_blockfloat_hip: exponents must be int8");
    TORCH_CHECK(codes.is_contiguous(), "tt_blockfloat_hip: codes must be contiguous");
    TORCH_CHECK(exponents.is_contiguous(),
                "tt_blockfloat_hip: exponents must be contiguous");
    const int64_t code_bytes = (bits == 4) ? rows * cols / 2 : rows * cols;
    TORCH_CHECK(codes.numel() == code_bytes,
                "tt_blockfloat_hip: codes has ", codes.numel(),
                " bytes, expected ", code_bytes);
    TORCH_CHECK(exponents.numel() == rows * (cols / 16),
                "tt_blockfloat_hip: exponents has ", exponents.numel(),
                " values, expected ", rows * (cols / 16));
}

torch::Tensor dequant_dispatch(torch::Tensor codes, torch::Tensor exponents,
                               int64_t bits, int64_t rows, int64_t cols,
                               int64_t dtype) {
    check_bits(bits);
    check_weight(codes, exponents, bits, rows, cols);
    TORCH_CHECK(codes.is_cuda() && exponents.is_cuda(),
                "tt_blockfloat_hip: weight must be on a CUDA/ROCm device");
    TORCH_CHECK(codes.device() == exponents.device(),
                "tt_blockfloat_hip: codes and exponents must share a device");

    auto options = torch::TensorOptions()
                       .dtype(scalar_type_from_code(dtype))
                       .device(codes.device());
    auto out = torch::empty({rows, cols}, options);
    const c10::cuda::CUDAGuard guard(codes.device());
    check_hip(ttbf_dequant(codes.data_ptr<uint8_t>(),
                           exponents.data_ptr<int8_t>(), static_cast<int>(bits),
                           rows, cols, out.data_ptr(),
                           static_cast<int>(dtype),
                           at::cuda::getCurrentCUDAStream().stream()),
              "ttbf_dequant");
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> quantize_dispatch(torch::Tensor x,
                                                           int64_t bits) {
    check_bits(bits);
    TORCH_CHECK(x.dim() == 2, "tt_blockfloat_hip: weight must be 2-D");
    TORCH_CHECK(x.scalar_type() == at::kFloat,
                "tt_blockfloat_hip: quantize input must be float32");
    const int64_t rows = x.size(0);
    const int64_t cols = x.size(1);
    TORCH_CHECK(cols % 16 == 0,
                "tt_blockfloat_hip: in_features must be a multiple of 16");
    auto xc = x.contiguous();
    const int64_t code_bytes = (bits == 4) ? rows * cols / 2 : rows * cols;
    auto codes = torch::empty(
        {code_bytes},
        torch::TensorOptions().dtype(at::kByte).device(x.device()));
    auto exponents = torch::empty(
        {rows, cols / 16},
        torch::TensorOptions().dtype(at::kChar).device(x.device()));
    const c10::cuda::CUDAGuard guard(x.device());
    check_hip(ttbf_quantize(xc.data_ptr<float>(), static_cast<int>(bits), rows,
                            cols, codes.data_ptr<uint8_t>(),
                            exponents.data_ptr<int8_t>(),
                            at::cuda::getCurrentCUDAStream().stream()),
              "ttbf_quantize");
    return {codes, exponents};
}

torch::Tensor embedding_dispatch(torch::Tensor ids, torch::Tensor codes,
                                 torch::Tensor exponents, int64_t bits,
                                 int64_t rows, int64_t cols, int64_t dtype) {
    check_bits(bits);
    check_weight(codes, exponents, bits, rows, cols);
    TORCH_CHECK(ids.scalar_type() == at::kLong,
                "tt_blockfloat_hip: embedding ids must be int64");
    TORCH_CHECK(ids.is_cuda() && codes.is_cuda(),
                "tt_blockfloat_hip: embedding tensors must be on device");
    TORCH_CHECK(ids.device() == codes.device(),
                "tt_blockfloat_hip: ids and weight must share a device");
    auto idc = ids.contiguous();
    auto shape = idc.sizes().vec();
    shape.push_back(cols);
    auto out = torch::empty(
        shape, torch::TensorOptions()
                   .dtype(scalar_type_from_code(dtype))
                   .device(codes.device()));
    const c10::cuda::CUDAGuard guard(codes.device());
    check_hip(ttbf_embedding(idc.data_ptr<int64_t>(),
                             codes.data_ptr<uint8_t>(),
                             exponents.data_ptr<int8_t>(), static_cast<int>(bits),
                             idc.numel(), rows, cols, out.data_ptr(),
                             static_cast<int>(dtype),
                             at::cuda::getCurrentCUDAStream().stream()),
              "ttbf_embedding");
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> decompose_dispatch(torch::Tensor x,
                                                            int64_t bits) {
    check_bits(bits);
    TORCH_CHECK(x.dim() == 2, "tt_blockfloat_hip: weight must be 2-D");
    TORCH_CHECK(x.scalar_type() == at::kFloat,
                "tt_blockfloat: decompose input must be float32");
    const int64_t rows = x.size(0);
    const int64_t cols = x.size(1);
    TORCH_CHECK(cols % 16 == 0,
                "tt_blockfloat: in_features must be a multiple of 16");
    auto xc = x.contiguous();
    const int64_t code_bytes = (bits == 4) ? rows * cols / 2 : rows * cols;
    auto codes = torch::empty(
        {code_bytes},
        torch::TensorOptions().dtype(at::kByte).device(x.device()));
    auto exponents = torch::empty(
        {rows, cols / 16},
        torch::TensorOptions().dtype(at::kChar).device(x.device()));
    auto failed = torch::zeros(
        {1}, torch::TensorOptions().dtype(at::kInt).device(x.device()));
    const c10::cuda::CUDAGuard guard(x.device());
    check_hip(ttbf_decompose(xc.data_ptr<float>(), static_cast<int>(bits), rows,
                             cols, codes.data_ptr<uint8_t>(),
                             exponents.data_ptr<int8_t>(),
                             failed.data_ptr<int32_t>(),
                             at::cuda::getCurrentCUDAStream().stream()),
              "ttbf_decompose");
    TORCH_CHECK(failed.item<int32_t>() == 0,
                "tt_blockfloat: weight is not exactly on the block-float grid");
    return {codes, exponents};
}

torch::Tensor linear_dispatch(torch::Tensor x, torch::Tensor codes,
                              torch::Tensor exponents, int64_t bits,
                              std::optional<torch::Tensor> bias) {
    check_bits(bits);
    TORCH_CHECK(x.dim() == 2, "tt_blockfloat_hip: activation must be 2-D");
    TORCH_CHECK(x.is_cuda(), "tt_blockfloat_hip: activation must be on device");
    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    TORCH_CHECK(k % 16 == 0,
                "tt_blockfloat: in_features must be a multiple of 16, got ", k);
    const int64_t n = codes.numel() * (bits == 4 ? 2 : 1) / k;
    check_weight(codes, exponents, bits, n, k);
    TORCH_CHECK(codes.device() == x.device(),
                "tt_blockfloat: weight and activation must share a device");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                    x.scalar_type() == at::kBFloat16,
                "tt_blockfloat: unsupported activation dtype ", x.scalar_type());
    auto xc = x.contiguous();
    auto out = torch::empty(
        {m, n}, torch::TensorOptions().dtype(x.scalar_type()).device(x.device()));

    const void* bias_ptr = nullptr;
    torch::Tensor bias_c;
    if (bias.has_value()) {
        TORCH_CHECK(bias->numel() == n, "tt_blockfloat: bias must have ", n,
                    " elements");
        bias_c = bias->to(x.scalar_type()).contiguous();
        TORCH_CHECK(bias_c.device() == x.device(),
                    "tt_blockfloat: bias must share the activation device");
        bias_ptr = bias_c.data_ptr();
    }

    const c10::cuda::CUDAGuard guard(x.device());
    check_hip(ttbf_linear(xc.data_ptr(), dtype_code(x.scalar_type()),
                          codes.data_ptr<uint8_t>(),
                          exponents.data_ptr<int8_t>(), static_cast<int>(bits),
                          m, n, k, bias_ptr, out.data_ptr(),
                          at::cuda::getCurrentCUDAStream().stream()),
              "ttbf_linear");
    return out;
}

bool selftest(const torch::Tensor& ref) {
    // The tensor argument exists only to establish the CUDA dispatch key.
    TORCH_CHECK(ref.is_cuda(), "tt_blockfloat_hip: selftest needs a CUDA tensor");
    auto flag = torch::zeros(
        {1}, torch::TensorOptions().dtype(at::kInt).device(ref.device()));
    const c10::cuda::CUDAGuard guard(ref.device());
    check_hip(ttbf_selftest(flag.data_ptr<int32_t>(),
                            at::cuda::getCurrentCUDAStream().stream()),
              "ttbf_selftest");
    return flag.item<int32_t>() == 0x54544642;
}

}  // namespace

TORCH_LIBRARY(tt_blockfloat_hip, m) {
    m.def(
        "dequant(Tensor codes, Tensor exponents, int bits, int rows, int cols, "
        "int dtype) -> Tensor");
    m.def("quantize(Tensor x, int bits) -> (Tensor codes, Tensor exponents)");
    m.def(
        "embedding(Tensor ids, Tensor codes, Tensor exponents, int bits, "
        "int rows, int cols, int dtype) -> Tensor");
    m.def("decompose(Tensor x, int bits) -> (Tensor codes, Tensor exponents)");
    m.def(
        "linear(Tensor x, Tensor codes, Tensor exponents, int bits, "
        "Tensor? bias=None) -> Tensor");
    m.def("selftest(Tensor ref) -> bool");
}

TORCH_LIBRARY_IMPL(tt_blockfloat_hip, CUDA, m) {
    m.impl("dequant", &dequant_dispatch);
    m.impl("quantize", &quantize_dispatch);
    m.impl("embedding", &embedding_dispatch);
    m.impl("decompose", &decompose_dispatch);
    m.impl("linear", &linear_dispatch);
    m.impl("selftest", &selftest);
}
