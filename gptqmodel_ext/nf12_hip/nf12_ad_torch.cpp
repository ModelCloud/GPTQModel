// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// Torch bindings for the NF12-AD HIP port (AMD RDNA4: gfx1200/gfx1201/gfx1250).
//
// Algorithm: "NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit
// Weight Compression", Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).
// Together AI: https://www.together.ai/
//
// Original NF12 reference implementation and CUDA kernels:
//   https://github.com/dphnAI/nf12
//   Copyright (c) 2026 AlpinDale and the nf12 contributors.
//   Licensed under the Apache License, Version 2.0.
//
// Written for GPT-QModel (ModelCloud.ai). Registers `torch.ops.nf12_ad_hip`
// with the same decode/linear contract as the CUDA `nf12` namespace so the
// Python layer routes by device family without changing callers.
//
// Host-only translation unit: the ROCm SDK here ships no thrust, so the .hip
// files must not see torch/ATen. Mirrors allspark_torch_hip.cpp.
#include <torch/extension.h>
#include <torch/library.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>

#include <hip/hip_runtime_api.h>

#include <optional>
#include <string>

// ---------------------------------------------------------------------------
// Device-side entry points (nf12_ad_hip.hip). No torch types cross this line.
// ---------------------------------------------------------------------------
extern "C" {

// `out` is deliberately `void*`: the .hip translation unit owns the bf16 type
// (__hip_bfloat16) and hipify would mangle any HIP bf16 spelling used here.
hipError_t nf12_ad_decode(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                          int64_t n, int64_t k, void* out, int64_t ldo,
                          hipStream_t stream);

hipError_t nf12_ad_selftest(uint32_t* out, hipStream_t stream);
}

namespace {

constexpr int64_t kMaxExpBase = 120;

inline hipStream_t current_stream() {
  return reinterpret_cast<hipStream_t>(c10::cuda::getCurrentCUDAStream().stream());
}

inline c10::cuda::CUDAGuard device_guard(const torch::Tensor& t) {
  return c10::cuda::CUDAGuard(t.device());
}

inline std::string arch_name(int device) {
  hipDeviceProp_t props;
  TORCH_CHECK(hipGetDeviceProperties(&props, device) == hipSuccess,
              "nf12_ad_hip: could not query device properties");
  const std::string full(props.gcnArchName);
  return full.substr(0, full.find(':'));
}

inline std::string check_arch(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda(), "nf12_ad_hip requires a CUDA/HIP tensor");
  const std::string arch = arch_name(t.get_device());
  TORCH_CHECK(arch == "gfx1200" || arch == "gfx1201" || arch == "gfx1250",
              "nf12_ad_hip requires RDNA4 (gfx1200/gfx1201/gfx1250), got ", arch);
  return arch;
}

// ---------------------------------------------------------------------------
// nf12_ad_hip::decode(packed, escapes, exp_base) -> bf16 [N, K]
// ---------------------------------------------------------------------------
torch::Tensor decode_dispatch(const torch::Tensor& packed, const torch::Tensor& escapes,
                              int64_t exp_base) {
  check_arch(packed);
  TORCH_CHECK(packed.dim() == 2 && packed.scalar_type() == at::kByte,
              "nf12_ad_hip.decode: packed must be a 2-D uint8 tensor");
  TORCH_CHECK(packed.is_contiguous(), "nf12_ad_hip.decode: packed must be contiguous");
  TORCH_CHECK(escapes.scalar_type() == at::kShort && escapes.dim() == 2 &&
                  escapes.size(1) == 8,
              "nf12_ad_hip.decode: escapes must be int16 [E, 8]");
  TORCH_CHECK(escapes.is_contiguous(), "nf12_ad_hip.decode: escapes must be contiguous");
  TORCH_CHECK(packed.device() == escapes.device(),
              "nf12_ad_hip.decode: packed and escapes must share a device");
  TORCH_CHECK(exp_base >= 0 && exp_base <= kMaxExpBase,
              "nf12_ad_hip.decode: exp_base must be in [0, 120], got ", exp_base);

  const int64_t n = packed.size(0);
  const int64_t row_bytes = packed.size(1);
  TORCH_CHECK(n > 0 && n % 16 == 0,
              "nf12_ad_hip.decode: N must be a positive multiple of 16");
  TORCH_CHECK(row_bytes % 96 == 0,
              "nf12_ad_hip.decode: packed row must be a multiple of 96 bytes");
  const int64_t k = row_bytes * 2 / 3;
  TORCH_CHECK(k % 64 == 0, "nf12_ad_hip.decode: K must be a multiple of 64");

  auto guard = device_guard(packed);
  auto out = torch::empty({n, k}, packed.options().dtype(at::kBFloat16));

  const hipError_t status = nf12_ad_decode(
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<const uint4*>(escapes.data_ptr<int16_t>()),
      static_cast<uint32_t>(exp_base), n, k,
      static_cast<void*>(out.data_ptr()), k, current_stream());
  TORCH_CHECK(status == hipSuccess, "nf12_ad_hip.decode failed: ", hipGetErrorString(status));
  return out;
}

// ---------------------------------------------------------------------------
// nf12_ad_hip::linear(x, packed, escapes, exp_base, bias) -> bf16
// ---------------------------------------------------------------------------
// SPEC.md 6.5 decode-then-GEMM: decode once into a caching-allocator buffer,
// then one BF16 GEMM. The buffer is a local temporary owned by this call; no
// dense weight is cached on the module (AGENTS.md quantized-storage rule).
torch::Tensor linear_dispatch(const torch::Tensor& x, const torch::Tensor& packed,
                              const torch::Tensor& escapes, int64_t exp_base,
                              const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16,
              "nf12_ad_hip.linear: x must be a bf16 CUDA tensor");
  const int64_t k = x.size(-1);
  const torch::Tensor w = decode_dispatch(packed, escapes, exp_base);
  TORCH_CHECK(w.size(1) == k, "nf12_ad_hip.linear: x's last dim (", k,
              ") must equal K (", w.size(1), ")");

  if (bias.has_value() && bias->defined()) {
    // Fold the bias into the GEMM epilogue (addmm), never add it to an
    // already-rounded BF16 product. Adding afterwards double-rounds: the
    // matmul rounds H, then H + bias rounds again, which measured 0.0017
    // RMS relative error and only 80% bit-exact outputs on gfx1201 versus
    // 2e-5 / 99.99% for addmm. The CUDA kernel also accumulates bias in
    // FP32 before writing BF16 (csrc/linear/small_m.cu), so addmm matches it.
    const torch::Tensor b = bias->to(w.device(), w.scalar_type());
    return at::addmm(b, x, w.t());
  }
  return at::matmul(x, w.t());
}

// ---------------------------------------------------------------------------
// nf12_ad_hip::selftest(ref) -> bool
// ---------------------------------------------------------------------------
bool selftest(const torch::Tensor& ref) {
  check_arch(ref);
  auto guard = device_guard(ref);
  auto sentinel = torch::zeros({1}, ref.options().dtype(at::kInt));
  const hipError_t status = nf12_ad_selftest(
      reinterpret_cast<uint32_t*>(sentinel.data_ptr<int32_t>()), current_stream());
  TORCH_CHECK(status == hipSuccess, "nf12_ad_hip.selftest failed: ", hipGetErrorString(status));
  return sentinel.cpu().data_ptr<int32_t>()[0] == 0x4E463132;
}

// ---------------------------------------------------------------------------
// nf12_ad_hip::arch_info() -> str
// ---------------------------------------------------------------------------
std::string arch_info() {
  TORCH_CHECK(torch::cuda::is_available(), "nf12_ad_hip.arch_info: no GPU");
  return arch_name(c10::cuda::current_device());
}

}  // namespace

TORCH_LIBRARY(nf12_ad_hip, m) {
  m.def("decode(Tensor packed, Tensor escapes, int exp_base) -> Tensor");
  m.def(
      "linear(Tensor x, Tensor packed, Tensor escapes, int exp_base, "
      "Tensor? bias=None) -> Tensor");
  m.def("selftest(Tensor ref) -> bool");
  m.def("arch_info() -> str");
}

TORCH_LIBRARY_IMPL(nf12_ad_hip, CUDA, m) {
  m.impl("decode", &decode_dispatch);
  m.impl("linear", &linear_dispatch);
  m.impl("selftest", &selftest);
}

// arch_info() takes no tensor arguments, so it cannot dispatch on a key set
// derived from inputs. CompositeExplicitAutograd makes it callable directly.
TORCH_LIBRARY_IMPL(nf12_ad_hip, CompositeExplicitAutograd, m) {
  m.impl("arch_info", &arch_info);
}
