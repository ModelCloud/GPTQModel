// SPDX-FileCopyrightText: 2025 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Torch bindings for the AllSpark HIP/RDNA4 W8A16 fused GEMM.
//
// Host-only translation unit: the ROCm pip SDK has no thrust headers, so the
// .cu files must not see torch/ATen. Note we deliberately avoid
// <ATen/cuda/CUDAContext.h> (it pulls hipsparse, which is not installed in the
// runtime wheel) and use c10/cuda/CUDAStream.h instead.

#include <torch/extension.h>
#include <torch/library.h>

#include <c10/cuda/CUDAStream.h>

#include <optional>

// ---------------------------------------------------------------------------
// Device-side entry points (allspark_hip.cu). No torch types cross this line.
// ---------------------------------------------------------------------------
extern "C" {

size_t allspark_hip_workspace_bytes(int M, int N, int splitk);
int allspark_hip_pick_splitk(int M, int N, int N32, int K, int sm_count);

void allspark_hip_launch(const void* A, const void* Bq, const void* Bs,
                         const void* Bz, void* C, void* ws, int M, int N,
                         int N32, int K, int splitk, int dtype, int out_fp32,
                         void* stream, uint32_t* ran);

void allspark_hip_launch_repack(const void* B, const void* Bs, const void* Bz,
                                void* Bq_out, void* Bs_out, void* Bz_out, int K,
                                int N, int N32, int dtype, void* stream);

void allspark_hip_launch_unpack(const void* Bq, const void* Bs, const void* Bz,
                                void* B, int K, int N, int N32, int dtype,
                                void* stream);
}

namespace {

inline int dtype_code(const torch::Tensor& t) {
  if (t.scalar_type() == at::ScalarType::Half) return 0;
  if (t.scalar_type() == at::ScalarType::BFloat16) return 1;
  TORCH_CHECK(false, "allspark_hip: activation/scale dtype must be fp16 or bf16, got ",
              t.scalar_type());
}

inline hipStream_t current_stream() {
  return reinterpret_cast<hipStream_t>(c10::cuda::getCurrentCUDAStream().stream());
}

torch::Tensor allspark_w8a16_gemm_dispatch(
    torch::Tensor a, torch::Tensor b_qweight, torch::Tensor b_scales,
    std::optional<torch::Tensor> b_qzeros, int64_t n, int64_t group_size,
    int64_t sm_count, int64_t sm_version, int64_t cublas_m_threshold,
    bool has_zp, bool n32k16_reorder) {
  TORCH_CHECK(a.is_cuda() && a.is_contiguous(), "a must be a contiguous CUDA tensor");
  TORCH_CHECK(b_qweight.is_cuda() && b_qweight.is_contiguous(),
              "b_qweight must be contiguous");
  TORCH_CHECK(b_scales.is_cuda() && b_scales.is_contiguous(),
              "b_scales must be contiguous");
  TORCH_CHECK(group_size == -1, "allspark_hip supports group_size=-1 only");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "activation and scale dtypes must match");

  const int64_t K = a.size(-1);
  const int64_t M = a.numel() / K;
  const int64_t N = n;
  TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16");
  TORCH_CHECK(N % 16 == 0, "N must be a multiple of 16");
  TORCH_CHECK(M <= 16,
              "allspark_hip native kernel covers M<=16; use the dequant fallback");

  // b_qweight holds N32*K uint8 either as (N32/4, K*4) or as the (N32, K) view;
  // derive N32 from the element count so both layouts work.
  TORCH_CHECK(b_qweight.scalar_type() == at::ScalarType::Byte,
              "b_qweight must be uint8");
  const int64_t N32 = b_qweight.numel() / K;
  TORCH_CHECK(N32 % 32 == 0 && N32 >= N, "packed weight has an invalid N padding");

  auto c = torch::empty({M, N}, a.options());

  int splitk = allspark_hip_pick_splitk(static_cast<int>(M), static_cast<int>(N),
                                        static_cast<int>(N32), static_cast<int>(K),
                                        static_cast<int>(sm_count));
  torch::Tensor ws;
  void* ws_ptr = nullptr;
  const size_t ws_bytes = allspark_hip_workspace_bytes(
      static_cast<int>(M), static_cast<int>(N), splitk);
  if (ws_bytes > 0) {
    ws = torch::empty({static_cast<int64_t>(ws_bytes / sizeof(float))},
                      a.options().dtype(torch::kFloat32));
    ws_ptr = ws.data_ptr();
  }

  const void* bz = nullptr;
  if (has_zp && b_qzeros.has_value()) {
    TORCH_CHECK(b_qzeros->is_contiguous() && b_qzeros->is_cuda(),
                "b_qzeros must be a contiguous CUDA tensor");
    bz = b_qzeros->data_ptr();
  }

  allspark_hip_launch(a.data_ptr(), b_qweight.data_ptr(), b_scales.data_ptr(), bz,
                      c.data_ptr(), ws_ptr, static_cast<int>(M),
                      static_cast<int>(N), static_cast<int>(N32),
                      static_cast<int>(K), splitk, dtype_code(a), 0,
                      current_stream(), nullptr);
  return c;
}

// Same GEMM, but the fp32 accumulators are written out directly. Used by the
// numerics tests to separate kernel arithmetic error from mandatory output
// rounding (mirrors the repo's `raw` fp32 comparison in the ROCm kernel tests).
torch::Tensor allspark_hip_gemm_f32_dispatch(torch::Tensor a,
                                             torch::Tensor b_qweight,
                                             torch::Tensor b_scales, int64_t n,
                                             int64_t sm_count) {
  const int64_t K = a.size(-1);
  const int64_t M = a.numel() / K;
  const int64_t N = n;
  TORCH_CHECK(M <= 16, "fp32 output path supports M<=16 only");
  const int64_t N32 = b_qweight.numel() / K;
  auto c = torch::empty({M, N}, a.options().dtype(torch::kFloat32));
  const int sm = static_cast<int>(sm_count);
  int splitk = allspark_hip_pick_splitk(static_cast<int>(M), static_cast<int>(N),
                                        static_cast<int>(N32), static_cast<int>(K), sm);
  torch::Tensor ws;
  void* ws_ptr = nullptr;
  const size_t ws_bytes = allspark_hip_workspace_bytes(
      static_cast<int>(M), static_cast<int>(N), splitk);
  if (ws_bytes > 0) {
    ws = torch::empty({static_cast<int64_t>(ws_bytes / sizeof(float))},
                      a.options().dtype(torch::kFloat32));
    ws_ptr = ws.data_ptr();
  }
  allspark_hip_launch(a.data_ptr(), b_qweight.data_ptr(), b_scales.data_ptr(),
                      nullptr, c.data_ptr(), ws_ptr, static_cast<int>(M),
                      static_cast<int>(N), static_cast<int>(N32),
                      static_cast<int>(K), splitk, dtype_code(a), 1,
                      current_stream(), nullptr);
  return c;
}

void rearrange_kn_weight_as_n32k16_order_dispatch(
    torch::Tensor b_qweight, torch::Tensor b_scales,
    std::optional<torch::Tensor> b_zeros, bool has_zp,
    torch::Tensor b_qweight_reorder, torch::Tensor b_scales_reorder,
    std::optional<torch::Tensor> b_zeros_reorder, int64_t k, int64_t n,
    int64_t n_32align) {
  TORCH_CHECK(b_qweight.is_cuda() && b_qweight.is_contiguous(),
              "b_qweight must be contiguous");
  TORCH_CHECK(b_qweight.scalar_type() == at::ScalarType::Byte,
              "b_qweight must be uint8 codes in (K, N) order");
  TORCH_CHECK(b_scales.is_cuda() && b_scales.is_contiguous(),
              "b_scales must be contiguous");
  TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16");
  TORCH_CHECK(n_32align % 32 == 0 && n_32align >= n, "invalid n_32align");

  const void* bz = nullptr;
  void* bz_out = nullptr;
  if (has_zp && b_zeros.has_value() && b_zeros_reorder.has_value()) {
    bz = b_zeros->data_ptr();
    bz_out = b_zeros_reorder->data_ptr();
  }

  allspark_hip_launch_repack(
      b_qweight.data_ptr(), b_scales.data_ptr(), bz,
      b_qweight_reorder.data_ptr(), b_scales_reorder.data_ptr(), bz_out,
      static_cast<int>(k), static_cast<int>(n), static_cast<int>(n_32align),
      dtype_code(b_scales), current_stream());
}

// Fragment-order packed weights -> dense (K, N). Only used by the M > 16
// fallback; the result is a per-call temporary and is never cached.
torch::Tensor allspark_hip_unpack_dispatch(torch::Tensor b_qweight,
                                           torch::Tensor b_scales, int64_t n,
                                           int64_t k) {
  const int64_t N32 = b_qweight.numel() / k;
  auto out = torch::empty({k, n}, b_scales.options());
  allspark_hip_launch_unpack(b_qweight.data_ptr(), b_scales.data_ptr(), nullptr,
                             out.data_ptr(), static_cast<int>(k),
                             static_cast<int>(n), static_cast<int>(N32),
                             dtype_code(b_scales), current_stream());
  return out;
}

// One-shot proof that the device code actually executed. A silently empty
// kernel body (the `#if __CUDA_ARCH__` failure mode) leaves the sentinel
// untouched and this returns false.
bool allspark_hip_selftest(torch::Tensor ref) {
  auto opts = torch::TensorOptions().dtype(torch::kHalf).device(ref.device());
  constexpr int M = 1, N = 32, K = 16;
  auto a = torch::zeros({M, K}, opts);
  auto bq = torch::zeros({(N / 4), K * 4}, opts.dtype(torch::kByte));
  auto bs = torch::ones({N}, opts);
  auto c = torch::empty({M, N}, opts);
  auto ran = torch::zeros({1}, opts.dtype(torch::kInt32));

  allspark_hip_launch(a.data_ptr(), bq.data_ptr(), bs.data_ptr(), nullptr,
                      c.data_ptr(), nullptr, M, N, N, K, 1, 0, 0,
                      current_stream(),
                      reinterpret_cast<uint32_t*>(ran.data_ptr()));
  auto host = ran.cpu();
  const uint32_t got = static_cast<uint32_t>(host.data_ptr<int32_t>()[0]);
  return got == 0x41535350u;
}

}  // namespace

TORCH_LIBRARY(gptqmodel_allspark_hip, m) {
  m.def(
      "allspark_w8a16_gemm(Tensor a, Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_qzeros, int n, int group_size, int sm_count, int sm_version, "
      "int cublas_m_threshold, bool has_zp, bool n32k16_reorder) -> Tensor");
  m.def(
      "rearrange_kn_weight_as_n32k16_order(Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_zeros, bool has_zp, Tensor(a!) b_qweight_reorder, "
      "Tensor(b!) b_scales_reorder, Tensor(c!)? b_zeros_reorder, int k, int n, "
      "int n_32align) -> ()");
  m.def("allspark_hip_unpack(Tensor b_qweight, Tensor b_scales, int n, int k) -> Tensor");
  m.def("allspark_hip_gemm_f32(Tensor a, Tensor b_qweight, Tensor b_scales, int n, int sm_count) -> Tensor");
  m.def("allspark_hip_selftest(Tensor ref) -> bool");
}

TORCH_LIBRARY_IMPL(gptqmodel_allspark_hip, CUDA, m) {
  m.impl("allspark_w8a16_gemm", &allspark_w8a16_gemm_dispatch);
  m.impl("rearrange_kn_weight_as_n32k16_order",
         &rearrange_kn_weight_as_n32k16_order_dispatch);
  m.impl("allspark_hip_unpack", &allspark_hip_unpack_dispatch);
  m.impl("allspark_hip_gemm_f32", &allspark_hip_gemm_f32_dispatch);
  m.impl("allspark_hip_selftest", &allspark_hip_selftest);
}
