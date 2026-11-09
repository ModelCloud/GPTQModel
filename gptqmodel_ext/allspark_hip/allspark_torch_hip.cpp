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
#include <c10/cuda/CUDAGuard.h>

#include <optional>
#include <limits>
#include <string>
#include <hip/hip_runtime_api.h>

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

// Custom ops do not get a device guard from the dispatcher. Without one the
// launcher runs on whatever device happens to be current and reads/writes
// another device's pointers, which shows up as NaN on multi-GPU splits.
inline c10::cuda::CUDAGuard device_guard(const torch::Tensor& t) {
  return c10::cuda::CUDAGuard(t.device());
}

// Validate every public operator before allocation or pointer reinterpretation.
inline void check_tensor(const torch::Tensor& t, const torch::Tensor& ref,
                         at::ScalarType dtype, int64_t count, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == ref.device(), name,
              " must be on the same CUDA device as the input");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(t.numel() == count, name, " has an invalid element count");
}

inline void check_arch(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda(), "allspark_hip requires a CUDA tensor");
  hipDeviceProp_t props;
  TORCH_CHECK(hipGetDeviceProperties(&props, t.get_device()) == hipSuccess,
              "allspark_hip could not query device properties");
  const std::string arch = std::string(props.gcnArchName).substr(
      0, std::string(props.gcnArchName).find(':'));
  TORCH_CHECK(arch == "gfx1200" || arch == "gfx1201",
              "allspark_hip requires gfx1200/gfx1201, got ", arch);
}

inline void check_launch() {
  const auto status = hipGetLastError();
  TORCH_CHECK(status == hipSuccess, "allspark_hip launch failed: ", hipGetErrorString(status));
}

inline void check_dimensions(int64_t k, int64_t n) {
  TORCH_CHECK(k > 0 && k % 16 == 0, "K must be positive and a multiple of 16");
  TORCH_CHECK(n > 0 && n % 16 == 0, "N must be positive and a multiple of 16");
  // Launch arguments and reduction indexing use signed 32-bit integers.
  TORCH_CHECK(k <= std::numeric_limits<int>::max() - 16 &&
              n <= std::numeric_limits<int>::max() / 128,
              "allspark_hip dimensions exceed the kernel index range");
}

inline int64_t check_packed(const torch::Tensor& bq, const torch::Tensor& bs,
                           const std::optional<torch::Tensor>& bz,
                           int64_t k, int64_t n) {
  check_dimensions(k, n);
  check_arch(bq);
  TORCH_CHECK(bq.dim() == 2 && bq.numel() % k == 0,
              "packed weight must be a matrix with an exact multiple of K elements");
  const int64_t n32 = bq.numel() / k;
  TORCH_CHECK(n32 >= n && n32 % 128 == 0 &&
              n32 <= std::numeric_limits<int>::max() - 128,
              "packed weight padding must cover N in complete 128-column tiles");
  check_tensor(bq, bq, at::ScalarType::Byte, n32 * k, "b_qweight");
  dtype_code(bs);
  check_tensor(bs, bq, bs.scalar_type(), n32, "b_scales");
  if (bz.has_value()) check_tensor(*bz, bq, bs.scalar_type(), n32, "b_qzeros");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(bq.data_ptr()) % 16 == 0,
              "packed weight must be 16-byte aligned");
  return n32;
}

inline void check_activation(const torch::Tensor& a, const torch::Tensor& bs) {
  TORCH_CHECK(a.dim() == 2, "a must be a matrix");
  TORCH_CHECK(a.size(0) <= 16, "native kernel supports M<=16 only");
  check_tensor(a, bs, bs.scalar_type(), a.numel(), "a");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0,
              "a must be 16-byte aligned");
}

torch::Tensor allspark_w8a16_gemm_dispatch(
    torch::Tensor a, torch::Tensor b_qweight, torch::Tensor b_scales,
    std::optional<torch::Tensor> b_qzeros, int64_t n, int64_t group_size,
    int64_t sm_count, int64_t sm_version, int64_t cublas_m_threshold,
    bool has_zp, bool n32k16_reorder) {
  TORCH_CHECK(a.dim() == 2, "a must be a matrix");
  const int64_t K = a.size(1), M = a.size(0), N = n;
  TORCH_CHECK(group_size == -1 || group_size == K,
              "allspark_hip group_size must be -1 or K (one full-K group)");
  TORCH_CHECK(n32k16_reorder, "allspark_hip requires reordered weights");
  TORCH_CHECK(has_zp == b_qzeros.has_value(), "has_zp must match b_qzeros presence");
  const int64_t N32 = check_packed(b_qweight, b_scales, b_qzeros, K, N);
  check_activation(a, b_scales);
  const c10::cuda::CUDAGuard device_guard_ = device_guard(a);
  auto c = torch::empty({M, N}, a.options());
  if (M == 0) return c;

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
  check_launch();
  return c;
}

// Same GEMM, but the fp32 accumulators are written out directly. Used by the
// numerics tests to separate kernel arithmetic error from mandatory output
// rounding (mirrors the repo's `raw` fp32 comparison in the ROCm kernel tests).
torch::Tensor allspark_hip_gemm_f32_dispatch(torch::Tensor a,
                                             torch::Tensor b_qweight,
                                             torch::Tensor b_scales, int64_t n,
                                             int64_t sm_count,
                                             std::optional<torch::Tensor> b_qzeros) {
  TORCH_CHECK(a.dim() == 2, "a must be a matrix");
  const int64_t K = a.size(1), M = a.size(0), N = n;
  const int64_t N32 = check_packed(b_qweight, b_scales, b_qzeros, K, N);
  check_activation(a, b_scales);
  const c10::cuda::CUDAGuard device_guard_ = device_guard(a);
  auto c = torch::empty({M, N}, a.options().dtype(torch::kFloat32));
  if (M == 0) return c;
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
  const void* bz = nullptr;
  if (b_qzeros.has_value()) {
    TORCH_CHECK(b_qzeros->is_contiguous() && b_qzeros->is_cuda(),
                "b_qzeros must be a contiguous CUDA tensor");
    bz = b_qzeros->data_ptr();
  }
  allspark_hip_launch(a.data_ptr(), b_qweight.data_ptr(), b_scales.data_ptr(),
                      bz, c.data_ptr(), ws_ptr, static_cast<int>(M),
                      static_cast<int>(N), static_cast<int>(N32),
                      static_cast<int>(K), splitk, dtype_code(a), 1,
                      current_stream(), nullptr);
  check_launch();
  return c;
}

void rearrange_kn_weight_as_n32k16_order_dispatch(
    torch::Tensor b_qweight, torch::Tensor b_scales,
    std::optional<torch::Tensor> b_zeros, bool has_zp,
    torch::Tensor b_qweight_reorder, torch::Tensor b_scales_reorder,
    std::optional<torch::Tensor> b_zeros_reorder, int64_t k, int64_t n,
    int64_t n_32align) {
  check_dimensions(k, n);
  check_arch(b_qweight);
  TORCH_CHECK(n_32align >= n && n_32align % 128 == 0 &&
              n_32align <= std::numeric_limits<int>::max() - 128,
              "n_32align must cover N in complete 128-column tiles");
  TORCH_CHECK(b_qweight.dim() == 2 && b_qweight.size(0) == k && b_qweight.size(1) == n,
              "b_qweight must have shape (K, N)");
  check_tensor(b_qweight, b_qweight, at::ScalarType::Byte, k * n, "b_qweight");
  dtype_code(b_scales);
  check_tensor(b_scales, b_qweight, b_scales.scalar_type(), n, "b_scales");
  check_tensor(b_qweight_reorder, b_qweight, at::ScalarType::Byte,
               n_32align * k, "b_qweight_reorder");
  check_tensor(b_scales_reorder, b_qweight, b_scales.scalar_type(),
               n_32align, "b_scales_reorder");
  TORCH_CHECK(has_zp == b_zeros.has_value() && has_zp == b_zeros_reorder.has_value(),
              "has_zp must match input and output zero presence");
  const void* bz = nullptr;
  void* bz_out = nullptr;
  if (has_zp) {
    check_tensor(*b_zeros, b_qweight, b_scales.scalar_type(), n, "b_zeros");
    check_tensor(*b_zeros_reorder, b_qweight, b_scales.scalar_type(),
                 n_32align, "b_zeros_reorder");
    bz = b_zeros->data_ptr();
    bz_out = b_zeros_reorder->data_ptr();
  }
  const c10::cuda::CUDAGuard device_guard_ = device_guard(b_qweight);
  allspark_hip_launch_repack(
      b_qweight.data_ptr(), b_scales.data_ptr(), bz,
      b_qweight_reorder.data_ptr(), b_scales_reorder.data_ptr(), bz_out,
      static_cast<int>(k), static_cast<int>(n), static_cast<int>(n_32align),
      dtype_code(b_scales), current_stream());
  check_launch();
}

// Fragment-order packed weights -> dense (K, N). Only used by the M > 16
// fallback; the result is a per-call temporary and is never cached.
torch::Tensor allspark_hip_unpack_dispatch(torch::Tensor b_qweight,
                                           torch::Tensor b_scales, int64_t n,
                                           int64_t k,
                                           std::optional<torch::Tensor> b_qzeros) {
  check_packed(b_qweight, b_scales, b_qzeros, k, n);
  const c10::cuda::CUDAGuard device_guard_ = device_guard(b_qweight);
  const int64_t N32 = b_qweight.numel() / k;
  auto out = torch::empty({k, n}, b_scales.options());
  const void* bz = nullptr;
  if (b_qzeros.has_value()) {
    bz = b_qzeros->data_ptr();
  }
  allspark_hip_launch_unpack(b_qweight.data_ptr(), b_scales.data_ptr(), bz,
                             out.data_ptr(), static_cast<int>(k),
                             static_cast<int>(n), static_cast<int>(N32),
                             dtype_code(b_scales), current_stream());
  check_launch();
  return out;
}

// One-shot proof that the device code actually executed. A silently empty
// kernel body (the `#if __CUDA_ARCH__` failure mode) leaves the sentinel
// untouched and this returns false.
bool allspark_hip_selftest(torch::Tensor ref) {
  check_arch(ref);
  const c10::cuda::CUDAGuard device_guard_ = device_guard(ref);
  auto opts = torch::TensorOptions().dtype(torch::kHalf).device(ref.device());
  constexpr int M = 1, N = 128, K = 16;
  auto a = torch::randn({M, K}, opts);
  auto bq = torch::randint(1, 256, {(N / 4), K * 4}, opts.dtype(torch::kByte));
  auto bs = torch::ones({N}, opts);
  auto c = torch::empty({M, N}, opts);
  auto ran = torch::zeros({1}, opts.dtype(torch::kInt32));

  allspark_hip_launch(a.data_ptr(), bq.data_ptr(), bs.data_ptr(), nullptr,
                      c.data_ptr(), nullptr, M, N, N, K, 1, 0, 0,
                      current_stream(),
                      reinterpret_cast<uint32_t*>(ran.data_ptr()));
  check_launch();
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
  m.def(
      "allspark_hip_unpack(Tensor b_qweight, Tensor b_scales, int n, int k, "
      "Tensor? b_qzeros) -> Tensor");
  m.def(
      "allspark_hip_gemm_f32(Tensor a, Tensor b_qweight, Tensor b_scales, int n, "
      "int sm_count, Tensor? b_qzeros) -> Tensor");
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
