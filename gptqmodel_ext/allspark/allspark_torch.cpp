// SPDX-FileCopyrightText: 2025 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Torch bindings for the AllSpark Ampere W8A16 fused GEMM.
// Kernel sources are vendored from vLLM (vllm-project/vllm#12931, Apache-2.0).

#include <torch/extension.h>
#include <torch/library.h>
#include <optional>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

torch::Tensor allspark_w8a16_gemm(
    torch::Tensor const& a, torch::Tensor const& b_qweight,
    torch::Tensor const& b_scales, c10::optional<torch::Tensor> const& b_qzeros,
    int64_t n, int64_t group_size, int64_t sm_count, int64_t sm_version,
    int64_t cublas_m_threshold, bool has_zp, bool n32k16_reorder);

void rearrange_kn_weight_as_n32k16_order(
    torch::Tensor const& b_qweight, torch::Tensor const& b_scales,
    c10::optional<torch::Tensor> const& b_zeros, bool has_zp,
    torch::Tensor& b_qweight_reorder, torch::Tensor& b_scales_reorder,
    c10::optional<torch::Tensor> const& b_zeros_reorder,
    const int64_t k, const int64_t n, const int64_t n_32align);

namespace {

void check_tensor(const torch::Tensor& tensor, const torch::Tensor& reference,
                  at::ScalarType dtype, int64_t elements, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == reference.device(),
              name, " must be on the same CUDA device");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has incorrect dtype");
  TORCH_CHECK(tensor.numel() == elements, name, " has incorrect shape");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
              name, " must be 16-byte aligned");
}

torch::Tensor allspark_w8a16_gemm_dispatch(
    torch::Tensor a, torch::Tensor b_qweight, torch::Tensor b_scales,
    std::optional<torch::Tensor> b_qzeros, int64_t n, int64_t group_size,
    int64_t sm_count, int64_t sm_version, int64_t cublas_m_threshold,
    bool has_zp, bool n32k16_reorder) {
  TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.is_contiguous(),
              "A must be a contiguous CUDA matrix");
  TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
              "A must be float16 or bfloat16");
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0,
              "A must be 16-byte aligned");
  TORCH_CHECK(n > 0 && n % 16 == 0 && a.size(1) > 0 && a.size(1) % 16 == 0,
              "N and K must be positive multiples of 16");
  TORCH_CHECK(group_size == -1 && n32k16_reorder,
              "AllSpark requires group_size=-1 and N32K16 weights");
  TORCH_CHECK(sm_count > 0 && sm_version >= 80 && sm_version <= 120 && cublas_m_threshold >= 0,
              "Invalid AllSpark launch configuration");
  const int64_t padded_n = (n + 31) / 32 * 32;
  TORCH_CHECK(b_qweight.dim() == 2, "b_qweight must be a matrix");
  check_tensor(b_qweight, a, at::kByte, padded_n * a.size(1), "b_qweight");
  check_tensor(b_scales, a, a.scalar_type(), padded_n, "b_scales");
  TORCH_CHECK(has_zp == b_qzeros.has_value(), "has_zp must match b_qzeros");
  if (has_zp) check_tensor(*b_qzeros, a, a.scalar_type(), padded_n, "b_qzeros");
  if (a.size(0) == 0) return torch::empty({0, n}, a.options());
  auto result = allspark_w8a16_gemm(a, b_qweight, b_scales, b_qzeros, n, group_size,
                             sm_count, sm_version, cublas_m_threshold, has_zp,
                             n32k16_reorder);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

// The kernel writes into pre-allocated buffers, so the dispatcher hands us
// tensors by value; passing the locals as lvalues keeps the in-place writes.
void rearrange_kn_weight_as_n32k16_order_dispatch(
    torch::Tensor b_qweight, torch::Tensor b_scales,
    std::optional<torch::Tensor> b_zeros, bool has_zp,
    torch::Tensor b_qweight_reorder, torch::Tensor b_scales_reorder,
    std::optional<torch::Tensor> b_zeros_reorder, int64_t k, int64_t n,
    int64_t n_32align) {
  TORCH_CHECK(b_qweight.is_cuda(), "b_qweight must be on CUDA");
  const c10::cuda::CUDAGuard guard(b_qweight.device());
  TORCH_CHECK(k > 0 && n > 0 && k % 16 == 0 && n % 16 == 0 &&
              n_32align == (n + 31) / 32 * 32,
              "N and K must be positive multiples of 16 with N rounded to 32");
  TORCH_CHECK(b_scales.scalar_type() == at::kHalf || b_scales.scalar_type() == at::kBFloat16,
              "scales must be float16 or bfloat16");
  check_tensor(b_qweight, b_qweight, at::kByte, k * n, "b_qweight");
  check_tensor(b_scales, b_qweight, b_scales.scalar_type(), n, "b_scales");
  check_tensor(b_qweight_reorder, b_qweight, at::kByte, k * n_32align, "b_qweight_reorder");
  check_tensor(b_scales_reorder, b_qweight, b_scales.scalar_type(), n_32align, "b_scales_reorder");
  TORCH_CHECK(has_zp == b_zeros.has_value() && has_zp == b_zeros_reorder.has_value(),
              "has_zp must match zero buffers");
  if (has_zp) {
    check_tensor(*b_zeros, b_qweight, b_scales.scalar_type(), n, "b_zeros");
    check_tensor(*b_zeros_reorder, b_qweight, b_scales.scalar_type(), n_32align, "b_zeros_reorder");
  }
  rearrange_kn_weight_as_n32k16_order(
      b_qweight, b_scales, b_zeros, has_zp, b_qweight_reorder,
      b_scales_reorder, b_zeros_reorder, k, n, n_32align);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(gptqmodel_allspark, m) {
  m.def(
      "allspark_w8a16_gemm(Tensor a, Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_qzeros, int n, int group_size, int sm_count, int sm_version, "
      "int cublas_m_threshold, bool has_zp, bool n32k16_reorder) -> Tensor");
  m.def(
      "rearrange_kn_weight_as_n32k16_order(Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_zeros, bool has_zp, Tensor(a!) b_qweight_reorder, "
      "Tensor(b!) b_scales_reorder, Tensor(c!)? b_zeros_reorder, int k, int n, "
      "int n_32align) -> ()");
}

TORCH_LIBRARY_IMPL(gptqmodel_allspark, CUDA, m) {
  m.impl("allspark_w8a16_gemm", &allspark_w8a16_gemm_dispatch);
  m.impl("rearrange_kn_weight_as_n32k16_order",
         &rearrange_kn_weight_as_n32k16_order_dispatch);
}
