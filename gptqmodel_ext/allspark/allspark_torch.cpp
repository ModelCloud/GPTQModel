// SPDX-FileCopyrightText: 2025 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Torch bindings for the AllSpark Ampere W8A16 fused GEMM.
// Kernel sources are vendored from vLLM (vllm-project/vllm#12931, Apache-2.0).

#include <torch/extension.h>
#include <torch/library.h>
#include <optional>

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

torch::Tensor allspark_w8a16_gemm_dispatch(
    torch::Tensor a, torch::Tensor b_qweight, torch::Tensor b_scales,
    std::optional<torch::Tensor> b_qzeros, int64_t n, int64_t group_size,
    int64_t sm_count, int64_t sm_version, int64_t cublas_m_threshold,
    bool has_zp, bool n32k16_reorder) {
  return allspark_w8a16_gemm(a, b_qweight, b_scales, b_qzeros, n, group_size,
                             sm_count, sm_version, cublas_m_threshold, has_zp,
                             n32k16_reorder);
}

// The kernel writes into pre-allocated buffers, so the dispatcher hands us
// tensors by value; passing the locals as lvalues keeps the in-place writes.
void rearrange_kn_weight_as_n32k16_order_dispatch(
    torch::Tensor b_qweight, torch::Tensor b_scales,
    std::optional<torch::Tensor> b_zeros, bool has_zp,
    torch::Tensor b_qweight_reorder, torch::Tensor b_scales_reorder,
    std::optional<torch::Tensor> b_zeros_reorder, int64_t k, int64_t n,
    int64_t n_32align) {
  rearrange_kn_weight_as_n32k16_order(
      b_qweight, b_scales, b_zeros, has_zp, b_qweight_reorder,
      b_scales_reorder, b_zeros_reorder, k, n, n_32align);
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
