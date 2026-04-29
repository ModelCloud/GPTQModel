#include <torch/library.h>

#include "awq_marlin_repack.cuh"
#include "core/scalar_type.hpp"
#include "gptq_marlin_repack.cuh"

torch::Tensor gptq_marlin_gemm_fp16(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    vllm::ScalarTypeId const& b_q_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float);

namespace {

torch::Tensor gptq_marlin_gemm_fp16_dispatch(
    torch::Tensor a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor b_q_weight, std::optional<torch::Tensor> const& b_bias_or_none,
    torch::Tensor b_scales,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor workspace,
    int64_t b_q_type_id, int64_t size_m, int64_t size_n, int64_t size_k,
    bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float) {
  return gptq_marlin_gemm_fp16(
      a, c_or_none, b_q_weight, b_bias_or_none, b_scales, global_scale_or_none,
      b_zeros_or_none, g_idx_or_none, perm_or_none, workspace,
      static_cast<vllm::ScalarTypeId>(b_q_type_id), size_m, size_n, size_k,
      is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}

torch::Tensor gptq_marlin_repack_dispatch(torch::Tensor b_q_weight,
                                          torch::Tensor perm, int64_t size_k,
                                          int64_t size_n, int64_t num_bits) {
  return gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits);
}

torch::Tensor awq_marlin_repack_dispatch(torch::Tensor b_q_weight,
                                         int64_t size_k, int64_t size_n,
                                         int64_t num_bits) {
  return awq_marlin_repack(b_q_weight, size_k, size_n, num_bits);
}

}  // namespace

TORCH_LIBRARY(gptqmodel_marlin_fp16, m) {
  m.def(
      "gptq_marlin_gemm_fp16(Tensor a, Tensor? c, Tensor b_q_weight, Tensor? b_bias, Tensor b_scales, "
      "Tensor? global_scale, Tensor? b_zeros, Tensor? g_idx, Tensor? perm, Tensor workspace, int b_q_type_id, "
      "int size_m, int size_n, int size_k, bool is_k_full=True, bool use_atomic_add=False, "
      "bool use_fp32_reduce=False, bool is_zp_float=False) -> Tensor");
  m.def("gptq_marlin_repack(Tensor b_q_weight, Tensor perm, int size_k, int size_n, int num_bits) -> Tensor");
  m.def("awq_marlin_repack(Tensor b_q_weight, int size_k, int size_n, int num_bits) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_marlin_fp16, CUDA, m) {
  m.impl("gptq_marlin_gemm_fp16", &gptq_marlin_gemm_fp16_dispatch);
  m.impl("gptq_marlin_repack", &gptq_marlin_repack_dispatch);
  m.impl("awq_marlin_repack", &awq_marlin_repack_dispatch);
}
