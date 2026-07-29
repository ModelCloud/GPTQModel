// Copyright (C) 2024 Marlin. Adapted for GPT-QModel.
// Native batched/offset Marlin MoE kernel registration and dispatcher.

#include <torch/library.h>
#include <torch/types.h>

#include "../marlin/core/scalar_type.hpp"

torch::Tensor moe_wna16_marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none,
    torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& a_scales_or_none,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none,
    torch::Tensor& workspace, torch::Tensor& sorted_token_ids,
    torch::Tensor& expert_ids, torch::Tensor& num_tokens_past_padded,
    torch::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, vllm::ScalarTypeId const& b_type_id,
    int64_t size_m, int64_t size_n, int64_t size_k, bool is_k_full,
    bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float,
    int64_t thread_k, int64_t thread_n, int64_t blocks_per_sm);

namespace {

torch::Tensor moe_wna16_marlin_gemm_dispatch(
    torch::Tensor a, std::optional<torch::Tensor> c, torch::Tensor b_q_weight,
    std::optional<torch::Tensor> b_bias, torch::Tensor b_scales,
    std::optional<torch::Tensor> a_scales,
    std::optional<torch::Tensor> global_scale,
    std::optional<torch::Tensor> b_zeros,
    std::optional<torch::Tensor> g_idx,
    std::optional<torch::Tensor> perm, torch::Tensor workspace,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_past_padded, torch::Tensor topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    int64_t b_q_type_id, int64_t size_m, int64_t size_n, int64_t size_k,
    bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float, int64_t thread_k, int64_t thread_n,
    int64_t blocks_per_sm) {
  return moe_wna16_marlin_gemm(
      a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros,
      g_idx, perm, workspace, sorted_token_ids, expert_ids,
      num_tokens_past_padded, topk_weights, moe_block_size, top_k,
      mul_topk_weights, static_cast<vllm::ScalarTypeId>(b_q_type_id), size_m,
      size_n, size_k, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float,
      thread_k, thread_n, blocks_per_sm);
}

}  // namespace

TORCH_LIBRARY(gptqmodel_marlin_moe, m) {
  m.def(
      "moe_wna16_marlin_gemm(Tensor a, Tensor? c, Tensor b_q_weight, Tensor? "
      "b_bias, Tensor b_scales, Tensor? a_scales, Tensor? global_scale, "
      "Tensor? b_zeros, Tensor? g_idx, Tensor? perm, Tensor workspace, Tensor "
      "sorted_token_ids, Tensor expert_ids, Tensor num_tokens_past_padded, "
      "Tensor topk_weights, int moe_block_size, int top_k, bool "
      "mul_topk_weights, int b_q_type_id, int size_m, int size_n, int size_k, "
      "bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool "
      "is_zp_float, int thread_k, int thread_n, int blocks_per_sm) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_marlin_moe, CUDA, m) {
  m.impl("moe_wna16_marlin_gemm", &moe_wna16_marlin_gemm_dispatch);
}
