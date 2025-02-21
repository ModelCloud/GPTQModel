#include <torch/all.h>

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& b_zeros,
                               torch::Tensor& g_idx, torch::Tensor& perm,
                               torch::Tensor& workspace, int64_t num_bits,
                               int64_t size_m, int64_t size_n, int64_t size_k,
                               bool is_k_full, bool has_zp, bool use_fp32_reduce);
