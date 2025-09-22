#include <torch/extension.h>

torch::Tensor awq_marlin_repack(torch::Tensor& b_q_weight, int64_t size_k,
                                int64_t size_n, int64_t num_bits);