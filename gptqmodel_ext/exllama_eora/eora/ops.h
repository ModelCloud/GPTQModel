#pragma once

#include "torch/library.h"
#include <torch/script.h> // One-stop header.

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit);

torch::Tensor gptq_gemm_eora(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit,
                        torch::Tensor eora_ax, torch::Tensor eora_b);

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit);