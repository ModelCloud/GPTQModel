// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <torch/library.h>
#include <vector>

torch::Tensor gemm_int4_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &zeros,
    const torch::Tensor &absmax,
    int64_t blocksize);

std::vector<torch::Tensor>
pack_gptq_int4_cpu_forward(
    const torch::Tensor &qweight,
    const torch::Tensor &qzeros,
    const torch::Tensor &ret_idx,
    int64_t out_features,
    int64_t in_features);

int64_t parallel_thread_count_cpu(const torch::Tensor &input, int64_t rows);

TORCH_LIBRARY(gptqmodel_gptq_cpu, ops) {
    ops.def(
        "gemm_int4_forward(Tensor input, Tensor weight, Tensor zeros, Tensor absmax, "
        "int blocksize) -> Tensor");
    ops.impl("gemm_int4_forward", torch::kCPU, &gemm_int4_cpu_forward);
    ops.def(
        "pack_gptq_int4(Tensor qweight, Tensor qzeros, Tensor ret_idx, "
        "int out_features, int in_features) -> Tensor[]");
    ops.impl("pack_gptq_int4", torch::kCPU, &pack_gptq_int4_cpu_forward);
    // Diagnostic hook used by tests to verify that the OpenMP region follows
    // PyTorch's intra-op thread budget rather than OMP_NUM_THREADS.
    ops.def("_parallel_thread_count(Tensor input, int rows) -> int");
    ops.impl("_parallel_thread_count", torch::kCPU, &parallel_thread_count_cpu);
}
