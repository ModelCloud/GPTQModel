// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "../torch_stable_compat.h"
#include <optional>
#include <vector>

torch::Tensor gemm_int4_qzero_group_scaled_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &zeros,
    const torch::Tensor &absmax,
    int64_t blocksize,
    const std::optional<torch::Tensor> &bias);

std::vector<torch::Tensor>
pack_gptq_int4_cpu_forward(
    const torch::Tensor &qweight,
    const torch::Tensor &qzeros,
    const torch::Tensor &ret_idx,
    int64_t out_features,
    int64_t in_features);

std::vector<torch::Tensor>
pack_awq_int4_cpu_forward(
    const torch::Tensor &qweight,
    const torch::Tensor &qzeros,
    int64_t out_features,
    int64_t in_features);

int64_t parallel_thread_count_cpu(const torch::Tensor &input, int64_t rows);

STABLE_TORCH_LIBRARY(gptqmodel_int4_qzero_group_scaled_cpu, ops) {
    ops.def(
        "gemm_int4_qzero_group_scaled_cpu_forward(Tensor input, Tensor weight, "
        "Tensor zeros, Tensor absmax, int blocksize, Tensor? bias=None) -> Tensor");
    ops.def(
        "pack_gptq_int4(Tensor qweight, Tensor qzeros, Tensor ret_idx, "
        "int out_features, int in_features) -> Tensor[]");
    ops.def(
        "pack_awq_int4(Tensor qweight, Tensor qzeros, "
        "int out_features, int in_features) -> Tensor[]");
    // Diagnostic hook used by tests to verify that the OpenMP region follows
    // PyTorch's intra-op thread budget rather than OMP_NUM_THREADS.
    ops.def("_parallel_thread_count(Tensor input, int rows) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(gptqmodel_int4_qzero_group_scaled_cpu, CPU, ops) {
    ops.impl(
        "gemm_int4_qzero_group_scaled_cpu_forward",
        TORCH_BOX(&gemm_int4_qzero_group_scaled_cpu_forward));
    ops.impl("pack_gptq_int4", TORCH_BOX(&pack_gptq_int4_cpu_forward));
    ops.impl("pack_awq_int4", TORCH_BOX(&pack_awq_int4_cpu_forward));
    ops.impl("_parallel_thread_count", TORCH_BOX(&parallel_thread_count_cpu));
}
