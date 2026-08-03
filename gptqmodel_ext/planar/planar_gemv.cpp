// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
// SPDX-License-Identifier: Apache-2.0
// Contact: qubitium@modelcloud.ai, x.com/qubitium

#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>

torch::Tensor pangolin_gemv_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor g_idx,
    int64_t bits);

TORCH_LIBRARY(gptqmodel_pangolin, m) {
  m.def("gemv(Tensor input, Tensor qweight, Tensor scales, Tensor qzeros, Tensor g_idx, int bits) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_pangolin, CUDA, m) {
  m.impl("gemv", pangolin_gemv_cuda);
}
