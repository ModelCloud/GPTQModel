// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "../torch_stable_compat.h"

namespace gptqmodel_gptq_block {

void block_update_cuda(
    at::Tensor work, const at::Tensor& hinv, const at::Tensor& scale,
    const at::Tensor& zero, const at::Tensor& column_group,
    at::Tensor quantized, at::Tensor errors, at::Tensor losses,
    int64_t maxq);

}  // namespace gptqmodel_gptq_block

STABLE_TORCH_LIBRARY(gptqmodel_gptq_block, m) {
    m.def("block_update(Tensor(a!) work, Tensor hinv, Tensor scale, Tensor zero, Tensor column_group, Tensor(b!) quantized, Tensor(c!) errors, Tensor(d!) losses, int maxq) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(gptqmodel_gptq_block, CUDA, m) {
    m.impl("block_update", TORCH_BOX(&gptqmodel_gptq_block::block_update_cuda));
}
