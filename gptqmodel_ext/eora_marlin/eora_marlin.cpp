// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <torch/library.h>
#include <torch/types.h>

torch::Tensor eora_marlin_lora_up_add_cuda(torch::Tensor down,
                                           torch::Tensor up,
                                           torch::Tensor out);

namespace {

torch::Tensor eora_marlin_lora_up_add_dispatch(torch::Tensor down,
                                               torch::Tensor up,
                                               torch::Tensor out) {
  return eora_marlin_lora_up_add_cuda(down, up, out);
}

}  // namespace

TORCH_LIBRARY(gptqmodel_eora_marlin, m) {
  m.def("lora_up_add(Tensor down, Tensor up, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(gptqmodel_eora_marlin, CUDA, m) {
  m.impl("lora_up_add", &eora_marlin_lora_up_add_dispatch);
}
