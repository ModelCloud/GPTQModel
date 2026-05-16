// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <torch/library.h>
#include <torch/types.h>

torch::Tensor vecquant3_gptq_gemv_cuda(torch::Tensor vec,
                                       torch::Tensor qweight,
                                       torch::Tensor scales,
                                       torch::Tensor qzeros,
                                       int64_t group_size);

torch::Tensor vecquant3_gptq_gemv_lora_cuda(torch::Tensor vec,
                                            torch::Tensor qweight,
                                            torch::Tensor scales,
                                            torch::Tensor qzeros,
                                            torch::Tensor down,
                                            torch::Tensor up,
                                            int64_t group_size);

namespace {

torch::Tensor vecquant3_gptq_gemv_dispatch(torch::Tensor vec,
                                           torch::Tensor qweight,
                                           torch::Tensor scales,
                                           torch::Tensor qzeros,
                                           int64_t group_size) {
  return vecquant3_gptq_gemv_cuda(vec, qweight, scales, qzeros, group_size);
}

torch::Tensor vecquant3_gptq_gemv_lora_dispatch(torch::Tensor vec,
                                                torch::Tensor qweight,
                                                torch::Tensor scales,
                                                torch::Tensor qzeros,
                                                torch::Tensor down,
                                                torch::Tensor up,
                                                int64_t group_size) {
  return vecquant3_gptq_gemv_lora_cuda(vec, qweight, scales, qzeros, down, up,
                                       group_size);
}

}  // namespace

TORCH_LIBRARY(gptqmodel_vecquant3, m) {
  m.def("gemv(Tensor vec, Tensor qweight, Tensor scales, Tensor qzeros, int group_size) -> Tensor");
  m.def("gemv_lora(Tensor vec, Tensor qweight, Tensor scales, Tensor qzeros, Tensor down, Tensor up, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_vecquant3, CUDA, m) {
  m.impl("gemv", &vecquant3_gptq_gemv_dispatch);
  m.impl("gemv_lora", &vecquant3_gptq_gemv_lora_dispatch);
}
