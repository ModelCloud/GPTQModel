// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <torch/library.h>
#include <torch/types.h>

torch::Tensor grasshopper_gptq_gemv_cuda(torch::Tensor vec,
                                       torch::Tensor qweight,
                                       torch::Tensor scales,
                                       torch::Tensor qzeros,
                                       int64_t group_size,
                                       int64_t accumulation_type,
                                       int64_t bits);

torch::Tensor grasshopper_gptq_gemv_lora_cuda(torch::Tensor vec,
                                            torch::Tensor qweight,
                                            torch::Tensor scales,
                                            torch::Tensor qzeros,
                                            torch::Tensor down,
                                            torch::Tensor up,
                                            int64_t group_size,
                                            int64_t accumulation_type,
                                            int64_t bits);

torch::Tensor grasshopper_gptq_gemv_lora_int8_cuda(torch::Tensor vec,
                                                 torch::Tensor qweight,
                                                 torch::Tensor scales,
                                                 torch::Tensor qzeros,
                                                 torch::Tensor down,
                                                 torch::Tensor up_qweight,
                                                 torch::Tensor up_scales,
                                                 int64_t group_size,
                                                 int64_t lora_group_size,
                                                 int64_t accumulation_type,
                                                 int64_t bits);

torch::Tensor grasshopper_gptq_gemm_cuda(torch::Tensor vec,
                                       torch::Tensor qweight,
                                       torch::Tensor scales,
                                       torch::Tensor qzeros,
                                       int64_t group_size,
                                       int64_t accumulation_type,
                                       int64_t bits);

torch::Tensor grasshopper_gptq_gemm_lora_cuda(torch::Tensor vec,
                                            torch::Tensor qweight,
                                            torch::Tensor scales,
                                            torch::Tensor qzeros,
                                            torch::Tensor down,
                                            torch::Tensor up,
                                            int64_t group_size,
                                            int64_t accumulation_type,
                                            int64_t bits);

torch::Tensor grasshopper_gptq_gemm_lora_int8_cuda(torch::Tensor vec,
                                                 torch::Tensor qweight,
                                                 torch::Tensor scales,
                                                 torch::Tensor qzeros,
                                                 torch::Tensor down,
                                                 torch::Tensor up_qweight,
                                                 torch::Tensor up_scales,
                                                 int64_t group_size,
                                                 int64_t lora_group_size,
                                                 int64_t accumulation_type,
                                                 int64_t bits);

namespace {

torch::Tensor grasshopper_gptq_gemv_dispatch(torch::Tensor vec,
                                           torch::Tensor qweight,
                                           torch::Tensor scales,
                                           torch::Tensor qzeros,
                                           int64_t group_size,
                                           int64_t accumulation_type,
                                           int64_t bits) {
  return grasshopper_gptq_gemv_cuda(vec, qweight, scales, qzeros, group_size,
                                  accumulation_type, bits);
}

torch::Tensor grasshopper_gptq_gemv_lora_dispatch(torch::Tensor vec,
                                                torch::Tensor qweight,
                                                torch::Tensor scales,
                                                torch::Tensor qzeros,
                                                torch::Tensor down,
                                                torch::Tensor up,
                                                int64_t group_size,
                                                int64_t accumulation_type,
                                                int64_t bits) {
  return grasshopper_gptq_gemv_lora_cuda(vec, qweight, scales, qzeros, down, up,
                                       group_size, accumulation_type, bits);
}

torch::Tensor grasshopper_gptq_gemv_lora_int8_dispatch(torch::Tensor vec,
                                                     torch::Tensor qweight,
                                                     torch::Tensor scales,
                                                     torch::Tensor qzeros,
                                                     torch::Tensor down,
                                                     torch::Tensor up_qweight,
                                                     torch::Tensor up_scales,
                                                     int64_t group_size,
                                                     int64_t lora_group_size,
                                                     int64_t accumulation_type,
                                                     int64_t bits) {
  return grasshopper_gptq_gemv_lora_int8_cuda(
      vec, qweight, scales, qzeros, down, up_qweight, up_scales, group_size,
      lora_group_size, accumulation_type, bits);
}

torch::Tensor grasshopper_gptq_gemm_dispatch(torch::Tensor vec,
                                           torch::Tensor qweight,
                                           torch::Tensor scales,
                                           torch::Tensor qzeros,
                                           int64_t group_size,
                                           int64_t accumulation_type,
                                           int64_t bits) {
  return grasshopper_gptq_gemm_cuda(vec, qweight, scales, qzeros, group_size,
                                  accumulation_type, bits);
}

torch::Tensor grasshopper_gptq_gemm_lora_dispatch(torch::Tensor vec,
                                                torch::Tensor qweight,
                                                torch::Tensor scales,
                                                torch::Tensor qzeros,
                                                torch::Tensor down,
                                                torch::Tensor up,
                                                int64_t group_size,
                                                int64_t accumulation_type,
                                                int64_t bits) {
  return grasshopper_gptq_gemm_lora_cuda(vec, qweight, scales, qzeros, down, up,
                                       group_size, accumulation_type, bits);
}

torch::Tensor grasshopper_gptq_gemm_lora_int8_dispatch(torch::Tensor vec,
                                                     torch::Tensor qweight,
                                                     torch::Tensor scales,
                                                     torch::Tensor qzeros,
                                                     torch::Tensor down,
                                                     torch::Tensor up_qweight,
                                                     torch::Tensor up_scales,
                                                     int64_t group_size,
                                                     int64_t lora_group_size,
                                                     int64_t accumulation_type,
                                                     int64_t bits) {
  return grasshopper_gptq_gemm_lora_int8_cuda(
      vec, qweight, scales, qzeros, down, up_qweight, up_scales, group_size,
      lora_group_size, accumulation_type, bits);
}

}  // namespace

TORCH_LIBRARY(gptqmodel_grasshopper, m) {
  m.def("gemv(Tensor vec, Tensor qweight, Tensor scales, Tensor qzeros, int group_size, int accumulation_type=0, int bits=3) -> Tensor");
  m.def("gemv_lora(Tensor vec, Tensor qweight, Tensor scales, Tensor qzeros, Tensor down, Tensor up, int group_size, int accumulation_type=0, int bits=3) -> Tensor");
  m.def("gemv_lora_int8(Tensor vec, Tensor qweight, Tensor scales, Tensor qzeros, Tensor down, Tensor up_qweight, Tensor up_scales, int group_size, int lora_group_size, int accumulation_type=0, int bits=3) -> Tensor");
  m.def("gemm(Tensor vec, Tensor qweight, Tensor scales, Tensor qzeros, int group_size, int accumulation_type=0, int bits=3) -> Tensor");
  m.def("gemm_lora(Tensor vec, Tensor qweight, Tensor scales, Tensor qzeros, Tensor down, Tensor up, int group_size, int accumulation_type=0, int bits=3) -> Tensor");
  m.def("gemm_lora_int8(Tensor vec, Tensor qweight, Tensor scales, Tensor qzeros, Tensor down, Tensor up_qweight, Tensor up_scales, int group_size, int lora_group_size, int accumulation_type=0, int bits=3) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_grasshopper, CUDA, m) {
  m.impl("gemv", &grasshopper_gptq_gemv_dispatch);
  m.impl("gemv_lora", &grasshopper_gptq_gemv_lora_dispatch);
  m.impl("gemv_lora_int8", &grasshopper_gptq_gemv_lora_int8_dispatch);
  m.impl("gemm", &grasshopper_gptq_gemm_dispatch);
  m.impl("gemm_lora", &grasshopper_gptq_gemm_lora_dispatch);
  m.impl("gemm_lora_int8", &grasshopper_gptq_gemm_lora_int8_dispatch);
}
