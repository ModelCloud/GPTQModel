/*
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 *
 * LICENSE: GPTQModel/licenses/LICENSE.apache
 */

#include <torch/extension.h>

#include "gptq_marlin.cuh"
#include "gptq_marlin_repack.cuh"
#include "awq_marlin_repack.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gptq_marlin_gemm", &gptq_marlin_gemm, "Marlin FP16xINT4 matmul.");
  m.def("gptq_marlin_repack", &gptq_marlin_repack, "Repack GPTQ checkpoints for Marlin.");
  m.def("awq_marlin_repack", &awq_marlin_repack, "Repack AWQ checkpoints for Marlin.");
}