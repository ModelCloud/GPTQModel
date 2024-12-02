/*
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 *
 * LICENSE: GPTQModel/licenses/LICENSE.apache
 */

#include <torch/all.h>
#include <torch/python.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "marlin_cuda_kernel.cuh"
#include "marlin_repack.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gptq_marlin_gemm", &gptq_marlin_gemm, "Marlin FP16xINT4 matmul.");
  m.def("gptq_marlin_repack", &gptq_marlin_repack, "Repack GPTQ checkpoints for Marlin.");
}