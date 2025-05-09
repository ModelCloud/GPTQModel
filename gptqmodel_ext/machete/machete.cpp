#include <torch/all.h>
#include <torch/python.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "machete_pytorch.cuh"
#include "permute_cols.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("machete_prepack_B", &machete::prepack_B, "machete_prepack_B");
  m.def("machete_mm", &machete::mm, "machete_mm");
  m.def("permute_cols", &permute_cols, "permute_cols");
}