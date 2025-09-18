#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "quantization_new/gemm/gemm_cuda.h"
#include "quantization_new/gemv/gemv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_forward_cuda_prefill", &gemm_forward_cuda_prefill, "New quantized GEMM kernel.");
    m.def("gemv_forward_cuda_decode", &gemv_forward_cuda_decode, "New quantized GEMM kernel.");
}