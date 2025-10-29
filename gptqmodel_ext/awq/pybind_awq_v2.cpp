#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "quantization_new/gemm/gemm_cuda.h"
#include "quantization_new/gemv/gemv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_forward_cuda", &gemm_forward_cuda_prefill, "Quantized GEMM kernel (v2).");
    m.def("gemm_forward_cuda_prefill", &gemm_forward_cuda_prefill, "Quantized GEMM kernel (v2).");
    m.def("gemv_forward_cuda", &gemv_forward_cuda_decode, "Quantized GEMV kernel (v2).");
    m.def("gemv_forward_cuda_decode", &gemv_forward_cuda_decode, "Quantized GEMV kernel (v2).");
}
