#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "quantization/gemm_cuda.h"
#include "quantization/gemv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_forward_cuda", &gemm_forward_cuda, "Quantized GEMM kernel.");
    m.def("grouped_gemm_forward", &grouped_gemm_forward, "Quantized grouped GEMM kernel.");
    m.def("gemmv2_forward_cuda", &gemmv2_forward_cuda, "Quantized v2 GEMM kernel.");
    m.def("gemv_forward_cuda", &gemv_forward_cuda, "Quantized GEMV kernel.");
    m.def("dequantize_weights_cuda", &dequantize_weights_cuda, "Dequantize weights.");
}