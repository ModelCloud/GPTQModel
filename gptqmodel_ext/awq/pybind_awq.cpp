#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "quantization/gemm_cuda.h"
#include "quantization/gemv_cuda.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "gemm_forward_cuda",
        &gemm_forward_cuda,
        py::arg("in_feats"),
        py::arg("kernel"),
        py::arg("scaling_factors"),
        py::arg("zeros"),
        py::arg("split_k_iters"),
        py::arg("fp32_accum") = false,
        "Quantized GEMM kernel."
    );
    m.def("gemm_forward_cuda_fp32_reduce", &gemm_forward_cuda_fp32_reduce, "Quantized GEMM kernel with fp32 split-K reduction.");
    m.def("grouped_gemm_forward", &grouped_gemm_forward, "Quantized grouped GEMM kernel.");
    m.def("gemmv2_forward_cuda", &gemmv2_forward_cuda, "Quantized v2 GEMM kernel.");
    m.def("gemv_forward_cuda", &gemv_forward_cuda, "Quantized GEMV kernel.");
    m.def("dequantize_weights_cuda", &dequantize_weights_cuda, "Dequantize weights.");
}
