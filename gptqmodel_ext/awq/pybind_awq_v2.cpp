#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "quantization_new/gemm/gemm_cuda.h"
#include "quantization_new/gemv/gemv_cuda.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "gemm_forward_cuda",
        &gemm_forward_cuda_prefill,
        "Quantized GEMM kernel (v2).",
        py::arg("in_feats"),
        py::arg("kernel"),
        py::arg("scales"),
        py::arg("zeros"),
        py::arg("use_fp32") = false);
    m.def(
        "gemm_forward_cuda_prefill",
        &gemm_forward_cuda_prefill,
        "Quantized GEMM kernel (v2).",
        py::arg("in_feats"),
        py::arg("kernel"),
        py::arg("scales"),
        py::arg("zeros"),
        py::arg("use_fp32") = false);
    m.def("gemv_forward_cuda", &gemv_forward_cuda_decode, "Quantized GEMV kernel (v2).");
    m.def("gemv_forward_cuda_decode", &gemv_forward_cuda_decode, "Quantized GEMV kernel (v2).");
}
