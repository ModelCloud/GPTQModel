#include <torch/extension.h>
#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gptq_gemm", &gptq_gemm, "gptq_gemm")
    .def("gptq_gemm_lora", &gptq_gemm_lora, "gptq_gemm_lora")
    .def("gptq_shuffle", &gptq_shuffle, "gptq_shuffle")
    ;
}