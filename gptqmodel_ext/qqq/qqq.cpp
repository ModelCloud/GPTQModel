// Adapted from https://github.com/HandH1998/QQQ

#include "qqq_gemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qqq_gemm", &qqq_gemm, "INT8xINT4 matmul based marlin FP16xINT4 kernel.");
}
