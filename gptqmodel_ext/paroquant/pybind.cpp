// Empty pybind module. The rotation kernel registers itself via TORCH_LIBRARY in rotation.cu.

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
