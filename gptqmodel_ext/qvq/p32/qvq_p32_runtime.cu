// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "qvq_p32_abi.h"
#include "qvq_p32_internal.h"

#include <cuda_runtime.h>

#include <cstdio>

namespace qvq_p32_internal {

thread_local char last_error[256] = {};

void set_last_error(const char* message) {
  const char* text = message == nullptr ? "unknown CUDA error" : message;
  std::snprintf(last_error, sizeof(last_error), "%s", text);
}

}  // namespace qvq_p32_internal

extern "C" int qvq_p32_abi_version(void) {
  return QVQ_P32_ABI_VERSION;
}

extern "C" int qvq_p32_kernel_version(void) {
  return QVQ_P32_KERNEL_VERSION;
}

extern "C" int qvq_compiled_sm(void) {
  return QVQ_P32_COMPILED_SM;
}

extern "C" int qvq_device_sm(int device) {
  cudaDeviceProp properties{};
  const cudaError_t error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) return 0;
  return properties.major * 10 + properties.minor;
}

extern "C" int qvq_driver_version(void) {
  int version = 0;
  return cudaDriverGetVersion(&version) == cudaSuccess ? version : 0;
}

extern "C" int qvq_runtime_version(void) {
  int version = 0;
  return cudaRuntimeGetVersion(&version) == cudaSuccess ? version : 0;
}

extern "C" int qvq_toolkit_version(void) {
  return CUDART_VERSION;
}

extern "C" int qvq_device_is_supported(int device) {
  cudaDeviceProp properties{};
  const cudaError_t error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) return 0;
  return properties.major * 10 + properties.minor == QVQ_P32_COMPILED_SM ? 1 : 0;
}

extern "C" const char* qvq_last_error(void) {
  return qvq_p32_internal::last_error;
}
