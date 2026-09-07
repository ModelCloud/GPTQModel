// SPDX-License-Identifier: Apache-2.0
// Experimental FP16-input / FP32-output baseline; not production dispatch.
#include <hip/hip_runtime.h>
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include "qvq_gfx950_rocblas.h"
#include <algorithm>
#include <cstring>
#include <new>
#include <vector>

struct NativeBlasPlan {
  rocblas_handle handle;
  hipStream_t stream;
  int device, m, k, n;
  int solution = 0;
};

// Workspace belongs to the caller and must outlive the plan and all graphs.
// Use a separate plan/workspace for each concurrent execution stream.
// Return negative HIP errors, positive rocBLAS errors, or zero on success.
extern "C" int qvq_gfx950_rocblas_prepare(int m, int k, int n, void* stream,
    void* workspace, size_t workspace_bytes, void** result) {
  if (!result) return -int(hipErrorInvalidValue);
  *result = nullptr;
  if (m <= 0 || k <= 0 || n <= 0 || !workspace || !workspace_bytes)
    return -int(hipErrorInvalidValue);
  auto hip_stream = static_cast<hipStream_t>(stream);
  hipStreamCaptureStatus capture;
  auto hs = hipStreamIsCapturing(hip_stream, &capture);
  if (hs != hipSuccess) return -int(hs);
  if (capture != hipStreamCaptureStatusNone) return -int(hipErrorStreamCaptureUnsupported);
  int device;
  hs = hipGetDevice(&device);
  if (hs != hipSuccess) return -int(hs);
  hipDeviceProp_t props{};
  hs = hipGetDeviceProperties(&props, device);
  if (hs != hipSuccess) return -int(hs);
  if (std::strncmp(props.gcnArchName, "gfx950", 6) != 0 ||
      (props.gcnArchName[6] && props.gcnArchName[6] != ':'))
    return -int(hipErrorNoBinaryForGpu);
  auto* plan = new (std::nothrow) NativeBlasPlan{nullptr, hip_stream, device, m, k, n};
  if (!plan) return -int(hipErrorOutOfMemory);
  rocblas_initialize();
  auto status = rocblas_create_handle(&plan->handle);
  if (status == rocblas_status_success)
    status = rocblas_set_stream(plan->handle, hip_stream);
  if (status == rocblas_status_success)
    status = rocblas_set_workspace(plan->handle, workspace, workspace_bytes);
  if (status != rocblas_status_success) {
    if (plan->handle) {
      const auto ignored = rocblas_destroy_handle(plan->handle);
      (void)ignored;
    }
    delete plan;
    return int(status);
  }
  *result = plan;
  return 0;
}

// X[M,K] and decoded W[N,K] are FP16; Y[M,N] is FP32, all row-major.
// Interpret Y as column-major [N,M]: Y^T = transpose(W_col[K,N]) * X_col[K,M].
// Caller must warm this exact operation outside capture before recording it.
static int execute_blas(void* opaque, const void* x,
    const void* weights, void* y, void* stream, bool capture_only) {
  auto* plan = static_cast<NativeBlasPlan*>(opaque);
  if (!plan || !x || !weights || !y || (!capture_only && stream != plan->stream))
    return -int(hipErrorInvalidValue);
  int device;
  const auto hs = hipGetDevice(&device);
  if (hs != hipSuccess) return -int(hs);
  if (device != plan->device) return -int(hipErrorInvalidDevice);
  if (capture_only) {
    hipStreamCaptureStatus capture;
    const auto status = hipStreamIsCapturing(static_cast<hipStream_t>(stream), &capture);
    if (status != hipSuccess) return -int(status);
    if (capture != hipStreamCaptureStatusActive) return -int(hipErrorStreamCaptureUnsupported);
    const auto bind = rocblas_set_stream(plan->handle, static_cast<hipStream_t>(stream));
    if (bind != rocblas_status_success) return int(bind);
  }
  const float alpha = 1, beta = 0;
  const int result = int(rocblas_gemm_ex(plan->handle, rocblas_operation_transpose,
      rocblas_operation_none, plan->n, plan->m, plan->k, &alpha,
      weights, rocblas_datatype_f16_r, plan->k,
      x, rocblas_datatype_f16_r, plan->k, &beta,
      y, rocblas_datatype_f32_r, plan->n,
      y, rocblas_datatype_f32_r, plan->n,
      rocblas_datatype_f32_r,
      plan->solution ? rocblas_gemm_algo_solution_index : rocblas_gemm_algo_standard,
      plan->solution, 0));
  if (capture_only) {
    const auto restore = rocblas_set_stream(plan->handle, plan->stream);
    if (!result && restore != rocblas_status_success) return int(restore);
  }
  return result;
}

extern "C" int qvq_gfx950_rocblas_execute(void* opaque, const void* x,
    const void* weights, void* y, void* stream) {
  return execute_blas(opaque, x, weights, y, stream, false);
}

extern "C" int qvq_gfx950_rocblas_execute_capture(void* opaque, const void* x,
    const void* weights, void* y, void* stream) {
  return execute_blas(opaque, x, weights, y, stream, true);
}

extern "C" int qvq_gfx950_rocblas_get_config(void* opaque, qvq_gfx950_blas_config* out) {
  const auto* plan = static_cast<NativeBlasPlan*>(opaque);
  if (!plan || !out || out->struct_size != sizeof(*out) || out->version != 1)
    return -int(hipErrorInvalidValue);
  *out = {sizeof(*out), 1, plan->m, plan->k, plan->n, 1, plan->solution, 0};
  return 0;
}

extern "C" int qvq_gfx950_rocblas_solutions(void* opaque, const void* x,
    const void* weights, void* y, int32_t* list, int32_t* count) {
  auto* plan = static_cast<NativeBlasPlan*>(opaque);
  if (!plan || !x || !weights || !y || !count || (list && *count < 0))
    return -int(hipErrorInvalidValue);
  int device;
  auto hs = hipGetDevice(&device);
  if (hs != hipSuccess) return -int(hs);
  if (device != plan->device) return -int(hipErrorInvalidDevice);
  hipStreamCaptureStatus capture;
  hs = hipStreamIsCapturing(plan->stream, &capture);
  if (hs != hipSuccess) return -int(hs);
  if (capture != hipStreamCaptureStatusNone) return -int(hipErrorStreamCaptureUnsupported);
  const float alpha = 1, beta = 0;
  // The installed 5.6 library exports this beta API. Keep its use isolated;
  // solution IDs are library-build-specific, never portable tuning cache keys.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  const auto status = rocblas_gemm_ex_get_solutions(plan->handle,
      rocblas_operation_transpose, rocblas_operation_none,
      plan->n, plan->m, plan->k, &alpha,
      weights, rocblas_datatype_f16_r, plan->k,
      x, rocblas_datatype_f16_r, plan->k, &beta,
      y, rocblas_datatype_f32_r, plan->n, y, rocblas_datatype_f32_r, plan->n,
      rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0, list, count);
#pragma GCC diagnostic pop
  return int(status);
}

extern "C" int qvq_gfx950_rocblas_prepare_config(const qvq_gfx950_blas_config* config,
    const void* x, const void* weights, void* y, void* stream,
    void* workspace, size_t workspace_bytes, void** result) {
  if (!result) return -int(hipErrorInvalidValue);
  *result = nullptr;
  if (!config || config->struct_size != sizeof(*config) || config->version != 1 ||
      config->e != 1 || config->solution_index < 0 || config->reserved || !x || !weights || !y)
    return -int(hipErrorInvalidValue);
  void* opaque = nullptr;
  int status = qvq_gfx950_rocblas_prepare(config->m, config->k, config->n, stream,
      workspace, workspace_bytes, &opaque);
  if (status) return status;
  if (config->solution_index) {
    int32_t count = 0;
    status = qvq_gfx950_rocblas_solutions(opaque, x, weights, y, nullptr, &count);
    if (!status && count > 0) {
      try {
        std::vector<int32_t> solutions(count);
        status = qvq_gfx950_rocblas_solutions(opaque, x, weights, y, solutions.data(), &count);
        if (!status && std::find(solutions.begin(), solutions.end(), config->solution_index) == solutions.end())
          status = -int(hipErrorInvalidValue);
      } catch (const std::bad_alloc&) {
        status = -int(hipErrorOutOfMemory);
      }
    } else if (!status) {
      status = -int(hipErrorInvalidValue);
    }
  }
  if (status) {
    const int cleanup = qvq_gfx950_rocblas_destroy(opaque);
    (void)cleanup;
    return status;
  }
  static_cast<NativeBlasPlan*>(opaque)->solution = config->solution_index;
  *result = opaque;
  return 0;
}

// All asynchronous uses must finish and referencing graphs must be destroyed.
extern "C" int qvq_gfx950_rocblas_destroy(void* opaque) {
  auto* plan = static_cast<NativeBlasPlan*>(opaque);
  if (!plan) return -int(hipErrorInvalidValue);
  int device;
  const auto hs = hipGetDevice(&device);
  if (hs != hipSuccess) return -int(hs);
  if (device != plan->device) return -int(hipErrorInvalidDevice);
  const auto status = rocblas_destroy_handle(plan->handle);
  if (status == rocblas_status_success) delete plan;
  return int(status);
}
