// SPDX-License-Identifier: Apache-2.0
#include "qvq_gfx950_native.h"
#include <hip/hip_runtime.h>
#include <new>

extern "C" int qvq_gfx950_decode_window(const void*, const void*, const void*,
    void*, unsigned, unsigned, unsigned, unsigned, void*);
extern "C" int qvq_gfx950_rocblas_execute_capture(void*, const void*, const void*, void*, void*);

struct NativeWindowPlan {
  qvq_gfx950_native_config config;
  void* blas;
  void* scratch;
  void* stream;
  int device;
  void* owned_workspace = nullptr;
  const void* decoded_window = nullptr;
  const void* decoded_levels = nullptr;
  const void* decoded_banks = nullptr;
  bool decoded_cache_valid = false;
};

extern "C" int qvq_gfx950_native_scratch_bytes(const qvq_gfx950_native_config* c,
    size_t* bytes) {
  if (!bytes) return -int(hipErrorInvalidValue);
  *bytes = 0;
  if (!c || c->struct_size != sizeof(*c) || c->version != 1 ||
      c->gemm.struct_size != sizeof(c->gemm) || c->gemm.version != 1 ||
      c->gemm.m <= 0 || c->gemm.k <= 0 || c->gemm.n <= 0 || c->gemm.e != 1 ||
      c->gemm.k % 16 || c->gemm.n % 16 || c->gemm.k / 16 > 65535 ||
      c->gemm.solution_index < 0 || c->gemm.reserved ||
      c->transition_bits < 4 || c->transition_bits > 8 ||
      (c->transition_bits == 8 ? c->bank_alt_id != 0 : c->bank_alt_id > 3) ||
      c->decode_threads != 256 || c->cache_policy > 1)
    return -int(hipErrorInvalidValue);
  const uint64_t required = uint64_t(c->gemm.k) * c->gemm.n * 2;
  if (required > SIZE_MAX) return -int(hipErrorInvalidValue);
  *bytes = size_t(required);
  return 0;
}

static int execute_window(void* opaque, const void* x,
    const void* window, const void* levels, const void* banks, void* y, void* stream,
    bool capture_only) {
  auto* p = static_cast<NativeWindowPlan*>(opaque);
  if (!p || !x || !window || !levels || !banks || !y || (!capture_only && stream != p->stream))
    return -int(hipErrorInvalidValue);
  int device;
  const auto hs = hipGetDevice(&device);
  if (hs != hipSuccess) return -int(hs);
  if (device != p->device) return -int(hipErrorInvalidDevice);
  if (capture_only) {
    hipStreamCaptureStatus capture;
    const auto status = hipStreamIsCapturing(static_cast<hipStream_t>(stream), &capture);
    if (status != hipSuccess) return -int(status);
    if (capture != hipStreamCaptureStatusActive) return -int(hipErrorStreamCaptureUnsupported);
  }
  const auto& c = p->config;
  const bool reuse_decoded = c.cache_policy == 1 && p->decoded_cache_valid &&
      p->decoded_window == window && p->decoded_levels == levels &&
      p->decoded_banks == banks;
  if (!reuse_decoded) {
    const int decode_status = qvq_gfx950_decode_window(window, levels, banks,
        p->scratch, c.gemm.k, c.gemm.n, c.transition_bits, c.bank_alt_id, stream);
    if (decode_status) return -decode_status;
    if (c.cache_policy == 1) {
      p->decoded_window = window;
      p->decoded_levels = levels;
      p->decoded_banks = banks;
      p->decoded_cache_valid = true;
    }
  }
  return capture_only ? qvq_gfx950_rocblas_execute_capture(p->blas, x, p->scratch, y, stream)
                      : qvq_gfx950_rocblas_execute(p->blas, x, p->scratch, y, stream);
}

extern "C" int qvq_gfx950_native_execute(void* opaque, const void* x,
    const void* window, const void* levels, const void* banks, void* y, void* stream) {
  return execute_window(opaque, x, window, levels, banks, y, stream, false);
}

extern "C" int qvq_gfx950_native_execute_capture(void* opaque, const void* x,
    const void* window, const void* levels, const void* banks, void* y, void* stream) {
  return execute_window(opaque, x, window, levels, banks, y, stream, true);
}

extern "C" int qvq_gfx950_native_prepare(const qvq_gfx950_native_config* c,
    const void* x, const void* window, const void* levels, const void* banks, void* y,
    void* scratch, size_t scratch_bytes, void* workspace, size_t workspace_bytes,
    void* stream, void** result) {
  if (!result) return -int(hipErrorInvalidValue);
  *result = nullptr;
  size_t required;
  int status = qvq_gfx950_native_scratch_bytes(c, &required);
  if (status) return status;
  if (!scratch || scratch_bytes < required || !x || !window || !levels || !banks || !y)
    return -int(hipErrorInvalidValue);
  void* blas = nullptr;
  // BLAS preparation checks device/capture before any GPU work is submitted.
  status = qvq_gfx950_rocblas_prepare_config(&c->gemm, x, scratch, y, stream,
      workspace, workspace_bytes, &blas);
  if (status) return status;
  int device;
  const auto device_status = hipGetDevice(&device);
  if (device_status != hipSuccess) {
    const int ignored = qvq_gfx950_rocblas_destroy(blas);
    (void)ignored;
    return -int(device_status);
  }
  auto* p = new (std::nothrow) NativeWindowPlan{*c, blas, scratch, stream, device};
  if (!p) {
    const int ignored = qvq_gfx950_rocblas_destroy(blas);
    (void)ignored;
    return -int(hipErrorOutOfMemory);
  }
  status = qvq_gfx950_native_execute(p, x, window, levels, banks, y, stream);
  // Even if GEMM submission failed, a preceding decode may be in flight.
  const auto hs = hipStreamSynchronize(static_cast<hipStream_t>(stream));
  if (!status && hs != hipSuccess) status = -int(hs);
  if (status) {
    const int ignored = qvq_gfx950_rocblas_destroy(blas);
    (void)ignored;
    delete p;
    return status;
  }
  *result = p;
  return 0;
}

extern "C" int qvq_gfx950_native_get_config(void* opaque, qvq_gfx950_native_config* out) {
  const auto* p = static_cast<NativeWindowPlan*>(opaque);
  if (!p || !out || out->struct_size != sizeof(*out) || out->version != 1)
    return -int(hipErrorInvalidValue);
  *out = p->config;
  return 0;
}

extern "C" int qvq_gfx950_native_destroy(void* opaque) {
  auto* p = static_cast<NativeWindowPlan*>(opaque);
  if (!p) return -int(hipErrorInvalidValue);
  const int status = qvq_gfx950_rocblas_destroy(p->blas);
  if (!status) {
    if (p->owned_workspace) {
      const auto scratch_status = hipFree(p->scratch);
      const auto workspace_status = hipFree(p->owned_workspace);
      delete p;
      if (scratch_status != hipSuccess) return -int(scratch_status);
      return workspace_status == hipSuccess ? 0 : -int(workspace_status);
    }
    delete p;
  }
  return status;
}

extern "C" int qvq_gfx950_native_prepare_runtime(const qvq_gfx950_native_config* c,
    void* scratch, size_t scratch_bytes, void* workspace, size_t workspace_bytes,
    void* stream, void** result) {
  if (!result) return -int(hipErrorInvalidValue);
  *result = nullptr;
  size_t required;
  int status = qvq_gfx950_native_scratch_bytes(c, &required);
  if (status) return status;
  if (!scratch || scratch_bytes < required || !workspace || !workspace_bytes)
    return -int(hipErrorInvalidValue);
  auto hs_stream = static_cast<hipStream_t>(stream);
  hipStreamCaptureStatus capture;
  auto hs = hipStreamIsCapturing(hs_stream, &capture);
  if (hs != hipSuccess) return -int(hs);
  if (capture != hipStreamCaptureStatusNone) return -int(hipErrorStreamCaptureUnsupported);
  const size_t tiles = size_t(c->gemm.k / 16) * (c->gemm.n / 16);
  // X, packed window, LUT, bank bytes, Y. The LUT is zero, so warmup outputs
  // are finite and do not depend on user weight/input contents.
  const size_t sizes[5] = {
      size_t(c->gemm.m) * c->gemm.k * 2,
      tiles * c->transition_bits * 16, 512, tiles,
      size_t(c->gemm.m) * c->gemm.n * 4};
  void* buffers[5] = {};
  for (unsigned i = 0; i < 5; ++i) {
    hs = hipMalloc(&buffers[i], sizes[i]);
    if (hs != hipSuccess) { status = -int(hs); break; }
    hs = hipMemsetAsync(buffers[i], 0, sizes[i], hs_stream);
    if (hs != hipSuccess) { status = -int(hs); break; }
  }
  if (!status) {
    status = qvq_gfx950_native_prepare(c, buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
        scratch, scratch_bytes, workspace, workspace_bytes, stream, result);
  }
  // Initialization can fail after queuing a memset or warmup. Finish private
  // work before reclaiming storage even on that error path.
  hs = hipStreamSynchronize(hs_stream);
  if (!status && hs != hipSuccess) status = -int(hs);
  for (void* buffer : buffers) {
    if (buffer) {
      hs = hipFree(buffer);
      if (!status && hs != hipSuccess) status = -int(hs);
    }
  }
  if (status && *result) {
    const int ignored = qvq_gfx950_native_destroy(*result);
    (void)ignored;
    *result = nullptr;
  }
  return status;
}

extern "C" int qvq_gfx950_native_prepare_owned(const qvq_gfx950_native_config* c,
    size_t workspace_bytes, void* stream, void** result) {
  if (!result) return -int(hipErrorInvalidValue);
  *result = nullptr;
  size_t scratch_bytes;
  int status = qvq_gfx950_native_scratch_bytes(c, &scratch_bytes);
  if (status) return status;
  if (!workspace_bytes) return -int(hipErrorInvalidValue);
  hipStreamCaptureStatus capture;
  auto hs = hipStreamIsCapturing(static_cast<hipStream_t>(stream), &capture);
  if (hs != hipSuccess) return -int(hs);
  if (capture != hipStreamCaptureStatusNone) return -int(hipErrorStreamCaptureUnsupported);
  void* scratch = nullptr;
  void* workspace = nullptr;
  hs = hipMalloc(&scratch, scratch_bytes);
  if (hs != hipSuccess) return -int(hs);
  hs = hipMalloc(&workspace, workspace_bytes);
  if (hs == hipSuccess)
    status = qvq_gfx950_native_prepare_runtime(c, scratch, scratch_bytes,
        workspace, workspace_bytes, stream, result);
  else
    status = -int(hs);
  if (status) {
    if (workspace) { const auto ignored = hipFree(workspace); (void)ignored; }
    const auto ignored = hipFree(scratch); (void)ignored;
    return status;
  }
  static_cast<NativeWindowPlan*>(*result)->owned_workspace = workspace;
  return 0;
}

extern "C" int qvq_gfx950_native_prepare_payload(void* opaque,
    const void* window, const void* levels, const void* banks, void* stream) {
  auto* p = static_cast<NativeWindowPlan*>(opaque);
  if (!p || p->config.cache_policy != 1 || !window || !levels || !banks ||
      stream != p->stream)
    return -int(hipErrorInvalidValue);
  int device;
  auto hs = hipGetDevice(&device);
  if (hs != hipSuccess) return -int(hs);
  if (device != p->device) return -int(hipErrorInvalidDevice);
  auto hip_stream = static_cast<hipStream_t>(stream);
  hipStreamCaptureStatus capture;
  hs = hipStreamIsCapturing(hip_stream, &capture);
  if (hs != hipSuccess) return -int(hs);
  if (capture != hipStreamCaptureStatusNone)
    return -int(hipErrorStreamCaptureUnsupported);
  if (p->decoded_cache_valid && p->decoded_window == window &&
      p->decoded_levels == levels && p->decoded_banks == banks)
    return 0;
  const auto& c = p->config;
  const int status = qvq_gfx950_decode_window(window, levels, banks, p->scratch,
      c.gemm.k, c.gemm.n, c.transition_bits, c.bank_alt_id, stream);
  if (status) return -status;
  hs = hipStreamSynchronize(hip_stream);
  if (hs != hipSuccess) return -int(hs);
  p->decoded_window = window;
  p->decoded_levels = levels;
  p->decoded_banks = banks;
  p->decoded_cache_valid = true;
  return 0;
}
