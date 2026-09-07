// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qvq_gfx950_rocblas.h"
#ifdef __cplusplus
extern "C" {
#endif

// Experimental versioned P32 operation. Published AOT ABI v1 is unchanged.
typedef struct qvq_gfx950_native_config {
  uint32_t struct_size, version;
  qvq_gfx950_blas_config gemm;
  uint32_t transition_bits, bank_alt_id;
  uint32_t decode_threads; // currently exactly 256
  uint32_t cache_policy;   // 0: decode each execution; 1: cache by input pointers
} qvq_gfx950_native_config;

// No device initialization: query caller-owned scratch requirements. No hidden
// immutable weight cache is selected by policy 0. Policy 1 reuses decoded W
// while window/levels/banks pointers remain unchanged. E=1, FP16 X/LUT/decoded
// W, FP32 Y, contiguous layouts.
int qvq_gfx950_native_scratch_bytes(const qvq_gfx950_native_config*, size_t* bytes);

// Caller supplies writable sample Y, scratch[N,K] FP16 and BLAS workspace.
// Preparation validates configuration, decodes, warms the selected GEMM and
// synchronizes this stream to finish lazy setup before capture. No ownership
// of caller buffers transfers; scratch/workspace must outlive the plan/graphs.
int qvq_gfx950_native_prepare(const qvq_gfx950_native_config*, const void* x,
    const void* window, const void* levels, const void* banks, void* y,
    void* scratch, size_t scratch_bytes, void* workspace, size_t workspace_bytes,
    void* stream, void** result);
int qvq_gfx950_native_get_config(void* plan, qvq_gfx950_native_config* result);
// For external-runtime initialization: never accesses executable argument
// buffers. Uses temporary zero-filled private inputs/output to warm the plan,
// then frees them after stream synchronization. Scratch/workspace remain owned
// by the caller. This is preparation only and rejects capture.
int qvq_gfx950_native_prepare_runtime(const qvq_gfx950_native_config*,
    void* scratch, size_t scratch_bytes, void* workspace, size_t workspace_bytes,
    void* stream, void** result);
// Allocate plan-owned scratch plus the explicit workspace budget, then warm
// with private inputs outside capture. Persistent bytes are scratch_bytes()
// plus workspace_bytes; temporary warmup storage is additional. No XLA argument
// contents are accessed. Independent plans own independent storage. Destroy
// releases owned storage after the caller completes all uses/destroys graphs.
int qvq_gfx950_native_prepare_owned(const qvq_gfx950_native_config*,
    size_t workspace_bytes, void* stream, void** result);
// Populate the policy-1 decoded cache from the caller's real immutable
// payloads before command-buffer capture. This function never launches GEMM,
// rejects active capture, synchronizes the preparation stream, and is a
// no-op when the same payload pointer tuple is already cached. It requires
// cache_policy=1 and the plan's prepared stream/device.
int qvq_gfx950_native_prepare_payload(void* plan, const void* window,
    const void* levels, const void* banks, void* stream);
// Execute on the prepared stream/device. No tuning, allocation, or sync.
// With cache_policy=0, input/weight contents may change at stable addresses
// between ordered calls. With cache_policy=1, X/Y may change but the window,
// levels and banks pointers must remain stable and their contents immutable;
// changing any of those pointers invalidates and rebuilds the decoded cache.
// Do not overlap executions sharing this plan's scratch/workspace.
int qvq_gfx950_native_execute(void* plan, const void* x, const void* window,
    const void* levels, const void* banks, void* y, void* stream);
// External runtimes may record on a different stream from preparation. This
// entrypoint requires ACTIVE capture on the supplied stream and the same device.
// Caller must exclusively own the plan during recording, with all prior uses
// completed. It temporarily binds the warmed BLAS handle and restores its
// original stream; it does not allocate, synchronize, tune, or change config.
// Serialize all graph replays/eager uses sharing scratch/workspace. Independent
// execution lanes require independent plans and storage. Capture failure must
// discard the partial graph. Ordinary execute retains its strict stream check.
int qvq_gfx950_native_execute_capture(void* plan, const void* x, const void* window,
    const void* levels, const void* banks, void* y, void* capture_stream);
// Synchronize outstanding uses and destroy referencing graphs before destroy.
int qvq_gfx950_native_destroy(void* plan);
#ifdef __cplusplus
}
#endif
