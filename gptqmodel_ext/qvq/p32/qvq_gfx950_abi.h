// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QVQ_GFX950_ABI_VERSION 1
#define QVQ_GFX950_OPERATION_VERSION 1

// AOT artifacts are produced by QVQ's export_qvq_gfx950.py, not by the
// consuming framework. All dimensions and launch metadata are frozen before
// capture. This ABI is independent of the CUDA/SM80 ABI.
typedef struct qvq_gfx950_spec {
    uint32_t abi_version;
    uint32_t operation_version;
    uint32_t m, k, n;
    uint32_t transition_bits;
    uint32_t bank_alt_id;
    uint32_t grid_x, threads, shared_bytes;
} qvq_gfx950_spec;

typedef struct qvq_gfx950_plan qvq_gfx950_plan;
int qvq_gfx950_abi_version(void);
const char* qvq_gfx950_last_error(void);
int qvq_gfx950_device_supported(int device);

// Current HIP device must equal device. Module loading is preparation, never
// capture-safe. The caller verifies the artifact hash and supplies its exact
// QVQ-produced spec/symbol. No ownership of image/symbol is retained.
int qvq_gfx950_prepare(const qvq_gfx950_spec* spec, const void* image,
    size_t image_size, const char* symbol, int device, void* stream,
    qvq_gfx950_plan** result);

// X[M,K] fp16, circular-window words[(K/16)*(N/16),4*transition_bits]
// int32, canonical PGC16 levels[256] fp16, packed binary bank bytes per tile,
// Y[M,N] fp32; all contiguous, caller-owned, on the prepared device.
// W4 uses transition_bits=8, bank_alt_id=0, and zero bank bytes.
// Y = fp32 accumulation of X times decoded fp16 weights. Scales/transforms
// are separate operations; no implicit input/output precision conversion.
// No allocation, lazy compilation/loading, synchronization, or stream change.
// The plan and all buffers must outlive in-flight work and captured graphs.
int qvq_gfx950_execute(const qvq_gfx950_plan* plan, const void* input,
    const void* window, const void* levels, const void* banks,
    void* output, void* stream);

// Call only after all uses finish and all referencing graphs are destroyed.
int qvq_gfx950_destroy(qvq_gfx950_plan* plan);

#ifdef __cplusplus
}
#endif
