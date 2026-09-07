// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// Experimental dense-GEMM sub-ABI. Does not replace published QVQ ABI v1.
typedef struct qvq_gfx950_blas_config {
  uint32_t struct_size, version;
  int32_t m, k, n, e;
  int32_t solution_index; // zero: standard/default; nonzero: explicit solution
  uint32_t reserved;     // must be zero
} qvq_gfx950_blas_config;

// Standalone tuning is preparation-only and never runs during capture/replay.
// A null options pointer selects the bounded defaults (one warmup, three
// timed samples). Solution IDs are valid only for the exact rocBLAS build.
typedef struct qvq_gfx950_tuning_options {
  uint32_t struct_size, version;
  uint32_t warmup_iterations, benchmark_iterations;
  uint32_t reserved;
} qvq_gfx950_tuning_options;

typedef struct qvq_gfx950_tuning_result {
  uint32_t struct_size, version;
  int32_t solution_index;
  uint32_t candidates_tested, candidates_failed, samples;
  float median_us;
  uint32_t reserved;
} qvq_gfx950_tuning_result;

int qvq_gfx950_rocblas_prepare(int m, int k, int n, void* stream,
    void* workspace, size_t workspace_bytes, void** result);
// Query matching library solutions outside capture. Caller supplies valid X,
// decoded W and Y buffers for this plan's geometry, and host list/count storage.
// A null list queries count; otherwise count is host capacity on entry.
// Solutions require independent accuracy certification before tuning selection.
int qvq_gfx950_rocblas_solutions(void* plan, const void* x, const void* weights,
    void* y, int32_t* solutions, int32_t* count);
// Enumerate and time complete GEMM candidates on the prepared stream. The
// selected solution is retained in the plan and returned in result. Solution
// zero is a valid measured standard-algorithm winner. This call
// synchronizes only for measurement and rejects active stream capture; callers
// must independently certify numerical accuracy before using the winner.
int qvq_gfx950_rocblas_autotune(void* plan, const void* x, const void* weights,
    void* y, const qvq_gfx950_tuning_options* options,
    qvq_gfx950_tuning_result* result);
// Explicit selection is validated and frozen before returning. No autotuning or
// silent solution substitution. E>1 is unsupported. Workspace belongs to caller.
int qvq_gfx950_rocblas_prepare_config(const qvq_gfx950_blas_config* config,
    const void* x, const void* weights, void* y, void* stream,
    void* workspace, size_t workspace_bytes, void** result);
int qvq_gfx950_rocblas_get_config(void* plan, qvq_gfx950_blas_config* result);
int qvq_gfx950_rocblas_execute(void* plan, const void* x,
    const void* weights, void* y, void* stream);
int qvq_gfx950_rocblas_destroy(void* plan);
#ifdef __cplusplus
}
#endif
