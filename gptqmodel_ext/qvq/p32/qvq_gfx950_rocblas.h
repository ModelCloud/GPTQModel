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
  int32_t solution_index; // zero: library heuristic; positive: explicit solution
  uint32_t reserved;     // must be zero
} qvq_gfx950_blas_config;

int qvq_gfx950_rocblas_prepare(int m, int k, int n, void* stream,
    void* workspace, size_t workspace_bytes, void** result);
// Query matching library solutions outside capture. Caller supplies valid X,
// decoded W and Y buffers for this plan's geometry, and host list/count storage.
// A null list queries count; otherwise count is host capacity on entry.
// Solutions require independent accuracy certification before tuning selection.
int qvq_gfx950_rocblas_solutions(void* plan, const void* x, const void* weights,
    void* y, int32_t* solutions, int32_t* count);
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
