// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define QVQ_MARLIN_ABI_VERSION 1
#define QVQ_MARLIN_KERNEL_VERSION 1
// Prepared Marlin layout, not an on-disk GPTQ/AWQ packing declaration.
typedef struct {
  uint32_t struct_bytes, abi_version;
  int32_t m, k, n, e, lda, device;
  int32_t dtype; // 0=FP16, 1=BF16 (activation and output)
  int64_t weight_type; // QvQ ScalarType id, including its integer bias
  int32_t groups, group_size, has_bias, has_act_order, is_k_full;
  int32_t has_zero_points, zero_points_float;
} QvqMarlinProblem;
typedef struct {
  uint32_t struct_bytes, abi_version;
  int32_t tile_m, tile_n, tile_k, threads, stages;
  int32_t sm_count, blocks_per_sm, max_parallel;
  int32_t shared_memory_bytes;
  int32_t atomic_add, fp32_reduce;
  int32_t prefill; // 0=striped Marlin; 1..4=exact packed-prefill specialization
} QvqMarlinConfig;
typedef struct { void* data; uint64_t bytes; } QvqMarlinBuffer;
typedef struct {
  QvqMarlinBuffer a, weight, scales, global_scale, zeros, group_index, permutation;
  QvqMarlinBuffer bias, output, reduction, permuted_a, locks;
} QvqMarlinBuffers;
typedef struct {
  uint64_t reduction_bytes, permuted_a_bytes, locks_bytes;
  int32_t launch_count, compiled_sm, kernel_version;
} QvqMarlinResources;
typedef struct QvqMarlinPlan QvqMarlinPlan;

// Host preparation selects exactly the supplied specialization. No retuning or
// fallback. E != 1 is rejected: expert batching is not the dense Marlin kernel.
// Create outside capture on problem.device; destroy after all users finish.
// Buffers remain caller-owned through all stream/graph work, with independent
// scratch per concurrent execution lane. Launch performs no allocations/tuning.
int qvq_marlin_prepare(const QvqMarlinProblem*, const QvqMarlinConfig*,
                       QvqMarlinPlan**, QvqMarlinResources*, char*, uint64_t);
int qvq_marlin_launch(const QvqMarlinPlan*, const QvqMarlinBuffers*,
                      void* stream, char*, uint64_t);
void qvq_marlin_destroy(QvqMarlinPlan*);
int qvq_marlin_abi_version(void);
#ifdef __cplusplus
}
#endif
