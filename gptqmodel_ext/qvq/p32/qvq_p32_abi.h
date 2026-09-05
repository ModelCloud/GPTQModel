// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Stable constants shared by runtime implementations and framework/compiler
// adapters. The operation version describes the mathematical P32 contract;
// the ABI version describes this C surface; the kernel version invalidates
// launch-autotune entries when implementation details change.
#define QVQ_P32_OPERATION_VERSION 1
#define QVQ_P32_ABI_VERSION 1
#define QVQ_P32_KERNEL_VERSION 10
#define QVQ_P32_COMPILED_SM 80

#define QVQ_P32_TILE_SIZE 16
#define QVQ_P32_LEVEL_COUNT 256
#define QVQ_P32_TRANSITION_BITS_MIN 4
#define QVQ_P32_TRANSITION_BITS_MAX 7
#define QVQ_P32_SPLIT_COUNT_MAX 128
#define QVQ_P32_STAGE_K_TILES_MIN 1
#define QVQ_P32_STAGE_K_TILES_MAX 4
#define QVQ_P32_SCALAR_M_MAX 4
#define QVQ_P32_GROUPED_M_MAX 16
#define QVQ_P32_GROUP_COUNT_MIN 2
#define QVQ_P32_GROUP_COUNT_MAX 3
#define QVQ_P32_ROW_GROUPS_AUTO 0
#define QVQ_P32_ROW_GROUPS_MIN 1
#define QVQ_P32_ROW_GROUPS_MAX 16

// Kernel variants are compile-time specializations selected by the host
// tuner. The scalar variant is intended for small M; the block variant
// uses one output tile per warp. `threads` must be a multiple of 32 and
// `stage_k_tiles` is the number of K tiles in each cp.async stage.
enum qvq_p32_kernel_variant {
  QVQ_P32_VARIANT_SCALAR = 1,
  QVQ_P32_VARIANT_BLOCK = 2,
};

// Reduction ownership is selected by the framework adapter. Native reduction
// keeps QvQ's specialized CUDA reducer, while graph-visible reduction returns
// split partials for a compiler-managed reduction and its consumers.
enum qvq_p32_reduction_mode {
  QVQ_P32_REDUCTION_NATIVE = 1,
  QVQ_P32_REDUCTION_PARTIALS = 2,
};

struct qvq_p32_config {
  int split_count;
  int kernel_variant;
  int threads;
  int stage_k_tiles;
  int static_n;
  int reduction_mode;
};

// The ABI intentionally uses opaque pointers so framework adapters do not need
// to include CUDA headers. All pointers refer to device memory except `stream`,
// which is the caller-owned CUDA stream. For split_count >
// 1, partial_output must point to split_count * M * N float elements. Native
// reduction writes their sum to output; partials mode leaves the workspace for
// the surrounding graph to reduce. For M > 16, only native reduction is
// supported; the launcher batches 16-row kernel tiles behind this one ABI call.
int qvq_p32_abi_version(void);
int qvq_p32_kernel_version(void);
int qvq_compiled_sm(void);
int qvq_device_sm(int device);
int qvq_driver_version(void);
int qvq_runtime_version(void);
int qvq_toolkit_version(void);
int qvq_device_is_supported(int device);
const char* qvq_last_error(void);

int qvq_p32_window(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int split_count,
    int kernel_variant,
    int threads,
    int stage_k_tiles,
    int static_n,
    int reduction_mode,
    void* stream);

// Explicit large-M scheduling entry point for compiler-owned autotuners.
// `row_groups` selects how many adjacent M16 tiles share one packed weight
// decode and must be 1, 2, 4, 8, or 16. The legacy qvq_p32_window entry point
// retains QvQ's automatic row-group policy for framework callers that do not
// own this tuning axis.
int qvq_p32_window_with_row_groups(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int split_count,
    int kernel_variant,
    int threads,
    int stage_k_tiles,
    int static_n,
    int reduction_mode,
    int row_groups,
    void* stream);

// Native grouped V2B2-P32 entry point. `bank_alt_ids` has one uint8 selector
// per group (two or three groups, with selectors in [1, 3]). M=1..4 uses the
// scalar kernel and M=5..16 uses the block kernel. The final output is row-major
// [M, size_n], while partial buffers use group-major contiguous storage: each
// group owns an [split, M, N_group] slice. `n_tile_end_0` and `n_tile_end_1` are
// cumulative N16 boundaries; the final boundary is size_n / 16.
int qvq_p32_grouped_window(
    const void* input,
    const void* trellis,
    const void* levels,
    const void* bank_ids,
    const void* bank_alt_ids,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int transition_bits,
    int split_count,
    int split_count_0,
    int split_count_1,
    int split_count_2,
    int kernel_variant,
    int threads,
    int stage_k_tiles,
    int static_n,
    int reduction_mode,
    int group_count,
    int n_tile_end_0,
    int n_tile_end_1,
    void* stream);

#ifdef __cplusplus
}
#endif
