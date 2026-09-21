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
#define QVQ_P32_ABI_VERSION 3
#define QVQ_P32_KERNEL_VERSION 16
#define QVQ_P32_COMPILED_SM 90

#define QVQ_P32_TILE_SIZE 16
#define QVQ_P32_LEVEL_COUNT 256
#define QVQ_P32_TRANSITION_BITS_MIN 4
#define QVQ_P32_TRANSITION_BITS_MAX 7
#define QVQ_P32_SPLIT_COUNT_MAX 128
#define QVQ_P32_STAGE_K_TILES_MIN 1
#define QVQ_P32_STAGE_K_TILES_MAX 4
#define QVQ_P32_SCALAR_M_MAX 4
#define QVQ_P32_GROUPED_M_MAX 16
#define QVQ_P32_RANK8_MAX_COUNT 24
#define QVQ_P32_GROUP_COUNT_MIN 2
#define QVQ_P32_GROUP_COUNT_MAX 3
#define QVQ_P32_ROW_GROUPS_AUTO 0
#define QVQ_P32_ROW_GROUPS_MIN 1
#define QVQ_P32_ROW_GROUPS_MAX 16
#define QVQ_P32_TUNING_AUTO 0
#define QVQ_P32_TUNING_EXTERNAL 1
#define QVQ_P32_N_TILES_AUTO 0
#define QVQ_P32_WARPS_AUTO 0
#define QVQ_P32_LAUNCH_PLAN_MAX_LAUNCHES 5
#define QVQ_P32_LAUNCH_PLAN_MAX_ARGS 16
#define QVQ_P32_LAUNCH_PLAN_HOST_STORAGE_WORDS 128

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
  // External bridges set tuning_mode=EXTERNAL and may provide the derived
  // N-tile and warp choices explicitly. Zero selects the native policy for
  // that axis. The runtime rejects a non-native geometry until a matching
  // compiled specialization is available; it never silently ignores a
  // requested tuning value.
  int tuning_mode;
  int n_tiles_per_block;
  int n_warps;
};

// Framework-neutral CUDA launch metadata for compiler-owned command buffers.
// `kernel_symbol` is the CUDA C++ host symbol accepted by
// cudaGetFuncBySymbol. Device-pointer arguments store the device pointer
// directly in `address`; host-value arguments point into the plan's inline
// storage and are copied by the consuming compiler runtime. Dependencies are
// launch indexes in this plan. The descriptor exposes launch structure without
// introducing an XLA/PJRT dependency into QvQ.
// Host-value addresses refer into this exact plan instance: do not copy or move
// a populated plan. Consume its arguments before destroying or rebuilding it.
enum qvq_p32_launch_arg_type {
  QVQ_P32_LAUNCH_ARG_DEVICE_POINTER = 1,
  QVQ_P32_LAUNCH_ARG_HOST_VALUE = 2,
};

struct qvq_p32_launch_arg {
  const void* address;
  long long size;
  int type;
};

struct qvq_p32_launch_descriptor {
  const void* kernel_symbol;
  const char* kernel_name;
  unsigned grid_x;
  unsigned grid_y;
  unsigned grid_z;
  unsigned block_x;
  unsigned block_y;
  unsigned block_z;
  unsigned shared_memory_bytes;
  int uses_pdl;
  int arg_count;
  struct qvq_p32_launch_arg args[QVQ_P32_LAUNCH_PLAN_MAX_ARGS];
  int dependency_count;
  int dependencies[QVQ_P32_LAUNCH_PLAN_MAX_LAUNCHES];
};

struct qvq_p32_launch_plan {
  int launch_count;
  struct qvq_p32_launch_descriptor
      launches[QVQ_P32_LAUNCH_PLAN_MAX_LAUNCHES];
  // Eight-byte alignment covers every by-value field in the current P32 ABI.
  unsigned long long host_storage[QVQ_P32_LAUNCH_PLAN_HOST_STORAGE_WORDS];
};

// The ABI intentionally uses opaque pointers so framework adapters do not need
// to include CUDA headers. All pointers refer to device memory except `stream`,
// which is the caller-owned CUDA stream. For split_count >
// 1, partial_output must point to split_count * M * N float elements. Native
// reduction writes their sum to output; partials mode leaves the workspace for
// the surrounding graph to reduce. Kernel v11 also supports partials for M > 16:
// batched row tiles retain the global [split_count, M, N] workspace layout.
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

int qvq_p32_window_tuned(
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
    int row_groups,
    const struct qvq_p32_config* config,
    void* stream);

// Add a rank-8 (or packed rank-8) correction to an existing FP32 P32 output.
// `hidden` is row-major FP16 [M, rank_count], `rank8_b` is row-major FP32
// [rank_count, N], and `output` may alias `base_output`. rank_count is 8, 16,
// or 24 so grouped ZML projections can share one epilogue. The kernel keeps
// the QvQ/ZML arithmetic boundary: hidden is already rounded to FP16 and the
// correction is accumulated in FP32 before being added to the base output.
int qvq_p32_rank8_epilogue(
    const float* base_output,
    const void* hidden,
    const void* rank8_b,
    float* output,
    int size_m,
    int size_n,
    int rank_count,
    void* stream);

// Native rank-8 recovery projection. `input` is row-major FP32 [M,K],
// `rank8_a` is row-major FP16 [K,rank_count], and `hidden` is row-major FP16
// [M,rank_count]. The implementation accumulates in FP32 and rounds once to
// FP16, matching the framework recovery boundary. rank_count is 8, 16, or 24.
int qvq_p32_rank8_project(
    const void* input,
    const void* rank8_a,
    void* hidden,
    int size_m,
    int size_k,
    int rank_count,
    void* stream);

// Native grouped V2B2-P32 entry point. `bank_alt_ids` has one uint8 selector
// per group (two or three groups, with selectors in [1, 3]). M=1..4 uses the
// scalar kernel and M=5..16 uses the block kernel. The final output is row-major
// [M, size_n], while partial buffers use group-major contiguous storage: each
// group owns an [split, M, N_group] slice. `n_tile_end_0` and `n_tile_end_1` are
// cumulative N16 boundaries; the final boundary is size_n / 16.
// In PARTIALS mode no reducers are launched: only groups with split_count=1
// write their columns of output. Split groups write partial_output, packed in
// group order as [split_count, M, N_group], excluding unsplit groups. Other
// output columns and unused partial storage are untouched and must not be read.
// The caller owns both buffers until all compiler-managed consumers complete.
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

// Versioned configuration entry point for external compiler bridges. The
// legacy grouped function above remains source-compatible and uses native
// tuning policy. This entry point lets ZML/XLA supply the full launch policy
// in one object, including N-tile and warp choices, without depending on
// private CUDA templates.
int qvq_p32_grouped_window_tuned(
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
    int group_count,
    int n_tile_end_0,
    int n_tile_end_1,
    const struct qvq_p32_config* config,
    void* stream);

// Build the exact launch sequence used by qvq_p32_grouped_window without
// issuing work. ZML uses this to create/update native nodes in an XLA command
// buffer after autotuning has selected a fixed configuration. Other framework
// integrations can continue calling qvq_p32_grouped_window unchanged.
// PARTIALS mode returns exactly the main-product descriptor, with no reducers.
int qvq_p32_grouped_launch_plan(
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
    struct qvq_p32_launch_plan* plan);

int qvq_p32_grouped_launch_plan_tuned(
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
    int group_count,
    int n_tile_end_0,
    int n_tile_end_1,
    const struct qvq_p32_config* config,
    struct qvq_p32_launch_plan* plan);

#ifdef __cplusplus
}
#endif
