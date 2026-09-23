// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "qvq_p32_abi.h"

using WindowFn = decltype(qvq_p32_window);
using WindowRowGroupsFn = decltype(qvq_p32_window_with_row_groups);
using WindowTunedFn = decltype(qvq_p32_window_tuned);
using WindowFnPtr = decltype(&qvq_p32_window);
using WindowRowGroupsFnPtr = decltype(&qvq_p32_window_with_row_groups);
using WindowTunedFnPtr = decltype(&qvq_p32_window_tuned);
using GroupedWindowFn = decltype(qvq_p32_grouped_window);
using GroupedWindowTunedFn = decltype(qvq_p32_grouped_window_tuned);
using GroupedPlanFn = decltype(qvq_p32_grouped_launch_plan);
using GroupedPlanTunedFn = decltype(qvq_p32_grouped_launch_plan_tuned);

extern "C" {
#define QVQ_DECLARE_STANDARD_KIND(BITS, KIND)                              \
  WindowFn qvq_p32_window_bits##BITS##KIND;                                \
  WindowRowGroupsFn qvq_p32_window_with_row_groups_bits##BITS##KIND;       \
  WindowTunedFn qvq_p32_window_tuned_bits##BITS##KIND
#define QVQ_DECLARE_STANDARD(BITS)                                        \
  QVQ_DECLARE_STANDARD_KIND(BITS, _scalar);                               \
  QVQ_DECLARE_STANDARD_KIND(BITS, _block);                                \
  QVQ_DECLARE_STANDARD_KIND(BITS, _large_m2_low);                         \
  QVQ_DECLARE_STANDARD_KIND(BITS, _large_m2_stage3);                      \
  QVQ_DECLARE_STANDARD_KIND(BITS, _large_m2_stage4);                      \
  QVQ_DECLARE_STANDARD_KIND(BITS, _large_m_grid_low);                     \
  QVQ_DECLARE_STANDARD_KIND(BITS, _large_m_grid_high)
#define QVQ_DECLARE_GROUPED(BITS)                                         \
  GroupedWindowFn qvq_p32_grouped_window_bits##BITS;                      \
  GroupedWindowTunedFn qvq_p32_grouped_window_tuned_bits##BITS;            \
  GroupedPlanFn qvq_p32_grouped_launch_plan_bits##BITS;                    \
  GroupedPlanTunedFn qvq_p32_grouped_launch_plan_tuned_bits##BITS
QVQ_DECLARE_STANDARD(4);
QVQ_DECLARE_STANDARD(5);
QVQ_DECLARE_STANDARD(6);
QVQ_DECLARE_STANDARD(7);
QVQ_DECLARE_GROUPED(4);
QVQ_DECLARE_GROUPED(5);
QVQ_DECLARE_GROUPED(6);
QVQ_DECLARE_GROUPED(7);
#undef QVQ_DECLARE_GROUPED
#undef QVQ_DECLARE_STANDARD
#undef QVQ_DECLARE_STANDARD_KIND
}

#define QVQ_STANDARD_SHARD_TABLES(BASE, FUNCTION_PTR)                       \
  static constexpr FUNCTION_PTR scalar_shards[4] = {                      \
      BASE##_bits4_scalar, BASE##_bits5_scalar,                            \
      BASE##_bits6_scalar, BASE##_bits7_scalar};                           \
  static constexpr FUNCTION_PTR block_shards[4] = {                       \
      BASE##_bits4_block, BASE##_bits5_block,                              \
      BASE##_bits6_block, BASE##_bits7_block};                             \
  static constexpr FUNCTION_PTR large_m2_shards[4][4] = {                  \
      {BASE##_bits4_large_m2_low, BASE##_bits5_large_m2_low,              \
       BASE##_bits6_large_m2_low, BASE##_bits7_large_m2_low},              \
      {BASE##_bits4_large_m2_low, BASE##_bits5_large_m2_low,              \
       BASE##_bits6_large_m2_low, BASE##_bits7_large_m2_low},              \
      {BASE##_bits4_large_m2_stage3, BASE##_bits5_large_m2_stage3,         \
       BASE##_bits6_large_m2_stage3, BASE##_bits7_large_m2_stage3},        \
      {BASE##_bits4_large_m2_stage4, BASE##_bits5_large_m2_stage4,         \
       BASE##_bits6_large_m2_stage4, BASE##_bits7_large_m2_stage4}};       \
  static constexpr FUNCTION_PTR large_m_grid_low_shards[4] = {            \
      BASE##_bits4_large_m_grid_low, BASE##_bits5_large_m_grid_low,       \
      BASE##_bits6_large_m_grid_low, BASE##_bits7_large_m_grid_low};      \
  static constexpr FUNCTION_PTR large_m_grid_high_shards[4] = {           \
      BASE##_bits4_large_m_grid_high, BASE##_bits5_large_m_grid_high,     \
      BASE##_bits6_large_m_grid_high, BASE##_bits7_large_m_grid_high}

namespace {

template <typename Function>
Function select_shard(
    int transition_bits,
    Function bits4,
    Function bits5,
    Function bits6,
    Function bits7) {
  switch (transition_bits) {
    case 5: return bits5;
    case 6: return bits6;
    case 7: return bits7;
    case 4:
    default: return bits4;
  }
}

template <typename Function>
Function select_shard(int transition_bits, const Function* shards) {
  switch (transition_bits) {
    case 5: return shards[1];
    case 6: return shards[2];
    case 7: return shards[3];
    case 4:
    default: return shards[0];
  }
}

template <typename Function>
Function select_standard_shard(
    int transition_bits,
    int size_m,
    int kernel_variant,
    int threads,
    int row_groups,
    int stage_k_tiles,
    const Function* scalar_shards,
    const Function* block_shards,
    const Function (*large_m2_shards)[4],
    const Function* large_m_grid_low_shards,
    const Function* large_m_grid_high_shards) {
  if (size_m > 16) {
    const bool use_large_m2 = threads == 128 && size_m % 32 == 0 &&
        row_groups != 1;
    const int stage_index = stage_k_tiles >= 1 && stage_k_tiles <= 4
        ? stage_k_tiles - 1
        : 0;
    if (use_large_m2) {
      return select_shard(transition_bits, large_m2_shards[stage_index]);
    }
    return select_shard(
        transition_bits,
        stage_k_tiles >= 3 ? large_m_grid_high_shards
                           : large_m_grid_low_shards);
  }
  if (kernel_variant == QVQ_P32_VARIANT_BLOCK) {
    return select_shard(transition_bits, block_shards);
  }
  return select_shard(transition_bits, scalar_shards);
}

int resolve_requested_threads(const qvq_p32_config* requested) {
  if (requested == nullptr) return 0;
  if (requested->n_warps != QVQ_P32_WARPS_AUTO) {
    return requested->n_warps * 32;
  }
  return requested->threads == 0 ? 128 : requested->threads;
}

int resolve_requested_stage(const qvq_p32_config* requested) {
  return requested == nullptr ? 0 : requested->stage_k_tiles;
}

}  // namespace

extern "C" int qvq_p32_window(
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
    void* stream) {
  QVQ_STANDARD_SHARD_TABLES(qvq_p32_window, WindowFnPtr);
  return select_standard_shard(
      transition_bits, size_m, kernel_variant, threads,
      QVQ_P32_ROW_GROUPS_AUTO, stage_k_tiles,
      scalar_shards, block_shards, large_m2_shards,
      large_m_grid_low_shards, large_m_grid_high_shards)(
      input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,
      size_m, size_k, size_n, transition_bits, split_count, kernel_variant,
      threads, stage_k_tiles, static_n, reduction_mode, stream);
}

extern "C" int qvq_p32_window_with_row_groups(
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
    void* stream) {
  QVQ_STANDARD_SHARD_TABLES(
      qvq_p32_window_with_row_groups, WindowRowGroupsFnPtr);
  return select_standard_shard(
      transition_bits, size_m, kernel_variant, threads, row_groups,
      stage_k_tiles,
      scalar_shards, block_shards, large_m2_shards,
      large_m_grid_low_shards, large_m_grid_high_shards)(
      input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,
      size_m, size_k, size_n, transition_bits, split_count, kernel_variant,
      threads, stage_k_tiles, static_n, reduction_mode, row_groups, stream);
}

extern "C" int qvq_p32_window_tuned(
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
    const qvq_p32_config* requested,
    void* stream) {
  const int kernel_variant = requested == nullptr
      ? QVQ_P32_VARIANT_SCALAR
      : requested->kernel_variant;
  const int threads = resolve_requested_threads(requested);
  const int stage_k_tiles = resolve_requested_stage(requested);
  QVQ_STANDARD_SHARD_TABLES(qvq_p32_window_tuned, WindowTunedFnPtr);
  return select_standard_shard(
      transition_bits, size_m, kernel_variant, threads, row_groups,
      stage_k_tiles,
      scalar_shards, block_shards, large_m2_shards,
      large_m_grid_low_shards, large_m_grid_high_shards)(
      input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,
      size_m, size_k, size_n, transition_bits, row_groups, requested, stream);
}

extern "C" int qvq_p32_grouped_window(
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
    void* stream) {
  return select_shard(
      transition_bits, qvq_p32_grouped_window_bits4,
      qvq_p32_grouped_window_bits5, qvq_p32_grouped_window_bits6,
      qvq_p32_grouped_window_bits7)(
      input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
      size_m, size_k, size_n, transition_bits, split_count, split_count_0,
      split_count_1, split_count_2, kernel_variant, threads, stage_k_tiles,
      static_n, reduction_mode, group_count, n_tile_end_0, n_tile_end_1,
      stream);
}

extern "C" int qvq_p32_grouped_window_tuned(
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
    const qvq_p32_config* requested,
    void* stream) {
  return select_shard(
      transition_bits, qvq_p32_grouped_window_tuned_bits4,
      qvq_p32_grouped_window_tuned_bits5,
      qvq_p32_grouped_window_tuned_bits6,
      qvq_p32_grouped_window_tuned_bits7)(
      input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
      size_m, size_k, size_n, transition_bits, split_count, split_count_0,
      split_count_1, split_count_2, group_count, n_tile_end_0, n_tile_end_1,
      requested, stream);
}

extern "C" int qvq_p32_grouped_launch_plan(
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
    qvq_p32_launch_plan* plan) {
  return select_shard(
      transition_bits, qvq_p32_grouped_launch_plan_bits4,
      qvq_p32_grouped_launch_plan_bits5,
      qvq_p32_grouped_launch_plan_bits6,
      qvq_p32_grouped_launch_plan_bits7)(
      input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
      size_m, size_k, size_n, transition_bits, split_count, split_count_0,
      split_count_1, split_count_2, kernel_variant, threads, stage_k_tiles,
      static_n, reduction_mode, group_count, n_tile_end_0, n_tile_end_1,
      plan);
}

extern "C" int qvq_p32_grouped_launch_plan_tuned(
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
    const qvq_p32_config* requested,
    qvq_p32_launch_plan* plan) {
  return select_shard(
      transition_bits, qvq_p32_grouped_launch_plan_tuned_bits4,
      qvq_p32_grouped_launch_plan_tuned_bits5,
      qvq_p32_grouped_launch_plan_tuned_bits6,
      qvq_p32_grouped_launch_plan_tuned_bits7)(
      input, trellis, levels, bank_ids, bank_alt_ids, output, partial_output,
      size_m, size_k, size_n, transition_bits, split_count, split_count_0,
      split_count_1, split_count_2, group_count, n_tile_end_0, n_tile_end_1,
      requested, plan);
}
