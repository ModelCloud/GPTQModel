// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "qvq_p32_abi.h"

using WindowFn = decltype(qvq_p32_window);
using WindowRowGroupsFn = decltype(qvq_p32_window_with_row_groups);
using WindowTunedFn = decltype(qvq_p32_window_tuned);
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
  QVQ_DECLARE_STANDARD_KIND(BITS, _large_m2_high);                        \
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
Function select_standard_shard(
    int transition_bits,
    int size_m,
    int kernel_variant,
    int threads,
    int row_groups,
    int stage_k_tiles,
    Function scalar_bits4,
    Function scalar_bits5,
    Function scalar_bits6,
    Function scalar_bits7,
    Function block_bits4,
    Function block_bits5,
    Function block_bits6,
    Function block_bits7,
    Function large_m2_low_bits4,
    Function large_m2_low_bits5,
    Function large_m2_low_bits6,
    Function large_m2_low_bits7,
    Function large_m2_high_bits4,
    Function large_m2_high_bits5,
    Function large_m2_high_bits6,
    Function large_m2_high_bits7,
    Function large_m_grid_low_bits4,
    Function large_m_grid_low_bits5,
    Function large_m_grid_low_bits6,
    Function large_m_grid_low_bits7,
    Function large_m_grid_high_bits4,
    Function large_m_grid_high_bits5,
    Function large_m_grid_high_bits6,
    Function large_m_grid_high_bits7) {
  if (size_m > 16) {
    const bool use_large_m2 = threads == 128 && size_m % 32 == 0 &&
        row_groups != 1;
    if (use_large_m2) {
      if (stage_k_tiles >= 3) {
        return select_shard(
            transition_bits, large_m2_high_bits4, large_m2_high_bits5,
            large_m2_high_bits6, large_m2_high_bits7);
      }
      return select_shard(
          transition_bits, large_m2_low_bits4, large_m2_low_bits5,
          large_m2_low_bits6, large_m2_low_bits7);
    }
    if (stage_k_tiles >= 3) {
      return select_shard(
          transition_bits, large_m_grid_high_bits4, large_m_grid_high_bits5,
          large_m_grid_high_bits6, large_m_grid_high_bits7);
    }
    return select_shard(
        transition_bits, large_m_grid_low_bits4, large_m_grid_low_bits5,
        large_m_grid_low_bits6, large_m_grid_low_bits7);
  }
  if (kernel_variant == QVQ_P32_VARIANT_BLOCK) {
    return select_shard(
        transition_bits, block_bits4, block_bits5,
        block_bits6, block_bits7);
  }
  return select_shard(
      transition_bits, scalar_bits4, scalar_bits5,
      scalar_bits6, scalar_bits7);
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
  return select_standard_shard(
      transition_bits, size_m, kernel_variant, threads,
      QVQ_P32_ROW_GROUPS_AUTO, stage_k_tiles,
      qvq_p32_window_bits4_scalar, qvq_p32_window_bits5_scalar,
      qvq_p32_window_bits6_scalar, qvq_p32_window_bits7_scalar,
      qvq_p32_window_bits4_block, qvq_p32_window_bits5_block,
      qvq_p32_window_bits6_block, qvq_p32_window_bits7_block,
      qvq_p32_window_bits4_large_m2_low,
      qvq_p32_window_bits5_large_m2_low,
      qvq_p32_window_bits6_large_m2_low,
      qvq_p32_window_bits7_large_m2_low,
      qvq_p32_window_bits4_large_m2_high,
      qvq_p32_window_bits5_large_m2_high,
      qvq_p32_window_bits6_large_m2_high,
      qvq_p32_window_bits7_large_m2_high,
      qvq_p32_window_bits4_large_m_grid_low,
      qvq_p32_window_bits5_large_m_grid_low,
      qvq_p32_window_bits6_large_m_grid_low,
      qvq_p32_window_bits7_large_m_grid_low,
      qvq_p32_window_bits4_large_m_grid_high,
      qvq_p32_window_bits5_large_m_grid_high,
      qvq_p32_window_bits6_large_m_grid_high,
      qvq_p32_window_bits7_large_m_grid_high)(
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
  return select_standard_shard(
      transition_bits, size_m, kernel_variant, threads, row_groups,
      stage_k_tiles,
      qvq_p32_window_with_row_groups_bits4_scalar,
      qvq_p32_window_with_row_groups_bits5_scalar,
      qvq_p32_window_with_row_groups_bits6_scalar,
      qvq_p32_window_with_row_groups_bits7_scalar,
      qvq_p32_window_with_row_groups_bits4_block,
      qvq_p32_window_with_row_groups_bits5_block,
      qvq_p32_window_with_row_groups_bits6_block,
      qvq_p32_window_with_row_groups_bits7_block,
      qvq_p32_window_with_row_groups_bits4_large_m2_low,
      qvq_p32_window_with_row_groups_bits5_large_m2_low,
      qvq_p32_window_with_row_groups_bits6_large_m2_low,
      qvq_p32_window_with_row_groups_bits7_large_m2_low,
      qvq_p32_window_with_row_groups_bits4_large_m2_high,
      qvq_p32_window_with_row_groups_bits5_large_m2_high,
      qvq_p32_window_with_row_groups_bits6_large_m2_high,
      qvq_p32_window_with_row_groups_bits7_large_m2_high,
      qvq_p32_window_with_row_groups_bits4_large_m_grid_low,
      qvq_p32_window_with_row_groups_bits5_large_m_grid_low,
      qvq_p32_window_with_row_groups_bits6_large_m_grid_low,
      qvq_p32_window_with_row_groups_bits7_large_m_grid_low,
      qvq_p32_window_with_row_groups_bits4_large_m_grid_high,
      qvq_p32_window_with_row_groups_bits5_large_m_grid_high,
      qvq_p32_window_with_row_groups_bits6_large_m_grid_high,
      qvq_p32_window_with_row_groups_bits7_large_m_grid_high)(
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
  return select_standard_shard(
      transition_bits, size_m, kernel_variant, threads, row_groups,
      stage_k_tiles,
      qvq_p32_window_tuned_bits4_scalar,
      qvq_p32_window_tuned_bits5_scalar,
      qvq_p32_window_tuned_bits6_scalar,
      qvq_p32_window_tuned_bits7_scalar,
      qvq_p32_window_tuned_bits4_block,
      qvq_p32_window_tuned_bits5_block,
      qvq_p32_window_tuned_bits6_block,
      qvq_p32_window_tuned_bits7_block,
      qvq_p32_window_tuned_bits4_large_m2_low,
      qvq_p32_window_tuned_bits5_large_m2_low,
      qvq_p32_window_tuned_bits6_large_m2_low,
      qvq_p32_window_tuned_bits7_large_m2_low,
      qvq_p32_window_tuned_bits4_large_m2_high,
      qvq_p32_window_tuned_bits5_large_m2_high,
      qvq_p32_window_tuned_bits6_large_m2_high,
      qvq_p32_window_tuned_bits7_large_m2_high,
      qvq_p32_window_tuned_bits4_large_m_grid_low,
      qvq_p32_window_tuned_bits5_large_m_grid_low,
      qvq_p32_window_tuned_bits6_large_m_grid_low,
      qvq_p32_window_tuned_bits7_large_m_grid_low,
      qvq_p32_window_tuned_bits4_large_m_grid_high,
      qvq_p32_window_tuned_bits5_large_m_grid_high,
      qvq_p32_window_tuned_bits6_large_m_grid_high,
      qvq_p32_window_tuned_bits7_large_m_grid_high)(
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
