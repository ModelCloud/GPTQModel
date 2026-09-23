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
#define QVQ_DECLARE_SHARD(BITS)                                            \
  WindowFn qvq_p32_window_bits##BITS;                                     \
  WindowRowGroupsFn qvq_p32_window_with_row_groups_bits##BITS;             \
  WindowTunedFn qvq_p32_window_tuned_bits##BITS;                           \
  GroupedWindowFn qvq_p32_grouped_window_bits##BITS;                       \
  GroupedWindowTunedFn qvq_p32_grouped_window_tuned_bits##BITS;             \
  GroupedPlanFn qvq_p32_grouped_launch_plan_bits##BITS;                     \
  GroupedPlanTunedFn qvq_p32_grouped_launch_plan_tuned_bits##BITS
QVQ_DECLARE_SHARD(4);
QVQ_DECLARE_SHARD(5);
QVQ_DECLARE_SHARD(6);
QVQ_DECLARE_SHARD(7);
#undef QVQ_DECLARE_SHARD
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
  return select_shard(
      transition_bits, qvq_p32_window_bits4, qvq_p32_window_bits5,
      qvq_p32_window_bits6, qvq_p32_window_bits7)(
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
  return select_shard(
      transition_bits, qvq_p32_window_with_row_groups_bits4,
      qvq_p32_window_with_row_groups_bits5,
      qvq_p32_window_with_row_groups_bits6,
      qvq_p32_window_with_row_groups_bits7)(
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
  return select_shard(
      transition_bits, qvq_p32_window_tuned_bits4,
      qvq_p32_window_tuned_bits5, qvq_p32_window_tuned_bits6,
      qvq_p32_window_tuned_bits7)(
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
