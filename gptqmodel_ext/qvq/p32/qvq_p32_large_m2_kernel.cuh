// Keep the M2 kernel body in its own input so device-loop edits rebuild only
// the M2 shards. Reuse one packed N-tile set across two adjacent 16-row groups.
// Each warp keeps two accumulator fragments, so the second row group avoids repeating
// trellis loads and state decoding while retaining the four-warp N layout.
template <int TransitionBits, int StaticN, int StaticK = 0,
          int StageKTiles = 1, int RowGroups = 2>
__global__ __launch_bounds__(128) void p32_window_ampere_large_m2_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const half* __restrict__ levels,
    const uint8_t* __restrict__ bank_ids,
    float* __restrict__ partial_output,
    float* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    const uint8_t* __restrict__ bank_alt_id) {
  const int row_offset = static_cast<int>(blockIdx.y) * (RowGroups * kRows);
  const int local_m = min(RowGroups * kRows, size_m - row_offset);
  if (local_m < RowGroups * kRows) return;
  const int n_tiles = size_n / kTileColumns;
  const int n_block = static_cast<int>(blockIdx.x);
  const int split = static_cast<int>(blockIdx.z);
  p32_window_ampere_kernel_body<
      TransitionBits, true, 0, StaticN, 128, 4, StageKTiles,
      true, false, StaticK, RowGroups,
      (RowGroups == 16 && StageKTiles >= 3)>(
      input + static_cast<int64_t>(row_offset) * size_k,
      trellis, levels, bank_ids, partial_output,
      output + static_cast<int64_t>(row_offset) * size_n, local_m, size_k,
      size_n, split_count, bank_alt_id, n_block, split, n_tiles, 0, 0,
      size_n, static_cast<int64_t>(row_offset) * size_n,
      static_cast<int64_t>(size_m) * size_n);
}
