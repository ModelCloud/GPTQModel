// Large prefills share one 2-D grid across all 16-row tiles.  This keeps the
// ABI's global [split, M, N] workspace layout while avoiding one host launch
// (and one reduction) per row chunk.
template <
    int TransitionBits,
    int Threads,
    int StageKTiles,
    int StaticN = 0,
    int StaticK = 0>
__global__ __launch_bounds__(Threads) void p32_window_ampere_large_m_kernel(
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
  const int row_offset = static_cast<int>(blockIdx.y) * kRows;
  const int local_m = min(kRows, size_m - row_offset);
  if (local_m <= 0) return;
  const int n_tiles = size_n / kTileColumns;
  const int n_block = static_cast<int>(blockIdx.x);
  const int split = static_cast<int>(blockIdx.z);
#define QVQ_LARGE_M_BODY(FULL_ROWS) \
  p32_window_ampere_kernel_body< \
      TransitionBits, FULL_ROWS, 0, StaticN, Threads, Threads / 32, \
      StageKTiles, false, false, StaticK>( \
      input + static_cast<int64_t>(row_offset) * size_k, \
      trellis, levels, bank_ids, partial_output, \
      output + static_cast<int64_t>(row_offset) * size_n, local_m, size_k, \
      size_n, split_count, bank_alt_id, n_block, split, n_tiles, 0, 0, \
      size_n, static_cast<int64_t>(row_offset) * size_n, \
      static_cast<int64_t>(size_m) * size_n)
  if (local_m == kRows) QVQ_LARGE_M_BODY(true);
  else QVQ_LARGE_M_BODY(false);
#undef QVQ_LARGE_M_BODY
}
