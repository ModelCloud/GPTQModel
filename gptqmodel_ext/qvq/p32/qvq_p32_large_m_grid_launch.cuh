template <int TransitionBits, int Threads, int StageKTiles, int StaticN,
          int StaticK = 0>
int launch_p32_large_m_grid(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    const uint8_t* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    cudaStream_t stream) {
  constexpr int tiles_per_block = Threads / 32;
  const int n_tiles = size_n / kTileColumns;
  const dim3 grid(
      static_cast<unsigned>((n_tiles + tiles_per_block - 1) / tiles_per_block),
      static_cast<unsigned>((size_m + kRows - 1) / kRows),
      static_cast<unsigned>(split_count));
  p32_window_ampere_large_m_kernel<
      TransitionBits, Threads, StageKTiles, StaticN, StaticK>
      <<<grid, Threads, 0, stream>>>(
      input, trellis, levels, bank_ids, partial_output, output, size_m, size_k,
      size_n, split_count, bank_alt_id);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    set_last_error(cudaGetErrorString(error));
    return static_cast<int>(error);
  }
  return 0;
}

template <int TransitionBits, int Threads, int StageKTiles>
int launch_p32_large_m_grid_dispatch(
    const half* input,
    const uint32_t* trellis,
    const half* levels,
    const uint8_t* bank_ids,
    const uint8_t* bank_alt_id,
    float* output,
    float* partial_output,
    int size_m,
    int size_k,
    int size_n,
    int split_count,
    bool static_n,
    cudaStream_t stream) {
  if (static_n && size_k == 5120) {
    switch (size_n) {
#define QVQ_LARGE_M_STATIC_N(N) \
      case N: \
        return launch_p32_large_m_grid< \
            TransitionBits, Threads, StageKTiles, N, 5120>( \
            input, trellis, levels, bank_ids, bank_alt_id, output, \
            partial_output, size_m, size_k, size_n, split_count, stream)
      QVQ_LARGE_M_STATIC_N(512);
      QVQ_LARGE_M_STATIC_N(2048);
      QVQ_LARGE_M_STATIC_N(8192);
      QVQ_LARGE_M_STATIC_N(1024);
      QVQ_LARGE_M_STATIC_N(5120);
      QVQ_LARGE_M_STATIC_N(6144);
      QVQ_LARGE_M_STATIC_N(10240);
      QVQ_LARGE_M_STATIC_N(12288);
      QVQ_LARGE_M_STATIC_N(17408);
#undef QVQ_LARGE_M_STATIC_N
      default:
        set_last_error("QVQ P32 large-M static_n does not support this N");
        return -1;
    }
  }
  if (static_n) {
    switch (size_n) {
#define QVQ_LARGE_M_STATIC_N(N) \
      case N: \
        return launch_p32_large_m_grid< \
            TransitionBits, Threads, StageKTiles, N, 0>( \
            input, trellis, levels, bank_ids, bank_alt_id, output, \
            partial_output, size_m, size_k, size_n, split_count, stream)
      QVQ_LARGE_M_STATIC_N(512);
      QVQ_LARGE_M_STATIC_N(2048);
      QVQ_LARGE_M_STATIC_N(8192);
      QVQ_LARGE_M_STATIC_N(1024);
      QVQ_LARGE_M_STATIC_N(5120);
      QVQ_LARGE_M_STATIC_N(6144);
      QVQ_LARGE_M_STATIC_N(10240);
      QVQ_LARGE_M_STATIC_N(12288);
      QVQ_LARGE_M_STATIC_N(17408);
#undef QVQ_LARGE_M_STATIC_N
      default:
        set_last_error("QVQ P32 large-M static_n does not support this N");
        return -1;
    }
  }
  return launch_p32_large_m_grid<
      TransitionBits, Threads, StageKTiles, 0>(
      input, trellis, levels, bank_ids, bank_alt_id, output, partial_output,
      size_m, size_k, size_n, split_count, stream);
}
