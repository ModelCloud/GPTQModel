// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kRows = 16;
constexpr int kTileRows = 16;
constexpr int kTileColumns = 16;
constexpr int kTilesPerBlock = kWarps;
constexpr int kPairsPerTile = 128;
constexpr int kLevels = 256;
constexpr uint32_t kPgc16Multiplier = 40503u;
constexpr uint32_t kPgc16Increment = 17011u;

__device__ __forceinline__ uint32_t pgc16_mix(uint32_t state) {
  uint32_t mixed = state ^ (state >> 8);
  mixed = (mixed * kPgc16Multiplier + kPgc16Increment) & 0xffffu;
  return mixed ^ (mixed >> 7);
}

template <int TransitionBits>
__device__ __forceinline__ uint32_t alternate_bank_mask(int bank_alt_id) {
  static_assert(TransitionBits >= 4 && TransitionBits <= 7);
  if (bank_alt_id == 0) {
    return 0u;
  }
  if constexpr (TransitionBits == 4) {
    return bank_alt_id == 1 ? 0x5a5au : bank_alt_id == 2 ? 0x3c3cu : 0xc3c3u;
  } else if constexpr (TransitionBits == 5) {
    return bank_alt_id == 1 ? 0x9696u : bank_alt_id == 2 ? 0x3c3cu : 0xc3c3u;
  } else if constexpr (TransitionBits == 6) {
    return bank_alt_id == 1 ? 0x6969u : bank_alt_id == 2 ? 0x5a5au : 0x3c3cu;
  } else {
    return bank_alt_id == 1 ? 0xc3c3u : bank_alt_id == 2 ? 0x9696u : 0x5a5au;
  }
}

// Window states pair naturally at a distance of 64 pairs: both use the same
// funnel-shift amount and their first words are exactly 2 * TransitionBits
// words apart. This is the storage-neutral paired extraction used by the
// Hopper kernel, expressed here with ordinary Ampere integer instructions.
template <int TransitionBits>
__device__ __forceinline__ void window_state_pair64(
    const uint32_t* __restrict__ words,
    int pair,
    uint32_t& first,
    uint32_t& second) {
  constexpr int kWordsPerTile = 4 * TransitionBits;
  constexpr int kPairWordDistance = 2 * TransitionBits;
  const int bit_position = (kPairsPerTile - 1 - pair) * TransitionBits;
  const int first_word = bit_position >> 5;
  const int shift = bit_position & 31;
  const int first_next = first_word + 1 == kWordsPerTile ? 0 : first_word + 1;
  const int second_word = first_word - kPairWordDistance;
  const uint64_t first_window = static_cast<uint64_t>(words[first_word]) |
      (static_cast<uint64_t>(words[first_next]) << 32);
  const uint64_t second_window = static_cast<uint64_t>(words[second_word]) |
      (static_cast<uint64_t>(words[second_word + 1]) << 32);
  first = static_cast<uint32_t>((first_window >> shift) & 0xffffu);
  second = static_cast<uint32_t>((second_window >> shift) & 0xffffu);
}

template <int TransitionBits>
__device__ __forceinline__ void decode_pair(
    half* __restrict__ weight,
    int pair,
    uint32_t state,
    uint8_t packed_bank_id,
    uint32_t alt_mask,
    const half* __restrict__ levels) {
  const uint32_t bank_mask =
      ((static_cast<uint32_t>(packed_bank_id) >> (pair >> 4)) & 1u) * alt_mask;
  const uint32_t mixed = pgc16_mix(state ^ bank_mask);
  weight[pair * 2] = levels[mixed >> 8];
  weight[pair * 2 + 1] = levels[mixed & 0xffu];
}

template <int TransitionBits, bool FullRows>
__global__ __launch_bounds__(kThreads) void p32_window_ampere_kernel(
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
    int bank_alt_id) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  constexpr int kWordsPerTile = 4 * TransitionBits;
  __shared__ __align__(32) half input_tile[2][kRows * kTileRows];
  __shared__ __align__(16) uint32_t packed_words[2][kTilesPerBlock][kWordsPerTile];
  __shared__ uint8_t packed_bank_ids[2][kTilesPerBlock];
  __shared__ __align__(32) half decoded_weight[kTilesPerBlock][kTileRows * kTileColumns];
  __shared__ half cached_levels[kLevels];
  __shared__ __align__(32) float output_tile[kTilesPerBlock][kRows * kTileColumns];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int n_tiles = size_n / kTileColumns;
  const int n_tile_base = static_cast<int>(blockIdx.x) * kTilesPerBlock;
  const bool active_tile = n_tile_base + warp < n_tiles;
  const int split = static_cast<int>(blockIdx.z);
  const int k_tiles = size_k / kTileRows;
  const int k_tile_begin = (k_tiles * split) / split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / split_count;
  const uint32_t alt_mask = alternate_bank_mask<TransitionBits>(bank_alt_id);

  for (int index = thread; index < kLevels; index += kThreads) {
    cached_levels[index] = levels[index];
  }

  auto stage = [&](int k_tile, int destination) {
    auto* input_vectors = reinterpret_cast<uint4*>(input_tile[destination]);
    for (int index = thread; index < kRows * kTileRows / 8; index += kThreads) {
      const int row = index / (kTileRows / 8);
      const int vector = index - row * (kTileRows / 8);
      if constexpr (FullRows) {
        const half* source = input + static_cast<int64_t>(row) * size_k +
            k_tile * kTileRows + vector * 8;
        __pipeline_memcpy_async(input_vectors + index, reinterpret_cast<const uint4*>(source), 16);
      } else if (row < size_m) {
        const half* source = input + static_cast<int64_t>(row) * size_k +
            k_tile * kTileRows + vector * 8;
        __pipeline_memcpy_async(input_vectors + index, reinterpret_cast<const uint4*>(source), 16);
      } else {
        input_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
      }
    }

    constexpr int kVectorsPerTile = kWordsPerTile / 4;
    constexpr int kVectorsPerBlock = kTilesPerBlock * kVectorsPerTile;
    auto* destination_vectors = reinterpret_cast<uint4*>(packed_words[destination]);
    for (int index = thread; index < kVectorsPerBlock; index += kThreads) {
      const int tile = index / kVectorsPerTile;
      const int vector = index - tile * kVectorsPerTile;
      const int n_tile = n_tile_base + tile;
      if (n_tile < n_tiles) {
        const int64_t global_tile = static_cast<int64_t>(k_tile) * n_tiles + n_tile;
        const auto* source_vectors = reinterpret_cast<const uint4*>(
            trellis + global_tile * kWordsPerTile);
        __pipeline_memcpy_async(destination_vectors + index, source_vectors + vector, 16);
      } else {
        destination_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    if (thread < kTilesPerBlock) {
      const int n_tile = n_tile_base + thread;
      packed_bank_ids[destination][thread] = n_tile < n_tiles
          ? bank_ids[static_cast<int64_t>(k_tile) * n_tiles + n_tile]
          : 0;
    }
    __pipeline_commit();
  };

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> input_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> weight_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  stage(k_tile_begin, 0);
  int parity = 0;
  for (int k_tile = k_tile_begin; k_tile < k_tile_end; ++k_tile) {
    const bool has_next = k_tile + 1 < k_tile_end;
    if (has_next) {
      stage(k_tile + 1, parity ^ 1);
    }
    __pipeline_wait_prior(has_next ? 1 : 0);
    __syncthreads();

    if (active_tile) {
      half* weight = decoded_weight[warp];
      const uint32_t* words = packed_words[parity][warp];
      const uint8_t packed_bank_id = packed_bank_ids[parity][warp];
#pragma unroll
      for (int pair_base = 0; pair_base < 64; pair_base += 32) {
        const int first_pair = lane + pair_base;
        uint32_t first_state;
        uint32_t second_state;
        window_state_pair64<TransitionBits>(words, first_pair, first_state, second_state);
        decode_pair<TransitionBits>(
            weight, first_pair, first_state, packed_bank_id, alt_mask, cached_levels);
        decode_pair<TransitionBits>(
            weight, first_pair + 64, second_state, packed_bank_id, alt_mask, cached_levels);
      }
      __syncwarp();
      wmma::load_matrix_sync(input_fragment, input_tile[parity], kTileRows);
      wmma::load_matrix_sync(weight_fragment, weight, kTileColumns);
      wmma::mma_sync(accumulator, input_fragment, weight_fragment, accumulator);
    }
    // Every warp must finish consuming the current stage before any lane can
    // enter the next iteration and cp.async-overwrite that buffer two stages
    // later. Independent thread scheduling makes warp-local completion alone
    // insufficient for this block-wide producer/consumer handoff.
    __syncthreads();
    parity ^= 1;
  }

  float* target = split_count == 1
      ? output
      : partial_output + static_cast<int64_t>(split) * size_m * size_n;
  if constexpr (FullRows) {
    if (active_tile) {
      wmma::store_matrix_sync(
          target + n_tile_base * kTileColumns + warp * kTileColumns,
          accumulator,
          size_n,
          wmma::mem_row_major);
    }
  } else {
    if (active_tile) {
      wmma::store_matrix_sync(
          output_tile[warp], accumulator, kTileColumns, wmma::mem_row_major);
    }
    __syncthreads();
    const int active_tiles = min(kTilesPerBlock, n_tiles - n_tile_base);
    const int output_values = active_tiles * size_m * kTileColumns;
    for (int index = thread; index < output_values; index += kThreads) {
      const int tile = index / (size_m * kTileColumns);
      const int tile_index = index - tile * size_m * kTileColumns;
      const int row = tile_index / kTileColumns;
      const int column = tile_index - row * kTileColumns;
      target[static_cast<int64_t>(row) * size_n +
             (n_tile_base + tile) * kTileColumns + column] =
          output_tile[tile][row * kTileColumns + column];
    }
  }
#endif
}

__global__ void reduce_split_kernel(
    const float* __restrict__ partial_output,
    float* __restrict__ output,
    int output_values,
    int split_count) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= output_values) {
    return;
  }
  float accumulator = 0.0f;
  for (int split = 0; split < split_count; ++split) {
    accumulator += partial_output[static_cast<int64_t>(split) * output_values + index];
  }
  output[index] = accumulator;
}

template <int TransitionBits>
at::Tensor p32_window_ampere_impl(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t out_features,
    int64_t bank_alt_id,
    int64_t split_count) {
  constexpr int kWordsPerTile = 4 * TransitionBits;
  TORCH_CHECK(input.is_cuda(), "QVQ P32 Ampere input must be CUDA");
  TORCH_CHECK(
      trellis.device() == input.device() && levels.device() == input.device() &&
          bank_ids.device() == input.device(),
      "QVQ P32 Ampere tensors must share one CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf && levels.scalar_type() == at::kHalf,
      "QVQ P32 Ampere input and levels must be float16");
  TORCH_CHECK(trellis.scalar_type() == at::kInt, "QVQ P32 Ampere trellis must be int32");
  TORCH_CHECK(bank_ids.scalar_type() == at::kByte, "QVQ P32 Ampere bank ids must be uint8");
  TORCH_CHECK(
      input.is_contiguous() && trellis.is_contiguous() && levels.is_contiguous() &&
          bank_ids.is_contiguous(),
      "QVQ P32 Ampere tensors must be contiguous");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= kRows,
      "QVQ P32 Ampere input must have between 1 and 16 rows");
  TORCH_CHECK(
      input.size(1) > 0 && input.size(1) % kTileRows == 0,
      "QVQ P32 Ampere K must be positive and divisible by 16");
  TORCH_CHECK(
      out_features > 0 && out_features % kTileColumns == 0,
      "QVQ P32 Ampere N must be positive and divisible by 16");
  TORCH_CHECK(split_count >= 1 && split_count <= 8, "QVQ P32 Ampere split count must be in [1, 8]");
  TORCH_CHECK(bank_alt_id >= 0 && bank_alt_id <= 3, "QVQ P32 Ampere bank ID must be in [0, 3]");

  const c10::cuda::CUDAGuard device_guard(input.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(
      properties.major == 8 && properties.minor == 0,
      "QVQ P32 Ampere WMMA requires compute capability 8.0, got ",
      properties.major,
      ".",
      properties.minor);

  const int size_m = static_cast<int>(input.size(0));
  const int size_k = static_cast<int>(input.size(1));
  const int size_n = static_cast<int>(out_features);
  const int k_tiles = size_k / kTileRows;
  const int n_tiles = size_n / kTileColumns;
  TORCH_CHECK(split_count <= k_tiles, "QVQ P32 Ampere split count cannot exceed the K16 tile count");
  const int64_t expected_tiles = static_cast<int64_t>(k_tiles) * n_tiles;
  TORCH_CHECK(
      trellis.numel() == expected_tiles * kWordsPerTile,
      "QVQ P32 Ampere trellis has the wrong word count");
  TORCH_CHECK(bank_ids.numel() == expected_tiles, "QVQ P32 Ampere bank ids have the wrong length");
  TORCH_CHECK(levels.numel() == kLevels, "QVQ P32 Ampere requires 256 PGC16 levels");

  auto output = at::empty({size_m, size_n}, input.options().dtype(at::kFloat));
  auto partial_output = split_count == 1
      ? output
      : at::empty({split_count, size_m, size_n}, input.options().dtype(at::kFloat));
  const dim3 grid(
      static_cast<unsigned>((n_tiles + kTilesPerBlock - 1) / kTilesPerBlock),
      1,
      static_cast<unsigned>(split_count));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  if (size_m == kRows) {
    p32_window_ampere_kernel<TransitionBits, true><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else {
    p32_window_ampere_kernel<TransitionBits, false><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(levels.data_ptr<at::Half>()),
        bank_ids.data_ptr<uint8_t>(),
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        size_m,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (split_count > 1) {
    constexpr int kReductionThreads = 256;
    const int output_values = size_m * size_n;
    const int blocks = (output_values + kReductionThreads - 1) / kReductionThreads;
    reduce_split_kernel<<<blocks, kReductionThreads, 0, stream>>>(
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        output_values,
        static_cast<int>(split_count));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

at::Tensor p32_window_ampere(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    int64_t out_features,
    int64_t bank_alt_id,
    int64_t split_count) {
  switch (transition_bits) {
    case 4:
      return p32_window_ampere_impl<4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 5:
      return p32_window_ampere_impl<5>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 6:
      return p32_window_ampere_impl<6>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 7:
      return p32_window_ampere_impl<7>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    default:
      TORCH_CHECK(false, "QVQ P32 Ampere transition bits must be in [4, 7]");
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq_ampere, m) {
  m.def("p32_window(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq_ampere, CUDA, m) {
  m.impl("p32_window", p32_window_ampere);
}
