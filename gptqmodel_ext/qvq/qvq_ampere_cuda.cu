// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kRows = 16;
constexpr int kTileRows = 16;
constexpr int kTileColumns = 16;
constexpr int kTilesPerBlock = kWarps;
constexpr int kStageKTiles = 2;
constexpr int kStageColumns = kStageKTiles * kTileRows;
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

struct MmaFragmentA {
  uint32_t values[4];
};

struct MmaFragmentB {
  uint32_t values[2];
};

struct MmaFragmentC {
  float values[4];
};

__device__ __forceinline__ void load_mma_fragment_a(
    MmaFragmentA& fragment,
    const void* shared_source) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(fragment.values[0]),
        "=r"(fragment.values[1]),
        "=r"(fragment.values[2]),
        "=r"(fragment.values[3])
      : "r"(shared_address));
}

__device__ __forceinline__ void mma_m16n8k16(
    const MmaFragmentA& input,
    const MmaFragmentB& weight,
    MmaFragmentC& accumulator) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(accumulator.values[0]),
        "=f"(accumulator.values[1]),
        "=f"(accumulator.values[2]),
        "=f"(accumulator.values[3])
      : "r"(input.values[0]),
        "r"(input.values[1]),
        "r"(input.values[2]),
        "r"(input.values[3]),
        "r"(weight.values[0]),
        "r"(weight.values[1]),
        "f"(accumulator.values[0]),
        "f"(accumulator.values[1]),
        "f"(accumulator.values[2]),
        "f"(accumulator.values[3]));
}

template <int TransitionBits>
__device__ __forceinline__ uint32_t decode_pair_bits(
    int pair,
    uint32_t state,
    uint8_t packed_bank_id,
    uint32_t alt_mask,
    const half* __restrict__ levels) {
  const uint32_t bank_mask =
      ((static_cast<uint32_t>(packed_bank_id) >> (pair >> 4)) & 1u) * alt_mask;
  const uint32_t mixed = pgc16_mix(state ^ bank_mask);
  union {
    uint32_t bits;
    half2 values;
  } decoded;
  decoded.values = __halves2half2(levels[mixed >> 8], levels[mixed & 0xffu]);
  return decoded.bits;
}

__device__ __forceinline__ uint32_t pack_low_halves(uint32_t first, uint32_t second) {
  return (first & 0xffffu) | (second << 16);
}

__device__ __forceinline__ uint32_t pack_high_halves(uint32_t first, uint32_t second) {
  return (first >> 16) | (second & 0xffff0000u);
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
  __shared__ __align__(32) half input_tile[2][kRows * kStageColumns];
  __shared__ __align__(16) uint32_t packed_words[2][kStageKTiles][kTilesPerBlock][kWordsPerTile];
  __shared__ uint8_t packed_bank_ids[2][kStageKTiles][kTilesPerBlock];
  __shared__ half cached_levels[kLevels];

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

  auto stage = [&](int k_tile_base, int destination) {
    auto* input_vectors = reinterpret_cast<uint4*>(input_tile[destination]);
    for (int index = thread; index < kRows * kStageColumns / 8; index += kThreads) {
      const int row = index / (kStageColumns / 8);
      const int vector = index - row * (kStageColumns / 8);
      const int source_column = k_tile_base * kTileRows + vector * 8;
      if constexpr (FullRows) {
        if (source_column < size_k) {
          const half* source = input + static_cast<int64_t>(row) * size_k + source_column;
          __pipeline_memcpy_async(input_vectors + index, reinterpret_cast<const uint4*>(source), 16);
        } else {
          input_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
        }
      } else if (row < size_m && source_column < size_k) {
        const half* source = input + static_cast<int64_t>(row) * size_k +
            source_column;
        __pipeline_memcpy_async(input_vectors + index, reinterpret_cast<const uint4*>(source), 16);
      } else {
        input_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
      }
    }

    constexpr int kVectorsPerTile = kWordsPerTile / 4;
    constexpr int kVectorsPerKTile = kTilesPerBlock * kVectorsPerTile;
    constexpr int kVectorsPerBlock = kStageKTiles * kVectorsPerKTile;
    auto* destination_vectors = reinterpret_cast<uint4*>(packed_words[destination]);
    for (int index = thread; index < kVectorsPerBlock; index += kThreads) {
      const int stage_k_tile = index / kVectorsPerKTile;
      const int tile_index = index - stage_k_tile * kVectorsPerKTile;
      const int tile = tile_index / kVectorsPerTile;
      const int vector = tile_index - tile * kVectorsPerTile;
      const int n_tile = n_tile_base + tile;
      const int k_tile = k_tile_base + stage_k_tile;
      if (n_tile < n_tiles && k_tile < k_tiles) {
        const int64_t global_tile = static_cast<int64_t>(k_tile) * n_tiles + n_tile;
        const auto* source_vectors = reinterpret_cast<const uint4*>(
            trellis + global_tile * kWordsPerTile);
        __pipeline_memcpy_async(destination_vectors + index, source_vectors + vector, 16);
      } else {
        destination_vectors[index] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    if (thread < kStageKTiles * kTilesPerBlock) {
      const int stage_k_tile = thread / kTilesPerBlock;
      const int tile = thread - stage_k_tile * kTilesPerBlock;
      const int n_tile = n_tile_base + tile;
      const int k_tile = k_tile_base + stage_k_tile;
      packed_bank_ids[destination][stage_k_tile][tile] = n_tile < n_tiles && k_tile < k_tiles
          ? bank_ids[static_cast<int64_t>(k_tile) * n_tiles + n_tile]
          : 0;
    }
    __pipeline_commit();
  };

  MmaFragmentC accumulator_0 = {};
  MmaFragmentC accumulator_1 = {};

  stage(k_tile_begin, 0);
  int parity = 0;
  for (int k_tile = k_tile_begin; k_tile < k_tile_end; k_tile += kStageKTiles) {
    const bool has_next = k_tile + kStageKTiles < k_tile_end;
    if (has_next) {
      stage(k_tile + kStageKTiles, parity ^ 1);
    }
    __pipeline_wait_prior(has_next ? 1 : 0);
    __syncthreads();

    if (active_tile) {
#pragma unroll
      for (int stage_k_tile = 0; stage_k_tile < kStageKTiles; ++stage_k_tile) {
        if (k_tile + stage_k_tile >= k_tile_end) {
          continue;
        }
        const uint32_t* words = packed_words[parity][stage_k_tile][warp];
        const uint8_t packed_bank_id = packed_bank_ids[parity][stage_k_tile][warp];
        const int producer_fragment = lane >> 4;
        const int producer_pair_column = producer_fragment * 4 + ((lane >> 2) & 3);
        const int producer_row_pair = lane & 3;
        const int first_pair = producer_row_pair * 16 + producer_pair_column;
        const int second_pair = first_pair + 8;
        uint32_t state_row_0;
        uint32_t state_row_8;
        uint32_t state_row_1;
        uint32_t state_row_9;
        window_state_pair64<TransitionBits>(words, first_pair, state_row_0, state_row_8);
        window_state_pair64<TransitionBits>(words, second_pair, state_row_1, state_row_9);
        const uint32_t decoded_row_0 = decode_pair_bits<TransitionBits>(
            first_pair, state_row_0, packed_bank_id, alt_mask, cached_levels);
        const uint32_t decoded_row_8 = decode_pair_bits<TransitionBits>(
            first_pair + 64, state_row_8, packed_bank_id, alt_mask, cached_levels);
        const uint32_t decoded_row_1 = decode_pair_bits<TransitionBits>(
            second_pair, state_row_1, packed_bank_id, alt_mask, cached_levels);
        const uint32_t decoded_row_9 = decode_pair_bits<TransitionBits>(
            second_pair + 64, state_row_9, packed_bank_id, alt_mask, cached_levels);

        const uint32_t low_rows_01 = pack_low_halves(decoded_row_0, decoded_row_1);
        const uint32_t high_rows_01 = pack_high_halves(decoded_row_0, decoded_row_1);
        const uint32_t low_rows_89 = pack_low_halves(decoded_row_8, decoded_row_9);
        const uint32_t high_rows_89 = pack_high_halves(decoded_row_8, decoded_row_9);
        const int target_column = lane >> 2;
        const int source_lane_0 = ((target_column >> 1) << 2) + (lane & 3);
        const int source_lane_1 = source_lane_0 + 16;
        const bool select_high = (target_column & 1) != 0;
        constexpr uint32_t kFullWarpMask = 0xffffffffu;
        MmaFragmentB weight_fragment_0;
        MmaFragmentB weight_fragment_1;
        const uint32_t low_01_0 = __shfl_sync(kFullWarpMask, low_rows_01, source_lane_0);
        const uint32_t high_01_0 = __shfl_sync(kFullWarpMask, high_rows_01, source_lane_0);
        const uint32_t low_89_0 = __shfl_sync(kFullWarpMask, low_rows_89, source_lane_0);
        const uint32_t high_89_0 = __shfl_sync(kFullWarpMask, high_rows_89, source_lane_0);
        const uint32_t low_01_1 = __shfl_sync(kFullWarpMask, low_rows_01, source_lane_1);
        const uint32_t high_01_1 = __shfl_sync(kFullWarpMask, high_rows_01, source_lane_1);
        const uint32_t low_89_1 = __shfl_sync(kFullWarpMask, low_rows_89, source_lane_1);
        const uint32_t high_89_1 = __shfl_sync(kFullWarpMask, high_rows_89, source_lane_1);
        weight_fragment_0.values[0] = select_high ? high_01_0 : low_01_0;
        weight_fragment_0.values[1] = select_high ? high_89_0 : low_89_0;
        weight_fragment_1.values[0] = select_high ? high_01_1 : low_01_1;
        weight_fragment_1.values[1] = select_high ? high_89_1 : low_89_1;

        const int address_row = (lane & 7) + ((lane >> 3) & 1) * 8;
        const int address_column = (lane >> 4) * 8;
        MmaFragmentA input_fragment;
        load_mma_fragment_a(
            input_fragment,
            input_tile[parity] +
                address_row * kStageColumns +
                stage_k_tile * kTileRows +
                address_column);
        mma_m16n8k16(input_fragment, weight_fragment_0, accumulator_0);
        mma_m16n8k16(input_fragment, weight_fragment_1, accumulator_1);
      }
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
  if (active_tile) {
    const int output_row_0 = lane >> 2;
    const int output_row_1 = output_row_0 + 8;
    const int output_column =
        (n_tile_base + warp) * kTileColumns + (lane & 3) * 2;
    if constexpr (FullRows) {
      target[static_cast<int64_t>(output_row_0) * size_n + output_column] = accumulator_0.values[0];
      target[static_cast<int64_t>(output_row_0) * size_n + output_column + 1] = accumulator_0.values[1];
      target[static_cast<int64_t>(output_row_1) * size_n + output_column] = accumulator_0.values[2];
      target[static_cast<int64_t>(output_row_1) * size_n + output_column + 1] = accumulator_0.values[3];
      target[static_cast<int64_t>(output_row_0) * size_n + output_column + 8] = accumulator_1.values[0];
      target[static_cast<int64_t>(output_row_0) * size_n + output_column + 9] = accumulator_1.values[1];
      target[static_cast<int64_t>(output_row_1) * size_n + output_column + 8] = accumulator_1.values[2];
      target[static_cast<int64_t>(output_row_1) * size_n + output_column + 9] = accumulator_1.values[3];
    } else {
      if (output_row_0 < size_m) {
        target[static_cast<int64_t>(output_row_0) * size_n + output_column] = accumulator_0.values[0];
        target[static_cast<int64_t>(output_row_0) * size_n + output_column + 1] = accumulator_0.values[1];
        target[static_cast<int64_t>(output_row_0) * size_n + output_column + 8] = accumulator_1.values[0];
        target[static_cast<int64_t>(output_row_0) * size_n + output_column + 9] = accumulator_1.values[1];
      }
      if (output_row_1 < size_m) {
        target[static_cast<int64_t>(output_row_1) * size_n + output_column] = accumulator_0.values[2];
        target[static_cast<int64_t>(output_row_1) * size_n + output_column + 1] = accumulator_0.values[3];
        target[static_cast<int64_t>(output_row_1) * size_n + output_column + 8] = accumulator_1.values[2];
        target[static_cast<int64_t>(output_row_1) * size_n + output_column + 9] = accumulator_1.values[3];
      }
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
