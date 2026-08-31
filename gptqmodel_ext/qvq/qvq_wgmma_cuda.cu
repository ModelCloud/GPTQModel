// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cstdint>

namespace {

using Element = cutlass::half_t;
using WgmmaTileShape = cute::Shape<cute::_64, cute::_16, cute::_16>;
using WgmmaTiledMma = decltype(cute::make_tiled_mma(
    cute::GMMA::rs_op_selector<
        Element,
        Element,
        float,
        WgmmaTileShape,
        cute::GMMA::Major::K,
        cute::GMMA::Major::K>()));
using WgmmaSmemLayoutAtomB = decltype(
    cutlass::gemm::collective::detail::rs_smem_selector<
        cute::GMMA::Major::K,
        Element,
        cute::_16,
        cute::_256,
        false>());
using WgmmaSmemLayoutB = decltype(cute::tile_to_shape(
    WgmmaSmemLayoutAtomB{}, cute::make_shape(cute::_16{}, cute::_256{})));

constexpr int kThreads = 128;
constexpr int kTmaThreads = 160;
constexpr int kRows = 16;
constexpr int kOutputColumns = 64;
constexpr int kK32Rows = 32;
constexpr int kN8Columns = 8;
constexpr int kW3TransitionBits = 6;
constexpr int kWordsPerTile = 4 * kW3TransitionBits;
constexpr int kK32TilesPerStage = 8;
constexpr int kKPerStage = kK32TilesPerStage * kK32Rows;
constexpr int kN8TilesPerBlock = kOutputColumns / kN8Columns;
constexpr int kTmaStages = 2;
constexpr uint32_t kPgc16Multiplier = 40503u;
constexpr uint32_t kPgc16Increment = 17011u;

// Standard P32 uses K16 x N16 tiles.  Four adjacent tiles form the register
// sourced A operand of one m64n16k16 WGMMA.  The continuous-window payload is
// storage-neutral with canonical planar P32, but makes every 16-bit state a
// direct circular bit-window load instead of a reconstructed recurrence.
constexpr int kP32TileRows = 16;
constexpr int kP32TileColumns = 16;
constexpr int kP32PairsPerTile = 128;
constexpr int kP32N16TilesPerBlock = kOutputColumns / kP32TileColumns;
constexpr int kP32K16TilesPerStage = kKPerStage / kP32TileRows;

using WgmmaTmaSmemLayoutB = decltype(cute::tile_to_shape(
    WgmmaSmemLayoutAtomB{},
    cute::make_shape(cute::_16{}, cute::_256{}, cute::Int<kTmaStages>{})));
using TrellisTmaSmemLayout = decltype(cute::make_layout(
    cute::make_shape(
        cute::Int<kWordsPerTile>{},
        cute::Int<kN8TilesPerBlock>{},
        cute::Int<kK32TilesPerStage>{},
        cute::Int<kTmaStages>{}),
    cute::make_stride(
        cute::_1{},
        cute::Int<kWordsPerTile>{},
        cute::Int<kWordsPerTile * kN8TilesPerBlock>{},
        cute::Int<kWordsPerTile * kN8TilesPerBlock * kK32TilesPerStage>{})));
using BankTmaSmemLayout = decltype(cute::make_layout(
    cute::make_shape(cute::_16{}, cute::_8{}, cute::Int<kTmaStages>{}),
    cute::make_stride(cute::_1{}, cute::_16{}, cute::_128{})));
template <int TransitionBits>
using P32TrellisTmaSmemLayoutFor = decltype(cute::make_layout(
    cute::make_shape(
        cute::Int<4 * TransitionBits>{},
        cute::Int<kP32N16TilesPerBlock>{},
        cute::Int<kP32K16TilesPerStage>{},
        cute::Int<kTmaStages>{}),
    cute::make_stride(
        cute::_1{},
        cute::Int<4 * TransitionBits>{},
        cute::Int<4 * TransitionBits * kP32N16TilesPerBlock>{},
        cute::Int<4 * TransitionBits * kP32N16TilesPerBlock * kP32K16TilesPerStage>{})));
using P32BankTmaSmemLayout = decltype(cute::make_layout(
    cute::make_shape(cute::_16{}, cute::_16{}, cute::Int<kTmaStages>{}),
    cute::make_stride(cute::_1{}, cute::_16{}, cute::_256{})));
using WgmmaTmaPipeline = cutlass::PipelineTmaAsync<kTmaStages>;
using WgmmaTmaPipelineState = cutlass::PipelineState<kTmaStages>;

struct alignas(128) WgmmaTmaSharedStorage {
  typename WgmmaTmaPipeline::SharedStorage pipeline;
  alignas(128) cute::ArrayEngine<Element, cute::cosize_v<WgmmaTmaSmemLayoutB>> input;
  alignas(128) cute::ArrayEngine<uint32_t, cute::cosize_v<TrellisTmaSmemLayout>> trellis;
  alignas(128) cute::ArrayEngine<uint8_t, cute::cosize_v<BankTmaSmemLayout>> bank_ids;
};

template <int TransitionBits>
struct alignas(128) P32WgmmaTmaSharedStorageFor {
  typename WgmmaTmaPipeline::SharedStorage pipeline;
  alignas(128) cute::ArrayEngine<Element, cute::cosize_v<WgmmaTmaSmemLayoutB>> input;
  alignas(128) cute::ArrayEngine<
      uint32_t,
      cute::cosize_v<P32TrellisTmaSmemLayoutFor<TransitionBits>>> trellis;
  alignas(128) cute::ArrayEngine<uint8_t, cute::cosize_v<P32BankTmaSmemLayout>> bank_ids;
};

static_assert(cute::size(WgmmaTiledMma{}) == kThreads);

__device__ __forceinline__ uint32_t qvq_wgmma_pgc16_mix(uint32_t state) {
  uint32_t mixed = state ^ (state >> 8);
  mixed = (mixed * kPgc16Multiplier + kPgc16Increment) & 0xffffu;
  return mixed ^ (mixed >> 7);
}

__device__ __forceinline__ Element qvq_wgmma_load_level(
    const Element* __restrict__ levels,
    uint32_t index) {
  uint64_t address;
  asm("mad.wide.u32 %0, %1, 2, %2;"
      : "=l"(address)
      : "r"(index), "l"(levels));
  uint16_t bits;
  asm("ld.global.nc.L1::evict_last.u16 %0, [%1];"
      : "=h"(bits)
      : "l"(address));
  return Element::bitcast(bits);
}

template <int TransitionBits>
__device__ __forceinline__ uint32_t qvq_wgmma_v2_alternate_bank_mask(int bank_alt_id) {
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

__device__ __forceinline__ uint32_t qvq_wgmma_w3_transition(
    uint32_t low_first,
    uint32_t low_second,
    uint32_t high,
    int edge) {
  const uint32_t low_word = edge < 8 ? low_first : low_second;
  const uint32_t low = (low_word >> (4 * (edge & 7))) & 0xfu;
  const uint32_t high_bits = (high >> (2 * edge)) & 0x3u;
  return low | (high_bits << 4);
}

__device__ __forceinline__ uint32_t qvq_wgmma_w3_state(
    uint32_t low_first,
    uint32_t low_second,
    uint32_t high,
    int pair) {
  const int edge0 = (pair + 14) & 15;
  const int edge1 = (pair + 15) & 15;
  const uint32_t transition0 = qvq_wgmma_w3_transition(low_first, low_second, high, edge0);
  const uint32_t transition1 = qvq_wgmma_w3_transition(low_first, low_second, high, edge1);
  const uint32_t transition2 = qvq_wgmma_w3_transition(low_first, low_second, high, pair & 15);
  return ((transition0 << 12) | (transition1 << 6) | transition2) & 0xffffu;
}

template <class FragmentA>
__device__ __forceinline__ void qvq_wgmma_decode_w3_fragment(
    FragmentA& fragment,
    const uint32_t* __restrict__ packed_first,
    const uint32_t* __restrict__ packed_second,
    uint8_t bank_id_first,
    uint8_t bank_id_second,
    const Element* __restrict__ levels,
    int column_in_n8,
    int k16_half,
    uint32_t alternate_bank_mask) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int pair_lane = lane & 3;
  const int source_lane = lane & ~3;
  const int edge_block = column_in_n8 >> 1;
  const int edge_offset = (column_in_n8 & 1) * 16;
  const int word_base = edge_block * kW3TransitionBits;

  uint32_t first_low0 = 0;
  uint32_t first_low1 = 0;
  uint32_t first_high = 0;
  uint32_t second_low0 = 0;
  uint32_t second_low1 = 0;
  uint32_t second_high = 0;
  if (pair_lane == 0) {
    first_low0 = packed_first[word_base + edge_offset / 8];
    first_low1 = packed_first[word_base + edge_offset / 8 + 1];
    first_high = packed_first[word_base + 4 + edge_offset / 16];
    second_low0 = packed_second[word_base + edge_offset / 8];
    second_low1 = packed_second[word_base + edge_offset / 8 + 1];
    second_high = packed_second[word_base + 4 + edge_offset / 16];
  }
  first_low0 = __shfl_sync(0xffffffffu, first_low0, source_lane);
  first_low1 = __shfl_sync(0xffffffffu, first_low1, source_lane);
  first_high = __shfl_sync(0xffffffffu, first_high, source_lane);
  second_low0 = __shfl_sync(0xffffffffu, second_low0, source_lane);
  second_low1 = __shfl_sync(0xffffffffu, second_low1, source_lane);
  second_high = __shfl_sync(0xffffffffu, second_high, source_lane);

  const uint32_t first_bank_bit = (static_cast<uint32_t>(bank_id_first) >> column_in_n8) & 1u;
  const uint32_t second_bank_bit = (static_cast<uint32_t>(bank_id_second) >> column_in_n8) & 1u;
  const uint32_t first_bank_mask = (0u - first_bank_bit) & alternate_bank_mask;
  const uint32_t second_bank_mask = (0u - second_bank_bit) & alternate_bank_mask;
  const int first_pair = k16_half * 8 + pair_lane;
  const int second_pair = first_pair + 4;

  const uint32_t first_mixed0 = qvq_wgmma_pgc16_mix(
      qvq_wgmma_w3_state(first_low0, first_low1, first_high, first_pair) ^ first_bank_mask);
  const uint32_t second_mixed0 = qvq_wgmma_pgc16_mix(
      qvq_wgmma_w3_state(second_low0, second_low1, second_high, first_pair) ^ second_bank_mask);
  const uint32_t first_mixed1 = qvq_wgmma_pgc16_mix(
      qvq_wgmma_w3_state(first_low0, first_low1, first_high, second_pair) ^ first_bank_mask);
  const uint32_t second_mixed1 = qvq_wgmma_pgc16_mix(
      qvq_wgmma_w3_state(second_low0, second_low1, second_high, second_pair) ^ second_bank_mask);

  // CuTe's m64n16k16 RS fragment maps each lane to four half2 values:
  // (row0,kpair0), (row1,kpair0), (row0,kpair1), (row1,kpair1).
  fragment(0) = levels[first_mixed0 >> 8];
  fragment(1) = levels[first_mixed0 & 0xffu];
  fragment(2) = levels[second_mixed0 >> 8];
  fragment(3) = levels[second_mixed0 & 0xffu];
  fragment(4) = levels[first_mixed1 >> 8];
  fragment(5) = levels[first_mixed1 & 0xffu];
  fragment(6) = levels[second_mixed1 >> 8];
  fragment(7) = levels[second_mixed1 & 0xffu];
}

template <int TransitionBits>
__device__ __forceinline__ uint32_t qvq_p32_window_state(
    const uint32_t* __restrict__ window_words,
    int pair) {
  constexpr int kWordsPerP32Tile = 4 * TransitionBits;
  const int bit_position = (kP32PairsPerTile - 1 - pair) * TransitionBits;
  const int word = bit_position >> 5;
  const int shift = bit_position & 31;
  const int next_word = word + 1 == kWordsPerP32Tile ? 0 : word + 1;
  const uint64_t window = static_cast<uint64_t>(window_words[word]) |
      (static_cast<uint64_t>(window_words[next_word]) << 32);
  return static_cast<uint32_t>((window >> shift) & 0xffffu);
}

template <int TransitionBits>
__device__ __forceinline__ void qvq_p32_window_state_pair(
    const uint32_t* __restrict__ window_words,
    int pair,
    uint32_t& first,
    uint32_t& second) {
  constexpr int kWordsPerP32Tile = 4 * TransitionBits;
  constexpr int kKPairWordDistance = 2 * TransitionBits;
  const int bit_position = (kP32PairsPerTile - 1 - pair) * TransitionBits;
  const int first_word = bit_position >> 5;
  const int shift = bit_position & 31;
  const int first_next_word = first_word + 1 == kWordsPerP32Tile ? 0 : first_word + 1;
  const int second_word = first_word - kKPairWordDistance;
  const uint64_t first_window = static_cast<uint64_t>(window_words[first_word]) |
      (static_cast<uint64_t>(window_words[first_next_word]) << 32);
  const uint64_t second_window = static_cast<uint64_t>(window_words[second_word]) |
      (static_cast<uint64_t>(window_words[second_word + 1]) << 32);
  first = static_cast<uint32_t>((first_window >> shift) & 0xffffu);
  second = static_cast<uint32_t>((second_window >> shift) & 0xffffu);
}

template <int TransitionBits, class FragmentA>
__device__ __forceinline__ void qvq_p32_window_decode_fragment(
    FragmentA& fragment,
    const uint32_t* __restrict__ window_words,
    uint8_t bank_id,
    const Element* __restrict__ levels,
    uint32_t alternate_bank_mask) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int n_pair = lane >> 2;
  const int k_pair0 = lane & 3;
  const int k_pair1 = k_pair0 + 4;
  const int pair00 = k_pair0 * 16 + n_pair;
  const int pair01 = pair00 + 8;
  const uint32_t bank_pair_bits = static_cast<uint32_t>(bank_id) >> k_pair0;
  const uint32_t bank_mask0 = (bank_pair_bits & 1u) * alternate_bank_mask;
  const uint32_t bank_mask1 = ((bank_pair_bits >> 4) & 1u) * alternate_bank_mask;

  uint32_t state00;
  uint32_t state01;
  uint32_t state10;
  uint32_t state11;
  qvq_p32_window_state_pair<TransitionBits>(window_words, pair00, state00, state10);
  qvq_p32_window_state_pair<TransitionBits>(window_words, pair01, state01, state11);
  const uint32_t mixed00 = qvq_wgmma_pgc16_mix(state00 ^ bank_mask0);
  const uint32_t mixed01 = qvq_wgmma_pgc16_mix(state01 ^ bank_mask0);
  const uint32_t mixed10 = qvq_wgmma_pgc16_mix(state10 ^ bank_mask1);
  const uint32_t mixed11 = qvq_wgmma_pgc16_mix(state11 ^ bank_mask1);

  // CuTe maps each lane to two A rows and two K pairs.  Map those two rows to
  // adjacent P32 N values so each decoded state feeds both output columns.
  fragment(0) = levels[mixed00 >> 8];
  fragment(1) = levels[mixed01 >> 8];
  if constexpr (TransitionBits == kW3TransitionBits) {
    fragment(2) = qvq_wgmma_load_level(levels, mixed00 & 0xffu);
    fragment(3) = qvq_wgmma_load_level(levels, mixed01 & 0xffu);
  } else {
    fragment(2) = levels[mixed00 & 0xffu];
    fragment(3) = levels[mixed01 & 0xffu];
  }
  fragment(4) = levels[mixed10 >> 8];
  fragment(5) = levels[mixed11 >> 8];
  if constexpr (TransitionBits == kW3TransitionBits) {
    fragment(6) = qvq_wgmma_load_level(levels, mixed10 & 0xffu);
    fragment(7) = qvq_wgmma_load_level(levels, mixed11 & 0xffu);
  } else {
    fragment(6) = levels[mixed10 & 0xffu];
    fragment(7) = levels[mixed11 & 0xffu];
  }
}

__global__ __launch_bounds__(kThreads) void qvq_wgmma_w3_m16_kernel(
    const Element* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const uint8_t* __restrict__ bank_ids,
    const Element* __restrict__ levels,
    float* __restrict__ partial_output,
    int size_k,
    int size_n,
    int split_count,
    int bank_alt_id) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  __shared__ __align__(16) uint32_t packed_words
      [kK32TilesPerStage][kN8TilesPerBlock][kWordsPerTile];
  __shared__ uint8_t packed_bank_ids[kK32TilesPerStage][kN8TilesPerBlock];
  __shared__ __align__(128) Element shared_input[cute::cosize_v<WgmmaSmemLayoutB>];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int n64_block = static_cast<int>(blockIdx.x);
  const int n8_tile_base = n64_block * kN8TilesPerBlock;
  const int n_tiles = size_n / kN8Columns;
  const int k_tiles = size_k / kK32Rows;
  const int split = static_cast<int>(blockIdx.z);
  const int k_tile_begin = (k_tiles * split) / split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / split_count;
  const uint32_t alternate_bank_mask = bank_alt_id == 0 ? 0u
      : bank_alt_id == 1 ? 0x6969u
      : bank_alt_id == 2 ? 0x5a5au
      : 0x3c3cu;

  auto sB = cute::make_tensor(cute::make_smem_ptr(shared_input), WgmmaSmemLayoutB{});
  WgmmaTiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(thread);
  auto thread_shared_b = thread_mma.partition_B(sB);
  auto fragment_b = thread_mma.make_fragment_B(thread_shared_b);
  static_assert(cute::size<2>(decltype(fragment_b){}) == 16);

  auto coordinate_a = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_a = thread_mma.partition_A(coordinate_a);
  auto fragment_a0 = cute::make_tensor<Element>(thread_coordinate_a.shape());
  auto fragment_a1 = cute::make_tensor<Element>(thread_coordinate_a.shape());
  static_assert(cute::size(decltype(fragment_a0){}) == 8);

  auto coordinate_c = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_c = thread_mma.partition_C(coordinate_c);
  auto accumulator = cute::make_tensor<float>(thread_coordinate_c.shape());
  cute::clear(accumulator);
  tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;

  const int row_in_warp = lane >> 2;
  const int column_in_n8 = row_in_warp;
  const int first_n8_tile = warp * 2;
  const int second_n8_tile = first_n8_tile + 1;

  for (int kb = k_tile_begin; kb < k_tile_end; kb += kK32TilesPerStage) {
    auto* packed_vectors = reinterpret_cast<uint4*>(&packed_words[0][0][0]);
    const auto* trellis_vectors = reinterpret_cast<const uint4*>(trellis);
    constexpr int kVectorsPerTile = kWordsPerTile / 4;
    constexpr int kStageVectors = kK32TilesPerStage * kN8TilesPerBlock * kVectorsPerTile;
    for (int index = thread; index < kStageVectors; index += kThreads) {
      const int tile_slot = index / kVectorsPerTile;
      const int vector_in_tile = index - tile_slot * kVectorsPerTile;
      const int k32_in_stage = tile_slot / kN8TilesPerBlock;
      const int n8_in_block = tile_slot - k32_in_stage * kN8TilesPerBlock;
      const int64_t global_tile =
          static_cast<int64_t>(kb + k32_in_stage) * n_tiles + n8_tile_base + n8_in_block;
      packed_vectors[index] = trellis_vectors[global_tile * kVectorsPerTile + vector_in_tile];
    }
    for (int index = thread;
         index < kK32TilesPerStage * kN8TilesPerBlock;
         index += kThreads) {
      const int k32_in_stage = index / kN8TilesPerBlock;
      const int n8_in_block = index - k32_in_stage * kN8TilesPerBlock;
      const int64_t global_tile =
          static_cast<int64_t>(kb + k32_in_stage) * n_tiles + n8_tile_base + n8_in_block;
      packed_bank_ids[k32_in_stage][n8_in_block] = bank_ids[global_tile];
    }
    for (int index = thread; index < kRows * kKPerStage; index += kThreads) {
      const int row = index / kKPerStage;
      const int k_in_stage = index - row * kKPerStage;
      sB(row, k_in_stage) = input[
          static_cast<int64_t>(row) * size_k + kb * kK32Rows + k_in_stage];
    }
    __syncthreads();

#pragma unroll
    for (int k_block = 0; k_block < 16; ++k_block) {
      const int k32_in_stage = k_block >> 1;
      const int k16_half = k_block & 1;
      auto& fragment_a = (k_block & 1) == 0 ? fragment_a0 : fragment_a1;

      if (k_block >= 2) {
        cute::warpgroup_wait<1>();
      }
      qvq_wgmma_decode_w3_fragment(
          fragment_a,
          &packed_words[k32_in_stage][first_n8_tile][0],
          &packed_words[k32_in_stage][second_n8_tile][0],
          packed_bank_ids[k32_in_stage][first_n8_tile],
          packed_bank_ids[k32_in_stage][second_n8_tile],
          levels,
          column_in_n8,
          k16_half,
          alternate_bank_mask);
      cute::warpgroup_fence_operand(fragment_a);
      cute::warpgroup_arrive();
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b(cute::_, cute::_, k_block),
          accumulator);
      tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
      cute::warpgroup_commit_batch();
    }
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(accumulator);
    __syncthreads();
  }

  constexpr int kAccumulatorValuesPerThread = cute::size(decltype(thread_coordinate_c){});
#pragma unroll
  for (int index = 0; index < kAccumulatorValuesPerThread; ++index) {
    const auto coordinate = thread_coordinate_c(index);
    const int output_column_in_block = static_cast<int>(cute::get<0>(coordinate));
    const int output_row = static_cast<int>(cute::get<1>(coordinate));
    const int64_t output_index =
        static_cast<int64_t>(output_row) * size_n + n64_block * kOutputColumns + output_column_in_block;
    partial_output[static_cast<int64_t>(split) * kRows * size_n + output_index] = accumulator(index);
  }
#endif
}

__global__ __launch_bounds__(kThreads) void qvq_p32_window_wgmma_w3_m16_kernel(
    const Element* __restrict__ input,
    const uint32_t* __restrict__ trellis,
    const uint8_t* __restrict__ bank_ids,
    const Element* __restrict__ levels,
    float* __restrict__ partial_output,
    int size_k,
    int size_n,
    int split_count,
    int bank_alt_id) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  constexpr int kWordsPerP32Tile = 4 * kW3TransitionBits;
  __shared__ __align__(16) uint32_t packed_words
      [kP32K16TilesPerStage][kP32N16TilesPerBlock][kWordsPerP32Tile];
  __shared__ uint8_t packed_bank_ids[kP32K16TilesPerStage][kP32N16TilesPerBlock];
  __shared__ __align__(128) Element shared_input[cute::cosize_v<WgmmaSmemLayoutB>];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int n64_block = static_cast<int>(blockIdx.x);
  const int n16_tile_base = n64_block * kP32N16TilesPerBlock;
  const int n_tiles = size_n / kP32TileColumns;
  const int k_tiles = size_k / kP32TileRows;
  const int split = static_cast<int>(blockIdx.z);
  const int k_tile_begin = (k_tiles * split) / split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / split_count;
  const uint32_t alternate_bank_mask = bank_alt_id == 0 ? 0u
      : bank_alt_id == 1 ? 0x6969u
      : bank_alt_id == 2 ? 0x5a5au
      : 0x3c3cu;

  auto sB = cute::make_tensor(cute::make_smem_ptr(shared_input), WgmmaSmemLayoutB{});
  WgmmaTiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(thread);
  auto thread_shared_b = thread_mma.partition_B(sB);
  auto fragment_b = thread_mma.make_fragment_B(thread_shared_b);
  static_assert(cute::size<2>(decltype(fragment_b){}) == 16);

  auto coordinate_a = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_a = thread_mma.partition_A(coordinate_a);
  auto fragment_a0 = cute::make_tensor<Element>(thread_coordinate_a.shape());
  auto fragment_a1 = cute::make_tensor<Element>(thread_coordinate_a.shape());
  static_assert(cute::size(decltype(fragment_a0){}) == 8);

  auto coordinate_c = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_c = thread_mma.partition_C(coordinate_c);
  auto accumulator = cute::make_tensor<float>(thread_coordinate_c.shape());
  cute::clear(accumulator);
  tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;

  for (int kb = k_tile_begin; kb < k_tile_end; kb += kP32K16TilesPerStage) {
    auto* packed_vectors = reinterpret_cast<uint4*>(&packed_words[0][0][0]);
    const auto* trellis_vectors = reinterpret_cast<const uint4*>(trellis);
    constexpr int kVectorsPerTile = kWordsPerP32Tile / 4;
    constexpr int kStageVectors =
        kP32K16TilesPerStage * kP32N16TilesPerBlock * kVectorsPerTile;
    for (int index = thread; index < kStageVectors; index += kThreads) {
      const int tile_slot = index / kVectorsPerTile;
      const int vector_in_tile = index - tile_slot * kVectorsPerTile;
      const int k16_in_stage = tile_slot / kP32N16TilesPerBlock;
      const int n16_in_block = tile_slot - k16_in_stage * kP32N16TilesPerBlock;
      const int64_t global_tile =
          static_cast<int64_t>(kb + k16_in_stage) * n_tiles + n16_tile_base + n16_in_block;
      packed_vectors[index] = trellis_vectors[global_tile * kVectorsPerTile + vector_in_tile];
    }
    for (int index = thread;
         index < kP32K16TilesPerStage * kP32N16TilesPerBlock;
         index += kThreads) {
      const int k16_in_stage = index / kP32N16TilesPerBlock;
      const int n16_in_block = index - k16_in_stage * kP32N16TilesPerBlock;
      const int64_t global_tile =
          static_cast<int64_t>(kb + k16_in_stage) * n_tiles + n16_tile_base + n16_in_block;
      packed_bank_ids[k16_in_stage][n16_in_block] = bank_ids[global_tile];
    }
    for (int index = thread; index < kRows * kKPerStage; index += kThreads) {
      const int row = index / kKPerStage;
      const int k_in_stage = index - row * kKPerStage;
      sB(row, k_in_stage) = input[
          static_cast<int64_t>(row) * size_k + kb * kP32TileRows + k_in_stage];
    }
    __syncthreads();

#pragma unroll
    for (int k_block = 0; k_block < kP32K16TilesPerStage; ++k_block) {
      auto& fragment_a = (k_block & 1) == 0 ? fragment_a0 : fragment_a1;
      if (k_block >= 2) {
        cute::warpgroup_wait<1>();
      }
      qvq_p32_window_decode_fragment<kW3TransitionBits>(
          fragment_a,
          &packed_words[k_block][warp][0],
          packed_bank_ids[k_block][warp],
          levels,
          alternate_bank_mask);
      cute::warpgroup_fence_operand(fragment_a);
      cute::warpgroup_arrive();
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b(cute::_, cute::_, k_block),
          accumulator);
      tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
      cute::warpgroup_commit_batch();
    }
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(accumulator);
    __syncthreads();
  }

  constexpr int kAccumulatorValuesPerThread = cute::size(decltype(thread_coordinate_c){});
#pragma unroll
  for (int index = 0; index < kAccumulatorValuesPerThread; ++index) {
    const auto coordinate = thread_coordinate_c(index);
    const int wgmma_column = static_cast<int>(cute::get<0>(coordinate));
    const int output_row = static_cast<int>(cute::get<1>(coordinate));
    const int tile_column = wgmma_column & 15;
    const int p32_column = (wgmma_column & ~15) + ((tile_column & 7) << 1) + (tile_column >> 3);
    const int64_t output_index =
        static_cast<int64_t>(output_row) * size_n + n64_block * kOutputColumns + p32_column;
    partial_output[static_cast<int64_t>(split) * kRows * size_n + output_index] = accumulator(index);
  }
#endif
}

template <int TransitionBits, class InputTma, class TrellisTma, class BankTma>
__global__ __launch_bounds__(kTmaThreads) void qvq_p32_window_wgmma_m16_tma_kernel(
    CUTE_GRID_CONSTANT InputTma const input_tma,
    CUTE_GRID_CONSTANT TrellisTma const trellis_tma,
    CUTE_GRID_CONSTANT BankTma const bank_tma,
    const Element* __restrict__ levels,
    float* __restrict__ partial_output,
    int size_k,
    int size_n,
    int split_count,
    int bank_alt_id) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  constexpr int kWordsPerP32Tile = 4 * TransitionBits;
  using TrellisSmemLayout = P32TrellisTmaSmemLayoutFor<TransitionBits>;
  using SharedStorage = P32WgmmaTmaSharedStorageFor<TransitionBits>;
  __shared__ __align__(128) char shared_buffer[sizeof(SharedStorage)];
  auto& shared = *reinterpret_cast<SharedStorage*>(shared_buffer);

  const int thread = static_cast<int>(threadIdx.x);
  const bool is_consumer = thread < kThreads;
  const bool is_producer = !is_consumer;
  const int n64_block = static_cast<int>(blockIdx.x);
  const int n_tiles = size_n / kP32TileColumns;
  const int k_tiles = size_k / kP32TileRows;
  const int split = static_cast<int>(blockIdx.z);
  const int k_tile_begin = (k_tiles * split) / split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / split_count;
  const int stage_begin = k_tile_begin / kP32K16TilesPerStage;
  const int stage_count = (k_tile_end - k_tile_begin) / kP32K16TilesPerStage;
  const uint32_t alternate_bank_mask =
      qvq_wgmma_v2_alternate_bank_mask<TransitionBits>(bank_alt_id);

  typename WgmmaTmaPipeline::Params pipeline_params;
  pipeline_params.role = is_producer
      ? WgmmaTmaPipeline::ThreadCategory::Producer
      : WgmmaTmaPipeline::ThreadCategory::Consumer;
  pipeline_params.is_leader = thread == kThreads;
  pipeline_params.num_consumers = kThreads;
  pipeline_params.transaction_bytes =
      kRows * kKPerStage * sizeof(Element) +
      kWordsPerP32Tile * kP32N16TilesPerBlock * kP32K16TilesPerStage * sizeof(uint32_t) +
      16 * kP32K16TilesPerStage * sizeof(uint8_t);
  WgmmaTmaPipeline pipeline(
      shared.pipeline,
      pipeline_params,
      cute::Shape<cute::_1, cute::_1, cute::_1>{});

  auto s_input = cute::make_tensor(
      cute::make_smem_ptr(shared.input.begin()), WgmmaTmaSmemLayoutB{});
  auto s_trellis = cute::make_tensor(
      cute::make_smem_ptr(shared.trellis.begin()), TrellisSmemLayout{});
  auto s_bank_ids = cute::make_tensor(
      cute::make_smem_ptr(shared.bank_ids.begin()), P32BankTmaSmemLayout{});

  auto full_input = input_tma.get_tma_tensor(cute::make_shape(cute::_16{}, size_k));
  auto tiled_input = cute::local_tile(
      full_input,
      cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(cute::_0{}, cute::_));
  auto [tma_global_input, tma_shared_input] = cute::tma_partition(
      input_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input),
      cute::group_modes<0, 2>(tiled_input));

  auto p32_full_trellis = trellis_tma.get_tma_tensor(
      cute::make_shape(cute::Int<kWordsPerP32Tile>{}, n_tiles, k_tiles));
  auto tiled_trellis = cute::local_tile(
      p32_full_trellis,
      cute::make_shape(
          cute::Int<kWordsPerP32Tile>{},
          cute::Int<kP32N16TilesPerBlock>{},
          cute::Int<kP32K16TilesPerStage>{}),
      cute::make_coord(cute::_0{}, n64_block, cute::_));
  auto [tma_global_trellis, tma_shared_trellis] = cute::tma_partition(
      trellis_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 3>(s_trellis),
      cute::group_modes<0, 3>(tiled_trellis));

  auto full_bank_ids = bank_tma.get_tma_tensor(cute::make_shape(n_tiles, k_tiles));
  auto tiled_bank_ids = cute::local_tile(
      full_bank_ids,
      cute::make_shape(cute::_16{}, cute::_16{}),
      cute::make_coord(n64_block >> 2, cute::_));
  auto [tma_global_bank_ids, tma_shared_bank_ids] = cute::tma_partition(
      bank_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_bank_ids),
      cute::group_modes<0, 2>(tiled_bank_ids));

  __syncthreads();

  if (is_producer) {
    if (cute::elect_one_sync()) {
      auto write_state = cutlass::make_producer_start_state<WgmmaTmaPipeline>();
      for (int stage_offset = 0; stage_offset < stage_count; ++stage_offset) {
        pipeline.producer_acquire(write_state);
        using Barrier = typename WgmmaTmaPipeline::ProducerBarrierType;
        Barrier* barrier = pipeline.producer_get_barrier(write_state);
        const int global_stage = stage_begin + stage_offset;
        const int write_stage = write_state.index();
        cute::copy(
            input_tma.with(*barrier),
            tma_global_input(cute::_, global_stage),
            tma_shared_input(cute::_, write_stage));
        cute::copy(
            trellis_tma.with(*barrier),
            tma_global_trellis(cute::_, global_stage),
            tma_shared_trellis(cute::_, write_stage));
        cute::copy(
            bank_tma.with(*barrier),
            tma_global_bank_ids(cute::_, global_stage),
            tma_shared_bank_ids(cute::_, write_stage));
        ++write_state;
      }
      pipeline.producer_tail(write_state);
    }
    return;
  }

  WgmmaTiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(thread);
  auto thread_shared_b = thread_mma.partition_B(s_input);
  auto fragment_b = thread_mma.make_fragment_B(thread_shared_b);
  static_assert(cute::size<2>(decltype(fragment_b){}) == 16);

  auto coordinate_a = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_a = thread_mma.partition_A(coordinate_a);
  auto fragment_a0 = cute::make_tensor<Element>(thread_coordinate_a.shape());
  auto fragment_a1 = cute::make_tensor<Element>(thread_coordinate_a.shape());
  static_assert(cute::size(decltype(fragment_a0){}) == 8);

  auto coordinate_c = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_c = thread_mma.partition_C(coordinate_c);
  auto accumulator = cute::make_tensor<float>(thread_coordinate_c.shape());
  cute::clear(accumulator);
  tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;

  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int bank_n16_offset = (n64_block & 3) * kP32N16TilesPerBlock;
  WgmmaTmaPipelineState read_state;
  WgmmaTmaPipelineState release_state;

  for (int stage_offset = 0; stage_offset < stage_count; ++stage_offset) {
    auto wait_token = pipeline.consumer_try_wait(read_state);
    pipeline.consumer_wait(read_state, wait_token);
    const int read_stage = read_state.index();

#pragma unroll
    for (int k_block = 0; k_block < kP32K16TilesPerStage; ++k_block) {
      auto& fragment_a = (k_block & 1) == 0 ? fragment_a0 : fragment_a1;
      uint32_t bank_id = 0;
      if (lane == 0) {
        bank_id = s_bank_ids(bank_n16_offset + warp, k_block, read_stage);
      }
      bank_id = __shfl_sync(0xffffffffu, bank_id, 0);
      if (k_block >= 2) {
        cute::warpgroup_wait<1>();
      }
      const auto trellis_layout = TrellisSmemLayout{};
      const uint32_t* window_words = shared.trellis.begin() +
          trellis_layout(0, warp, k_block, read_stage);
      qvq_p32_window_decode_fragment<TransitionBits>(
          fragment_a,
          window_words,
          static_cast<uint8_t>(bank_id),
          levels,
          alternate_bank_mask);
      cute::warpgroup_fence_operand(fragment_a);
      cute::warpgroup_arrive();
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b(cute::_, cute::_, k_block, read_stage),
          accumulator);
      tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
      cute::warpgroup_commit_batch();
    }
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(accumulator);
    pipeline.consumer_release(release_state);
    ++read_state;
    ++release_state;
  }

  constexpr int kAccumulatorValuesPerThread = cute::size(decltype(thread_coordinate_c){});
#pragma unroll
  for (int index = 0; index < kAccumulatorValuesPerThread; ++index) {
    const auto coordinate = thread_coordinate_c(index);
    const int wgmma_column = static_cast<int>(cute::get<0>(coordinate));
    const int output_row = static_cast<int>(cute::get<1>(coordinate));
    const int tile_column = wgmma_column & 15;
    const int p32_column = (wgmma_column & ~15) + ((tile_column & 7) << 1) + (tile_column >> 3);
    const int64_t output_index =
        static_cast<int64_t>(output_row) * size_n + n64_block * kOutputColumns + p32_column;
    partial_output[static_cast<int64_t>(split) * kRows * size_n + output_index] = accumulator(index);
  }
#endif
}

template <class InputTma, class TrellisTma, class BankTma>
__global__ __launch_bounds__(kTmaThreads) void qvq_wgmma_w3_m16_tma_kernel(
    CUTE_GRID_CONSTANT InputTma const input_tma,
    CUTE_GRID_CONSTANT TrellisTma const trellis_tma,
    CUTE_GRID_CONSTANT BankTma const bank_tma,
    const Element* __restrict__ levels,
    float* __restrict__ partial_output,
    int size_k,
    int size_n,
    int split_count,
    int bank_alt_id) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  __shared__ __align__(128) char shared_buffer[sizeof(WgmmaTmaSharedStorage)];
  auto& shared = *reinterpret_cast<WgmmaTmaSharedStorage*>(shared_buffer);

  const int thread = static_cast<int>(threadIdx.x);
  const bool is_consumer = thread < kThreads;
  const bool is_producer = !is_consumer;
  const int n64_block = static_cast<int>(blockIdx.x);
  const int n8_tile_base = n64_block * kN8TilesPerBlock;
  const int n_tiles = size_n / kN8Columns;
  const int k_tiles = size_k / kK32Rows;
  const int split = static_cast<int>(blockIdx.z);
  const int k_tile_begin = (k_tiles * split) / split_count;
  const int k_tile_end = (k_tiles * (split + 1)) / split_count;
  const int stage_begin = k_tile_begin / kK32TilesPerStage;
  const int stage_count = (k_tile_end - k_tile_begin) / kK32TilesPerStage;
  const uint32_t alternate_bank_mask = bank_alt_id == 0 ? 0u
      : bank_alt_id == 1 ? 0x6969u
      : bank_alt_id == 2 ? 0x5a5au
      : 0x3c3cu;

  typename WgmmaTmaPipeline::Params pipeline_params;
  pipeline_params.role = is_producer
      ? WgmmaTmaPipeline::ThreadCategory::Producer
      : WgmmaTmaPipeline::ThreadCategory::Consumer;
  pipeline_params.is_leader = thread == kThreads;
  pipeline_params.num_consumers = kThreads;
  pipeline_params.transaction_bytes =
      kRows * kKPerStage * sizeof(Element) +
      kWordsPerTile * kN8TilesPerBlock * kK32TilesPerStage * sizeof(uint32_t) +
      16 * kK32TilesPerStage * sizeof(uint8_t);
  WgmmaTmaPipeline pipeline(
      shared.pipeline,
      pipeline_params,
      cute::Shape<cute::_1, cute::_1, cute::_1>{});

  auto s_input = cute::make_tensor(
      cute::make_smem_ptr(shared.input.begin()), WgmmaTmaSmemLayoutB{});
  auto s_trellis = cute::make_tensor(
      cute::make_smem_ptr(shared.trellis.begin()), TrellisTmaSmemLayout{});
  auto s_bank_ids = cute::make_tensor(
      cute::make_smem_ptr(shared.bank_ids.begin()), BankTmaSmemLayout{});

  auto full_input = input_tma.get_tma_tensor(cute::make_shape(cute::_16{}, size_k));
  auto tiled_input = cute::local_tile(
      full_input,
      cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(cute::_0{}, cute::_));
  auto [tma_global_input, tma_shared_input] = cute::tma_partition(
      input_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input),
      cute::group_modes<0, 2>(tiled_input));

  auto full_trellis = trellis_tma.get_tma_tensor(
      cute::make_shape(cute::Int<kWordsPerTile>{}, n_tiles, k_tiles));
  auto tiled_trellis = cute::local_tile(
      full_trellis,
      cute::make_shape(
          cute::Int<kWordsPerTile>{},
          cute::Int<kN8TilesPerBlock>{},
          cute::Int<kK32TilesPerStage>{}),
      cute::make_coord(cute::_0{}, n64_block, cute::_));
  auto [tma_global_trellis, tma_shared_trellis] = cute::tma_partition(
      trellis_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 3>(s_trellis),
      cute::group_modes<0, 3>(tiled_trellis));

  auto full_bank_ids = bank_tma.get_tma_tensor(cute::make_shape(n_tiles, k_tiles));
  auto tiled_bank_ids = cute::local_tile(
      full_bank_ids,
      cute::make_shape(cute::_16{}, cute::_8{}),
      cute::make_coord(n64_block >> 1, cute::_));
  auto [tma_global_bank_ids, tma_shared_bank_ids] = cute::tma_partition(
      bank_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_bank_ids),
      cute::group_modes<0, 2>(tiled_bank_ids));

  // PipelineTmaAsync initializes its transaction and empty barriers from warp
  // zero.  All roles must observe that initialization before the producer and
  // consumer control flows diverge, especially once a two-stage buffer wraps.
  __syncthreads();

  if (is_producer) {
    if (cute::elect_one_sync()) {
      auto write_state = cutlass::make_producer_start_state<WgmmaTmaPipeline>();
      for (int stage_offset = 0; stage_offset < stage_count; ++stage_offset) {
        pipeline.producer_acquire(write_state);
        using Barrier = typename WgmmaTmaPipeline::ProducerBarrierType;
        Barrier* barrier = pipeline.producer_get_barrier(write_state);
        const int global_stage = stage_begin + stage_offset;
        const int write_stage = write_state.index();
        cute::copy(
            input_tma.with(*barrier),
            tma_global_input(cute::_, global_stage),
            tma_shared_input(cute::_, write_stage));
        cute::copy(
            trellis_tma.with(*barrier),
            tma_global_trellis(cute::_, global_stage),
            tma_shared_trellis(cute::_, write_stage));
        cute::copy(
            bank_tma.with(*barrier),
            tma_global_bank_ids(cute::_, global_stage),
            tma_shared_bank_ids(cute::_, write_stage));
        ++write_state;
      }
      pipeline.producer_tail(write_state);
    }
    return;
  }

  WgmmaTiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(thread);
  auto thread_shared_b = thread_mma.partition_B(s_input);
  auto fragment_b = thread_mma.make_fragment_B(thread_shared_b);
  static_assert(cute::size<2>(decltype(fragment_b){}) == 16);

  auto coordinate_a = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_a = thread_mma.partition_A(coordinate_a);
  auto fragment_a0 = cute::make_tensor<Element>(thread_coordinate_a.shape());
  auto fragment_a1 = cute::make_tensor<Element>(thread_coordinate_a.shape());
  static_assert(cute::size(decltype(fragment_a0){}) == 8);

  auto coordinate_c = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_c = thread_mma.partition_C(coordinate_c);
  auto accumulator = cute::make_tensor<float>(thread_coordinate_c.shape());
  cute::clear(accumulator);
  tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;

  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int column_in_n8 = lane >> 2;
  const int first_n8_tile = warp * 2;
  const int second_n8_tile = first_n8_tile + 1;
  const int bank_n8_offset = (n64_block & 1) * kN8TilesPerBlock;
  WgmmaTmaPipelineState read_state;
  WgmmaTmaPipelineState release_state;

  for (int stage_offset = 0; stage_offset < stage_count; ++stage_offset) {
    auto wait_token = pipeline.consumer_try_wait(read_state);
    pipeline.consumer_wait(read_state, wait_token);
    const int read_stage = read_state.index();

#pragma unroll
    for (int k_block = 0; k_block < 16; ++k_block) {
      const int k32_in_stage = k_block >> 1;
      const int k16_half = k_block & 1;
      auto& fragment_a = (k_block & 1) == 0 ? fragment_a0 : fragment_a1;
      uint32_t first_bank_id = 0;
      uint32_t second_bank_id = 0;
      if (lane == 0) {
        first_bank_id = s_bank_ids(
            bank_n8_offset + first_n8_tile, k32_in_stage, read_stage);
        second_bank_id = s_bank_ids(
            bank_n8_offset + second_n8_tile, k32_in_stage, read_stage);
      }
      first_bank_id = __shfl_sync(0xffffffffu, first_bank_id, 0);
      second_bank_id = __shfl_sync(0xffffffffu, second_bank_id, 0);

      if (k_block >= 2) {
        cute::warpgroup_wait<1>();
      }
      const auto trellis_layout = TrellisTmaSmemLayout{};
      const uint32_t* packed_first = shared.trellis.begin() +
          trellis_layout(0, first_n8_tile, k32_in_stage, read_stage);
      const uint32_t* packed_second = shared.trellis.begin() +
          trellis_layout(0, second_n8_tile, k32_in_stage, read_stage);
      qvq_wgmma_decode_w3_fragment(
          fragment_a,
          packed_first,
          packed_second,
          static_cast<uint8_t>(first_bank_id),
          static_cast<uint8_t>(second_bank_id),
          levels,
          column_in_n8,
          k16_half,
          alternate_bank_mask);
      cute::warpgroup_fence_operand(fragment_a);
      cute::warpgroup_arrive();
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b(cute::_, cute::_, k_block, read_stage),
          accumulator);
      tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
      cute::warpgroup_commit_batch();
    }
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(accumulator);
    pipeline.consumer_release(release_state);
    ++read_state;
    ++release_state;
  }

  constexpr int kAccumulatorValuesPerThread = cute::size(decltype(thread_coordinate_c){});
#pragma unroll
  for (int index = 0; index < kAccumulatorValuesPerThread; ++index) {
    const auto coordinate = thread_coordinate_c(index);
    const int output_column_in_block = static_cast<int>(cute::get<0>(coordinate));
    const int output_row = static_cast<int>(cute::get<1>(coordinate));
    const int64_t output_index =
        static_cast<int64_t>(output_row) * size_n + n64_block * kOutputColumns + output_column_in_block;
    partial_output[static_cast<int64_t>(split) * kRows * size_n + output_index] = accumulator(index);
  }
#endif
}

__global__ void qvq_wgmma_reduce_split_kernel(
    const float* __restrict__ partial_output,
    float* __restrict__ output,
    int output_values,
    int split_count) {
  const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
      static_cast<int>(threadIdx.x);
  if (index >= output_values) {
    return;
  }
  float value = 0.0f;
  for (int split = 0; split < split_count; ++split) {
    value += partial_output[static_cast<int64_t>(split) * output_values + index];
  }
  output[index] = value;
}

at::Tensor qvq_wgmma_w3_m16(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t out_features,
    int64_t bank_alt_id,
    int64_t split_count) {
  TORCH_CHECK(input.is_cuda(), "QVQ WGMMA input must be CUDA");
  c10::cuda::CUDAGuard device_guard(input.device());
  TORCH_CHECK(trellis.device() == input.device() && levels.device() == input.device() &&
                  bank_ids.device() == input.device(),
              "QVQ WGMMA tensors must share one CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kHalf && levels.scalar_type() == at::kHalf,
              "QVQ WGMMA prototype requires FP16 input and levels");
  TORCH_CHECK(trellis.scalar_type() == at::kInt, "QVQ WGMMA trellis must be int32");
  TORCH_CHECK(bank_ids.scalar_type() == at::kByte, "QVQ WGMMA bank ids must be uint8");
  TORCH_CHECK(input.is_contiguous() && trellis.is_contiguous() && levels.is_contiguous() &&
                  bank_ids.is_contiguous(),
              "QVQ WGMMA tensors must be contiguous");
  TORCH_CHECK(input.dim() == 2 && input.size(0) == kRows,
              "QVQ WGMMA prototype requires M=16");
  TORCH_CHECK(out_features > 0 && out_features % kOutputColumns == 0,
              "QVQ WGMMA output features must be a positive multiple of 64");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) % kKPerStage == 0,
              "QVQ WGMMA input features must be a positive multiple of 256");
  TORCH_CHECK(split_count >= 1 && split_count <= 64,
              "QVQ WGMMA split count must be in [1, 64]");
  TORCH_CHECK(bank_alt_id >= 0 && bank_alt_id <= 3,
              "QVQ WGMMA alternate bank id must be in [0, 3]");

  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(properties.major == 9 && properties.minor == 0,
              "QVQ WGMMA prototype requires an SM90 H100/H200 device");

  const int size_k = static_cast<int>(input.size(1));
  const int size_n = static_cast<int>(out_features);
  const int k_tiles = size_k / kK32Rows;
  const int n_tiles = size_n / kN8Columns;
  TORCH_CHECK(k_tiles % split_count == 0 && (k_tiles / split_count) % kK32TilesPerStage == 0,
              "QVQ WGMMA split partitions must contain a multiple of eight K32 tiles");
  const int64_t expected_tiles = static_cast<int64_t>(k_tiles) * n_tiles;
  TORCH_CHECK(trellis.numel() == expected_tiles * kWordsPerTile,
              "QVQ WGMMA trellis size mismatch");
  TORCH_CHECK(bank_ids.numel() == expected_tiles,
              "QVQ WGMMA bank-id size mismatch");
  TORCH_CHECK(levels.numel() == 256, "QVQ WGMMA requires 256 PGC16 levels");

  auto output = at::empty({kRows, size_n}, input.options().dtype(at::kFloat));
  // A single K partition already produces the final FP32 result.  Alias the
  // output directly so the common split=1 path does not pay for a redundant
  // conversion/reduction kernel launch.
  auto partial_output = split_count == 1
      ? output
      : at::empty({split_count, kRows, size_n}, input.options().dtype(at::kFloat));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const dim3 grid(static_cast<unsigned>(size_n / kOutputColumns), 1, static_cast<unsigned>(split_count));
  qvq_wgmma_w3_m16_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const Element*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
      bank_ids.data_ptr<uint8_t>(),
      reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
      partial_output.data_ptr<float>(),
      size_k,
      size_n,
      static_cast<int>(split_count),
      static_cast<int>(bank_alt_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (split_count > 1) {
    constexpr int kReductionThreads = 256;
    const int output_values = kRows * size_n;
    const int reduction_blocks = (output_values + kReductionThreads - 1) / kReductionThreads;
    qvq_wgmma_reduce_split_kernel<<<reduction_blocks, kReductionThreads, 0, stream>>>(
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        output_values,
        static_cast<int>(split_count));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

at::Tensor qvq_p32_window_wgmma_w3_m16(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t out_features,
    int64_t bank_alt_id,
    int64_t split_count) {
  TORCH_CHECK(input.is_cuda(), "QVQ P32 WGMMA input must be CUDA");
  c10::cuda::CUDAGuard device_guard(input.device());
  TORCH_CHECK(trellis.device() == input.device() && levels.device() == input.device() &&
                  bank_ids.device() == input.device(),
              "QVQ P32 WGMMA tensors must share one CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kHalf && levels.scalar_type() == at::kHalf,
              "QVQ P32 WGMMA prototype requires FP16 input and levels");
  TORCH_CHECK(trellis.scalar_type() == at::kInt, "QVQ P32 WGMMA trellis must be int32");
  TORCH_CHECK(bank_ids.scalar_type() == at::kByte, "QVQ P32 WGMMA bank ids must be uint8");
  TORCH_CHECK(input.is_contiguous() && trellis.is_contiguous() && levels.is_contiguous() &&
                  bank_ids.is_contiguous(),
              "QVQ P32 WGMMA tensors must be contiguous");
  TORCH_CHECK(input.dim() == 2 && input.size(0) == kRows,
              "QVQ P32 WGMMA prototype requires M=16");
  TORCH_CHECK(out_features > 0 && out_features % kOutputColumns == 0,
              "QVQ P32 WGMMA output features must be a positive multiple of 64");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) % kKPerStage == 0,
              "QVQ P32 WGMMA input features must be a positive multiple of 256");
  TORCH_CHECK(split_count >= 1 && split_count <= 64,
              "QVQ P32 WGMMA split count must be in [1, 64]");
  TORCH_CHECK(bank_alt_id >= 0 && bank_alt_id <= 3,
              "QVQ P32 WGMMA alternate bank id must be in [0, 3]");

  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(properties.major == 9 && properties.minor == 0,
              "QVQ P32 WGMMA prototype requires an SM90 H100/H200 device");

  const int size_k = static_cast<int>(input.size(1));
  const int size_n = static_cast<int>(out_features);
  const int k_tiles = size_k / kP32TileRows;
  const int n_tiles = size_n / kP32TileColumns;
  TORCH_CHECK(k_tiles % split_count == 0 &&
                  (k_tiles / split_count) % kP32K16TilesPerStage == 0,
              "QVQ P32 WGMMA split partitions must contain a multiple of sixteen K16 tiles");
  const int64_t expected_tiles = static_cast<int64_t>(k_tiles) * n_tiles;
  constexpr int kWordsPerP32Tile = 4 * kW3TransitionBits;
  TORCH_CHECK(trellis.numel() == expected_tiles * kWordsPerP32Tile,
              "QVQ P32 WGMMA trellis size mismatch");
  TORCH_CHECK(bank_ids.numel() == expected_tiles,
              "QVQ P32 WGMMA bank-id size mismatch");
  TORCH_CHECK(levels.numel() == 256, "QVQ P32 WGMMA requires 256 PGC16 levels");

  auto output = at::empty({kRows, size_n}, input.options().dtype(at::kFloat));
  auto partial_output = split_count == 1
      ? output
      : at::empty({split_count, kRows, size_n}, input.options().dtype(at::kFloat));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const dim3 grid(static_cast<unsigned>(size_n / kOutputColumns), 1, static_cast<unsigned>(split_count));
  qvq_p32_window_wgmma_w3_m16_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const Element*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
      bank_ids.data_ptr<uint8_t>(),
      reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
      partial_output.data_ptr<float>(),
      size_k,
      size_n,
      static_cast<int>(split_count),
      static_cast<int>(bank_alt_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (split_count > 1) {
    constexpr int kReductionThreads = 256;
    const int output_values = kRows * size_n;
    const int reduction_blocks = (output_values + kReductionThreads - 1) / kReductionThreads;
    qvq_wgmma_reduce_split_kernel<<<reduction_blocks, kReductionThreads, 0, stream>>>(
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        output_values,
        static_cast<int>(split_count));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

template <int TransitionBits>
at::Tensor qvq_p32_window_wgmma_m16_tma_impl(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t out_features,
    int64_t bank_alt_id,
    int64_t split_count) {
  TORCH_CHECK(input.is_cuda(), "QVQ P32 TMA WGMMA input must be CUDA");
  c10::cuda::CUDAGuard device_guard(input.device());
  TORCH_CHECK(trellis.device() == input.device() && levels.device() == input.device() &&
                  bank_ids.device() == input.device(),
              "QVQ P32 TMA WGMMA tensors must share one CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kHalf && levels.scalar_type() == at::kHalf,
              "QVQ P32 TMA WGMMA prototype requires FP16 input and levels");
  TORCH_CHECK(trellis.scalar_type() == at::kInt, "QVQ P32 TMA WGMMA trellis must be int32");
  TORCH_CHECK(bank_ids.scalar_type() == at::kByte, "QVQ P32 TMA WGMMA bank ids must be uint8");
  TORCH_CHECK(input.is_contiguous() && trellis.is_contiguous() && levels.is_contiguous() &&
                  bank_ids.is_contiguous(),
              "QVQ P32 TMA WGMMA tensors must be contiguous");
  TORCH_CHECK(input.dim() == 2 && input.size(0) == kRows,
              "QVQ P32 TMA WGMMA prototype requires M=16");
  TORCH_CHECK(out_features > 0 && out_features % 256 == 0,
              "QVQ P32 TMA WGMMA output features must be a positive multiple of 256");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) % kKPerStage == 0,
              "QVQ P32 TMA WGMMA input features must be a positive multiple of 256");
  TORCH_CHECK(split_count >= 1 && split_count <= 64,
              "QVQ P32 TMA WGMMA split count must be in [1, 64]");
  TORCH_CHECK(bank_alt_id >= 0 && bank_alt_id <= 3,
              "QVQ P32 TMA WGMMA alternate bank id must be in [0, 3]");

  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(properties.major == 9 && properties.minor == 0,
              "QVQ P32 TMA WGMMA prototype requires an SM90 H100/H200 device");

  const int size_k = static_cast<int>(input.size(1));
  const int size_n = static_cast<int>(out_features);
  const int k_tiles = size_k / kP32TileRows;
  const int n_tiles = size_n / kP32TileColumns;
  TORCH_CHECK(k_tiles % split_count == 0 &&
                  (k_tiles / split_count) % kP32K16TilesPerStage == 0,
              "QVQ P32 TMA WGMMA split partitions must contain a multiple of sixteen K16 tiles");
  const int64_t expected_tiles = static_cast<int64_t>(k_tiles) * n_tiles;
  constexpr int kWordsPerP32Tile = 4 * TransitionBits;
  using TrellisSmemLayout = P32TrellisTmaSmemLayoutFor<TransitionBits>;
  TORCH_CHECK(trellis.numel() == expected_tiles * kWordsPerP32Tile,
              "QVQ P32 TMA WGMMA trellis size mismatch");
  TORCH_CHECK(bank_ids.numel() == expected_tiles,
              "QVQ P32 TMA WGMMA bank-id size mismatch");
  TORCH_CHECK(levels.numel() == 256, "QVQ P32 TMA WGMMA requires 256 PGC16 levels");

  const auto* input_ptr = reinterpret_cast<const Element*>(input.data_ptr<at::Half>());
  const auto* trellis_ptr = reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>());
  auto input_tensor = cute::make_tensor(
      input_ptr,
      cute::make_shape(kRows, size_k),
      cute::make_stride(static_cast<int64_t>(size_k), cute::_1{}));
  auto trellis_tensor = cute::make_tensor(
      trellis_ptr,
      cute::make_shape(kWordsPerP32Tile, n_tiles, k_tiles),
      cute::make_stride(
          cute::_1{},
          cute::Int<kWordsPerP32Tile>{},
          static_cast<int64_t>(n_tiles) * kWordsPerP32Tile));
  auto bank_tensor = cute::make_tensor(
      bank_ids.data_ptr<uint8_t>(),
      cute::make_shape(n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, static_cast<int64_t>(n_tiles)));
  auto input_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{},
      input_tensor,
      WgmmaTmaSmemLayoutB{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_256{}));
  auto trellis_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{},
      trellis_tensor,
      TrellisSmemLayout{}(cute::_, cute::_, cute::_, cute::_0{}),
      cute::make_shape(
          cute::Int<kWordsPerP32Tile>{},
          cute::Int<kP32N16TilesPerBlock>{},
          cute::Int<kP32K16TilesPerStage>{}));
  auto bank_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{},
      bank_tensor,
      P32BankTmaSmemLayout{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_16{}));

  auto output = at::empty({kRows, size_n}, input.options().dtype(at::kFloat));
  auto partial_output = split_count == 1
      ? output
      : at::empty({split_count, kRows, size_n}, input.options().dtype(at::kFloat));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const dim3 grid(static_cast<unsigned>(size_n / kOutputColumns), 1, static_cast<unsigned>(split_count));
  qvq_p32_window_wgmma_m16_tma_kernel<TransitionBits><<<grid, kTmaThreads, 0, stream>>>(
      input_tma,
      trellis_tma,
      bank_tma,
      reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
      partial_output.data_ptr<float>(),
      size_k,
      size_n,
      static_cast<int>(split_count),
      static_cast<int>(bank_alt_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (split_count > 1) {
    constexpr int kReductionThreads = 256;
    const int output_values = kRows * size_n;
    const int reduction_blocks = (output_values + kReductionThreads - 1) / kReductionThreads;
    qvq_wgmma_reduce_split_kernel<<<reduction_blocks, kReductionThreads, 0, stream>>>(
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        output_values,
        static_cast<int>(split_count));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

at::Tensor qvq_p32_window_wgmma_w3_m16_tma(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t out_features,
    int64_t bank_alt_id,
    int64_t split_count) {
  return qvq_p32_window_wgmma_m16_tma_impl<kW3TransitionBits>(
      input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
}

at::Tensor qvq_p32_window_wgmma_m16_tma(
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
      return qvq_p32_window_wgmma_m16_tma_impl<4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 5:
      return qvq_p32_window_wgmma_m16_tma_impl<5>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 6:
      return qvq_p32_window_wgmma_m16_tma_impl<6>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 7:
      return qvq_p32_window_wgmma_m16_tma_impl<7>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    default:
      TORCH_CHECK(false, "QVQ P32 TMA WGMMA transition bits must be in [4, 7]");
  }
}

at::Tensor qvq_wgmma_w3_m16_tma(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t out_features,
    int64_t bank_alt_id,
    int64_t split_count) {
  TORCH_CHECK(input.is_cuda(), "QVQ TMA WGMMA input must be CUDA");
  c10::cuda::CUDAGuard device_guard(input.device());
  TORCH_CHECK(trellis.device() == input.device() && levels.device() == input.device() &&
                  bank_ids.device() == input.device(),
              "QVQ TMA WGMMA tensors must share one CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kHalf && levels.scalar_type() == at::kHalf,
              "QVQ TMA WGMMA prototype requires FP16 input and levels");
  TORCH_CHECK(trellis.scalar_type() == at::kInt, "QVQ TMA WGMMA trellis must be int32");
  TORCH_CHECK(bank_ids.scalar_type() == at::kByte, "QVQ TMA WGMMA bank ids must be uint8");
  TORCH_CHECK(input.is_contiguous() && trellis.is_contiguous() && levels.is_contiguous() &&
                  bank_ids.is_contiguous(),
              "QVQ TMA WGMMA tensors must be contiguous");
  TORCH_CHECK(input.dim() == 2 && input.size(0) == kRows,
              "QVQ TMA WGMMA prototype requires M=16");
  TORCH_CHECK(out_features > 0 && out_features % kOutputColumns == 0,
              "QVQ TMA WGMMA output features must be a positive multiple of 64");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) % kKPerStage == 0,
              "QVQ TMA WGMMA input features must be a positive multiple of 256");
  TORCH_CHECK(split_count >= 1 && split_count <= 64,
              "QVQ TMA WGMMA split count must be in [1, 64]");
  TORCH_CHECK(bank_alt_id >= 0 && bank_alt_id <= 3,
              "QVQ TMA WGMMA alternate bank id must be in [0, 3]");

  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(properties.major == 9 && properties.minor == 0,
              "QVQ TMA WGMMA prototype requires an SM90 H100/H200 device");

  const int size_k = static_cast<int>(input.size(1));
  const int size_n = static_cast<int>(out_features);
  const int k_tiles = size_k / kK32Rows;
  const int n_tiles = size_n / kN8Columns;
  TORCH_CHECK(k_tiles % split_count == 0 && (k_tiles / split_count) % kK32TilesPerStage == 0,
              "QVQ TMA WGMMA split partitions must contain a multiple of eight K32 tiles");
  const int64_t expected_tiles = static_cast<int64_t>(k_tiles) * n_tiles;
  TORCH_CHECK(trellis.numel() == expected_tiles * kWordsPerTile,
              "QVQ TMA WGMMA trellis size mismatch");
  TORCH_CHECK(bank_ids.numel() == expected_tiles,
              "QVQ TMA WGMMA bank-id size mismatch");
  TORCH_CHECK(levels.numel() == 256, "QVQ TMA WGMMA requires 256 PGC16 levels");

  const auto* input_ptr = reinterpret_cast<const Element*>(input.data_ptr<at::Half>());
  const auto* trellis_ptr = reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>());
  auto input_tensor = cute::make_tensor(
      input_ptr,
      cute::make_shape(kRows, size_k),
      cute::make_stride(static_cast<int64_t>(size_k), cute::_1{}));
  auto trellis_tensor = cute::make_tensor(
      trellis_ptr,
      cute::make_shape(kWordsPerTile, n_tiles, k_tiles),
      cute::make_stride(
          cute::_1{},
          cute::Int<kWordsPerTile>{},
          static_cast<int64_t>(n_tiles) * kWordsPerTile));
  auto bank_tensor = cute::make_tensor(
      bank_ids.data_ptr<uint8_t>(),
      cute::make_shape(n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, static_cast<int64_t>(n_tiles)));
  auto input_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{},
      input_tensor,
      WgmmaTmaSmemLayoutB{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_256{}));
  auto trellis_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{},
      trellis_tensor,
      TrellisTmaSmemLayout{}(cute::_, cute::_, cute::_, cute::_0{}),
      cute::make_shape(
          cute::Int<kWordsPerTile>{},
          cute::Int<kN8TilesPerBlock>{},
          cute::Int<kK32TilesPerStage>{}));
  auto bank_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{},
      bank_tensor,
      BankTmaSmemLayout{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_8{}));

  auto output = at::empty({kRows, size_n}, input.options().dtype(at::kFloat));
  auto partial_output = split_count == 1
      ? output
      : at::empty({split_count, kRows, size_n}, input.options().dtype(at::kFloat));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const dim3 grid(static_cast<unsigned>(size_n / kOutputColumns), 1, static_cast<unsigned>(split_count));
  qvq_wgmma_w3_m16_tma_kernel<<<grid, kTmaThreads, 0, stream>>>(
      input_tma,
      trellis_tma,
      bank_tma,
      reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
      partial_output.data_ptr<float>(),
      size_k,
      size_n,
      static_cast<int>(split_count),
      static_cast<int>(bank_alt_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (split_count > 1) {
    constexpr int kReductionThreads = 256;
    const int output_values = kRows * size_n;
    const int reduction_blocks = (output_values + kReductionThreads - 1) / kReductionThreads;
    qvq_wgmma_reduce_split_kernel<<<reduction_blocks, kReductionThreads, 0, stream>>>(
        partial_output.data_ptr<float>(),
        output.data_ptr<float>(),
        output_values,
        static_cast<int>(split_count));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq_wgmma, m) {
  m.def("w3_m16(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("w3_m16_tma(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("p32_window_w3_m16(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("p32_window_w3_m16_tma(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("p32_window_m16_tma(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq_wgmma, CUDA, m) {
  m.impl("w3_m16", qvq_wgmma_w3_m16);
  m.impl("w3_m16_tma", qvq_wgmma_w3_m16_tma);
  m.impl("p32_window_w3_m16", qvq_p32_window_wgmma_w3_m16);
  m.impl("p32_window_w3_m16_tma", qvq_p32_window_wgmma_w3_m16_tma);
  m.impl("p32_window_m16_tma", qvq_p32_window_wgmma_m16_tma);
}
