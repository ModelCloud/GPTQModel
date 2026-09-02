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
#include <limits>

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
constexpr int kW3TransitionBits = 6;
constexpr int kKPerStage = 256;
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
constexpr int kMaxGroupedP32Segments = 3;

struct HopperGroupedP32LaunchParams {
  int segment_count;
  int n_tile_start[kMaxGroupedP32Segments];
  int n_tiles[kMaxGroupedP32Segments];
  int bank_alt_id[kMaxGroupedP32Segments];
  int split_count[kMaxGroupedP32Segments];
  int work_item_start[kMaxGroupedP32Segments];
  int64_t output_offset[kMaxGroupedP32Segments];
  int64_t partial_output_offset[kMaxGroupedP32Segments];
};

using WgmmaTmaSmemLayoutB = decltype(cute::tile_to_shape(
    WgmmaSmemLayoutAtomB{},
    cute::make_shape(cute::_16{}, cute::_256{}, cute::Int<kTmaStages>{})));
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

template <int TransitionBits>
struct alignas(128) P32WgmmaTmaSharedStorageFor {
  typename WgmmaTmaPipeline::SharedStorage pipeline;
  alignas(128) cute::ArrayEngine<Element, cute::cosize_v<WgmmaTmaSmemLayoutB>> input;
  alignas(128) cute::ArrayEngine<
      uint32_t,
      cute::cosize_v<P32TrellisTmaSmemLayoutFor<TransitionBits>>> trellis;
  alignas(128) cute::ArrayEngine<uint8_t, cute::cosize_v<P32BankTmaSmemLayout>> bank_ids;
  // Reused by every decode lane and K16 tile; avoid dependent L1/global
  // lookups for the small, read-only PGC level table.
  alignas(128) cute::ArrayEngine<Element, 256> levels;
  // The PGC high byte is b ^ (b >> 7).  Store that fixed permutation once so
  // the hot loop can index it directly with affine-product byte 1.
  alignas(128) cute::ArrayEngine<Element, 256> levels_high;
};

static_assert(cute::size(WgmmaTiledMma{}) == kThreads);

__device__ __forceinline__ uint32_t qvq_wgmma_pgc16_mix(uint32_t state) {
  uint32_t mixed = state ^ (state >> 8);
  mixed = (mixed * kPgc16Multiplier + kPgc16Increment) & 0xffffu;
  return mixed ^ (mixed >> 7);
}

__device__ __forceinline__ uint32_t qvq_wgmma_pgc16_product_masked(
    uint32_t state,
    uint32_t bank_mask) {
  // Every supported alternate-bank mask repeats one byte in both halves.
  // Therefore, for x = state ^ bank_mask:
  //   x ^ (x >> 8) == state ^ (state >> 8) ^ (bank_mask & 0xff00).
  // The affine PGC step is reduced modulo 2^16, so bits above the low state
  // word are irrelevant.  Extract just state byte 1 instead of first masking
  // the raw circular-window funnel result and shifting that temporary again.
  const uint32_t high_byte = __byte_perm(state, 0u, 0x4441u);
  uint32_t mixed = state ^ high_byte ^ bank_mask;
  uint32_t product;
  asm("mad.lo.u32 %0, %1, 40503, 17011;"
      : "=r"(product)
      : "r"(mixed));
  return product;
}

__device__ __forceinline__ uint32_t qvq_wgmma_pgc16_finish(uint32_t mixed) {
  uint32_t shifted;
  asm("bfe.u32 %0, %1, 7, 9;" : "=r"(shifted) : "r"(mixed));
  return mixed ^ shifted;
}

__device__ __forceinline__ uint32_t qvq_wgmma_high_byte(uint32_t value) {
  return __byte_perm(value, 0u, 0x4441u);
}

__device__ __forceinline__ uint32_t qvq_wgmma_pgc16_low_byte_offset(uint32_t& product) {
  // Form the two-byte table offset directly:
  //   2 * ((product ^ (product >> 7)) & 0xff)
  //     == ((product << 1) ^ (product >> 6)) & 0x1fe.
  // Distributing the scale removes a separate post-XOR shift in the hot path.
  uint32_t shifted;
  asm("shr.u32 %0, %1, 6;" : "=r"(shifted) : "r"(product));
  asm("shl.b32 %0, %0, 1;" : "+r"(product));
  // 0x28 is (a ^ b) & c for PTX lop3 inputs a, b, c.
  asm("lop3.b32 %0, %0, %1, 0x1fe, 0x28;"
      : "+r"(product)
      : "r"(shifted));
  return product;
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

__device__ __forceinline__ Element qvq_wgmma_load_level_shared(
    uint32_t levels_base,
    uint32_t index) {
  const uint32_t address = levels_base + index * sizeof(Element);
  uint16_t bits;
  asm("ld.shared.u16 %0, [%1];" : "=h"(bits) : "r"(address));
  return Element::bitcast(bits);
}

__device__ __forceinline__ Element qvq_wgmma_load_level_shared_byte_offset(
    uint32_t levels_base,
    uint32_t byte_offset) {
  const uint32_t address = levels_base + byte_offset;
  uint16_t bits;
  asm("ld.shared.u16 %0, [%1];" : "=h"(bits) : "r"(address));
  return Element::bitcast(bits);
}

template <int TransitionBits, bool LevelsInShared>
__device__ __forceinline__ Element qvq_wgmma_decode_level(
    const Element* __restrict__ levels,
    uint32_t levels_shared_base,
    uint32_t index) {
  if constexpr (LevelsInShared) {
    return qvq_wgmma_load_level_shared(levels_shared_base, index);
  } else if constexpr (TransitionBits == kW3TransitionBits) {
    return qvq_wgmma_load_level(levels, index);
  } else {
    return levels[index];
  }
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
  first = static_cast<uint32_t>(first_window >> shift);
  second = static_cast<uint32_t>(second_window >> shift);
}

// W3.5 is the only rate whose seven-bit windows cross enough word boundaries
// to make the lane geometry arithmetic visible in the hot loop.  Hoist the
// two word pairs and shifts once per lane; lower rates retain the compact
// generic path that ptxas already optimizes well.
struct QvqP32WindowLanePlan {
  int first_word0;
  int first_next_word0;
  int shift0;
  int first_word1;
  int first_next_word1;
  int shift1;
  int bank_shift;
};

template <int TransitionBits>
__device__ __forceinline__ QvqP32WindowLanePlan qvq_p32_window_lane_plan(int lane) {
  constexpr int kWordsPerP32Tile = 4 * TransitionBits;
  const int n_pair = lane >> 2;
  const int k_pair = lane & 3;
  const int pair0 = k_pair * 16 + n_pair;
  const int bit_position0 = (kP32PairsPerTile - 1 - pair0) * TransitionBits;
  const int bit_position1 = bit_position0 - 8 * TransitionBits;
  const int first_word0 = bit_position0 >> 5;
  const int first_word1 = bit_position1 >> 5;
  return {
      first_word0,
      first_word0 + 1 == kWordsPerP32Tile ? 0 : first_word0 + 1,
      bit_position0 & 31,
      first_word1,
      first_word1 + 1,
      bit_position1 & 31,
      k_pair,
  };
}

template <int TransitionBits>
__device__ __forceinline__ void qvq_p32_window_state_pair_planned(
    const uint32_t* __restrict__ window_words,
    int first_word,
    int first_next_word,
    int shift,
    uint32_t& first,
    uint32_t& second) {
  constexpr int kKPairWordDistance = 2 * TransitionBits;
  const int second_word = first_word - kKPairWordDistance;
  const uint64_t first_window = static_cast<uint64_t>(window_words[first_word]) |
      (static_cast<uint64_t>(window_words[first_next_word]) << 32);
  const uint64_t second_window = static_cast<uint64_t>(window_words[second_word]) |
      (static_cast<uint64_t>(window_words[second_word + 1]) << 32);
  first = static_cast<uint32_t>(first_window >> shift);
  second = static_cast<uint32_t>(second_window >> shift);
}

template <int TransitionBits, bool LevelsInShared, class FragmentA>
__device__ __forceinline__ void qvq_p32_window_decode_fragment(
    FragmentA& fragment,
    const uint32_t* __restrict__ window_words,
    const QvqP32WindowLanePlan& plan,
    uint8_t bank_id,
    const Element* __restrict__ levels,
    uint32_t levels_shared_base,
    uint32_t levels_high_shared_base,
    uint32_t alternate_bank_mask) {
  uint32_t state00;
  uint32_t state01;
  uint32_t state10;
  uint32_t state11;
  uint32_t bank_pair_bits;
  if constexpr (TransitionBits == 7) {
    bank_pair_bits = static_cast<uint32_t>(bank_id) >> plan.bank_shift;
    qvq_p32_window_state_pair_planned<TransitionBits>(
        window_words, plan.first_word0, plan.first_next_word0, plan.shift0, state00, state10);
    qvq_p32_window_state_pair_planned<TransitionBits>(
        window_words, plan.first_word1, plan.first_next_word1, plan.shift1, state01, state11);
  } else {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int n_pair = lane >> 2;
    const int k_pair0 = lane & 3;
    const int pair00 = k_pair0 * 16 + n_pair;
    const int pair01 = pair00 + 8;
    bank_pair_bits = static_cast<uint32_t>(bank_id) >> k_pair0;
    qvq_p32_window_state_pair<TransitionBits>(window_words, pair00, state00, state10);
    qvq_p32_window_state_pair<TransitionBits>(window_words, pair01, state01, state11);
  }
  const uint32_t alternate_mix_mask = alternate_bank_mask & 0xff00u;
  const uint32_t bank_mask0 = (bank_pair_bits & 1u) * alternate_mix_mask;
  const uint32_t bank_mask1 = ((bank_pair_bits >> 4) & 1u) * alternate_mix_mask;
  uint32_t product00 = qvq_wgmma_pgc16_product_masked(state00, bank_mask0);
  uint32_t product01 = qvq_wgmma_pgc16_product_masked(state01, bank_mask0);
  uint32_t product10 = qvq_wgmma_pgc16_product_masked(state10, bank_mask1);
  uint32_t product11 = qvq_wgmma_pgc16_product_masked(state11, bank_mask1);

  // CuTe maps each lane to two A rows and two K pairs.  Map those two rows to
  // adjacent P32 N values so each decoded state feeds both output columns.
  if constexpr (LevelsInShared) {
    fragment(0) = qvq_wgmma_load_level_shared(levels_high_shared_base, qvq_wgmma_high_byte(product00));
    fragment(1) = qvq_wgmma_load_level_shared(levels_high_shared_base, qvq_wgmma_high_byte(product01));
    fragment(2) = qvq_wgmma_load_level_shared_byte_offset(
        levels_shared_base, qvq_wgmma_pgc16_low_byte_offset(product00));
    fragment(3) = qvq_wgmma_load_level_shared_byte_offset(
        levels_shared_base, qvq_wgmma_pgc16_low_byte_offset(product01));
    fragment(4) = qvq_wgmma_load_level_shared(levels_high_shared_base, qvq_wgmma_high_byte(product10));
    fragment(5) = qvq_wgmma_load_level_shared(levels_high_shared_base, qvq_wgmma_high_byte(product11));
    fragment(6) = qvq_wgmma_load_level_shared_byte_offset(
        levels_shared_base, qvq_wgmma_pgc16_low_byte_offset(product10));
    fragment(7) = qvq_wgmma_load_level_shared_byte_offset(
        levels_shared_base, qvq_wgmma_pgc16_low_byte_offset(product11));
  } else {
    const uint32_t mixed00 = qvq_wgmma_pgc16_finish(product00);
    const uint32_t mixed01 = qvq_wgmma_pgc16_finish(product01);
    const uint32_t mixed10 = qvq_wgmma_pgc16_finish(product10);
    const uint32_t mixed11 = qvq_wgmma_pgc16_finish(product11);
    fragment(0) = qvq_wgmma_decode_level<TransitionBits, false>(levels, 0, qvq_wgmma_high_byte(mixed00));
    fragment(1) = qvq_wgmma_decode_level<TransitionBits, false>(levels, 0, qvq_wgmma_high_byte(mixed01));
    fragment(2) = qvq_wgmma_decode_level<TransitionBits, false>(levels, 0, mixed00 & 0xffu);
    fragment(3) = qvq_wgmma_decode_level<TransitionBits, false>(levels, 0, mixed01 & 0xffu);
    fragment(4) = qvq_wgmma_decode_level<TransitionBits, false>(levels, 0, qvq_wgmma_high_byte(mixed10));
    fragment(5) = qvq_wgmma_decode_level<TransitionBits, false>(levels, 0, qvq_wgmma_high_byte(mixed11));
    fragment(6) = qvq_wgmma_decode_level<TransitionBits, false>(levels, 0, mixed10 & 0xffu);
    fragment(7) = qvq_wgmma_decode_level<TransitionBits, false>(levels, 0, mixed11 & 0xffu);
  }
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
  const auto decode_plan = qvq_p32_window_lane_plan<kW3TransitionBits>(thread & 31);

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
      qvq_p32_window_decode_fragment<kW3TransitionBits, false>(
          fragment_a,
          &packed_words[k_block][warp][0],
          decode_plan,
          packed_bank_ids[k_block][warp],
          levels,
          0,
          0,
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

template <
    int TransitionBits,
    bool Grouped,
    bool OrderedSplit,
    class InputTma,
    class TrellisTma,
    class BankTma>
__global__ __launch_bounds__(kTmaThreads) void qvq_p32_window_wgmma_m16_tma_kernel(
    CUTE_GRID_CONSTANT InputTma const input_tma,
    CUTE_GRID_CONSTANT TrellisTma const trellis_tma,
    CUTE_GRID_CONSTANT BankTma const bank_tma,
    const Element* __restrict__ levels,
    float* __restrict__ partial_output,
    HopperGroupedP32LaunchParams grouped_params,
    int size_k,
    int launch_size_n,
    int launch_split_count,
    int launch_bank_alt_id) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  constexpr int kWordsPerP32Tile = 4 * TransitionBits;
  using TrellisSmemLayout = P32TrellisTmaSmemLayoutFor<TransitionBits>;
  using SharedStorage = P32WgmmaTmaSharedStorageFor<TransitionBits>;
  __shared__ __align__(128) char shared_buffer[sizeof(SharedStorage)];
  auto& shared = *reinterpret_cast<SharedStorage*>(shared_buffer);

  const int thread = static_cast<int>(threadIdx.x);
  const bool is_consumer = thread < kThreads;
  const bool is_producer = !is_consumer;
  int segment = static_cast<int>(blockIdx.y);
  int n64_block = static_cast<int>(blockIdx.x);
  int split = static_cast<int>(blockIdx.z);
  int n64_block_global = n64_block;
  int total_n_tiles = launch_size_n / kP32TileColumns;
  int size_n = launch_size_n;
  int split_count = launch_split_count;
  int bank_alt_id = launch_bank_alt_id;
  int64_t output_offset = 0;
  int64_t partial_output_offset = 0;
  if constexpr (Grouped) {
    if constexpr (OrderedSplit) {
      const int work_item = static_cast<int>(blockIdx.x);
      segment = 0;
#pragma unroll
      for (int candidate = 1; candidate < kMaxGroupedP32Segments; ++candidate) {
        if (candidate < grouped_params.segment_count &&
            work_item >= grouped_params.work_item_start[candidate]) {
          segment = candidate;
        }
      }
      const int segment_work_item =
          work_item - grouped_params.work_item_start[segment];
      const int segment_n64_blocks =
          grouped_params.n_tiles[segment] / kP32N16TilesPerBlock;
      split = segment_work_item / segment_n64_blocks;
      n64_block = segment_work_item - split * segment_n64_blocks;
    }
    if (segment >= grouped_params.segment_count) {
      return;
    }
    const int segment_n_tiles = grouped_params.n_tiles[segment];
    split_count = grouped_params.split_count[segment];
    if (n64_block * kP32N16TilesPerBlock >= segment_n_tiles ||
        static_cast<int>(blockIdx.z) >= split_count) {
      return;
    }
    n64_block_global =
        grouped_params.n_tile_start[segment] / kP32N16TilesPerBlock + n64_block;
    size_n = segment_n_tiles * kP32TileColumns;
    total_n_tiles = launch_size_n / kP32TileColumns;
    bank_alt_id = grouped_params.bank_alt_id[segment];
    output_offset = grouped_params.output_offset[segment];
    partial_output_offset = grouped_params.partial_output_offset[segment];
  }
  const int k_tiles = size_k / kP32TileRows;
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
      cute::make_shape(cute::Int<kWordsPerP32Tile>{}, total_n_tiles, k_tiles));
  auto tiled_trellis = cute::local_tile(
      p32_full_trellis,
      cute::make_shape(
          cute::Int<kWordsPerP32Tile>{},
          cute::Int<kP32N16TilesPerBlock>{},
          cute::Int<kP32K16TilesPerStage>{}),
      cute::make_coord(cute::_0{}, n64_block_global, cute::_));
  auto [tma_global_trellis, tma_shared_trellis] = cute::tma_partition(
      trellis_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 3>(s_trellis),
      cute::group_modes<0, 3>(tiled_trellis));

  auto full_bank_ids =
      bank_tma.get_tma_tensor(cute::make_shape(total_n_tiles, k_tiles));
  auto tiled_bank_ids = cute::local_tile(
      full_bank_ids,
      cute::make_shape(cute::_16{}, cute::_16{}),
      cute::make_coord(n64_block_global >> 2, cute::_));
  auto [tma_global_bank_ids, tma_shared_bank_ids] = cute::tma_partition(
      bank_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_bank_ids),
      cute::group_modes<0, 2>(tiled_bank_ids));

  for (int index = thread; index < 256; index += kTmaThreads) {
    shared.levels.begin()[index] = levels[index];
    shared.levels_high.begin()[index] = levels[index ^ (index >> 7)];
  }

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
  const int bank_n16_offset =
      (n64_block_global & 3) * kP32N16TilesPerBlock;
  const auto decode_plan = qvq_p32_window_lane_plan<TransitionBits>(lane);
  const uint32_t shared_levels_base = __cvta_generic_to_shared(shared.levels.begin());
  const uint32_t shared_levels_high_base = __cvta_generic_to_shared(shared.levels_high.begin());
  WgmmaTmaPipelineState read_state;
  WgmmaTmaPipelineState release_state;

  for (int stage_offset = 0; stage_offset < stage_count; ++stage_offset) {
    auto wait_token = pipeline.consumer_try_wait(read_state);
    pipeline.consumer_wait(read_state, wait_token);
    const int read_stage = read_state.index();

#pragma unroll
    for (int k_block = 0; k_block < kP32K16TilesPerStage; ++k_block) {
      auto& fragment_a = (k_block & 1) == 0 ? fragment_a0 : fragment_a1;
      const uint32_t bank_id = s_bank_ids(bank_n16_offset + warp, k_block, read_stage);
      if (k_block >= 2) {
        cute::warpgroup_wait<1>();
      }
      const auto trellis_layout = TrellisSmemLayout{};
      const uint32_t* window_words = shared.trellis.begin() +
          trellis_layout(0, warp, k_block, read_stage);
      qvq_p32_window_decode_fragment<TransitionBits, true>(
          fragment_a,
          window_words,
          decode_plan,
          static_cast<uint8_t>(bank_id),
          shared.levels.begin(),
          shared_levels_base,
          shared_levels_high_base,
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
    const int64_t local_output_index =
        static_cast<int64_t>(output_row) * size_n +
        n64_block * kOutputColumns + p32_column;
    const int64_t output_index = output_offset + local_output_index;
    if constexpr (OrderedSplit) {
      partial_output[
          partial_output_offset +
          static_cast<int64_t>(split) * kRows * size_n +
          local_output_index] =
          accumulator(index);
    } else if (split_count > 1) {
      atomicAdd(partial_output + output_index, accumulator(index));
    } else {
      partial_output[output_index] = accumulator(index);
    }
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

template <int SplitCount>
__global__ void qvq_wgmma_reduce_split_fixed_kernel(
    const float* __restrict__ partial_output,
    float* __restrict__ output,
    int output_values) {
  const int vector_index = static_cast<int>(blockIdx.x) *
          static_cast<int>(blockDim.x) +
      static_cast<int>(threadIdx.x);
  const int vector_count = output_values / 4;
  if (vector_index >= vector_count) {
    return;
  }
  float value0 = 0.0f;
  float value1 = 0.0f;
  float value2 = 0.0f;
  float value3 = 0.0f;
#pragma unroll
  for (int split = 0; split < SplitCount; ++split) {
    const auto value = reinterpret_cast<const float4*>(partial_output)[
        static_cast<int64_t>(split) * vector_count + vector_index];
    value0 += value.x;
    value1 += value.y;
    value2 += value.z;
    value3 += value.w;
  }
  reinterpret_cast<float4*>(output)[vector_index] =
      make_float4(value0, value1, value2, value3);
}

void qvq_wgmma_launch_ordered_split_reduction(
    const at::Tensor& partial_output,
    int64_t partial_output_offset,
    at::Tensor& output,
    int64_t output_offset,
    int output_values,
    int split_count,
    cudaStream_t stream) {
  constexpr int kReductionThreads = 256;
  const int vector_count = output_values / 4;
  const int reduction_blocks =
      (vector_count + kReductionThreads - 1) / kReductionThreads;
#define QVQ_LAUNCH_FIXED_REDUCER(SPLIT_COUNT)                                      \
  qvq_wgmma_reduce_split_fixed_kernel<SPLIT_COUNT>                                \
      <<<reduction_blocks, kReductionThreads, 0, stream>>>(                        \
          partial_output.data_ptr<float>() + partial_output_offset,                \
          output.data_ptr<float>() + output_offset,                               \
          output_values)
  switch (split_count) {
    case 1:
      QVQ_LAUNCH_FIXED_REDUCER(1);
      break;
    case 2:
      QVQ_LAUNCH_FIXED_REDUCER(2);
      break;
    case 4:
      QVQ_LAUNCH_FIXED_REDUCER(4);
      break;
    case 8:
      QVQ_LAUNCH_FIXED_REDUCER(8);
      break;
    case 16:
      QVQ_LAUNCH_FIXED_REDUCER(16);
      break;
    case 32:
      QVQ_LAUNCH_FIXED_REDUCER(32);
      break;
    default: {
      const int scalar_reduction_blocks =
          (output_values + kReductionThreads - 1) / kReductionThreads;
      qvq_wgmma_reduce_split_kernel
          <<<scalar_reduction_blocks, kReductionThreads, 0, stream>>>(
              partial_output.data_ptr<float>() + partial_output_offset,
              output.data_ptr<float>() + output_offset,
              output_values,
              split_count);
      break;
    }
  }
#undef QVQ_LAUNCH_FIXED_REDUCER
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

template <int TransitionBits, bool OrderedSplit = false>
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
      : OrderedSplit
          ? at::empty(
                {split_count, kRows, size_n},
                input.options().dtype(at::kFloat))
          : at::zeros({kRows, size_n}, input.options().dtype(at::kFloat));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const dim3 grid(static_cast<unsigned>(size_n / kOutputColumns), 1, static_cast<unsigned>(split_count));
  HopperGroupedP32LaunchParams grouped_params{};
  qvq_p32_window_wgmma_m16_tma_kernel<TransitionBits, false, OrderedSplit>
      <<<grid, kTmaThreads, 0, stream>>>(
      input_tma,
      trellis_tma,
      bank_tma,
      reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
      partial_output.data_ptr<float>(),
      grouped_params,
      size_k,
      size_n,
      static_cast<int>(split_count),
      static_cast<int>(bank_alt_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if constexpr (OrderedSplit) {
    if (split_count > 1) {
      qvq_wgmma_launch_ordered_split_reduction(
          partial_output,
          0,
          output,
          0,
          static_cast<int>(output.numel()),
          static_cast<int>(split_count),
          stream);
    }
  } else if (split_count > 1) {
    output = partial_output;
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

at::Tensor qvq_p32_window_wgmma_m16_tma_ordered_split(
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
      return qvq_p32_window_wgmma_m16_tma_impl<4, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 5:
      return qvq_p32_window_wgmma_m16_tma_impl<5, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 6:
      return qvq_p32_window_wgmma_m16_tma_impl<6, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 7:
      return qvq_p32_window_wgmma_m16_tma_impl<7, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    default:
      TORCH_CHECK(
          false,
          "ordered-split QVQ P32 TMA WGMMA transition bits must be in [4, 7]");
  }
}

template <int TransitionBits, bool OrderedSplit = false>
at::Tensor qvq_p32_window_wgmma_m16_tma_grouped_impl(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts) {
  TORCH_CHECK(input.is_cuda(), "grouped QVQ P32 TMA WGMMA input must be CUDA");
  c10::cuda::CUDAGuard device_guard(input.device());
  TORCH_CHECK(
      trellis.device() == input.device() && levels.device() == input.device() &&
          bank_ids.device() == input.device(),
      "grouped QVQ P32 TMA WGMMA tensors must share one CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf && levels.scalar_type() == at::kHalf,
      "grouped QVQ P32 TMA WGMMA requires FP16 input and levels");
  TORCH_CHECK(
      trellis.scalar_type() == at::kInt,
      "grouped QVQ P32 TMA WGMMA trellis must be int32");
  TORCH_CHECK(
      bank_ids.scalar_type() == at::kByte,
      "grouped QVQ P32 TMA WGMMA bank ids must be uint8");
  TORCH_CHECK(
      input.is_contiguous() && trellis.is_contiguous() && levels.is_contiguous() &&
          bank_ids.is_contiguous(),
      "grouped QVQ P32 TMA WGMMA tensors must be contiguous");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) == kRows,
      "grouped QVQ P32 TMA WGMMA requires M=16");
  TORCH_CHECK(
      input.size(1) > 0 && input.size(1) % kKPerStage == 0,
      "grouped QVQ P32 TMA WGMMA K must be a positive multiple of 256");
  const int64_t segment_count = static_cast<int64_t>(out_features.size());
  TORCH_CHECK(
      segment_count >= 1 && segment_count <= kMaxGroupedP32Segments,
      "grouped QVQ P32 TMA WGMMA requires one to three segments");
  TORCH_CHECK(
      static_cast<int64_t>(bank_alt_ids.size()) == segment_count &&
          static_cast<int64_t>(split_counts.size()) == segment_count,
      "grouped QVQ P32 TMA WGMMA metadata lengths must match");

  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(
      properties.major == 9 && properties.minor == 0,
      "grouped QVQ P32 TMA WGMMA requires an SM90 H100/H200 device");

  const int size_k = static_cast<int>(input.size(1));
  const int k_tiles = size_k / kP32TileRows;
  int64_t total_n = 0;
  int max_n64_blocks = 0;
  int max_split_count = 1;
  int64_t total_work_items = 0;
  int64_t total_partial_values = 0;
  HopperGroupedP32LaunchParams grouped_params{};
  grouped_params.segment_count = static_cast<int>(segment_count);
  for (int segment = 0; segment < segment_count; ++segment) {
    const int64_t width = out_features[segment];
    const int64_t alt_id = bank_alt_ids[segment];
    const int64_t split_count = split_counts[segment];
    TORCH_CHECK(
        width > 0 && width % 256 == 0,
        "grouped QVQ P32 TMA WGMMA widths must be positive multiples of 256");
    TORCH_CHECK(
        alt_id >= 0 && alt_id <= 3,
        "grouped QVQ P32 TMA WGMMA alternative bank IDs must be in [0, 3]");
    TORCH_CHECK(
        split_count >= 1 && split_count <= 64,
        "grouped QVQ P32 TMA WGMMA split counts must be in [1, 64]");
    TORCH_CHECK(
        k_tiles % split_count == 0 &&
            (k_tiles / split_count) % kP32K16TilesPerStage == 0,
        "grouped QVQ P32 TMA WGMMA split partitions must contain a multiple "
        "of sixteen K16 tiles");
    grouped_params.n_tile_start[segment] =
        static_cast<int>(total_n / kP32TileColumns);
    grouped_params.n_tiles[segment] =
        static_cast<int>(width / kP32TileColumns);
    grouped_params.bank_alt_id[segment] = static_cast<int>(alt_id);
    grouped_params.split_count[segment] = static_cast<int>(split_count);
    grouped_params.work_item_start[segment] =
        static_cast<int>(total_work_items);
    grouped_params.output_offset[segment] = total_n * kRows;
    grouped_params.partial_output_offset[segment] = total_partial_values;
    total_work_items += (width / kOutputColumns) * split_count;
    total_partial_values += split_count * kRows * width;
    total_n += width;
    max_n64_blocks = std::max(
        max_n64_blocks, static_cast<int>(width / kOutputColumns));
    max_split_count =
        std::max(max_split_count, static_cast<int>(split_count));
  }
  TORCH_CHECK(
      total_n <= std::numeric_limits<int>::max(),
      "grouped QVQ P32 TMA WGMMA total N exceeds int32 range");
  TORCH_CHECK(
      total_work_items <= std::numeric_limits<int>::max(),
      "grouped QVQ P32 TMA WGMMA work count exceeds int32 range");
  const int total_n_tiles = static_cast<int>(total_n / kP32TileColumns);
  const int64_t expected_tiles = static_cast<int64_t>(k_tiles) * total_n_tiles;
  constexpr int kWordsPerP32Tile = 4 * TransitionBits;
  using TrellisSmemLayout = P32TrellisTmaSmemLayoutFor<TransitionBits>;
  TORCH_CHECK(
      trellis.numel() == expected_tiles * kWordsPerP32Tile,
      "grouped QVQ P32 TMA WGMMA trellis size mismatch");
  TORCH_CHECK(
      bank_ids.numel() == expected_tiles,
      "grouped QVQ P32 TMA WGMMA bank-id size mismatch");
  TORCH_CHECK(
      levels.numel() == 256,
      "grouped QVQ P32 TMA WGMMA requires 256 PGC16 levels");

  const auto* input_ptr =
      reinterpret_cast<const Element*>(input.data_ptr<at::Half>());
  const auto* trellis_ptr =
      reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>());
  auto input_tensor = cute::make_tensor(
      input_ptr,
      cute::make_shape(kRows, size_k),
      cute::make_stride(static_cast<int64_t>(size_k), cute::_1{}));
  auto trellis_tensor = cute::make_tensor(
      trellis_ptr,
      cute::make_shape(kWordsPerP32Tile, total_n_tiles, k_tiles),
      cute::make_stride(
          cute::_1{},
          cute::Int<kWordsPerP32Tile>{},
          static_cast<int64_t>(total_n_tiles) * kWordsPerP32Tile));
  auto bank_tensor = cute::make_tensor(
      bank_ids.data_ptr<uint8_t>(),
      cute::make_shape(total_n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, static_cast<int64_t>(total_n_tiles)));
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

  auto output = OrderedSplit || max_split_count == 1
      ? at::empty({kRows * total_n}, input.options().dtype(at::kFloat))
      : at::zeros({kRows * total_n}, input.options().dtype(at::kFloat));
  auto partial_output = OrderedSplit
      ? at::empty({total_partial_values}, input.options().dtype(at::kFloat))
      : output;
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(input.get_device());
  const dim3 grid = OrderedSplit
      ? dim3(static_cast<unsigned>(total_work_items), 1, 1)
      : dim3(
            static_cast<unsigned>(max_n64_blocks),
            static_cast<unsigned>(segment_count),
            static_cast<unsigned>(max_split_count));
  qvq_p32_window_wgmma_m16_tma_kernel<TransitionBits, true, OrderedSplit>
      <<<grid, kTmaThreads, 0, stream>>>(
          input_tma,
          trellis_tma,
          bank_tma,
          reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
          partial_output.data_ptr<float>(),
          grouped_params,
          size_k,
          static_cast<int>(total_n),
          1,
          0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if constexpr (OrderedSplit) {
    for (int segment = 0; segment < segment_count; ++segment) {
      const int output_values =
          kRows * static_cast<int>(out_features[segment]);
      qvq_wgmma_launch_ordered_split_reduction(
          partial_output,
          grouped_params.partial_output_offset[segment],
          output,
          grouped_params.output_offset[segment],
          output_values,
          grouped_params.split_count[segment],
          stream);
    }
  }
  return output;
}

at::Tensor qvq_p32_window_wgmma_m16_tma_grouped(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts) {
  switch (transition_bits) {
    case 4:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    case 5:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<5>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    case 6:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<6>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    case 7:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<7>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    default:
      TORCH_CHECK(
          false,
          "grouped QVQ P32 TMA WGMMA transition bits must be in [4, 7]");
  }
}

at::Tensor qvq_p32_window_wgmma_m16_tma_grouped_ordered_split(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts) {
  switch (transition_bits) {
    case 4:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<4, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    case 5:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<5, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    case 6:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<6, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    case 7:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<7, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    default:
      TORCH_CHECK(
          false,
          "ordered grouped QVQ P32 TMA WGMMA transition bits must be in [4, 7]");
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq_wgmma, m) {
  m.def("p32_window_w3_m16(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("p32_window_w3_m16_tma(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("p32_window_m16_tma(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("p32_window_m16_tma_ordered_split(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("p32_window_m16_tma_grouped(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts) -> Tensor");
  m.def("p32_window_m16_tma_grouped_ordered_split(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq_wgmma, CUDA, m) {
  m.impl("p32_window_w3_m16", qvq_p32_window_wgmma_w3_m16);
  m.impl("p32_window_w3_m16_tma", qvq_p32_window_wgmma_w3_m16_tma);
  m.impl("p32_window_m16_tma", qvq_p32_window_wgmma_m16_tma);
  m.impl("p32_window_m16_tma_ordered_split", qvq_p32_window_wgmma_m16_tma_ordered_split);
  m.impl("p32_window_m16_tma_grouped", qvq_p32_window_wgmma_m16_tma_grouped);
  m.impl("p32_window_m16_tma_grouped_ordered_split", qvq_p32_window_wgmma_m16_tma_grouped_ordered_split);
}
