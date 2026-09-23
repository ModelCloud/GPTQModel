// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#ifndef QVQ_WGMMA_DEVICE_ONLY
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>
#else
#include <cutlass/numeric_types.h>
namespace c10 {
using Float8_e4m3fn = cutlass::float_e4m3_t;
using Float8_e5m2 = cutlass::float_e5m2_t;
}  // namespace c10
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#ifndef QVQ_WGMMA_DEVICE_ONLY
#include <c10/util/Float8_e5m2.h>
#endif

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

using Element = cutlass::half_t;
using Fp8Element = cutlass::float_e4m3_t;
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

// True A8 P32 execution combines two adjacent canonical K16 tiles into the
// K32 operand required by Hopper's E4M3 x E4M3 RS-WGMMA.  P32 storage remains
// unchanged: only decoded register values and the transformed activation tile
// are narrowed to E4M3 for the tensor-core operation.
using Fp8WgmmaTileShape = cute::Shape<cute::_64, cute::_16, cute::_32>;
using Fp8WgmmaTiledMma = decltype(cute::make_tiled_mma(
    cute::GMMA::rs_op_selector<
        Fp8Element,
        Fp8Element,
        float,
        Fp8WgmmaTileShape,
        cute::GMMA::Major::K,
        cute::GMMA::Major::K>()));
using Fp8WgmmaSmemLayoutAtomB = decltype(
    cutlass::gemm::collective::detail::rs_smem_selector<
        cute::GMMA::Major::K,
        Fp8Element,
        cute::_32,
        cute::_32,
        false>());
using Fp8WgmmaSmemLayoutB = decltype(cute::tile_to_shape(
    Fp8WgmmaSmemLayoutAtomB{}, cute::make_shape(cute::_16{}, cute::_32{})));

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
constexpr int kFixedGateUpK = 2048;
constexpr int kFixedGateUpN = 8192;
constexpr int kFixedQwenGateUpK = 5120;
constexpr int kFixedQwenGateUpN = 17408;
constexpr int kFixedQwenGateUpSplit = 5;
constexpr int kFixedQwenLinearQkvN = 10240;
constexpr int kFixedQwenLinearZN = 6144;
constexpr int kFixedFlashNextFullQn = 12288;
constexpr int kFixedFlashNextFullKvN = 512;
// The W2.5 N128 block has two independent consumer warpgroups and needs only
// three decoded A fragments in flight.  A matched H100 depth-2/3/4 sweep
// selected depth three at every Llama 3.2 1B decode M.
constexpr int kW25N128DecodeDepth = 3;

struct HopperGroupedP32LaunchParams {
  int segment_count;
  int n_tile_start[kMaxGroupedP32Segments];
  int n_tiles[kMaxGroupedP32Segments];
  int bank_alt_id[kMaxGroupedP32Segments];
  int split_count[kMaxGroupedP32Segments];
  int work_item_start[kMaxGroupedP32Segments];
  int64_t output_offset[kMaxGroupedP32Segments];
  int64_t partial_output_offset[kMaxGroupedP32Segments];
  // Optional device scalar used by framework-neutral callers. Torch callers
  // leave this null and retain the launch-time scalar ABI.
  const uint8_t* launch_bank_alt_ids;
};

struct HopperGroupedP32DecodeParams {
  int segment_count;
  int n64_end[kMaxGroupedP32Segments];
  int bank_alt_id[kMaxGroupedP32Segments];
};

struct HopperGroupedP32FoldParams {
  int segment_count;
  int n_start[kMaxGroupedP32Segments];
  int width[kMaxGroupedP32Segments];
  int output_hadamard[kMaxGroupedP32Segments];
  const float* output_scale[kMaxGroupedP32Segments];
};

struct HopperFixedGateUpLaunchParams {
  int bank_alt_id[2];
};

using WgmmaTmaSmemLayoutB = decltype(cute::tile_to_shape(
    WgmmaSmemLayoutAtomB{},
    cute::make_shape(cute::_16{}, cute::_256{}, cute::Int<kTmaStages>{})));
template <int TransitionBits, int N64BlocksPerCta = 1>
using P32TrellisTmaSmemLayoutFor = decltype(cute::make_layout(
    cute::make_shape(
        cute::Int<4 * TransitionBits>{},
        cute::Int<kP32N16TilesPerBlock * N64BlocksPerCta>{},
        cute::Int<kP32K16TilesPerStage>{},
        cute::Int<kTmaStages>{}),
    cute::make_stride(
        cute::_1{},
        cute::Int<4 * TransitionBits>{},
        cute::Int<4 * TransitionBits * kP32N16TilesPerBlock * N64BlocksPerCta>{},
        cute::Int<4 * TransitionBits * kP32N16TilesPerBlock *
                  N64BlocksPerCta * kP32K16TilesPerStage>{})));
using P32BankTmaSmemLayout = decltype(cute::make_layout(
    cute::make_shape(cute::_16{}, cute::_16{}, cute::Int<kTmaStages>{}),
    cute::make_stride(cute::_1{}, cute::_16{}, cute::_256{})));
using WgmmaTmaPipeline = cutlass::PipelineTmaAsync<kTmaStages>;
using WgmmaTmaPipelineState = cutlass::PipelineState<kTmaStages>;

template <
    int TransitionBits,
    int N64BlocksPerCta = 1,
    int RowTilesPerCta = 1>
struct alignas(128) P32WgmmaTmaSharedStorageFor {
  typename WgmmaTmaPipeline::SharedStorage pipeline;
  alignas(128) cute::ArrayEngine<
      Element,
      RowTilesPerCta * cute::cosize_v<WgmmaTmaSmemLayoutB>> input;
  alignas(128) cute::ArrayEngine<
      uint32_t,
      cute::cosize_v<P32TrellisTmaSmemLayoutFor<
          TransitionBits, N64BlocksPerCta>>> trellis;
  alignas(128) cute::ArrayEngine<uint8_t, cute::cosize_v<P32BankTmaSmemLayout>> bank_ids;
  // Reused by every decode lane and K16 tile; avoid dependent L1/global
  // lookups for the small, read-only PGC level table. W2-W3.5 store
  // levels[index][lane], assigning each lane pair its own two alternating
  // shared banks. Even lane slots hold the canonical view and odd slots hold
  // the fixed high-byte permutation. The extra shared footprint removes
  // cross-pair conflicts without retaining a separate high table.
  static constexpr int kLevelEntries =
      TransitionBits >= 4 ? 256 * 32 : 256;
  alignas(128) cute::ArrayEngine<Element, kLevelEntries> levels;
  // The PGC high byte is b ^ (b >> 7).  Store that fixed permutation once so
  // the hot loop can index it directly with affine-product byte 1.
  static constexpr int kHighLevelEntries =
      TransitionBits >= 4 ? 1 : 256;
  alignas(128) cute::ArrayEngine<Element, kHighLevelEntries> levels_high;
};

static_assert(
    sizeof(P32WgmmaTmaSharedStorageFor<kW3TransitionBits>) <= 48 * 1024,
    "W3 lane-interleaved levels must fit the default Hopper shared-memory limit");
static_assert(
    sizeof(P32WgmmaTmaSharedStorageFor<4>) <= 48 * 1024,
    "W2 lane-interleaved levels must fit the default Hopper shared-memory limit");
static_assert(
    sizeof(P32WgmmaTmaSharedStorageFor<5>) <= 48 * 1024,
    "W2.5 lane-interleaved levels must fit the default Hopper shared-memory limit");
static_assert(
    sizeof(P32WgmmaTmaSharedStorageFor<7>) <= 48 * 1024,
    "W3.5 lane-interleaved levels must fit the default Hopper shared-memory limit");

static_assert(cute::size(WgmmaTiledMma{}) == kThreads);
static_assert(cute::size(Fp8WgmmaTiledMma{}) == kThreads);

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

__device__ __forceinline__ Element qvq_wgmma_load_level_shared_lane(
    uint32_t levels_base,
    uint32_t index,
    uint32_t lane) {
  const uint32_t address = levels_base + (index << 6) + (lane << 1);
  uint16_t bits;
  asm("ld.shared.u16 %0, [%1];" : "=h"(bits) : "r"(address));
  return Element::bitcast(bits);
}

__device__ __forceinline__ Element qvq_wgmma_load_level_shared_lane_byte_offset(
    uint32_t levels_base,
    uint32_t byte_offset,
    uint32_t lane) {
  const uint32_t address = levels_base + (byte_offset << 5) + (lane << 1);
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
  if constexpr (TransitionBits == 7) {
    // Shift-7 only needs the low 16 bits of each circular window. Express the
    // existing low-word result directly as a 32-bit funnel shift so ptxas
    // does not retain the unused high half of four 64-bit shifts in the hot
    // decode loop.
    first = __funnelshift_r(
        window_words[first_word], window_words[first_next_word], shift);
    second = __funnelshift_r(
        window_words[second_word], window_words[second_word + 1], shift);
  } else {
    const uint64_t first_window = static_cast<uint64_t>(window_words[first_word]) |
        (static_cast<uint64_t>(window_words[first_next_word]) << 32);
    const uint64_t second_window = static_cast<uint64_t>(window_words[second_word]) |
        (static_cast<uint64_t>(window_words[second_word + 1]) << 32);
    first = static_cast<uint32_t>(first_window >> shift);
    second = static_cast<uint32_t>(second_window >> shift);
  }
}

// W2.5, W3, and W3.5 cross enough word boundaries for the lane geometry
// arithmetic to remain visible in the hot loop. Hoist the two word pairs and
// shifts once per lane; W2 retains the compact generic path that ptxas
// optimizes.
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
__device__ __forceinline__ void qvq_p32_window_load_fragment_levels(
    FragmentA& fragment,
    const Element* __restrict__ levels,
    uint32_t levels_shared_base,
    uint32_t levels_high_shared_base,
    uint32_t product00,
    uint32_t product01,
    uint32_t product10,
    uint32_t product11) {
  // CuTe maps each lane to two A rows and two K pairs. Map those two rows to
  // adjacent P32 N values so each decoded state feeds both output columns.
  if constexpr (LevelsInShared) {
    if constexpr (TransitionBits >= 4) {
      const uint32_t lane = static_cast<uint32_t>(threadIdx.x) & 31u;
      const uint32_t low_lane = lane & ~1u;
      const uint32_t high_lane = lane | 1u;
      fragment(0) = qvq_wgmma_load_level_shared_lane(
          levels_shared_base, qvq_wgmma_high_byte(product00), high_lane);
      fragment(1) = qvq_wgmma_load_level_shared_lane(
          levels_shared_base, qvq_wgmma_high_byte(product01), high_lane);
      fragment(2) = qvq_wgmma_load_level_shared_lane_byte_offset(
          levels_shared_base, qvq_wgmma_pgc16_low_byte_offset(product00), low_lane);
      fragment(3) = qvq_wgmma_load_level_shared_lane_byte_offset(
          levels_shared_base, qvq_wgmma_pgc16_low_byte_offset(product01), low_lane);
      fragment(4) = qvq_wgmma_load_level_shared_lane(
          levels_shared_base, qvq_wgmma_high_byte(product10), high_lane);
      fragment(5) = qvq_wgmma_load_level_shared_lane(
          levels_shared_base, qvq_wgmma_high_byte(product11), high_lane);
      fragment(6) = qvq_wgmma_load_level_shared_lane_byte_offset(
          levels_shared_base, qvq_wgmma_pgc16_low_byte_offset(product10), low_lane);
      fragment(7) = qvq_wgmma_load_level_shared_lane_byte_offset(
          levels_shared_base, qvq_wgmma_pgc16_low_byte_offset(product11), low_lane);
    } else {
      fragment(0) = qvq_wgmma_load_level_shared(
          levels_high_shared_base, qvq_wgmma_high_byte(product00));
      fragment(1) = qvq_wgmma_load_level_shared(
          levels_high_shared_base, qvq_wgmma_high_byte(product01));
      fragment(2) = qvq_wgmma_load_level_shared_byte_offset(
          levels_shared_base, qvq_wgmma_pgc16_low_byte_offset(product00));
      fragment(3) = qvq_wgmma_load_level_shared_byte_offset(
          levels_shared_base, qvq_wgmma_pgc16_low_byte_offset(product01));
      fragment(4) = qvq_wgmma_load_level_shared(
          levels_high_shared_base, qvq_wgmma_high_byte(product10));
      fragment(5) = qvq_wgmma_load_level_shared(
          levels_high_shared_base, qvq_wgmma_high_byte(product11));
      fragment(6) = qvq_wgmma_load_level_shared_byte_offset(
          levels_shared_base, qvq_wgmma_pgc16_low_byte_offset(product10));
      fragment(7) = qvq_wgmma_load_level_shared_byte_offset(
          levels_shared_base, qvq_wgmma_pgc16_low_byte_offset(product11));
    }
  } else {
    const uint32_t mixed00 = qvq_wgmma_pgc16_finish(product00);
    const uint32_t mixed01 = qvq_wgmma_pgc16_finish(product01);
    const uint32_t mixed10 = qvq_wgmma_pgc16_finish(product10);
    const uint32_t mixed11 = qvq_wgmma_pgc16_finish(product11);
    fragment(0) = qvq_wgmma_decode_level<TransitionBits, false>(
        levels, 0, qvq_wgmma_high_byte(mixed00));
    fragment(1) = qvq_wgmma_decode_level<TransitionBits, false>(
        levels, 0, qvq_wgmma_high_byte(mixed01));
    fragment(2) = qvq_wgmma_decode_level<TransitionBits, false>(levels, 0, mixed00 & 0xffu);
    fragment(3) = qvq_wgmma_decode_level<TransitionBits, false>(levels, 0, mixed01 & 0xffu);
    fragment(4) = qvq_wgmma_decode_level<TransitionBits, false>(
        levels, 0, qvq_wgmma_high_byte(mixed10));
    fragment(5) = qvq_wgmma_decode_level<TransitionBits, false>(
        levels, 0, qvq_wgmma_high_byte(mixed11));
    fragment(6) = qvq_wgmma_decode_level<TransitionBits, false>(levels, 0, mixed10 & 0xffu);
    fragment(7) = qvq_wgmma_decode_level<TransitionBits, false>(levels, 0, mixed11 & 0xffu);
  }
}

template <
    int TransitionBits,
    bool LevelsInShared,
    bool PrefetchDecodedLevels = false,
    int ReuseWaitGroups = 3,
    class FragmentA>
__device__ __forceinline__ void qvq_p32_window_decode_fragment(
    FragmentA& fragment,
    const uint32_t* __restrict__ window_words,
    const QvqP32WindowLanePlan& plan,
    uint8_t bank_id,
    const Element* __restrict__ levels,
    uint32_t levels_shared_base,
    uint32_t levels_high_shared_base,
    uint32_t alternate_bank_mask,
    bool wait_before_fragment_reuse = false) {
  uint32_t state00;
  uint32_t state01;
  uint32_t state10;
  uint32_t state11;
  uint32_t bank_pair_bits;
  if constexpr (TransitionBits >= 5) {
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

  if constexpr (LevelsInShared && PrefetchDecodedLevels) {
    // Depth-four W2.5/W3 can profitably prefetch all eight levels into an
    // independent register fragment while the old WGMMA source remains live.
    auto decoded_fragment = cute::make_tensor<Element>(fragment.shape());
    qvq_p32_window_load_fragment_levels<TransitionBits, LevelsInShared>(
        decoded_fragment,
        levels,
        levels_shared_base,
        levels_high_shared_base,
        product00,
        product01,
        product10,
        product11);
    // Express the decoded values as the four register pairs consumed by
    // WGMMA before waiting for the old source.  This lets ptxas retain the
    // halfword loads while removing the generic tensor-copy plumbing; the
    // final four PRMT writes may still follow DEPBAR because the destination
    // fragment registers remain live until that barrier.
    uint32_t decoded_packed[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      decoded_packed[pair] =
          static_cast<uint32_t>(decoded_fragment(2 * pair).storage) |
          (static_cast<uint32_t>(decoded_fragment(2 * pair + 1).storage) << 16);
    }
    if (wait_before_fragment_reuse) {
      cute::warpgroup_fence_operand(fragment);
      cute::warpgroup_wait<ReuseWaitGroups>();
    }
    auto packed_fragment = cute::recast<uint32_t>(fragment);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      packed_fragment(pair) = decoded_packed[pair];
    }
  } else {
    if (wait_before_fragment_reuse) {
      if constexpr (TransitionBits == 5 || TransitionBits == kW3TransitionBits) {
        cute::warpgroup_wait<3>();
      } else {
        cute::warpgroup_wait<1>();
      }
    }
    qvq_p32_window_load_fragment_levels<TransitionBits, LevelsInShared>(
        fragment,
        levels,
        levels_shared_base,
        levels_high_shared_base,
        product00,
        product01,
        product10,
        product11);
  }
}

__device__ __forceinline__ int qvq_p32_wgmma_logical_column(int wgmma_column) {
  const int tile_column = wgmma_column & 15;
  return (wgmma_column & ~15) + ((tile_column & 7) << 1) + (tile_column >> 3);
}

__device__ __forceinline__ void qvq_p32_copy_async_cg_16(
    void* destination,
    const void* source) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :
      : "r"(shared_address), "l"(source));
}

template <int TransitionBits, int RowTilesPerCta = 1>
__global__ __launch_bounds__(kThreads) void qvq_p32_window_wgmma_fp8_m16_kernel(
    const Fp8Element* __restrict__ input,
    const float* __restrict__ input_scale,
    const uint32_t* __restrict__ trellis,
    const uint8_t* __restrict__ bank_ids,
    const Fp8Element* __restrict__ levels,
    float level_scale,
    float* __restrict__ output,
    int size_k,
    int size_n,
    int bank_alt_id) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  static_assert(
      RowTilesPerCta == 1 || RowTilesPerCta == 2 || RowTilesPerCta == 4 ||
      RowTilesPerCta == 8);
  constexpr int kInputTileElements = cute::cosize_v<Fp8WgmmaSmemLayoutB>;
  // Keep eight addressable tiles for the explicitly named CUTE tensor views.
  // Inactive views are compile-time dead for the narrower variants.
  __shared__ __align__(128) Fp8Element shared_input[8 * kInputTileElements];
  __shared__ __align__(16) float shared_input_scale[8 * kRows];

  const int thread = static_cast<int>(threadIdx.x);
  const int n64_block = static_cast<int>(blockIdx.x);
  const int row_base =
      static_cast<int>(blockIdx.y) * RowTilesPerCta * kRows;
  const int n_base = n64_block * kOutputColumns;
  const int n_tiles = size_n / kP32TileColumns;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int n_pair = lane >> 2;
  const int k4 = (lane & 3) << 2;
  const int n16_tile = n64_block * kP32N16TilesPerBlock + warp;
  const uint32_t alternate_bank_mask =
      qvq_wgmma_v2_alternate_bank_mask<TransitionBits>(bank_alt_id);
  const uint32_t alternate_mix_mask = alternate_bank_mask & 0xff00u;

  if (thread < RowTilesPerCta * kRows) {
    shared_input_scale[thread] = input_scale[row_base + thread];
  }
  __syncthreads();

  auto sB = cute::make_tensor(cute::make_smem_ptr(shared_input), Fp8WgmmaSmemLayoutB{});
  auto sB1 = cute::make_tensor(
      cute::make_smem_ptr(shared_input + kInputTileElements),
      Fp8WgmmaSmemLayoutB{});
  auto sB2 = cute::make_tensor(
      cute::make_smem_ptr(shared_input + 2 * kInputTileElements),
      Fp8WgmmaSmemLayoutB{});
  auto sB3 = cute::make_tensor(
      cute::make_smem_ptr(shared_input + 3 * kInputTileElements),
      Fp8WgmmaSmemLayoutB{});
  auto sB4 = cute::make_tensor(
      cute::make_smem_ptr(shared_input + 4 * kInputTileElements),
      Fp8WgmmaSmemLayoutB{});
  auto sB5 = cute::make_tensor(
      cute::make_smem_ptr(shared_input + 5 * kInputTileElements),
      Fp8WgmmaSmemLayoutB{});
  auto sB6 = cute::make_tensor(
      cute::make_smem_ptr(shared_input + 6 * kInputTileElements),
      Fp8WgmmaSmemLayoutB{});
  auto sB7 = cute::make_tensor(
      cute::make_smem_ptr(shared_input + 7 * kInputTileElements),
      Fp8WgmmaSmemLayoutB{});
  Fp8WgmmaTiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(thread);
  auto thread_shared_b = thread_mma.partition_B(sB);
  auto fragment_b = thread_mma.make_fragment_B(thread_shared_b);
  auto fragment_b1 = thread_mma.make_fragment_B(thread_mma.partition_B(sB1));
  auto fragment_b2 = thread_mma.make_fragment_B(thread_mma.partition_B(sB2));
  auto fragment_b3 = thread_mma.make_fragment_B(thread_mma.partition_B(sB3));
  auto fragment_b4 = thread_mma.make_fragment_B(thread_mma.partition_B(sB4));
  auto fragment_b5 = thread_mma.make_fragment_B(thread_mma.partition_B(sB5));
  auto fragment_b6 = thread_mma.make_fragment_B(thread_mma.partition_B(sB6));
  auto fragment_b7 = thread_mma.make_fragment_B(thread_mma.partition_B(sB7));

  auto coordinate_a = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_32{}));
  auto thread_coordinate_a = thread_mma.partition_A(coordinate_a);
  auto fragment_a = cute::make_tensor<Fp8Element>(thread_coordinate_a.shape());
  static_assert(cute::size(decltype(thread_coordinate_a){}) == 16);

  auto coordinate_c = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_c = thread_mma.partition_C(coordinate_c);
  auto accumulator = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator1 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator2 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator3 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator4 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator5 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator6 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator7 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto scaled_accumulator = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto scaled_accumulator1 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto scaled_accumulator2 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto scaled_accumulator3 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto scaled_accumulator4 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto scaled_accumulator5 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto scaled_accumulator6 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto scaled_accumulator7 = cute::make_tensor<float>(thread_coordinate_c.shape());
  cute::clear(accumulator);
  cute::clear(accumulator1);
  cute::clear(accumulator2);
  cute::clear(accumulator3);
  cute::clear(accumulator4);
  cute::clear(accumulator5);
  cute::clear(accumulator6);
  cute::clear(accumulator7);
  cute::clear(scaled_accumulator);
  cute::clear(scaled_accumulator1);
  cute::clear(scaled_accumulator2);
  cute::clear(scaled_accumulator3);
  cute::clear(scaled_accumulator4);
  cute::clear(scaled_accumulator5);
  cute::clear(scaled_accumulator6);
  cute::clear(scaled_accumulator7);
  tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
  constexpr int kAccumulatorValuesPerThread = cute::size(decltype(thread_coordinate_c){});

  for (int k_base = 0; k_base < size_k; k_base += 32) {
    // Hopper's FP8 WGMMA layout keeps each aligned K16 half-row contiguous
    // even where its row-level swizzle exchanges the two halves. Copy those
    // 16-byte units directly instead of issuing one global/shared byte pair
    // for every activation element.
    constexpr int kVectorElements = sizeof(uint4) / sizeof(Fp8Element);
    constexpr int kVectorsPerRow = 32 / kVectorElements;
    for (int index = thread;
         index < RowTilesPerCta * kRows * kVectorsPerRow;
         index += kThreads) {
      const int row_tile = index / (kRows * kVectorsPerRow);
      const int tile_index = index - row_tile * kRows * kVectorsPerRow;
      const int row = tile_index / kVectorsPerRow;
      const int column = (tile_index - row * kVectorsPerRow) * kVectorElements;
      const int shared_offset = row_tile * kInputTileElements +
          Fp8WgmmaSmemLayoutB{}(row, column);
      qvq_p32_copy_async_cg_16(
          shared_input + shared_offset,
          input +
              static_cast<int64_t>(row_base + row_tile * kRows + row) * size_k +
              k_base + column);
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // Each lane owns four K values for one adjacent pair of P32 N columns in
    // each of the two canonical K16 tiles.  The even and odd columns select
    // the high and low bytes of the same PGC state, respectively.  Decode the
    // state once and feed both fragment positions instead of rediscovering
    // identical tile, bank, window, and affine state for all 16 elements.
#pragma unroll
    for (int k16_half = 0; k16_half < 2; ++k16_half) {
      constexpr int kWordsPerP32Tile = 4 * TransitionBits;
      const int k16_tile = (k_base >> 4) + k16_half;
      const int64_t tile = static_cast<int64_t>(k16_tile) * n_tiles + n16_tile;
      const uint32_t* window_words = trellis + tile * kWordsPerP32Tile;
      const uint32_t bank_id = static_cast<uint32_t>(bank_ids[tile]);
      const uint32_t bank_pair_bits = bank_id >> (k4 >> 1);
      const uint32_t bank_mask0 = (bank_pair_bits & 1u) * alternate_mix_mask;
      const uint32_t bank_mask1 = ((bank_pair_bits >> 1) & 1u) * alternate_mix_mask;
#pragma unroll
      for (int k_in_group = 0; k_in_group < 4; ++k_in_group) {
        const int pair = (k4 + k_in_group) * 8 + n_pair;
        const uint32_t state =
            qvq_p32_window_state<TransitionBits>(window_words, pair);
        const uint32_t bank_mask = k_in_group < 2 ? bank_mask0 : bank_mask1;
        const uint32_t product =
            qvq_wgmma_pgc16_product_masked(state, bank_mask);
        const uint32_t mixed = qvq_wgmma_pgc16_finish(product);
        const int fragment_base = k16_half * 8 + k_in_group;
        fragment_a(fragment_base) = levels[qvq_wgmma_high_byte(mixed)];
        fragment_a(fragment_base + 4) = levels[mixed & 0xffu];
      }
    }

    cute::warpgroup_fence_operand(fragment_a);
    cute::warpgroup_fence_operand(accumulator);
    if constexpr (RowTilesPerCta >= 2) {
      cute::warpgroup_fence_operand(accumulator1);
    }
    if constexpr (RowTilesPerCta == 4) {
      cute::warpgroup_fence_operand(accumulator2);
      cute::warpgroup_fence_operand(accumulator3);
    }
    if constexpr (RowTilesPerCta == 8) {
      cute::warpgroup_fence_operand(accumulator2);
      cute::warpgroup_fence_operand(accumulator3);
      cute::warpgroup_fence_operand(accumulator4);
      cute::warpgroup_fence_operand(accumulator5);
      cute::warpgroup_fence_operand(accumulator6);
      cute::warpgroup_fence_operand(accumulator7);
    }
    cute::warpgroup_arrive();
    cute::gemm(
        tiled_mma,
        fragment_a(cute::_, cute::_, cute::_0{}),
        fragment_b(cute::_, cute::_, cute::_0{}),
        accumulator);
    if constexpr (RowTilesPerCta >= 2) {
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b1(cute::_, cute::_, cute::_0{}),
          accumulator1);
    }
    if constexpr (RowTilesPerCta == 4) {
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b2(cute::_, cute::_, cute::_0{}),
          accumulator2);
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b3(cute::_, cute::_, cute::_0{}),
          accumulator3);
    }
    if constexpr (RowTilesPerCta == 8) {
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b2(cute::_, cute::_, cute::_0{}),
          accumulator2);
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b3(cute::_, cute::_, cute::_0{}),
          accumulator3);
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b4(cute::_, cute::_, cute::_0{}),
          accumulator4);
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b5(cute::_, cute::_, cute::_0{}),
          accumulator5);
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b6(cute::_, cute::_, cute::_0{}),
          accumulator6);
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b7(cute::_, cute::_, cute::_0{}),
          accumulator7);
    }
    tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(accumulator);
    if constexpr (RowTilesPerCta >= 2) {
      cute::warpgroup_fence_operand(accumulator1);
    }
    if constexpr (RowTilesPerCta == 4) {
      cute::warpgroup_fence_operand(accumulator2);
      cute::warpgroup_fence_operand(accumulator3);
    }
    if constexpr (RowTilesPerCta == 8) {
      cute::warpgroup_fence_operand(accumulator2);
      cute::warpgroup_fence_operand(accumulator3);
      cute::warpgroup_fence_operand(accumulator4);
      cute::warpgroup_fence_operand(accumulator5);
      cute::warpgroup_fence_operand(accumulator6);
      cute::warpgroup_fence_operand(accumulator7);
    }
#pragma unroll
    for (int index = 0; index < kAccumulatorValuesPerThread; ++index) {
      const auto coordinate = thread_coordinate_c(index);
      const int output_row = static_cast<int>(cute::get<1>(coordinate));
      scaled_accumulator(index) +=
          accumulator(index) * shared_input_scale[output_row] * level_scale;
      if constexpr (RowTilesPerCta >= 2) {
        scaled_accumulator1(index) += accumulator1(index) *
            shared_input_scale[kRows + output_row] * level_scale;
      }
      if constexpr (RowTilesPerCta == 4) {
        scaled_accumulator2(index) += accumulator2(index) *
            shared_input_scale[2 * kRows + output_row] * level_scale;
        scaled_accumulator3(index) += accumulator3(index) *
            shared_input_scale[3 * kRows + output_row] * level_scale;
      }
      if constexpr (RowTilesPerCta == 8) {
        scaled_accumulator2(index) += accumulator2(index) *
            shared_input_scale[2 * kRows + output_row] * level_scale;
        scaled_accumulator3(index) += accumulator3(index) *
            shared_input_scale[3 * kRows + output_row] * level_scale;
        scaled_accumulator4(index) += accumulator4(index) *
            shared_input_scale[4 * kRows + output_row] * level_scale;
        scaled_accumulator5(index) += accumulator5(index) *
            shared_input_scale[5 * kRows + output_row] * level_scale;
        scaled_accumulator6(index) += accumulator6(index) *
            shared_input_scale[6 * kRows + output_row] * level_scale;
        scaled_accumulator7(index) += accumulator7(index) *
            shared_input_scale[7 * kRows + output_row] * level_scale;
      }
    }
    cute::clear(accumulator);
    cute::clear(accumulator1);
    cute::clear(accumulator2);
    cute::clear(accumulator3);
    cute::clear(accumulator4);
    cute::clear(accumulator5);
    cute::clear(accumulator6);
    cute::clear(accumulator7);
    tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
    __syncthreads();
  }

#pragma unroll
  for (int index = 0; index < kAccumulatorValuesPerThread; ++index) {
    const auto coordinate = thread_coordinate_c(index);
    const int wgmma_column = static_cast<int>(cute::get<0>(coordinate));
    const int output_row = static_cast<int>(cute::get<1>(coordinate));
    const int logical_n = n_base + qvq_p32_wgmma_logical_column(wgmma_column);
    output[static_cast<int64_t>(row_base + output_row) * size_n + logical_n] =
        scaled_accumulator(index);
    if constexpr (RowTilesPerCta >= 2) {
      output[static_cast<int64_t>(row_base + kRows + output_row) * size_n +
             logical_n] = scaled_accumulator1(index);
    }
    if constexpr (RowTilesPerCta == 4) {
      output[static_cast<int64_t>(row_base + 2 * kRows + output_row) * size_n +
             logical_n] = scaled_accumulator2(index);
      output[static_cast<int64_t>(row_base + 3 * kRows + output_row) * size_n +
             logical_n] = scaled_accumulator3(index);
    }
    if constexpr (RowTilesPerCta == 8) {
      output[static_cast<int64_t>(row_base + 2 * kRows + output_row) * size_n +
             logical_n] = scaled_accumulator2(index);
      output[static_cast<int64_t>(row_base + 3 * kRows + output_row) * size_n +
             logical_n] = scaled_accumulator3(index);
      output[static_cast<int64_t>(row_base + 4 * kRows + output_row) * size_n +
             logical_n] = scaled_accumulator4(index);
      output[static_cast<int64_t>(row_base + 5 * kRows + output_row) * size_n +
             logical_n] = scaled_accumulator5(index);
      output[static_cast<int64_t>(row_base + 6 * kRows + output_row) * size_n +
             logical_n] = scaled_accumulator6(index);
      output[static_cast<int64_t>(row_base + 7 * kRows + output_row) * size_n +
             logical_n] = scaled_accumulator7(index);
    }
  }
#endif
}

template <int TransitionBits>
__global__ __launch_bounds__(kThreads) void qvq_p32_window_decode_fp16_kernel(
    const uint32_t* __restrict__ trellis,
    const uint8_t* __restrict__ bank_ids,
    const Element* __restrict__ levels,
    Element* __restrict__ output,
    HopperGroupedP32DecodeParams grouped_params,
    int size_k,
    int size_n,
    bool transpose_output) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  constexpr int kWordsPerP32Tile = 4 * TransitionBits;
  constexpr int kVectorsPerP32Tile = kWordsPerP32Tile / 4;
  __shared__ __align__(16) uint32_t
      packed_words[kP32K16TilesPerStage][kP32N16TilesPerBlock]
                  [kWordsPerP32Tile];
  __shared__ uint8_t
      packed_bank_ids[kP32K16TilesPerStage][kP32N16TilesPerBlock];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int n64_block = static_cast<int>(blockIdx.x);
  const int k256_stage = static_cast<int>(blockIdx.y);
  const int n_tiles = size_n / kP32TileColumns;
  const int n16_tile_base = n64_block * kP32N16TilesPerBlock;

  auto* shared_vectors = reinterpret_cast<uint4*>(&packed_words[0][0][0]);
  const auto* global_vectors = reinterpret_cast<const uint4*>(trellis);
  constexpr int kVectorsPerBlock = kP32K16TilesPerStage *
      kP32N16TilesPerBlock * kVectorsPerP32Tile;
  for (int vector = thread; vector < kVectorsPerBlock; vector += kThreads) {
    const int tile_local = vector / kVectorsPerP32Tile;
    const int vector_in_tile = vector - tile_local * kVectorsPerP32Tile;
    const int k16_local = tile_local / kP32N16TilesPerBlock;
    const int n16_local =
        tile_local - k16_local * kP32N16TilesPerBlock;
    const int64_t global_tile =
        static_cast<int64_t>(k256_stage * kP32K16TilesPerStage + k16_local) *
            n_tiles +
        n16_tile_base + n16_local;
    shared_vectors[vector] =
        global_vectors[global_tile * kVectorsPerP32Tile + vector_in_tile];
  }
  constexpr int kBankIdsPerBlock =
      kP32K16TilesPerStage * kP32N16TilesPerBlock;
  for (int index = thread; index < kBankIdsPerBlock; index += kThreads) {
    const int k16_local = index / kP32N16TilesPerBlock;
    const int n16_local = index - k16_local * kP32N16TilesPerBlock;
    const int64_t global_tile =
        static_cast<int64_t>(k256_stage * kP32K16TilesPerStage + k16_local) *
            n_tiles +
        n16_tile_base + n16_local;
    packed_bank_ids[k16_local][n16_local] = bank_ids[global_tile];
  }
  __syncthreads();

  int segment = 0;
#pragma unroll
  for (int candidate = 1; candidate < kMaxGroupedP32Segments; ++candidate) {
    if (candidate < grouped_params.segment_count &&
        n64_block >= grouped_params.n64_end[candidate - 1]) {
      segment = candidate;
    }
  }
  const uint32_t alternate_bank_mask =
      qvq_wgmma_v2_alternate_bank_mask<TransitionBits>(
          grouped_params.bank_alt_id[segment]);

  WgmmaTiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(thread);
  auto coordinate_a =
      cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_a = thread_mma.partition_A(coordinate_a);
  auto fragment_a = cute::make_tensor<Element>(thread_coordinate_a.shape());
  static_assert(cute::size(decltype(fragment_a){}) == 8);
  const auto decode_plan =
      qvq_p32_window_lane_plan<TransitionBits>(thread & 31);

#pragma unroll
  for (int k16_local = 0; k16_local < kP32K16TilesPerStage; ++k16_local) {
    qvq_p32_window_decode_fragment<TransitionBits, false>(
        fragment_a,
        &packed_words[k16_local][warp][0],
        decode_plan,
        packed_bank_ids[k16_local][warp],
        levels,
        0,
        0,
        alternate_bank_mask);
#pragma unroll
    for (int index = 0; index < cute::size(decltype(fragment_a){}); ++index) {
      const auto coordinate = thread_coordinate_a(index);
      const int wgmma_column = static_cast<int>(cute::get<0>(coordinate));
      const int k16_row = static_cast<int>(cute::get<1>(coordinate));
      const int tile_column = wgmma_column & 15;
      const int p32_column =
          (wgmma_column & ~15) + ((tile_column & 7) << 1) +
          (tile_column >> 3);
      const int global_k =
          k256_stage * kKPerStage + k16_local * kP32TileRows + k16_row;
      const int global_n = n64_block * kOutputColumns + p32_column;
      const int64_t output_index = transpose_output
          ? static_cast<int64_t>(global_n) * size_k + global_k
          : static_cast<int64_t>(global_k) * size_n + global_n;
      output[output_index] = fragment_a(index);
    }
  }
#endif
}

constexpr int kPrefillFoldThreads = 1024;
constexpr int kPrefillTransposeTile = 32;
constexpr int kPrefillTransposeRows = 8;

// Phase-2 folded-FP16 preparation keeps the exact Phase-1 operation order:
// normalized K-axis Hadamard, SU multiplication, normalized child-local
// N-axis Hadamard, SV multiplication, then one FP16 rounding store.  The
// decoder writes [N,K], so the first transform has contiguous rows and avoids
// Phase 1's half->float cast plus transpose materialization.
__global__ __launch_bounds__(kPrefillFoldThreads)
void qvq_p32_prefill_fold_k_axis_kernel(
    const __half* __restrict__ decoded_transposed,
    float* __restrict__ transformed_transposed,
    const float* __restrict__ input_scale,
    int size_k) {
  extern __shared__ float shared[];
  const auto padded = [](int index) { return index + (index >> 5); };
  const int row = static_cast<int>(blockIdx.x);
  const int64_t row_offset = static_cast<int64_t>(row) * size_k;

  for (int index = static_cast<int>(threadIdx.x); index < size_k;
       index += static_cast<int>(blockDim.x)) {
    shared[padded(index)] =
        __half2float(decoded_transposed[row_offset + index]);
  }
  __syncthreads();

  for (int bit = 1; bit < size_k; bit <<= 1) {
    for (int index = static_cast<int>(threadIdx.x); index < size_k;
         index += static_cast<int>(blockDim.x)) {
      const int peer = index ^ bit;
      if (index < peer) {
        const float a = shared[padded(index)];
        const float b = shared[padded(peer)];
        shared[padded(index)] = a + b;
        shared[padded(peer)] = a - b;
      }
    }
    __syncthreads();
  }

  const float reciprocal = 1.0f / sqrtf(static_cast<float>(size_k));
  for (int index = static_cast<int>(threadIdx.x); index < size_k;
       index += static_cast<int>(blockDim.x)) {
    float value = shared[padded(index)];
    value *= reciprocal;
    value *= input_scale[index];
    transformed_transposed[row_offset + index] = value;
  }
}

__global__ __launch_bounds__(kPrefillFoldThreads)
void qvq_p32_prefill_fold_k_axis_fp16_kernel(
    const __half* __restrict__ decoded_transposed,
    __half* __restrict__ transformed_transposed,
    const float* __restrict__ input_scale,
    int size_k) {
  extern __shared__ __half2 shared_pairs[];
  const auto padded_pair = [](int index) { return index + (index >> 4); };
  const int row = static_cast<int>(blockIdx.x);
  const int64_t row_offset = static_cast<int64_t>(row) * size_k;
  const int pair_count = size_k >> 1;

  for (int pair = static_cast<int>(threadIdx.x); pair < pair_count;
       pair += static_cast<int>(blockDim.x)) {
    shared_pairs[padded_pair(pair)] =
        reinterpret_cast<const __half2*>(decoded_transposed + row_offset)[pair];
  }
  __syncthreads();

  for (int pair = static_cast<int>(threadIdx.x); pair < pair_count;
       pair += static_cast<int>(blockDim.x)) {
    const __half2 value = shared_pairs[padded_pair(pair)];
    const __half a = __low2half(value);
    const __half b = __high2half(value);
    shared_pairs[padded_pair(pair)] =
        __halves2half2(__hadd(a, b), __hsub(a, b));
  }
  __syncthreads();

  for (int bit = 2; bit < size_k; bit <<= 1) {
    const int pair_bit = bit >> 1;
    for (int pair = static_cast<int>(threadIdx.x); pair < pair_count;
         pair += static_cast<int>(blockDim.x)) {
      const int peer = pair ^ pair_bit;
      if (pair < peer) {
        const __half2 a = shared_pairs[padded_pair(pair)];
        const __half2 b = shared_pairs[padded_pair(peer)];
        shared_pairs[padded_pair(pair)] = __hadd2(a, b);
        shared_pairs[padded_pair(peer)] = __hsub2(a, b);
      }
    }
    __syncthreads();
  }

  const float reciprocal = 1.0f / sqrtf(static_cast<float>(size_k));
  for (int pair = static_cast<int>(threadIdx.x); pair < pair_count;
       pair += static_cast<int>(blockDim.x)) {
    const __half2 value = shared_pairs[padded_pair(pair)];
    const int index = pair << 1;
    transformed_transposed[row_offset + index] = __float2half_rn(
        __half2float(__low2half(value)) * reciprocal * input_scale[index]);
    transformed_transposed[row_offset + index + 1] = __float2half_rn(
        __half2float(__high2half(value)) * reciprocal * input_scale[index + 1]);
  }
}

__global__ void qvq_p32_prefill_transpose_fp32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int columns) {
  __shared__ float tile[kPrefillTransposeTile][kPrefillTransposeTile + 1];
  int column = static_cast<int>(blockIdx.x) * kPrefillTransposeTile +
      static_cast<int>(threadIdx.x);
  int row = static_cast<int>(blockIdx.y) * kPrefillTransposeTile +
      static_cast<int>(threadIdx.y);
#pragma unroll
  for (int offset = 0; offset < kPrefillTransposeTile;
       offset += kPrefillTransposeRows) {
    if (column < columns && row + offset < rows) {
      tile[threadIdx.y + offset][threadIdx.x] =
          input[static_cast<int64_t>(row + offset) * columns + column];
    }
  }
  __syncthreads();

  column = static_cast<int>(blockIdx.y) * kPrefillTransposeTile +
      static_cast<int>(threadIdx.x);
  row = static_cast<int>(blockIdx.x) * kPrefillTransposeTile +
      static_cast<int>(threadIdx.y);
#pragma unroll
  for (int offset = 0; offset < kPrefillTransposeTile;
       offset += kPrefillTransposeRows) {
    if (column < rows && row + offset < columns) {
      output[static_cast<int64_t>(row + offset) * rows + column] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

__global__ void qvq_p32_prefill_transpose_fp16_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int rows,
    int columns) {
  __shared__ __half tile[kPrefillTransposeTile][kPrefillTransposeTile + 1];
  int column = static_cast<int>(blockIdx.x) * kPrefillTransposeTile +
      static_cast<int>(threadIdx.x);
  int row = static_cast<int>(blockIdx.y) * kPrefillTransposeTile +
      static_cast<int>(threadIdx.y);
#pragma unroll
  for (int offset = 0; offset < kPrefillTransposeTile;
       offset += kPrefillTransposeRows) {
    if (column < columns && row + offset < rows) {
      tile[threadIdx.y + offset][threadIdx.x] =
          input[static_cast<int64_t>(row + offset) * columns + column];
    }
  }
  __syncthreads();

  column = static_cast<int>(blockIdx.y) * kPrefillTransposeTile +
      static_cast<int>(threadIdx.x);
  row = static_cast<int>(blockIdx.x) * kPrefillTransposeTile +
      static_cast<int>(threadIdx.y);
#pragma unroll
  for (int offset = 0; offset < kPrefillTransposeTile;
       offset += kPrefillTransposeRows) {
    if (column < rows && row + offset < columns) {
      output[static_cast<int64_t>(row + offset) * rows + column] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

__global__ __launch_bounds__(kPrefillFoldThreads)
void qvq_p32_prefill_fold_n_axis_fp16_to_fp8_kernel(
    const __half* __restrict__ transformed,
    c10::Float8_e4m3fn* __restrict__ output,
    const float* __restrict__ weight_scale,
    HopperGroupedP32FoldParams params,
    int size_k,
    int size_n) {
  extern __shared__ __half2 shared_pairs[];
  const auto padded_pair = [](int index) { return index + (index >> 4); };
  const int segment = static_cast<int>(blockIdx.x) % params.segment_count;
  const int row = static_cast<int>(blockIdx.x) / params.segment_count;
  const int width = params.width[segment];
  const int start = params.n_start[segment];
  const int64_t row_offset = static_cast<int64_t>(row) * size_n + start;
  const int pair_count = width >> 1;

  if (params.output_hadamard[segment]) {
    for (int pair = static_cast<int>(threadIdx.x); pair < pair_count;
         pair += static_cast<int>(blockDim.x)) {
      shared_pairs[padded_pair(pair)] =
          reinterpret_cast<const __half2*>(transformed + row_offset)[pair];
    }
    __syncthreads();

    for (int pair = static_cast<int>(threadIdx.x); pair < pair_count;
         pair += static_cast<int>(blockDim.x)) {
      const __half2 value = shared_pairs[padded_pair(pair)];
      const __half a = __low2half(value);
      const __half b = __high2half(value);
      shared_pairs[padded_pair(pair)] =
          __halves2half2(__hadd(a, b), __hsub(a, b));
    }
    __syncthreads();

    for (int bit = 2; bit < width; bit <<= 1) {
      const int pair_bit = bit >> 1;
      for (int pair = static_cast<int>(threadIdx.x); pair < pair_count;
           pair += static_cast<int>(blockDim.x)) {
        const int peer = pair ^ pair_bit;
        if (pair < peer) {
          const __half2 a = shared_pairs[padded_pair(pair)];
          const __half2 b = shared_pairs[padded_pair(peer)];
          shared_pairs[padded_pair(pair)] = __hadd2(a, b);
          shared_pairs[padded_pair(peer)] = __hsub2(a, b);
        }
      }
      __syncthreads();
    }
    const float reciprocal = 1.0f / sqrtf(static_cast<float>(width));
    for (int pair = static_cast<int>(threadIdx.x); pair < pair_count;
         pair += static_cast<int>(blockDim.x)) {
      const __half2 packed = shared_pairs[padded_pair(pair)];
      const int index = pair << 1;
      float low = __half2float(__low2half(packed));
      low *= reciprocal;
      low *= params.output_scale[segment][index];
      low = __half2float(__float2half_rn(low));
      low = __half2float(__float2half_rn(low / weight_scale[0]));
      low = fminf(448.0f, fmaxf(-448.0f, low));
      output[static_cast<int64_t>(start + index) * size_k + row] =
          c10::Float8_e4m3fn(low);
      float high = __half2float(__high2half(packed));
      high *= reciprocal;
      high *= params.output_scale[segment][index + 1];
      high = __half2float(__float2half_rn(high));
      high = __half2float(__float2half_rn(high / weight_scale[0]));
      high = fminf(448.0f, fmaxf(-448.0f, high));
      output[static_cast<int64_t>(start + index + 1) * size_k + row] =
          c10::Float8_e4m3fn(high);
    }
  } else {
    for (int pair = static_cast<int>(threadIdx.x); pair < pair_count;
         pair += static_cast<int>(blockDim.x)) {
      const int index = pair << 1;
      const __half2 packed =
          reinterpret_cast<const __half2*>(transformed + row_offset)[pair];
      float low = __half2float(__low2half(packed));
      low *= params.output_scale[segment][index];
      low = __half2float(__float2half_rn(low));
      low = __half2float(__float2half_rn(low / weight_scale[0]));
      low = fminf(448.0f, fmaxf(-448.0f, low));
      output[static_cast<int64_t>(start + index) * size_k + row] =
          c10::Float8_e4m3fn(low);
      float high = __half2float(__high2half(packed));
      high *= params.output_scale[segment][index + 1];
      high = __half2float(__float2half_rn(high));
      high = __half2float(__float2half_rn(high / weight_scale[0]));
      high = fminf(448.0f, fmaxf(-448.0f, high));
      output[static_cast<int64_t>(start + index + 1) * size_k + row] =
          c10::Float8_e4m3fn(high);
    }
  }
}

template <bool OutputFp8>
__global__ __launch_bounds__(kPrefillFoldThreads)
void qvq_p32_prefill_fold_n_axis_kernel(
    const float* __restrict__ transformed,
    std::conditional_t<OutputFp8, c10::Float8_e4m3fn, __half>*
        __restrict__ output,
    const float* __restrict__ weight_scale,
    HopperGroupedP32FoldParams params,
    int size_k,
    int size_n) {
  extern __shared__ float shared[];
  const auto padded = [](int index) { return index + (index >> 5); };
  const int segment = static_cast<int>(blockIdx.x) % params.segment_count;
  const int row = static_cast<int>(blockIdx.x) / params.segment_count;
  const int width = params.width[segment];
  const int start = params.n_start[segment];
  const int64_t row_offset = static_cast<int64_t>(row) * size_n + start;

  if (params.output_hadamard[segment]) {
    for (int index = static_cast<int>(threadIdx.x); index < width;
         index += static_cast<int>(blockDim.x)) {
      shared[padded(index)] = transformed[row_offset + index];
    }
    __syncthreads();
    for (int bit = 1; bit < width; bit <<= 1) {
      for (int index = static_cast<int>(threadIdx.x); index < width;
           index += static_cast<int>(blockDim.x)) {
        const int peer = index ^ bit;
        if (index < peer) {
          const float a = shared[padded(index)];
          const float b = shared[padded(peer)];
          shared[padded(index)] = a + b;
          shared[padded(peer)] = a - b;
        }
      }
      __syncthreads();
    }
    const float reciprocal = 1.0f / sqrtf(static_cast<float>(width));
    for (int index = static_cast<int>(threadIdx.x); index < width;
         index += static_cast<int>(blockDim.x)) {
      float value = shared[padded(index)];
      value *= reciprocal;
      value *= params.output_scale[segment][index];
      if constexpr (OutputFp8) {
        // Match the persistent FP8 control exactly: the folded effective
        // weight first crosses its FP16 boundary, then is quantized to E4M3.
        const float rounded = __half2float(__float2half_rn(value));
        const float normalized =
            __half2float(__float2half_rn(rounded / weight_scale[0]));
        output[static_cast<int64_t>(start + index) * size_k + row] =
            c10::Float8_e4m3fn(normalized);
      } else {
        output[row_offset + index] = __float2half_rn(value);
      }
    }
  } else {
    for (int index = static_cast<int>(threadIdx.x); index < width;
         index += static_cast<int>(blockDim.x)) {
      float value = transformed[row_offset + index];
      value *= params.output_scale[segment][index];
      if constexpr (OutputFp8) {
        const float rounded = __half2float(__float2half_rn(value));
        const float normalized =
            __half2float(__float2half_rn(rounded / weight_scale[0]));
        output[static_cast<int64_t>(start + index) * size_k + row] =
            c10::Float8_e4m3fn(normalized);
      } else {
        output[row_offset + index] = __float2half_rn(value);
      }
    }
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
    bool FixedGateUp = false,
    bool PrefetchDecodedLevels = false,
    int N64BlocksPerCta = 1,
    int FixedQwenProjection = 0,
    int RowTilesPerCta = 1,
    class InputTma,
    class TrellisTma,
    class BankTma,
    class GroupedParams>
__global__ __launch_bounds__(
    kTmaThreads + (N64BlocksPerCta - 1) * kThreads)
void qvq_p32_window_wgmma_m16_tma_kernel(
    CUTE_GRID_CONSTANT InputTma const input_tma,
    CUTE_GRID_CONSTANT TrellisTma const trellis_tma,
    CUTE_GRID_CONSTANT BankTma const bank_tma,
    const Element* __restrict__ levels,
    float* __restrict__ partial_output,
    GroupedParams grouped_params,
    int launch_size_m,
    int size_k,
    int launch_size_n,
    int launch_split_count,
    int launch_bank_alt_id) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  static_assert(!FixedGateUp || Grouped);
  static_assert(
      FixedQwenProjection >= 0 && FixedQwenProjection <= 4);
  static_assert(
      !FixedQwenProjection ||
      (Grouped && !FixedGateUp &&
       ((FixedQwenProjection == 3 && !OrderedSplit) ||
        (FixedQwenProjection != 3 && OrderedSplit))));
  static_assert(N64BlocksPerCta == 1 || N64BlocksPerCta == 2);
  static_assert(
      RowTilesPerCta == 1 || RowTilesPerCta == 2 || RowTilesPerCta == 4 ||
          RowTilesPerCta == 5 ||
          RowTilesPerCta == 8 || RowTilesPerCta == 11);
  constexpr int kWordsPerP32Tile = 4 * TransitionBits;
  using TrellisSmemLayout =
      P32TrellisTmaSmemLayoutFor<TransitionBits, N64BlocksPerCta>;
  using SharedStorage =
      P32WgmmaTmaSharedStorageFor<
          TransitionBits, N64BlocksPerCta, RowTilesPerCta>;
  extern __shared__ __align__(128) char dynamic_shared_buffer[];
  __shared__ __align__(128) char static_shared_buffer[
      N64BlocksPerCta == 1 && RowTilesPerCta == 1
      ? sizeof(SharedStorage)
      : 1];
  auto& shared = *reinterpret_cast<SharedStorage*>(
      N64BlocksPerCta == 1 && RowTilesPerCta == 1
      ? static_shared_buffer
      : dynamic_shared_buffer);

  const int thread = static_cast<int>(threadIdx.x);
  constexpr int kConsumerThreads = kThreads * N64BlocksPerCta;
  constexpr int kLaunchThreads =
      kTmaThreads + (N64BlocksPerCta - 1) * kThreads;
  const bool is_consumer = thread < kConsumerThreads;
  const bool is_producer = !is_consumer;
  int consumer_group = 0;
  int consumer_thread = thread;
  if constexpr (N64BlocksPerCta == 2) {
    consumer_group = is_consumer ? thread / kThreads : 0;
    consumer_thread = thread - consumer_group * kThreads;
  }
  int row_tile = 0;
  int segment = static_cast<int>(blockIdx.y);
  int n64_block = static_cast<int>(blockIdx.x);
  int split = static_cast<int>(blockIdx.z);
  int n64_block_global = n64_block;
  int total_n_tiles = launch_size_n / kP32TileColumns;
  int size_n = launch_size_n;
  int split_count = launch_split_count;
  int bank_alt_id = launch_bank_alt_id;
  if constexpr (!Grouped) {
    row_tile = static_cast<int>(blockIdx.y);
    n64_block = static_cast<int>(blockIdx.x) * N64BlocksPerCta + consumer_group;
    n64_block_global = n64_block;
    if (grouped_params.launch_bank_alt_ids != nullptr) {
      bank_alt_id = grouped_params.launch_bank_alt_ids[0];
    }
  }
  int64_t output_offset = 0;
  int64_t partial_output_offset = 0;
  int trellis_block_global = n64_block_global;
  int bank_n64_block_global = n64_block_global;
  if constexpr (FixedGateUp) {
    // Equal-width grouped gate/up plans can use their rectangular CUDA grid
    // directly.  Avoid the generic flattened-work segment search and integer
    // division in every thread while retaining child-local banks and ordered
    // partial planes.
    // Ordered split is the fixed Qwen decode schedule.  The unsplit
    // row-reuse path also serves the measured Llama and Qwen large-M
    // geometries, so take its equal child width from the launch instead of
    // hard-coding Llama's N=8192.  The division is outside the K hot loop.
    const int fixed_n =
        OrderedSplit ? kFixedQwenGateUpN : launch_size_n / 2;
    constexpr int kFixedSplit =
        OrderedSplit ? kFixedQwenGateUpSplit : 1;
    const int n64_blocks = fixed_n / kOutputColumns;
    const int n16_tiles = fixed_n / kP32TileColumns;
    segment = static_cast<int>(blockIdx.y) & 1;
    row_tile = static_cast<int>(blockIdx.y) >> 1;
    const int n64_block_base =
        static_cast<int>(blockIdx.x) * N64BlocksPerCta;
    n64_block = n64_block_base + consumer_group;
    split = OrderedSplit ? static_cast<int>(blockIdx.z) : 0;
    n64_block_global = segment * n64_blocks + n64_block;
    trellis_block_global =
        segment * (n64_blocks / N64BlocksPerCta) +
        static_cast<int>(blockIdx.x);
    bank_n64_block_global = segment * n64_blocks + n64_block_base;
    total_n_tiles = 2 * n16_tiles;
    size_n = fixed_n;
    split_count = kFixedSplit;
    bank_alt_id = grouped_params.bank_alt_id[segment];
    output_offset = static_cast<int64_t>(segment) * launch_size_m * fixed_n;
    partial_output_offset = OrderedSplit
        ? static_cast<int64_t>(segment) * kFixedSplit * launch_size_m * fixed_n
        : 0;
  } else if constexpr (FixedQwenProjection == 3) {
    // Flash-Next full-attention Q/K/V has one 192-block query child and two
    // eight-block key/value children.  Flatten the three unsplit ranges into
    // 208 useful CTAs instead of launching a 192 x 3 rectangle whose 368
    // narrow-child CTAs immediately return.
    constexpr int kQBlocks = kFixedFlashNextFullQn / kOutputColumns;
    constexpr int kKvBlocks = kFixedFlashNextFullKvN / kOutputColumns;
    const int work_item = static_cast<int>(blockIdx.x);
    segment = work_item < kQBlocks ? 0 : work_item < kQBlocks + kKvBlocks ? 1 : 2;
    const int segment_start =
        segment == 0 ? 0 : segment == 1 ? kQBlocks : kQBlocks + kKvBlocks;
    n64_block = work_item - segment_start;
    n64_block_global = work_item;
    size_n = segment == 0 ? kFixedFlashNextFullQn : kFixedFlashNextFullKvN;
    total_n_tiles =
        (kFixedFlashNextFullQn + 2 * kFixedFlashNextFullKvN) /
        kP32TileColumns;
    split = 0;
    split_count = 1;
    bank_alt_id = grouped_params.bank_alt_id[segment];
    output_offset = static_cast<int64_t>(launch_size_m) *
        (segment == 0
             ? 0
             : segment == 1
                 ? kFixedFlashNextFullQn
                 : kFixedFlashNextFullQn + kFixedFlashNextFullKvN);
  } else if constexpr (FixedQwenProjection == 4) {
    // Ordered Flash-Next full Q/K/V uses one rate-specific query split and
    // split ten for both narrow children.  Resolve each flattened work range
    // with compile-time boundaries and divisors rather than scanning runtime
    // segment metadata in every thread.
    static_assert(TransitionBits == 5 || TransitionBits == 7);
    constexpr int kQBlocks = kFixedFlashNextFullQn / kOutputColumns;
    constexpr int kKvBlocks = kFixedFlashNextFullKvN / kOutputColumns;
    constexpr int kQSplit = TransitionBits == 5 ? 5 : 2;
    constexpr int kKvSplit = 10;
    constexpr int kQWorkItems = kQBlocks * kQSplit;
    constexpr int kKvWorkItems = kKvBlocks * kKvSplit;
    const int work_item = static_cast<int>(blockIdx.x);
    segment =
        work_item < kQWorkItems
        ? 0
        : work_item < kQWorkItems + kKvWorkItems ? 1 : 2;
    const int segment_work_item =
        segment == 0
        ? work_item
        : segment == 1
            ? work_item - kQWorkItems
            : work_item - kQWorkItems - kKvWorkItems;
    if (segment == 0) {
      split = segment_work_item / kQBlocks;
      n64_block = segment_work_item - split * kQBlocks;
      n64_block_global = n64_block;
      size_n = kFixedFlashNextFullQn;
      split_count = kQSplit;
    } else {
      split = segment_work_item / kKvBlocks;
      n64_block = segment_work_item - split * kKvBlocks;
      n64_block_global =
          kQBlocks + (segment - 1) * kKvBlocks + n64_block;
      size_n = kFixedFlashNextFullKvN;
      split_count = kKvSplit;
    }
    total_n_tiles =
        (kFixedFlashNextFullQn + 2 * kFixedFlashNextFullKvN) /
        kP32TileColumns;
    bank_alt_id = grouped_params.bank_alt_id[segment];
    output_offset = grouped_params.output_offset[segment];
    partial_output_offset = grouped_params.partial_output_offset[segment];
  } else if constexpr (FixedQwenProjection) {
    // Qwen3.8 linear-attention input projections have two fixed unequal
    // children.  Resolve the segment from one compile-time work boundary and
    // divide by compile-time N64 counts instead of searching runtime segment
    // descriptors and dividing by runtime widths in every thread.
    constexpr int kQkvN64Blocks = kFixedQwenLinearQkvN / kOutputColumns;
    constexpr int kZN64Blocks = kFixedQwenLinearZN / kOutputColumns;
    // The original Qwen3-Next schedule uses K=5120 and asymmetric 10/20
    // (W2/W2.5) or 4/20 (W3) splits.  Flash-Next keeps the same two output
    // widths but halves K and uses the measured 2/2 schedule.  Both layouts
    // preserve child-local ordered reduction, so share the direct mapping
    // while selecting the boundary from the launch's fixed geometry.
    constexpr bool kIsFlashNext = FixedQwenProjection == 2;
    constexpr int kQkvSplit =
        kIsFlashNext ? 2 : (TransitionBits <= 5 ? 10 : 4);
    constexpr int kZSplit = kIsFlashNext ? 2 : 20;
    constexpr int kQkvWorkItems = kQkvN64Blocks * kQkvSplit;
    row_tile = static_cast<int>(blockIdx.y);
    const int work_item = static_cast<int>(blockIdx.x);
    segment = work_item >= kQkvWorkItems ? 1 : 0;
    const int segment_work_item =
        segment == 0 ? work_item : work_item - kQkvWorkItems;
    if (segment == 0) {
      split = segment_work_item / kQkvN64Blocks;
      n64_block = segment_work_item - split * kQkvN64Blocks;
      size_n = kFixedQwenLinearQkvN;
      split_count = kQkvSplit;
      n64_block_global = n64_block;
      output_offset = 0;
      partial_output_offset = 0;
    } else {
      split = segment_work_item / kZN64Blocks;
      n64_block = segment_work_item - split * kZN64Blocks;
      size_n = kFixedQwenLinearZN;
      split_count = kZSplit;
      n64_block_global = kQkvN64Blocks + n64_block;
      output_offset = static_cast<int64_t>(launch_size_m) * kFixedQwenLinearQkvN;
      partial_output_offset =
          static_cast<int64_t>(kQkvSplit) * launch_size_m * kFixedQwenLinearQkvN;
    }
    total_n_tiles =
        (kFixedQwenLinearQkvN + kFixedQwenLinearZN) / kP32TileColumns;
    bank_alt_id = grouped_params.bank_alt_id[segment];
  } else if constexpr (Grouped) {
    if constexpr (OrderedSplit) {
      row_tile = static_cast<int>(blockIdx.y);
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
    } else {
      const int grid_segment = static_cast<int>(blockIdx.y);
      segment = grid_segment % grouped_params.segment_count;
      row_tile = grid_segment / grouped_params.segment_count;
      if constexpr (N64BlocksPerCta == 2) {
        n64_block = static_cast<int>(blockIdx.x) * 2 + consumer_group;
      }
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
  if constexpr (N64BlocksPerCta == 1) {
    trellis_block_global = n64_block_global;
    bank_n64_block_global = n64_block_global;
  } else if constexpr (!FixedGateUp) {
    // The measured generic N128 path is a single-child Qwen down projection.
    // Its two consumers own adjacent N64 blocks while one TMA tile stages
    // their eight contiguous N16 payloads and their shared input rows.
    trellis_block_global = n64_block_global / N64BlocksPerCta;
    bank_n64_block_global = n64_block_global - consumer_group;
  }
  const int row_tile_begin = row_tile * RowTilesPerCta;
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
  pipeline_params.is_leader = thread == kConsumerThreads;
  pipeline_params.num_consumers = kConsumerThreads;
  pipeline_params.transaction_bytes =
      RowTilesPerCta * kRows * kKPerStage * sizeof(Element) +
      kWordsPerP32Tile * kP32N16TilesPerBlock * N64BlocksPerCta *
          kP32K16TilesPerStage * sizeof(uint32_t) +
      16 * kP32K16TilesPerStage * sizeof(uint8_t);
  WgmmaTmaPipeline pipeline(
      shared.pipeline,
      pipeline_params,
      cute::Shape<cute::_1, cute::_1, cute::_1>{});

  auto s_input = cute::make_tensor(
      cute::make_smem_ptr(shared.input.begin()), WgmmaTmaSmemLayoutB{});
  auto s_input1 = cute::make_tensor(
      cute::make_smem_ptr(
          shared.input.begin() + cute::cosize_v<WgmmaTmaSmemLayoutB>),
      WgmmaTmaSmemLayoutB{});
  auto s_input2 = cute::make_tensor(
      cute::make_smem_ptr(
          shared.input.begin() + 2 * cute::cosize_v<WgmmaTmaSmemLayoutB>),
      WgmmaTmaSmemLayoutB{});
  auto s_input3 = cute::make_tensor(
      cute::make_smem_ptr(
          shared.input.begin() + 3 * cute::cosize_v<WgmmaTmaSmemLayoutB>),
      WgmmaTmaSmemLayoutB{});
  auto s_input4 = cute::make_tensor(
      cute::make_smem_ptr(
          shared.input.begin() + 4 * cute::cosize_v<WgmmaTmaSmemLayoutB>),
      WgmmaTmaSmemLayoutB{});
  auto s_input5 = cute::make_tensor(
      cute::make_smem_ptr(
          shared.input.begin() + 5 * cute::cosize_v<WgmmaTmaSmemLayoutB>),
      WgmmaTmaSmemLayoutB{});
  auto s_input6 = cute::make_tensor(
      cute::make_smem_ptr(
          shared.input.begin() + 6 * cute::cosize_v<WgmmaTmaSmemLayoutB>),
      WgmmaTmaSmemLayoutB{});
  auto s_input7 = cute::make_tensor(
      cute::make_smem_ptr(
          shared.input.begin() + 7 * cute::cosize_v<WgmmaTmaSmemLayoutB>),
      WgmmaTmaSmemLayoutB{});
  auto s_input8 = cute::make_tensor(
      cute::make_smem_ptr(
          shared.input.begin() + 8 * cute::cosize_v<WgmmaTmaSmemLayoutB>),
      WgmmaTmaSmemLayoutB{});
  auto s_input9 = cute::make_tensor(
      cute::make_smem_ptr(
          shared.input.begin() + 9 * cute::cosize_v<WgmmaTmaSmemLayoutB>),
      WgmmaTmaSmemLayoutB{});
  auto s_input10 = cute::make_tensor(
      cute::make_smem_ptr(
          shared.input.begin() + 10 * cute::cosize_v<WgmmaTmaSmemLayoutB>),
      WgmmaTmaSmemLayoutB{});
  auto s_trellis = cute::make_tensor(
      cute::make_smem_ptr(shared.trellis.begin()), TrellisSmemLayout{});
  auto s_bank_ids = cute::make_tensor(
      cute::make_smem_ptr(shared.bank_ids.begin()), P32BankTmaSmemLayout{});

  auto full_input = input_tma.get_tma_tensor(cute::make_shape(launch_size_m, size_k));
  auto tiled_input = cute::local_tile(
      full_input,
      cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(row_tile_begin, cute::_));
  auto [tma_global_input, tma_shared_input] = cute::tma_partition(
      input_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input),
      cute::group_modes<0, 2>(tiled_input));
  auto tiled_input1 = cute::local_tile(
      full_input,
      cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(row_tile_begin + 1, cute::_));
  auto [tma_global_input1, tma_shared_input1] = cute::tma_partition(
      input_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input1),
      cute::group_modes<0, 2>(tiled_input1));
  auto tiled_input2 = cute::local_tile(
      full_input,
      cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(row_tile_begin + 2, cute::_));
  auto [tma_global_input2, tma_shared_input2] = cute::tma_partition(
      input_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input2),
      cute::group_modes<0, 2>(tiled_input2));
  auto tiled_input3 = cute::local_tile(
      full_input,
      cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(row_tile_begin + 3, cute::_));
  auto [tma_global_input3, tma_shared_input3] = cute::tma_partition(
      input_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input3),
      cute::group_modes<0, 2>(tiled_input3));
  auto tiled_input4 = cute::local_tile(
      full_input,
      cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(row_tile_begin + 4, cute::_));
  auto [tma_global_input4, tma_shared_input4] = cute::tma_partition(
      input_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input4),
      cute::group_modes<0, 2>(tiled_input4));
  auto tiled_input5 = cute::local_tile(
      full_input,
      cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(row_tile_begin + 5, cute::_));
  auto [tma_global_input5, tma_shared_input5] = cute::tma_partition(
      input_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input5),
      cute::group_modes<0, 2>(tiled_input5));
  auto tiled_input6 = cute::local_tile(
      full_input,
      cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(row_tile_begin + 6, cute::_));
  auto [tma_global_input6, tma_shared_input6] = cute::tma_partition(
      input_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input6),
      cute::group_modes<0, 2>(tiled_input6));
  auto tiled_input7 = cute::local_tile(
      full_input,
      cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(row_tile_begin + 7, cute::_));
  auto [tma_global_input7, tma_shared_input7] = cute::tma_partition(
      input_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input7),
      cute::group_modes<0, 2>(tiled_input7));
  auto tiled_input8 = cute::local_tile(
      full_input,
      cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(row_tile_begin + 8, cute::_));
  auto [tma_global_input8, tma_shared_input8] = cute::tma_partition(
      input_tma, cute::Int<0>{}, cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input8), cute::group_modes<0, 2>(tiled_input8));
  auto tiled_input9 = cute::local_tile(
      full_input, cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(row_tile_begin + 9, cute::_));
  auto [tma_global_input9, tma_shared_input9] = cute::tma_partition(
      input_tma, cute::Int<0>{}, cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input9), cute::group_modes<0, 2>(tiled_input9));
  auto tiled_input10 = cute::local_tile(
      full_input, cute::make_shape(cute::_16{}, cute::_256{}),
      cute::make_coord(row_tile_begin + 10, cute::_));
  auto [tma_global_input10, tma_shared_input10] = cute::tma_partition(
      input_tma, cute::Int<0>{}, cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_input10), cute::group_modes<0, 2>(tiled_input10));
  auto p32_full_trellis = trellis_tma.get_tma_tensor(
      cute::make_shape(cute::Int<kWordsPerP32Tile>{}, total_n_tiles, k_tiles));
  auto tiled_trellis = cute::local_tile(
      p32_full_trellis,
      cute::make_shape(
          cute::Int<kWordsPerP32Tile>{},
          cute::Int<kP32N16TilesPerBlock * N64BlocksPerCta>{},
          cute::Int<kP32K16TilesPerStage>{}),
      cute::make_coord(cute::_0{}, trellis_block_global, cute::_));
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
      cute::make_coord(bank_n64_block_global >> 2, cute::_));
  auto [tma_global_bank_ids, tma_shared_bank_ids] = cute::tma_partition(
      bank_tma,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(s_bank_ids),
      cute::group_modes<0, 2>(tiled_bank_ids));

  if constexpr (TransitionBits >= 4) {
    auto* vectors = reinterpret_cast<uint4*>(shared.levels.begin());
    for (int entry = thread; entry < 256 * 4; entry += kLaunchThreads) {
      const int index = entry >> 2;
      const auto* level_bits = reinterpret_cast<const uint16_t*>(levels);
      const uint16_t low_bits = level_bits[index];
      const uint16_t high_bits = level_bits[index ^ (index >> 7)];
      const uint32_t pair = static_cast<uint32_t>(low_bits) |
          (static_cast<uint32_t>(high_bits) << 16);
      const uint4 replicated = make_uint4(pair, pair, pair, pair);
      vectors[entry] = replicated;
    }
  } else {
    for (int index = thread; index < 256; index += kLaunchThreads) {
      shared.levels.begin()[index] = levels[index];
      shared.levels_high.begin()[index] = levels[index ^ (index >> 7)];
    }
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
        if constexpr (RowTilesPerCta == 2) {
          cute::copy(
              input_tma.with(*barrier),
              tma_global_input1(cute::_, global_stage),
              tma_shared_input1(cute::_, write_stage));
        }
        if constexpr (RowTilesPerCta == 4 || RowTilesPerCta == 5) {
          cute::copy(
              input_tma.with(*barrier),
              tma_global_input1(cute::_, global_stage),
              tma_shared_input1(cute::_, write_stage));
          cute::copy(
              input_tma.with(*barrier),
              tma_global_input2(cute::_, global_stage),
              tma_shared_input2(cute::_, write_stage));
          cute::copy(
              input_tma.with(*barrier),
              tma_global_input3(cute::_, global_stage),
              tma_shared_input3(cute::_, write_stage));
        }
        if constexpr (RowTilesPerCta == 5) {
          cute::copy(
              input_tma.with(*barrier),
              tma_global_input4(cute::_, global_stage),
              tma_shared_input4(cute::_, write_stage));
        }
        if constexpr (RowTilesPerCta == 8) {
          cute::copy(
              input_tma.with(*barrier),
              tma_global_input1(cute::_, global_stage),
              tma_shared_input1(cute::_, write_stage));
          cute::copy(
              input_tma.with(*barrier),
              tma_global_input2(cute::_, global_stage),
              tma_shared_input2(cute::_, write_stage));
          cute::copy(
              input_tma.with(*barrier),
              tma_global_input3(cute::_, global_stage),
              tma_shared_input3(cute::_, write_stage));
          cute::copy(
              input_tma.with(*barrier),
              tma_global_input4(cute::_, global_stage),
              tma_shared_input4(cute::_, write_stage));
          cute::copy(
              input_tma.with(*barrier),
              tma_global_input5(cute::_, global_stage),
              tma_shared_input5(cute::_, write_stage));
          cute::copy(
              input_tma.with(*barrier),
              tma_global_input6(cute::_, global_stage),
              tma_shared_input6(cute::_, write_stage));
          cute::copy(
              input_tma.with(*barrier),
              tma_global_input7(cute::_, global_stage),
              tma_shared_input7(cute::_, write_stage));
        }
        if constexpr (RowTilesPerCta > 8) {
          cute::copy(input_tma.with(*barrier), tma_global_input1(cute::_, global_stage), tma_shared_input1(cute::_, write_stage));
          cute::copy(input_tma.with(*barrier), tma_global_input2(cute::_, global_stage), tma_shared_input2(cute::_, write_stage));
          cute::copy(input_tma.with(*barrier), tma_global_input3(cute::_, global_stage), tma_shared_input3(cute::_, write_stage));
          cute::copy(input_tma.with(*barrier), tma_global_input4(cute::_, global_stage), tma_shared_input4(cute::_, write_stage));
          cute::copy(input_tma.with(*barrier), tma_global_input5(cute::_, global_stage), tma_shared_input5(cute::_, write_stage));
          cute::copy(input_tma.with(*barrier), tma_global_input6(cute::_, global_stage), tma_shared_input6(cute::_, write_stage));
          cute::copy(input_tma.with(*barrier), tma_global_input7(cute::_, global_stage), tma_shared_input7(cute::_, write_stage));
          cute::copy(input_tma.with(*barrier), tma_global_input8(cute::_, global_stage), tma_shared_input8(cute::_, write_stage));
          cute::copy(input_tma.with(*barrier), tma_global_input9(cute::_, global_stage), tma_shared_input9(cute::_, write_stage));
          cute::copy(input_tma.with(*barrier), tma_global_input10(cute::_, global_stage), tma_shared_input10(cute::_, write_stage));
        }
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
  auto thread_mma = tiled_mma.get_thread_slice(consumer_thread);
  auto thread_shared_b = thread_mma.partition_B(s_input);
  auto fragment_b = thread_mma.make_fragment_B(thread_shared_b);
  auto thread_shared_b1 = thread_mma.partition_B(s_input1);
  auto fragment_b1 = thread_mma.make_fragment_B(thread_shared_b1);
  auto thread_shared_b2 = thread_mma.partition_B(s_input2);
  auto fragment_b2 = thread_mma.make_fragment_B(thread_shared_b2);
  auto thread_shared_b3 = thread_mma.partition_B(s_input3);
  auto fragment_b3 = thread_mma.make_fragment_B(thread_shared_b3);
  auto thread_shared_b4 = thread_mma.partition_B(s_input4);
  auto fragment_b4 = thread_mma.make_fragment_B(thread_shared_b4);
  auto thread_shared_b5 = thread_mma.partition_B(s_input5);
  auto fragment_b5 = thread_mma.make_fragment_B(thread_shared_b5);
  auto thread_shared_b6 = thread_mma.partition_B(s_input6);
  auto fragment_b6 = thread_mma.make_fragment_B(thread_shared_b6);
  auto thread_shared_b7 = thread_mma.partition_B(s_input7);
  auto fragment_b7 = thread_mma.make_fragment_B(thread_shared_b7);
  auto thread_shared_b8 = thread_mma.partition_B(s_input8);
  auto fragment_b8 = thread_mma.make_fragment_B(thread_shared_b8);
  auto thread_shared_b9 = thread_mma.partition_B(s_input9);
  auto fragment_b9 = thread_mma.make_fragment_B(thread_shared_b9);
  auto thread_shared_b10 = thread_mma.partition_B(s_input10);
  auto fragment_b10 = thread_mma.make_fragment_B(thread_shared_b10);
  static_assert(cute::size<2>(decltype(fragment_b){}) == 16);

  auto coordinate_a = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_a = thread_mma.partition_A(coordinate_a);
  auto fragment_a0 = cute::make_tensor<Element>(thread_coordinate_a.shape());
  auto fragment_a1 = cute::make_tensor<Element>(thread_coordinate_a.shape());
  auto fragment_a2 = cute::make_tensor<Element>(thread_coordinate_a.shape());
  auto fragment_a3 = cute::make_tensor<Element>(thread_coordinate_a.shape());
  static_assert(cute::size(decltype(fragment_a0){}) == 8);

  auto coordinate_c = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_c = thread_mma.partition_C(coordinate_c);
  auto accumulator = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator1 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator2 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator3 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator4 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator5 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator6 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator7 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator8 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator9 = cute::make_tensor<float>(thread_coordinate_c.shape());
  auto accumulator10 = cute::make_tensor<float>(thread_coordinate_c.shape());
  cute::clear(accumulator);
  cute::clear(accumulator1);
  cute::clear(accumulator2);
  cute::clear(accumulator3);
  cute::clear(accumulator4);
  cute::clear(accumulator5);
  cute::clear(accumulator6);
  cute::clear(accumulator7);
  cute::clear(accumulator8);
  cute::clear(accumulator9);
  cute::clear(accumulator10);
  tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;

  const int warp = consumer_thread >> 5;
  const int lane = consumer_thread & 31;
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
      // W2.5/W3 use the measured depth-four schedule now that the fragment
      // reuse wait occurs at the first overwrite. W2/W3.5 retain depth two.
      constexpr int kDefaultDecodeDepth =
          TransitionBits == 5 || TransitionBits == kW3TransitionBits ? 4 : 2;
      constexpr int kDecodeDepth =
          N64BlocksPerCta == 2 && TransitionBits == 5
          ? kW25N128DecodeDepth
          : kDefaultDecodeDepth;
      auto& fragment_a = (k_block % kDecodeDepth) == 0 ? fragment_a0
          : (k_block % kDecodeDepth) == 1 ? fragment_a1
          : (k_block % kDecodeDepth) == 2 ? fragment_a2
                                         : fragment_a3;
      const uint32_t bank_id = s_bank_ids(bank_n16_offset + warp, k_block, read_stage);
      const auto trellis_layout = TrellisSmemLayout{};
      const uint32_t* window_words = shared.trellis.begin() +
          trellis_layout(
              0,
              consumer_group * kP32N16TilesPerBlock + warp,
              k_block,
              read_stage);
      qvq_p32_window_decode_fragment<
          TransitionBits,
          true,
          PrefetchDecodedLevels,
          kDecodeDepth - 1>(
          fragment_a,
          window_words,
          decode_plan,
          static_cast<uint8_t>(bank_id),
          shared.levels.begin(),
          shared_levels_base,
          shared_levels_high_base,
          alternate_bank_mask,
          k_block >= kDecodeDepth);
      cute::warpgroup_fence_operand(fragment_a);
      cute::warpgroup_arrive();
      cute::gemm(
          tiled_mma,
          fragment_a(cute::_, cute::_, cute::_0{}),
          fragment_b(cute::_, cute::_, k_block, read_stage),
          accumulator);
      if constexpr (RowTilesPerCta == 2) {
        cute::gemm(
            tiled_mma,
            fragment_a(cute::_, cute::_, cute::_0{}),
            fragment_b1(cute::_, cute::_, k_block, read_stage),
            accumulator1);
      }
      if constexpr (RowTilesPerCta == 4 || RowTilesPerCta == 5) {
        cute::gemm(
            tiled_mma,
            fragment_a(cute::_, cute::_, cute::_0{}),
            fragment_b1(cute::_, cute::_, k_block, read_stage),
            accumulator1);
        cute::gemm(
            tiled_mma,
            fragment_a(cute::_, cute::_, cute::_0{}),
            fragment_b2(cute::_, cute::_, k_block, read_stage),
            accumulator2);
        cute::gemm(
            tiled_mma,
            fragment_a(cute::_, cute::_, cute::_0{}),
            fragment_b3(cute::_, cute::_, k_block, read_stage),
            accumulator3);
      }
      if constexpr (RowTilesPerCta == 5) {
        cute::gemm(
            tiled_mma,
            fragment_a(cute::_, cute::_, cute::_0{}),
            fragment_b4(cute::_, cute::_, k_block, read_stage),
            accumulator4);
      }
      if constexpr (RowTilesPerCta == 8) {
        cute::gemm(
            tiled_mma,
            fragment_a(cute::_, cute::_, cute::_0{}),
            fragment_b1(cute::_, cute::_, k_block, read_stage),
            accumulator1);
        cute::gemm(
            tiled_mma,
            fragment_a(cute::_, cute::_, cute::_0{}),
            fragment_b2(cute::_, cute::_, k_block, read_stage),
            accumulator2);
        cute::gemm(
            tiled_mma,
            fragment_a(cute::_, cute::_, cute::_0{}),
            fragment_b3(cute::_, cute::_, k_block, read_stage),
            accumulator3);
        cute::gemm(
            tiled_mma,
            fragment_a(cute::_, cute::_, cute::_0{}),
            fragment_b4(cute::_, cute::_, k_block, read_stage),
            accumulator4);
        cute::gemm(
            tiled_mma,
            fragment_a(cute::_, cute::_, cute::_0{}),
            fragment_b5(cute::_, cute::_, k_block, read_stage),
            accumulator5);
        cute::gemm(
            tiled_mma,
            fragment_a(cute::_, cute::_, cute::_0{}),
            fragment_b6(cute::_, cute::_, k_block, read_stage),
            accumulator6);
        cute::gemm(
            tiled_mma,
            fragment_a(cute::_, cute::_, cute::_0{}),
            fragment_b7(cute::_, cute::_, k_block, read_stage),
            accumulator7);
      }
      if constexpr (RowTilesPerCta > 8) {
        cute::gemm(tiled_mma, fragment_a(cute::_, cute::_, cute::_0{}), fragment_b1(cute::_, cute::_, k_block, read_stage), accumulator1);
        cute::gemm(tiled_mma, fragment_a(cute::_, cute::_, cute::_0{}), fragment_b2(cute::_, cute::_, k_block, read_stage), accumulator2);
        cute::gemm(tiled_mma, fragment_a(cute::_, cute::_, cute::_0{}), fragment_b3(cute::_, cute::_, k_block, read_stage), accumulator3);
        cute::gemm(tiled_mma, fragment_a(cute::_, cute::_, cute::_0{}), fragment_b4(cute::_, cute::_, k_block, read_stage), accumulator4);
        cute::gemm(tiled_mma, fragment_a(cute::_, cute::_, cute::_0{}), fragment_b5(cute::_, cute::_, k_block, read_stage), accumulator5);
        cute::gemm(tiled_mma, fragment_a(cute::_, cute::_, cute::_0{}), fragment_b6(cute::_, cute::_, k_block, read_stage), accumulator6);
        cute::gemm(tiled_mma, fragment_a(cute::_, cute::_, cute::_0{}), fragment_b7(cute::_, cute::_, k_block, read_stage), accumulator7);
        cute::gemm(tiled_mma, fragment_a(cute::_, cute::_, cute::_0{}), fragment_b8(cute::_, cute::_, k_block, read_stage), accumulator8);
        cute::gemm(tiled_mma, fragment_a(cute::_, cute::_, cute::_0{}), fragment_b9(cute::_, cute::_, k_block, read_stage), accumulator9);
        cute::gemm(tiled_mma, fragment_a(cute::_, cute::_, cute::_0{}), fragment_b10(cute::_, cute::_, k_block, read_stage), accumulator10);
      }
      tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
      cute::warpgroup_commit_batch();
    }
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(accumulator);
    if constexpr (RowTilesPerCta == 2) {
      cute::warpgroup_fence_operand(accumulator1);
    }
    if constexpr (RowTilesPerCta == 4 || RowTilesPerCta == 5) {
      cute::warpgroup_fence_operand(accumulator1);
      cute::warpgroup_fence_operand(accumulator2);
      cute::warpgroup_fence_operand(accumulator3);
    }
    if constexpr (RowTilesPerCta == 5) {
      cute::warpgroup_fence_operand(accumulator4);
    }
    if constexpr (RowTilesPerCta == 8) {
      cute::warpgroup_fence_operand(accumulator1);
      cute::warpgroup_fence_operand(accumulator2);
      cute::warpgroup_fence_operand(accumulator3);
      cute::warpgroup_fence_operand(accumulator4);
      cute::warpgroup_fence_operand(accumulator5);
      cute::warpgroup_fence_operand(accumulator6);
      cute::warpgroup_fence_operand(accumulator7);
    }
    if constexpr (RowTilesPerCta > 8) {
      cute::warpgroup_fence_operand(accumulator1);
      cute::warpgroup_fence_operand(accumulator2);
      cute::warpgroup_fence_operand(accumulator3);
      cute::warpgroup_fence_operand(accumulator4);
      cute::warpgroup_fence_operand(accumulator5);
      cute::warpgroup_fence_operand(accumulator6);
      cute::warpgroup_fence_operand(accumulator7);
      cute::warpgroup_fence_operand(accumulator8);
      cute::warpgroup_fence_operand(accumulator9);
      cute::warpgroup_fence_operand(accumulator10);
    }
    pipeline.consumer_release(release_state);
    ++read_state;
    ++release_state;
  }

  constexpr int kAccumulatorValuesPerThread = cute::size(decltype(thread_coordinate_c){});
  constexpr bool kUseCoalescedOutput =
      FixedGateUp && !OrderedSplit && N64BlocksPerCta == 2 &&
      (RowTilesPerCta == 4 || RowTilesPerCta == 8 || RowTilesPerCta == 11);
  if constexpr (kUseCoalescedOutput) {
    // The RS-WGMMA accumulator mapping gives every consumer eight scattered
    // FP32 values. Direct stores therefore generate almost twice the ideal
    // number of global sectors on H100. All TMA stages are dead here, so
    // reclaim the input buffer as one 16x64 scratch tile per consumer group,
    // then write the tile as aligned float4 vectors. The first named barrier
    // keeps either group from overwriting pipeline storage before the other
    // group has completed its final WGMMA stage. Separate group-local
    // barriers let the independent N64 consumers progress without meeting on
    // every output row tile.
    asm volatile("bar.sync 1, 256;" ::: "memory");
    float* output_scratch =
        reinterpret_cast<float*>(shared.input.begin()) + consumer_group * 1024;
#pragma unroll
    for (int row_local = 0; row_local < RowTilesPerCta; ++row_local) {
#pragma unroll
      for (int index = 0; index < kAccumulatorValuesPerThread; ++index) {
        const auto coordinate = thread_coordinate_c(index);
        const int wgmma_column = static_cast<int>(cute::get<0>(coordinate));
        const int output_row = static_cast<int>(cute::get<1>(coordinate));
        const int tile_column = wgmma_column & 15;
        const int p32_column =
            (wgmma_column & ~15) + ((tile_column & 7) << 1) +
            (tile_column >> 3);
        const float value = row_local == 0 ? accumulator(index)
            : row_local == 1             ? accumulator1(index)
            : row_local == 2             ? accumulator2(index)
            : row_local == 3             ? accumulator3(index)
            : row_local == 4             ? accumulator4(index)
            : row_local == 5             ? accumulator5(index)
            : row_local == 6             ? accumulator6(index)
            : row_local == 7             ? accumulator7(index)
            : row_local == 8             ? accumulator8(index)
            : row_local == 9             ? accumulator9(index)
                                          : accumulator10(index);
        output_scratch[output_row * kOutputColumns + p32_column] = value;
      }
      if (consumer_group == 0) {
        asm volatile("bar.sync 2, 128;" ::: "memory");
      } else {
        asm volatile("bar.sync 3, 128;" ::: "memory");
      }

#pragma unroll
      for (int vector = consumer_thread; vector < 256; vector += kThreads) {
        const int output_row = vector >> 4;
        const int output_column = (vector & 15) << 2;
        const int global_output_row =
            (row_tile_begin + row_local) * kRows + output_row;
        const int64_t output_index =
            output_offset + static_cast<int64_t>(global_output_row) * size_n +
            n64_block * kOutputColumns + output_column;
        *reinterpret_cast<float4*>(partial_output + output_index) =
            reinterpret_cast<const float4*>(output_scratch)[vector];
      }
      if (consumer_group == 0) {
        asm volatile("bar.sync 2, 128;" ::: "memory");
      } else {
        asm volatile("bar.sync 3, 128;" ::: "memory");
      }
    }
    return;
  }
#pragma unroll
  for (int index = 0; index < kAccumulatorValuesPerThread; ++index) {
    const auto coordinate = thread_coordinate_c(index);
    const int wgmma_column = static_cast<int>(cute::get<0>(coordinate));
    const int output_row = static_cast<int>(cute::get<1>(coordinate));
    const int tile_column = wgmma_column & 15;
    const int p32_column = (wgmma_column & ~15) + ((tile_column & 7) << 1) + (tile_column >> 3);
#pragma unroll
    for (int row_local = 0; row_local < RowTilesPerCta; ++row_local) {
      const int global_output_row =
          (row_tile_begin + row_local) * kRows + output_row;
      const int64_t local_output_index =
          static_cast<int64_t>(global_output_row) * size_n +
          n64_block * kOutputColumns + p32_column;
      const int64_t output_index = output_offset + local_output_index;
      const float value = row_local == 0 ? accumulator(index)
          : row_local == 1             ? accumulator1(index)
          : row_local == 2             ? accumulator2(index)
          : row_local == 3             ? accumulator3(index)
          : row_local == 4             ? accumulator4(index)
          : row_local == 5             ? accumulator5(index)
          : row_local == 6             ? accumulator6(index)
          : row_local == 7             ? accumulator7(index)
          : row_local == 8             ? accumulator8(index)
          : row_local == 9             ? accumulator9(index)
                                        : accumulator10(index);
      if constexpr (OrderedSplit) {
        partial_output[
            partial_output_offset +
            static_cast<int64_t>(split) * launch_size_m * size_n +
            local_output_index] = value;
      } else if (split_count > 1) {
        atomicAdd(partial_output + output_index, value);
      } else {
        partial_output[output_index] = value;
      }
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

__global__ void qvq_fp16_to_fp8_e5m2_clamped_vector8_kernel(
    const Element* __restrict__ input,
    c10::Float8_e5m2* __restrict__ output,
    int vector_count) {
  const int vector_index = static_cast<int>(blockIdx.x) * blockDim.x +
      static_cast<int>(threadIdx.x);
  if (vector_index >= vector_count) {
    return;
  }
  const auto* input2 = reinterpret_cast<const __half2*>(input);
  const int pair_index = vector_index << 2;
  const __half2 value0 = input2[pair_index];
  const __half2 value1 = input2[pair_index + 1];
  const __half2 value2 = input2[pair_index + 2];
  const __half2 value3 = input2[pair_index + 3];
  const auto packed0 = __nv_fp8x4_e5m2(value0, value1).__x;
  const auto packed1 = __nv_fp8x4_e5m2(value2, value3).__x;
  reinterpret_cast<uint2*>(output)[vector_index] = make_uint2(packed0, packed1);
}

__global__ void qvq_fp16_to_fp8_e5m2_clamped_vector4_kernel(
    const Element* __restrict__ input,
    c10::Float8_e5m2* __restrict__ output,
    int vector_count) {
  const int vector_index = static_cast<int>(blockIdx.x) * blockDim.x +
      static_cast<int>(threadIdx.x);
  if (vector_index >= vector_count) {
    return;
  }
  const auto* input2 = reinterpret_cast<const __half2*>(input);
  const int pair_index = vector_index << 1;
  const __half2 value0 = input2[pair_index];
  const __half2 value1 = input2[pair_index + 1];
  reinterpret_cast<__nv_fp8x4_storage_t*>(output)[vector_index] =
      __nv_fp8x4_e5m2(value0, value1).__x;
}

__global__ void qvq_fp16_to_fp8_e5m2_clamped_generic_kernel(
    const Element* __restrict__ input,
    c10::Float8_e5m2* __restrict__ output,
    int64_t value_count) {
  const int64_t pair_index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      static_cast<int64_t>(threadIdx.x);
  const int64_t pair_count = value_count >> 1;
  if (pair_index < pair_count) {
    const auto value = reinterpret_cast<const __half2*>(input)[pair_index];
    reinterpret_cast<__nv_fp8x2_storage_t*>(output)[pair_index] =
        __nv_fp8x2_e5m2(value).__x;
  }
  if ((value_count & 1) && pair_index == pair_count) {
    float value = static_cast<float>(input[value_count - 1]);
    if (isfinite(value)) {
      value = fminf(57344.0f, fmaxf(-57344.0f, value));
    }
    output[value_count - 1] = c10::Float8_e5m2(value);
  }
}

#ifndef QVQ_WGMMA_DEVICE_ONLY
at::Tensor qvq_fp16_to_fp8_e5m2_clamped(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "FP8 prefill conversion requires CUDA input");
  TORCH_CHECK(input.scalar_type() == at::kHalf, "FP8 prefill conversion requires FP16 input");
  TORCH_CHECK(input.is_contiguous(), "FP8 prefill conversion requires contiguous input");
  auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e5m2));
  const int64_t value_count = input.numel();
  constexpr int kConvertThreads = 256;
  const int64_t pair_count = (value_count + 1) / 2;
  const int64_t blocks = (pair_count + kConvertThreads - 1) / kConvertThreads;
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if ((value_count & 7) == 0 &&
      value_count / 8 <= std::numeric_limits<int>::max()) {
    const int vector_count = static_cast<int>(value_count / 8);
    const int vector_blocks =
        (vector_count + kConvertThreads - 1) / kConvertThreads;
    qvq_fp16_to_fp8_e5m2_clamped_vector8_kernel
        <<<vector_blocks, kConvertThreads, 0, stream>>>(
            reinterpret_cast<const Element*>(input.data_ptr<at::Half>()),
            output.data_ptr<c10::Float8_e5m2>(),
            vector_count);
  } else if ((value_count & 3) == 0 &&
      value_count / 4 <= std::numeric_limits<int>::max()) {
    const int vector_count = static_cast<int>(value_count / 4);
    const int vector_blocks =
        (vector_count + kConvertThreads - 1) / kConvertThreads;
    qvq_fp16_to_fp8_e5m2_clamped_vector4_kernel
        <<<vector_blocks, kConvertThreads, 0, stream>>>(
            reinterpret_cast<const Element*>(input.data_ptr<at::Half>()),
            output.data_ptr<c10::Float8_e5m2>(),
            vector_count);
  } else {
    qvq_fp16_to_fp8_e5m2_clamped_generic_kernel
        <<<blocks, kConvertThreads, 0, stream>>>(
            reinterpret_cast<const Element*>(input.data_ptr<at::Half>()),
            output.data_ptr<c10::Float8_e5m2>(),
            value_count);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
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
    case 5:
      QVQ_LAUNCH_FIXED_REDUCER(5);
      break;
    case 8:
      QVQ_LAUNCH_FIXED_REDUCER(8);
      break;
    case 10:
      QVQ_LAUNCH_FIXED_REDUCER(10);
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

template <int TransitionBits, bool OrderedSplit = false, bool ReturnPartials = false>
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
  TORCH_CHECK(
      input.dim() == 2 &&
          (input.size(0) == kRows ||
           (OrderedSplit && !ReturnPartials && input.size(0) > 0 &&
            input.size(0) < kRows)),
      "QVQ P32 TMA WGMMA requires M=16, or logical M1..M15 for ordered output");
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
  const int logical_size_m = static_cast<int>(input.size(0));
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
      cute::make_shape(logical_size_m, size_k),
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

  static_assert(!ReturnPartials || OrderedSplit,
                "returning split planes requires ordered split output");
  TORCH_CHECK(!ReturnPartials || split_count > 1,
              "ordered partial output requires split_count greater than one");
  auto output = ReturnPartials
      ? at::Tensor()
      : at::empty({kRows, size_n}, input.options().dtype(at::kFloat));
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
  const bool use_h100_llama_down_prefetch =
      OrderedSplit && size_k == 8192 && size_n == 2048 && split_count == 16 &&
      std::strcmp(properties.name, "NVIDIA H100") == 0;
  const bool use_h100_qwen_down_prefetch =
      size_k == 17408 && size_n == 5120 && split_count == 17 &&
      OrderedSplit && TransitionBits == kW3TransitionBits &&
      std::strcmp(properties.name, "NVIDIA H100") == 0;
  const bool use_h100_qwen_atomic_down_prefetch =
      !OrderedSplit && TransitionBits <= 5 &&
      size_k == 17408 && size_n == 5120 && split_count == 34 &&
      std::strcmp(properties.name, "NVIDIA H100") == 0;
  if (use_h100_llama_down_prefetch || use_h100_qwen_down_prefetch ||
      use_h100_qwen_atomic_down_prefetch) {
    qvq_p32_window_wgmma_m16_tma_kernel<
        TransitionBits,
        false,
        OrderedSplit,
        false,
        true><<<grid, kTmaThreads, 0, stream>>>(
        input_tma,
        trellis_tma,
        bank_tma,
        reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
        partial_output.data_ptr<float>(),
        grouped_params,
        kRows,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  } else {
    qvq_p32_window_wgmma_m16_tma_kernel<TransitionBits, false, OrderedSplit>
        <<<grid, kTmaThreads, 0, stream>>>(
        input_tma,
        trellis_tma,
        bank_tma,
        reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
        partial_output.data_ptr<float>(),
        grouped_params,
        kRows,
        size_k,
        size_n,
        static_cast<int>(split_count),
        static_cast<int>(bank_alt_id));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if constexpr (ReturnPartials) {
    return partial_output;
  }
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

at::Tensor qvq_p32_window_wgmma_m16_tma_ordered_partials(
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
      return qvq_p32_window_wgmma_m16_tma_impl<4, true, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 5:
      return qvq_p32_window_wgmma_m16_tma_impl<5, true, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 6:
      return qvq_p32_window_wgmma_m16_tma_impl<6, true, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    case 7:
      return qvq_p32_window_wgmma_m16_tma_impl<7, true, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_id, split_count);
    default:
      TORCH_CHECK(
          false,
          "ordered-partial QVQ P32 TMA WGMMA transition bits must be in [4, 7]");
  }
}

template <
    int TransitionBits,
    bool OrderedSplit = false,
    bool ReturnPartials = false,
    int RowTilesPerCta = 1>
at::Tensor qvq_p32_window_wgmma_m16_tma_grouped_impl(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts,
    int64_t block_n = 0) {
  static_assert(!ReturnPartials || OrderedSplit);
  TORCH_CHECK(block_n == 0 || block_n == 64 || block_n == 128,
              "explicit Hopper BN must be 64 or 128");
  TORCH_CHECK(block_n == 0 || (RowTilesPerCta > 1 && !OrderedSplit),
              "explicit Hopper BN requires unsplit row reuse");
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
      input.dim() == 2 && input.size(0) >= kRows &&
          input.size(0) % kRows == 0 &&
          input.size(0) <= (RowTilesPerCta == 11 ? 4224 : 8192),
      "grouped QVQ P32 TMA WGMMA row count exceeds its specialization limit");
  TORCH_CHECK(
      input.size(1) > 0 && input.size(1) % kKPerStage == 0,
      "grouped QVQ P32 TMA WGMMA K must be a positive multiple of 256");
  const int64_t segment_count = static_cast<int64_t>(out_features.size());
  TORCH_CHECK(
      segment_count >= 1 && segment_count <= kMaxGroupedP32Segments,
      "grouped QVQ P32 TMA WGMMA requires one to three segments");
  TORCH_CHECK(
      block_n != 128 || segment_count == 1,
      "explicit grouped BN128 requires a single-segment specialization");
  TORCH_CHECK(
      static_cast<int64_t>(bank_alt_ids.size()) == segment_count &&
          static_cast<int64_t>(split_counts.size()) == segment_count,
      "grouped QVQ P32 TMA WGMMA metadata lengths must match");

  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(
      properties.major == 9 && properties.minor == 0,
      "grouped QVQ P32 TMA WGMMA requires an SM90 H100/H200 device");

  const int size_m = static_cast<int>(input.size(0));
  const int row_tiles = size_m / kRows;
  TORCH_CHECK(
      row_tiles % RowTilesPerCta == 0,
      "grouped QVQ P32 row count must be divisible by the CTA row reuse factor");
  const int row_ctas = row_tiles / RowTilesPerCta;
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
    grouped_params.output_offset[segment] = total_n * size_m;
    grouped_params.partial_output_offset[segment] = total_partial_values;
    total_work_items += (width / kOutputColumns) * split_count;
    total_partial_values += split_count * size_m * width;
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
  using WideTrellisSmemLayout =
      P32TrellisTmaSmemLayoutFor<TransitionBits, 2>;
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
      cute::make_shape(size_m, size_k),
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
  auto wide_trellis_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{},
      trellis_tensor,
      WideTrellisSmemLayout{}(cute::_, cute::_, cute::_, cute::_0{}),
      cute::make_shape(
          cute::Int<kWordsPerP32Tile>{},
          cute::Int<2 * kP32N16TilesPerBlock>{},
          cute::Int<kP32K16TilesPerStage>{}));
  auto bank_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{},
      bank_tensor,
      P32BankTmaSmemLayout{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_16{}));

  auto output = ReturnPartials
      ? at::empty({0}, input.options().dtype(at::kFloat))
      : OrderedSplit || max_split_count == 1
          ? at::empty({size_m * total_n}, input.options().dtype(at::kFloat))
          : at::zeros({size_m * total_n}, input.options().dtype(at::kFloat));
  auto partial_output = OrderedSplit
      ? at::empty({total_partial_values}, input.options().dtype(at::kFloat))
      : output;
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(input.get_device());
  const dim3 grid = OrderedSplit
      ? dim3(
            static_cast<unsigned>(total_work_items),
            static_cast<unsigned>(row_ctas),
            1)
      : dim3(
            static_cast<unsigned>(max_n64_blocks),
            static_cast<unsigned>(segment_count * row_ctas),
            static_cast<unsigned>(max_split_count));
  const bool use_gate_up_geometry =
      !OrderedSplit && size_k == kFixedGateUpK && segment_count == 2 &&
      out_features[0] == kFixedGateUpN && out_features[1] == kFixedGateUpN &&
      split_counts[0] == 1 && split_counts[1] == 1;
  const bool use_qwen_unsplit_gate_up_geometry =
      !OrderedSplit && size_k == kFixedQwenGateUpK && segment_count == 2 &&
      out_features[0] == kFixedQwenGateUpN &&
      out_features[1] == kFixedQwenGateUpN &&
      split_counts[0] == 1 && split_counts[1] == 1;
  const bool use_fixed_gate_up =
      use_gate_up_geometry &&
      (TransitionBits == 4 || TransitionBits == kW3TransitionBits);
  const bool use_prefetch_gate_up =
      use_gate_up_geometry &&
      (TransitionBits == 5 || TransitionBits == kW3TransitionBits ||
       TransitionBits == 7);
  const bool use_wide_gate_up =
      use_gate_up_geometry && TransitionBits == 5 &&
      std::strcmp(properties.name, "NVIDIA H100") == 0;
  const bool use_h100_qwen_ordered_fixed =
      OrderedSplit && size_k == 5120 && segment_count == 2 &&
      out_features[0] == 17408 && out_features[1] == 17408 &&
      split_counts[0] == split_counts[1] &&
      split_counts[0] == kFixedQwenGateUpSplit &&
      std::strcmp(properties.name, "NVIDIA H100") == 0;
  const bool use_h100_flash_next_full_qkv_fixed =
      !OrderedSplit && size_k == 2560 && segment_count == 3 &&
      out_features[0] == kFixedFlashNextFullQn &&
      out_features[1] == kFixedFlashNextFullKvN &&
      out_features[2] == kFixedFlashNextFullKvN &&
      split_counts[0] == 1 && split_counts[1] == 1 &&
      split_counts[2] == 1 &&
      (TransitionBits == 4 || TransitionBits == kW3TransitionBits) &&
      std::strcmp(properties.name, "NVIDIA H100") == 0;
  const int flash_next_full_q_split = TransitionBits == 5 ? 5 : 2;
  const bool use_h100_flash_next_full_qkv_ordered_fixed =
      OrderedSplit && size_k == 2560 && segment_count == 3 &&
      out_features[0] == kFixedFlashNextFullQn &&
      out_features[1] == kFixedFlashNextFullKvN &&
      out_features[2] == kFixedFlashNextFullKvN &&
      split_counts[0] == flash_next_full_q_split &&
      split_counts[1] == 10 && split_counts[2] == 10 &&
      (TransitionBits == 5 || TransitionBits == 7) &&
      std::strcmp(properties.name, "NVIDIA H100") == 0;
  const int qwen_linear_qkv_split = TransitionBits <= 5 ? 10 : 4;
  const bool use_h100_qwen_linear_legacy =
      OrderedSplit && size_k == 5120 && segment_count == 2 &&
      out_features[0] == kFixedQwenLinearQkvN &&
      out_features[1] == kFixedQwenLinearZN &&
      split_counts[0] == qwen_linear_qkv_split &&
      split_counts[1] == 20 &&
      TransitionBits <= kW3TransitionBits &&
      std::strcmp(properties.name, "NVIDIA H100") == 0;
  const bool use_h100_flash_next_linear_fixed =
      OrderedSplit && size_k == 2560 && segment_count == 2 &&
      out_features[0] == kFixedQwenLinearQkvN &&
      out_features[1] == kFixedQwenLinearZN &&
      split_counts[0] == 2 && split_counts[1] == 2 &&
      TransitionBits >= kW3TransitionBits &&
      std::strcmp(properties.name, "NVIDIA H100") == 0;
  if constexpr (RowTilesPerCta > 1) {
    // At M>=128 the physical H100 benefits from two independent N64
    // consumers sharing the same four staged M16 input tiles.  M64 retains
    // the narrower CTA: its shorter grid does not amortize the larger block.
    const bool use_h100_wide_reuse_gate_up =
        block_n == 0 && (use_gate_up_geometry || use_qwen_unsplit_gate_up_geometry) &&
        size_m >= 128 &&
        std::strcmp(properties.name, "NVIDIA H100") == 0;
    const bool use_h100_wide_reuse_qwen_down =
        block_n == 128 || (block_n == 0 && segment_count == 1 && size_k == 17408 && out_features[0] == 5120 &&
        split_counts[0] == 1 && size_m >= 128 &&
        std::strcmp(properties.name, "NVIDIA H100") == 0);
    if (use_h100_wide_reuse_gate_up) {
      const HopperFixedGateUpLaunchParams fixed_params{
          {grouped_params.bank_alt_id[0], grouped_params.bank_alt_id[1]}};
      using WideReuseSharedStorage =
          P32WgmmaTmaSharedStorageFor<TransitionBits, 2, RowTilesPerCta>;
      auto wide_reuse_kernel = qvq_p32_window_wgmma_m16_tma_kernel<
          TransitionBits,
          true,
          OrderedSplit,
          true,
          true,
          2,
          false,
          RowTilesPerCta,
          decltype(input_tma),
          decltype(wide_trellis_tma),
          decltype(bank_tma),
          HopperFixedGateUpLaunchParams>;
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          wide_reuse_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(sizeof(WideReuseSharedStorage))));
      const dim3 wide_reuse_grid(
          static_cast<unsigned>(max_n64_blocks / 2),
          static_cast<unsigned>(segment_count * row_ctas),
          1);
      wide_reuse_kernel<<<
          wide_reuse_grid,
          kTmaThreads + kThreads,
          sizeof(WideReuseSharedStorage),
          stream>>>(
          input_tma,
          wide_trellis_tma,
          bank_tma,
          reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
          partial_output.data_ptr<float>(),
          fixed_params,
          size_m,
          size_k,
          static_cast<int>(total_n),
          1,
          0);
    } else if (use_h100_wide_reuse_qwen_down) {
      using WideReuseSharedStorage =
          P32WgmmaTmaSharedStorageFor<TransitionBits, 2, RowTilesPerCta>;
      auto wide_reuse_kernel = qvq_p32_window_wgmma_m16_tma_kernel<
          TransitionBits,
          true,
          OrderedSplit,
          false,
          true,
          2,
          false,
          RowTilesPerCta,
          decltype(input_tma),
          decltype(wide_trellis_tma),
          decltype(bank_tma),
          HopperGroupedP32LaunchParams>;
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          wide_reuse_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(sizeof(WideReuseSharedStorage))));
      const dim3 wide_reuse_grid(
          static_cast<unsigned>(max_n64_blocks / 2),
          static_cast<unsigned>(row_ctas),
          1);
      wide_reuse_kernel<<<
          wide_reuse_grid,
          kTmaThreads + kThreads,
          sizeof(WideReuseSharedStorage),
          stream>>>(
          input_tma,
          wide_trellis_tma,
          bank_tma,
          reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
          partial_output.data_ptr<float>(),
          grouped_params,
          size_m,
          size_k,
          static_cast<int>(total_n),
          1,
          0);
    } else {
      if (block_n == 128) {
        // Explicit BN128 uses two independent N64 consumers per CTA.  This
        // branch is reserved for the validated single-segment specialization;
        // generic multi-segment grouped policies fail closed above.
        using WideReuseSharedStorage =
            P32WgmmaTmaSharedStorageFor<TransitionBits, 2, RowTilesPerCta>;
        auto wide_reuse_kernel = qvq_p32_window_wgmma_m16_tma_kernel<
            TransitionBits,
            true,
            OrderedSplit,
            false,
            true,
            2,
            false,
            RowTilesPerCta,
            decltype(input_tma),
            decltype(trellis_tma),
            decltype(bank_tma),
            HopperGroupedP32LaunchParams>;
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            wide_reuse_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(sizeof(WideReuseSharedStorage))));
        const dim3 wide_reuse_grid(
            static_cast<unsigned>(max_n64_blocks / 2),
            static_cast<unsigned>(segment_count * row_ctas),
            static_cast<unsigned>(max_split_count));
        wide_reuse_kernel<<<
            wide_reuse_grid,
            kTmaThreads + kThreads,
            sizeof(WideReuseSharedStorage),
            stream>>>(
            input_tma,
            trellis_tma,
            bank_tma,
            reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
            partial_output.data_ptr<float>(),
            grouped_params,
            size_m,
            size_k,
            static_cast<int>(total_n),
            1,
            0);
      } else {
        using ReuseSharedStorage =
            P32WgmmaTmaSharedStorageFor<TransitionBits, 1, RowTilesPerCta>;
        auto reuse_kernel = qvq_p32_window_wgmma_m16_tma_kernel<
            TransitionBits,
            true,
            OrderedSplit,
            false,
            true,
            1,
            false,
            RowTilesPerCta,
            decltype(input_tma),
            decltype(trellis_tma),
            decltype(bank_tma),
            HopperGroupedP32LaunchParams>;
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            reuse_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(sizeof(ReuseSharedStorage))));
        reuse_kernel<<<
            grid,
            kTmaThreads,
            sizeof(ReuseSharedStorage),
            stream>>>(
            input_tma,
            trellis_tma,
            bank_tma,
            reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
            partial_output.data_ptr<float>(),
            grouped_params,
            size_m,
            size_k,
            static_cast<int>(total_n),
            1,
            0);
      }
    }
  } else if (use_h100_flash_next_full_qkv_fixed) {
    const dim3 full_qkv_grid(
        static_cast<unsigned>(total_n / kOutputColumns),
        static_cast<unsigned>(row_ctas),
        1);
    qvq_p32_window_wgmma_m16_tma_kernel<
        TransitionBits,
        true,
        false,
        false,
        false,
        1,
        3><<<full_qkv_grid, kTmaThreads, 0, stream>>>(
            input_tma,
            trellis_tma,
            bank_tma,
            reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
            partial_output.data_ptr<float>(),
            grouped_params,
            size_m,
            size_k,
            static_cast<int>(total_n),
            1,
            0);
  } else if (use_h100_flash_next_full_qkv_ordered_fixed) {
    if constexpr (TransitionBits == 5 || TransitionBits == 7) {
      qvq_p32_window_wgmma_m16_tma_kernel<
          TransitionBits,
          true,
          true,
          false,
          false,
          1,
          4><<<grid, kTmaThreads, 0, stream>>>(
              input_tma,
              trellis_tma,
              bank_tma,
              reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
              partial_output.data_ptr<float>(),
              grouped_params,
              size_m,
              size_k,
              static_cast<int>(total_n),
              1,
              0);
    }
  } else if (use_h100_qwen_ordered_fixed) {
    const HopperFixedGateUpLaunchParams fixed_params{
        {grouped_params.bank_alt_id[0], grouped_params.bank_alt_id[1]}};
    const dim3 qwen_grid(
        static_cast<unsigned>(kFixedQwenGateUpN / kOutputColumns),
        static_cast<unsigned>(2 * row_tiles),
        kFixedQwenGateUpSplit);
    qvq_p32_window_wgmma_m16_tma_kernel<
        TransitionBits,
        true,
        true,
        true,
        TransitionBits <= kW3TransitionBits>
        <<<qwen_grid, kTmaThreads, 0, stream>>>(
            input_tma,
            trellis_tma,
            bank_tma,
            reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
            partial_output.data_ptr<float>(),
            fixed_params,
            size_m,
            kFixedQwenGateUpK,
            2 * kFixedQwenGateUpN,
            kFixedQwenGateUpSplit,
            0);
  } else if (use_h100_qwen_linear_legacy) {
    qvq_p32_window_wgmma_m16_tma_kernel<
        TransitionBits,
        true,
        true,
        false,
        TransitionBits != 5,
        1,
        true><<<grid, kTmaThreads, 0, stream>>>(
            input_tma,
            trellis_tma,
            bank_tma,
            reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
            partial_output.data_ptr<float>(),
            grouped_params,
            size_m,
            size_k,
            static_cast<int>(total_n),
            1,
            0);
  } else if (use_h100_flash_next_linear_fixed) {
    qvq_p32_window_wgmma_m16_tma_kernel<
        TransitionBits,
        true,
        true,
        false,
        true,
        1,
        2><<<grid, kTmaThreads, 0, stream>>>(
            input_tma,
            trellis_tma,
            bank_tma,
            reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
            partial_output.data_ptr<float>(),
            grouped_params,
            size_m,
            size_k,
            static_cast<int>(total_n),
            1,
            0);
  } else if (use_wide_gate_up) {
    const HopperFixedGateUpLaunchParams fixed_params{
        {grouped_params.bank_alt_id[0], grouped_params.bank_alt_id[1]}};
    using WideSharedStorage =
        P32WgmmaTmaSharedStorageFor<TransitionBits, 2>;
    auto wide_kernel = qvq_p32_window_wgmma_m16_tma_kernel<
        TransitionBits,
        true,
        false,
        true,
        true,
        2,
        false,
        1,
        decltype(input_tma),
        decltype(wide_trellis_tma),
        decltype(bank_tma),
        HopperFixedGateUpLaunchParams>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        wide_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sizeof(WideSharedStorage))));
    const dim3 wide_grid(
        static_cast<unsigned>(max_n64_blocks / 2),
        static_cast<unsigned>(segment_count * row_tiles),
        1);
    wide_kernel<<<
        wide_grid,
        kTmaThreads + kThreads,
        sizeof(WideSharedStorage),
        stream>>>(
        input_tma,
        wide_trellis_tma,
        bank_tma,
        reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
        partial_output.data_ptr<float>(),
        fixed_params,
        size_m,
        kFixedGateUpK,
        2 * kFixedGateUpN,
        1,
        0);
  } else if (use_fixed_gate_up) {
    const HopperFixedGateUpLaunchParams fixed_params{
        {grouped_params.bank_alt_id[0], grouped_params.bank_alt_id[1]}};
    qvq_p32_window_wgmma_m16_tma_kernel<
        TransitionBits,
        true,
        false,
        true,
        true>
        <<<grid, kTmaThreads, 0, stream>>>(
            input_tma,
            trellis_tma,
            bank_tma,
            reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
            partial_output.data_ptr<float>(),
            fixed_params,
            size_m,
            kFixedGateUpK,
            2 * kFixedGateUpN,
            1,
            0);
  } else if (use_prefetch_gate_up) {
    qvq_p32_window_wgmma_m16_tma_kernel<
        TransitionBits,
        true,
        false,
        false,
        true><<<grid, kTmaThreads, 0, stream>>>(
        input_tma,
        trellis_tma,
        bank_tma,
        reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
        partial_output.data_ptr<float>(),
        grouped_params,
        size_m,
        size_k,
        static_cast<int>(total_n),
        1,
        0);
  } else {
    qvq_p32_window_wgmma_m16_tma_kernel<TransitionBits, true, OrderedSplit>
        <<<grid, kTmaThreads, 0, stream>>>(
            input_tma,
            trellis_tma,
            bank_tma,
            reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
            partial_output.data_ptr<float>(),
            grouped_params,
            size_m,
            size_k,
            static_cast<int>(total_n),
            1,
            0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if constexpr (OrderedSplit && !ReturnPartials) {
    for (int segment = 0; segment < segment_count; ++segment) {
      const int output_values =
          size_m * static_cast<int>(out_features[segment]);
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
  if constexpr (ReturnPartials) {
    return partial_output;
  } else {
    return output;
  }
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

at::Tensor qvq_p32_window_wgmma_m16_tma_grouped_ordered_partials(
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
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<4, true, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    case 5:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<5, true, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    case 6:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<6, true, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    case 7:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<7, true, true>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts);
    default:
      TORCH_CHECK(
          false,
          "ordered-partial grouped QVQ P32 TMA WGMMA transition bits must be in [4, 7]");
  }
}

template <int TransitionBits>
at::Tensor qvq_p32_window_wgmma_fp8_m16_impl(
    const at::Tensor& input,
    const at::Tensor& input_scale,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t out_features,
    int64_t bank_alt_id,
    double level_scale) {
  TORCH_CHECK(input.is_cuda(), "QVQ P32 FP8 WGMMA input must be CUDA");
  c10::cuda::CUDAGuard device_guard(input.device());
  TORCH_CHECK(
      input_scale.device() == input.device() && trellis.device() == input.device() &&
          levels.device() == input.device() && bank_ids.device() == input.device(),
      "QVQ P32 FP8 WGMMA tensors must share one CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kFloat8_e4m3fn && levels.scalar_type() == at::kFloat8_e4m3fn,
      "QVQ P32 FP8 WGMMA requires float8_e4m3fn input and levels");
  TORCH_CHECK(input_scale.scalar_type() == at::kFloat,
              "QVQ P32 FP8 WGMMA input scale must be float32");
  TORCH_CHECK(trellis.scalar_type() == at::kInt,
              "QVQ P32 FP8 WGMMA trellis must be int32");
  TORCH_CHECK(bank_ids.scalar_type() == at::kByte,
              "QVQ P32 FP8 WGMMA bank ids must be uint8");
  TORCH_CHECK(
      input.is_contiguous() && input_scale.is_contiguous() && trellis.is_contiguous() &&
          levels.is_contiguous() && bank_ids.is_contiguous(),
      "QVQ P32 FP8 WGMMA tensors must be contiguous");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= kRows &&
          input.size(0) % kRows == 0 && input.size(0) <= 4096,
      "QVQ P32 FP8 WGMMA requires M in [16, 4096] and divisible by 16");
  TORCH_CHECK(input_scale.numel() == input.size(0),
              "QVQ P32 FP8 WGMMA requires one input scale per row");
  TORCH_CHECK(out_features > 0 && out_features % kOutputColumns == 0,
              "QVQ P32 FP8 WGMMA output features must be a positive multiple of 64");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) % 32 == 0,
              "QVQ P32 FP8 WGMMA input features must be a positive multiple of 32");
  TORCH_CHECK(bank_alt_id >= 1 && bank_alt_id <= 3,
              "QVQ P32 FP8 WGMMA alternate bank id must be in [1, 3]");
  TORCH_CHECK(std::isfinite(level_scale) && level_scale > 0.0,
              "QVQ P32 FP8 WGMMA level scale must be finite and positive");

  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(properties.major == 9 && properties.minor == 0,
              "QVQ P32 FP8 WGMMA requires an SM90 H100/H200 device");

  const int size_k = static_cast<int>(input.size(1));
  const int size_n = static_cast<int>(out_features);
  const int64_t expected_tiles =
      static_cast<int64_t>(size_k / kP32TileRows) * (size_n / kP32TileColumns);
  constexpr int kWordsPerP32Tile = 4 * TransitionBits;
  TORCH_CHECK(trellis.numel() == expected_tiles * kWordsPerP32Tile,
              "QVQ P32 FP8 WGMMA trellis size mismatch");
  TORCH_CHECK(bank_ids.numel() == expected_tiles,
              "QVQ P32 FP8 WGMMA bank-id size mismatch");
  TORCH_CHECK(levels.numel() == 256,
              "QVQ P32 FP8 WGMMA requires 256 quantized PGC16 levels");

  const int size_m = static_cast<int>(input.size(0));
  auto output = at::empty({size_m, size_n}, input.options().dtype(at::kFloat));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  // Two independent M16 CTAs are measurably faster than reuse-2 at M32 on
  // H200 because they provide a second wave across wide-N projections.  Once
  // M reaches 64, reuse amortizes P32 decoding and wins; reuse-8 crosses over
  // at M256 on H200.  Retain reuse-2 for larger shapes divisible by 32 but
  // not 64.
  const int row_tiles_per_cta = size_m >= 16 * kRows && size_m % (8 * kRows) == 0
      ? 8
      : size_m >= 4 * kRows && size_m % (4 * kRows) == 0
      ? 4
      : size_m > 2 * kRows && size_m % (2 * kRows) == 0 ? 2 : 1;
  const dim3 grid(
      static_cast<unsigned>(size_n / kOutputColumns),
      static_cast<unsigned>(size_m / (row_tiles_per_cta * kRows)));
#define QVQ_LAUNCH_FP8_ROW_REUSE(ROW_TILES)                                      \
  qvq_p32_window_wgmma_fp8_m16_kernel<TransitionBits, ROW_TILES>                 \
      <<<grid, kThreads, 0, stream>>>(                                            \
          reinterpret_cast<const Fp8Element*>(input.data_ptr()),                 \
          input_scale.data_ptr<float>(),                                         \
          reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),        \
          bank_ids.data_ptr<uint8_t>(),                                          \
          reinterpret_cast<const Fp8Element*>(levels.data_ptr()),                \
          static_cast<float>(level_scale),                                       \
          output.data_ptr<float>(),                                              \
          size_k,                                                                \
          size_n,                                                                \
          static_cast<int>(bank_alt_id))
  if (row_tiles_per_cta == 8) {
    QVQ_LAUNCH_FP8_ROW_REUSE(8);
  } else if (row_tiles_per_cta == 4) {
    QVQ_LAUNCH_FP8_ROW_REUSE(4);
  } else if (row_tiles_per_cta == 2) {
    QVQ_LAUNCH_FP8_ROW_REUSE(2);
  } else {
    QVQ_LAUNCH_FP8_ROW_REUSE(1);
  }
#undef QVQ_LAUNCH_FP8_ROW_REUSE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qvq_p32_window_wgmma_fp8_m16(
    const at::Tensor& input,
    const at::Tensor& input_scale,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    int64_t out_features,
    int64_t bank_alt_id,
    double level_scale) {
  switch (transition_bits) {
    case 4:
      return qvq_p32_window_wgmma_fp8_m16_impl<4>(
          input, input_scale, trellis, levels, bank_ids, out_features, bank_alt_id, level_scale);
    case 5:
      return qvq_p32_window_wgmma_fp8_m16_impl<5>(
          input, input_scale, trellis, levels, bank_ids, out_features, bank_alt_id, level_scale);
    case 6:
      return qvq_p32_window_wgmma_fp8_m16_impl<6>(
          input, input_scale, trellis, levels, bank_ids, out_features, bank_alt_id, level_scale);
    case 7:
      return qvq_p32_window_wgmma_fp8_m16_impl<7>(
          input, input_scale, trellis, levels, bank_ids, out_features, bank_alt_id, level_scale);
    default:
      TORCH_CHECK(false, "QVQ P32 FP8 WGMMA transition bits must be in [4, 7]");
  }
}

template <int TransitionBits>
at::Tensor qvq_p32_window_wgmma_tuned_impl(
    const at::Tensor& input, const at::Tensor& trellis,
    const at::Tensor& levels, const at::Tensor& bank_ids,
    int64_t out_features, int64_t bank_alt_id, int64_t block_m, int64_t block_n) {
  // Each branch already exists in grouped reuse2/4/8. No new device variants.
  const std::vector<int64_t> widths{out_features}, banks{bank_alt_id}, splits{1};
#define QVQ_TUNED_ROWS(BM, REUSE) \
  case BM: return qvq_p32_window_wgmma_m16_tma_grouped_impl< \
      TransitionBits, false, false, REUSE>( \
          input, trellis, levels, bank_ids, widths, banks, splits, block_n)
  switch (block_m) {
    QVQ_TUNED_ROWS(32, 2);
    QVQ_TUNED_ROWS(64, 4);
    QVQ_TUNED_ROWS(128, 8);
    default: TORCH_CHECK(false, "explicit Hopper BM must be 32, 64 or 128");
  }
#undef QVQ_TUNED_ROWS
}

at::Tensor qvq_p32_window_wgmma_tuned(
    const at::Tensor& input, const at::Tensor& trellis,
    const at::Tensor& levels, const at::Tensor& bank_ids,
    int64_t transition_bits, int64_t out_features, int64_t bank_alt_id,
    int64_t block_m, int64_t block_n) {
  TORCH_CHECK(block_n == 64 || block_n == 128,
              "explicit Hopper BN must be 64 or 128");
  switch (transition_bits) {
#define QVQ_TUNED_RATE(RATE) \
    case RATE: return qvq_p32_window_wgmma_tuned_impl<RATE>( \
        input, trellis, levels, bank_ids, out_features, bank_alt_id, block_m, block_n)
    QVQ_TUNED_RATE(4);
    QVQ_TUNED_RATE(5);
    QVQ_TUNED_RATE(6);
    QVQ_TUNED_RATE(7);
#undef QVQ_TUNED_RATE
    default: TORCH_CHECK(false, "explicit Hopper transition bits must be in [4, 7]");
  }
}

at::Tensor qvq_p32_window_wgmma_m32_tma_grouped_reuse2(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts,
    int64_t block_n) {
  switch (transition_bits) {
    case 4:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<4, false, false, 2>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 5:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<5, false, false, 2>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 6:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<6, false, false, 2>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 7:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<7, false, false, 2>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    default:
      TORCH_CHECK(
          false,
          "M32-reuse grouped QVQ P32 transition bits must be in [4, 7]");
  }
}

at::Tensor qvq_p32_window_wgmma_m32_tma_grouped_ordered_reuse2(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts,
    int64_t block_n) {
  switch (transition_bits) {
    case 4:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<4, true, false, 2>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 5:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<5, true, false, 2>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 6:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<6, true, false, 2>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 7:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<7, true, false, 2>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    default:
      TORCH_CHECK(
          false,
          "ordered M32-reuse grouped QVQ P32 transition bits must be in [4, 7]");
  }
}

at::Tensor qvq_p32_window_wgmma_m64_tma_grouped_reuse4(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts,
    int64_t block_n) {
  switch (transition_bits) {
    case 4:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<4, false, false, 4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 5:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<5, false, false, 4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 6:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<6, false, false, 4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 7:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<7, false, false, 4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    default:
      TORCH_CHECK(
          false,
          "M64-reuse grouped QVQ P32 transition bits must be in [4, 7]");
  }
}

at::Tensor qvq_p32_window_wgmma_m128_tma_grouped_reuse8(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts,
    int64_t block_n) {
  switch (transition_bits) {
    case 4:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<4, false, false, 8>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 5:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<5, false, false, 8>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 6:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<6, false, false, 8>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 7:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<7, false, false, 8>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    default:
      TORCH_CHECK(
          false,
          "M128-reuse grouped QVQ P32 transition bits must be in [4, 7]");
  }
}

at::Tensor qvq_p32_window_wgmma_m176_tma_grouped_reuse11(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts,
    int64_t block_n) {
  switch (transition_bits) {
    case 4:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<4, false, false, 11>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 5:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<5, false, false, 11>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 6:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<6, false, false, 11>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 7:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<7, false, false, 11>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    default:
      TORCH_CHECK(
          false,
          "M176-reuse grouped QVQ P32 transition bits must be in [4, 7]");
  }
}

at::Tensor qvq_p32_window_wgmma_m64_tma_grouped_ordered_reuse4(
    const at::Tensor& input,
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef split_counts,
    int64_t block_n) {
  switch (transition_bits) {
    case 4:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<4, true, false, 4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 5:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<5, true, false, 4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 6:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<6, true, false, 4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    case 7:
      return qvq_p32_window_wgmma_m16_tma_grouped_impl<7, true, false, 4>(
          input, trellis, levels, bank_ids, out_features, bank_alt_ids, split_counts, block_n);
    default:
      TORCH_CHECK(
          false,
          "ordered M64-reuse grouped QVQ P32 transition bits must be in [4, 7]");
  }
}

template <int TransitionBits>
at::Tensor qvq_p32_window_decode_grouped_fp16_impl(
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t in_features,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    bool transpose_output = false) {
  TORCH_CHECK(trellis.is_cuda(), "grouped P32 FP16 decoder requires CUDA tensors");
  c10::cuda::CUDAGuard device_guard(trellis.device());
  TORCH_CHECK(
      levels.device() == trellis.device() && bank_ids.device() == trellis.device(),
      "grouped P32 FP16 decoder tensors must share one CUDA device");
  TORCH_CHECK(
      trellis.scalar_type() == at::kInt && levels.scalar_type() == at::kHalf &&
          bank_ids.scalar_type() == at::kByte,
      "grouped P32 FP16 decoder requires int32 trellis, FP16 levels, and uint8 bank IDs");
  TORCH_CHECK(
      trellis.is_contiguous() && levels.is_contiguous() && bank_ids.is_contiguous(),
      "grouped P32 FP16 decoder tensors must be contiguous");
  TORCH_CHECK(
      in_features > 0 && in_features % kKPerStage == 0,
      "grouped P32 FP16 decoder K must be a positive multiple of 256");
  const int64_t segment_count = static_cast<int64_t>(out_features.size());
  TORCH_CHECK(
      segment_count >= 1 && segment_count <= kMaxGroupedP32Segments &&
          static_cast<int64_t>(bank_alt_ids.size()) == segment_count,
      "grouped P32 FP16 decoder requires one to three matching segments");
  TORCH_CHECK(
      levels.numel() == 256,
      "grouped P32 FP16 decoder requires 256 PGC16 levels");

  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, trellis.get_device()));
  TORCH_CHECK(
      properties.major == 9 && properties.minor == 0,
      "grouped P32 FP16 decoder requires an SM90 H100/H200 device");

  HopperGroupedP32DecodeParams grouped_params{};
  grouped_params.segment_count = static_cast<int>(segment_count);
  int64_t total_n = 0;
  for (int segment = 0; segment < segment_count; ++segment) {
    const int64_t width = out_features[segment];
    const int64_t alt_id = bank_alt_ids[segment];
    TORCH_CHECK(
        width > 0 && width % kOutputColumns == 0,
        "grouped P32 FP16 decoder widths must be positive multiples of 64");
    TORCH_CHECK(
        alt_id >= 0 && alt_id <= 3,
        "grouped P32 FP16 decoder alternative bank IDs must be in [0, 3]");
    total_n += width;
    TORCH_CHECK(
        total_n <= std::numeric_limits<int>::max(),
        "grouped P32 FP16 decoder total N exceeds int32 range");
    grouped_params.n64_end[segment] =
        static_cast<int>(total_n / kOutputColumns);
    grouped_params.bank_alt_id[segment] = static_cast<int>(alt_id);
  }

  constexpr int kWordsPerP32Tile = 4 * TransitionBits;
  const int64_t k_tiles = in_features / kP32TileRows;
  const int64_t n_tiles = total_n / kP32TileColumns;
  TORCH_CHECK(
      trellis.numel() == k_tiles * n_tiles * kWordsPerP32Tile,
      "grouped P32 FP16 decoder trellis size mismatch");
  TORCH_CHECK(
      bank_ids.numel() == k_tiles * n_tiles,
      "grouped P32 FP16 decoder bank-ID size mismatch");

  auto output = transpose_output
      ? at::empty({total_n, in_features}, levels.options().dtype(at::kHalf))
      : at::empty({in_features, total_n}, levels.options().dtype(at::kHalf));
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(trellis.get_device());
  const dim3 grid(
      static_cast<unsigned>(total_n / kOutputColumns),
      static_cast<unsigned>(in_features / kKPerStage),
      1);
  qvq_p32_window_decode_fp16_kernel<TransitionBits>
      <<<grid, kThreads, 0, stream>>>(
          reinterpret_cast<const uint32_t*>(trellis.data_ptr<int32_t>()),
          bank_ids.data_ptr<uint8_t>(),
          reinterpret_cast<const Element*>(levels.data_ptr<at::Half>()),
          reinterpret_cast<Element*>(output.data_ptr<at::Half>()),
          grouped_params,
          static_cast<int>(in_features),
          static_cast<int>(total_n),
          transpose_output);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qvq_p32_window_decode_grouped_fp16(
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    int64_t transition_bits,
    int64_t in_features,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids) {
  switch (transition_bits) {
    case 4:
      return qvq_p32_window_decode_grouped_fp16_impl<4>(
          trellis, levels, bank_ids, in_features, out_features, bank_alt_ids);
    case 5:
      return qvq_p32_window_decode_grouped_fp16_impl<5>(
          trellis, levels, bank_ids, in_features, out_features, bank_alt_ids);
    case 6:
      return qvq_p32_window_decode_grouped_fp16_impl<6>(
          trellis, levels, bank_ids, in_features, out_features, bank_alt_ids);
    case 7:
      return qvq_p32_window_decode_grouped_fp16_impl<7>(
          trellis, levels, bank_ids, in_features, out_features, bank_alt_ids);
    default:
      TORCH_CHECK(
          false,
          "grouped P32 FP16 decoder transition bits must be in [4, 7]");
  }
}

template <int TransitionBits, bool OutputFp8 = false, bool HalfFold = false>
at::Tensor qvq_p32_window_prepare_grouped_fp16_impl(
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    const at::Tensor& input_scale,
    at::TensorList output_scales,
    int64_t in_features,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef output_hadamards,
    const at::Tensor* weight_scale = nullptr) {
  static_assert(!HalfFold || OutputFp8);
  TORCH_CHECK(
      in_features >= 256 && in_features <= 16384 &&
          (in_features & (in_features - 1)) == 0,
      "grouped P32 folded-FP16 preparation requires power-of-two K in [256, 16384]");
  const int64_t segment_count = static_cast<int64_t>(out_features.size());
  TORCH_CHECK(
      segment_count >= 1 && segment_count <= kMaxGroupedP32Segments &&
          static_cast<int64_t>(bank_alt_ids.size()) == segment_count &&
          static_cast<int64_t>(output_scales.size()) == segment_count &&
          static_cast<int64_t>(output_hadamards.size()) == segment_count,
      "grouped P32 folded-FP16 preparation requires one to three matching segments");
  TORCH_CHECK(
      input_scale.is_cuda() && input_scale.device() == trellis.device() &&
          input_scale.scalar_type() == at::kFloat && input_scale.is_contiguous() &&
          input_scale.numel() == in_features,
      "grouped P32 folded-FP16 preparation requires contiguous FP32 input scale [K]");
  if constexpr (OutputFp8) {
    TORCH_CHECK(
        weight_scale != nullptr && weight_scale->is_cuda() &&
            weight_scale->device() == trellis.device() &&
            weight_scale->scalar_type() == at::kFloat &&
            weight_scale->is_contiguous() && weight_scale->numel() == 1,
        "grouped P32 folded-FP8 preparation requires one contiguous CUDA FP32 weight scale");
  }

  HopperGroupedP32FoldParams fold_params{};
  fold_params.segment_count = static_cast<int>(segment_count);
  int64_t total_n = 0;
  int64_t max_hadamard_width = 0;
  for (int segment = 0; segment < segment_count; ++segment) {
    const int64_t width = out_features[segment];
    const int64_t output_hadamard = output_hadamards[segment];
    const at::Tensor& output_scale = output_scales[segment];
    TORCH_CHECK(
        output_scale.is_cuda() && output_scale.device() == trellis.device() &&
            output_scale.scalar_type() == at::kFloat &&
            output_scale.is_contiguous() && output_scale.numel() == width,
        "grouped P32 folded-FP16 preparation requires contiguous FP32 output scales [N]");
    TORCH_CHECK(
        output_hadamard == 0 || output_hadamard == 1,
        "grouped P32 folded-FP16 output-H flags must be zero or one");
    if (output_hadamard) {
      TORCH_CHECK(
          width >= 2 && width <= 16384 && (width & (width - 1)) == 0,
          "grouped P32 folded-FP16 output Hadamard requires power-of-two child N");
      max_hadamard_width = std::max(max_hadamard_width, width);
    }
    fold_params.n_start[segment] = static_cast<int>(total_n);
    fold_params.width[segment] = static_cast<int>(width);
    fold_params.output_hadamard[segment] = static_cast<int>(output_hadamard);
    fold_params.output_scale[segment] = output_scale.data_ptr<float>();
    total_n += width;
    TORCH_CHECK(
        total_n <= std::numeric_limits<int>::max(),
        "grouped P32 folded-FP16 preparation total N exceeds int32 range");
  }

  auto decoded_transposed = qvq_p32_window_decode_grouped_fp16_impl<TransitionBits>(
      trellis,
      levels,
      bank_ids,
      in_features,
      out_features,
      bank_alt_ids,
      true);
  auto transformed_transposed = at::empty(
      {total_n, in_features},
      levels.options().dtype(HalfFold ? at::kHalf : at::kFloat));
  auto transformed = at::empty(
      {in_features, total_n},
      levels.options().dtype(HalfFold ? at::kHalf : at::kFloat));
  auto output = OutputFp8
      ? at::empty(
            {total_n, in_features},
            levels.options().dtype(at::kFloat8_e4m3fn))
      : at::empty(
            {in_features, total_n}, levels.options().dtype(at::kHalf));

  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(trellis.get_device());
  const size_t k_shared_bytes = static_cast<size_t>(
      in_features + in_features / (HalfFold ? 16 : 32)) *
      (HalfFold ? sizeof(__half) : sizeof(float));
  if constexpr (HalfFold) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        qvq_p32_prefill_fold_k_axis_fp16_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(k_shared_bytes)));
    qvq_p32_prefill_fold_k_axis_fp16_kernel<<<
        static_cast<unsigned>(total_n),
        kPrefillFoldThreads,
        k_shared_bytes,
        stream>>>(
            reinterpret_cast<const __half*>(decoded_transposed.const_data_ptr()),
            reinterpret_cast<__half*>(transformed_transposed.mutable_data_ptr()),
            input_scale.data_ptr<float>(),
            static_cast<int>(in_features));
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        qvq_p32_prefill_fold_k_axis_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(k_shared_bytes)));
    qvq_p32_prefill_fold_k_axis_kernel<<<
        static_cast<unsigned>(total_n),
        kPrefillFoldThreads,
        k_shared_bytes,
        stream>>>(
            reinterpret_cast<const __half*>(decoded_transposed.const_data_ptr()),
            transformed_transposed.data_ptr<float>(),
            input_scale.data_ptr<float>(),
            static_cast<int>(in_features));
  }

  const dim3 transpose_block(kPrefillTransposeTile, kPrefillTransposeRows, 1);
  const dim3 transpose_grid(
      static_cast<unsigned>(
          (in_features + kPrefillTransposeTile - 1) / kPrefillTransposeTile),
      static_cast<unsigned>(
          (total_n + kPrefillTransposeTile - 1) / kPrefillTransposeTile),
      1);
  if constexpr (HalfFold) {
    qvq_p32_prefill_transpose_fp16_kernel<<<
        transpose_grid, transpose_block, 0, stream>>>(
            reinterpret_cast<const __half*>(
                transformed_transposed.const_data_ptr()),
            reinterpret_cast<__half*>(transformed.mutable_data_ptr()),
            static_cast<int>(total_n),
            static_cast<int>(in_features));
  } else {
    qvq_p32_prefill_transpose_fp32_kernel<<<
        transpose_grid, transpose_block, 0, stream>>>(
            transformed_transposed.data_ptr<float>(),
            transformed.data_ptr<float>(),
            static_cast<int>(total_n),
            static_cast<int>(in_features));
  }

  const size_t n_shared_bytes = static_cast<size_t>(
      max_hadamard_width + max_hadamard_width / (HalfFold ? 16 : 32)) *
      (HalfFold ? sizeof(__half) : sizeof(float));
  if constexpr (HalfFold) {
    if (n_shared_bytes > 0) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qvq_p32_prefill_fold_n_axis_fp16_to_fp8_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(n_shared_bytes)));
    }
    qvq_p32_prefill_fold_n_axis_fp16_to_fp8_kernel<<<
        static_cast<unsigned>(in_features * segment_count),
        kPrefillFoldThreads,
        n_shared_bytes,
        stream>>>(
            reinterpret_cast<const __half*>(transformed.const_data_ptr()),
            output.data_ptr<c10::Float8_e4m3fn>(),
            weight_scale->data_ptr<float>(),
            fold_params,
            static_cast<int>(in_features),
            static_cast<int>(total_n));
  } else {
    if (n_shared_bytes > 0) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qvq_p32_prefill_fold_n_axis_kernel<OutputFp8>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(n_shared_bytes)));
    }
    qvq_p32_prefill_fold_n_axis_kernel<OutputFp8><<<
        static_cast<unsigned>(in_features * segment_count),
        kPrefillFoldThreads,
        n_shared_bytes,
        stream>>>(
            transformed.data_ptr<float>(),
            reinterpret_cast<std::conditional_t<
                OutputFp8, c10::Float8_e4m3fn, __half>*>(
                output.mutable_data_ptr()),
            OutputFp8 ? weight_scale->data_ptr<float>() : nullptr,
            fold_params,
            static_cast<int>(in_features),
            static_cast<int>(total_n));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qvq_p32_window_prepare_grouped_fp16(
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    const at::Tensor& input_scale,
    at::TensorList output_scales,
    int64_t transition_bits,
    int64_t in_features,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef output_hadamards) {
  switch (transition_bits) {
    case 4:
      return qvq_p32_window_prepare_grouped_fp16_impl<4>(
          trellis, levels, bank_ids, input_scale, output_scales, in_features,
          out_features, bank_alt_ids, output_hadamards);
    case 5:
      return qvq_p32_window_prepare_grouped_fp16_impl<5>(
          trellis, levels, bank_ids, input_scale, output_scales, in_features,
          out_features, bank_alt_ids, output_hadamards);
    case 6:
      return qvq_p32_window_prepare_grouped_fp16_impl<6>(
          trellis, levels, bank_ids, input_scale, output_scales, in_features,
          out_features, bank_alt_ids, output_hadamards);
    case 7:
      return qvq_p32_window_prepare_grouped_fp16_impl<7>(
          trellis, levels, bank_ids, input_scale, output_scales, in_features,
          out_features, bank_alt_ids, output_hadamards);
    default:
      TORCH_CHECK(
          false,
          "grouped P32 folded-FP16 preparation transition bits must be in [4, 7]");
  }
}

at::Tensor qvq_p32_window_prepare_grouped_fp8(
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    const at::Tensor& input_scale,
    at::TensorList output_scales,
    const at::Tensor& weight_scale,
    int64_t transition_bits,
    int64_t in_features,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef output_hadamards) {
#define QVQ_PREPARE_GROUPED_FP8(TRANSITION_BITS)                               \
  return qvq_p32_window_prepare_grouped_fp16_impl<TRANSITION_BITS, true>(      \
      trellis, levels, bank_ids, input_scale, output_scales, in_features,     \
      out_features, bank_alt_ids, output_hadamards, &weight_scale)
  switch (transition_bits) {
    case 4:
      QVQ_PREPARE_GROUPED_FP8(4);
    case 5:
      QVQ_PREPARE_GROUPED_FP8(5);
    case 6:
      QVQ_PREPARE_GROUPED_FP8(6);
    case 7:
      QVQ_PREPARE_GROUPED_FP8(7);
    default:
      TORCH_CHECK(
          false,
          "grouped P32 folded-FP8 preparation transition bits must be in [4, 7]");
  }
#undef QVQ_PREPARE_GROUPED_FP8
}

at::Tensor qvq_p32_window_prepare_grouped_fp8_half_fold(
    const at::Tensor& trellis,
    const at::Tensor& levels,
    const at::Tensor& bank_ids,
    const at::Tensor& input_scale,
    at::TensorList output_scales,
    const at::Tensor& weight_scale,
    int64_t transition_bits,
    int64_t in_features,
    at::IntArrayRef out_features,
    at::IntArrayRef bank_alt_ids,
    at::IntArrayRef output_hadamards) {
#define QVQ_PREPARE_GROUPED_FP8_HALF(TRANSITION_BITS)                          \
  return qvq_p32_window_prepare_grouped_fp16_impl<                             \
      TRANSITION_BITS, true, true>(                                            \
      trellis, levels, bank_ids, input_scale, output_scales, in_features,     \
      out_features, bank_alt_ids, output_hadamards, &weight_scale)
  switch (transition_bits) {
    case 4:
      QVQ_PREPARE_GROUPED_FP8_HALF(4);
    case 5:
      QVQ_PREPARE_GROUPED_FP8_HALF(5);
    case 6:
      QVQ_PREPARE_GROUPED_FP8_HALF(6);
    case 7:
      QVQ_PREPARE_GROUPED_FP8_HALF(7);
    default:
      TORCH_CHECK(
          false,
          "grouped P32 half-folded FP8 preparation transition bits must be in [4, 7]");
  }
#undef QVQ_PREPARE_GROUPED_FP8_HALF
}

#endif  // QVQ_WGMMA_DEVICE_ONLY

}  // namespace

#ifndef QVQ_WGMMA_DEVICE_ONLY
TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq_wgmma, m) {
  m.def("fp16_to_fp8_e5m2_clamped(Tensor input) -> Tensor");
  m.def("p32_window_w3_m16(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("p32_window_w3_m16_tma(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("p32_window_m16_tma(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("p32_window_m16_tma_ordered_split(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("p32_window_m16_tma_ordered_partials(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
  m.def("p32_window_m16_tma_grouped(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts) -> Tensor");
  m.def("p32_window_m16_tma_grouped_ordered_split(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts) -> Tensor");
  m.def("p32_window_m16_tma_grouped_ordered_partials(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts) -> Tensor");
  m.def("p32_window_fp8_m16(Tensor input, Tensor input_scale, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int out_features, int bank_alt_id, float level_scale) -> Tensor");
  m.def("p32_window_tuned(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int out_features, int bank_alt_id, int block_m, int block_n) -> Tensor");
  m.def("p32_window_m32_tma_grouped_reuse2(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts, int block_n=0) -> Tensor");
  m.def("p32_window_m32_tma_grouped_ordered_reuse2(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts, int block_n=0) -> Tensor");
  m.def("p32_window_m64_tma_grouped_reuse4(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts, int block_n=0) -> Tensor");
  m.def("p32_window_m64_tma_grouped_ordered_reuse4(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts, int block_n=0) -> Tensor");
  m.def("p32_window_m128_tma_grouped_reuse8(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts, int block_n=0) -> Tensor");
  m.def("p32_window_m176_tma_grouped_reuse11(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int[] out_features, int[] bank_alt_ids, int[] split_counts, int block_n=0) -> Tensor");
  m.def("p32_window_decode_grouped_fp16(Tensor trellis, Tensor levels, Tensor bank_ids, int transition_bits, int in_features, int[] out_features, int[] bank_alt_ids) -> Tensor");
  m.def("p32_window_prepare_grouped_fp16(Tensor trellis, Tensor levels, Tensor bank_ids, Tensor input_scale, Tensor[] output_scales, int transition_bits, int in_features, int[] out_features, int[] bank_alt_ids, int[] output_hadamards) -> Tensor");
  m.def("p32_window_prepare_grouped_fp8(Tensor trellis, Tensor levels, Tensor bank_ids, Tensor input_scale, Tensor[] output_scales, Tensor weight_scale, int transition_bits, int in_features, int[] out_features, int[] bank_alt_ids, int[] output_hadamards) -> Tensor");
  m.def("p32_window_prepare_grouped_fp8_half_fold(Tensor trellis, Tensor levels, Tensor bank_ids, Tensor input_scale, Tensor[] output_scales, Tensor weight_scale, int transition_bits, int in_features, int[] out_features, int[] bank_alt_ids, int[] output_hadamards) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq_wgmma, CUDA, m) {
  m.impl("fp16_to_fp8_e5m2_clamped", qvq_fp16_to_fp8_e5m2_clamped);
  m.impl("p32_window_w3_m16", qvq_p32_window_wgmma_w3_m16);
  m.impl("p32_window_w3_m16_tma", qvq_p32_window_wgmma_w3_m16_tma);
  m.impl("p32_window_m16_tma", qvq_p32_window_wgmma_m16_tma);
  m.impl("p32_window_m16_tma_ordered_split", qvq_p32_window_wgmma_m16_tma_ordered_split);
  m.impl("p32_window_m16_tma_ordered_partials", qvq_p32_window_wgmma_m16_tma_ordered_partials);
  m.impl("p32_window_m16_tma_grouped", qvq_p32_window_wgmma_m16_tma_grouped);
  m.impl("p32_window_m16_tma_grouped_ordered_split", qvq_p32_window_wgmma_m16_tma_grouped_ordered_split);
  m.impl("p32_window_m16_tma_grouped_ordered_partials", qvq_p32_window_wgmma_m16_tma_grouped_ordered_partials);
  m.impl("p32_window_fp8_m16", qvq_p32_window_wgmma_fp8_m16);
  m.impl("p32_window_tuned", qvq_p32_window_wgmma_tuned);
  m.impl("p32_window_m32_tma_grouped_reuse2", qvq_p32_window_wgmma_m32_tma_grouped_reuse2);
  m.impl("p32_window_m32_tma_grouped_ordered_reuse2", qvq_p32_window_wgmma_m32_tma_grouped_ordered_reuse2);
  m.impl("p32_window_m64_tma_grouped_reuse4", qvq_p32_window_wgmma_m64_tma_grouped_reuse4);
  m.impl("p32_window_m64_tma_grouped_ordered_reuse4", qvq_p32_window_wgmma_m64_tma_grouped_ordered_reuse4);
  m.impl("p32_window_m128_tma_grouped_reuse8", qvq_p32_window_wgmma_m128_tma_grouped_reuse8);
  m.impl("p32_window_m176_tma_grouped_reuse11", qvq_p32_window_wgmma_m176_tma_grouped_reuse11);
  m.impl("p32_window_decode_grouped_fp16", qvq_p32_window_decode_grouped_fp16);
  m.impl("p32_window_prepare_grouped_fp16", qvq_p32_window_prepare_grouped_fp16);
  m.impl("p32_window_prepare_grouped_fp8", qvq_p32_window_prepare_grouped_fp8);
  m.impl("p32_window_prepare_grouped_fp8_half_fold", qvq_p32_window_prepare_grouped_fp8_half_fold);
}
#endif  // QVQ_WGMMA_DEVICE_ONLY
