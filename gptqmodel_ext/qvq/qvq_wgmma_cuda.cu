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
constexpr int kRows = 16;
constexpr int kOutputColumns = 64;
constexpr int kK32Rows = 32;
constexpr int kN8Columns = 8;
constexpr int kW3TransitionBits = 6;
constexpr int kWordsPerTile = 4 * kW3TransitionBits;
constexpr int kK32TilesPerStage = 8;
constexpr int kKPerStage = kK32TilesPerStage * kK32Rows;
constexpr int kN8TilesPerBlock = kOutputColumns / kN8Columns;
constexpr uint32_t kPgc16Multiplier = 40503u;
constexpr uint32_t kPgc16Increment = 17011u;

static_assert(cute::size(WgmmaTiledMma{}) == kThreads);

__device__ __forceinline__ uint32_t qvq_wgmma_pgc16_mix(uint32_t state) {
  uint32_t mixed = state ^ (state >> 8);
  mixed = (mixed * kPgc16Multiplier + kPgc16Increment) & 0xffffu;
  return mixed ^ (mixed >> 7);
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

__global__ void qvq_wgmma_reduce_split_kernel(
    const float* __restrict__ partial_output,
    at::Half* __restrict__ output,
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
  output[index] = static_cast<at::Half>(value);
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

  auto output = at::empty({kRows, size_n}, input.options());
  auto partial_output = at::empty(
      {split_count, kRows, size_n},
      input.options().dtype(at::kFloat));
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

  constexpr int kReductionThreads = 256;
  const int output_values = kRows * size_n;
  const int reduction_blocks = (output_values + kReductionThreads - 1) / kReductionThreads;
  qvq_wgmma_reduce_split_kernel<<<reduction_blocks, kReductionThreads, 0, stream>>>(
      partial_output.data_ptr<float>(),
      output.data_ptr<at::Half>(),
      output_values,
      static_cast<int>(split_count));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq_wgmma, m) {
  m.def("w3_m16(Tensor input, Tensor trellis, Tensor levels, Tensor bank_ids, int out_features, int bank_alt_id=3, int split_count=1) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq_wgmma, CUDA, m) {
  m.impl("w3_m16", qvq_wgmma_w3_m16);
}
