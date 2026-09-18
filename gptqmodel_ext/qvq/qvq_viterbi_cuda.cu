// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// QVQ Viterbi: shared-memory G-only reformulation.
//
// The per-step recurrence is equivalent to keeping only the per-suffix
// minimum vector G (fits in shared memory) instead of materializing the
// 65536-state cost array in global memory every step:
//
//   G_0[x]       = min_h  e_0((h << (16-shift)) | x)          (h == overlap if constrained)
//   G_i[x]       = min_h [ G_{i-1}[s >> shift] + e_i(s) ]     for s = (h << (16-shift)) | x
//   f_last(s)    = G_{steps-2}[s >> shift] + e_{steps-1}(s)
//   final state  = argmin_{s} f_last(s)   (lowest state index on ties; x == overlap if constrained)
//
// Every value, arithmetic operation, and tie-break is identical to the
// reference kernel: G holds the same fp32 minima, the emission expression is
// unchanged (precomputed codebook norms use the exact same IEEE expression),
// and lowest-index tie-breaking is preserved by lower_pair(). Results are
// therefore bit-identical, with zero per-state global-memory traffic.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math_constants.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <type_traits>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kStateCount = 1 << 16;
constexpr int kThreads = 1024;
constexpr size_t kMaxNormCacheEntries = 32;
namespace cg = cooperative_groups;

__device__ __forceinline__ bool lower_pair(float candidate, int candidate_index, float current, int current_index);

__device__ __forceinline__ void block_argmin(float& value, int& index) {
  constexpr unsigned mask = 0xffffffffu;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  __shared__ float warp_values[32];
  __shared__ int warp_indices[32];
  #pragma unroll
  for (int offset = 16; offset != 0; offset >>= 1) {
    const float candidate_value = __shfl_down_sync(mask, value, offset);
    const int candidate_index = __shfl_down_sync(mask, index, offset);
    if (lane + offset < 32 && lower_pair(candidate_value, candidate_index, value, index)) {
      value = candidate_value;
      index = candidate_index;
    }
  }
  if (lane == 0) {
    warp_values[warp] = value;
    warp_indices[warp] = index;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < (kThreads / 32) ? warp_values[lane] : CUDART_INF_F;
    index = lane < (kThreads / 32) ? warp_indices[lane] : INT_MAX;
    #pragma unroll
    for (int offset = 16; offset != 0; offset >>= 1) {
      const float candidate_value = __shfl_down_sync(mask, value, offset);
      const int candidate_index = __shfl_down_sync(mask, index, offset);
      if (lane + offset < 32 && lower_pair(candidate_value, candidate_index, value, index)) {
        value = candidate_value;
        index = candidate_index;
      }
    }
  }
}

template <int VectorSize, typename CodebookScalar>
__device__ __forceinline__ float emission(
    const float* __restrict__ target,
    const CodebookScalar* __restrict__ codebook,
    const float* __restrict__ codebook_norm,
    int state,
    float step_weight,
    float target_norm) {
  float dot = 0.0f;
  float norm;
  if constexpr (VectorSize == 4 && std::is_same_v<CodebookScalar, float>) {
    const float4 target4 = *reinterpret_cast<const float4*>(target);
    const float4 code4 = reinterpret_cast<const float4*>(codebook)[state];
    dot = __fmaf_rn(target4.x, code4.x, dot);
    dot = __fmaf_rn(target4.y, code4.y, dot);
    dot = __fmaf_rn(target4.z, code4.z, dot);
    dot = __fmaf_rn(target4.w, code4.w, dot);
    norm = codebook_norm[state];
  } else if constexpr (VectorSize == 4) {
    const float4 target4 = *reinterpret_cast<const float4*>(target);
    const half2 code01 = reinterpret_cast<const half2*>(codebook)[2 * state];
    const half2 code23 = reinterpret_cast<const half2*>(codebook)[2 * state + 1];
    const float2 code01f = __half22float2(code01);
    const float2 code23f = __half22float2(code23);
    dot = __fmaf_rn(target4.x, code01f.x, dot);
    dot = __fmaf_rn(target4.y, code01f.y, dot);
    dot = __fmaf_rn(target4.z, code23f.x, dot);
    dot = __fmaf_rn(target4.w, code23f.y, dot);
    norm = codebook_norm[state];
  } else if constexpr (std::is_same_v<CodebookScalar, float>) {
    if constexpr (VectorSize == 2) {
      // One aligned 8-byte load supplies both FP32 coordinates; the FP32
      // operation sequence below is unchanged from the scalar version.
      const float2 code2 = *reinterpret_cast<const float2*>(codebook + 2 * state);
      dot = __fmaf_rn(target[0], code2.x, dot);
      dot = __fmaf_rn(target[1], code2.y, dot);
    } else {
      #pragma unroll
      for (int i = 0; i < VectorSize; ++i) {
        dot = __fmaf_rn(target[i], codebook[VectorSize * state + i], dot);
      }
    }
    norm = codebook_norm[state];
  } else {
    // V2's immutable FP16 coordinates and their cached FP32 norm occupy one
    // aligned 64-bit record. One scoreboard dependency now supplies both
    // values without changing either bit pattern or the FP32 operation order.
    const uint64_t packed = reinterpret_cast<const uint64_t*>(codebook_norm)[state];
    const uint32_t code_bits = static_cast<uint32_t>(packed);
    const half2 code2_half = *reinterpret_cast<const half2*>(&code_bits);
    const float2 code2 = __half22float2(code2_half);
    norm = __uint_as_float(static_cast<uint32_t>(packed >> 32));
    dot = __fmaf_rn(target[0], code2.x, dot);
    dot = __fmaf_rn(target[1], code2.y, dot);
  }
  const float distance = __fsub_rn(__fadd_rn(target_norm, norm), __fmul_rn(2.0f, dot));
  return __fmul_rn(fmaxf(distance, 0.0f), step_weight);
}

template <bool DirectDistance, typename CodebookScalar>
__device__ __forceinline__ float grid_emission_v2(
    const float* __restrict__ target,
    const CodebookScalar* __restrict__ codebook,
    const float* __restrict__ codebook_norm,
    int state,
    float step_weight,
    float target_norm) {
  if constexpr (DirectDistance && std::is_same_v<CodebookScalar, half>) {
    const float2 code = __half22float2(reinterpret_cast<const half2*>(codebook)[state]);
    const float d0 = __fsub_rn(target[0], code.x);
    const float d1 = __fsub_rn(target[1], code.y);
    const float distance = __fmaf_rn(d1, d1, __fmul_rn(d0, d0));
    return __fmul_rn(distance, step_weight);
  } else {
    return emission<2, CodebookScalar>(
        target, codebook, codebook_norm, state, step_weight, target_norm);
  }
}

template <int VectorSize, typename CodebookScalar>
__device__ __forceinline__ float codebook_norm_value(const CodebookScalar* __restrict__ codebook, int state) {
  float norm = 0.0f;
  if constexpr (VectorSize == 4 && std::is_same_v<CodebookScalar, float>) {
    const float4 code4 = reinterpret_cast<const float4*>(codebook)[state];
    norm = __fadd_rn(norm, __fmul_rn(code4.x, code4.x));
    norm = __fadd_rn(norm, __fmul_rn(code4.y, code4.y));
    norm = __fadd_rn(norm, __fmul_rn(code4.z, code4.z));
    norm = __fadd_rn(norm, __fmul_rn(code4.w, code4.w));
  } else if constexpr (VectorSize == 4) {
    const float2 code01 = __half22float2(reinterpret_cast<const half2*>(codebook)[2 * state]);
    const float2 code23 = __half22float2(reinterpret_cast<const half2*>(codebook)[2 * state + 1]);
    norm = __fadd_rn(norm, __fmul_rn(code01.x, code01.x));
    norm = __fadd_rn(norm, __fmul_rn(code01.y, code01.y));
    norm = __fadd_rn(norm, __fmul_rn(code23.x, code23.x));
    norm = __fadd_rn(norm, __fmul_rn(code23.y, code23.y));
  } else if constexpr (std::is_same_v<CodebookScalar, float>) {
    #pragma unroll
    for (int i = 0; i < VectorSize; ++i) {
      const float value = codebook[VectorSize * state + i];
      norm = __fadd_rn(norm, __fmul_rn(value, value));
    }
  } else {
    const float2 code2 = __half22float2(reinterpret_cast<const half2*>(codebook)[state]);
    norm = __fadd_rn(norm, __fmul_rn(code2.x, code2.x));
    norm = __fadd_rn(norm, __fmul_rn(code2.y, code2.y));
  }
  return norm;
}

template <int VectorSize, typename CodebookScalar>
__global__ void qvq_codebook_norm_kernel(
    const CodebookScalar* __restrict__ codebook,
    float* __restrict__ codebook_norm,
    int bank_count) {
  const int total_states = kStateCount * bank_count;
  for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       index < total_states;
       index += static_cast<int>(gridDim.x * blockDim.x)) {
    const int bank = index / kStateCount;
    const int state = index - bank * kStateCount;
    const CodebookScalar* bank_codebook = codebook + static_cast<int64_t>(bank) * kStateCount * VectorSize;
    codebook_norm[index] = codebook_norm_value<VectorSize, CodebookScalar>(bank_codebook, state);
  }
}

__global__ void qvq_half2_codebook_norm_pack_kernel(
    const half* __restrict__ codebook,
    uint64_t* __restrict__ packed_codebook_norm,
    int bank_count) {
  const int total_states = kStateCount * bank_count;
  for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       index < total_states;
       index += static_cast<int>(gridDim.x * blockDim.x)) {
    const half2 code2_half = reinterpret_cast<const half2*>(codebook)[index];
    const float2 code2 = __half22float2(code2_half);
    float norm = 0.0f;
    norm = __fadd_rn(norm, __fmul_rn(code2.x, code2.x));
    norm = __fadd_rn(norm, __fmul_rn(code2.y, code2.y));
    const uint32_t code_bits = *reinterpret_cast<const uint32_t*>(&code2_half);
    packed_codebook_norm[index] = static_cast<uint64_t>(code_bits) |
        (static_cast<uint64_t>(__float_as_uint(norm)) << 32);
  }
}

template <int VectorSize, typename CodebookScalar>
__global__ void qvq_memoryless_kernel(
    const float* __restrict__ sequences,
    const CodebookScalar* __restrict__ codebook,
    const float* __restrict__ codebook_norm,
    const float* __restrict__ step_weights,
    int steps,
    int batch_size,
    int bank_count,
    bool bank_specific_sequences,
    float* __restrict__ step_error,
    int64_t* __restrict__ states) {
  const int work = static_cast<int>(blockIdx.x);
  const int launch_batch = work / steps;
  const int batch = launch_batch % batch_size;
  const int bank = launch_batch / batch_size;
  const int step = work % steps;
  const int thread = static_cast<int>(threadIdx.x);
  const int64_t sequence_bank_base = bank_specific_sequences
      ? static_cast<int64_t>(bank) * batch_size * steps * VectorSize
      : 0;
  const float* target = sequences + sequence_bank_base +
      (static_cast<int64_t>(batch) * steps + step) * VectorSize;
  codebook += static_cast<int64_t>(bank) * kStateCount * VectorSize;
  constexpr int norm_values_per_state =
      VectorSize == 2 && std::is_same_v<CodebookScalar, half> ? 2 : 1;
  codebook_norm += static_cast<int64_t>(bank) * kStateCount * norm_values_per_state;
  const float weight = step_weights == nullptr ? 1.0f : step_weights[static_cast<int64_t>(batch) * steps + step];
  float target_norm = 0.0f;
  #pragma unroll
  for (int i = 0; i < VectorSize; ++i) {
    target_norm = __fadd_rn(target_norm, __fmul_rn(target[i], target[i]));
  }
  float best = CUDART_INF_F;
  int best_state = kStateCount;
  for (int state = thread; state < kStateCount; state += kThreads) {
    const float value = emission<VectorSize, CodebookScalar>(
        target, codebook, codebook_norm, state, weight, target_norm);
    if (lower_pair(value, state, best, best_state)) {
      best = value;
      best_state = state;
    }
  }
  __shared__ float partial_value[kThreads];
  __shared__ int partial_state[kThreads];
  partial_value[thread] = best;
  partial_state[thread] = best_state;
  __syncthreads();
  if (thread == 0) {
    for (int i = 1; i < kThreads; ++i) {
      if (lower_pair(partial_value[i], partial_state[i], partial_value[0], partial_state[0])) {
        partial_value[0] = partial_value[i];
        partial_state[0] = partial_state[i];
      }
    }
    step_error[static_cast<int64_t>(launch_batch) * steps + step] = partial_value[0];
    states[static_cast<int64_t>(launch_batch) * steps + step] = partial_state[0];
  }
}

struct NormCacheEntry {
  at::Tensor codebook;
  at::Tensor norm;
  uint32_t codebook_version = 0;
  cudaEvent_t ready{};
  int device = -1;
  int vector_size = 0;
  at::ScalarType scalar_type = at::kFloat;
  int bank_count = 1;
};

std::mutex g_norm_cache_mutex;
std::vector<NormCacheEntry> g_norm_cache;

// cudaEventDestroy is device-sensitive: an event is owned by the device that
// was current at cudaEventCreate*, and destroying it while another device is
// current is invalid.  Both caches mix devices.  Use a guard so the previous
// device is restored even when C10_CUDA_CHECK throws.
void destroy_cuda_event_on_device(cudaEvent_t event, int device) {
  const c10::cuda::CUDAGuard device_guard(device);
  C10_CUDA_CHECK(cudaEventDestroy(event));
}

void evict_norm_cache_entry() {
  if (g_norm_cache.size() < kMaxNormCacheEntries) {
    return;
  }
  destroy_cuda_event_on_device(g_norm_cache.front().ready, g_norm_cache.front().device);
  g_norm_cache.erase(g_norm_cache.begin());
}

void record_norm_cache_use(const at::Tensor& codebook, const at::Tensor& norm, cudaStream_t stream) {
  const auto cuda_stream = c10::cuda::getStreamFromExternal(stream, codebook.get_device());
  c10::cuda::CUDACachingAllocator::recordStream(codebook.storage().data_ptr(), cuda_stream);
  c10::cuda::CUDACachingAllocator::recordStream(norm.storage().data_ptr(), cuda_stream);
}

template <int VectorSize, typename CodebookScalar>
at::Tensor build_codebook_norm(const at::Tensor& codebook, int bank_count, cudaStream_t stream) {
  at::Tensor norm;
  if constexpr (VectorSize == 2 && std::is_same_v<CodebookScalar, half>) {
    if (bank_count == 1) {
      norm = at::empty({kStateCount, 2}, codebook.options().dtype(at::kFloat));
    } else {
      norm = at::empty({bank_count, kStateCount, 2}, codebook.options().dtype(at::kFloat));
    }
    qvq_half2_codebook_norm_pack_kernel<<<256 * bank_count, 256, 0, stream>>>(
        reinterpret_cast<const half*>(codebook.const_data_ptr()),
        reinterpret_cast<uint64_t*>(norm.mutable_data_ptr<float>()),
        bank_count);
  } else {
    if (bank_count == 1) {
      norm = at::empty({kStateCount}, codebook.options().dtype(at::kFloat));
    } else {
      norm = at::empty({bank_count, kStateCount}, codebook.options().dtype(at::kFloat));
    }
    qvq_codebook_norm_kernel<VectorSize, CodebookScalar>
        <<<256 * bank_count, 256, 0, stream>>>(
            reinterpret_cast<const CodebookScalar*>(codebook.const_data_ptr()),
            norm.mutable_data_ptr<float>(), bank_count);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return norm;
}

template <int VectorSize, typename CodebookScalar>
at::Tensor cached_codebook_norm(const at::Tensor& codebook, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(g_norm_cache_mutex);
  const int device = codebook.get_device();
  const void* pointer = codebook.data_ptr();
  auto* codebook_impl = const_cast<c10::TensorImpl*>(codebook.unsafeGetTensorImpl());
  // Inference tensors deliberately do not maintain a useful version counter.
  // They can still be mutated inside an inference_mode() block, so caching by
  // pointer would make a later call consume stale norms.  Keep this path
  // uncached; the caller retains the returned tensor through the Viterbi call.
  if (codebook_impl->is_inference()) {
    return build_codebook_norm<VectorSize, CodebookScalar>(codebook, 1, stream);
  }
  const uint32_t codebook_version = codebook_impl->version_counter().current_version();
  for (const auto& entry : g_norm_cache) {
    if (entry.device == device && entry.vector_size == VectorSize && entry.scalar_type == codebook.scalar_type() &&
        entry.bank_count == 1 && entry.codebook_version == codebook_version &&
        entry.codebook.data_ptr() == pointer) {
      C10_CUDA_CHECK(cudaStreamWaitEvent(stream, entry.ready, 0));
      return entry.norm;
    }
  }

  NormCacheEntry entry;
  entry.codebook = codebook;
  entry.norm = build_codebook_norm<VectorSize, CodebookScalar>(codebook, 1, stream);
  entry.device = device;
  entry.vector_size = VectorSize;
  entry.scalar_type = codebook.scalar_type();
  entry.bank_count = 1;
  entry.codebook_version = codebook_version;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&entry.ready, cudaEventDisableTiming));
  C10_CUDA_CHECK(cudaEventRecord(entry.ready, stream));
  at::Tensor result = entry.norm;
  evict_norm_cache_entry();
  g_norm_cache.push_back(std::move(entry));
  return result;
}

template <int VectorSize, typename CodebookScalar>
at::Tensor cached_banked_codebook_norm(const at::Tensor& codebook, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(g_norm_cache_mutex);
  const int device = codebook.get_device();
  const void* pointer = codebook.data_ptr();
  const int bank_count = static_cast<int>(codebook.size(0));
  auto* codebook_impl = const_cast<c10::TensorImpl*>(codebook.unsafeGetTensorImpl());
  if (codebook_impl->is_inference()) {
    return build_codebook_norm<VectorSize, CodebookScalar>(codebook, bank_count, stream);
  }
  const uint32_t codebook_version = codebook_impl->version_counter().current_version();
  for (const auto& entry : g_norm_cache) {
    if (entry.device == device && entry.vector_size == VectorSize && entry.scalar_type == codebook.scalar_type() &&
        entry.bank_count == bank_count && entry.codebook_version == codebook_version &&
        entry.codebook.data_ptr() == pointer) {
      C10_CUDA_CHECK(cudaStreamWaitEvent(stream, entry.ready, 0));
      return entry.norm;
    }
  }

  NormCacheEntry entry;
  entry.codebook = codebook;
  entry.norm = build_codebook_norm<VectorSize, CodebookScalar>(codebook, bank_count, stream);
  entry.device = device;
  entry.vector_size = VectorSize;
  entry.scalar_type = codebook.scalar_type();
  entry.bank_count = bank_count;
  entry.codebook_version = codebook_version;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&entry.ready, cudaEventDisableTiming));
  C10_CUDA_CHECK(cudaEventRecord(entry.ready, stream));
  at::Tensor result = entry.norm;
  evict_norm_cache_entry();
  g_norm_cache.push_back(std::move(entry));
  return result;
}

// ---------------------------------------------------------------------------
// Norm-rank contiguous-band tables.
//
// For one suffix column x the segmented grid recurrence evaluates the
// prefix_count states h * suffix_count + x, h in [0, prefix_count).  These
// tables hold that candidate list once per codebook, sorted ascending by the
// state's cached FP32 squared norm (ties by original prefix) and cut into
// ChunkWidth-wide chunks, chunk-major so a warp visiting one chunk index
// streams fully coalesced lines exactly like the baseline codebook walk:
//
//   records[bank][chunk][x][slot]   the state's exact 64-bit {norm, code}
//   prefixes[bank][chunk][x][slot]  the state's original prefix h, one byte
//   low_norms[bank][chunk][x]       smallest norm in the chunk (slot 0)
//
// One lane reads its chunk as two adjacent 16-byte words, so a warp's chunk
// visit is a contiguous 32 * 32-byte block; the prefix bytes sit
// slot-innermost, so the kernel reads a lane's chunk worth as one
// little-endian NormRankPrefixPack integer.
//
// The original h travels with every record so the (value, lowest original
// prefix) lexicographic tie contract of lower_pair() survives a scan order
// that is by norm rather than by prefix.
// ---------------------------------------------------------------------------
template <int ChunkWidth>
struct NormRankPrefixPack;
template <>
struct NormRankPrefixPack<2> {
  using type = uint16_t;
};
template <>
struct NormRankPrefixPack<4> {
  using type = uint32_t;
};
template <>
struct NormRankPrefixPack<8> {
  using type = uint64_t;
};

// Width of the straight-line evaluation block, per rate.  Four keeps the
// unrolled body inside the 32-register budget that holds two 1024-thread CTAs
// per SM and loads one lane's chunk in two 16-byte transactions; the per-rate
// choices are the fastest of the 2/4/8 sweep recorded in OPTIMIZATION_LOG.md.
constexpr int kNormRankChunkWidthW25 = 4;
constexpr int kNormRankChunkWidthW3 = 4;
constexpr int kNormRankChunkWidthW35 = 4;

// Rank of each state inside its own suffix column, ordered by (norm, original
// prefix).  One thread per state, prefix_count comparisons each; runs once per
// codebook.
template <int Shift, int ChunkWidth>
__global__ void qvq_norm_rank_sort_kernel(
    const uint64_t* __restrict__ packed_records,
    uint64_t* __restrict__ sorted_records,
    uint8_t* __restrict__ sorted_prefixes,
    int bank_count) {
  constexpr int prefix_count = 1 << Shift;
  constexpr int suffix_count = 1 << (16 - Shift);
  const int64_t total = static_cast<int64_t>(bank_count) * kStateCount;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int bank = static_cast<int>(index >> 16);
    const int state = static_cast<int>(index & (kStateCount - 1));
    const int prefix = state >> (16 - Shift);
    const int suffix = state - prefix * suffix_count;
    const uint64_t* bank_records = packed_records + static_cast<int64_t>(bank) * kStateCount;
    const uint64_t self = bank_records[state];
    const float self_norm = __uint_as_float(static_cast<uint32_t>(self >> 32));
    int rank = 0;
    for (int other = 0; other < prefix_count; ++other) {
      const float other_norm = __uint_as_float(
          static_cast<uint32_t>(bank_records[other * suffix_count + suffix] >> 32));
      rank += (other_norm < self_norm || (other_norm == self_norm && other < prefix)) ? 1 : 0;
    }
    const int64_t out = static_cast<int64_t>(bank) * kStateCount +
        static_cast<int64_t>(rank / ChunkWidth) * suffix_count * ChunkWidth +
        static_cast<int64_t>(suffix) * ChunkWidth + rank % ChunkWidth;
    sorted_records[out] = self;
    sorted_prefixes[out] = static_cast<uint8_t>(prefix);
  }
}

// Smallest norm of every chunk, read back from slot zero of the sorted records.
template <int Shift, int ChunkWidth>
__global__ void qvq_norm_rank_bounds_kernel(
    const uint64_t* __restrict__ sorted_records,
    float* __restrict__ low_norms,
    int bank_count) {
  constexpr int prefix_count = 1 << Shift;
  constexpr int chunk_count = prefix_count / ChunkWidth;
  const int64_t total = static_cast<int64_t>(bank_count) * kStateCount / ChunkWidth;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // [bank][chunk][x] is index-compatible with slot zero of the records.
    low_norms[index] = __uint_as_float(
        static_cast<uint32_t>(sorted_records[index * ChunkWidth] >> 32));
  }
}

struct NormRankTables {
  at::Tensor records;
  at::Tensor low_norms;
  at::Tensor prefixes;
};

struct NormRankCacheEntry {
  at::Tensor codebook;
  NormRankTables tables;
  uint32_t codebook_version = 0;
  cudaEvent_t ready{};
  int device = -1;
  int bank_count = 0;
  int shift = 0;
  int chunk_width = 0;
  at::ScalarType scalar_type = at::kHalf;
};

std::mutex g_norm_rank_cache_mutex;
std::vector<NormRankCacheEntry> g_norm_rank_cache;
constexpr size_t kMaxNormRankCacheEntries = 8;

template <int Shift, int ChunkWidth>
NormRankTables build_norm_rank_tables(
    const at::Tensor& codebooks,
    const at::Tensor& packed_norm,
    int bank_count,
    cudaStream_t stream) {
  constexpr int prefix_count = 1 << Shift;
  constexpr int suffix_count = 1 << (16 - Shift);
  constexpr int chunk_count = prefix_count / ChunkWidth;
  NormRankTables tables;
  tables.records = at::empty({bank_count, chunk_count, suffix_count, ChunkWidth},
                             codebooks.options().dtype(at::kLong));
  tables.low_norms = at::empty({bank_count, chunk_count, suffix_count},
                               codebooks.options().dtype(at::kFloat));
  tables.prefixes = at::empty({bank_count, chunk_count, suffix_count, ChunkWidth},
                              codebooks.options().dtype(at::kByte));

  const uint64_t* records_in = reinterpret_cast<const uint64_t*>(packed_norm.const_data_ptr<float>());
  uint64_t* records_out = reinterpret_cast<uint64_t*>(tables.records.mutable_data_ptr<int64_t>());
  constexpr int kBuildBlock = 256;
  const int64_t sort_total = static_cast<int64_t>(bank_count) * kStateCount;
  const int sort_blocks = static_cast<int>(
      std::min<int64_t>((sort_total + kBuildBlock - 1) / kBuildBlock, 8192));
  qvq_norm_rank_sort_kernel<Shift, ChunkWidth><<<sort_blocks, kBuildBlock, 0, stream>>>(
      records_in, records_out, tables.prefixes.mutable_data_ptr<uint8_t>(), bank_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t bounds_total = sort_total / ChunkWidth;
  const int bounds_blocks = static_cast<int>(
      std::min<int64_t>((bounds_total + kBuildBlock - 1) / kBuildBlock, 8192));
  qvq_norm_rank_bounds_kernel<Shift, ChunkWidth><<<bounds_blocks, kBuildBlock, 0, stream>>>(
      records_out, tables.low_norms.mutable_data_ptr<float>(), bank_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tables;
}

void record_norm_rank_cache_use(
    const NormRankTables& tables, int device, cudaStream_t stream) {
  const auto cuda_stream = c10::cuda::getStreamFromExternal(stream, device);
  c10::cuda::CUDACachingAllocator::recordStream(tables.records.storage().data_ptr(), cuda_stream);
  c10::cuda::CUDACachingAllocator::recordStream(tables.low_norms.storage().data_ptr(), cuda_stream);
  c10::cuda::CUDACachingAllocator::recordStream(tables.prefixes.storage().data_ptr(), cuda_stream);
}

// Cached by (device, scalar type, bank count, shift, chunk width, codebook
// storage, codebook version).  A mutated codebook bumps the version counter
// and therefore misses; inference-mode tensors keep no useful version counter,
// so they build uncached exactly like cached_codebook_norm().  A hit waits on
// the producing stream's ready event, so a consumer on another stream never
// reads a table that is still being written.
template <int Shift, int ChunkWidth>
NormRankTables cached_norm_rank_tables(
    const at::Tensor& codebooks,
    const at::Tensor& packed_norm,
    int bank_count,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(g_norm_rank_cache_mutex);
  const int device = codebooks.get_device();
  const void* pointer = codebooks.data_ptr();
  auto* codebook_impl = const_cast<c10::TensorImpl*>(codebooks.unsafeGetTensorImpl());
  if (codebook_impl->is_inference()) {
    return build_norm_rank_tables<Shift, ChunkWidth>(codebooks, packed_norm, bank_count, stream);
  }
  const uint32_t codebook_version = codebook_impl->version_counter().current_version();
  for (const auto& entry : g_norm_rank_cache) {
    if (entry.device == device && entry.bank_count == bank_count && entry.shift == Shift &&
        entry.chunk_width == ChunkWidth && entry.scalar_type == codebooks.scalar_type() &&
        entry.codebook_version == codebook_version && entry.codebook.data_ptr() == pointer) {
      C10_CUDA_CHECK(cudaStreamWaitEvent(stream, entry.ready, 0));
      return entry.tables;
    }
  }
  NormRankCacheEntry entry;
  entry.codebook = codebooks;
  entry.tables = build_norm_rank_tables<Shift, ChunkWidth>(
      codebooks, packed_norm, bank_count, stream);
  entry.device = device;
  entry.bank_count = bank_count;
  entry.shift = Shift;
  entry.chunk_width = ChunkWidth;
  entry.scalar_type = codebooks.scalar_type();
  entry.codebook_version = codebook_version;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&entry.ready, cudaEventDisableTiming));
  C10_CUDA_CHECK(cudaEventRecord(entry.ready, stream));
  NormRankTables result = entry.tables;
  if (g_norm_rank_cache.size() >= kMaxNormRankCacheEntries) {
    // The cache mixes devices: destroy the evicted entry's event on the
    // device that created it, not on the calling thread's current device.
    destroy_cuda_event_on_device(
        g_norm_rank_cache.front().ready, g_norm_rank_cache.front().device);
    g_norm_rank_cache.erase(g_norm_rank_cache.begin());
  }
  g_norm_rank_cache.push_back(std::move(entry));
  return result;
}

__device__ __forceinline__ bool lower_pair(float candidate, int candidate_index, float current, int current_index) {
  return candidate < current || (candidate == current && candidate_index < current_index);
}

template <int Shift, int VectorSize, typename CodebookScalar, typename Backpointer>
__global__ __launch_bounds__(kThreads) void qvq_viterbi_kernel(
    const float* __restrict__ sequences,
    const CodebookScalar* __restrict__ codebook,
    const float* __restrict__ codebook_norm,
    const int64_t* __restrict__ overlap,
    const float* __restrict__ step_weights,
    Backpointer* __restrict__ backpointers,
    int64_t* __restrict__ states,
    float* __restrict__ squared_error,
    int steps,
    int batch_size,
    int bank_count,
    bool bank_specific_sequences,
    bool constrained,
    bool weighted) {
  constexpr int shift = Shift;
  constexpr int suffix_count = 1 << (16 - shift);
  constexpr int overlap_bits = 16 - shift;
  constexpr int overlap_mask = overlap_bits == 0 ? 0 : (1 << overlap_bits) - 1;
  // shift <= 6: every suffix's 2^shift states are owned by one thread
  // (state mod 1024 == suffix mod 1024), so the per-suffix min is local.
  constexpr bool multi_thread_suffix = shift > 6;
  constexpr int suffixes_per_thread = shift <= 6 ? (1 << (6 - shift)) : 1;  // shift <= 6
  constexpr int threads_per_suffix = shift >= 7 ? (1 << (shift - 6)) : 1;   // shift >= 7
  // Fully unroll only W1/W2 transitions; W3's 64-way loop is faster rolled
  // because full expansion increases instruction footprint and register use.

  extern __shared__ float g_shared[];
  float* G_prev = g_shared;
  float* G_next = g_shared + suffix_count;
  __shared__ float partial_val[kThreads];
  __shared__ int partial_h[kThreads];

  const int thread = static_cast<int>(threadIdx.x);
  const int launch_batch = static_cast<int>(blockIdx.x);
  const int batch = launch_batch % batch_size;
  const int bank = launch_batch / batch_size;
  codebook += static_cast<int64_t>(bank) * kStateCount * VectorSize;
  constexpr int norm_values_per_state =
      VectorSize == 2 && std::is_same_v<CodebookScalar, half> ? 2 : 1;
  codebook_norm += static_cast<int64_t>(bank) * kStateCount * norm_values_per_state;
  const int64_t sequence_base = (bank_specific_sequences
      ? static_cast<int64_t>(bank) * batch_size * steps * VectorSize
      : 0) + static_cast<int64_t>(batch) * steps * VectorSize;
  const int64_t pointer_base = static_cast<int64_t>(launch_batch) * steps * suffix_count;
  const int64_t state_base = static_cast<int64_t>(launch_batch) * steps;
  const int required_overlap = constrained ? static_cast<int>(overlap[launch_batch]) : 0;
  const bool constrain_init = constrained && overlap_bits != 0;

  // ---- Step 0: G_0[x] = min_h e_0((h << (16-shift)) | x) ----
  {
    const float* target = sequences + sequence_base;
    const float step_weight = weighted ? step_weights[static_cast<int64_t>(batch) * steps] : 1.0f;
    float target_norm = 0.0f;
    #pragma unroll
    for (int i = 0; i < VectorSize; ++i) {
      target_norm = __fadd_rn(target_norm, __fmul_rn(target[i], target[i]));
    }
    if (!multi_thread_suffix) {
      for (int m = 0; m < suffixes_per_thread; ++m) {
        const int x = (m << 10) | thread;
        const int h0 = x >> (16 - shift);
        float best = CUDART_INF_F;
        int best_h = h0;
        if constexpr (shift <= 4) {
          #pragma unroll
          for (int q = 0; q < (1 << shift); ++q) {
            const int h = h0 + q;
            const int s = (h << (16 - shift)) | x;
            float value = emission<VectorSize, CodebookScalar>(
                target, codebook, codebook_norm, s, step_weight, target_norm);
            if (constrain_init && (s >> shift) != required_overlap) {
              value = CUDART_INF_F;
            }
            if (lower_pair(value, h, best, best_h)) {
              best = value;
              best_h = h;
            }
          }
        } else {
          for (int q = 0; q < (1 << shift); ++q) {
            const int h = h0 + q;
            const int s = (h << (16 - shift)) | x;
            float value = emission<VectorSize, CodebookScalar>(
                target, codebook, codebook_norm, s, step_weight, target_norm);
            if (constrain_init && (s >> shift) != required_overlap) {
              value = CUDART_INF_F;
            }
            if (lower_pair(value, h, best, best_h)) {
              best = value;
              best_h = h;
            }
          }
        }
        G_next[x] = best;
        backpointers[pointer_base + x] = best_h;
      }
    } else {
      // shift >= 7: one suffix per thread, 64 states each.
      const int x = thread & (suffix_count - 1);
      const int h0 = thread >> (16 - shift);
      float best = CUDART_INF_F;
      int best_h = h0;
      // Batched-ILP form; see the main step loop for the bit-exactness note.
      constexpr int kChunk = 8;
      for (int k0 = 0; k0 < 64; k0 += kChunk) {
        float values[kChunk];
        int h_values[kChunk];
        _Pragma("unroll")
        for (int j = 0; j < kChunk; ++j) {
          const int h = h0 + (k0 + j) * threads_per_suffix;
          const int s = (h << (16 - shift)) | x;
          h_values[j] = h;
          values[j] = emission<VectorSize, CodebookScalar>(
              target, codebook, codebook_norm, s, step_weight, target_norm);
          if (constrain_init && (s >> shift) != required_overlap) {
            values[j] = CUDART_INF_F;
          }
        }
        _Pragma("unroll")
        for (int j = 0; j < kChunk; ++j) {
          if (lower_pair(values[j], h_values[j], best, best_h)) {
            best = values[j];
            best_h = h_values[j];
          }
        }
      }
      partial_val[thread] = best;
      partial_h[thread] = best_h;
    }
  }
  __syncthreads();
  if (multi_thread_suffix) {
    if (thread < suffix_count) {
      float best = CUDART_INF_F;
      int best_h = 0;
      for (int m = 0; m < threads_per_suffix; ++m) {
        const int p = thread + m * suffix_count;
        const int candidate_h = partial_h[p];
        const float candidate = partial_val[p];
        if (lower_pair(candidate, candidate_h, best, best_h)) {
          best = candidate;
          best_h = candidate_h;
        }
      }
      G_next[thread] = best;
      backpointers[pointer_base + thread] = best_h;
    }
    __syncthreads();
  }
  float* swap = G_prev;
  G_prev = G_next;
  G_next = swap;

  // ---- Steps 1 .. steps-2 produce G_i; the final step is folded into the
  //      end-state selection below so no extra G pass is needed. ----
  for (int step = 1; step < steps - 1; ++step) {
    const float* target = sequences + sequence_base + static_cast<int64_t>(step) * VectorSize;
    const float step_weight = weighted ? step_weights[static_cast<int64_t>(batch) * steps + step] : 1.0f;
    float target_norm = 0.0f;
    #pragma unroll
    for (int i = 0; i < VectorSize; ++i) {
      target_norm = __fadd_rn(target_norm, __fmul_rn(target[i], target[i]));
    }
    if (!multi_thread_suffix) {
      if constexpr (shift == 5) {
        const int x0 = thread;
        const int x1 = thread + kThreads;
        float best0 = CUDART_INF_F;
        float best1 = CUDART_INF_F;
        int best_h0 = 0;
        int best_h1 = 0;
        for (int h = 0; h < (1 << shift); ++h) {
          const int s0 = (h << (16 - shift)) | x0;
          const int s1 = (h << (16 - shift)) | x1;
          const float value0 = G_prev[s0 >> shift] + emission<VectorSize, CodebookScalar>(
              target, codebook, codebook_norm, s0, step_weight, target_norm);
          const float value1 = G_prev[s1 >> shift] + emission<VectorSize, CodebookScalar>(
              target, codebook, codebook_norm, s1, step_weight, target_norm);
          if (lower_pair(value0, h, best0, best_h0)) {
            best0 = value0;
            best_h0 = h;
          }
          if (lower_pair(value1, h, best1, best_h1)) {
            best1 = value1;
            best_h1 = h;
          }
        }
        G_next[x0] = best0;
        G_next[x1] = best1;
        backpointers[pointer_base + static_cast<int64_t>(step) * suffix_count + x0] = best_h0;
        backpointers[pointer_base + static_cast<int64_t>(step) * suffix_count + x1] = best_h1;
      } else {
        for (int m = 0; m < suffixes_per_thread; ++m) {
        const int x = (m << 10) | thread;
        const int h0 = x >> (16 - shift);
        float best = CUDART_INF_F;
        int best_h = h0;
        if constexpr (shift <= 4) {
          #pragma unroll
          for (int q = 0; q < (1 << shift); ++q) {
            const int h = h0 + q;
            const int s = (h << (16 - shift)) | x;
            const float value =
                G_prev[s >> shift] + emission<VectorSize, CodebookScalar>(
                    target, codebook, codebook_norm, s, step_weight, target_norm);
            if (lower_pair(value, h, best, best_h)) {
              best = value;
              best_h = h;
            }
          }
        } else {
          for (int q = 0; q < (1 << shift); ++q) {
            const int h = h0 + q;
            const int s = (h << (16 - shift)) | x;
            const float value =
                G_prev[s >> shift] + emission<VectorSize, CodebookScalar>(
                    target, codebook, codebook_norm, s, step_weight, target_norm);
            if (lower_pair(value, h, best, best_h)) {
              best = value;
              best_h = h;
            }
          }
        }
        G_next[x] = best;
        backpointers[pointer_base + static_cast<int64_t>(step) * suffix_count + x] = best_h;
        }
      }
      __syncthreads();
    } else {
      const int x = thread & (suffix_count - 1);
      const int h0 = thread >> (16 - shift);
      float best = CUDART_INF_F;
      int best_h = h0;
      // Batched-ILP form: all loads/FMAs of a chunk are issued before any
      // min-fold consumes them, hiding L2 latency that serialized the naive
      // loop. Min/argmin over a fixed set is fold-order invariant, so results
      // are bit-identical.
      constexpr int kChunk = 8;
      const int h_step = threads_per_suffix * kChunk;
      for (int k0 = 0; k0 < 64; k0 += kChunk) {
        float values[kChunk];
        float g_values[kChunk];
        int h_values[kChunk];
        #pragma unroll
        for (int j = 0; j < kChunk; ++j) {
          const int h = h0 + (k0 + j) * threads_per_suffix;
          const int s = (h << (16 - shift)) | x;
          h_values[j] = h;
          g_values[j] = G_prev[s >> shift];
          values[j] = emission<VectorSize, CodebookScalar>(
              target, codebook, codebook_norm, s, step_weight, target_norm);
        }
        _Pragma("unroll")
        for (int j = 0; j < kChunk; ++j) {
          const float value = g_values[j] + values[j];
          if (lower_pair(value, h_values[j], best, best_h)) {
            best = value;
            best_h = h_values[j];
          }
        }
      }
      partial_val[thread] = best;
      partial_h[thread] = best_h;
    }
    if (multi_thread_suffix) {
      __syncthreads();
      if (thread < suffix_count) {
        float best = CUDART_INF_F;
        int best_h = 0;
        for (int m = 0; m < threads_per_suffix; ++m) {
          const int p = thread + m * suffix_count;
          const int candidate_h = partial_h[p];
          const float candidate = partial_val[p];
          if (lower_pair(candidate, candidate_h, best, best_h)) {
            best = candidate;
            best_h = candidate_h;
          }
        }
        G_next[thread] = best;
        backpointers[pointer_base + static_cast<int64_t>(step) * suffix_count + thread] = best_h;
      }
      __syncthreads();
    }
    swap = G_prev;
    G_prev = G_next;
    G_next = swap;
  }

  // ---- Final end-state selection over f_last(s); also writes the last
  //      backpointer (index steps-2) that the step loop does not produce. ----
  const float* target = sequences + sequence_base + static_cast<int64_t>(steps - 1) * VectorSize;
  const float step_weight = weighted ? step_weights[static_cast<int64_t>(batch) * steps + steps - 1] : 1.0f;
  float target_norm = 0.0f;
  #pragma unroll
  for (int i = 0; i < VectorSize; ++i) {
    target_norm = __fadd_rn(target_norm, __fmul_rn(target[i], target[i]));
  }
  const bool constrain_final = constrained && overlap_bits != 0;
  const bool write_last_backpointer = steps > 1;
  float best = CUDART_INF_F;
  int best_state = kStateCount;
  if (!multi_thread_suffix) {
    for (int m = 0; m < suffixes_per_thread; ++m) {
      const int x = (m << 10) | thread;
      const int h0 = x >> (16 - shift);
      float suffix_best = CUDART_INF_F;
      int suffix_best_h = h0;
      if constexpr (shift <= 4) {
        #pragma unroll
        for (int q = 0; q < (1 << shift); ++q) {
          const int h = h0 + q;
          const int s = (h << (16 - shift)) | x;
          float value;
          if (steps == 1) {
            value = emission<VectorSize, CodebookScalar>(
                target, codebook, codebook_norm, s, step_weight, target_norm);
          } else {
            value = G_prev[s >> shift] + emission<VectorSize, CodebookScalar>(
                target, codebook, codebook_norm, s, step_weight, target_norm);
          }
          if (constrain_final && (s & overlap_mask) != required_overlap) {
            value = CUDART_INF_F;
          }
          if (lower_pair(value, h, suffix_best, suffix_best_h)) {
            suffix_best = value;
            suffix_best_h = h;
          }
          if (lower_pair(value, s, best, best_state)) {
            best = value;
            best_state = s;
          }
        }
      } else {
        for (int q = 0; q < (1 << shift); ++q) {
        const int h = h0 + q;
        const int s = (h << (16 - shift)) | x;
        float value;
        if (steps == 1) {
          value = emission<VectorSize, CodebookScalar>(
              target, codebook, codebook_norm, s, step_weight, target_norm);
        } else {
          value = G_prev[s >> shift] + emission<VectorSize, CodebookScalar>(
              target, codebook, codebook_norm, s, step_weight, target_norm);
        }
        // End-state tail-biting constraint: the final state's low (L-shift)
        // bits must equal the overlap. The multi-thread path applies this via
        // `!constrain_final || x == required_overlap`; mirror it here so the
        // constrained search never selects a non-circular end state.
        if (constrain_final && (s & overlap_mask) != required_overlap) {
          value = CUDART_INF_F;
        }
        if (lower_pair(value, h, suffix_best, suffix_best_h)) {
          suffix_best = value;
          suffix_best_h = h;
        }
          if (lower_pair(value, s, best, best_state)) {
            best = value;
            best_state = s;
          }
        }
      }
      if (write_last_backpointer) {
        backpointers[pointer_base + static_cast<int64_t>(steps - 1) * suffix_count + x] = suffix_best_h;
      }
    }
  } else {
    const int x = thread & (suffix_count - 1);
    float suffix_best = CUDART_INF_F;
    int suffix_best_h = 0;
    if (!constrain_final || x == required_overlap) {
      const int h0 = thread >> (16 - shift);
      // Batched-ILP form; see the main step loop for the bit-exactness note.
      constexpr int kChunk = 8;
      for (int k0 = 0; k0 < 64; k0 += kChunk) {
        float values[kChunk];
        float g_values[kChunk];
        int h_values[kChunk];
        int s_values[kChunk];
        _Pragma("unroll")
        for (int j = 0; j < kChunk; ++j) {
          const int h = h0 + (k0 + j) * threads_per_suffix;
          const int s = (h << (16 - shift)) | x;
          h_values[j] = h;
          s_values[j] = s;
          g_values[j] = steps == 1 ? 0.0f : G_prev[s >> shift];
          values[j] = emission<VectorSize, CodebookScalar>(
              target, codebook, codebook_norm, s, step_weight, target_norm);
        }
        _Pragma("unroll")
        for (int j = 0; j < kChunk; ++j) {
          const float value = g_values[j] + values[j];
          if (lower_pair(value, h_values[j], suffix_best, suffix_best_h)) {
            suffix_best = value;
            suffix_best_h = h_values[j];
          }
          if (lower_pair(value, s_values[j], best, best_state)) {
            best = value;
            best_state = s_values[j];
          }
        }
      }
    }
    partial_val[thread] = suffix_best;
    partial_h[thread] = suffix_best_h;
    if (write_last_backpointer) {
      __syncthreads();
      if (thread < suffix_count) {
        float suffix_best_final = CUDART_INF_F;
        int suffix_best_h_final = 0;
        for (int m = 0; m < threads_per_suffix; ++m) {
          const int p = thread + m * suffix_count;
          const int candidate_h = partial_h[p];
          const float candidate = partial_val[p];
          if (lower_pair(candidate, candidate_h, suffix_best_final, suffix_best_h_final)) {
            suffix_best_final = candidate;
            suffix_best_h_final = candidate_h;
          }
        }
        backpointers[pointer_base + static_cast<int64_t>(steps - 1) * suffix_count + thread] = suffix_best_h_final;
      }
      __syncthreads();
    }
  }
  partial_val[thread] = best;
  partial_h[thread] = best_state;
  __syncthreads();
  if (thread == 0) {
    for (int candidate_thread = 1; candidate_thread < kThreads; ++candidate_thread) {
      const float candidate = partial_val[candidate_thread];
      const int candidate_state = partial_h[candidate_thread];
      if (lower_pair(candidate, candidate_state, best, best_state)) {
        best = candidate;
        best_state = candidate_state;
      }
    }

    squared_error[launch_batch] = best;
    states[state_base + steps - 1] = best_state;
#if defined(QVQ_DEBUG_NO_TRACEBACK)
    for (int step = steps - 1; step > 0; --step) {
      states[state_base + step - 1] = best_state;
    }
#else
    for (int step = steps - 1; step > 0; --step) {
      const int x_var = best_state >> shift;
      const int h_p = backpointers[pointer_base + static_cast<int64_t>(step - 1) * suffix_count + x_var];
      best_state = (h_p << (16 - shift)) | x_var;
      states[state_base + step - 1] = best_state;
    }
#endif

  }
}

// Coupled segmented-bank V2 recurrence retained for V2B4-P64.
//
// A selector is constant for SegmentSteps transitions.  Inside a segment the
// recurrence minimizes predecessor prefixes within the current bank.  At a
// segment boundary it additionally minimizes the predecessor bank, with the
// flattened (bank, prefix) ordering matching torch.min exactly: bank zero and
// then the lowest prefix win exact ties.  Costs remain in global workspace so
// W1's four-bank 4x65536 state frontier does not exceed A100 shared memory;
// one persistent CTA per sequence removes the 128 Python/Torch launch chain.
template <int Shift, typename CodebookScalar, typename BackpointerScalar>
__global__ __launch_bounds__(kThreads) void qvq_v2_segment_banked_kernel(
    const float* __restrict__ sequences,
    const CodebookScalar* __restrict__ codebooks,
    const float* __restrict__ codebook_norms,
    const int64_t* __restrict__ overlap,
    const float* __restrict__ step_weights,
    float* __restrict__ costs_a,
    float* __restrict__ costs_b,
    float* __restrict__ reduced,
    BackpointerScalar* __restrict__ backpointers,
    int64_t* __restrict__ states,
    uint8_t* __restrict__ segment_bank_ids,
    float* __restrict__ squared_error,
    int steps,
    int bank_count,
    int segment_steps,
    bool constrained,
    bool weighted) {
  constexpr int shift = Shift;
  constexpr int prefix_count = 1 << shift;
  constexpr int suffix_count = 1 << (16 - shift);
  constexpr int overlap_mask = suffix_count - 1;

  __shared__ float partial_value[kThreads];
  __shared__ int partial_index[kThreads];

  const int thread = static_cast<int>(threadIdx.x);
  const int batch = static_cast<int>(blockIdx.x);
  const int bank_state_count = bank_count * kStateCount;
  const int bank_suffix_count = bank_count * suffix_count;
  const int segment_count = steps / segment_steps;
  const int required_overlap = constrained ? static_cast<int>(overlap[batch]) : 0;
  const int64_t sequence_base = static_cast<int64_t>(batch) * steps * 2;
  const int64_t cost_base = static_cast<int64_t>(batch) * bank_state_count;
  const int64_t reduced_base = static_cast<int64_t>(batch) * bank_suffix_count;
  const int64_t pointer_base = static_cast<int64_t>(batch) * (steps - 1) * bank_suffix_count;
  const int64_t state_base = static_cast<int64_t>(batch) * steps;
  const int64_t selector_base = static_cast<int64_t>(batch) * segment_count;
  float* current_costs = costs_a + cost_base;
  float* next_costs = costs_b + cost_base;
  float* batch_reduced = reduced + reduced_base;

  // Step zero retains one complete state frontier per bank.  The constrained
  // pass applies the same start-state suffix rule as the eager recurrence.
  {
    const float* target = sequences + sequence_base;
    const float weight = weighted ? step_weights[static_cast<int64_t>(batch) * steps] : 1.0f;
    const float target_norm = __fadd_rn(
        __fmul_rn(target[0], target[0]),
        __fmul_rn(target[1], target[1]));
    for (int flat = thread; flat < bank_state_count; flat += kThreads) {
      const int bank = flat / kStateCount;
      const int state = flat - bank * kStateCount;
      const CodebookScalar* bank_codebook = codebooks + static_cast<int64_t>(bank) * kStateCount * 2;
      const float* bank_norm = codebook_norms + static_cast<int64_t>(bank) * kStateCount *
          (std::is_same_v<CodebookScalar, half> ? 2 : 1);
      float value = emission<2, CodebookScalar>(
          target, bank_codebook, bank_norm, state, weight, target_norm);
      if (constrained && (state >> shift) != required_overlap) {
        value = CUDART_INF_F;
      }
      current_costs[flat] = value;
    }
  }
  __syncthreads();

  for (int step = 1; step < steps; ++step) {
    const bool boundary = step % segment_steps == 0;

    // Reduce the predecessor frontier to one cost per retained suffix.  At a
    // boundary the bank-major loop order is the public deterministic contract.
    for (int flat = thread; flat < bank_suffix_count; flat += kThreads) {
      const int current_bank = flat / suffix_count;
      const int suffix = flat - current_bank * suffix_count;
      float best = CUDART_INF_F;
      int best_pointer = 0;
      const int first_bank = boundary ? 0 : current_bank;
      const int last_bank = boundary ? bank_count : current_bank + 1;
      for (int previous_bank = first_bank; previous_bank < last_bank; ++previous_bank) {
        #pragma unroll
        for (int prefix = 0; prefix < prefix_count; ++prefix) {
          const int predecessor_state = prefix * suffix_count + suffix;
          const float candidate = current_costs[previous_bank * kStateCount + predecessor_state];
          const int candidate_pointer = boundary ? previous_bank * prefix_count + prefix : prefix;
          if (lower_pair(candidate, candidate_pointer, best, best_pointer)) {
            best = candidate;
            best_pointer = candidate_pointer;
          }
        }
      }
      batch_reduced[flat] = best;
      backpointers[pointer_base +
                   static_cast<int64_t>(step - 1) * bank_suffix_count + flat] =
          static_cast<BackpointerScalar>(best_pointer);
    }
    __syncthreads();

    const float* target = sequences + sequence_base + static_cast<int64_t>(step) * 2;
    const float weight = weighted
        ? step_weights[static_cast<int64_t>(batch) * steps + step]
        : 1.0f;
    const float target_norm = __fadd_rn(
        __fmul_rn(target[0], target[0]),
        __fmul_rn(target[1], target[1]));
    for (int flat = thread; flat < bank_state_count; flat += kThreads) {
      const int bank = flat / kStateCount;
      const int state = flat - bank * kStateCount;
      const int predecessor_suffix = state >> shift;
      const CodebookScalar* bank_codebook = codebooks + static_cast<int64_t>(bank) * kStateCount * 2;
      const float* bank_norm = codebook_norms + static_cast<int64_t>(bank) * kStateCount *
          (std::is_same_v<CodebookScalar, half> ? 2 : 1);
      next_costs[flat] = __fadd_rn(
          batch_reduced[bank * suffix_count + predecessor_suffix],
          emission<2, CodebookScalar>(
              target, bank_codebook, bank_norm, state, weight, target_norm));
    }
    __syncthreads();
    float* swap = current_costs;
    current_costs = next_costs;
    next_costs = swap;
  }

  float best = CUDART_INF_F;
  int best_flat = bank_state_count;
  for (int flat = thread; flat < bank_state_count; flat += kThreads) {
    const int state = flat % kStateCount;
    float candidate = current_costs[flat];
    if (constrained && (state & overlap_mask) != required_overlap) {
      candidate = CUDART_INF_F;
    }
    if (lower_pair(candidate, flat, best, best_flat)) {
      best = candidate;
      best_flat = flat;
    }
  }
  partial_value[thread] = best;
  partial_index[thread] = best_flat;
  __syncthreads();

  if (thread == 0) {
    for (int candidate_thread = 1; candidate_thread < kThreads; ++candidate_thread) {
      if (lower_pair(
              partial_value[candidate_thread], partial_index[candidate_thread],
              best, best_flat)) {
        best = partial_value[candidate_thread];
        best_flat = partial_index[candidate_thread];
      }
    }
    squared_error[batch] = best;
    int current_bank = best_flat / kStateCount;
    int current_state = best_flat - current_bank * kStateCount;
    states[state_base + steps - 1] = current_state;
    if ((steps - 1) % segment_steps == 0) {
      segment_bank_ids[selector_base + (steps - 1) / segment_steps] =
          static_cast<uint8_t>(current_bank);
    }
    for (int step = steps - 1; step > 0; --step) {
      const int predecessor_suffix = current_state >> shift;
      const int pointer = static_cast<int>(backpointers[
          pointer_base + static_cast<int64_t>(step - 1) * bank_suffix_count +
          current_bank * suffix_count + predecessor_suffix]);
      int prefix;
      if (step % segment_steps == 0) {
        current_bank = pointer / prefix_count;
        prefix = pointer % prefix_count;
      } else {
        prefix = pointer;
      }
      current_state = prefix * suffix_count + predecessor_suffix;
      states[state_base + step - 1] = current_state;
      if ((step - 1) % segment_steps == 0) {
        segment_bank_ids[selector_base + (step - 1) / segment_steps] =
            static_cast<uint8_t>(current_bank);
      }
    }
  }
}

// Accuracy-first G-only reformulation for B2-P32 and B4-P64.  G[b, x] is the minimum
// complete-state cost in bank b whose low retained suffix is x.  Segment
// boundaries need only an additional previous-bank choice per suffix; the
// winning prefix remains in the ordinary G traceback.  This removes both
// 65,536-state cost frontiers without changing emission or addition order.
template <int Shift, int BankCount, int SegmentSteps, typename CodebookScalar, typename BackpointerScalar>
__global__ __launch_bounds__(kThreads) void qvq_v2_segment_g_kernel(
    const float* __restrict__ sequences,
    const CodebookScalar* __restrict__ codebooks,
    const float* __restrict__ codebook_norms,
    const int64_t* __restrict__ overlap,
    const float* __restrict__ step_weights,
    float* __restrict__ g_a,
    float* __restrict__ g_b,
    BackpointerScalar* __restrict__ backpointers,
    uint8_t* __restrict__ boundary_banks,
    int64_t* __restrict__ states,
    uint8_t* __restrict__ segment_bank_ids,
    float* __restrict__ squared_error,
    bool constrained,
    bool weighted) {
  constexpr int shift = Shift;
  constexpr int prefix_count = 1 << shift;
  constexpr int suffix_count = 1 << (16 - shift);
  constexpr int overlap_mask = suffix_count - 1;
  constexpr int bank_count = BankCount;
  constexpr int steps = 128;
  constexpr int segment_steps = SegmentSteps;
  constexpr int segment_count = steps / segment_steps;
  constexpr int bank_suffix_count = bank_count * suffix_count;

  __shared__ float partial_value[kThreads];
  __shared__ int partial_index[kThreads];

  const int thread = static_cast<int>(threadIdx.x);
  const int batch = static_cast<int>(blockIdx.x);
  const int required_overlap = constrained ? static_cast<int>(overlap[batch]) : 0;
  const int64_t sequence_base = static_cast<int64_t>(batch) * steps * 2;
  const int64_t g_base = static_cast<int64_t>(batch) * bank_suffix_count;
  const int64_t pointer_base = static_cast<int64_t>(batch) * (steps - 1) * bank_suffix_count;
  const int64_t boundary_base = static_cast<int64_t>(batch) * (segment_count - 1) * suffix_count;
  const int64_t state_base = static_cast<int64_t>(batch) * steps;
  const int64_t selector_base = static_cast<int64_t>(batch) * segment_count;
  float* g_previous = g_a + g_base;
  float* g_next = g_b + g_base;

  // G_0[b,x] = min_h emission_b((h << (16-shift)) | x).
  {
    const float* target = sequences + sequence_base;
    const float weight = weighted ? step_weights[static_cast<int64_t>(batch) * steps] : 1.0f;
    const float target_norm = __fadd_rn(
        __fmul_rn(target[0], target[0]),
        __fmul_rn(target[1], target[1]));
    for (int flat = thread; flat < bank_suffix_count; flat += kThreads) {
      const int bank = flat / suffix_count;
      const int x = flat - bank * suffix_count;
      const CodebookScalar* bank_codebook =
          codebooks + static_cast<int64_t>(bank) * kStateCount * 2;
      const float* bank_norm = codebook_norms + static_cast<int64_t>(bank) * kStateCount *
          (std::is_same_v<CodebookScalar, half> ? 2 : 1);
      float best = CUDART_INF_F;
      int best_h = 0;
      for (int h = 0; h < prefix_count; ++h) {
        const int state = h * suffix_count + x;
        float candidate = emission<2, CodebookScalar>(
            target, bank_codebook, bank_norm, state, weight, target_norm);
        if (constrained && (state >> shift) != required_overlap) {
          candidate = CUDART_INF_F;
        }
        if (lower_pair(candidate, h, best, best_h)) {
          best = candidate;
          best_h = h;
        }
      }
      g_previous[flat] = best;
      backpointers[pointer_base + flat] = static_cast<BackpointerScalar>(best_h);
    }
  }
  __syncthreads();

  // G_1 through G_126.  G_127 is folded into the final state reduction.
  for (int step = 1; step < steps - 1; ++step) {
    const bool boundary = step % segment_steps == 0;
    const int boundary_index = step / segment_steps - 1;
    if (boundary) {
      for (int x = thread; x < suffix_count; x += kThreads) {
        float best_bank_cost = g_previous[x];
        int best_bank = 0;
        for (int candidate_bank = 1; candidate_bank < bank_count; ++candidate_bank) {
          const float candidate = g_previous[candidate_bank * suffix_count + x];
          if (candidate < best_bank_cost) {
            best_bank_cost = candidate;
            best_bank = candidate_bank;
          }
        }
        // Flattened old ordering is (bank, prefix), hence bank zero wins every
        // exact cross-bank tie regardless of each bank's winning prefix.
        boundary_banks[boundary_base + static_cast<int64_t>(boundary_index) * suffix_count + x] =
            static_cast<uint8_t>(best_bank);
      }
      __syncthreads();
    }

    const float* target = sequences + sequence_base + static_cast<int64_t>(step) * 2;
    const float weight = weighted
        ? step_weights[static_cast<int64_t>(batch) * steps + step]
        : 1.0f;
    const float target_norm = __fadd_rn(
        __fmul_rn(target[0], target[0]),
        __fmul_rn(target[1], target[1]));
    for (int flat = thread; flat < bank_suffix_count; flat += kThreads) {
      const int bank = flat / suffix_count;
      const int x = flat - bank * suffix_count;
      const CodebookScalar* bank_codebook =
          codebooks + static_cast<int64_t>(bank) * kStateCount * 2;
      const float* bank_norm = codebook_norms + static_cast<int64_t>(bank) * kStateCount *
          (std::is_same_v<CodebookScalar, half> ? 2 : 1);
      float best = CUDART_INF_F;
      int best_h = 0;
      for (int h = 0; h < prefix_count; ++h) {
        const int state = h * suffix_count + x;
        const int predecessor_suffix = state >> shift;
        const int previous_bank = boundary
            ? static_cast<int>(boundary_banks[
                  boundary_base + static_cast<int64_t>(boundary_index) * suffix_count +
                  predecessor_suffix])
            : bank;
        const float candidate = __fadd_rn(
            g_previous[previous_bank * suffix_count + predecessor_suffix],
            emission<2, CodebookScalar>(
                target, bank_codebook, bank_norm, state, weight, target_norm));
        if (lower_pair(candidate, h, best, best_h)) {
          best = candidate;
          best_h = h;
        }
      }
      g_next[flat] = best;
      backpointers[pointer_base + static_cast<int64_t>(step) * bank_suffix_count + flat] =
          static_cast<BackpointerScalar>(best_h);
    }
    __syncthreads();
    float* swap = g_previous;
    g_previous = g_next;
    g_next = swap;
  }

  const float* target = sequences + sequence_base + static_cast<int64_t>(steps - 1) * 2;
  const float weight = weighted
      ? step_weights[static_cast<int64_t>(batch) * steps + steps - 1]
      : 1.0f;
  const float target_norm = __fadd_rn(
      __fmul_rn(target[0], target[0]),
      __fmul_rn(target[1], target[1]));
  float best = CUDART_INF_F;
  int best_flat = bank_count * kStateCount;
  for (int flat = thread; flat < bank_count * kStateCount; flat += kThreads) {
    const int bank = flat / kStateCount;
    const int state = flat - bank * kStateCount;
    const CodebookScalar* bank_codebook =
        codebooks + static_cast<int64_t>(bank) * kStateCount * 2;
    const float* bank_norm = codebook_norms + static_cast<int64_t>(bank) * kStateCount *
        (std::is_same_v<CodebookScalar, half> ? 2 : 1);
    float candidate = __fadd_rn(
        g_previous[bank * suffix_count + (state >> shift)],
        emission<2, CodebookScalar>(
            target, bank_codebook, bank_norm, state, weight, target_norm));
    if (constrained && (state & overlap_mask) != required_overlap) {
      candidate = CUDART_INF_F;
    }
    if (lower_pair(candidate, flat, best, best_flat)) {
      best = candidate;
      best_flat = flat;
    }
  }
  partial_value[thread] = best;
  partial_index[thread] = best_flat;
  __syncthreads();

  if (thread == 0) {
    for (int candidate_thread = 1; candidate_thread < kThreads; ++candidate_thread) {
      if (lower_pair(
              partial_value[candidate_thread], partial_index[candidate_thread],
              best, best_flat)) {
        best = partial_value[candidate_thread];
        best_flat = partial_index[candidate_thread];
      }
    }
    squared_error[batch] = best;
    int current_bank = best_flat / kStateCount;
    int current_state = best_flat - current_bank * kStateCount;
    states[state_base + steps - 1] = current_state;
    segment_bank_ids[selector_base + segment_count - 1] = static_cast<uint8_t>(current_bank);
    for (int step = steps - 1; step > 0; --step) {
      const int predecessor_suffix = current_state >> shift;
      if (step % segment_steps == 0) {
        const int boundary_index = step / segment_steps - 1;
        current_bank = static_cast<int>(boundary_banks[
            boundary_base + static_cast<int64_t>(boundary_index) * suffix_count +
            predecessor_suffix]);
        segment_bank_ids[selector_base + boundary_index] = static_cast<uint8_t>(current_bank);
      }
      const int prefix = static_cast<int>(backpointers[
          pointer_base + static_cast<int64_t>(step - 1) * bank_suffix_count +
          current_bank * suffix_count + predecessor_suffix]);
      current_state = prefix * suffix_count + predecessor_suffix;
      states[state_base + step - 1] = current_state;
    }
  }
}

// Grid-parallel form of the exact G-only recurrence.  A persistent CTA per
// sequence leaves small YAQA anti-diagonals unable to fill an A100.  Here one
// CTA owns (sequence, current bank) for one selector segment.  Kernel-launch
// ordering is the global barrier between segments, while the ordinary G
// frontier and traceback tensors preserve exactly the same dynamic program.
template <int Shift, int BankCount, int SegmentSteps>
__global__ __launch_bounds__(kThreads) void qvq_v2_segment_boundary_kernel(
    const float* __restrict__ g_previous,
    uint8_t* __restrict__ boundary_banks,
    int batch,
    int boundary_index) {
  constexpr int suffix_count = 1 << (16 - Shift);
  constexpr int bank_count = BankCount;
  constexpr int segment_count = 128 / SegmentSteps;
  const int sequence = static_cast<int>(blockIdx.x);
  if (sequence >= batch) {
    return;
  }
  const int64_t g_base = static_cast<int64_t>(sequence) * bank_count * suffix_count;
  const int64_t boundary_base =
      static_cast<int64_t>(sequence) * (segment_count - 1) * suffix_count +
      static_cast<int64_t>(boundary_index) * suffix_count;
  for (int x = static_cast<int>(threadIdx.x); x < suffix_count; x += kThreads) {
    float best = g_previous[g_base + x];
    int best_bank = 0;
    #pragma unroll
    for (int bank = 1; bank < bank_count; ++bank) {
      const float candidate = g_previous[g_base + bank * suffix_count + x];
      // Strict comparison preserves bank-zero/lowest-bank tie precedence.
      if (candidate < best) {
        best = candidate;
        best_bank = bank;
      }
    }
    boundary_banks[boundary_base + x] = static_cast<uint8_t>(best_bank);
  }
}

template <
    int Shift,
    int BankCount,
    int SegmentSteps,
    bool FuseBoundary,
    bool MidpointOnly,
    bool DirectDistance,
    typename CodebookScalar,
    typename BackpointerScalar>
__global__ __launch_bounds__(kThreads) void qvq_v2_segment_grid_kernel(
    const float* __restrict__ sequences,
    const CodebookScalar* __restrict__ codebooks,
    const float* __restrict__ codebook_norms,
    const int64_t* __restrict__ overlap,
    const float* __restrict__ step_weights,
    const float* __restrict__ g_input_all,
    float* __restrict__ g_output_all,
    BackpointerScalar* __restrict__ backpointers,
    uint8_t* __restrict__ boundary_banks,
    int batch,
    int family_batch,
    int segment_index,
    bool constrained,
    bool weighted) {
  constexpr int shift = Shift;
  constexpr int prefix_count = 1 << shift;
  constexpr int suffix_count = 1 << (16 - shift);
  constexpr int bank_count = BankCount;
  constexpr int segment_steps = SegmentSteps;
  constexpr int segment_count = 128 / segment_steps;
  constexpr int bank_suffix_count = bank_count * suffix_count;

  const int flat_block = static_cast<int>(blockIdx.x);
  const int sequence = flat_block / bank_count;
  const int bank = flat_block - sequence * bank_count;
  if (sequence >= batch) {
    return;
  }
  const int thread = static_cast<int>(threadIdx.x);
  const int first_step = segment_index * segment_steps;
  // Step 127 remains folded into the final reduction to preserve the exact
  // bank-major then state-major terminal tie ordering.
  const int end_step = min(first_step + segment_steps, 127);
  const int required_overlap = constrained ? static_cast<int>(overlap[sequence]) : 0;
  const int64_t sequence_base = static_cast<int64_t>(sequence) * 128 * 2;
  const int64_t g_base = static_cast<int64_t>(sequence) * bank_suffix_count + bank * suffix_count;
  constexpr int first_pointer_step = MidpointOnly ? 63 : 0;
  constexpr int stored_pointer_steps = 127 - first_pointer_step;
  const int64_t pointer_base =
      static_cast<int64_t>(sequence) * stored_pointer_steps * bank_suffix_count;
  const int64_t boundary_base = static_cast<int64_t>(sequence) * (segment_count - 1) * suffix_count;
  const int family = family_batch == 0 ? 0 : sequence / family_batch;
  const int physical_bank = family * bank_count + bank;
  const CodebookScalar* bank_codebook =
      codebooks + static_cast<int64_t>(physical_bank) * kStateCount * 2;
  const float* bank_norm = codebook_norms + static_cast<int64_t>(physical_bank) * kStateCount *
      (std::is_same_v<CodebookScalar, half> ? 2 : 1);
  extern __shared__ float shared_frontiers[];
  float* g_previous = shared_frontiers;
  float* g_scratch = shared_frontiers + suffix_count;
  float* partial_values = shared_frontiers + 2 * suffix_count;
  int* partial_prefixes = reinterpret_cast<int*>(partial_values + (shift == 7 ? kThreads : 0));

  if (segment_index == 0) {
    const float* target = sequences + sequence_base;
    const float weight = weighted ? step_weights[static_cast<int64_t>(sequence) * 128] : 1.0f;
    const float target_norm = __fadd_rn(
        __fmul_rn(target[0], target[0]),
        __fmul_rn(target[1], target[1]));
    if constexpr (shift == 7) {
      const int x = thread & (suffix_count - 1);
      const int first_h = thread >> 9;
      float best = CUDART_INF_F;
      int best_h = first_h;
      for (int h = first_h; h < prefix_count; h += 2) {
        const int state = h * suffix_count + x;
        float candidate = grid_emission_v2<DirectDistance, CodebookScalar>(
            target, bank_codebook, bank_norm, state, weight, target_norm);
        if (constrained && (state >> shift) != required_overlap) {
          candidate = CUDART_INF_F;
        }
        if (lower_pair(candidate, h, best, best_h)) {
          best = candidate;
          best_h = h;
        }
      }
      partial_values[thread] = best;
      partial_prefixes[thread] = best_h;
      __syncthreads();
      if (thread < suffix_count) {
        const float other = partial_values[thread + suffix_count];
        const int other_h = partial_prefixes[thread + suffix_count];
        if (lower_pair(other, other_h, best, best_h)) {
          best = other;
          best_h = other_h;
        }
        g_previous[x] = best;
        if constexpr (!MidpointOnly) {
          backpointers[pointer_base + bank * suffix_count + x] =
              static_cast<BackpointerScalar>(best_h);
        }
      }
      __syncthreads();
    } else {
      for (int x = thread; x < suffix_count; x += kThreads) {
        float best = CUDART_INF_F;
        int best_h = 0;
        for (int h = 0; h < prefix_count; ++h) {
          const int state = h * suffix_count + x;
          float candidate = grid_emission_v2<DirectDistance, CodebookScalar>(
              target, bank_codebook, bank_norm, state, weight, target_norm);
          if (constrained && (state >> shift) != required_overlap) {
            candidate = CUDART_INF_F;
          }
          if (lower_pair(candidate, h, best, best_h)) {
            best = candidate;
            best_h = h;
          }
        }
        g_previous[x] = best;
        if constexpr (!MidpointOnly) {
          backpointers[pointer_base + bank * suffix_count + x] =
              static_cast<BackpointerScalar>(best_h);
        }
      }
      __syncthreads();
    }
  } else if constexpr (FuseBoundary) {
    for (int x = thread; x < suffix_count; x += kThreads) {
      const int64_t input_base = static_cast<int64_t>(sequence) * bank_suffix_count;
      float best = g_input_all[input_base + x];
      int best_bank = 0;
      #pragma unroll
      for (int previous_bank = 1; previous_bank < bank_count; ++previous_bank) {
        const float candidate = g_input_all[input_base + previous_bank * suffix_count + x];
        if (candidate < best) {
          best = candidate;
          best_bank = previous_bank;
        }
      }
      // Every current-bank CTA needs the same reduced input frontier.  Bank
      // zero alone records the selector for traceback after all CTAs finish.
      g_previous[x] = best;
      if (bank == 0) {
        boundary_banks[
            boundary_base + static_cast<int64_t>(segment_index - 1) * suffix_count + x] =
            static_cast<uint8_t>(best_bank);
      }
    }
    __syncthreads();
  } else {
    for (int x = thread; x < suffix_count; x += kThreads) {
      g_previous[x] = g_input_all[g_base + x];
    }
    __syncthreads();
  }

  const int recurrence_start = segment_index == 0 ? 1 : first_step;
  for (int step = recurrence_start; step < end_step; ++step) {
    const bool boundary = !FuseBoundary && step == first_step && segment_index != 0;
    const int boundary_index = segment_index - 1;
    const float* target = sequences + sequence_base + static_cast<int64_t>(step) * 2;
    const float weight = weighted
        ? step_weights[static_cast<int64_t>(sequence) * 128 + step]
        : 1.0f;
    const float target_norm = __fadd_rn(
        __fmul_rn(target[0], target[0]),
        __fmul_rn(target[1], target[1]));
    if constexpr (shift == 7) {
      const int x = thread & (suffix_count - 1);
      const int first_h = thread >> 9;
      float best = CUDART_INF_F;
      int best_h = first_h;
      for (int h = first_h; h < prefix_count; h += 2) {
        const int state = h * suffix_count + x;
        const int predecessor_suffix = state >> shift;
        float predecessor_cost;
        if constexpr (FuseBoundary) {
          predecessor_cost = g_previous[predecessor_suffix];
        } else {
          const int previous_bank = boundary
              ? static_cast<int>(boundary_banks[
                    boundary_base + static_cast<int64_t>(boundary_index) * suffix_count +
                    predecessor_suffix])
              : bank;
          predecessor_cost = boundary
              ? g_input_all[static_cast<int64_t>(sequence) * bank_suffix_count +
                            previous_bank * suffix_count + predecessor_suffix]
              : g_previous[predecessor_suffix];
        }
        const float candidate = __fadd_rn(
            predecessor_cost,
            grid_emission_v2<DirectDistance, CodebookScalar>(
                target, bank_codebook, bank_norm, state, weight, target_norm));
        if (lower_pair(candidate, h, best, best_h)) {
          best = candidate;
          best_h = h;
        }
      }
      partial_values[thread] = best;
      partial_prefixes[thread] = best_h;
      __syncthreads();
      if (thread < suffix_count) {
        const float other = partial_values[thread + suffix_count];
        const int other_h = partial_prefixes[thread + suffix_count];
        if (lower_pair(other, other_h, best, best_h)) {
          best = other;
          best_h = other_h;
        }
        g_scratch[x] = best;
        if constexpr (!MidpointOnly) {
          backpointers[pointer_base + static_cast<int64_t>(step - first_pointer_step) * bank_suffix_count +
                       bank * suffix_count + x] = static_cast<BackpointerScalar>(best_h);
        } else if (step >= first_pointer_step) {
          backpointers[pointer_base + static_cast<int64_t>(step - first_pointer_step) * bank_suffix_count +
                       bank * suffix_count + x] = static_cast<BackpointerScalar>(best_h);
        }
      }
      __syncthreads();
    } else {
      for (int x = thread; x < suffix_count; x += kThreads) {
        float best = CUDART_INF_F;
        int best_h = 0;
        for (int h = 0; h < prefix_count; ++h) {
          const int state = h * suffix_count + x;
          const int predecessor_suffix = state >> shift;
          float predecessor_cost;
          if constexpr (FuseBoundary) {
            predecessor_cost = g_previous[predecessor_suffix];
          } else {
            const int previous_bank = boundary
                ? static_cast<int>(boundary_banks[
                      boundary_base + static_cast<int64_t>(boundary_index) * suffix_count +
                      predecessor_suffix])
                : bank;
            predecessor_cost = boundary
                ? g_input_all[static_cast<int64_t>(sequence) * bank_suffix_count +
                              previous_bank * suffix_count + predecessor_suffix]
                : g_previous[predecessor_suffix];
          }
          const float candidate = __fadd_rn(
              predecessor_cost,
              grid_emission_v2<DirectDistance, CodebookScalar>(
                  target, bank_codebook, bank_norm, state, weight, target_norm));
          if (lower_pair(candidate, h, best, best_h)) {
            best = candidate;
            best_h = h;
          }
        }
        g_scratch[x] = best;
        if constexpr (!MidpointOnly) {
          backpointers[pointer_base + static_cast<int64_t>(step - first_pointer_step) * bank_suffix_count +
                       bank * suffix_count + x] = static_cast<BackpointerScalar>(best_h);
        } else if (step >= first_pointer_step) {
          backpointers[pointer_base + static_cast<int64_t>(step - first_pointer_step) * bank_suffix_count +
                       bank * suffix_count + x] = static_cast<BackpointerScalar>(best_h);
        }
      }
      __syncthreads();
    }
    float* swap = g_previous;
    g_previous = g_scratch;
    g_scratch = swap;
  }
  for (int x = thread; x < suffix_count; x += kThreads) {
    g_output_all[g_base + x] = g_previous[x];
  }
}

// ---------------------------------------------------------------------------
// Norm-rank contiguous-band exact bound-pruned segmented grid recurrence.
//
// Same thread mapping (one thread per suffix column x, 1024 threads per CTA,
// one CTA per (sequence, bank)), same FP32 candidate arithmetic, same
// frontier/backpointer/boundary layouts, and the same (value, lowest original
// prefix) tie contract as qvq_v2_segment_grid_kernel.  The difference is that
// each thread evaluates only a provably sufficient contiguous norm-band of its
// prefix_count-entry sorted candidate list instead of all of it, and every
// range decision is widened to the warp so each chunk visit is a warp-uniform,
// perfectly coalesced, branch-free straight-line block:
//
//   seed:  the warp range of the previous step's winning chunks (the first
//          step of a segment brackets tn per lane instead).  Evaluating it
//          hands every lane an incumbent value U; seed quality affects only
//          how tight U is, never correctness.
//   walk:  each lane converts U into a provable norm interval; the warp walks
//          outward from the seed range and one all-lanes-agree ballot against
//          the sorted chunk-minimum norms ends each direction for everyone.
//
// Warp-widening only ever ADDS evaluated chunks, so exactness is per-lane.
// Measured on real Qwen3 weight tiles the warp union costs the same as the
// per-lane maximum (bands overlap almost perfectly within a warp).  A column
// whose band degenerates to the full list drops to a straight full scan - the
// baseline body - and re-probes its band every sixteenth step.
//
// Notation for one step: t = (tx, ty), tn = fl(fl(tx*tx) + fl(ty*ty)) is the
// kernel's own FP32 target norm, nu is a state's cached FP32 norm, gp is its
// predecessor frontier cost, and u = 2^-24 bounds the relative error of one
// FP32 round-to-nearest operation.  The reference candidate value is
//   cand = fl(gp + max(fl(fl(tn + nu) - fl(2 * fl(dot))), 0)).
//
// Let floor <= gp be the minimum of the previous frontier over this column's
// predecessor class and U the lane's evaluated candidate value from the seed
// range.  If a state can still win or tie then cand <= U.  Chaining
// |fl(y) - y| <= u|y| per operation with Cauchy-Schwarz on the exact reals
// (dot <= sqrt(TN * NU), TN <= tn/(1-u)^2, NU <= nu/(1-u)^2) yields, for
// constants c1, c2 < 5,
//
//   (sqrt(tn) - sqrt(nu))^2 <= (U - floor)(1 + c1*u) + c2*u*(tn + sqrt(tn*nu)).
//
// The code widens every coefficient to kNormRankEps = 2^-19 = 32u with
// outward-directed rounding:
//
//   slack  = max(U*(1+eps)[ru] - floor, 0)          >= (U - floor)(1 + c1*u)
//   radius = sqrt_ru(slack*(1+eps)[ru] + tn*eps[ru])
//   root   = sqrt_rn(tn)
//   root_lo/root_hi = adjacent FP32 values below/above root
//   band   = [ (root_lo - radius)^2_rd , (root_hi*(1+eps)[ru] + radius)^2_ru ]
//
// The eps*slack term inside the square root covers every multiplicative error
// (32u versus the required < 5u) and the eps*tn term covers every additive
// error including the self-referential sqrt(tn*nu) cross term (resolving the
// quadratic in r = |sqrt(tn) - sqrt(nu)| consumes < 9u*tn); the remaining
// >= 3x margin enters under the square root, so the band widens by an
// O(sqrt(u)) sliver that admits essentially no extra survivors.  Nothing here
// needs directed-rounding square roots: correctly rounded sqrt_rn lies within
// its two adjacent representable FP32 values, so those neighbors are an exact
// outward enclosure of the real square root while requiring only one sqrt.
// Nothing here relies on the FP32 bound being monotone in nu: the band is contiguous
// because the *table* is sorted by the very same cached FP32 norms, and every
// rounding decision above only widens the interval.  Non-finite targets or
// frontiers collapse the lane's band to the full list, which is exactly the
// baseline scan.
//
// The chunk skip test uses only the sorted per-chunk minimum norms lo[]: a
// chunk is skippable above the band iff lo[c] > high (and then so is every
// later chunk), and below the band iff its maximum norm < low, for which
// lo[c+1] < low is a sufficient (sortedness: max of chunk c <= lo[c+1]) and
// cheaper test (and then every earlier chunk is skippable too).
// ---------------------------------------------------------------------------
constexpr float kNormRankEps = 1.9073486328125e-06f;         // 2^-19, exact in FP32
constexpr float kNormRankRelax = 1.0000019073486328125f;     // 1 + 2^-19, exact in FP32

// Telemetry is opt-in through GPTQMODEL_QVQ_TELEMETRY.  Each CTA contributes
// one pair of 64-bit atomics after reducing its per-thread totals in shared
// memory, keeping observability out of the inner candidate loop.
__device__ unsigned long long g_norm_rank_candidates_evaluated = 0;
__device__ unsigned long long g_norm_rank_candidates_possible = 0;

template <int Shift, int BankCount, int SegmentSteps, int ChunkWidth, int BlockThreads>
__global__ __launch_bounds__(BlockThreads, Shift >= 6 ? 1 : 2)
void qvq_v2_segment_grid_norm_rank_kernel(
    const float* __restrict__ sequences,
    const uint64_t* __restrict__ baseline_records,
    const uint64_t* __restrict__ sorted_records,
    const typename NormRankPrefixPack<ChunkWidth>::type* __restrict__ chunk_prefixes,
    const float* __restrict__ chunk_low_norms,
    const int64_t* __restrict__ overlap,
    const float* __restrict__ g_input_all,
    float* __restrict__ g_output_all,
    uint8_t* __restrict__ backpointers,
    uint8_t* __restrict__ boundary_banks,
    int batch,
    int family_batch,
    int segment_index,
    bool constrained,
    bool collect_telemetry) {
  using PackType = typename NormRankPrefixPack<ChunkWidth>::type;
  constexpr int shift = Shift;
  constexpr int prefix_count = 1 << shift;
  constexpr int suffix_count = 1 << (16 - shift);
  constexpr int bank_count = BankCount;
  constexpr int segment_steps = SegmentSteps;
  constexpr int segment_count = 128 / segment_steps;
  constexpr int bank_suffix_count = bank_count * suffix_count;
  constexpr int chunk_count = prefix_count / ChunkWidth;
  constexpr int column_iterations = suffix_count / BlockThreads;
  // Predecessor of state h * suffix_count + x is h * group_count + (x >> shift):
  // exactly the suffixes congruent to (x >> shift) modulo group_count.
  constexpr int group_shift = 16 - 2 * shift;
  constexpr int group_count = 1 << group_shift;
  // Skew the shared frontier by one slot per predecessor group: the in-chunk
  // reads go to prefix * (group_count + 1) + group, an odd stride that spreads
  // the warp's distinct prefixes over distinct shared banks (the natural
  // power-of-two stride would serialize them onto one).
  constexpr int frontier_stride = suffix_count + prefix_count;
  constexpr unsigned kInfBits = 0x7f800000u;
  constexpr unsigned kFullMask = 0xffffffffu;
  static_assert(group_count * prefix_count == suffix_count, "group decomposition");
  static_assert(prefix_count % ChunkWidth == 0, "chunk width must divide the prefix count");
  static_assert(ChunkWidth % 2 == 0, "chunk records are loaded in aligned 16-byte pairs");
  static_assert(suffix_count % BlockThreads == 0, "warps must stay fully populated");

  const int flat_block = static_cast<int>(blockIdx.x);
  const int sequence = flat_block / bank_count;
  const int bank = flat_block - sequence * bank_count;
  if (sequence >= batch) {
    return;
  }
  const int thread = static_cast<int>(threadIdx.x);
  unsigned long long candidates_evaluated = 0;
  unsigned long long candidates_possible = 0;
  const int first_step = segment_index * segment_steps;
  // Step 127 stays folded into the final reduction, as in the baseline.
  const int end_step = min(first_step + segment_steps, 127);
  const int64_t sequence_base = static_cast<int64_t>(sequence) * 128 * 2;
  const int64_t g_base = static_cast<int64_t>(sequence) * bank_suffix_count + bank * suffix_count;
  const int64_t pointer_base = static_cast<int64_t>(sequence) * 127 * bank_suffix_count +
      static_cast<int64_t>(bank) * suffix_count;
  const int64_t boundary_base = static_cast<int64_t>(sequence) * (segment_count - 1) * suffix_count;

  extern __shared__ float shared_frontiers[];
  float* g_previous = shared_frontiers;
  float* g_scratch = shared_frontiers + frontier_stride;
  // Three rotating per-group minimum buffers: one read this step, one filled
  // for the next step, one already cleared for the step after.  This keeps the
  // recurrence at exactly one __syncthreads() per step, like the baseline.
  unsigned* group_min_all = reinterpret_cast<unsigned*>(shared_frontiers + 2 * frontier_stride);
  unsigned* group_min = group_min_all;
  unsigned* group_min_next = group_min_all + group_count;
  unsigned* group_min_spare = group_min_all + 2 * group_count;

  const int family = family_batch == 0 ? 0 : sequence / family_batch;
  const int physical_bank = family * bank_count + bank;
  const uint64_t* bank_baseline = baseline_records + static_cast<int64_t>(physical_bank) * kStateCount;
  const uint64_t* bank_records = sorted_records + static_cast<int64_t>(physical_bank) * kStateCount;
  const PackType* bank_prefixes =
      chunk_prefixes + static_cast<int64_t>(physical_bank) * chunk_count * suffix_count;
  const float* bank_low = chunk_low_norms +
      static_cast<int64_t>(physical_bank) * chunk_count * suffix_count;

  for (int slot = thread; slot < 3 * group_count; slot += BlockThreads) {
    group_min_all[slot] = kInfBits;
  }
  __syncthreads();
  if (segment_index == 0) {
    // Step 0 has no predecessor. A zero frontier makes __fadd_rn(0, e) == e
    // for the non-negative emission e, so the shared step body is bit-identical
    // to the baseline's emission-only prologue. Tail-biting's constrained pass
    // admits exactly the predecessor suffix named by overlap[sequence]; all
    // other prefixes begin at infinity, matching the baseline state mask.
    const int required_overlap = constrained ? static_cast<int>(overlap[sequence]) : 0;
    for (int x = thread; x < suffix_count; x += BlockThreads) {
      g_previous[x + (x >> group_shift)] =
          !constrained || x == required_overlap ? 0.0f : CUDART_INF_F;
    }
    if (thread < group_count) {
      group_min[thread] =
          !constrained || thread == (required_overlap & (group_count - 1)) ? 0u : kInfBits;
    }
  } else {
    const int64_t input_base = static_cast<int64_t>(sequence) * bank_suffix_count;
    for (int x = thread; x < suffix_count; x += BlockThreads) {
      float best = g_input_all[input_base + x];
      int best_bank = 0;
      #pragma unroll
      for (int previous_bank = 1; previous_bank < bank_count; ++previous_bank) {
        const float candidate = g_input_all[input_base + previous_bank * suffix_count + x];
        // Strict comparison preserves bank-zero/lowest-bank tie precedence.
        if (candidate < best) {
          best = candidate;
          best_bank = previous_bank;
        }
      }
      g_previous[x + (x >> group_shift)] = best;
      if (bank == 0) {
        boundary_banks[boundary_base +
                       static_cast<int64_t>(segment_index - 1) * suffix_count + x] =
            static_cast<uint8_t>(best_bank);
      }
      atomicMin(&group_min[x & (group_count - 1)], __float_as_uint(best));
    }
  }
  __syncthreads();

  // Per-column carried state: last step's winning chunk (next step's seed) and
  // whether the last band probe degenerated to the full list.
  int winner_chunk[column_iterations];
  bool degenerate[column_iterations];
  #pragma unroll
  for (int column = 0; column < column_iterations; ++column) {
    winner_chunk[column] = 0;
    degenerate[column] = false;
  }

  const int recurrence_start = segment_index == 0 ? 0 : first_step;
  for (int step = recurrence_start; step < end_step; ++step) {
    for (int slot = thread; slot < group_count; slot += BlockThreads) {
      group_min_spare[slot] = kInfBits;
    }
    const bool reseed = step == recurrence_start;
    const bool probe = reseed || (step & 15) == 0;
    const float* target = sequences + sequence_base + static_cast<int64_t>(step) * 2;
    const float tx = target[0];
    const float ty = target[1];
    const float tn = __fadd_rn(__fmul_rn(tx, tx), __fmul_rn(ty, ty));
    const float root = __fsqrt_rn(tn);
    const unsigned root_bits = __float_as_uint(root);
    const float root_low = root_bits == 0u ? 0.0f : __uint_as_float(root_bits - 1u);
    const float root_upper = __uint_as_float(root_bits + 1u);
    const float root_high = __fmul_ru(root_upper, kNormRankRelax);
    const float target_slack = __fmul_ru(tn, kNormRankEps);
    const int64_t step_pointer_base =
        pointer_base + static_cast<int64_t>(step) * bank_suffix_count;
    #pragma unroll
    for (int column = 0; column < column_iterations; ++column) {
      if (collect_telemetry) {
        candidates_possible += prefix_count;
      }
      const int x = thread + column * BlockThreads;
      const int group = x >> shift;
      const float floor_cost = __uint_as_float(group_min[group]);

      float best = CUDART_INF_F;
      int best_prefix = 0;
      int best_chunk = 0;
      // One warp-uniform straight-line block of ChunkWidth candidates.
      const auto evaluate_chunk = [&](int chunk) {
        if (collect_telemetry) {
          candidates_evaluated += ChunkWidth;
        }
        const PackType packed = bank_prefixes[static_cast<int64_t>(chunk) * suffix_count + x];
        const uint64_t* lane_records =
            bank_records + (static_cast<int64_t>(chunk) * suffix_count + x) * ChunkWidth;
        #pragma unroll
        for (int pair = 0; pair < ChunkWidth / 2; ++pair) {
          const ulonglong2 records =
              *reinterpret_cast<const ulonglong2*>(lane_records + 2 * pair);
          #pragma unroll
          for (int half_slot = 0; half_slot < 2; ++half_slot) {
            const uint64_t record = half_slot == 0 ? records.x : records.y;
            const int prefix = static_cast<int>(
                (packed >> (8 * (2 * pair + half_slot))) & 0xFFu);
            const uint32_t code_bits = static_cast<uint32_t>(record);
            const float2 code = __half22float2(*reinterpret_cast<const half2*>(&code_bits));
            const float norm = __uint_as_float(static_cast<uint32_t>(record >> 32));
            float dot = 0.0f;
            dot = __fmaf_rn(tx, code.x, dot);
            dot = __fmaf_rn(ty, code.y, dot);
            const float distance = __fsub_rn(__fadd_rn(tn, norm), __fmul_rn(2.0f, dot));
            const float candidate = __fadd_rn(
                g_previous[prefix * (group_count + 1) + group], fmaxf(distance, 0.0f));
            if (lower_pair(candidate, prefix, best, best_prefix)) {
              best = candidate;
              best_prefix = prefix;
              best_chunk = chunk;
            }
          }
        }
      };

      if (degenerate[column] && !probe) {
        if (collect_telemetry) {
          candidates_evaluated += prefix_count;
        }
        // The last band probe covered most of the list: until the next probe,
        // run the baseline's own prefix-order scan over the original packed
        // records - the identical straight-line body ptxas fully unrolls, so
        // a degenerate column pays no band machinery at all.  The stale seed
        // is refreshed by the next probe's walk, which stays exact under any
        // seed.
        #pragma unroll
        for (int h = 0; h < prefix_count; ++h) {
          const uint64_t record = bank_baseline[h * suffix_count + x];
          const uint32_t code_bits = static_cast<uint32_t>(record);
          const float2 code = __half22float2(*reinterpret_cast<const half2*>(&code_bits));
          const float norm = __uint_as_float(static_cast<uint32_t>(record >> 32));
          float dot = 0.0f;
          dot = __fmaf_rn(tx, code.x, dot);
          dot = __fmaf_rn(ty, code.y, dot);
          const float distance = __fsub_rn(__fadd_rn(tn, norm), __fmul_rn(2.0f, dot));
          const float candidate = __fadd_rn(
              g_previous[h * (group_count + 1) + group], fmaxf(distance, 0.0f));
          if (lower_pair(candidate, h, best, best_prefix)) {
            best = candidate;
            best_prefix = h;
          }
        }
      } else {
        // Seed: last step's winning chunks, or a per-lane bracket of tn (the
        // unconstrained minimiser of the distance bound) on a segment's first
        // step.  Any seed is correct; a good one makes U tight.
        int seed = winner_chunk[column];
        if (reseed) {
          int rank_tn = 0;
          #pragma unroll
          for (int chunk = 0; chunk < chunk_count; ++chunk) {
            rank_tn += (bank_low[static_cast<int64_t>(chunk) * suffix_count + x] <= tn) ? 1 : 0;
          }
          seed = max(rank_tn - 1, 0);
        }
        const int seed_first = __reduce_min_sync(kFullMask, seed);
        const int seed_last = __reduce_max_sync(kFullMask, seed);
        for (int chunk = seed_first; chunk <= seed_last; ++chunk) {
          evaluate_chunk(chunk);
        }

        // Per-lane provable norm band, walked outward warp-uniformly.  One
        // ballot per candidate chunk: the sorted minima let the whole warp
        // stop at the first chunk no lane can use.
        const float slack =
            fmaxf(__fsub_ru(__fmul_ru(best, kNormRankRelax), floor_cost), 0.0f);
        // A correctly rounded square root under-estimates the real value by at
        // most one ulp, so scaling it up by 1 + 32u keeps the radius a strict
        // upper bound while avoiding the long directed-rounding sqrt sequence.
        const float radius = __fmul_ru(
            __fsqrt_rn(__fadd_ru(__fmul_ru(slack, kNormRankRelax), target_slack)),
            kNormRankRelax);
        const float inner = __fsub_rd(root_low, radius);
        float norm_low = inner > 0.0f ? __fmul_rd(inner, inner) : 0.0f;
        const float outer = __fadd_ru(root_high, radius);
        float norm_high = __fmul_ru(outer, outer);
        if (!(norm_high >= norm_low)) {  // non-finite target/frontier: full scan
          norm_low = -CUDART_INF_F;
          norm_high = CUDART_INF_F;
        }
        int walk_last = seed_last;
        for (int chunk = seed_last + 1; chunk < chunk_count; ++chunk) {
          const float low_norm = bank_low[static_cast<int64_t>(chunk) * suffix_count + x];
          if (__ballot_sync(kFullMask, low_norm <= norm_high) == 0u) {
            break;
          }
          evaluate_chunk(chunk);
          walk_last = chunk;
        }
        int walk_first = seed_first;
        for (int chunk = seed_first - 1; chunk >= 0; --chunk) {
          const float next_low =
              bank_low[(static_cast<int64_t>(chunk) + 1) * suffix_count + x];
          if (__ballot_sync(kFullMask, next_low >= norm_low) == 0u) {
            break;
          }
          evaluate_chunk(chunk);
          walk_first = chunk;
        }
        // Past ~5/8 of the list the fully unrolled full scan beats the
        // dynamically bounded band walk, so wide bands drop to it until the
        // next probe.  Real weight tiles measure 12-37% wide and never trip
        // this; it bounds the worst case on adversarial/random sequences.
        degenerate[column] = 8 * (walk_last - walk_first + 1) >= 5 * chunk_count;
        winner_chunk[column] = best_chunk;
      }

      g_scratch[x + (x >> group_shift)] = best;
      backpointers[step_pointer_base + x] = static_cast<uint8_t>(best_prefix);
      atomicMin(&group_min_next[x & (group_count - 1)], __float_as_uint(best));
    }
    __syncthreads();
    float* swap_frontier = g_previous;
    g_previous = g_scratch;
    g_scratch = swap_frontier;
    unsigned* rotate = group_min;
    group_min = group_min_next;
    group_min_next = group_min_spare;
    group_min_spare = rotate;
  }
  for (int x = thread; x < suffix_count; x += BlockThreads) {
    g_output_all[g_base + x] = g_previous[x + (x >> group_shift)];
  }
  if (collect_telemetry) {
    // The recurrence no longer needs shared frontier storage.  Reduce within
    // each warp in registers, then let warp zero reduce the warp totals.  This
    // preserves the exact uint64 totals and two atomics per CTA while replacing
    // the former log2(BlockThreads) barrier tree with one block barrier.
    constexpr int warp_count = BlockThreads / 32;
    const int lane = thread & 31;
    const int warp = thread >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      candidates_evaluated += __shfl_down_sync(kFullMask, candidates_evaluated, offset);
      candidates_possible += __shfl_down_sync(kFullMask, candidates_possible, offset);
    }
    auto* reduction = reinterpret_cast<unsigned long long*>(shared_frontiers);
    if (lane == 0) {
      reduction[warp] = candidates_evaluated;
      reduction[warp_count + warp] = candidates_possible;
    }
    __syncthreads();
    if (warp == 0) {
      unsigned long long block_evaluated = lane < warp_count ? reduction[lane] : 0;
      unsigned long long block_possible = lane < warp_count ? reduction[warp_count + lane] : 0;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        block_evaluated += __shfl_down_sync(kFullMask, block_evaluated, offset);
        block_possible += __shfl_down_sync(kFullMask, block_possible, offset);
      }
      if (lane == 0) {
        atomicAdd(&g_norm_rank_candidates_evaluated, block_evaluated);
        atomicAdd(&g_norm_rank_candidates_possible, block_possible);
      }
    }
  }
}

// W2.5 B2-P32 is badly underfilled by the one-CTA-per-bank recurrence: a
// singleton tile launches only two 1024-thread CTAs on a 124-SM local sm_80
// device.  This exact cooperative variant partitions the 2048 suffixes into
// eight 256-thread CTAs per bank.  All predecessor candidates for one suffix
// remain in one thread and are visited in ascending prefix order, preserving
// the reference arithmetic and lowest-prefix tie precedence.  Grid barriers
// make the global ping-pong frontiers safe between dependent recurrence steps.
template <int BlockThreads, typename CodebookScalar, typename BackpointerScalar>
__global__ __launch_bounds__(BlockThreads) void qvq_v2_segment_w25_cooperative_kernel(
    const float* __restrict__ sequences,
    const CodebookScalar* __restrict__ codebooks,
    const float* __restrict__ codebook_norms,
    const int64_t* __restrict__ overlap,
    const float* __restrict__ step_weights,
    float* __restrict__ costs_a,
    float* __restrict__ costs_b,
    BackpointerScalar* __restrict__ backpointers,
    uint8_t* __restrict__ boundary_banks,
    int batch,
    int family_batch,
    int segment_index,
    bool constrained,
    bool weighted) {
  constexpr int shift = 5;
  constexpr int bank_count = 2;
  constexpr int suffix_count = 1 << (16 - shift);
  constexpr int prefix_count = 1 << shift;
  constexpr int segment_steps = 16;
  constexpr int segment_count = 128 / segment_steps;
  constexpr int partitions = suffix_count / BlockThreads;
  constexpr int bank_suffix_count = bank_count * suffix_count;
  const int flat_block = static_cast<int>(blockIdx.x);
  const int partition = flat_block % partitions;
  const int bank_sequence = flat_block / partitions;
  const int bank = bank_sequence % bank_count;
  const int sequence = bank_sequence / bank_count;
  const int x = partition * BlockThreads + static_cast<int>(threadIdx.x);
  if (sequence >= batch) {
    return;
  }

  cg::grid_group grid = cg::this_grid();
  const int first_step = segment_index * segment_steps;
  const int end_step = min(first_step + segment_steps, 127);
  const int required_overlap = constrained ? static_cast<int>(overlap[sequence]) : 0;
  const int family = family_batch == 0 ? 0 : sequence / family_batch;
  const int physical_bank = family * bank_count + bank;
  const CodebookScalar* bank_codebook =
      codebooks + static_cast<int64_t>(physical_bank) * kStateCount * 2;
  const float* bank_norm = codebook_norms + static_cast<int64_t>(physical_bank) * kStateCount *
      (std::is_same_v<CodebookScalar, half> ? 2 : 1);
  const int64_t sequence_base = static_cast<int64_t>(sequence) * 128 * 2;
  const int64_t g_base = static_cast<int64_t>(sequence) * bank_suffix_count + bank * suffix_count;
  const int64_t pointer_base = static_cast<int64_t>(sequence) * 127 * bank_suffix_count;
  const int64_t boundary_base = static_cast<int64_t>(sequence) * (segment_count - 1) * suffix_count;

  if (segment_index == 0) {
    const float* target = sequences + sequence_base;
    const float weight = weighted ? step_weights[static_cast<int64_t>(sequence) * 128] : 1.0f;
    const float target_norm = __fadd_rn(
        __fmul_rn(target[0], target[0]),
        __fmul_rn(target[1], target[1]));
    float best = CUDART_INF_F;
    int best_h = 0;
    #pragma unroll
    for (int h = 0; h < prefix_count; ++h) {
      const int state = h * suffix_count + x;
      float candidate = emission<2, CodebookScalar>(
          target, bank_codebook, bank_norm, state, weight, target_norm);
      if (constrained && (state >> shift) != required_overlap) {
        candidate = CUDART_INF_F;
      }
      if (lower_pair(candidate, h, best, best_h)) {
        best = candidate;
        best_h = h;
      }
    }
    costs_a[g_base + x] = best;
    backpointers[pointer_base + bank * suffix_count + x] =
        static_cast<BackpointerScalar>(best_h);
    grid.sync();
  } else {
    const float* previous = ((first_step - 1) & 1) == 0 ? costs_a : costs_b;
    if (bank == 0) {
      const int64_t input_base = static_cast<int64_t>(sequence) * bank_suffix_count;
      float best = previous[input_base + x];
      int best_bank = 0;
      const float candidate = previous[input_base + suffix_count + x];
      if (candidate < best) {
        best_bank = 1;
      }
      boundary_banks[
          boundary_base + static_cast<int64_t>(segment_index - 1) * suffix_count + x] =
          static_cast<uint8_t>(best_bank);
    }
    grid.sync();
  }

  const int recurrence_start = segment_index == 0 ? 1 : first_step;
  for (int step = recurrence_start; step < end_step; ++step) {
    const float* previous = ((step - 1) & 1) == 0 ? costs_a : costs_b;
    float* current = (step & 1) == 0 ? costs_a : costs_b;
    const bool boundary = step == first_step && segment_index != 0;
    const float* target = sequences + sequence_base + static_cast<int64_t>(step) * 2;
    const float weight = weighted
        ? step_weights[static_cast<int64_t>(sequence) * 128 + step]
        : 1.0f;
    const float target_norm = __fadd_rn(
        __fmul_rn(target[0], target[0]),
        __fmul_rn(target[1], target[1]));
    float best = CUDART_INF_F;
    int best_h = 0;
    #pragma unroll
    for (int h = 0; h < prefix_count; ++h) {
      const int state = h * suffix_count + x;
      const int predecessor_suffix = state >> shift;
      const int previous_bank = boundary
          ? static_cast<int>(boundary_banks[
                boundary_base + static_cast<int64_t>(segment_index - 1) * suffix_count +
                predecessor_suffix])
          : bank;
      const float predecessor_cost = previous[
          static_cast<int64_t>(sequence) * bank_suffix_count +
          previous_bank * suffix_count + predecessor_suffix];
      const float candidate = __fadd_rn(
          predecessor_cost,
          emission<2, CodebookScalar>(
              target, bank_codebook, bank_norm, state, weight, target_norm));
      if (lower_pair(candidate, h, best, best_h)) {
        best = candidate;
        best_h = h;
      }
    }
    current[g_base + x] = best;
    backpointers[
        pointer_base + static_cast<int64_t>(step) * bank_suffix_count +
        bank * suffix_count + x] = static_cast<BackpointerScalar>(best_h);
    grid.sync();
  }
}

template <
    int Shift,
    int BankCount,
    int SegmentSteps,
    bool MidpointOnly,
    typename CodebookScalar,
    typename BackpointerScalar>
__global__ __launch_bounds__(kThreads) void qvq_v2_segment_grid_finalize_kernel(
    const float* __restrict__ sequences,
    const CodebookScalar* __restrict__ codebooks,
    const float* __restrict__ codebook_norms,
    const int64_t* __restrict__ overlap,
    const float* __restrict__ step_weights,
    const float* __restrict__ g_previous,
    const BackpointerScalar* __restrict__ backpointers,
    const uint8_t* __restrict__ boundary_banks,
    int64_t* __restrict__ states,
    uint8_t* __restrict__ segment_bank_ids,
    float* __restrict__ squared_error,
    int batch,
    int family_batch,
    bool constrained,
    bool weighted) {
  constexpr int shift = Shift;
  constexpr int suffix_count = 1 << (16 - shift);
  constexpr int overlap_mask = suffix_count - 1;
  constexpr int bank_count = BankCount;
  constexpr int segment_steps = SegmentSteps;
  constexpr int segment_count = 128 / segment_steps;
  constexpr int bank_suffix_count = bank_count * suffix_count;
  const int sequence = static_cast<int>(blockIdx.x);
  if (sequence >= batch) {
    return;
  }
  const int thread = static_cast<int>(threadIdx.x);
  const int required_overlap = constrained ? static_cast<int>(overlap[sequence]) : 0;
  const int64_t sequence_base = static_cast<int64_t>(sequence) * 128 * 2;
  const int64_t g_base = static_cast<int64_t>(sequence) * bank_suffix_count;
  constexpr int first_pointer_step = MidpointOnly ? 63 : 0;
  constexpr int stored_pointer_steps = 127 - first_pointer_step;
  const int64_t pointer_base =
      static_cast<int64_t>(sequence) * stored_pointer_steps * bank_suffix_count;
  const int64_t boundary_base = static_cast<int64_t>(sequence) * (segment_count - 1) * suffix_count;
  const int64_t state_base = static_cast<int64_t>(sequence) * 128;
  const int64_t selector_base = static_cast<int64_t>(sequence) * segment_count;
  const float* target = sequences + sequence_base + 127 * 2;
  const float weight = weighted ? step_weights[static_cast<int64_t>(sequence) * 128 + 127] : 1.0f;
  const float target_norm = __fadd_rn(
      __fmul_rn(target[0], target[0]),
      __fmul_rn(target[1], target[1]));

  float best = CUDART_INF_F;
  int best_flat = bank_count * kStateCount;
  for (int flat = thread; flat < bank_count * kStateCount; flat += kThreads) {
    const int bank = flat / kStateCount;
    const int state = flat - bank * kStateCount;
    const int family = family_batch == 0 ? 0 : sequence / family_batch;
    const int physical_bank = family * bank_count + bank;
    const CodebookScalar* bank_codebook =
        codebooks + static_cast<int64_t>(physical_bank) * kStateCount * 2;
    const float* bank_norm = codebook_norms + static_cast<int64_t>(physical_bank) * kStateCount *
        (std::is_same_v<CodebookScalar, half> ? 2 : 1);
    float candidate = __fadd_rn(
        g_previous[g_base + bank * suffix_count + (state >> shift)],
        emission<2, CodebookScalar>(
            target, bank_codebook, bank_norm, state, weight, target_norm));
    if (constrained && (state & overlap_mask) != required_overlap) {
      candidate = CUDART_INF_F;
    }
    if (lower_pair(candidate, flat, best, best_flat)) {
      best = candidate;
      best_flat = flat;
    }
  }
  block_argmin(best, best_flat);

  if (thread == 0) {
    int current_bank = best_flat / kStateCount;
    int current_state = best_flat - current_bank * kStateCount;
    if constexpr (!MidpointOnly) {
      squared_error[sequence] = best;
      states[state_base + 127] = current_state;
      segment_bank_ids[selector_base + segment_count - 1] = static_cast<uint8_t>(current_bank);
    }
    constexpr int final_traceback_step = MidpointOnly ? 63 : 0;
    for (int step = 127; step > final_traceback_step; --step) {
      const int predecessor_suffix = current_state >> shift;
      if (step % segment_steps == 0) {
        const int boundary_index = step / segment_steps - 1;
        current_bank = static_cast<int>(boundary_banks[
            boundary_base + static_cast<int64_t>(boundary_index) * suffix_count + predecessor_suffix]);
        if constexpr (!MidpointOnly) {
          segment_bank_ids[selector_base + boundary_index] = static_cast<uint8_t>(current_bank);
        }
      }
      const int prefix = static_cast<int>(backpointers[
          pointer_base + static_cast<int64_t>(step - 1 - first_pointer_step) * bank_suffix_count +
          current_bank * suffix_count + predecessor_suffix]);
      current_state = prefix * suffix_count + predecessor_suffix;
      if constexpr (!MidpointOnly) {
        states[state_base + step - 1] = current_state;
      }
    }
    if constexpr (MidpointOnly) {
      states[sequence] = current_state & overlap_mask;
    }
  }
}

template <int VectorSize>
std::tuple<at::Tensor, at::Tensor> qvq_viterbi_cuda_impl(
    const at::Tensor& sequences,
    const at::Tensor& codebook,
    int64_t transition_bits,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights,
    int64_t bank_count = 1,
    bool validate_values = true) {
  TORCH_CHECK(sequences.is_cuda() && codebook.is_cuda(), "sequences and codebook must be CUDA tensors");
  TORCH_CHECK((sequences.dim() == 3 && sequences.size(2) == VectorSize) ||
                  (bank_count > 1 && sequences.dim() == 4 && sequences.size(3) == VectorSize),
              "sequences have an invalid vector size");
  TORCH_CHECK(bank_count >= 1, "bank_count must be positive");
  if (bank_count > 1 && sequences.dim() == 4) {
    TORCH_CHECK(sequences.size(0) == bank_count,
                "bank-specific sequences must have shape [bank_count, batch, steps, vector_size]");
  }
  TORCH_CHECK(
      (bank_count == 1 && codebook.sizes() == at::IntArrayRef({kStateCount, VectorSize})) ||
          (bank_count > 1 && codebook.dim() == 3 && codebook.size(0) == bank_count &&
           codebook.size(1) == kStateCount && codebook.size(2) == VectorSize),
      "codebook has an invalid banked vector size");
  TORCH_CHECK(sequences.scalar_type() == at::kFloat &&
                  (codebook.scalar_type() == at::kHalf || codebook.scalar_type() == at::kFloat),
              "sequences must use float32 and codebook must use float16 or float32");
  TORCH_CHECK(sequences.device() == codebook.device(), "sequences and codebook must share one CUDA device");
  TORCH_CHECK(sequences.is_contiguous() && codebook.is_contiguous(),
              "sequences and codebook must be contiguous");
  const uintptr_t sequence_alignment = VectorSize == 4 ? 16 : 4;
  const uintptr_t codebook_alignment = VectorSize == 4 && codebook.scalar_type() == at::kFloat ? 16 : 4;
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(sequences.data_ptr()) % sequence_alignment == 0 &&
          reinterpret_cast<uintptr_t>(codebook.data_ptr()) % codebook_alignment == 0,
      "sequences and codebook do not satisfy the native vector-load alignment contract");
  TORCH_CHECK(transition_bits >= 2 && transition_bits <= 16,
              "transition_bits must be in [2, 16]");
  if constexpr (VectorSize == 4) {
    TORCH_CHECK(transition_bits == 4 || transition_bits == 6 || transition_bits == 8 ||
                    transition_bits == 10 || transition_bits == 12 || transition_bits == 14 ||
                    transition_bits == 16,
                "V4 transition_bits must be one of 4, 6, 8, 10, 12, 14, 16");
  }
  if (validate_values) {
    TORCH_CHECK(at::isfinite(sequences).all().item<bool>() &&
                    at::isfinite(codebook).all().item<bool>(),
                "sequences and codebook must be finite");
  }
  const bool bank_specific_sequences = sequences.dim() == 4;
  const bool constrained = overlap.has_value();
  const int batch = static_cast<int>(sequences.size(bank_specific_sequences ? 1 : 0));
  const int steps = static_cast<int>(sequences.size(bank_specific_sequences ? 2 : 1));
  TORCH_CHECK(batch > 0 && steps > 0,
              "sequences must contain at least one batch and one vector");
  TORCH_CHECK(batch <= std::numeric_limits<int>::max() &&
                  steps <= std::numeric_limits<int>::max(),
              "Viterbi dimensions exceed the int32 kernel limit");
  const int launch_batch = batch * static_cast<int>(bank_count);
  const at::Tensor overlap_tensor = constrained ? *overlap : at::Tensor();
  if (constrained) {
    TORCH_CHECK(overlap_tensor.is_cuda() && overlap_tensor.device() == sequences.device(),
                "overlap must share the sequence CUDA device");
    TORCH_CHECK(overlap_tensor.scalar_type() == at::kLong && overlap_tensor.is_contiguous(),
                "overlap must be contiguous int64");
    TORCH_CHECK(overlap_tensor.dim() == 1 && overlap_tensor.size(0) == launch_batch,
                "overlap must have shape [bank_count * batch]");
    const int64_t overlap_limit = int64_t{1} << (16 - transition_bits);
    if (validate_values) {
      TORCH_CHECK(overlap_tensor.ge(0).all().item<bool>() &&
                      overlap_tensor.lt(overlap_limit).all().item<bool>(),
                  "overlap values are outside the transition width");
    }
  }
  const bool weighted = step_weights.has_value();
  const at::Tensor step_weights_tensor = weighted ? *step_weights : at::Tensor();
  if (weighted) {
    TORCH_CHECK(step_weights_tensor.is_cuda() && step_weights_tensor.device() == sequences.device(),
                "step_weights must share the sequence CUDA device");
    TORCH_CHECK(step_weights_tensor.scalar_type() == at::kFloat && step_weights_tensor.is_contiguous(),
                "step_weights must be contiguous float32");
    TORCH_CHECK(step_weights_tensor.dim() == 2 && step_weights_tensor.size(0) == batch &&
                    step_weights_tensor.size(1) == steps,
                "step_weights must have shape [batch, steps]");
    if (validate_values) {
      TORCH_CHECK(at::isfinite(step_weights_tensor).all().item<bool>() &&
                      step_weights_tensor.ge(0).all().item<bool>(),
                  "step_weights must be finite and nonnegative");
    }
  }
  // The emission uses the expanded squared-distance identity in FP32.  A
  // finite input can still make ||x||^2 or ||c||^2 overflow, after which
  // inf - inf becomes NaN and fmaxf can silently turn every state into a
  // false zero-cost tie.  The recurrence also accumulates every weighted
  // step, so include the worst-case number and weight of terms.  Reject that
  // domain at the native boundary instead of clamping and corrupting state
  // selection.
  if (validate_values) {
    const double maximum_weight = weighted ? step_weights_tensor.abs().amax().item<double>() : 1.0;
    const double accumulation_terms = std::max(1.0, static_cast<double>(steps) * maximum_weight);
    const double safe_magnitude = std::sqrt(static_cast<double>(std::numeric_limits<float>::max()) /
                                             accumulation_terms) /
        (2.0 * std::sqrt(static_cast<double>(VectorSize)));
    TORCH_CHECK(
        sequences.abs().amax().item<double>() <= safe_magnitude &&
            codebook.abs().amax().item<double>() <= safe_magnitude,
        "sequences and codebook magnitudes are too large for finite FP32 squared-distance arithmetic");
  }

  const c10::cuda::CUDAGuard device_guard(sequences.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, sequences.get_device()));
  TORCH_CHECK(properties.major >= 8, "QVQ CUDA Viterbi requires compute capability >= 8.0");

  const int shift = static_cast<int>(transition_bits);
  const int suffix_count = kStateCount >> shift;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(sequences.get_device());
  at::Tensor codebook_norm;
  // At E=16 each state is a complete transition: there is no predecessor
  // recurrence or tail constraint. Evaluate all steps independently.
  if (shift == 16) {
    at::Tensor states = at::empty({launch_batch, steps}, sequences.options().dtype(at::kLong));
    at::Tensor step_error = at::empty({launch_batch, steps}, sequences.options());
    if (codebook.scalar_type() == at::kHalf) {
      codebook_norm = bank_count == 1
          ? cached_codebook_norm<VectorSize, half>(codebook, stream)
          : cached_banked_codebook_norm<VectorSize, half>(codebook, stream);
      qvq_memoryless_kernel<VectorSize, half><<<launch_batch * steps, kThreads, 0, stream>>>(
          sequences.const_data_ptr<float>(), reinterpret_cast<const half*>(codebook.const_data_ptr()),
          codebook_norm.const_data_ptr<float>(),
          step_weights.has_value() ? step_weights_tensor.const_data_ptr<float>() : nullptr, steps, batch,
          static_cast<int>(bank_count), bank_specific_sequences,
          step_error.mutable_data_ptr<float>(), states.mutable_data_ptr<int64_t>());
    } else {
      codebook_norm = bank_count == 1
          ? cached_codebook_norm<VectorSize, float>(codebook, stream)
          : cached_banked_codebook_norm<VectorSize, float>(codebook, stream);
      qvq_memoryless_kernel<VectorSize, float><<<launch_batch * steps, kThreads, 0, stream>>>(
          sequences.const_data_ptr<float>(), codebook.const_data_ptr<float>(), codebook_norm.const_data_ptr<float>(),
          step_weights.has_value() ? step_weights_tensor.const_data_ptr<float>() : nullptr, steps, batch,
          static_cast<int>(bank_count), bank_specific_sequences,
          step_error.mutable_data_ptr<float>(), states.mutable_data_ptr<int64_t>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    record_norm_cache_use(codebook, codebook_norm, stream);
    return {states, step_error.sum(1)};
  }
  const auto backpointer_dtype = transition_bits <= 8 ? at::kByte : at::kUInt16;
  at::Tensor backpointers = at::empty(
      {launch_batch, steps, suffix_count}, sequences.options().dtype(backpointer_dtype));
  at::Tensor states = at::empty({launch_batch, steps}, sequences.options().dtype(at::kLong));
  at::Tensor squared_error = at::empty({launch_batch}, sequences.options());
  // Norms are immutable for a retained codebook and are built only once per
  // device/codebook.  A CUDA event makes reuse safe across streams.
  codebook_norm = bank_count == 1
      ? (codebook.scalar_type() == at::kHalf
             ? cached_codebook_norm<VectorSize, half>(codebook, stream)
             : cached_codebook_norm<VectorSize, float>(codebook, stream))
      : (codebook.scalar_type() == at::kHalf
             ? cached_banked_codebook_norm<VectorSize, half>(codebook, stream)
             : cached_banked_codebook_norm<VectorSize, float>(codebook, stream));
  const size_t shared_bytes = static_cast<size_t>(2 * suffix_count + 2 * kThreads) * sizeof(float);
  TORCH_CHECK(shared_bytes <= static_cast<size_t>(properties.sharedMemPerBlockOptin),
              "QVQ CUDA Viterbi requires ", shared_bytes,
              " bytes of dynamic shared memory, but this device supports ", properties.sharedMemPerBlockOptin);
  const int64_t* overlap_ptr = constrained ? overlap_tensor.const_data_ptr<int64_t>() : nullptr;
  const float* step_weights_ptr = weighted ? step_weights_tensor.const_data_ptr<float>() : nullptr;

#define QVQ_VITERBI_LAUNCH_TYPED(BITS, CODEBOOK_TYPE, CODEBOOK_POINTER, BACKPOINTER_TYPE)                      \
  do {                                                                                                             \
    C10_CUDA_CHECK(cudaFuncSetAttribute(qvq_viterbi_kernel<BITS, VectorSize, CODEBOOK_TYPE, BACKPOINTER_TYPE>,   \
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,                              \
        static_cast<int>(shared_bytes)));                                         \
    qvq_viterbi_kernel<BITS, VectorSize, CODEBOOK_TYPE, BACKPOINTER_TYPE><<<launch_batch, kThreads, shared_bytes, stream>>>( \
        sequences.const_data_ptr<float>(), CODEBOOK_POINTER,                                                      \
        codebook_norm.const_data_ptr<float>(), overlap_ptr, step_weights_ptr,                                     \
        backpointers.mutable_data_ptr<BACKPOINTER_TYPE>(), states.mutable_data_ptr<int64_t>(),                    \
        squared_error.mutable_data_ptr<float>(), steps, batch, static_cast<int>(bank_count),                 \
        bank_specific_sequences, constrained, weighted); \
  } while (0)
#define QVQ_VITERBI_LAUNCH(BITS, BACKPOINTER_TYPE)                                                               \
  do {                                                                                                             \
    if (codebook.scalar_type() == at::kHalf) {                                                                    \
      QVQ_VITERBI_LAUNCH_TYPED(BITS, half, reinterpret_cast<const half*>(codebook.const_data_ptr()),              \
                               BACKPOINTER_TYPE);                                                                \
    } else {                                                                                                       \
      QVQ_VITERBI_LAUNCH_TYPED(BITS, float, codebook.const_data_ptr<float>(), BACKPOINTER_TYPE);                \
    }                                                                                                              \
  } while (0)
  const auto launch_for_bits = [&](auto bits_tag) {
    constexpr int bits = decltype(bits_tag)::value;
    if (transition_bits <= 8) {
      QVQ_VITERBI_LAUNCH(bits, uint8_t);
    } else {
      QVQ_VITERBI_LAUNCH(bits, uint16_t);
    }
  };
  switch (transition_bits) {
    case 2:  launch_for_bits(std::integral_constant<int, 2>{}); break;
    case 3:  launch_for_bits(std::integral_constant<int, 3>{}); break;
    case 4:  launch_for_bits(std::integral_constant<int, 4>{}); break;
    case 5:  launch_for_bits(std::integral_constant<int, 5>{}); break;
    case 6:  launch_for_bits(std::integral_constant<int, 6>{}); break;
    case 7:  launch_for_bits(std::integral_constant<int, 7>{}); break;
    case 8:  launch_for_bits(std::integral_constant<int, 8>{}); break;
    case 9:  launch_for_bits(std::integral_constant<int, 9>{}); break;
    case 10: launch_for_bits(std::integral_constant<int, 10>{}); break;
    case 11: launch_for_bits(std::integral_constant<int, 11>{}); break;
    case 12: launch_for_bits(std::integral_constant<int, 12>{}); break;
    case 13: launch_for_bits(std::integral_constant<int, 13>{}); break;
    case 14: launch_for_bits(std::integral_constant<int, 14>{}); break;
    case 15: launch_for_bits(std::integral_constant<int, 15>{}); break;
  }
#undef QVQ_VITERBI_LAUNCH
#undef QVQ_VITERBI_LAUNCH_TYPED
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  record_norm_cache_use(codebook, codebook_norm, stream);
  return {states, squared_error};
}

std::tuple<at::Tensor, at::Tensor> qvq_viterbi_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebook,
    int64_t transition_bits,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights) {
  return qvq_viterbi_cuda_impl<2>(sequences, codebook, transition_bits, overlap, step_weights);
}

std::tuple<at::Tensor, at::Tensor> qvq_viterbi_trusted_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebook,
    int64_t transition_bits,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights) {
  return qvq_viterbi_cuda_impl<2>(
      sequences, codebook, transition_bits, overlap, step_weights, 1, false);
}

std::tuple<at::Tensor, at::Tensor> qvq_viterbi_tail_trusted_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebook,
    int64_t transition_bits,
    const c10::optional<at::Tensor>& step_weights) {
  const int64_t midpoint = sequences.size(1) / 2;
  auto rotated_sequences = at::roll(sequences, {midpoint}, {1}).contiguous();
  c10::optional<at::Tensor> rotated_weights = c10::nullopt;
  if (step_weights.has_value()) {
    rotated_weights = at::roll(*step_weights, {midpoint}, {1}).contiguous();
  }
  auto provisional = qvq_viterbi_cuda_impl<2>(
      rotated_sequences, codebook, transition_bits, c10::nullopt, rotated_weights, 1, false);
  const int64_t overlap_mask = (int64_t{1} << (16 - transition_bits)) - 1;
  auto overlap = std::get<0>(provisional).select(1, midpoint - 1).bitwise_and(overlap_mask).contiguous();
  return qvq_viterbi_cuda_impl<2>(
      sequences, codebook, transition_bits, overlap, step_weights, 1, false);
}

std::tuple<at::Tensor, at::Tensor> qvq_viterbi_v4_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebook,
    int64_t transition_bits,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights) {
  return qvq_viterbi_cuda_impl<4>(sequences, codebook, transition_bits, overlap, step_weights);
}

template <int VectorSize>
std::tuple<at::Tensor, at::Tensor> qvq_viterbi_banked_cuda_impl(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights) {
  TORCH_CHECK(codebooks.dim() == 3 && codebooks.size(0) >= 1 && codebooks.size(0) <= 4,
              "banked V4 Viterbi requires between one and four codebooks");
  return qvq_viterbi_cuda_impl<VectorSize>(
      sequences, codebooks, transition_bits, overlap, step_weights, codebooks.size(0));
}

std::tuple<at::Tensor, at::Tensor> qvq_viterbi_banked_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights) {
  return qvq_viterbi_banked_cuda_impl<4>(sequences, codebooks, transition_bits, overlap, step_weights);
}

template <int BlockThreads, typename CodebookScalar, typename BackpointerScalar>
bool can_launch_qvq_v2_segment_w25_cooperative(int batch, const cudaDeviceProp& properties) {
  constexpr int partitions = (1 << (16 - 5)) / BlockThreads;
  constexpr int bank_count = 2;
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      qvq_v2_segment_w25_cooperative_kernel<BlockThreads, CodebookScalar, BackpointerScalar>,
      BlockThreads,
      0));
  return properties.cooperativeLaunch &&
      batch * bank_count * partitions <= blocks_per_sm * properties.multiProcessorCount;
}

template <int BlockThreads, typename CodebookScalar, typename BackpointerScalar>
void launch_qvq_v2_segment_w25_cooperative(
    const float* sequences,
    const CodebookScalar* codebooks,
    const float* codebook_norms,
    const int64_t* overlap,
    const float* step_weights,
    float* costs_a,
    float* costs_b,
    BackpointerScalar* backpointers,
    uint8_t* boundary_banks,
    int batch,
    int family_batch,
    bool constrained,
    bool weighted,
    const cudaDeviceProp& properties,
    cudaStream_t stream) {
  constexpr int partitions = (1 << (16 - 5)) / BlockThreads;
  constexpr int bank_count = 2;
  constexpr int segment_count = 8;
  const int blocks = batch * bank_count * partitions;
  const bool can_launch =
      can_launch_qvq_v2_segment_w25_cooperative<BlockThreads, CodebookScalar, BackpointerScalar>(
          batch, properties);
  TORCH_CHECK(
      can_launch,
      "cooperative W2.5 segmented V2 grid exceeds resident-device capacity");

  for (int segment_index = 0; segment_index < segment_count; ++segment_index) {
    void* arguments[] = {
        &sequences,
        &codebooks,
        &codebook_norms,
        &overlap,
        &step_weights,
        &costs_a,
        &costs_b,
        &backpointers,
        &boundary_banks,
        &batch,
        &family_batch,
        &segment_index,
        &constrained,
        &weighted,
    };
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(
            qvq_v2_segment_w25_cooperative_kernel<BlockThreads, CodebookScalar, BackpointerScalar>),
        dim3(blocks),
        dim3(BlockThreads),
        arguments,
        0,
        stream));
  }
}

// ---------------------------------------------------------------------------
// Fused W2 (Shift=4) two-bank P32 family-grid recurrence.
//
// Replaces the 8 x qvq_v2_segment_grid_kernel<4,2,16,fused> + finalize +
// codebook-norm-pack launches of viterbi_v2_segment_family_grid_trusted with a
// persistent kernel that owns one whole sequence (both banks) per CTA:
//
//  * work unit = (sequence, 16-step segment); CTAs pull units from an atomic
//    queue in segment-major order and wait on a per-sequence done flag, so the
//    grid never suffers a partial wave and the segment boundary merge is a
//    plain __syncthreads instead of a kernel boundary;
//  * bank 1 of every PGC16 V2B4 family is bank 0 with the state index XOR'ed by
//    a rate-keyed mask (pgc16.py: PGC16_V2B4_BANK_XOR_MASKS_BY_TRANSITION_BITS),
//    i.e. e_1(s) == e_0(s ^ mask) bit for bit.  A device-side check confirms the
//    relation per family, after which every emission is evaluated once and fed
//    to both banks' min-plus updates (half the emission work, half the codebook
//    traffic).  Families whose codebooks are not related fall back to a general
//    two-codebook loop inside the same kernel;
//  * prefixes h < 8 of the fp16 bank-0 codebook live in shared memory (128 KB),
//    the other half streams from L2 as 4-byte fp16 pairs with the norm
//    recomputed exactly, instead of the 8-byte packed codebook/norm record;
//  * the step-127 argmin is fused; the serial traceback runs in a tiny follow-up
//    kernel so the SM is released as soon as the forward pass ends.
//
// Round 2: arithmetic is DECISION-EQUIVALENT to emission<2, half>() and the
// reference recurrence, not bit-identical — the emission is evaluated as
// (t0-c0)^2 + (t1-c1)^2 with the step weight folded into one FMA (see
// fused_emission/fused_candidate), so accumulated costs can differ from the
// reference in the last ulps and genuine near-ties may resolve differently
// (measured ~1e-5 of states, score-equal; see
// docs/qvq_grid_kernel_opt_status.md).  Tie precedence and visit order are
// unchanged: lowest prefix on equal candidates (bank-1 candidates visited in
// ascending bank-1 prefix order inside each block of four prefixes, block
// winners merged with lower_pair()), bank-0-first boundary merge and
// bank-major / state-major terminal argmin.
// ---------------------------------------------------------------------------

constexpr int kFusedThreads = 1024;
constexpr int kFusedShift = 4;
constexpr int kFusedPrefixCount = 1 << kFusedShift;            // 16
constexpr int kFusedSuffixCount = 1 << (16 - kFusedShift);     // 4096
constexpr int kFusedSegmentSteps = 16;
constexpr int kFusedSegmentCount = 128 / kFusedSegmentSteps;   // 8
constexpr int kFusedSharedPrefixes = 8;                        // prefixes cached in shared memory
constexpr int kFusedSharedStates = kFusedSharedPrefixes * kFusedSuffixCount;  // 32768
constexpr int kFusedMaskCandidates = 3;
constexpr size_t kFusedCodebookSharedBytes = static_cast<size_t>(kFusedSharedStates) * sizeof(uint32_t);  // 128 KB
constexpr size_t kFusedFrontierSharedBytes = 2 * static_cast<size_t>(kFusedSuffixCount) * sizeof(float); // 32 KB
constexpr size_t kFusedSharedBytes = kFusedCodebookSharedBytes + kFusedFrontierSharedBytes;

// PGC16_V2B4_BANK_XOR_MASKS_BY_TRANSITION_BITS[4][1:] -- the three non-canonical
// banks a W2 family can pair with canonical bank 0.
__constant__ uint16_t g_fused_w2_masks[kFusedMaskCandidates] = {0x5A5A, 0x3C3C, 0xC3C3};

// One CTA per (family, candidate mask): does bank 1 equal bank 0 permuted by mask?
__global__ __launch_bounds__(kFusedThreads) void qvq_fused_detect_family_mask_kernel(
    const uint32_t* __restrict__ codebooks,  // [families][2][65536] fp16 pairs as u32
    int* __restrict__ match) {               // [families][kFusedMaskCandidates]
  const int family = static_cast<int>(blockIdx.x) / kFusedMaskCandidates;
  const int candidate = static_cast<int>(blockIdx.x) - family * kFusedMaskCandidates;
  const int mask = g_fused_w2_masks[candidate];
  const uint32_t* bank0 = codebooks + static_cast<int64_t>(family) * 2 * kStateCount;
  const uint32_t* bank1 = bank0 + kStateCount;
  bool equal = true;
  for (int state = static_cast<int>(threadIdx.x); state < kStateCount; state += kFusedThreads) {
    equal &= bank1[state] == bank0[state ^ mask];
  }
  const bool all_equal = __syncthreads_and(equal);
  if (threadIdx.x == 0) {
    match[blockIdx.x] = all_equal ? 1 : 0;
  }
}

// Round-2 relaxed emission (decision-equivalent, NOT bit-identical to the
// reference emission<2, half>()): the squared distance is evaluated directly as
// (t0-c0)^2 + (t1-c1)^2 in four FP32 ops instead of the reference's
// norm/dot/expand form (~9 ops incl. the clip).  The sum-of-squares form is
// >= +0 by construction, so the reference's fmaxf(distance, 0) clip is free.
__device__ __forceinline__ float fused_emission(float t0, float t1, uint32_t code_bits) {
  const half2 code2_half = *reinterpret_cast<const half2*>(&code_bits);
  const float2 code2 = __half22float2(code2_half);
  const float d0 = __fadd_rn(t0, -code2.x);
  const float d1 = __fadd_rn(t1, -code2.y);
  return __fmaf_rn(d1, d1, __fmul_rn(d0, d0));
}

// Candidate cost: the step weight is folded into a single FMA instead of the
// reference's separate multiply + add (same relaxation as above).
template <bool Weighted>
__device__ __forceinline__ float fused_candidate(float emission, float weight, float predecessor) {
  if constexpr (Weighted) {
    return __fmaf_rn(emission, weight, predecessor);
  } else {
    return __fadd_rn(predecessor, emission);
  }
}

struct FusedThreadMap {
  int row;        // suffix >> 4 of the thread's four base suffixes
  int base;       // first base suffix (row * 16 + group * 4)
  int prow;       // partner row (row ^ (mask_x >> 4))
  int pbase;      // first partner suffix of the aligned partner group
};

__device__ __forceinline__ FusedThreadMap fused_thread_map(int thread, int mask_x) {
  FusedThreadMap m;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  m.row = warp * 8 + (lane >> 2);
  m.base = m.row * 16 + (lane & 3) * 4;
  m.prow = m.row ^ (mask_x >> 4);
  m.pbase = m.prow * 16 + (((lane & 3) ^ ((mask_x >> 2) & 3)) * 4);
  return m;
}

// Load the four fp16 pairs of states (h, base .. base+3) of bank 0.
__device__ __forceinline__ uint4 fused_load_codes(
    const uint32_t* __restrict__ shared_codes,
    const uint32_t* __restrict__ global_codes,
    int h,
    int base) {
  const int state = h * kFusedSuffixCount + base;
  if (h < kFusedSharedPrefixes) {
    return *reinterpret_cast<const uint4*>(shared_codes + state);
  }
  return __ldg(reinterpret_cast<const uint4*>(global_codes + state));
}

__device__ __forceinline__ uint32_t fused_code_at(const uint4& codes, int index) {
  return index == 0 ? codes.x : (index == 1 ? codes.y : (index == 2 ? codes.z : codes.w));
}

// One recurrence step for a thread's four base suffixes (bank 0) and their four
// partner suffixes (bank 1), with the emission shared between the banks.
template <int Mask, bool Weighted>
__device__ __forceinline__ void fused_step_shared(
    const FusedThreadMap& map,
    const uint32_t* __restrict__ shared_codes,
    const uint32_t* __restrict__ global_codes,
    const float* __restrict__ g0,
    const float* __restrict__ g1,
    float t0, float t1, float weight,
    float (&best0)[4], int (&best_h0)[4],
    float (&best1)[4], int (&best_h1)[4]) {
  constexpr int mask_h = Mask >> 12;
  constexpr int mask_h_block = mask_h >> 2;
  constexpr int mask_h_inner = mask_h & 3;
  #pragma unroll
  for (int jj = 0; jj < 4; ++jj) {
    best0[jj] = CUDART_INF_F;
    best_h0[jj] = 0;
    best1[jj] = CUDART_INF_F;
    best_h1[jj] = 0;
  }
  #pragma unroll
  for (int block = 0; block < kFusedPrefixCount / 4; ++block) {
    uint4 codes[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      codes[i] = fused_load_codes(shared_codes, global_codes, block * 4 + i, map.base);
    }
    float emissions[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        emissions[i][jj] = fused_emission(t0, t1, fused_code_at(codes[i], jj));
      }
    }
    // Bank 0: ascending prefix order, strict '<' keeps the lowest prefix on ties.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = block * 4 + i;
      const float predecessor = g0[h * 256 + map.row];
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const float candidate = fused_candidate<Weighted>(emissions[i][jj], weight, predecessor);
        if (candidate < best0[jj]) {
          best0[jj] = candidate;
          best_h0[jj] = h;
        }
      }
    }
    // Bank 1: partner state (h', base ^ mask_x) uses e_0(h' ^ mask_h, base).
    // Visit h' ascending inside the block (h' = 4*(block ^ mask_h_block) + i'),
    // then merge block winners with the reference tie precedence.
    const int block1 = block ^ mask_h_block;
    float block_best[4];
    int block_best_h[4];
    #pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      block_best[jj] = CUDART_INF_F;
      block_best_h[jj] = 0;
    }
    #pragma unroll
    for (int ip = 0; ip < 4; ++ip) {
      const int i = ip ^ mask_h_inner;
      const int hp = block1 * 4 + ip;
      const float predecessor = g1[hp * 256 + map.prow];
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const float candidate = fused_candidate<Weighted>(emissions[i][jj], weight, predecessor);
        if (candidate < block_best[jj]) {
          block_best[jj] = candidate;
          block_best_h[jj] = hp;
        }
      }
    }
    #pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      if constexpr (mask_h_block == 0) {
        // Blocks arrive in ascending h' order: strict '<' already keeps the lowest prefix.
        if (block_best[jj] < best1[jj]) {
          best1[jj] = block_best[jj];
          best_h1[jj] = block_best_h[jj];
        }
      } else {
        if (lower_pair(block_best[jj], block_best_h[jj], best1[jj], best_h1[jj])) {
          best1[jj] = block_best[jj];
          best_h1[jj] = block_best_h[jj];
        }
      }
    }
  }
}

// General fallback: bank 1 is an arbitrary codebook; both banks' emissions are
// evaluated separately (bank 0 h < 8 from shared memory, everything else from L2).
template <bool Weighted>
__device__ __forceinline__ void fused_step_general(
    const FusedThreadMap& map,
    const uint32_t* __restrict__ shared_codes,
    const uint32_t* __restrict__ global_codes0,
    const uint32_t* __restrict__ global_codes1,
    const float* __restrict__ g0,
    const float* __restrict__ g1,
    float t0, float t1, float weight,
    float (&best0)[4], int (&best_h0)[4],
    float (&best1)[4], int (&best_h1)[4]) {
  #pragma unroll
  for (int jj = 0; jj < 4; ++jj) {
    best0[jj] = CUDART_INF_F;
    best_h0[jj] = 0;
    best1[jj] = CUDART_INF_F;
    best_h1[jj] = 0;
  }
  #pragma unroll 4
  for (int h = 0; h < kFusedPrefixCount; ++h) {
    const uint4 codes0 = fused_load_codes(shared_codes, global_codes0, h, map.base);
    const uint4 codes1 = __ldg(reinterpret_cast<const uint4*>(
        global_codes1 + h * kFusedSuffixCount + map.base));
    const float predecessor0 = g0[h * 256 + map.row];
    const float predecessor1 = g1[h * 256 + map.row];
    #pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const float candidate0 = fused_candidate<Weighted>(
          fused_emission(t0, t1, fused_code_at(codes0, jj)), weight, predecessor0);
      if (candidate0 < best0[jj]) {
        best0[jj] = candidate0;
        best_h0[jj] = h;
      }
      const float candidate1 = fused_candidate<Weighted>(
          fused_emission(t0, t1, fused_code_at(codes1, jj)), weight, predecessor1);
      if (candidate1 < best1[jj]) {
        best1[jj] = candidate1;
        best_h1[jj] = h;
      }
    }
  }
}

struct FusedSequenceArgs {
  const float* __restrict__ sequences;      // [batch][128][2]
  const uint32_t* __restrict__ codebooks;   // [families][2][65536] fp16 pairs
  const int64_t* __restrict__ overlap;      // [batch] or nullptr
  const float* __restrict__ step_weights;   // [batch][128] or nullptr
  const int* __restrict__ family_mask_match;// [families][kFusedMaskCandidates]
  float* __restrict__ frontiers;            // [batch][2][4096] inter-segment hand-off
  uint8_t* __restrict__ backpointers;       // [batch][127][2][4096]
  uint8_t* __restrict__ boundary_banks;     // [batch][7][4096]
  int* __restrict__ best_flat;              // [batch]
  float* __restrict__ squared_error;        // [batch]
  int* __restrict__ queue;                  // [1 + batch]: unit counter, then done-segment count per sequence
  int batch;
  int family_batch;
  bool constrained;
};

template <int Mask, bool Weighted, bool Shared>
__device__ __forceinline__ void fused_run_segment(
    const FusedSequenceArgs& args,
    uint32_t* __restrict__ shared_codes,
    float* __restrict__ g0,
    float* __restrict__ g1,
    int sequence,
    int segment,
    int family) {
  constexpr int mask_x = Mask & 0xFFF;
  constexpr int mask_x_inner = mask_x & 3;
  const int thread = static_cast<int>(threadIdx.x);
  const FusedThreadMap map = fused_thread_map(thread, Shared ? mask_x : 0);
  const uint32_t* global_codes0 = args.codebooks + static_cast<int64_t>(family) * 2 * kStateCount;
  const uint32_t* global_codes1 = global_codes0 + kStateCount;
  const float* sequence_targets = args.sequences + static_cast<int64_t>(sequence) * 128 * 2;
  const float* sequence_weights =
      Weighted ? args.step_weights + static_cast<int64_t>(sequence) * 128 : nullptr;
  uint8_t* sequence_backpointers =
      args.backpointers + static_cast<int64_t>(sequence) * 127 * 2 * kFusedSuffixCount;
  const int first_step = segment * kFusedSegmentSteps;
  const int end_step = min(first_step + kFusedSegmentSteps, 127);

  // ---- frontier entry: reference step-0 constraint mask or segment-boundary merge ----
  if (segment == 0) {
    const int required_overlap = args.constrained ? static_cast<int>(args.overlap[sequence]) : 0;
    for (int x = thread; x < kFusedSuffixCount; x += kFusedThreads) {
      const float value = (args.constrained && x != required_overlap) ? CUDART_INF_F : 0.0f;
      g0[x] = value;
      g1[x] = value;
    }
  } else {
    const float* frontier = args.frontiers + static_cast<int64_t>(sequence) * 2 * kFusedSuffixCount;
    uint8_t* boundary = args.boundary_banks +
        (static_cast<int64_t>(sequence) * (kFusedSegmentCount - 1) + (segment - 1)) * kFusedSuffixCount;
    for (int x = thread; x < kFusedSuffixCount; x += kFusedThreads) {
      // Written by another SM: bypass the (non-coherent) L1.
      const float value0 = __ldcg(frontier + x);
      const float value1 = __ldcg(frontier + kFusedSuffixCount + x);
      const bool take1 = value1 < value0;
      const float merged = take1 ? value1 : value0;
      g0[x] = merged;
      g1[x] = merged;
      boundary[x] = static_cast<uint8_t>(take1);
    }
  }
  __syncthreads();

  // ---- recurrence ----
  for (int step = first_step; step < end_step; ++step) {
    const float t0 = sequence_targets[step * 2];
    const float t1 = sequence_targets[step * 2 + 1];
    const float weight = Weighted ? sequence_weights[step] : 1.0f;
    float best0[4], best1[4];
    int best_h0[4], best_h1[4];
    if constexpr (Shared) {
      fused_step_shared<Mask, Weighted>(
          map, shared_codes, global_codes0, g0, g1, t0, t1, weight,
          best0, best_h0, best1, best_h1);
    } else {
      fused_step_general<Weighted>(
          map, shared_codes, global_codes0, global_codes1, g0, g1, t0, t1, weight,
          best0, best_h0, best1, best_h1);
    }
    __syncthreads();
    *reinterpret_cast<float4*>(g0 + map.base) = make_float4(best0[0], best0[1], best0[2], best0[3]);
    uint8_t* step_pointers = sequence_backpointers + static_cast<int64_t>(step) * 2 * kFusedSuffixCount;
    *reinterpret_cast<uint32_t*>(step_pointers + map.base) =
        static_cast<uint32_t>(best_h0[0]) | (static_cast<uint32_t>(best_h0[1]) << 8) |
        (static_cast<uint32_t>(best_h0[2]) << 16) | (static_cast<uint32_t>(best_h0[3]) << 24);
    if constexpr (Shared) {
      // Partner suffix of base jj sits at position jj ^ mask_x_inner of the partner group.
      float out1[4];
      int out_h1[4];
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        out1[jj ^ mask_x_inner] = best1[jj];
        out_h1[jj ^ mask_x_inner] = best_h1[jj];
      }
      *reinterpret_cast<float4*>(g1 + map.pbase) = make_float4(out1[0], out1[1], out1[2], out1[3]);
      *reinterpret_cast<uint32_t*>(step_pointers + kFusedSuffixCount + map.pbase) =
          static_cast<uint32_t>(out_h1[0]) | (static_cast<uint32_t>(out_h1[1]) << 8) |
          (static_cast<uint32_t>(out_h1[2]) << 16) | (static_cast<uint32_t>(out_h1[3]) << 24);
    } else {
      *reinterpret_cast<float4*>(g1 + map.base) = make_float4(best1[0], best1[1], best1[2], best1[3]);
      *reinterpret_cast<uint32_t*>(step_pointers + kFusedSuffixCount + map.base) =
          static_cast<uint32_t>(best_h1[0]) | (static_cast<uint32_t>(best_h1[1]) << 8) |
          (static_cast<uint32_t>(best_h1[2]) << 16) | (static_cast<uint32_t>(best_h1[3]) << 24);
    }
    __syncthreads();
  }

  // ---- segment exit: hand the frontier to the next segment, or run the terminal argmin ----
  if (segment + 1 < kFusedSegmentCount) {
    float* frontier = args.frontiers + static_cast<int64_t>(sequence) * 2 * kFusedSuffixCount;
    for (int x = thread; x < kFusedSuffixCount; x += kFusedThreads) {
      frontier[x] = g0[x];
      frontier[kFusedSuffixCount + x] = g1[x];
    }
    __threadfence();
    __syncthreads();
    if (thread == 0) {
      atomicExch(args.queue + 1 + sequence, segment + 1);
    }
    return;
  }

  // Step 127 folded into the terminal reduction exactly as the reference finalize:
  // flat index = bank * 65536 + state, lowest flat index wins ties.
  {
    const int required_overlap = args.constrained ? static_cast<int>(args.overlap[sequence]) : 0;
    const float t0 = sequence_targets[127 * 2];
    const float t1 = sequence_targets[127 * 2 + 1];
    const float weight = Weighted ? sequence_weights[127] : 1.0f;
    float best = CUDART_INF_F;
    int best_flat = 2 * kStateCount;
    for (int flat = thread; flat < 2 * kStateCount; flat += kFusedThreads) {
      const int bank = flat >> 16;
      const int state = flat & (kStateCount - 1);
      const uint32_t code = bank == 0 ? __ldg(global_codes0 + state) : __ldg(global_codes1 + state);
      const float* g = bank == 0 ? g0 : g1;
      float candidate = fused_candidate<Weighted>(
          fused_emission(t0, t1, code), weight, g[state >> kFusedShift]);
      if (args.constrained && (state & (kFusedSuffixCount - 1)) != required_overlap) {
        candidate = CUDART_INF_F;
      }
      if (lower_pair(candidate, flat, best, best_flat)) {
        best = candidate;
        best_flat = flat;
      }
    }
    block_argmin(best, best_flat);
    if (thread == 0) {
      args.squared_error[sequence] = best;
      args.best_flat[sequence] = best_flat;
    }
    __syncthreads();
  }
}

template <bool Weighted>
__global__ __launch_bounds__(kFusedThreads, 1) void qvq_fused_w2_family_grid_kernel(FusedSequenceArgs args) {
  extern __shared__ __align__(16) unsigned char fused_shared_raw[];
  uint32_t* shared_codes = reinterpret_cast<uint32_t*>(fused_shared_raw);
  float* g0 = reinterpret_cast<float*>(fused_shared_raw + kFusedCodebookSharedBytes);
  float* g1 = g0 + kFusedSuffixCount;
  __shared__ int unit_shared;
  const int thread = static_cast<int>(threadIdx.x);
  const int total_units = args.batch * kFusedSegmentCount;
  int loaded_family = -1;
  int mask_id = -1;

  for (;;) {
    if (thread == 0) {
      unit_shared = atomicAdd(args.queue, 1);
    }
    __syncthreads();
    const int unit = unit_shared;
    __syncthreads();
    if (unit >= total_units) {
      break;
    }
    const int segment = unit / args.batch;
    const int sequence = unit - segment * args.batch;
    const int family = sequence / args.family_batch;
    if (family != loaded_family) {
      const uint4* source = reinterpret_cast<const uint4*>(
          args.codebooks + static_cast<int64_t>(family) * 2 * kStateCount);
      uint4* destination = reinterpret_cast<uint4*>(shared_codes);
      for (int i = thread; i < kFusedSharedStates / 4; i += kFusedThreads) {
        destination[i] = __ldg(source + i);
      }
      mask_id = -1;
      #pragma unroll
      for (int candidate = kFusedMaskCandidates - 1; candidate >= 0; --candidate) {
        if (args.family_mask_match[family * kFusedMaskCandidates + candidate] != 0) {
          mask_id = candidate;
        }
      }
      loaded_family = family;
      __syncthreads();
    }
    if (segment > 0) {
      if (thread == 0) {
        volatile int* done = args.queue + 1 + sequence;
        while (*done < segment) {
          __nanosleep(256);
        }
        __threadfence();
      }
      __syncthreads();
    }
    switch (mask_id) {
      case 0:
        fused_run_segment<0x5A5A, Weighted, true>(args, shared_codes, g0, g1, sequence, segment, family);
        break;
      case 1:
        fused_run_segment<0x3C3C, Weighted, true>(args, shared_codes, g0, g1, sequence, segment, family);
        break;
      case 2:
        fused_run_segment<0xC3C3, Weighted, true>(args, shared_codes, g0, g1, sequence, segment, family);
        break;
      default:
        fused_run_segment<0, Weighted, false>(args, shared_codes, g0, g1, sequence, segment, family);
        break;
    }
    __syncthreads();
  }
}

// Serial traceback, one thread per sequence: identical to the thread-0 tail of
// qvq_v2_segment_grid_finalize_kernel<4, 2, 16, false, ...>.
__global__ void qvq_fused_w2_traceback_kernel(
    const uint8_t* __restrict__ backpointers,
    const uint8_t* __restrict__ boundary_banks,
    const int* __restrict__ best_flat,
    int64_t* __restrict__ states,
    uint8_t* __restrict__ segment_bank_ids,
    int batch) {
  const int sequence = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (sequence >= batch) {
    return;
  }
  constexpr int suffix_count = kFusedSuffixCount;
  constexpr int bank_suffix_count = 2 * suffix_count;
  const int64_t pointer_base = static_cast<int64_t>(sequence) * 127 * bank_suffix_count;
  const int64_t boundary_base = static_cast<int64_t>(sequence) * (kFusedSegmentCount - 1) * suffix_count;
  const int64_t state_base = static_cast<int64_t>(sequence) * 128;
  const int64_t selector_base = static_cast<int64_t>(sequence) * kFusedSegmentCount;
  const int flat = best_flat[sequence];
  int current_bank = flat / kStateCount;
  int current_state = flat - current_bank * kStateCount;
  states[state_base + 127] = current_state;
  segment_bank_ids[selector_base + kFusedSegmentCount - 1] = static_cast<uint8_t>(current_bank);
  for (int step = 127; step > 0; --step) {
    const int predecessor_suffix = current_state >> kFusedShift;
    if (step % kFusedSegmentSteps == 0) {
      const int boundary_index = step / kFusedSegmentSteps - 1;
      current_bank = static_cast<int>(boundary_banks[
          boundary_base + static_cast<int64_t>(boundary_index) * suffix_count + predecessor_suffix]);
      segment_bank_ids[selector_base + boundary_index] = static_cast<uint8_t>(current_bank);
    }
    const int prefix = static_cast<int>(backpointers[
        pointer_base + static_cast<int64_t>(step - 1) * bank_suffix_count +
        current_bank * suffix_count + predecessor_suffix]);
    current_state = prefix * suffix_count + predecessor_suffix;
    states[state_base + step - 1] = current_state;
  }
}

// Number of times the fused W2 family-grid kernel was dispatched in this
// process; exposed as gptqmodel_qvq.fused_family_grid_dispatch_count() so tests
// can assert that the fused path (not the reference path) produced a result.
std::atomic<int64_t> g_fused_w2_family_grid_dispatches{0};

int64_t qvq_fused_family_grid_dispatch_count() {
  return g_fused_w2_family_grid_dispatches.load();
}

// Read once on first use (cached for the process lifetime).
bool fused_w2_family_grid_disabled() {
  static const bool disabled = [] {
    const char* value = std::getenv("QVQ_DISABLE_FUSED_FAMILY_GRID");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return disabled;
}

bool fused_w2_family_grid_supported(const cudaDeviceProp& properties) {
  return properties.major >= 8 &&
      static_cast<size_t>(properties.sharedMemPerBlockOptin) >= kFusedSharedBytes &&
      !fused_w2_family_grid_disabled();
}

enum class FamilyGridDirectDistanceMode {
  kOff,
  kAll,
  kProvisional,
  kFinal,
  kProvisionalLargeFinal,
};

FamilyGridDirectDistanceMode family_grid_direct_distance_mode() {
  const char* value = std::getenv("GPTQMODEL_QVQ_YAQA_FAST_VITERBI_DISTANCE");
  if (value == nullptr || value[0] == '\0' || (value[0] == '0' && value[1] == '\0')) {
    return FamilyGridDirectDistanceMode::kOff;
  }
  if (std::strcmp(value, "provisional") == 0) {
    return FamilyGridDirectDistanceMode::kProvisional;
  }
  if (std::strcmp(value, "final") == 0) {
    return FamilyGridDirectDistanceMode::kFinal;
  }
  if (std::strcmp(value, "provisional_large_final") == 0) {
    return FamilyGridDirectDistanceMode::kProvisionalLargeFinal;
  }
  return FamilyGridDirectDistanceMode::kAll;
}

// Number of times the norm-rank contiguous-band recurrence was dispatched in
// this process; exposed as gptqmodel_qvq.norm_rank_grid_dispatch_count() so
// tests can assert which path produced a result.
std::atomic<int64_t> g_norm_rank_grid_dispatches{0};
std::atomic<int64_t> g_norm_rank_baseline_fallbacks{0};

int64_t qvq_norm_rank_grid_dispatch_count() {
  return g_norm_rank_grid_dispatches.load();
}

std::vector<int64_t> qvq_norm_rank_telemetry_snapshot() {
  unsigned long long evaluated = 0;
  unsigned long long possible = 0;
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(
      &evaluated, g_norm_rank_candidates_evaluated, sizeof(evaluated), 0, cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(
      &possible, g_norm_rank_candidates_possible, sizeof(possible), 0, cudaMemcpyDeviceToHost));
  return {
      g_norm_rank_grid_dispatches.load(),
      g_norm_rank_baseline_fallbacks.load(),
      static_cast<int64_t>(evaluated),
      static_cast<int64_t>(possible),
  };
}

// Number of live norm-rank table cache entries; exposed as
// gptqmodel_qvq.norm_rank_cache_size() so tests can assert the cache stays
// bounded under eviction pressure.  Snapshot only: entries come and go under
// g_norm_rank_cache_mutex in whichever thread dispatched the recurrence.
int64_t qvq_norm_rank_cache_size() {
  std::lock_guard<std::mutex> lock(g_norm_rank_cache_mutex);
  return static_cast<int64_t>(g_norm_rank_cache.size());
}

int64_t qvq_norm_cache_size() {
  std::lock_guard<std::mutex> lock(g_norm_cache_mutex);
  return static_cast<int64_t>(g_norm_cache.size());
}

// Pristine A/B control.  See norm_rank_grid_disabled() below.
// Exact survivor-pruning policy codes shared with
// `gptqmodel/quantization/qvq_pruning.py`.  These are part of the native op
// schema and must stay stable.
constexpr int64_t kViterbiPruningAuto = 0;       // auto + fallback=baseline
constexpr int64_t kViterbiPruningOff = 1;        // off
constexpr int64_t kViterbiPruningAutoError = 2;  // auto + fallback=error
constexpr int64_t kViterbiPruningRequired = 3;   // required

// Pristine A/B control, re-read on every dispatch so a same-process mutation
// of GPTQMODEL_QVQ_DISABLE_OCTET_GRID between calls is observed
// deterministically, matching the documented precedence: the deprecated
// escape hatch applies under `mode="auto"` only, and there it reflects the
// environment at call time — never a stale cached process-global policy.
// The variable keeps its historical name so existing enabled-versus-pristine
// harnesses keep working.  Only unset, empty, or exactly "0" leave the fast
// path enabled; any other non-empty value (including "00" and "0foo") forces
// the unmodified baseline grid recurrence.  One getenv per dispatched batch
// is noise next to the kernel launch itself.
bool norm_rank_grid_disabled() {
  const char* value = std::getenv("GPTQMODEL_QVQ_DISABLE_OCTET_GRID");
  return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

bool norm_rank_telemetry_enabled() {
  const char* value = std::getenv("GPTQMODEL_QVQ_TELEMETRY");
  return value != nullptr &&
      (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
       std::strcmp(value, "yes") == 0 || std::strcmp(value, "on") == 0);
}

// Segment loop for the norm-rank contiguous-band recurrence.  Launch geometry,
// frontier ping-pong, and the final frontier parity all match
// QVQ_V2_SEGMENT_GRID_LAUNCH, so the shared finalize kernel is reused verbatim.
template <int Shift, int BankCount, int SegmentSteps, int ChunkWidth, int BlockThreads>
void launch_qvq_v2_segment_norm_rank_segments_block(
    const float* sequences,
    const uint64_t* baseline_records,
    const NormRankTables& tables,
    const int64_t* overlap,
    float* frontier_a,
    float* frontier_b,
    uint8_t* backpointers,
    uint8_t* boundary_banks,
    int batch,
    int family_batch,
    bool constrained,
    cudaStream_t stream,
    bool collect_telemetry) {
  using PackType = typename NormRankPrefixPack<ChunkWidth>::type;
  constexpr int prefix_count = 1 << Shift;
  constexpr int suffix_count = 1 << (16 - Shift);
  constexpr int group_count = 1 << (16 - 2 * Shift);
  constexpr int segments = 128 / SegmentSteps;
  constexpr size_t recurrence_shared_bytes =
      2 * static_cast<size_t>(suffix_count + prefix_count) * sizeof(float) +
      3 * static_cast<size_t>(group_count) * sizeof(unsigned);
  const size_t shared_bytes = collect_telemetry
      ? std::max(recurrence_shared_bytes,
                 2 * static_cast<size_t>(BlockThreads / 32) * sizeof(unsigned long long))
      : recurrence_shared_bytes;
  auto* kernel = qvq_v2_segment_grid_norm_rank_kernel<
      Shift, BankCount, SegmentSteps, ChunkWidth, BlockThreads>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));
  const uint64_t* records =
      reinterpret_cast<const uint64_t*>(tables.records.const_data_ptr<int64_t>());
  const PackType* prefixes =
      reinterpret_cast<const PackType*>(tables.prefixes.const_data_ptr<uint8_t>());
  const float* low_norms = tables.low_norms.const_data_ptr<float>();
  kernel<<<batch * BankCount, BlockThreads, shared_bytes, stream>>>(
      sequences, baseline_records, records, prefixes, low_norms, overlap, nullptr,
      frontier_a, backpointers, boundary_banks, batch, family_batch, 0,
      constrained, collect_telemetry);
  for (int segment = 1; segment < segments; ++segment) {
    const float* input = segment % 2 == 1 ? frontier_a : frontier_b;
    float* output = segment % 2 == 1 ? frontier_b : frontier_a;
    kernel<<<batch * BankCount, BlockThreads, shared_bytes, stream>>>(
        sequences, baseline_records, records, prefixes, low_norms, overlap, input, output,
        backpointers, boundary_banks, batch, family_batch, segment,
        constrained, collect_telemetry);
  }
}

template <int Shift, int BankCount, int SegmentSteps, int ChunkWidth>
void launch_qvq_v2_segment_norm_rank_segments(
    const float* sequences,
    const uint64_t* baseline_records,
    const NormRankTables& tables,
    const int64_t* overlap,
    float* frontier_a,
    float* frontier_b,
    uint8_t* backpointers,
    uint8_t* boundary_banks,
    int batch,
    int family_batch,
    bool constrained,
    cudaStream_t stream,
    bool collect_telemetry,
    int multiprocessor_count) {
  // Shift-7 has two suffix columns per thread at 256 threads.  The extra
  // instruction-level work loses on an underfilled grid, but wins once the
  // 512-thread launch reaches a full SM wave because the smaller CTA permits
  // more resident work.  Select from grid geometry, not a model-specific
  // batch constant, so B2/P32 and B4/P64 share the same policy.
  if constexpr (Shift == 7) {
    if (batch * BankCount >= multiprocessor_count) {
      launch_qvq_v2_segment_norm_rank_segments_block<
          Shift, BankCount, SegmentSteps, ChunkWidth, 256>(
          sequences, baseline_records, tables, overlap, frontier_a, frontier_b,
          backpointers, boundary_banks, batch, family_batch, constrained, stream,
          collect_telemetry);
      return;
    }
    launch_qvq_v2_segment_norm_rank_segments_block<
        Shift, BankCount, SegmentSteps, ChunkWidth, 512>(
        sequences, baseline_records, tables, overlap, frontier_a, frontier_b,
        backpointers, boundary_banks, batch, family_batch, constrained, stream,
        collect_telemetry);
  } else {
    launch_qvq_v2_segment_norm_rank_segments_block<
        Shift, BankCount, SegmentSteps, ChunkWidth, kThreads>(
        sequences, baseline_records, tables, overlap, frontier_a, frontier_b,
        backpointers, boundary_banks, batch, family_batch, constrained, stream,
        collect_telemetry);
  }
}

// Host side of the fused path; called only for the family op with
// transition_bits == 4, bank_count == 2, segment_steps == 16, fp16 codebooks.
std::tuple<at::Tensor, at::Tensor, at::Tensor> qvq_fused_w2_family_grid_launch(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights,
    int family_batch,
    const cudaDeviceProp& properties,
    cudaStream_t stream) {
  const int batch = static_cast<int>(sequences.size(0));
  const int families = static_cast<int>(codebooks.size(0)) / 2;
  const bool constrained = overlap.has_value();
  const bool weighted = step_weights.has_value();
  const auto float_options = sequences.options().dtype(at::kFloat);
  const auto byte_options = sequences.options().dtype(at::kByte);
  const auto int_options = sequences.options().dtype(at::kInt);
  at::Tensor frontiers = at::empty({batch, 2, kFusedSuffixCount}, float_options);
  at::Tensor backpointers = at::empty({batch, 127, 2, kFusedSuffixCount}, byte_options);
  at::Tensor boundary_banks = at::empty({batch, kFusedSegmentCount - 1, kFusedSuffixCount}, byte_options);
  at::Tensor states = at::empty({batch, 128}, sequences.options().dtype(at::kLong));
  at::Tensor segment_bank_ids = at::empty({batch, kFusedSegmentCount}, byte_options);
  at::Tensor squared_error = at::empty({batch}, float_options);
  at::Tensor best_flat = at::empty({batch}, int_options);
  at::Tensor queue = at::zeros({1 + batch}, int_options);
  at::Tensor family_mask_match = at::empty({families, kFusedMaskCandidates}, int_options);

  const uint32_t* codebook_words = reinterpret_cast<const uint32_t*>(codebooks.const_data_ptr());
  qvq_fused_detect_family_mask_kernel<<<families * kFusedMaskCandidates, kFusedThreads, 0, stream>>>(
      codebook_words, family_mask_match.mutable_data_ptr<int>());

  FusedSequenceArgs args;
  args.sequences = sequences.const_data_ptr<float>();
  args.codebooks = codebook_words;
  args.overlap = constrained ? overlap->const_data_ptr<int64_t>() : nullptr;
  args.step_weights = weighted ? step_weights->const_data_ptr<float>() : nullptr;
  args.family_mask_match = family_mask_match.const_data_ptr<int>();
  args.frontiers = frontiers.mutable_data_ptr<float>();
  args.backpointers = backpointers.mutable_data_ptr<uint8_t>();
  args.boundary_banks = boundary_banks.mutable_data_ptr<uint8_t>();
  args.best_flat = best_flat.mutable_data_ptr<int>();
  args.squared_error = squared_error.mutable_data_ptr<float>();
  args.queue = queue.mutable_data_ptr<int>();
  args.batch = batch;
  args.family_batch = family_batch;
  args.constrained = constrained;

  const int total_units = batch * kFusedSegmentCount;
  const int grid = std::min(properties.multiProcessorCount, total_units);
  g_fused_w2_family_grid_dispatches.fetch_add(1);
  if (weighted) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        qvq_fused_w2_family_grid_kernel<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kFusedSharedBytes)));
    qvq_fused_w2_family_grid_kernel<true><<<grid, kFusedThreads, kFusedSharedBytes, stream>>>(args);
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        qvq_fused_w2_family_grid_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kFusedSharedBytes)));
    qvq_fused_w2_family_grid_kernel<false><<<grid, kFusedThreads, kFusedSharedBytes, stream>>>(args);
  }
  qvq_fused_w2_traceback_kernel<<<(batch + 127) / 128, 128, 0, stream>>>(
      backpointers.const_data_ptr<uint8_t>(),
      boundary_banks.const_data_ptr<uint8_t>(),
      best_flat.const_data_ptr<int>(),
      states.mutable_data_ptr<int64_t>(),
      segment_bank_ids.mutable_data_ptr<uint8_t>(),
      batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {states, squared_error, segment_bank_ids};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qvq_viterbi_v2_segment_banked_cuda_impl(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights,
    int kernel_mode,
    bool validate_values = true,
    int family_batch = 0,
    int bank_count_override = 0,
    bool midpoint_only = false,
    // Exact survivor-pruning policy from `ViterbiPruningConfig`. Zero is the
    // historical automatic behavior, so every direct low-level caller that
    // omits it keeps the pre-policy dispatch exactly.
    int64_t pruning_policy = kViterbiPruningAuto) {
  const bool g_only = kernel_mode != 0;
  bool cooperative = false;
  int cooperative_threads = 0;
  bool grid_parallel = kernel_mode == 2;
  TORCH_CHECK(sequences.is_cuda() && codebooks.is_cuda(),
              "segmented-bank V2 sequences and codebooks must be CUDA tensors");
  TORCH_CHECK(sequences.dim() == 3 && sequences.size(1) == 128 && sequences.size(2) == 2,
              "segmented-bank V2 sequences must have shape [batch, 128, 2]");
  TORCH_CHECK(codebooks.dim() == 3 &&
                  ((bank_count_override == 0 && (codebooks.size(0) == 2 || codebooks.size(0) == 4)) ||
                   (bank_count_override == 2 && codebooks.size(0) % 2 == 0)) &&
                  codebooks.size(1) == kStateCount && codebooks.size(2) == 2,
              "segmented-bank V2 codebooks must have shape [2|4, 65536, 2]");
  const int batch = static_cast<int>(sequences.size(0));
  const int bank_count = bank_count_override == 0 ? static_cast<int>(codebooks.size(0)) : bank_count_override;
  constexpr int steps = 128;
  TORCH_CHECK(batch > 0, "segmented-bank V2 sequences must contain at least one batch");
  TORCH_CHECK(family_batch == 0 || (family_batch > 0 && batch % family_batch == 0 &&
                  codebooks.size(0) == (batch / family_batch) * bank_count),
              "family-batched segmented V2 tensors have inconsistent family dimensions");
  TORCH_CHECK(transition_bits >= 2 && transition_bits <= 7,
              "segmented-bank V2 transition_bits must be in [2, 7]");
  TORCH_CHECK((bank_count == 2 && segment_steps == 16) ||
                  (bank_count == 4 && segment_steps == 32),
              "segmented-bank V2 requires two P32 banks or four P64 banks");
  TORCH_CHECK(sequences.scalar_type() == at::kFloat &&
                  (codebooks.scalar_type() == at::kHalf || codebooks.scalar_type() == at::kFloat),
              "segmented-bank V2 requires float32 sequences and float16 or float32 codebooks");
  TORCH_CHECK(sequences.device() == codebooks.device(),
              "segmented-bank V2 sequences and codebooks must share one CUDA device");
  TORCH_CHECK(sequences.is_contiguous() && codebooks.is_contiguous(),
              "segmented-bank V2 sequences and codebooks must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(sequences.data_ptr()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(codebooks.data_ptr()) % 4 == 0,
              "segmented-bank V2 tensors do not satisfy the native vector-load alignment contract");
  if (validate_values) {
    TORCH_CHECK(at::isfinite(sequences).all().item<bool>() &&
                    at::isfinite(codebooks).all().item<bool>(),
                "segmented-bank V2 sequences and codebooks must be finite");
  }

  const bool constrained = overlap.has_value();
  TORCH_CHECK(!midpoint_only || (!constrained && kernel_mode == 2),
              "midpoint-only segmented V2 requires an unconstrained grid recurrence");
  const at::Tensor overlap_tensor = constrained ? *overlap : at::Tensor();
  if (constrained) {
    TORCH_CHECK(overlap_tensor.is_cuda() && overlap_tensor.device() == sequences.device() &&
                    overlap_tensor.scalar_type() == at::kLong && overlap_tensor.is_contiguous(),
                "segmented-bank V2 overlap must be contiguous int64 on the sequence device");
    TORCH_CHECK(overlap_tensor.dim() == 1 && overlap_tensor.size(0) == batch,
                "segmented-bank V2 overlap must have shape [batch]");
    const int64_t overlap_limit = int64_t{1} << (16 - transition_bits);
    if (validate_values) {
      TORCH_CHECK(overlap_tensor.ge(0).all().item<bool>() &&
                      overlap_tensor.lt(overlap_limit).all().item<bool>(),
                  "segmented-bank V2 overlap values are outside the retained-state range");
    }
  }

  const bool weighted = step_weights.has_value();
  const at::Tensor step_weights_tensor = weighted ? *step_weights : at::Tensor();
  if (weighted) {
    TORCH_CHECK(step_weights_tensor.is_cuda() && step_weights_tensor.device() == sequences.device() &&
                    step_weights_tensor.scalar_type() == at::kFloat && step_weights_tensor.is_contiguous(),
                "segmented-bank V2 step weights must be contiguous float32 on the sequence device");
    TORCH_CHECK(step_weights_tensor.sizes() == at::IntArrayRef({batch, steps}),
                "segmented-bank V2 step weights must have shape [batch, 128]");
    if (validate_values) {
      TORCH_CHECK(at::isfinite(step_weights_tensor).all().item<bool>() &&
                      step_weights_tensor.ge(0).all().item<bool>(),
                  "segmented-bank V2 step weights must be finite and nonnegative");
    }
  }

  if (validate_values) {
    const double maximum_weight = weighted ? step_weights_tensor.abs().amax().item<double>() : 1.0;
    const double accumulation_terms = std::max(1.0, static_cast<double>(steps) * maximum_weight);
    const double safe_magnitude = std::sqrt(static_cast<double>(std::numeric_limits<float>::max()) /
                                             accumulation_terms) /
        (2.0 * std::sqrt(2.0));
    TORCH_CHECK(sequences.abs().amax().item<double>() <= safe_magnitude &&
                    codebooks.abs().amax().item<double>() <= safe_magnitude,
                "segmented-bank V2 magnitudes are too large for finite FP32 distance accumulation");
  }

  const c10::cuda::CUDAGuard device_guard(sequences.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, sequences.get_device()));
  TORCH_CHECK(properties.major >= 8,
              "segmented-bank V2 CUDA Viterbi requires compute capability >= 8.0");
  const int suffix_count = kStateCount >> static_cast<int>(transition_bits);
  const size_t grid_shared_bytes = 2 * static_cast<size_t>(suffix_count) * sizeof(float);
  // Ada exposes less opt-in shared memory per CTA than Ampere/Hopper. Preserve
  // the exact persistent G-only path when a rate's two frontiers do not fit.
  grid_parallel = grid_parallel && grid_shared_bytes <= properties.sharedMemPerBlockOptin;
  if (kernel_mode == 2 && !midpoint_only && transition_bits == 5 &&
      bank_count == 2 && segment_steps == 16) {
    const bool can_launch_256 = batch <= 7 && (codebooks.scalar_type() == at::kHalf
        ? can_launch_qvq_v2_segment_w25_cooperative<256, half, uint8_t>(batch, properties)
        : can_launch_qvq_v2_segment_w25_cooperative<256, float, uint8_t>(batch, properties));
    // The 512-thread variant remains faster through batch eight on the local
    // 124-SM sm_80 devices.  Beyond that point global-frontier barriers cost
    // more than the shared-memory grid kernel, even while the launch remains
    // cooperatively resident.
    const bool can_launch_512 = batch == 8 && (codebooks.scalar_type() == at::kHalf
        ? can_launch_qvq_v2_segment_w25_cooperative<512, half, uint8_t>(batch, properties)
        : can_launch_qvq_v2_segment_w25_cooperative<512, float, uint8_t>(batch, properties));
    cooperative_threads = can_launch_256 ? 256 : (can_launch_512 ? 512 : 0);
    cooperative = cooperative_threads != 0;
    grid_parallel = !cooperative && grid_parallel;
  }
  TORCH_CHECK(!midpoint_only || grid_parallel,
              "midpoint-only segmented V2 requires the grid-parallel recurrence");
  TORCH_CHECK(!cooperative || (!midpoint_only && transition_bits == 5 && bank_count == 2 && segment_steps == 16),
              "cooperative segmented V2 currently supports only full W2.5 B2-P32 recurrence");
  TORCH_CHECK(pruning_policy >= kViterbiPruningAuto && pruning_policy <= kViterbiPruningRequired,
              "segmented-bank V2 Viterbi pruning policy must be 0 (auto), 1 (off), 2 (auto+error), "
              "or 3 (required)");
  // Explicit configuration is authoritative.  The deprecated
  // GPTQMODEL_QVQ_DISABLE_OCTET_GRID A/B escape hatch is consulted only for the
  // two `auto` policies; `off` and `required` ignore it entirely.
  const bool pruning_strict =
      pruning_policy == kViterbiPruningAutoError || pruning_policy == kViterbiPruningRequired;
  const bool pruning_env_honored =
      pruning_policy == kViterbiPruningAuto || pruning_policy == kViterbiPruningAutoError;
  const bool pruning_env_disabled = pruning_env_honored && norm_rank_grid_disabled();
  const bool pruning_requested = pruning_policy != kViterbiPruningOff && !pruning_env_disabled;
  const FamilyGridDirectDistanceMode direct_distance_mode = family_grid_direct_distance_mode();
  const bool direct_family_distance = family_batch > 0 && codebooks.scalar_type() == at::kHalf &&
      transition_bits >= 5 && transition_bits <= 7 &&
      (direct_distance_mode == FamilyGridDirectDistanceMode::kAll ||
       (direct_distance_mode == FamilyGridDirectDistanceMode::kProvisional && !constrained) ||
       (direct_distance_mode == FamilyGridDirectDistanceMode::kFinal && constrained) ||
       (direct_distance_mode == FamilyGridDirectDistanceMode::kProvisionalLargeFinal &&
        (!constrained || family_batch >= 64)));
  const bool norm_rank_eligible =
      grid_parallel && !midpoint_only && !cooperative &&
      !weighted && !direct_family_distance &&
      codebooks.scalar_type() == at::kHalf &&
      (bank_count == 2 || bank_count == 4) &&
      segment_steps == (bank_count == 2 ? 16 : 32) &&
      (transition_bits == 5 || transition_bits == 6 || transition_bits == 7);
  // Refuse before any kernel selection so `required` and `fallback="error"`
  // can never be satisfied by a silent fallback -- including the fused
  // family-grid and cooperative launches that return early below.
  if (pruning_strict && !(pruning_requested && norm_rank_eligible)) {
    std::ostringstream reason;
    if (pruning_env_disabled) {
      reason << "the deprecated GPTQMODEL_QVQ_DISABLE_OCTET_GRID escape hatch disabled it";
    } else if (weighted) {
      reason << "weighted (step_weights) calls keep the exact baseline recurrence";
    } else if (direct_family_distance) {
      reason << "direct-distance calls use a different FP32 arithmetic order";
    } else if (codebooks.scalar_type() != at::kHalf) {
      reason << "only float16 codebooks are supported (got " << codebooks.scalar_type() << ")";
    } else if (bank_count != 2 && bank_count != 4) {
      reason << "only two-bank P32 and four-bank P64 stacks are supported (got bank_count="
             << bank_count << ")";
    } else if (segment_steps != (bank_count == 2 ? 16 : 32)) {
      reason << "segment_steps=" << segment_steps << " does not match bank_count=" << bank_count;
    } else if (transition_bits != 5 && transition_bits != 6 && transition_bits != 7) {
      reason << "only the benchmark-supported W2.5/W3/W3.5 rates (transition_bits 5, 6, or 7) are supported "
                "(got transition_bits=" << transition_bits << ")";
    } else if (midpoint_only) {
      reason << "midpoint-only traceback is not a norm-band shape";
    } else if (cooperative) {
      reason << "the cooperative W2.5 recurrence is not a norm-band shape";
    } else {
      reason << "this call does not use the grid-parallel segmented recurrence";
    }
    TORCH_CHECK(false,
                "QVQ exact norm-band Viterbi pruning was requested with "
                "`viterbi_pruning.mode='required'` or `fallback='error'`, but this call cannot "
                "use it: ", reason.str(),
                ". Set `viterbi_pruning.mode='auto'` with the default "
                "`fallback='baseline'` to keep the exact baseline recurrence instead.");
  }
  if ((pruning_policy == kViterbiPruningAuto || pruning_policy == kViterbiPruningAutoError) &&
      !(pruning_requested && norm_rank_eligible)) {
    g_norm_rank_baseline_fallbacks.fetch_add(1, std::memory_order_relaxed);
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(sequences.get_device());
  const bool collect_pruning_telemetry = norm_rank_telemetry_enabled();
  // Below ~40 sequences both paths are bound by the serial 127-step chain of a
  // single sequence and the reference layout (one CTA per bank) has twice the
  // per-sequence parallelism; measured crossover on the 124-SM sm_80 device.
  constexpr int fused_minimum_batch = 40;
  if (grid_parallel && !midpoint_only && family_batch > 0 && batch >= fused_minimum_batch &&
      transition_bits == 4 && bank_count == 2 && segment_steps == 16 &&
      codebooks.scalar_type() == at::kHalf && fused_w2_family_grid_supported(properties)) {
    return qvq_fused_w2_family_grid_launch(
        sequences, codebooks, constrained ? overlap : c10::nullopt, weighted ? step_weights : c10::nullopt,
        family_batch, properties, stream);
  }
  const int segments = steps / static_cast<int>(segment_steps);
  const auto float_options = sequences.options().dtype(at::kFloat);
  at::Tensor costs_a = at::empty(
      {batch, bank_count, g_only ? suffix_count : kStateCount}, float_options);
  at::Tensor costs_b = at::empty_like(costs_a);
  at::Tensor reduced = g_only
      ? at::Tensor()
      : at::empty({batch, bank_count, suffix_count}, float_options);
  // Four banks at E=7 have 4 * 128 = 512 boundary predecessors, so the old
  // flattened pointer needs nine bits. G-only stores its bank choice
  // separately, so its prefix always fits a byte.
  at::Tensor backpointers = at::empty(
      {batch, midpoint_only ? steps / 2 : steps - 1, bank_count, suffix_count},
      sequences.options().dtype(!g_only && transition_bits == 7 ? at::kShort : at::kByte));
  at::Tensor states = at::empty(
      midpoint_only ? at::IntArrayRef({batch}) : at::IntArrayRef({batch, steps}),
      sequences.options().dtype(at::kLong));
  at::Tensor segment_bank_ids = at::empty(
      midpoint_only ? at::IntArrayRef({0}) : at::IntArrayRef({batch, segments}),
      sequences.options().dtype(at::kByte));
  at::Tensor boundary_banks = g_only
      ? at::empty({batch, segments - 1, suffix_count}, sequences.options().dtype(at::kByte))
      : at::Tensor();
  at::Tensor squared_error = at::empty(midpoint_only ? at::IntArrayRef({0}) : at::IntArrayRef({batch}), float_options);
  at::Tensor codebook_norm = codebooks.scalar_type() == at::kHalf
      ? cached_banked_codebook_norm<2, half>(codebooks, stream)
      : cached_banked_codebook_norm<2, float>(codebooks, stream);
  const int64_t* overlap_ptr = constrained ? overlap_tensor.const_data_ptr<int64_t>() : nullptr;
  const float* weight_ptr = weighted ? step_weights_tensor.const_data_ptr<float>() : nullptr;

  if (cooperative) {
#define QVQ_V2_SEGMENT_COOPERATIVE_LAUNCH(THREADS, CODEBOOK_TYPE, CODEBOOK_POINTER)                   \
    do {                                                                                              \
      launch_qvq_v2_segment_w25_cooperative<THREADS, CODEBOOK_TYPE, uint8_t>(                         \
          sequences.const_data_ptr<float>(), CODEBOOK_POINTER, codebook_norm.const_data_ptr<float>(), \
          overlap_ptr, weight_ptr, costs_a.mutable_data_ptr<float>(), costs_b.mutable_data_ptr<float>(), \
          backpointers.mutable_data_ptr<uint8_t>(), boundary_banks.mutable_data_ptr<uint8_t>(),      \
          batch, family_batch, constrained, weighted, properties, stream);                            \
      qvq_v2_segment_grid_finalize_kernel<5, 2, 16, false, CODEBOOK_TYPE, uint8_t>                    \
          <<<batch, kThreads, 0, stream>>>(                                                           \
              sequences.const_data_ptr<float>(), CODEBOOK_POINTER, codebook_norm.const_data_ptr<float>(), \
              overlap_ptr, weight_ptr, costs_a.const_data_ptr<float>(),                              \
              backpointers.const_data_ptr<uint8_t>(), boundary_banks.const_data_ptr<uint8_t>(),      \
              states.mutable_data_ptr<int64_t>(), segment_bank_ids.mutable_data_ptr<uint8_t>(),      \
              squared_error.mutable_data_ptr<float>(), batch, family_batch, constrained, weighted);  \
    } while (0)
    if (codebooks.scalar_type() == at::kHalf) {
      if (cooperative_threads == 256) {
        QVQ_V2_SEGMENT_COOPERATIVE_LAUNCH(
            256, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()));
      } else {
        QVQ_V2_SEGMENT_COOPERATIVE_LAUNCH(
            512, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()));
      }
    } else {
      if (cooperative_threads == 256) {
        QVQ_V2_SEGMENT_COOPERATIVE_LAUNCH(256, float, codebooks.const_data_ptr<float>());
      } else {
        QVQ_V2_SEGMENT_COOPERATIVE_LAUNCH(512, float, codebooks.const_data_ptr<float>());
      }
    }
#undef QVQ_V2_SEGMENT_COOPERATIVE_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    record_norm_cache_use(codebooks, codebook_norm, stream);
    return {states, squared_error, segment_bank_ids};
  }

#define QVQ_V2_SEGMENT_LAUNCH(BITS, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE)                    \
  qvq_v2_segment_banked_kernel<BITS, CODEBOOK_TYPE, POINTER_TYPE><<<batch, kThreads, 0, stream>>>(    \
      sequences.const_data_ptr<float>(), CODEBOOK_POINTER, codebook_norm.const_data_ptr<float>(),     \
      overlap_ptr, weight_ptr, costs_a.mutable_data_ptr<float>(), costs_b.mutable_data_ptr<float>(),   \
      reduced.mutable_data_ptr<float>(), backpointers.mutable_data_ptr<POINTER_TYPE>(),                \
      states.mutable_data_ptr<int64_t>(), segment_bank_ids.mutable_data_ptr<uint8_t>(),                \
      squared_error.mutable_data_ptr<float>(), steps, bank_count, static_cast<int>(segment_steps),     \
      constrained, weighted)
#define QVQ_V2_SEGMENT_G_LAUNCH(                                                                      \
    BITS, BANKS, SEGMENT_STEPS, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE)                         \
  qvq_v2_segment_g_kernel<                                                                            \
      BITS, BANKS, SEGMENT_STEPS, CODEBOOK_TYPE, POINTER_TYPE><<<batch, kThreads, 0, stream>>>(        \
      sequences.const_data_ptr<float>(), CODEBOOK_POINTER, codebook_norm.const_data_ptr<float>(),     \
      overlap_ptr, weight_ptr, costs_a.mutable_data_ptr<float>(), costs_b.mutable_data_ptr<float>(),   \
      backpointers.mutable_data_ptr<POINTER_TYPE>(), boundary_banks.mutable_data_ptr<uint8_t>(),       \
      states.mutable_data_ptr<int64_t>(), segment_bank_ids.mutable_data_ptr<uint8_t>(),                \
      squared_error.mutable_data_ptr<float>(), constrained, weighted)
#define QVQ_V2_SEGMENT_G_DISPATCH(BITS, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE)                \
  do {                                                                                                 \
    if (bank_count == 2) {                                                                             \
      QVQ_V2_SEGMENT_G_LAUNCH(                                                                         \
          BITS, 2, 16, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE);                                 \
    } else {                                                                                           \
      QVQ_V2_SEGMENT_G_LAUNCH(                                                                         \
          BITS, 4, 32, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE);                                 \
    }                                                                                                  \
  } while (0)
#define QVQ_V2_SEGMENT_GRID_LAUNCH(                                                                  \
    BITS, BANKS, SEGMENT_STEPS, MIDPOINT_ONLY, DIRECT_DISTANCE, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE) \
  do {                                                                                                \
    constexpr bool qvq_fuse_boundary =                                                               \
        (BANKS == 2 && BITS != 7) || (BANKS == 4 && BITS >= 3 && BITS <= 6);                         \
    constexpr int qvq_segment_shared_bytes =                                                         \
        2 * (1 << (16 - BITS)) * sizeof(float) +                                                      \
        (BITS == 7 ? kThreads * (sizeof(float) + sizeof(int)) : 0);                                  \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                                              \
        qvq_v2_segment_grid_kernel<                                                                   \
            BITS, BANKS, SEGMENT_STEPS, qvq_fuse_boundary, MIDPOINT_ONLY, DIRECT_DISTANCE, CODEBOOK_TYPE, POINTER_TYPE>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, qvq_segment_shared_bytes));                      \
    float* qvq_frontier_a = costs_a.mutable_data_ptr<float>();                                        \
    float* qvq_frontier_b = costs_b.mutable_data_ptr<float>();                                        \
    qvq_v2_segment_grid_kernel<                                                                       \
        BITS, BANKS, SEGMENT_STEPS, qvq_fuse_boundary, MIDPOINT_ONLY, DIRECT_DISTANCE, CODEBOOK_TYPE, POINTER_TYPE><<< \
        batch * BANKS, kThreads, qvq_segment_shared_bytes, stream>>>(                                 \
        sequences.const_data_ptr<float>(), CODEBOOK_POINTER, codebook_norm.const_data_ptr<float>(),   \
        overlap_ptr, weight_ptr, qvq_frontier_b, qvq_frontier_a,                                     \
        backpointers.mutable_data_ptr<POINTER_TYPE>(), boundary_banks.mutable_data_ptr<uint8_t>(),     \
        batch, family_batch, 0, constrained, weighted);                                               \
    for (int segment = 1; segment < segments; ++segment) {                                            \
      const float* qvq_segment_input = segment % 2 == 1 ? qvq_frontier_a : qvq_frontier_b;            \
      float* qvq_segment_output = segment % 2 == 1 ? qvq_frontier_b : qvq_frontier_a;                 \
      if constexpr (!qvq_fuse_boundary) {                                                             \
        qvq_v2_segment_boundary_kernel<BITS, BANKS, SEGMENT_STEPS><<<batch, kThreads, 0, stream>>>(   \
            qvq_segment_input, boundary_banks.mutable_data_ptr<uint8_t>(),                            \
            batch, segment - 1);                                                                      \
      }                                                                                               \
      qvq_v2_segment_grid_kernel<                                                                     \
          BITS, BANKS, SEGMENT_STEPS, qvq_fuse_boundary, MIDPOINT_ONLY, DIRECT_DISTANCE, CODEBOOK_TYPE, POINTER_TYPE><<< \
          batch * BANKS, kThreads, qvq_segment_shared_bytes, stream>>>(                               \
          sequences.const_data_ptr<float>(), CODEBOOK_POINTER, codebook_norm.const_data_ptr<float>(), \
          overlap_ptr, weight_ptr, qvq_segment_input, qvq_segment_output,                             \
          backpointers.mutable_data_ptr<POINTER_TYPE>(), boundary_banks.mutable_data_ptr<uint8_t>(),   \
          batch, family_batch, segment, constrained, weighted);                                       \
    }                                                                                                 \
    const float* qvq_final_frontier = segments % 2 == 1 ? qvq_frontier_a : qvq_frontier_b;           \
    qvq_v2_segment_grid_finalize_kernel<                                                              \
        BITS, BANKS, SEGMENT_STEPS, MIDPOINT_ONLY, CODEBOOK_TYPE, POINTER_TYPE><<<                    \
        batch, kThreads, 0, stream>>>(                                                                \
        sequences.const_data_ptr<float>(), CODEBOOK_POINTER, codebook_norm.const_data_ptr<float>(),   \
        overlap_ptr, weight_ptr, qvq_final_frontier,                                                  \
        backpointers.const_data_ptr<POINTER_TYPE>(), boundary_banks.const_data_ptr<uint8_t>(),         \
        states.mutable_data_ptr<int64_t>(), segment_bank_ids.mutable_data_ptr<uint8_t>(),              \
        squared_error.mutable_data_ptr<float>(), batch, family_batch, constrained, weighted);          \
  } while (0)
#define QVQ_V2_SEGMENT_GRID_DISPATCH(                                                                 \
    BITS, MIDPOINT_ONLY, DIRECT_DISTANCE, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE)              \
  do {                                                                                                \
    if (bank_count == 2) {                                                                            \
      QVQ_V2_SEGMENT_GRID_LAUNCH(                                                                     \
          BITS, 2, 16, MIDPOINT_ONLY, DIRECT_DISTANCE, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE); \
    } else {                                                                                          \
      QVQ_V2_SEGMENT_GRID_LAUNCH(                                                                     \
          BITS, 4, 32, MIDPOINT_ONLY, DIRECT_DISTANCE, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE); \
    }                                                                                                 \
  } while (0)
#define QVQ_V2_SEGMENT_DISPATCH(BITS, POINTER_TYPE)                                                    \
  do {                                                                                                 \
    if (codebooks.scalar_type() == at::kHalf) {                                                        \
      if (grid_parallel) {                                                                              \
        if (direct_family_distance) {                                                                    \
          if (midpoint_only) {                                                                           \
            QVQ_V2_SEGMENT_GRID_DISPATCH(                                                                \
                BITS, true, true, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()), POINTER_TYPE); \
          } else {                                                                                       \
            QVQ_V2_SEGMENT_GRID_DISPATCH(                                                                \
                BITS, false, true, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()), POINTER_TYPE); \
          }                                                                                              \
        } else if (midpoint_only) {                                                                      \
          QVQ_V2_SEGMENT_GRID_DISPATCH(                                                                  \
              BITS, true, false, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()), POINTER_TYPE); \
        } else {                                                                                         \
          QVQ_V2_SEGMENT_GRID_DISPATCH(                                                                  \
              BITS, false, false, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()), POINTER_TYPE); \
        }                                                                                                \
      } else if (g_only) {                                                                              \
        QVQ_V2_SEGMENT_G_DISPATCH(                                                                      \
            BITS, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()), POINTER_TYPE);      \
      } else {                                                                                          \
        QVQ_V2_SEGMENT_LAUNCH(                                                                          \
            BITS, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()), POINTER_TYPE);      \
      }                                                                                                 \
    } else {                                                                                           \
      if (grid_parallel) {                                                                              \
        if (midpoint_only) {                                                                             \
          QVQ_V2_SEGMENT_GRID_DISPATCH(                                                                  \
              BITS, true, false, float, codebooks.const_data_ptr<float>(), POINTER_TYPE);               \
        } else {                                                                                         \
          QVQ_V2_SEGMENT_GRID_DISPATCH(                                                                  \
              BITS, false, false, float, codebooks.const_data_ptr<float>(), POINTER_TYPE);              \
        }                                                                                                \
      } else if (g_only) {                                                                              \
        QVQ_V2_SEGMENT_G_DISPATCH(                                                                      \
            BITS, float, codebooks.const_data_ptr<float>(), POINTER_TYPE);                              \
      } else {                                                                                          \
        QVQ_V2_SEGMENT_LAUNCH(BITS, float, codebooks.const_data_ptr<float>(), POINTER_TYPE);           \
      }                                                                                                 \
    }                                                                                                  \
  } while (0)
  // Exact norm-rank contiguous-band fast path for the unweighted,
  // half-codebook grid recurrence at W2.5/W3/W3.5. Every other configuration
  // (weighted, direct-distance, cooperative,
  // midpoint-only, fp32 codebooks, other rates) falls through to the
  // unmodified baseline below, which remains the exact reference. Family
  // batches use physical-bank table addressing while retaining logical-bank
  // recurrence and traceback layouts.
#define QVQ_V2_NORM_RANK_DISPATCH(BITS, BANKS, SEGMENT_STEPS, WIDTH)                              \
  do {                                                                                             \
    const NormRankTables qvq_norm_rank_tables =                                                    \
        cached_norm_rank_tables<BITS, WIDTH>(                                                      \
            codebooks, codebook_norm, static_cast<int>(codebooks.size(0)), stream);                 \
    launch_qvq_v2_segment_norm_rank_segments<BITS, BANKS, SEGMENT_STEPS, WIDTH>(                   \
        sequences.const_data_ptr<float>(),                                                         \
        reinterpret_cast<const uint64_t*>(codebook_norm.const_data_ptr<float>()),                  \
        qvq_norm_rank_tables, overlap_ptr,                                                         \
        costs_a.mutable_data_ptr<float>(), costs_b.mutable_data_ptr<float>(),                      \
        backpointers.mutable_data_ptr<uint8_t>(), boundary_banks.mutable_data_ptr<uint8_t>(),      \
        batch, family_batch, constrained, stream, collect_pruning_telemetry,                       \
        properties.multiProcessorCount);                                                           \
    qvq_v2_segment_grid_finalize_kernel<BITS, BANKS, SEGMENT_STEPS, false, half, uint8_t>          \
        <<<batch, kThreads, 0, stream>>>(                                                          \
            sequences.const_data_ptr<float>(),                                                     \
            reinterpret_cast<const half*>(codebooks.const_data_ptr()),                             \
            codebook_norm.const_data_ptr<float>(), overlap_ptr, weight_ptr,                        \
            segments % 2 == 1 ? costs_a.const_data_ptr<float>()                                    \
                              : costs_b.const_data_ptr<float>(),                                   \
            backpointers.const_data_ptr<uint8_t>(),                                                \
            boundary_banks.const_data_ptr<uint8_t>(),                                              \
            states.mutable_data_ptr<int64_t>(),                                                    \
            segment_bank_ids.mutable_data_ptr<uint8_t>(),                                          \
            squared_error.mutable_data_ptr<float>(), batch, family_batch,                          \
            constrained, weighted);                                                                \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                \
    record_norm_cache_use(codebooks, codebook_norm, stream);                                       \
    record_norm_rank_cache_use(qvq_norm_rank_tables, codebooks.get_device(), stream);              \
    g_norm_rank_grid_dispatches.fetch_add(1, std::memory_order_relaxed);                            \
    return {states, squared_error, segment_bank_ids};                                              \
  } while (0)
  // W2.5 through W3.5: at shift 4 the 16-entry candidate list is too short
  // for the band overhead to pay for itself (measured 0.88-1.15x on real
  // tiles), so W2.0 keeps the pristine grid recurrence unconditionally.
  if (pruning_requested && norm_rank_eligible) {
    if (bank_count == 2) {
      if (transition_bits == 5) {
        QVQ_V2_NORM_RANK_DISPATCH(5, 2, 16, kNormRankChunkWidthW25);
      } else if (transition_bits == 6) {
        QVQ_V2_NORM_RANK_DISPATCH(6, 2, 16, kNormRankChunkWidthW3);
      } else {
        QVQ_V2_NORM_RANK_DISPATCH(7, 2, 16, kNormRankChunkWidthW35);
      }
    } else {
      if (transition_bits == 5) {
        QVQ_V2_NORM_RANK_DISPATCH(5, 4, 32, kNormRankChunkWidthW25);
      } else if (transition_bits == 6) {
        QVQ_V2_NORM_RANK_DISPATCH(6, 4, 32, kNormRankChunkWidthW3);
      } else {
        QVQ_V2_NORM_RANK_DISPATCH(7, 4, 32, kNormRankChunkWidthW35);
      }
    }
  }
#undef QVQ_V2_NORM_RANK_DISPATCH
  switch (transition_bits) {
    case 2: QVQ_V2_SEGMENT_DISPATCH(2, uint8_t); break;
    case 3: QVQ_V2_SEGMENT_DISPATCH(3, uint8_t); break;
    case 4: QVQ_V2_SEGMENT_DISPATCH(4, uint8_t); break;
    case 5: QVQ_V2_SEGMENT_DISPATCH(5, uint8_t); break;
    case 6: QVQ_V2_SEGMENT_DISPATCH(6, uint8_t); break;
    case 7:
      if (g_only) {
        QVQ_V2_SEGMENT_DISPATCH(7, uint8_t);
      } else {
        QVQ_V2_SEGMENT_DISPATCH(7, int16_t);
      }
      break;
  }
#undef QVQ_V2_SEGMENT_DISPATCH
#undef QVQ_V2_SEGMENT_GRID_DISPATCH
#undef QVQ_V2_SEGMENT_GRID_LAUNCH
#undef QVQ_V2_SEGMENT_G_DISPATCH
#undef QVQ_V2_SEGMENT_G_LAUNCH
#undef QVQ_V2_SEGMENT_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  record_norm_cache_use(codebooks, codebook_norm, stream);
  return {states, squared_error, segment_bank_ids};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qvq_viterbi_v2_segment_banked_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights,
    int64_t pruning_policy) {
  return qvq_viterbi_v2_segment_banked_cuda_impl(
      sequences, codebooks, transition_bits, segment_steps, overlap, step_weights, 0,
      true, 0, 0, false, pruning_policy);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qvq_viterbi_v2_segment_g_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights,
    int64_t pruning_policy) {
  return qvq_viterbi_v2_segment_banked_cuda_impl(
      sequences, codebooks, transition_bits, segment_steps, overlap, step_weights, 1,
      true, 0, 0, false, pruning_policy);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qvq_viterbi_v2_segment_grid_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights,
    int64_t pruning_policy) {
  return qvq_viterbi_v2_segment_banked_cuda_impl(
      sequences, codebooks, transition_bits, segment_steps, overlap, step_weights, 2,
      true, 0, 0, false, pruning_policy);
}

// Internal hot-path entry point. YAQA validates one complete anti-diagonal
// before chunking and derives overlap from a native traceback, so repeating
// value reductions in both tail-biting passes only serializes the launch
// stream. Keep all structural checks above active; only synchronized value
// scans are skipped here.
std::tuple<at::Tensor, at::Tensor, at::Tensor> qvq_viterbi_v2_segment_grid_trusted_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights,
    int64_t pruning_policy) {
  return qvq_viterbi_v2_segment_banked_cuda_impl(
      sequences, codebooks, transition_bits, segment_steps, overlap, step_weights, 2, false,
      0, 0, false, pruning_policy);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qvq_viterbi_v2_segment_tail_trusted_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& step_weights,
    int64_t pruning_policy) {
  constexpr int64_t midpoint = 64;
  auto rotated_sequences = at::roll(sequences, {midpoint}, {1}).contiguous();
  c10::optional<at::Tensor> rotated_weights = c10::nullopt;
  if (step_weights.has_value()) {
    rotated_weights = at::roll(*step_weights, {midpoint}, {1}).contiguous();
  }
  // A single two-bank codebook stack is one family whose batch is the whole
  // batch; that lets the W2 two-pass tail share the fused family-grid kernel.
  const int single_family_batch =
      (codebooks.dim() == 3 && codebooks.size(0) == 2) ? static_cast<int>(sequences.size(0)) : 0;
  auto provisional = qvq_viterbi_v2_segment_banked_cuda_impl(
      rotated_sequences,
      codebooks,
      transition_bits,
      segment_steps,
      c10::nullopt,
      rotated_weights,
      2,
      false,
      single_family_batch,
      0,
      false,
      pruning_policy);
  const int64_t overlap_mask = (int64_t{1} << (16 - transition_bits)) - 1;
  auto overlap = std::get<0>(provisional).select(1, midpoint - 1).bitwise_and(overlap_mask).contiguous();
  return qvq_viterbi_v2_segment_banked_cuda_impl(
      sequences, codebooks, transition_bits, segment_steps, overlap, step_weights, 2, false,
      single_family_batch, 0, false, pruning_policy);
}

at::Tensor qvq_viterbi_v2_segment_midpoint_trusted_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& step_weights) {
  auto result = qvq_viterbi_v2_segment_banked_cuda_impl(
      sequences,
      codebooks,
      transition_bits,
      segment_steps,
      c10::nullopt,
      step_weights,
      2,
      false,
      0,
      0,
      true);
  return std::get<0>(result);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qvq_viterbi_v2_segment_family_grid_trusted_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights,
    int64_t pruning_policy) {
  TORCH_CHECK(sequences.dim() == 4 && sequences.size(2) == 128 && sequences.size(3) == 2,
              "family-batched segmented V2 sequences must have shape [families, batch, 128, 2]");
  TORCH_CHECK(codebooks.dim() == 4 && codebooks.size(0) == sequences.size(0) && codebooks.size(1) == 2 &&
                  codebooks.size(2) == kStateCount && codebooks.size(3) == 2,
              "family-batched segmented V2 codebooks must have shape [families, 2, 65536, 2]");
  const int64_t families = sequences.size(0);
  const int64_t family_batch = sequences.size(1);
  const at::Tensor flat_sequences = sequences.view({families * family_batch, 128, 2});
  const at::Tensor flat_codebooks = codebooks.view({families * 2, kStateCount, 2});
  c10::optional<at::Tensor> flat_overlap = c10::nullopt;
  if (overlap.has_value()) {
    TORCH_CHECK(overlap->sizes() == at::IntArrayRef({families, family_batch}),
                "family-batched segmented V2 overlap must have shape [families, batch]");
    flat_overlap = overlap->view({families * family_batch});
  }
  c10::optional<at::Tensor> flat_weights = c10::nullopt;
  if (step_weights.has_value()) {
    TORCH_CHECK(step_weights->sizes() == at::IntArrayRef({families, family_batch, 128}),
                "family-batched segmented V2 weights must have shape [families, batch, 128]");
    flat_weights = step_weights->view({families * family_batch, 128});
  }
  auto result = qvq_viterbi_v2_segment_banked_cuda_impl(
      flat_sequences,
      flat_codebooks,
      transition_bits,
      segment_steps,
      flat_overlap,
      flat_weights,
      2,
      false,
      static_cast<int>(family_batch),
      2,
      false,
      pruning_policy);
  return {
      std::get<0>(result).view({families, family_batch, 128}),
      std::get<1>(result).view({families, family_batch}),
      std::get<2>(result).view({families, family_batch, 8}),
  };
}

at::Tensor qvq_viterbi_v2_segment_family_midpoint_trusted_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& step_weights) {
  TORCH_CHECK(sequences.dim() == 4 && sequences.size(2) == 128 && sequences.size(3) == 2,
              "family-batched segmented V2 sequences must have shape [families, batch, 128, 2]");
  TORCH_CHECK(codebooks.dim() == 4 && codebooks.size(0) == sequences.size(0) && codebooks.size(1) == 2 &&
                  codebooks.size(2) == kStateCount && codebooks.size(3) == 2,
              "family-batched segmented V2 codebooks must have shape [families, 2, 65536, 2]");
  const int64_t families = sequences.size(0);
  const int64_t family_batch = sequences.size(1);
  const at::Tensor flat_sequences = sequences.view({families * family_batch, 128, 2});
  const at::Tensor flat_codebooks = codebooks.view({families * 2, kStateCount, 2});
  c10::optional<at::Tensor> flat_weights = c10::nullopt;
  if (step_weights.has_value()) {
    TORCH_CHECK(step_weights->sizes() == at::IntArrayRef({families, family_batch, 128}),
                "family-batched segmented V2 weights must have shape [families, batch, 128]");
    flat_weights = step_weights->view({families * family_batch, 128});
  }
  auto result = qvq_viterbi_v2_segment_banked_cuda_impl(
      flat_sequences,
      flat_codebooks,
      transition_bits,
      segment_steps,
      c10::nullopt,
      flat_weights,
      2,
      false,
      static_cast<int>(family_batch),
      2,
      true);
  return std::get<0>(result).view({families, family_batch});
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def("viterbi(Tensor sequences, Tensor codebook, int transition_bits, Tensor? overlap=None, "
        "Tensor? step_weights=None) -> (Tensor, Tensor)");
  m.def("viterbi_trusted(Tensor sequences, Tensor codebook, int transition_bits, Tensor? overlap=None, "
        "Tensor? step_weights=None) -> (Tensor, Tensor)");
  m.def("viterbi_tail_trusted(Tensor sequences, Tensor codebook, int transition_bits, "
        "Tensor? step_weights=None) -> (Tensor, Tensor)");
  m.def("viterbi_v4(Tensor sequences, Tensor codebook, int transition_bits, Tensor? overlap=None, "
        "Tensor? step_weights=None) -> (Tensor, Tensor)");
  m.def("viterbi_banked(Tensor sequences, Tensor codebooks, int transition_bits, Tensor? overlap=None, "
        "Tensor? step_weights=None) -> (Tensor, Tensor)");
  // `pruning_policy` defaults to 0 (auto), so pre-policy callers keep the
  // historical automatic norm-band behavior with no source change.
  m.def("viterbi_v2_segment_banked(Tensor sequences, Tensor codebooks, int transition_bits, int segment_steps, "
        "Tensor? overlap=None, Tensor? step_weights=None, int pruning_policy=0) -> (Tensor, Tensor, Tensor)");
  m.def("viterbi_v2_segment_g(Tensor sequences, Tensor codebooks, int transition_bits, int segment_steps, "
        "Tensor? overlap=None, Tensor? step_weights=None, int pruning_policy=0) -> (Tensor, Tensor, Tensor)");
  m.def("viterbi_v2_segment_grid(Tensor sequences, Tensor codebooks, int transition_bits, int segment_steps, "
        "Tensor? overlap=None, Tensor? step_weights=None, int pruning_policy=0) -> (Tensor, Tensor, Tensor)");
  m.def("viterbi_v2_segment_grid_trusted(Tensor sequences, Tensor codebooks, int transition_bits, int segment_steps, "
        "Tensor? overlap=None, Tensor? step_weights=None, int pruning_policy=0) -> (Tensor, Tensor, Tensor)");
  m.def("viterbi_v2_segment_tail_trusted(Tensor sequences, Tensor codebooks, int transition_bits, int segment_steps, "
        "Tensor? step_weights=None, int pruning_policy=0) -> (Tensor, Tensor, Tensor)");
  m.def("viterbi_v2_segment_midpoint_trusted(Tensor sequences, Tensor codebooks, int transition_bits, "
        "int segment_steps, Tensor? step_weights=None) -> Tensor");
  m.def("viterbi_v2_segment_family_grid_trusted(Tensor sequences, Tensor codebooks, int transition_bits, "
        "int segment_steps, Tensor? overlap=None, Tensor? step_weights=None, int pruning_policy=0) "
        "-> (Tensor, Tensor, Tensor)");
  m.def("viterbi_v2_segment_family_midpoint_trusted(Tensor sequences, Tensor codebooks, "
        "int transition_bits, int segment_steps, Tensor? step_weights=None) -> Tensor");
  // No tensor arguments, so this is a catch-all (dispatch-key-free) kernel.
  m.def("fused_family_grid_dispatch_count() -> int", &qvq_fused_family_grid_dispatch_count);
  m.def("norm_rank_grid_dispatch_count() -> int", &qvq_norm_rank_grid_dispatch_count);
  m.def("norm_rank_telemetry_snapshot() -> int[]", &qvq_norm_rank_telemetry_snapshot);
  m.def("norm_rank_cache_size() -> int", &qvq_norm_rank_cache_size);
  m.def("norm_cache_size() -> int", &qvq_norm_cache_size);
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CUDA, m) {
  m.impl("viterbi", &qvq_viterbi_cuda);
  m.impl("viterbi_trusted", &qvq_viterbi_trusted_cuda);
  m.impl("viterbi_tail_trusted", &qvq_viterbi_tail_trusted_cuda);
  m.impl("viterbi_v4", &qvq_viterbi_v4_cuda);
  m.impl("viterbi_banked", &qvq_viterbi_banked_cuda);
  m.impl("viterbi_v2_segment_banked", &qvq_viterbi_v2_segment_banked_cuda);
  m.impl("viterbi_v2_segment_g", &qvq_viterbi_v2_segment_g_cuda);
  m.impl("viterbi_v2_segment_grid", &qvq_viterbi_v2_segment_grid_cuda);
  m.impl("viterbi_v2_segment_grid_trusted", &qvq_viterbi_v2_segment_grid_trusted_cuda);
  m.impl("viterbi_v2_segment_tail_trusted", &qvq_viterbi_v2_segment_tail_trusted_cuda);
  m.impl("viterbi_v2_segment_midpoint_trusted", &qvq_viterbi_v2_segment_midpoint_trusted_cuda);
  m.impl("viterbi_v2_segment_family_grid_trusted", &qvq_viterbi_v2_segment_family_grid_trusted_cuda);
  m.impl("viterbi_v2_segment_family_midpoint_trusted", &qvq_viterbi_v2_segment_family_midpoint_trusted_cuda);
}
