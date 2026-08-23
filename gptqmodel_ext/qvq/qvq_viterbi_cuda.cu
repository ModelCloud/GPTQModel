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
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <limits>
#include <mutex>
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

void evict_norm_cache_entry() {
  if (g_norm_cache.size() < kMaxNormCacheEntries) {
    return;
  }
  C10_CUDA_CHECK(cudaEventDestroy(g_norm_cache.front().ready));
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
            emission<2, CodebookScalar>(
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
              emission<2, CodebookScalar>(
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
// Arithmetic is identical to emission<2, half>() and the reference recurrence:
// same FP32 operation sequence, same lowest-prefix tie precedence (bank-1
// candidates are visited in ascending bank-1 prefix order inside each block of
// four prefixes and the block winners are merged with lower_pair()), same
// bank-0-first boundary merge and bank-major / state-major terminal argmin.
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

// Bit-identical to emission<2, half>() with the norm recomputed from the fp16
// pair: codebook_norm_value() is fadd(fadd(0, c0*c0), c1*c1) and fadd(0, v) == v
// for v = c0*c0 >= +0.  fsub(a, fmul(2, d)) == fma(-2, d, a) because 2*d is exact.
template <bool Weighted>
__device__ __forceinline__ float fused_emission(
    float t0, float t1, float target_norm, uint32_t code_bits, float weight) {
  const half2 code2_half = *reinterpret_cast<const half2*>(&code_bits);
  const float2 code2 = __half22float2(code2_half);
  const float norm = __fadd_rn(__fmul_rn(code2.x, code2.x), __fmul_rn(code2.y, code2.y));
  float dot = __fmaf_rn(t0, code2.x, 0.0f);
  dot = __fmaf_rn(t1, code2.y, dot);
  const float distance = __fmaf_rn(-2.0f, dot, __fadd_rn(target_norm, norm));
  const float clipped = fmaxf(distance, 0.0f);
  if constexpr (Weighted) {
    return __fmul_rn(clipped, weight);
  } else {
    return clipped;
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
    float t0, float t1, float target_norm, float weight,
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
        emissions[i][jj] = fused_emission<Weighted>(
            t0, t1, target_norm, fused_code_at(codes[i], jj), weight);
      }
    }
    // Bank 0: ascending prefix order, strict '<' keeps the lowest prefix on ties.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = block * 4 + i;
      const float predecessor = g0[h * 256 + map.row];
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const float candidate = __fadd_rn(predecessor, emissions[i][jj]);
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
        const float candidate = __fadd_rn(predecessor, emissions[i][jj]);
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
    float t0, float t1, float target_norm, float weight,
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
      const float candidate0 = __fadd_rn(
          predecessor0,
          fused_emission<Weighted>(t0, t1, target_norm, fused_code_at(codes0, jj), weight));
      if (candidate0 < best0[jj]) {
        best0[jj] = candidate0;
        best_h0[jj] = h;
      }
      const float candidate1 = __fadd_rn(
          predecessor1,
          fused_emission<Weighted>(t0, t1, target_norm, fused_code_at(codes1, jj), weight));
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
    const float target_norm = __fadd_rn(__fmul_rn(t0, t0), __fmul_rn(t1, t1));
    const float weight = Weighted ? sequence_weights[step] : 1.0f;
    float best0[4], best1[4];
    int best_h0[4], best_h1[4];
    if constexpr (Shared) {
      fused_step_shared<Mask, Weighted>(
          map, shared_codes, global_codes0, g0, g1, t0, t1, target_norm, weight,
          best0, best_h0, best1, best_h1);
    } else {
      fused_step_general<Weighted>(
          map, shared_codes, global_codes0, global_codes1, g0, g1, t0, t1, target_norm, weight,
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
    const float target_norm = __fadd_rn(__fmul_rn(t0, t0), __fmul_rn(t1, t1));
    const float weight = Weighted ? sequence_weights[127] : 1.0f;
    float best = CUDART_INF_F;
    int best_flat = 2 * kStateCount;
    for (int flat = thread; flat < 2 * kStateCount; flat += kFusedThreads) {
      const int bank = flat >> 16;
      const int state = flat & (kStateCount - 1);
      const uint32_t code = bank == 0 ? __ldg(global_codes0 + state) : __ldg(global_codes1 + state);
      const float* g = bank == 0 ? g0 : g1;
      float candidate = __fadd_rn(
          g[state >> kFusedShift],
          fused_emission<Weighted>(t0, t1, target_norm, code, weight));
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
    bool midpoint_only = false) {
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
  TORCH_CHECK(!midpoint_only || (grid_parallel && properties.major == 8 && properties.minor == 0),
              "midpoint-only segmented V2 currently requires an sm_80 grid recurrence");
  TORCH_CHECK(!cooperative || (!midpoint_only && transition_bits == 5 && bank_count == 2 && segment_steps == 16),
              "cooperative segmented V2 currently supports only full W2.5 B2-P32 recurrence");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(sequences.get_device());
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
    BITS, BANKS, SEGMENT_STEPS, MIDPOINT_ONLY, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE)        \
  do {                                                                                                \
    constexpr bool qvq_fuse_boundary =                                                               \
        (BANKS == 2 && BITS != 7) || (BANKS == 4 && BITS >= 3 && BITS <= 6);                         \
    constexpr int qvq_segment_shared_bytes =                                                         \
        2 * (1 << (16 - BITS)) * sizeof(float) +                                                      \
        (BITS == 7 ? kThreads * (sizeof(float) + sizeof(int)) : 0);                                  \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                                              \
        qvq_v2_segment_grid_kernel<                                                                   \
            BITS, BANKS, SEGMENT_STEPS, qvq_fuse_boundary, MIDPOINT_ONLY, CODEBOOK_TYPE, POINTER_TYPE>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, qvq_segment_shared_bytes));                      \
    float* qvq_frontier_a = costs_a.mutable_data_ptr<float>();                                        \
    float* qvq_frontier_b = costs_b.mutable_data_ptr<float>();                                        \
    qvq_v2_segment_grid_kernel<                                                                       \
        BITS, BANKS, SEGMENT_STEPS, qvq_fuse_boundary, MIDPOINT_ONLY, CODEBOOK_TYPE, POINTER_TYPE><<< \
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
          BITS, BANKS, SEGMENT_STEPS, qvq_fuse_boundary, MIDPOINT_ONLY, CODEBOOK_TYPE, POINTER_TYPE><<< \
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
    BITS, MIDPOINT_ONLY, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE)                               \
  do {                                                                                                \
    if (bank_count == 2) {                                                                            \
      QVQ_V2_SEGMENT_GRID_LAUNCH(                                                                     \
          BITS, 2, 16, MIDPOINT_ONLY, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE);                 \
    } else {                                                                                          \
      QVQ_V2_SEGMENT_GRID_LAUNCH(                                                                     \
          BITS, 4, 32, MIDPOINT_ONLY, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE);                 \
    }                                                                                                 \
  } while (0)
#define QVQ_V2_SEGMENT_DISPATCH(BITS, POINTER_TYPE)                                                    \
  do {                                                                                                 \
    if (codebooks.scalar_type() == at::kHalf) {                                                        \
      if (grid_parallel) {                                                                              \
        if (midpoint_only) {                                                                             \
          QVQ_V2_SEGMENT_GRID_DISPATCH(                                                                  \
              BITS, true, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()), POINTER_TYPE); \
        } else {                                                                                         \
          QVQ_V2_SEGMENT_GRID_DISPATCH(                                                                  \
              BITS, false, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()), POINTER_TYPE); \
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
              BITS, true, float, codebooks.const_data_ptr<float>(), POINTER_TYPE);                      \
        } else {                                                                                         \
          QVQ_V2_SEGMENT_GRID_DISPATCH(                                                                  \
              BITS, false, float, codebooks.const_data_ptr<float>(), POINTER_TYPE);                     \
        }                                                                                                \
      } else if (g_only) {                                                                              \
        QVQ_V2_SEGMENT_G_DISPATCH(                                                                      \
            BITS, float, codebooks.const_data_ptr<float>(), POINTER_TYPE);                              \
      } else {                                                                                          \
        QVQ_V2_SEGMENT_LAUNCH(BITS, float, codebooks.const_data_ptr<float>(), POINTER_TYPE);           \
      }                                                                                                 \
    }                                                                                                  \
  } while (0)
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
    const c10::optional<at::Tensor>& step_weights) {
  return qvq_viterbi_v2_segment_banked_cuda_impl(
      sequences, codebooks, transition_bits, segment_steps, overlap, step_weights, 0);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qvq_viterbi_v2_segment_g_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights) {
  return qvq_viterbi_v2_segment_banked_cuda_impl(
      sequences, codebooks, transition_bits, segment_steps, overlap, step_weights, 1);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qvq_viterbi_v2_segment_grid_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights) {
  return qvq_viterbi_v2_segment_banked_cuda_impl(
      sequences, codebooks, transition_bits, segment_steps, overlap, step_weights, 2);
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
    const c10::optional<at::Tensor>& step_weights) {
  return qvq_viterbi_v2_segment_banked_cuda_impl(
      sequences, codebooks, transition_bits, segment_steps, overlap, step_weights, 2, false);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qvq_viterbi_v2_segment_tail_trusted_cuda(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& step_weights) {
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
      single_family_batch);
  const int64_t overlap_mask = (int64_t{1} << (16 - transition_bits)) - 1;
  auto overlap = std::get<0>(provisional).select(1, midpoint - 1).bitwise_and(overlap_mask).contiguous();
  return qvq_viterbi_v2_segment_banked_cuda_impl(
      sequences, codebooks, transition_bits, segment_steps, overlap, step_weights, 2, false,
      single_family_batch);
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
      2);
  return {
      std::get<0>(result).view({families, family_batch, 128}),
      std::get<1>(result).view({families, family_batch}),
      std::get<2>(result).view({families, family_batch, 8}),
  };
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
  m.def("viterbi_v2_segment_banked(Tensor sequences, Tensor codebooks, int transition_bits, int segment_steps, "
        "Tensor? overlap=None, Tensor? step_weights=None) -> (Tensor, Tensor, Tensor)");
  m.def("viterbi_v2_segment_g(Tensor sequences, Tensor codebooks, int transition_bits, int segment_steps, "
        "Tensor? overlap=None, Tensor? step_weights=None) -> (Tensor, Tensor, Tensor)");
  m.def("viterbi_v2_segment_grid(Tensor sequences, Tensor codebooks, int transition_bits, int segment_steps, "
        "Tensor? overlap=None, Tensor? step_weights=None) -> (Tensor, Tensor, Tensor)");
  m.def("viterbi_v2_segment_grid_trusted(Tensor sequences, Tensor codebooks, int transition_bits, int segment_steps, "
        "Tensor? overlap=None, Tensor? step_weights=None) -> (Tensor, Tensor, Tensor)");
  m.def("viterbi_v2_segment_tail_trusted(Tensor sequences, Tensor codebooks, int transition_bits, int segment_steps, "
        "Tensor? step_weights=None) -> (Tensor, Tensor, Tensor)");
  m.def("viterbi_v2_segment_midpoint_trusted(Tensor sequences, Tensor codebooks, int transition_bits, "
        "int segment_steps, Tensor? step_weights=None) -> Tensor");
  m.def("viterbi_v2_segment_family_grid_trusted(Tensor sequences, Tensor codebooks, int transition_bits, "
        "int segment_steps, Tensor? overlap=None, Tensor? step_weights=None) -> (Tensor, Tensor, Tensor)");
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
}
