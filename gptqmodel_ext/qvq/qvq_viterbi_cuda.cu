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
#include <math_constants.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>
#include <type_traits>
#include <limits>
#include <mutex>
#include <vector>

namespace {

constexpr int kStateCount = 1 << 16;
constexpr int kThreads = 1024;
constexpr size_t kMaxNormCacheEntries = 32;

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
  if constexpr (VectorSize == 4 && std::is_same_v<CodebookScalar, float>) {
    const float4 target4 = *reinterpret_cast<const float4*>(target);
    const float4 code4 = reinterpret_cast<const float4*>(codebook)[state];
    dot = __fmaf_rn(target4.x, code4.x, dot);
    dot = __fmaf_rn(target4.y, code4.y, dot);
    dot = __fmaf_rn(target4.z, code4.z, dot);
    dot = __fmaf_rn(target4.w, code4.w, dot);
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
  } else if constexpr (std::is_same_v<CodebookScalar, float>) {
    #pragma unroll
    for (int i = 0; i < VectorSize; ++i) {
      dot = __fmaf_rn(target[i], codebook[VectorSize * state + i], dot);
    }
  } else {
    const float2 code2 = __half22float2(reinterpret_cast<const half2*>(codebook)[state]);
    dot = __fmaf_rn(target[0], code2.x, dot);
    dot = __fmaf_rn(target[1], code2.y, dot);
  }
  const float distance = __fsub_rn(__fadd_rn(target_norm, codebook_norm[state]), __fmul_rn(2.0f, dot));
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
  codebook_norm += static_cast<int64_t>(bank) * kStateCount;
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
  if (bank_count == 1) {
    norm = at::empty({kStateCount}, codebook.options().dtype(at::kFloat));
  } else {
    norm = at::empty({bank_count, kStateCount}, codebook.options().dtype(at::kFloat));
  }
  qvq_codebook_norm_kernel<VectorSize, CodebookScalar>
      <<<256 * bank_count, 256, 0, stream>>>(
          reinterpret_cast<const CodebookScalar*>(codebook.const_data_ptr()),
          norm.mutable_data_ptr<float>(), bank_count);
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
  codebook_norm += static_cast<int64_t>(bank) * kStateCount;
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
      for (int k = 0; k < 64; ++k) {
        const int h = h0 + k * threads_per_suffix;
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
      __syncthreads();
    } else {
      const int x = thread & (suffix_count - 1);
      const int h0 = thread >> (16 - shift);
      float best = CUDART_INF_F;
      int best_h = h0;
      for (int k = 0; k < 64; ++k) {
        const int h = h0 + k * threads_per_suffix;
        const int s = (h << (16 - shift)) | x;
        const float value =
          G_prev[s >> shift] + emission<VectorSize, CodebookScalar>(
              target, codebook, codebook_norm, s, step_weight, target_norm);
        if (lower_pair(value, h, best, best_h)) {
          best = value;
          best_h = h;
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
      for (int k = 0; k < 64; ++k) {
        const int h = h0 + k * threads_per_suffix;
        const int s = (h << (16 - shift)) | x;
        float value;
        if (steps == 1) {
        value = emission<VectorSize, CodebookScalar>(
            target, codebook, codebook_norm, s, step_weight, target_norm);
        } else {
          value = G_prev[s >> shift] + emission<VectorSize, CodebookScalar>(
              target, codebook, codebook_norm, s, step_weight, target_norm);
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
      const float* bank_norm = codebook_norms + static_cast<int64_t>(bank) * kStateCount;
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
      const float* bank_norm = codebook_norms + static_cast<int64_t>(bank) * kStateCount;
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
      const float* bank_norm = codebook_norms + static_cast<int64_t>(bank) * kStateCount;
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
      const float* bank_norm = codebook_norms + static_cast<int64_t>(bank) * kStateCount;
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
    const float* bank_norm = codebook_norms + static_cast<int64_t>(bank) * kStateCount;
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
  const int64_t pointer_base = static_cast<int64_t>(sequence) * 127 * bank_suffix_count;
  const int64_t boundary_base = static_cast<int64_t>(sequence) * (segment_count - 1) * suffix_count;
  const CodebookScalar* bank_codebook =
      codebooks + static_cast<int64_t>(bank) * kStateCount * 2;
  const float* bank_norm = codebook_norms + static_cast<int64_t>(bank) * kStateCount;
  extern __shared__ float shared_frontiers[];
  float* g_previous = shared_frontiers;
  float* g_scratch = shared_frontiers + suffix_count;

  if (segment_index == 0) {
    const float* target = sequences + sequence_base;
    const float weight = weighted ? step_weights[static_cast<int64_t>(sequence) * 128] : 1.0f;
    const float target_norm = __fadd_rn(
        __fmul_rn(target[0], target[0]),
        __fmul_rn(target[1], target[1]));
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
      backpointers[pointer_base + bank * suffix_count + x] =
          static_cast<BackpointerScalar>(best_h);
    }
    __syncthreads();
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
      backpointers[pointer_base + static_cast<int64_t>(step) * bank_suffix_count +
                   bank * suffix_count + x] = static_cast<BackpointerScalar>(best_h);
    }
    __syncthreads();
    float* swap = g_previous;
    g_previous = g_scratch;
    g_scratch = swap;
  }
  for (int x = thread; x < suffix_count; x += kThreads) {
    g_output_all[g_base + x] = g_previous[x];
  }
}

template <
    int Shift,
    int BankCount,
    int SegmentSteps,
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
  const int64_t pointer_base = static_cast<int64_t>(sequence) * 127 * bank_suffix_count;
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
    const CodebookScalar* bank_codebook =
        codebooks + static_cast<int64_t>(bank) * kStateCount * 2;
    const float* bank_norm = codebook_norms + static_cast<int64_t>(bank) * kStateCount;
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
    squared_error[sequence] = best;
    int current_bank = best_flat / kStateCount;
    int current_state = best_flat - current_bank * kStateCount;
    states[state_base + 127] = current_state;
    segment_bank_ids[selector_base + segment_count - 1] = static_cast<uint8_t>(current_bank);
    for (int step = 127; step > 0; --step) {
      const int predecessor_suffix = current_state >> shift;
      if (step % segment_steps == 0) {
        const int boundary_index = step / segment_steps - 1;
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
}

template <int VectorSize>
std::tuple<at::Tensor, at::Tensor> qvq_viterbi_cuda_impl(
    const at::Tensor& sequences,
    const at::Tensor& codebook,
    int64_t transition_bits,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights,
    int64_t bank_count = 1) {
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
  TORCH_CHECK(at::isfinite(sequences).all().item<bool>() &&
                  at::isfinite(codebook).all().item<bool>(),
              "sequences and codebook must be finite");
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
    TORCH_CHECK(overlap_tensor.ge(0).all().item<bool>() &&
                    overlap_tensor.lt(overlap_limit).all().item<bool>(),
                "overlap values are outside the transition width");
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
    TORCH_CHECK(at::isfinite(step_weights_tensor).all().item<bool>() &&
                    step_weights_tensor.ge(0).all().item<bool>(),
                "step_weights must be finite and nonnegative");
  }
  // The emission uses the expanded squared-distance identity in FP32.  A
  // finite input can still make ||x||^2 or ||c||^2 overflow, after which
  // inf - inf becomes NaN and fmaxf can silently turn every state into a
  // false zero-cost tie.  The recurrence also accumulates every weighted
  // step, so include the worst-case number and weight of terms.  Reject that
  // domain at the native boundary instead of clamping and corrupting state
  // selection.
  const double maximum_weight = weighted ? step_weights_tensor.abs().amax().item<double>() : 1.0;
  const double accumulation_terms = std::max(1.0, static_cast<double>(steps) * maximum_weight);
  const double safe_magnitude = std::sqrt(static_cast<double>(std::numeric_limits<float>::max()) /
                                           accumulation_terms) /
      (2.0 * std::sqrt(static_cast<double>(VectorSize)));
  TORCH_CHECK(
      sequences.abs().amax().item<double>() <= safe_magnitude &&
          codebook.abs().amax().item<double>() <= safe_magnitude,
      "sequences and codebook magnitudes are too large for finite FP32 squared-distance arithmetic");

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

std::tuple<at::Tensor, at::Tensor, at::Tensor> qvq_viterbi_v2_segment_banked_cuda_impl(
    const at::Tensor& sequences,
    const at::Tensor& codebooks,
    int64_t transition_bits,
    int64_t segment_steps,
    const c10::optional<at::Tensor>& overlap,
    const c10::optional<at::Tensor>& step_weights,
    int kernel_mode) {
  const bool g_only = kernel_mode != 0;
  bool grid_parallel = kernel_mode == 2;
  TORCH_CHECK(sequences.is_cuda() && codebooks.is_cuda(),
              "segmented-bank V2 sequences and codebooks must be CUDA tensors");
  TORCH_CHECK(sequences.dim() == 3 && sequences.size(1) == 128 && sequences.size(2) == 2,
              "segmented-bank V2 sequences must have shape [batch, 128, 2]");
  TORCH_CHECK(codebooks.dim() == 3 && (codebooks.size(0) == 2 || codebooks.size(0) == 4) &&
                  codebooks.size(1) == kStateCount && codebooks.size(2) == 2,
              "segmented-bank V2 codebooks must have shape [2|4, 65536, 2]");
  const int batch = static_cast<int>(sequences.size(0));
  const int bank_count = static_cast<int>(codebooks.size(0));
  constexpr int steps = 128;
  TORCH_CHECK(batch > 0, "segmented-bank V2 sequences must contain at least one batch");
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
  TORCH_CHECK(at::isfinite(sequences).all().item<bool>() &&
                  at::isfinite(codebooks).all().item<bool>(),
              "segmented-bank V2 sequences and codebooks must be finite");

  const bool constrained = overlap.has_value();
  const at::Tensor overlap_tensor = constrained ? *overlap : at::Tensor();
  if (constrained) {
    TORCH_CHECK(overlap_tensor.is_cuda() && overlap_tensor.device() == sequences.device() &&
                    overlap_tensor.scalar_type() == at::kLong && overlap_tensor.is_contiguous(),
                "segmented-bank V2 overlap must be contiguous int64 on the sequence device");
    TORCH_CHECK(overlap_tensor.dim() == 1 && overlap_tensor.size(0) == batch,
                "segmented-bank V2 overlap must have shape [batch]");
    const int64_t overlap_limit = int64_t{1} << (16 - transition_bits);
    TORCH_CHECK(overlap_tensor.ge(0).all().item<bool>() &&
                    overlap_tensor.lt(overlap_limit).all().item<bool>(),
                "segmented-bank V2 overlap values are outside the retained-state range");
  }

  const bool weighted = step_weights.has_value();
  const at::Tensor step_weights_tensor = weighted ? *step_weights : at::Tensor();
  if (weighted) {
    TORCH_CHECK(step_weights_tensor.is_cuda() && step_weights_tensor.device() == sequences.device() &&
                    step_weights_tensor.scalar_type() == at::kFloat && step_weights_tensor.is_contiguous(),
                "segmented-bank V2 step weights must be contiguous float32 on the sequence device");
    TORCH_CHECK(step_weights_tensor.sizes() == at::IntArrayRef({batch, steps}),
                "segmented-bank V2 step weights must have shape [batch, 128]");
    TORCH_CHECK(at::isfinite(step_weights_tensor).all().item<bool>() &&
                    step_weights_tensor.ge(0).all().item<bool>(),
                "segmented-bank V2 step weights must be finite and nonnegative");
  }

  const double maximum_weight = weighted ? step_weights_tensor.abs().amax().item<double>() : 1.0;
  const double accumulation_terms = std::max(1.0, static_cast<double>(steps) * maximum_weight);
  const double safe_magnitude = std::sqrt(static_cast<double>(std::numeric_limits<float>::max()) /
                                           accumulation_terms) /
      (2.0 * std::sqrt(2.0));
  TORCH_CHECK(sequences.abs().amax().item<double>() <= safe_magnitude &&
                  codebooks.abs().amax().item<double>() <= safe_magnitude,
              "segmented-bank V2 magnitudes are too large for finite FP32 distance accumulation");

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
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(sequences.get_device());
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
      {batch, steps - 1, bank_count, suffix_count},
      sequences.options().dtype(!g_only && transition_bits == 7 ? at::kShort : at::kByte));
  at::Tensor states = at::empty({batch, steps}, sequences.options().dtype(at::kLong));
  at::Tensor segment_bank_ids = at::empty({batch, segments}, sequences.options().dtype(at::kByte));
  at::Tensor boundary_banks = g_only
      ? at::empty({batch, segments - 1, suffix_count}, sequences.options().dtype(at::kByte))
      : at::Tensor();
  at::Tensor squared_error = at::empty({batch}, float_options);
  at::Tensor codebook_norm = codebooks.scalar_type() == at::kHalf
      ? cached_banked_codebook_norm<2, half>(codebooks, stream)
      : cached_banked_codebook_norm<2, float>(codebooks, stream);
  const int64_t* overlap_ptr = constrained ? overlap_tensor.const_data_ptr<int64_t>() : nullptr;
  const float* weight_ptr = weighted ? step_weights_tensor.const_data_ptr<float>() : nullptr;

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
    BITS, BANKS, SEGMENT_STEPS, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE)                       \
  do {                                                                                                \
    constexpr bool qvq_fuse_boundary =                                                               \
        (BANKS == 2 && BITS != 7) || (BANKS == 4 && BITS >= 3 && BITS <= 6);                         \
    constexpr int qvq_segment_shared_bytes = 2 * (1 << (16 - BITS)) * sizeof(float);                  \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                                              \
        qvq_v2_segment_grid_kernel<                                                                   \
            BITS, BANKS, SEGMENT_STEPS, qvq_fuse_boundary, CODEBOOK_TYPE, POINTER_TYPE>,             \
        cudaFuncAttributeMaxDynamicSharedMemorySize, qvq_segment_shared_bytes));                      \
    float* qvq_frontier_a = costs_a.mutable_data_ptr<float>();                                        \
    float* qvq_frontier_b = costs_b.mutable_data_ptr<float>();                                        \
    qvq_v2_segment_grid_kernel<                                                                       \
        BITS, BANKS, SEGMENT_STEPS, qvq_fuse_boundary, CODEBOOK_TYPE, POINTER_TYPE><<<                \
        batch * BANKS, kThreads, qvq_segment_shared_bytes, stream>>>(                                 \
        sequences.const_data_ptr<float>(), CODEBOOK_POINTER, codebook_norm.const_data_ptr<float>(),   \
        overlap_ptr, weight_ptr, qvq_frontier_b, qvq_frontier_a,                                     \
        backpointers.mutable_data_ptr<POINTER_TYPE>(), boundary_banks.mutable_data_ptr<uint8_t>(),     \
        batch, 0, constrained, weighted);                                                             \
    for (int segment = 1; segment < segments; ++segment) {                                            \
      const float* qvq_segment_input = segment % 2 == 1 ? qvq_frontier_a : qvq_frontier_b;            \
      float* qvq_segment_output = segment % 2 == 1 ? qvq_frontier_b : qvq_frontier_a;                 \
      if constexpr (!qvq_fuse_boundary) {                                                             \
        qvq_v2_segment_boundary_kernel<BITS, BANKS, SEGMENT_STEPS><<<batch, kThreads, 0, stream>>>(   \
            qvq_segment_input, boundary_banks.mutable_data_ptr<uint8_t>(),                            \
            batch, segment - 1);                                                                      \
      }                                                                                               \
      qvq_v2_segment_grid_kernel<                                                                     \
          BITS, BANKS, SEGMENT_STEPS, qvq_fuse_boundary, CODEBOOK_TYPE, POINTER_TYPE><<<              \
          batch * BANKS, kThreads, qvq_segment_shared_bytes, stream>>>(                               \
          sequences.const_data_ptr<float>(), CODEBOOK_POINTER, codebook_norm.const_data_ptr<float>(), \
          overlap_ptr, weight_ptr, qvq_segment_input, qvq_segment_output,                             \
          backpointers.mutable_data_ptr<POINTER_TYPE>(), boundary_banks.mutable_data_ptr<uint8_t>(),   \
          batch, segment, constrained, weighted);                                                     \
    }                                                                                                 \
    const float* qvq_final_frontier = segments % 2 == 1 ? qvq_frontier_a : qvq_frontier_b;           \
    qvq_v2_segment_grid_finalize_kernel<                                                              \
        BITS, BANKS, SEGMENT_STEPS, CODEBOOK_TYPE, POINTER_TYPE><<<batch, kThreads, 0, stream>>>(     \
        sequences.const_data_ptr<float>(), CODEBOOK_POINTER, codebook_norm.const_data_ptr<float>(),   \
        overlap_ptr, weight_ptr, qvq_final_frontier,                                                  \
        backpointers.const_data_ptr<POINTER_TYPE>(), boundary_banks.const_data_ptr<uint8_t>(),         \
        states.mutable_data_ptr<int64_t>(), segment_bank_ids.mutable_data_ptr<uint8_t>(),              \
        squared_error.mutable_data_ptr<float>(), batch, constrained, weighted);                        \
  } while (0)
#define QVQ_V2_SEGMENT_GRID_DISPATCH(BITS, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE)             \
  do {                                                                                                \
    if (bank_count == 2) {                                                                            \
      QVQ_V2_SEGMENT_GRID_LAUNCH(                                                                     \
          BITS, 2, 16, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE);                                \
    } else {                                                                                          \
      QVQ_V2_SEGMENT_GRID_LAUNCH(                                                                     \
          BITS, 4, 32, CODEBOOK_TYPE, CODEBOOK_POINTER, POINTER_TYPE);                                \
    }                                                                                                 \
  } while (0)
#define QVQ_V2_SEGMENT_DISPATCH(BITS, POINTER_TYPE)                                                    \
  do {                                                                                                 \
    if (codebooks.scalar_type() == at::kHalf) {                                                        \
      if (grid_parallel) {                                                                              \
        QVQ_V2_SEGMENT_GRID_DISPATCH(                                                                   \
            BITS, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()), POINTER_TYPE);      \
      } else if (g_only) {                                                                              \
        QVQ_V2_SEGMENT_G_DISPATCH(                                                                      \
            BITS, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()), POINTER_TYPE);      \
      } else {                                                                                          \
        QVQ_V2_SEGMENT_LAUNCH(                                                                          \
            BITS, half, reinterpret_cast<const half*>(codebooks.const_data_ptr()), POINTER_TYPE);      \
      }                                                                                                 \
    } else {                                                                                           \
      if (grid_parallel) {                                                                              \
        QVQ_V2_SEGMENT_GRID_DISPATCH(                                                                   \
            BITS, float, codebooks.const_data_ptr<float>(), POINTER_TYPE);                             \
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

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def("viterbi(Tensor sequences, Tensor codebook, int transition_bits, Tensor? overlap=None, "
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
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CUDA, m) {
  m.impl("viterbi", &qvq_viterbi_cuda);
  m.impl("viterbi_v4", &qvq_viterbi_v4_cuda);
  m.impl("viterbi_banked", &qvq_viterbi_banked_cuda);
  m.impl("viterbi_v2_segment_banked", &qvq_viterbi_v2_segment_banked_cuda);
  m.impl("viterbi_v2_segment_g", &qvq_viterbi_v2_segment_g_cuda);
  m.impl("viterbi_v2_segment_grid", &qvq_viterbi_v2_segment_grid_cuda);
}
