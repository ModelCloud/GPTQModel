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
at::Tensor cached_codebook_norm(const at::Tensor& codebook, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(g_norm_cache_mutex);
  const int device = codebook.get_device();
  const void* pointer = codebook.data_ptr();
  auto* codebook_impl = const_cast<c10::TensorImpl*>(codebook.unsafeGetTensorImpl());
  const uint32_t codebook_version = codebook_impl->is_inference()
      ? 0
      : codebook_impl->version_counter().current_version();
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
  entry.norm = at::empty({kStateCount}, codebook.options().dtype(at::kFloat));
  entry.device = device;
  entry.vector_size = VectorSize;
  entry.scalar_type = codebook.scalar_type();
  entry.bank_count = 1;
  entry.codebook_version = codebook_version;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&entry.ready, cudaEventDisableTiming));
  qvq_codebook_norm_kernel<VectorSize, CodebookScalar><<<256, 256, 0, stream>>>(
      reinterpret_cast<const CodebookScalar*>(codebook.const_data_ptr()), entry.norm.mutable_data_ptr<float>(), 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  const uint32_t codebook_version = codebook_impl->is_inference()
      ? 0
      : codebook_impl->version_counter().current_version();
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
  entry.norm = at::empty({bank_count, kStateCount}, codebook.options().dtype(at::kFloat));
  entry.device = device;
  entry.vector_size = VectorSize;
  entry.scalar_type = codebook.scalar_type();
  entry.bank_count = bank_count;
  entry.codebook_version = codebook_version;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&entry.ready, cudaEventDisableTiming));
  qvq_codebook_norm_kernel<VectorSize, CodebookScalar>
      <<<256 * bank_count, 256, 0, stream>>>(
          reinterpret_cast<const CodebookScalar*>(codebook.const_data_ptr()),
          entry.norm.mutable_data_ptr<float>(), bank_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  TORCH_CHECK(codebooks.dim() == 3 && codebooks.size(0) == 4,
              "banked V4 Viterbi requires exactly four codebooks");
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

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def("viterbi(Tensor sequences, Tensor codebook, int transition_bits, Tensor? overlap=None, "
        "Tensor? step_weights=None) -> (Tensor, Tensor)");
  m.def("viterbi_v4(Tensor sequences, Tensor codebook, int transition_bits, Tensor? overlap=None, "
        "Tensor? step_weights=None) -> (Tensor, Tensor)");
  m.def("viterbi_banked(Tensor sequences, Tensor codebooks, int transition_bits, Tensor? overlap=None, "
        "Tensor? step_weights=None) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CUDA, m) {
  m.impl("viterbi", &qvq_viterbi_cuda);
  m.impl("viterbi_v4", &qvq_viterbi_v4_cuda);
  m.impl("viterbi_banked", &qvq_viterbi_banked_cuda);
}
