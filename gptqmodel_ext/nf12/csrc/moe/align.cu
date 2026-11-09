// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12 / NF12-AD kernel computation.
//
// Algorithm: "NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit
// Weight Compression", Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).
// Together AI: https://www.together.ai/
//
// Original implementation: https://github.com/dphnAI/nf12
// Copyright (c) 2026 AlpinDale and the nf12 contributors.
// Licensed under the Apache License, Version 2.0.
//
// Vendored into GPT-QModel (https://github.com/ModelCloud/GPTQModel) and
// modified for the NF12-AD (AlpinDale) method/format. See
// gptqmodel_ext/nf12/README.md for the list of local modifications.
// ---------------------------------------------------------------------------
// Stable counting sort drops ids outside [0, E) (SPEC.md 9.2).
// Each warp owns a contiguous, increasing assignment-id segment. Ballots rank
// equal-expert lanes by lane id, and expert-major (expert, warp) counter scans
// preserve segment order without relying on atomic ordering.
//
// The odd (expert, warp) row stride separates distinct experts' counter banks.
// Counters exceeding kSmemBytes use workspace in one block.
// SM90+ follows linear/dispatch.cu's PDL contract with no global reads before
// griddepcontrol.wait.
#include <algorithm>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {
namespace {

constexpr int kSmemBytes = 48 * 1024;
constexpr int kMaxThreads = 1024;
constexpr int kTiles = 8;            // 32-id tiles a warp holds in registers
constexpr int kRound = kTiles * 32;  // ids per warp per round
constexpr int64_t kFusedMaxAssignments = 2048;
constexpr int64_t kChunk = 2048;
// Bound on the counters a scan covers, so 4 * len4 and every running sum fit in 32 bits.
constexpr int64_t kMaxCounters = int64_t{1} << 30;

__device__ __forceinline__ void pdl_entry() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Expert of id `a`, or -1 if it is dropped or past `end`.
__device__ __forceinline__ int expert_key(const int32_t* __restrict__ ids, int64_t a, int64_t end, int32_t num_experts) {
  if (a >= end) return -1;
  const int32_t e = __ldg(ids + a);
  return e >= 0 && e < num_experts ? e : -1;
}

__device__ __forceinline__ void load_keys(const int32_t* __restrict__ ids, int64_t base, int64_t end,
                                          int32_t num_experts, int (&keys)[kTiles]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int t = 0; t < kTiles; ++t) keys[t] = expert_key(ids, base + t * 32 + lane, end, num_experts);
}

// All lanes must participate, including dropped keys (key < 0).
__device__ __forceinline__ unsigned peer_mask(int key, int bits) {
  unsigned peers = __ballot_sync(0xffffffffu, key >= 0);
  for (int b = 0; b < bits; ++b) {
    const bool set = (key >> b) & 1;
    const unsigned ballot = __ballot_sync(0xffffffffu, set);
    peers &= set ? ballot : ~ballot;
  }
  return peers;
}

__device__ __forceinline__ void count_keys(const int (&keys)[kTiles], int64_t base, int64_t end, int bits,
                                           int* counts, int stride, int col) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
    if (base + t * 32 >= end) break;  // warp-uniform
    const unsigned peers = peer_mask(keys[t], bits);
    if (keys[t] >= 0 && lane == __ffs(peers) - 1) atomicAdd(&counts[keys[t] * stride + col], __popc(peers));
  }
}

// Advance each expert's cursor only after all peers have read it.
__device__ __forceinline__ void scatter_keys(const int (&keys)[kTiles], int64_t base, int64_t end, int bits,
                                             int* cursor, int stride, int col, int32_t* __restrict__ sorted) {
  const int lane = threadIdx.x & 31;
  const unsigned lower = (1u << lane) - 1u;
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
    if (base + t * 32 >= end) break;
    const int key = keys[t];
    const unsigned peers = peer_mask(key, bits);
    int* slot = &cursor[max(key, 0) * stride + col];
    const int start = key >= 0 ? *slot : 0;
    if (key >= 0) sorted[start + __popc(peers & lower)] = static_cast<int32_t>(base + t * 32 + lane);
    __syncwarp();  // every peer has read the cursor before the leader advances it
    if (key >= 0 && lane == __ffs(peers) - 1) *slot = start + __popc(peers);
    __syncwarp();
  }
}

struct AlignParams {
  const int32_t* ids;
  int64_t num_assignments;
  int32_t num_experts;
  int bits;  // bits of the largest expert id, ceil(log2(E))
  int32_t* sorted;
  int64_t* expert_offsets;
  int* global_counts;  // fused kernel: counters in the workspace instead of shared memory
  int* hist;           // multi-block path: [E, blocks] expert-major, padded to a multiple of 4
  int warps;           // counting warps per block, each owning `seg` ids
  int stride;          // counter row stride (warps | 1)
  int64_t seg;         // ids per counting warp, a multiple of 32
  int len4;            // counters (fused) or hist (multi-block) in int4 vectors
  int per4;            // int4 vectors per thread in scan_counters
  int64_t chunk;       // multi-block: ids per block
  int64_t count_seg;   // multi-block count kernel: ids per warp (every warp counts)
};

// Counts a warp's segment [begin, end). On return `keys` holds the last round,
// which is the whole segment when end - begin <= kRound.
__device__ __forceinline__ void warp_count(const AlignParams& p, int64_t begin, int64_t end, int* counts, int stride,
                                           int col, int (&keys)[kTiles]) {
  for (int64_t base = begin; base < end; base += kRound) {
    load_keys(p.ids, base, end, p.num_experts, keys);
    count_keys(keys, base, end, p.bits, counts, stride, col);
  }
}

__device__ __forceinline__ void warp_scatter(const AlignParams& p, int64_t begin, int64_t end, int* cursor, int stride,
                                             int col, int (&keys)[kTiles]) {
  const bool cached = end - begin <= kRound;
  for (int64_t base = begin; base < end; base += kRound) {
    if (!cached) load_keys(p.ids, base, end, p.num_experts, keys);
    scatter_keys(keys, base, end, p.bits, cursor, stride, col, p.sorted);
  }
}

// data requires 16-byte alignment for 4 * len4 counters. scratch needs 32 ints.
// Thread t owns [t * per4, (t + 1) * per4) int4 vectors, avoiding bank conflicts.
// Writes expert starts and the total to offsets, ending with a block barrier.
__device__ void scan_counters(int* data, int len4, int per4, unsigned stride, int32_t num_experts, int64_t* offsets,
                              int* scratch) {
  int4* vec = reinterpret_cast<int4*>(data);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int first = threadIdx.x * per4;
  const int last = min(len4, first + per4);
  int sum = 0;
  for (int i = first; i < last; ++i) {
    const int4 q = vec[i];
    sum += q.x + q.y + q.z + q.w;
  }
  int incl = sum;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int u = __shfl_up_sync(0xffffffffu, incl, d);
    if (lane >= d) incl += u;
  }
  if (lane == 31) scratch[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    int w = lane < static_cast<int>(blockDim.x >> 5) ? scratch[lane] : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, w, d);
      if (lane >= d) w += u;
    }
    scratch[lane] = w;  // inclusive warp prefix, with the total in scratch[31]
  }
  __syncthreads();
  if (threadIdx.x == 0) offsets[num_experts] = scratch[31];
  int run = (warp > 0 ? scratch[warp - 1] : 0) + incl - sum;
  // At most 2^31 + 3 counters (E <= 2^31 - 1, or kMaxCounters), so this fits.
  unsigned e = static_cast<unsigned>(first) * 4u / stride;
  unsigned c = static_cast<unsigned>(first) * 4u - e * stride;
  for (int i = first; i < last; ++i) {
    const int4 q = vec[i];
    const int in[4] = {q.x, q.y, q.z, q.w};
    int out[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[j] = run;
      run += in[j];
      if (c == 0 && e < static_cast<unsigned>(num_experts)) offsets[e] = out[j];
      if (++c == stride) {
        c = 0;
        ++e;
      }
    }
    vec[i] = make_int4(out[0], out[1], out[2], out[3]);
  }
  __syncthreads();
}

__device__ __forceinline__ void zero4(int* data, int len4) {
  int4* vec = reinterpret_cast<int4*>(data);
  for (int i = threadIdx.x; i < len4; i += blockDim.x) vec[i] = make_int4(0, 0, 0, 0);
}

__device__ __forceinline__ void warp_segment(int64_t chunk_begin, int64_t chunk_end, int64_t seg, int64_t* begin,
                                             int64_t* end) {
  *begin = min(chunk_end, chunk_begin + static_cast<int64_t>(threadIdx.x >> 5) * seg);
  *end = min(chunk_end, *begin + seg);
}

// Separate instantiations let the shared-memory path use shared atomics.
template <bool kGlobal>
__global__ void __launch_bounds__(kMaxThreads) align_fused_kernel(AlignParams p) {
  extern __shared__ int4 smem_counts[];
  __shared__ int scratch[32];
  int* counts = kGlobal ? p.global_counts : reinterpret_cast<int*>(smem_counts);
  const int w = threadIdx.x >> 5;
  pdl_entry();
  zero4(counts, p.len4);
  __syncthreads();
  int64_t begin = 0, end = 0;
  warp_segment(0, p.num_assignments, p.seg, &begin, &end);
  int keys[kTiles];
  if (w < p.warps) warp_count(p, begin, end, counts, p.stride, w, keys);
  __syncthreads();
  scan_counters(counts, p.len4, p.per4, p.stride, p.num_experts, p.expert_offsets, scratch);
  if (w < p.warps) warp_scatter(p, begin, end, counts, p.stride, w, keys);
}

__global__ void __launch_bounds__(kMaxThreads) align_count_kernel(AlignParams p) {
  extern __shared__ int4 smem_counts[];
  int* counts = reinterpret_cast<int*>(smem_counts);
  const int64_t chunk_begin = blockIdx.x * p.chunk;
  const int64_t chunk_end = min(p.num_assignments, chunk_begin + p.chunk);
  pdl_entry();
  zero4(counts, (p.num_experts + 3) / 4);
  __syncthreads();
  int64_t begin = 0, end = 0;
  warp_segment(chunk_begin, chunk_end, p.count_seg, &begin, &end);
  int keys[kTiles];
  warp_count(p, begin, end, counts, 1, 0, keys);
  __syncthreads();
  for (int e = threadIdx.x; e < p.num_experts; e += blockDim.x) {
    p.hist[static_cast<int64_t>(e) * gridDim.x + blockIdx.x] = counts[e];
  }
  // Zero padding because the scan reads full int4 vectors.
  const int64_t used = static_cast<int64_t>(p.num_experts) * gridDim.x;
  if (blockIdx.x == 0 && threadIdx.x < 4 * static_cast<int64_t>(p.len4) - used) p.hist[used + threadIdx.x] = 0;
}

__global__ void __launch_bounds__(kMaxThreads) align_scan_kernel(AlignParams p, int blocks) {
  __shared__ int scratch[32];
  pdl_entry();
  scan_counters(p.hist, p.len4, p.per4, blocks, p.num_experts, p.expert_offsets, scratch);
}

__global__ void __launch_bounds__(kMaxThreads) align_scatter_kernel(AlignParams p) {
  extern __shared__ int4 smem_counts[];
  int* counts = reinterpret_cast<int*>(smem_counts);
  const int w = threadIdx.x >> 5;
  const int64_t chunk_begin = blockIdx.x * p.chunk;
  const int64_t chunk_end = min(p.num_assignments, chunk_begin + p.chunk);
  pdl_entry();
  zero4(counts, p.len4);
  __syncthreads();
  int64_t begin = 0, end = 0;
  warp_segment(chunk_begin, chunk_end, p.seg, &begin, &end);
  int keys[kTiles];
  if (w < p.warps) warp_count(p, begin, end, counts, p.stride, w, keys);
  __syncthreads();
  for (int e = threadIdx.x; e < p.num_experts; e += blockDim.x) {
    int run = p.hist[static_cast<int64_t>(e) * gridDim.x + blockIdx.x];
    for (int i = 0; i < p.warps; ++i) {
      const int v = counts[e * p.stride + i];
      counts[e * p.stride + i] = run;
      run += v;
    }
  }
  __syncthreads();
  if (w < p.warps) warp_scatter(p, begin, end, counts, p.stride, w, keys);
}

// Zero means even one warp's [E, warps | 1] counters exceed shared memory.
int smem_warps(int64_t num_experts) {
  int warps = kMaxThreads / 32;
  while (warps > 0 && (num_experts * (warps | 1) + 3) / 4 * 16 > kSmemBytes) warps >>= 1;
  return warps;
}

int64_t round_up(int64_t v, int64_t m) { return (v + m - 1) / m * m; }

struct AlignPlan {
  bool fused;          // single block, otherwise count/scan/scatter over blocks
  int64_t blocks;
  int warps;           // counting warps per block
  int threads;         // fused and scatter kernels
  int64_t seg;         // ids per counting warp
  bool global_counts;  // fused kernel with counters in the workspace
  int64_t chunk;       // ids per block (multi-block path)
  int64_t counters;    // E * stride (fused) or E * blocks, before int4 padding
  size_t workspace_bytes;
};

AlignPlan plan(int64_t num_assignments, int64_t num_experts) {
  AlignPlan pl{};
  pl.fused = true;
  pl.blocks = 1;
  const int max_warps = smem_warps(num_experts);
  if (max_warps == 0) {
    // One counting warp bounds workspace to E ints.
    pl.warps = 1;
    pl.seg = round_up(std::max<int64_t>(num_assignments, 1), 32);
    pl.global_counts = true;
    pl.counters = num_experts;
    pl.threads = kMaxThreads;
    pl.workspace_bytes = static_cast<size_t>(round_up(num_experts, 4)) * sizeof(int);
    return pl;
  }
  if (num_assignments <= std::min<int64_t>(kFusedMaxAssignments, int64_t{max_warps} * kRound)) {
    // More warps shorten the serial tile chain but enlarge the counter scan.
    pl.warps = static_cast<int>(std::min<int64_t>(max_warps, std::max<int64_t>(1, (num_assignments + 31) / 32)));
    pl.seg = round_up(std::max<int64_t>(1, (num_assignments + pl.warps - 1) / pl.warps), 32);
    pl.counters = num_experts * (pl.warps | 1);
    pl.threads = static_cast<int>(
        std::clamp<int64_t>(round_up(std::max<int64_t>(pl.warps * 32, (pl.counters + 3) / 4), 32), 32, kMaxThreads));
    return pl;
  }
  pl.fused = false;
  pl.chunk = std::min<int64_t>(kChunk, int64_t{max_warps} * kRound);
  pl.warps = static_cast<int>(std::min<int64_t>(max_warps, pl.chunk / 32));
  // Grow chunks to keep the [E, blocks] histogram within kMaxCounters.
  const int64_t min_chunk = (num_assignments * num_experts + kMaxCounters - 1) / kMaxCounters;
  pl.chunk = round_up(std::max(pl.chunk, min_chunk), int64_t{32} * pl.warps);
  pl.seg = pl.chunk / pl.warps;
  pl.threads = std::max(pl.warps * 32, 256);
  pl.blocks = (num_assignments + pl.chunk - 1) / pl.chunk;
  pl.counters = num_experts * pl.blocks;
  pl.workspace_bytes = static_cast<size_t>(round_up(pl.counters, 4)) * sizeof(int);
  return pl;
}

template <typename... Args>
cudaError_t launch(void (*kernel)(Args...), int64_t blocks, int threads, size_t smem, cudaStream_t stream, bool pdl,
                   Args... args) {
  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(static_cast<unsigned>(blocks), 1, 1);
  lc.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
  lc.dynamicSmemBytes = smem;
  lc.stream = stream;
  cudaLaunchAttribute attr[1];
  if (pdl) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = attr;
    lc.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&lc, kernel, args...);
}

}  // namespace

size_t moe_align_workspace_bytes(int64_t num_tokens, int64_t topk, int64_t num_experts) {
  return plan(num_tokens * topk, num_experts).workspace_bytes;
}

cudaError_t moe_align(const int32_t* topk_ids, int64_t num_tokens, int64_t topk, int64_t num_experts, int32_t* sorted,
                      int64_t* expert_offsets, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  if (num_tokens < 0 || topk < 1 || num_experts < 1 || num_experts > INT32_MAX) return cudaErrorInvalidValue;
  if (num_tokens > INT32_MAX / topk) return cudaErrorInvalidValue;
  const int64_t num_assignments = num_tokens * topk;
  const AlignPlan pl = plan(num_assignments, num_experts);
  if (workspace_bytes < pl.workspace_bytes || (pl.workspace_bytes > 0 && workspace == nullptr)) {
    return cudaErrorInvalidValue;
  }
  int device = 0;
  if (cudaError_t err = cudaGetDevice(&device); err != cudaSuccess) return err;
  const bool pdl = device_info(device).sm_arch >= 90;

  AlignParams p{};
  p.ids = topk_ids;
  p.num_assignments = num_assignments;
  p.num_experts = static_cast<int32_t>(num_experts);
  p.bits = num_experts > 1 ? 64 - __builtin_clzll(static_cast<uint64_t>(num_experts - 1)) : 0;
  p.sorted = sorted;
  p.expert_offsets = expert_offsets;
  p.warps = pl.warps;
  p.stride = pl.warps | 1;
  p.seg = pl.seg;
  p.chunk = pl.chunk;
  const int64_t len4 = (pl.counters + 3) / 4;
  p.len4 = static_cast<int>(len4);
  if (pl.fused) {
    p.per4 = static_cast<int>((len4 + pl.threads - 1) / pl.threads);
    p.global_counts = pl.global_counts ? static_cast<int*>(workspace) : nullptr;
    return pl.global_counts ? launch(align_fused_kernel<true>, 1, pl.threads, 0, stream, pdl, p)
                            : launch(align_fused_kernel<false>, 1, pl.threads, static_cast<size_t>(len4) * 16, stream,
                                     pdl, p);
  }
  p.hist = static_cast<int*>(workspace);
  p.per4 = static_cast<int>((len4 + kMaxThreads - 1) / kMaxThreads);
  p.count_seg = round_up(pl.chunk / (kMaxThreads / 32), 32);
  const size_t hist_smem = static_cast<size_t>(round_up(num_experts, 4)) * sizeof(int);
  const size_t counts_smem = static_cast<size_t>(round_up(num_experts * p.stride, 4)) * sizeof(int);
  AlignParams scatter = p;
  scatter.len4 = static_cast<int>(counts_smem / 16);
  if (cudaError_t err = launch(align_count_kernel, pl.blocks, kMaxThreads, hist_smem, stream, pdl, p);
      err != cudaSuccess) {
    return err;
  }
  if (cudaError_t err = launch(align_scan_kernel, 1, kMaxThreads, 0, stream, pdl, p, static_cast<int>(pl.blocks));
      err != cudaSuccess) {
    return err;
  }
  return launch(align_scatter_kernel, pl.blocks, pl.threads, counts_smem, stream, pdl, scatter);
}

}  // namespace nf12::detail
