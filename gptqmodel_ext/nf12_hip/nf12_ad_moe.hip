// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12 (NyoomFloat12) mixture-of-experts support for AMD RDNA4
// (gfx1200 / gfx1201 / gfx1250).
//
// Credits: NyoomFloat12 algorithm by Liberman, Fang, Zhang, Dao and Fu
// (Together AI, 2026), https://www.together.ai/. NF12 format, reference codec
// and CUDA kernels by AlpinDale, https://github.com/dphnAI/nf12 (Apache-2.0).
//
// This is the routing/activation half of SPEC.md section 9. The GEMM itself is
// not duplicated here: `moe_linear` runs the same matrix-core kernel as the
// dense route (csrc/hip/wmma.hip) with a row map enabled, so decode, matrix
// cores and the epilogue are shared. What lives here is the part that has no
// dense equivalent:
//
//   moe_align            stable counting sort of assignment ids by expert
//   moe_combine          sum topk rows in increasing j, one BF16 rounding
//   moe_silu_mul         SiLU(gate) * up in FP32, one BF16 rounding
//   moe_decode           per-expert exact BF16 reconstruction
//   moe_grouped_operands gather x into expert-sorted order + group ends
//
// Everything is host-synchronisation free so a launch can be graph-captured:
// per-expert counts are device-side and the host only supplies bounds.
// ---------------------------------------------------------------------------
#include "nf12_ad_format_hip.cuh"
#include "nf12_ad_validation.cuh"

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <limits>

namespace nf12_hip {
namespace {

// ---------------------------------------------------------------------------
// moe_align: stable counting sort (SPEC.md 9.2)
// ---------------------------------------------------------------------------
// Three passes over the assignment array, none of which reads routing data on
// the host:
//
//   count   - per (block, warp) histogram, so the scatter pass can order
//             segments without atomics on the output
//   scan    - exclusive prefix over experts, then over (block, warp) inside an
//             expert, turning the histogram into write cursors
//   scatter - walk each warp's segment in increasing id order and place ids
//
// A warp owns a contiguous, increasing id segment, so ordering the segments
// (expert, block, warp) is exactly a stable sort.
constexpr int kAlignThreads = 256;
constexpr int kAlignWarps = kAlignThreads / kWarpSize;
constexpr int64_t kAlignChunk = 8192;              // assignments per block
constexpr int64_t kAlignMaxCounters = int64_t{1} << 29;
constexpr size_t kAlignSmemBytes = 48u * 1024u;

// Ids per counting/placing warp, rounded up to a whole wave. Count and scatter
// must agree on this partition or the cursor order is wrong.
__host__ __device__ __forceinline__ int64_t align_segment(int64_t chunk) {
  return ((chunk + kAlignWarps - 1) / kAlignWarps + kWarpSize - 1) / kWarpSize * kWarpSize;
}

__device__ __forceinline__ int align_key(const int32_t* __restrict__ ids, int64_t a,
                                         int64_t end, int32_t num_experts) {
  if (a >= end) return -1;
  const int32_t v = ids[a];
  return (v >= 0 && v < num_experts) ? v : -1;
}

// count[e * blocks * kAlignWarps + b * kAlignWarps + w]
__global__ void __launch_bounds__(kAlignThreads)
    moe_align_count_kernel(const int32_t* __restrict__ ids, int64_t num_assignments,
                           int32_t num_experts, int64_t chunk, int64_t blocks,
                           int32_t* __restrict__ counts, int* __restrict__ global_hist) {
  extern __shared__ int smem_hist[];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  // Private histogram per (block, warp): no cross-block or cross-warp
  // contention, and a warp's ids are a contiguous run so its updates stay hot.
  int* const hist = (global_hist != nullptr)
                        ? global_hist + (int64_t(blockIdx.x) * kAlignWarps + warp) * num_experts
                        : smem_hist + warp * num_experts;
  for (int e = lane; e < num_experts; e += kWarpSize) hist[e] = 0;
  __syncthreads();

  const int64_t begin = int64_t(blockIdx.x) * chunk;
  const int64_t end = min(num_assignments, begin + chunk);
  // Warp `w` owns the CONTIGUOUS segment [a0, a1) of the chunk. The scatter
  // pass uses the same partition, and it is what makes the per-(expert, warp)
  // cursors an increasing id order. A strided assignment here would count a
  // different id set than the scatter places, which silently breaks stability.
  const int64_t seg = align_segment(chunk);
  const int64_t a0 = begin + int64_t(warp) * seg;
  const int64_t a1 = min(end, a0 + seg);
  for (int64_t a = a0 + lane; a < a1; a += kWarpSize) {
    const int key = align_key(ids, a, a1, num_experts);
    if (key >= 0) atomicAdd(&hist[key], 1);
  }
  __syncthreads();

  for (int e = lane; e < num_experts; e += kWarpSize) {
    counts[(int64_t(e) * blocks + blockIdx.x) * kAlignWarps + warp] = hist[e];
  }
}

// One block: exclusive scan over experts, then over (block, warp) inside each
// expert. `totals` is scratch of num_experts ints.
__global__ void __launch_bounds__(kAlignThreads)
    moe_align_scan_kernel(int32_t* __restrict__ counts, int32_t* __restrict__ totals,
                          int64_t blocks, int32_t num_experts,
                          int64_t* __restrict__ expert_offsets) {
  const int64_t row = blocks * kAlignWarps;
  for (int32_t e = threadIdx.x; e < num_experts; e += blockDim.x) {
    int64_t s = 0;
    for (int64_t i = 0; i < row; ++i) s += counts[int64_t(e) * row + i];
    totals[e] = static_cast<int32_t>(s);
  }
  __syncthreads();

  // Exclusive scan over experts. Serial on purpose: a block-wide scan over a
  // runtime-length array needs a second level of scratch for no measurable gain
  // at the expert counts MoE layers actually use.
  if (threadIdx.x == 0) {
    int64_t run = 0;
    for (int32_t e = 0; e < num_experts; ++e) {
      expert_offsets[e] = run;
      run += totals[e];
    }
    expert_offsets[num_experts] = run;
  }
  __syncthreads();

  // Turn the histogram into write cursors, ordered (block, warp) inside an
  // expert so the final placement is stable.
  for (int32_t e = threadIdx.x; e < num_experts; e += blockDim.x) {
    int32_t run = static_cast<int32_t>(expert_offsets[e]);
    for (int64_t i = 0; i < row; ++i) {
      const int32_t c = counts[int64_t(e) * row + i];
      counts[int64_t(e) * row + i] = run;
      run += c;
    }
  }
}

__global__ void __launch_bounds__(kAlignThreads)
    moe_align_scatter_kernel(const int32_t* __restrict__ ids, int64_t num_assignments,
                             int32_t num_experts, int64_t chunk, int64_t blocks,
                             int32_t* __restrict__ counts, int32_t* __restrict__ sorted,
                             int bits) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t begin = int64_t(blockIdx.x) * chunk;
  const int64_t end = min(num_assignments, begin + chunk);
  const int64_t seg = align_segment(chunk);
  const int64_t a0 = begin + int64_t(warp) * seg;
  const int64_t a1 = min(end, a0 + seg);
  const unsigned lower = (1u << lane) - 1u;

  for (int64_t base = a0; base < a1; base += kWarpSize) {
    const int64_t a = base + lane;
    const int key = align_key(ids, a, a1, num_experts);
    // Rank equal-expert lanes by lane id, so peers keep increasing id order.
    unsigned peers = __builtin_amdgcn_ballot_w32(key >= 0);
    for (int b = 0; b < bits; ++b) {
      const bool set = ((key >> b) & 1) != 0;
      const unsigned ballot = __builtin_amdgcn_ballot_w32(set);
      peers &= set ? ballot : ~ballot;
    }
    int32_t* const slot =
        (key >= 0) ? &counts[(int64_t(key) * blocks + blockIdx.x) * kAlignWarps + warp] : nullptr;
    const int32_t start = (key >= 0) ? *slot : 0;
    if (key >= 0) sorted[start + __popc(peers & lower)] = static_cast<int32_t>(a);
    __syncwarp();  // every peer has read the cursor before the leader advances it
    if (key >= 0 && lane == __ffs(peers) - 1) *slot = start + __popc(peers);
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// moe_combine / moe_silu_mul (SPEC.md 9.2)
// ---------------------------------------------------------------------------
constexpr int kElemThreads = 256;

struct CombineParams {
  const __hip_bfloat16* y;
  int64_t ldy;
  int64_t num_tokens;
  int32_t topk;
  int64_t vecs_per_row;  // n / 8
  __hip_bfloat16* out;
  int64_t ldo;
};

__device__ __forceinline__ void combine_accumulate(float (&acc)[8], const uint4& v) {
  const __hip_bfloat162* h = reinterpret_cast<const __hip_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    acc[2 * i] += f.x;
    acc[2 * i + 1] += f.y;
  }
}

__global__ void __launch_bounds__(kElemThreads) moe_combine_kernel(CombineParams p) {
  const int64_t idx = int64_t(blockIdx.x) * kElemThreads + threadIdx.x;
  const int64_t t = idx / p.vecs_per_row;
  const int64_t c = idx - t * p.vecs_per_row;
  if (t >= p.num_tokens) return;
  const __hip_bfloat16* src = p.y + t * p.topk * p.ldy + c * 8;
  // Starting from -0.0 preserves a -0.0 first term.
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = -0.0f;
  for (int64_t j = 0; j < p.topk; ++j) {
    combine_accumulate(acc, *reinterpret_cast<const uint4*>(src + j * p.ldy));
  }
  uint4 o;
  __hip_bfloat162* h = reinterpret_cast<__hip_bfloat162*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __float22bfloat162_rn(make_float2(acc[2 * i], acc[2 * i + 1]));
  *reinterpret_cast<uint4*>(p.out + t * p.ldo + c * 8) = o;
}

struct SiluMulParams {
  const __hip_bfloat16* h;
  int64_t ldh;
  int64_t rows;
  int64_t inter;
  int64_t vecs_per_row;  // inter / 8
  __hip_bfloat16* out;
  int64_t ldo;
};

__global__ void __launch_bounds__(kElemThreads) moe_silu_mul_kernel(SiluMulParams p) {
  const int64_t idx = int64_t(blockIdx.x) * kElemThreads + threadIdx.x;
  const int64_t r = idx / p.vecs_per_row;
  const int64_t c = idx - r * p.vecs_per_row;
  if (r >= p.rows) return;
  const __hip_bfloat16* src = p.h + r * p.ldh + c * 8;
  const uint4 gv = *reinterpret_cast<const uint4*>(src);
  const uint4 uv = *reinterpret_cast<const uint4*>(src + p.inter);
  const __hip_bfloat162* g2 = reinterpret_cast<const __hip_bfloat162*>(&gv);
  const __hip_bfloat162* u2 = reinterpret_cast<const __hip_bfloat162*>(&uv);
  uint4 o;
  __hip_bfloat162* o2 = reinterpret_cast<__hip_bfloat162*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 g = __bfloat1622float2(g2[i]);
    const float2 u = __bfloat1622float2(u2[i]);
    // No fast-math in this build, so / and + are correctly rounded and cannot
    // be contracted: the result matches the reference to the last BF16 bit.
    const float sx = g.x / (1.0f + expf(-g.x));
    const float sy = g.y / (1.0f + expf(-g.y));
    o2[i] = __float22bfloat162_rn(make_float2(sx * u.x, sy * u.y));
  }
  *reinterpret_cast<uint4*>(p.out + r * p.ldo + c * 8) = o;
}

// ---------------------------------------------------------------------------
// moe_decode (SPEC.md 9.4)
// ---------------------------------------------------------------------------
__global__ void moe_decode_kernel(const uint4* __restrict__ packed_in,
                                  const uint4* __restrict__ escapes,
                                  const int64_t* __restrict__ escape_offsets,
                                  const uint8_t* __restrict__ exp_base, int32_t num_experts,
                                  int64_t n, int64_t k, const int64_t* __restrict__ expert_offsets,
                                  int64_t max_assignments, int64_t row0, int64_t rows,
                                  __hip_bfloat16* __restrict__ out, int64_t ldo) {
  const int32_t e = blockIdx.y;
  if (e >= num_experts) return;
  if (expert_offsets != nullptr) {
    int64_t lo = 0;
    for (int32_t i = 0; i <= e; ++i) lo = moe_next_offset(expert_offsets[i], lo, max_assignments);
    const int64_t hi = moe_next_offset(expert_offsets[e + 1], lo, max_assignments);
    if (lo >= hi) return;  // inactive expert: leave its output untouched
  }
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int64_t k_blocks = k / kSuperAtomCols;
  // `packed` is expert-major, so step to this expert's slice first.
  const uint4* const packed =
      packed_in + int64_t(e) * (n / kSuperAtomRows) * k_blocks *
                      SourceTraits<WeightSource::kNf12>::kVecs;
  const int64_t row_base = row0 + int64_t(blockIdx.x) * kSuperAtomRows;
  const int64_t nb = row_base / kSuperAtomRows;
  const uint4* const esc = (escapes != nullptr) ? escapes + escape_offsets[e] : nullptr;
  const uint32_t base_x4 = exp_base_x4(exp_base[e]);
  // `row_tile_offset` re-adds row_base through the super-atom's row index, so
  // the base pointer is shifted by (e * rows - row0).
  __hip_bfloat16* const dst = out + (int64_t(e) * rows - row0) * ldo;

  for (int64_t kb = warp; kb < k_blocks; kb += kWarpsPerBlock) {
    const int64_t sa = nb * k_blocks + kb;
    uint4 planes[SourceTraits<WeightSource::kNf12>::kLoads];
    load_source_planes<WeightSource::kNf12>(packed, sa, lane, planes);
    AtomRegs r;
    source_planes_to_atoms<WeightSource::kNf12>(planes, base_x4, esc, r);
    RowTile x;
    atoms_to_tile(r, x);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int p = 0; p < 2; ++p) quad_transpose(x[h][p], lane);
    }
    store_row_tile(dst, row_tile_offset(super_atom(sa, k_blocks), ldo, lane), ldo, x);
  }
}

// ---------------------------------------------------------------------------
// moe_grouped_operands
// ---------------------------------------------------------------------------
// group_ends[e] is expert e's clamped, monotone end, so expert e owns
// [group_ends[e-1], group_ends[e]) with group_ends[-1] = 0. Every block derives
// it in shared memory; block 0 also publishes it.
__global__ void moe_grouped_operands_kernel(const __hip_bfloat16* __restrict__ x, int64_t ldx,
                                            const int32_t* __restrict__
                                                gather,
                                            int64_t gather_div,
                                            const int64_t* __restrict__ expert_offsets,
                                            int64_t max_assignments, int32_t num_experts,
                                            int64_t k, __hip_bfloat16* __restrict__ x_out,
                                            int64_t ldo, int32_t* __restrict__
                                                group_ends) {
  extern __shared__ int32_t s_ends[];
  if (threadIdx.x == 0) {
    int64_t prev = moe_next_offset(expert_offsets[0], 0, max_assignments);
    for (int32_t e = 0; e < num_experts; ++e) {
      const int64_t hi = moe_next_offset(expert_offsets[e + 1], prev, max_assignments);
      s_ends[e] = static_cast<int32_t>(hi);
      prev = hi;
    }
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int32_t e = 0; e < num_experts; ++e) group_ends[e] = s_ends[e];
  }
  // Without a gather there is nothing to copy: the CUDA build publishes the
  // ends and returns (csrc/moe/grouped.cu). x and x_out are optional in that
  // mode, so falling through would read and write caller buffers that the
  // contract does not require to exist.
  if (gather == nullptr) return;
  const int32_t total = s_ends[num_experts - 1];
  const int64_t vecs = k / 8;
  for (int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < int64_t(total) * vecs;
       idx += int64_t(gridDim.x) * blockDim.x) {
    const int64_t p = idx / vecs;
    const int64_t c = idx - p * vecs;
    const int64_t in = int64_t(gather[p]) / gather_div;
    reinterpret_cast<uint4*>(x_out + p * ldo)[c] = reinterpret_cast<const uint4*>(x + in * ldx)[c];
  }
}

}  // namespace
}  // namespace nf12_hip

using namespace nf12_hip;

extern "C" {

namespace {

// Chunk size and block count for a given assignment count. The chunk grows so
// the counter array stays inside int32.
struct AlignPlan {
  int64_t chunk;
  int64_t blocks;
  size_t bytes;
};

AlignPlan align_plan(int64_t num_assignments, int64_t num_experts) {
  AlignPlan pl{};
  pl.chunk = kAlignChunk;
  const int64_t min_chunk = (num_assignments * num_experts + kAlignMaxCounters - 1) / kAlignMaxCounters;
  const int64_t step = int64_t{kWarpSize} * kAlignWarps;
  const int64_t grown = (min_chunk + step - 1) / step * step;
  if (grown > pl.chunk) pl.chunk = grown;
  pl.blocks = (num_assignments + pl.chunk - 1) / pl.chunk;
  const int64_t counters = num_experts * pl.blocks * kAlignWarps + num_experts + 8;
  int64_t ints = counters;
  // A per-(block, warp) histogram does not fit in shared memory above ~1500
  // experts; those shapes take a copy in the workspace instead.
  if (static_cast<size_t>(kAlignWarps) * num_experts * sizeof(int) > kAlignSmemBytes) {
    ints += num_experts * pl.blocks * kAlignWarps;
  }
  pl.bytes = static_cast<size_t>(ints) * sizeof(int32_t);
  return pl;
}

}  // namespace

size_t nf12_ad_moe_align_workspace_bytes(int64_t num_assignments, int64_t num_experts) {
  if (num_assignments <= 0 || num_experts < 1) return 0;
  return align_plan(num_assignments, num_experts).bytes;
}

hipError_t nf12_ad_moe_align(const int32_t* topk_ids, int64_t num_tokens, int64_t topk,
                              int64_t num_experts, int32_t* sorted_out,
                              int64_t* expert_offsets_out, void* workspace,
                              size_t workspace_bytes, hipStream_t stream) {
  if (num_tokens < 0 || topk < 1 || num_experts < 1 || num_experts > 0x7fffffffLL) {
    return hipErrorInvalidValue;
  }
  if (num_tokens > 0x7fffffffLL / topk) return hipErrorInvalidValue;
  const int64_t num_assignments = num_tokens * topk;
  if (num_assignments == 0) {
    // No ids: still publish zero offsets.
    if (expert_offsets_out != nullptr) {
      const size_t bytes = static_cast<size_t>(num_experts + 1) * sizeof(int64_t);
      return hipMemsetAsync(expert_offsets_out, 0, bytes, stream);
    }
    return hipSuccess;
  }
  if (topk_ids == nullptr || sorted_out == nullptr || expert_offsets_out == nullptr) {
    return hipErrorInvalidValue;
  }

  const AlignPlan pl = align_plan(num_assignments, num_experts);
  const size_t needed = pl.bytes;
  if (workspace_bytes < needed || (needed > 0 && workspace == nullptr)) {
    return hipErrorInvalidValue;
  }

  const int64_t chunk = pl.chunk;
  const int64_t blocks = pl.blocks;

  int32_t* const counts = static_cast<int32_t*>(workspace);
  int32_t* const totals = counts + num_experts * blocks * kAlignWarps;

  // A private histogram per (block, warp) is kAlignWarps * E ints. Keep it in
  // shared memory while that fits; above that take the workspace copy the size
  // query already reserved.
  const size_t smem = static_cast<size_t>(kAlignWarps) * num_experts * sizeof(int);
  int* global_hist = nullptr;
  if (smem > kAlignSmemBytes) global_hist = totals + num_experts;

  moe_align_count_kernel<<<static_cast<unsigned>(blocks), kAlignThreads,
                           global_hist != nullptr ? 0u : static_cast<unsigned>(smem), stream>>>(
      topk_ids, num_assignments, static_cast<int32_t>(num_experts), chunk, blocks, counts,
      global_hist);
  // Read the launch status once: hipGetLastError clears the error, so calling
  // it again in the return statement would report success for a failed launch.
  if (hipError_t err = hipGetLastError(); err != hipSuccess) return err;
  moe_align_scan_kernel<<<1, kAlignThreads, 0, stream>>>(counts, totals, blocks,
                                                        static_cast<int32_t>(num_experts),
                                                        expert_offsets_out);
  if (hipError_t err = hipGetLastError(); err != hipSuccess) return err;
  const int bits = num_experts > 1 ? 64 - __builtin_clzll(static_cast<uint64_t>(num_experts - 1)) : 0;
  moe_align_scatter_kernel<<<static_cast<unsigned>(blocks), kAlignThreads, 0, stream>>>(
      topk_ids, num_assignments, static_cast<int32_t>(num_experts), chunk, blocks, counts,
      sorted_out, bits);
  return hipGetLastError();
}

hipError_t nf12_ad_moe_combine(const void* y_bf16, int64_t ldy, int64_t num_tokens, int64_t topk,
                                int64_t n, void* out_bf16, int64_t ldo, hipStream_t stream) {
  if (num_tokens < 0 || topk < 1 || topk > 0x7fffffffLL || n <= 0 || n % 8 != 0) {
    return hipErrorInvalidValue;
  }
  if (num_tokens == 0) return hipSuccess;
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (!nf12_ad::validation::valid_bf16_matrix(y_bf16, n, ldy) ||
      !nf12_ad::validation::valid_bf16_matrix(out_bf16, n, ldo) ||
      num_tokens > kMax / 2 / topk / ldy || num_tokens > kMax / 2 / ldo) {
    return hipErrorInvalidValue;
  }
  const int64_t vecs = num_tokens * (n / 8);
  const int64_t blocks = (vecs + kElemThreads - 1) / kElemThreads;
  if (blocks > 0x7fffffffLL) return hipErrorInvalidValue;
  CombineParams p{static_cast<const __hip_bfloat16*>(y_bf16), ldy, num_tokens,
                  static_cast<int32_t>(topk), n / 8, static_cast<__hip_bfloat16*>(out_bf16), ldo};
  moe_combine_kernel<<<static_cast<unsigned>(blocks), kElemThreads, 0, stream>>>(p);
  return hipGetLastError();
}

hipError_t nf12_ad_moe_silu_mul(const void* h_bf16, int64_t ldh, int64_t rows, int64_t inter,
                                 void* out_bf16, int64_t ldo, hipStream_t stream) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (rows < 0 || inter <= 0 || inter % 8 != 0 || inter > kMax / 2) return hipErrorInvalidValue;
  if (rows == 0) return hipSuccess;
  if (!nf12_ad::validation::valid_bf16_matrix(h_bf16, 2 * inter, ldh) ||
      !nf12_ad::validation::valid_bf16_matrix(out_bf16, inter, ldo) ||
      rows > kMax / 2 / ldh || rows > kMax / 2 / ldo) {
    return hipErrorInvalidValue;
  }
  const int64_t vecs = rows * (inter / 8);
  const int64_t blocks = (vecs + kElemThreads - 1) / kElemThreads;
  if (blocks > 0x7fffffffLL) return hipErrorInvalidValue;
  SiluMulParams p{static_cast<const __hip_bfloat16*>(h_bf16), ldh, rows, inter, inter / 8,
                  static_cast<__hip_bfloat16*>(out_bf16), ldo};
  moe_silu_mul_kernel<<<static_cast<unsigned>(blocks), kElemThreads, 0, stream>>>(p);
  return hipGetLastError();
}

hipError_t nf12_ad_moe_decode(const uint8_t* packed, const uint4* escapes,
                               const int64_t* escape_offsets, const uint8_t* exp_base,
                               int32_t num_experts, int64_t n, int64_t k,
                               const int64_t* expert_offsets,
                               int64_t max_assignments, int64_t row0, int64_t rows, void* out,
                               int64_t ldo, hipStream_t stream) {
  using namespace nf12_ad::validation;
  if (!valid_moe_shape(num_experts, n, k) || packed == nullptr || !aligned16(packed) ||
      !aligned16(escapes) || escape_offsets == nullptr || !aligned(escape_offsets, 8) ||
      exp_base == nullptr || !aligned(expert_offsets, 8) || max_assignments < 0 ||
      max_assignments > std::numeric_limits<int32_t>::max() || row0 < 0 || rows < 0 ||
      row0 % kSuperAtomRows != 0 || rows % kSuperAtomRows != 0 || rows > n - row0) {
    return hipErrorInvalidValue;
  }
  if (rows == 0) return hipSuccess;
  if (!valid_bf16_matrix(out, k, ldo) ||
      num_experts > std::numeric_limits<int64_t>::max() / 2 / rows / ldo) {
    return hipErrorInvalidValue;
  }
  const int64_t blocks = rows / kSuperAtomRows;
  if (blocks > std::numeric_limits<int32_t>::max()) return hipErrorInvalidValue;
  if (num_experts > 0xffff) return hipErrorInvalidValue;
  const dim3 grid(static_cast<unsigned>(blocks), static_cast<unsigned>(num_experts), 1);
  moe_decode_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const uint4*>(packed), escapes, escape_offsets, exp_base, num_experts, n, k,
      expert_offsets, max_assignments, row0, rows, reinterpret_cast<__hip_bfloat16*>(out), ldo);
  return hipGetLastError();
}

hipError_t nf12_ad_moe_grouped_operands(const void* x_bf16, int64_t ldx, const int32_t* gather,
                                         int64_t gather_div, const int64_t* expert_offsets,
                                         int64_t max_assignments, int64_t num_experts, int64_t k,
                                         void* x_out_bf16, int64_t ldo, int32_t* group_ends,
                                         hipStream_t stream) {
  using namespace nf12_ad::validation;
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (num_experts < 1 || num_experts > 0x7fffffffLL || k <= 0 || k % 8 != 0 || gather_div < 1 ||
      max_assignments < 0 || max_assignments > std::numeric_limits<int32_t>::max() ||
      expert_offsets == nullptr || !aligned(expert_offsets, 8) || group_ends == nullptr ||
      !aligned(group_ends, 4) || !aligned(gather, 4)) {
    return hipErrorInvalidValue;
  }
  if (gather != nullptr && (!valid_bf16_matrix(x_bf16, k, ldx) ||
      !valid_bf16_matrix(x_out_bf16, k, ldo) ||
      ldx > kMax / 2 / std::numeric_limits<int32_t>::max() ||
      ldo > kMax / 2 / std::numeric_limits<int32_t>::max())) return hipErrorInvalidValue;
  const size_t smem = static_cast<size_t>(num_experts) * sizeof(int32_t);
  if (smem > 48u * 1024u) return hipErrorNotSupported;
  // `total` is device-side, so size the grid from the host-known bound.
  int64_t blocks = gather == nullptr ? 1 :
      (max_assignments * (k / 8) + kElemThreads - 1) / kElemThreads;
  if (blocks < 1) blocks = 1;
  if (blocks > 4096) blocks = 4096;
  moe_grouped_operands_kernel<<<static_cast<unsigned>(blocks), kElemThreads,
                                static_cast<unsigned>(smem), stream>>>(
      static_cast<const __hip_bfloat16*>(x_bf16), ldx, gather, gather_div, expert_offsets,
      max_assignments, static_cast<int32_t>(num_experts), k,
      static_cast<__hip_bfloat16*>(x_out_bf16), ldo, group_ends);
  return hipGetLastError();
}

}  // extern "C"
