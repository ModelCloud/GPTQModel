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
// Large-M mma.sync m16n8k16 kernels (SM80+), shared by
// linear/sm80/large_m_mma.cu and moe/sm80/large_mma.cu.
// Weights are the A operand ("swap-AB"), activations the B operand (8 tokens
// per MMA, ldmatrix from an XOR-swizzled tile). A CTA computes BN weight rows
// x BM tokens. The epilogue transposes through shared memory to y[token, n].
//
// Four mainloops share tiles and epilogue (all except kRing also split-K):
//  * Loop::kStream: every warp decodes its own A fragments from the
//    compressed planes in shared memory straight into registers, minimizing
//    shared-memory use and per-CTA work for bandwidth-bound shapes.
//  * Loop::kStaged: compressed tiles load into registers and decode once per
//    CTA into a bf16t-layout buffer a stage before use. This removes decode
//    from the MMA dependency chains and gives escape loads latency slack.
//  * Loop::kRing: three staged-loop slots provide two stages of fetch slack.
//    Split-K is unsupported.
//  * Loop::kBudget: the two-slot staged loop with one A fragment buffer
//    instead of two, retaining B-fragment lookahead. This reduces register
//    pressure for 64x64 warp tiles on SM8x.
// The bf16t source uses an identity decode. Both sources use the same
// per-accumulator MMA sequence, epilogue and split-K order.
//
// Split-K uses a thread-block cluster and reduces FP32 partials through
// distributed shared memory in fixed rank order, without workspace.
// Below SM90, problems exposing partials() use workspace and a second
// large_m_split_reduce_kernel launch with the same reduction order.
//
// The Problem policy maps CTAs to tiles and supplies operands:
//   using Params;  using Work  (int n0, m0, kb_begin, nkb plus its own fields)
//   Prob(const Params&, uint8_t* extra_smem)     extra_smem: past the pipeline
//   template <class C> bool next(Work&)          false once the CTA is done (tile_raster() orders tiles)
//   weights(w), escapes(w), num_escapes(w), base_x4(w), n(), kblocks()
//   template <class C> act(w)                    activation loader: load(dst, kb_rel)
//   template <class C> row_terms(w, n_local)     epilogue terms of a weight row pair (g, g + 8)
//   template <class C> fragment(w, terms, tok, v[4])   v = acc of tokens (tok, tok + 1), rows (g, g + 8)
//   out_row(w, row)                              y row of token row `row` + n0, or nullptr
//   template <class C> partials(w)               optional (split-K below SM90, see HasPartials)
// The kernel loops over next(): a problem may hand one CTA several tiles.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "linear/common.cuh"

namespace nf12::detail::mma {

enum class Loop { kStream, kStaged, kRing, kBudget };

// BN weight rows x BM tokens per CTA, WN x WM warps; STAGES k64-deep stages
// for the stream loop (the staged loop double-buffers).
template <int BN_, int BM_, int WN_, int WM_, int STAGES_>
struct Tile {
  static constexpr int BN = BN_, BM = BM_, WN = WN_, WM = WM_, STAGES = STAGES_;
  static constexpr int kWarps = WN * WM;
  static constexpr int kThreads = 32 * kWarps;
  static constexpr int TN = BN / WN;  // warp tile: weight rows
  static constexpr int TM = BM / WM;  // warp tile: tokens
  static constexpr int RB = TN / 16;  // 16-row blocks per warp
  static constexpr int TF = TM / 8;   // 8-token MMA fragments per warp
  static constexpr int TG = TM / 16;  // ldmatrix.x4 loads per k16 step
  static_assert(TN % 16 == 0 && TM % 16 == 0, "warp tile must be 16-aligned");

  static constexpr int kXBytes = BM * 128;          // one k64 block of activations
  static constexpr int kABytes = BN / 16 * 2048;    // one k64 block of bf16t weights
  static constexpr int kStagingStride = BN + 8;     // bf16 elements; +16 B keeps transposed writes conflict-free

  template <WeightSource S>
  __host__ __device__ static constexpr int loop_bytes(Loop loop) {
    constexpr int sa = BN / 16 * WeightTraits<S>::kSuperAtomBytes;
    if (loop == Loop::kStream) return STAGES * (sa + kXBytes);
    if (loop == Loop::kRing) return 3 * (kABytes + kXBytes);
    return 2 * (kABytes + kXBytes);
  }
  // The split-K FP32 dump goes through shared memory in kSplitPasses slices
  // of the warp tile's token fragments, so large tiles fit.
  static constexpr int kSplitPasses = BN * BM * 4 > 96 * 1024 ? 2 : 1;
  static_assert(TF % kSplitPasses == 0, "split passes must divide the token fragments");
  // Epilogue staging and the split-K FP32 dump reuse memory. A problem's
  // extra shared memory follows the unsplit size. Such problems never split.
  template <WeightSource S>
  __host__ __device__ static constexpr int smem_bytes(Loop loop, bool split) {
    const int a = loop_bytes<S>(loop), b = BM * kStagingStride * 2, c = split ? BN * BM * 4 / kSplitPasses : 0;
    return a > b ? (a > c ? a : c) : (b > c ? b : c);
  }
};

// Cache per-device grants to avoid repeated driver calls for the shared-memory cap.
inline cudaError_t ensure_smem(const void* kernel, std::atomic<int> (&granted)[64], int device, int bytes) {
  if (device >= 0 && device < 64 && granted[device].load(std::memory_order_relaxed) >= bytes) return cudaSuccess;
  const cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (err == cudaSuccess && device >= 0 && device < 64) granted[device].store(bytes, std::memory_order_relaxed);
  return err;
}

__device__ __forceinline__ uint32_t movmatrix_trans(uint32_t v) {
  uint32_t r;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(r) : "r"(v));
  return r;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
  return v;
}

// Escaped groups (meta == 0) fetch their original values into the decoded
// slot with predicated cp.async. The caller commits the cp.async group.
__device__ __forceinline__ void store_group(uint32_t dst, const uint32_t (&r)[4], uint32_t meta,
                                            const uint4* entry) {
  asm volatile(
      "{\n.reg .pred e;\nsetp.eq.u32 e, %5, 0;\n@!e st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
      "@e cp.async.cg.shared.global [%0], [%6], 16;\n}\n" ::"r"(dst),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(meta), "l"(entry));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
}
__device__ __forceinline__ float4 ld_dsmem_f32x4(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(local_addr), "r"(rank));
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "r"(remote));
  return v;
}
#endif

// cp.async producer for the CTA's BN/16 super-atoms (SA bytes each) of one
// k64 block, spread over all threads.
template <class C, int SA>
struct WeightLoader {
  static constexpr int kChunks = C::BN / 16 * SA / 16;
  static constexpr int kIters = (kChunks + C::kThreads - 1) / C::kThreads;
  // Addresses are recomputed per fetch from one base pointer: per-chunk
  // pointers would pin 2 registers each for the whole kernel.
  const uint8_t* base;  // super-atom (n0 / 16, kb_begin)
  int64_t row_stride;   // bytes between consecutive 16-row blocks (exceeds 2^31 for large K)
  int last_rb;          // last row block inside the matrix, relative to n0 / 16

  template <class Prob, class Work>
  __device__ __forceinline__ WeightLoader(const Prob& prob, const Work& w) {
    base = prob.weights(w) + (static_cast<int64_t>(w.n0 / 16) * prob.kblocks() + w.kb_begin) * SA;
    row_stride = static_cast<int64_t>(prob.kblocks()) * SA;
    last_rb = prob.n() / 16 - 1 - w.n0 / 16;
  }
  __device__ __forceinline__ void load(uint32_t dst, int kb_rel) const {
    const uint8_t* src = base + static_cast<int64_t>(kb_rel) * SA;
#pragma unroll
    for (int i = 0; i < kIters; ++i) {
      const int c = threadIdx.x + i * C::kThreads;
      if (kChunks % C::kThreads != 0 && c >= kChunks) continue;
      const int rb = min(c / (SA / 16), last_rb);  // tail rows re-read the last block; never stored
      cp_async_16(dst + c * 16, src + rb * row_stride + (c % (SA / 16)) * 16);
    }
  }
};

// Shared-memory destination of 16-byte activation chunk `ch` (0-7) of token
// row `row` in a BM x 64 block: rows XOR-swizzled in 16-byte chunks so
// ldmatrix reads are conflict-free.
__device__ __forceinline__ uint32_t act_chunk_dst(uint32_t dst, int row, int ch) {
  return dst + row * 128 + ((ch ^ (row & 7)) * 16);
}

// Per-lane ldmatrix.x4 address of the B fragments of k16 step kk: matrices
// (tok 0-7, k lo), (tok 0-7, k hi), (tok 8-15, k lo), (tok 8-15, k hi) of
// each 16-token group of the warp.
template <class C>
struct BFragments {
  uint32_t row_off;
  int hi, swz;
  __device__ __forceinline__ BFragments(int wm, int lane) {
    row_off = (wm * C::TM + (lane & 7) + ((lane >> 4) << 3)) * 128;
    hi = (lane >> 3) & 1;
    swz = lane & 7;  // == row & 7
  }
  __device__ __forceinline__ void load(uint32_t (&b)[C::TF][2], uint32_t x_base, int kk) const {
    const uint32_t a = x_base + row_off + (((2 * kk + hi) ^ swz) * 16);
#pragma unroll
    for (int tg = 0; tg < C::TG; ++tg) {
      uint32_t r[4];
      ldmatrix_x4(r, a + tg * 16 * 128);
      b[2 * tg][0] = r[0];
      b[2 * tg][1] = r[1];
      b[2 * tg + 1][0] = r[2];
      b[2 * tg + 1][1] = r[3];
    }
  }
};

// FP32 partials of one split-K rank of a tile (below SM90): one float4 per
// thread and (row block, token fragment), [rb][tf][thread] so stores coalesce.
template <class C>
inline constexpr int kPartialsPerRank = C::RB * C::TF * C::kThreads;

// Whether a problem supports split-K without clusters: it then provides
// `template <class C> float4* partials(const Work&)`, the tile's workspace
// slot of kPartialsPerRank<C> float4 per rank.
template <class C, class Prob, class Work, class = void>
struct HasPartials : std::false_type {};
template <class C, class Prob, class Work>
struct HasPartials<C, Prob, Work,
                   std::void_t<decltype(std::declval<const Prob&>().template partials<C>(std::declval<const Work&>()))>>
    : std::true_type {};

// Reduce split-K, apply per-fragment terms, round to BF16 and transpose to
// [token][n] through shared memory for coalesced stores. Requires drained
// cp.async traffic and a CTA barrier. A following tile must pass another
// barrier before reusing the staging memory read by this epilogue.
template <class C, class Prob, class Work>
__device__ __forceinline__ void finish(float (&acc)[C::RB][C::TF][4], const Prob& prob, uint8_t* smem,
                                       const Work& w) {
  constexpr int RB = C::RB, TF = C::TF;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int wn = warp % C::WN, wm = warp / C::WN;
  const int splits = gridDim.z;
  uint32_t rank = 0;
  if (splits > 1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    rank = cluster_rank();
    constexpr int kPassTF = TF / C::kSplitPasses;
    float4* dump = reinterpret_cast<float4*>(smem);
#pragma unroll
    for (int pass = 0; pass < C::kSplitPasses; ++pass) {
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int f = 0; f < kPassTF; ++f) {
          const int tf = pass * kPassTF + f;
          dump[((warp * RB + rb) * kPassTF + f) * 32 + lane] =
              make_float4(acc[rb][tf][0], acc[rb][tf][1], acc[rb][tf][2], acc[rb][tf][3]);
        }
      cluster_sync();
#pragma unroll
      for (int f = 0; f < kPassTF; ++f) {
        const int tf = pass * kPassTF + f;
        if ((wm * TF + tf) % splits != static_cast<int>(rank)) continue;
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) {
          const uint32_t addr = smem_addr(&dump[((warp * RB + rb) * kPassTF + f) * 32 + lane]);
          float4 s = ld_dsmem_f32x4(addr, 0);
          for (int r = 1; r < splits; ++r) {
            const float4 v = ld_dsmem_f32x4(addr, r);
            s.x += v.x;
            s.y += v.y;
            s.z += v.z;
            s.w += v.w;
          }
          acc[rb][tf][0] = s.x;
          acc[rb][tf][1] = s.y;
          acc[rb][tf][2] = s.z;
          acc[rb][tf][3] = s.w;
        }
      }
      // No CTA may overwrite, reuse or release its shared memory while peers
      // still read it.
      cluster_sync();
    }
#else
    // No clusters below SM90: rank blockIdx.z stores its FP32 fragments to
    // the problem's workspace slot and large_m_split_reduce_kernel sums the
    // ranks in order and runs this epilogue.
    if constexpr (HasPartials<C, Prob, Work>::value) {
      float4* dst = prob.template partials<C>(w) + static_cast<size_t>(blockIdx.z) * kPartialsPerRank<C>;
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int tf = 0; tf < TF; ++tf)
          dst[(rb * TF + tf) * C::kThreads + tid] =
              make_float4(acc[rb][tf][0], acc[rb][tf][1], acc[rb][tf][2], acc[rb][tf][3]);
      return;
    } else {
      __trap();
    }
#endif
  }

  const int g = lane >> 2, t = lane & 3;
  __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem);
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    const auto terms = prob.template row_terms<C>(w, wn * C::TN + rb * 16 + g);
#pragma unroll
    for (int tf = 0; tf < TF; ++tf) {
      if (splits > 1 && (wm * TF + tf) % splits != static_cast<int>(rank)) continue;
      float v[4] = {acc[rb][tf][0], acc[rb][tf][1], acc[rb][tf][2], acc[rb][tf][3]};
      prob.template fragment<C>(w, terms, wm * C::TM + tf * 8 + 2 * t, v);
      // v0/v1 represent row g, tokens 2t/2t+1 in ldmatrix layout.
      // movmatrix.trans produces token g, rows 2t/2t+1.
      const uint32_t lo = movmatrix_trans(pack_bf16x2(v[0], v[1]));
      const uint32_t hi = movmatrix_trans(pack_bf16x2(v[2], v[3]));
      const int tok = wm * C::TM + tf * 8 + g;
      const int col = wn * C::TN + rb * 16 + 2 * t;
      *reinterpret_cast<uint32_t*>(&staging[tok * C::kStagingStride + col]) = lo;
      *reinterpret_cast<uint32_t*>(&staging[tok * C::kStagingStride + col + 8]) = hi;
    }
  }
  __syncthreads();

  constexpr int kRowChunks = C::BN / 8;
#pragma unroll
  for (int c = tid; c < C::BM * kRowChunks; c += C::kThreads) {
    const int row = c / kRowChunks, ch = c % kRowChunks;
    __nv_bfloat16* dst = prob.out_row(w, row);
    if (dst != nullptr && ch * 8 < prob.n() - w.n0 && (splits == 1 || (row / 8) % splits == static_cast<int>(rank)))
      *reinterpret_cast<uint4*>(dst + ch * 8) = *reinterpret_cast<const uint4*>(&staging[row * C::kStagingStride + ch * 8]);
  }
}

template <class C>
__device__ __forceinline__ void zero(float (&acc)[C::RB][C::TF][4]) {
#pragma unroll
  for (int i = 0; i < C::RB; ++i)
#pragma unroll
    for (int j = 0; j < C::TF; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
}


template <class C, WeightSource S, class Prob>
__global__ void __launch_bounds__(C::kThreads, 1) large_m_stream_kernel(const __grid_constant__ typename Prob::Params params) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr bool kNF12 = S == WeightSource::kNF12;
  constexpr int SA = WeightTraits<S>::kSuperAtomBytes;
  constexpr int kWBytes = C::BN / 16 * SA;
  constexpr int kStageBytes = kWBytes + C::kXBytes;
  constexpr int RB = C::RB, TF = C::TF;

  Prob prob(params, smem + C::template smem_bytes<S>(Loop::kStream, false));
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int wn = warp % C::WN, wm = warp / C::WN;
  const uint32_t smem0 = smem_addr(smem);
  const BFragments<C> bfrag(wm, lane);
  // Weight planes of this warp's row blocks: lane-contiguous 16-byte loads.
  const uint32_t w_lane = (wn * RB) * SA + lane * 16;

  typename Prob::Work work;
  while (prob.template next<C>(work)) {
  const int nkb = work.nkb;
  const WeightLoader<C, SA> wload(prob, work);
  const auto xload = prob.template act<C>(work);
  const uint32_t base_x4 = prob.base_x4(work);
  const uint4* const escapes = prob.escapes(work);
  const uint32_t num_escapes = prob.num_escapes(work);
  auto load_stage = [&](int slot, int kb_rel) {
    wload.load(smem0 + slot * kStageBytes, kb_rel);
    xload.load(smem0 + slot * kStageBytes + kWBytes, kb_rel);
  };

  float acc[RB][TF][4];
  zero<C>(acc);

  // Software pipeline per k16 step kk: the A fragments of step kk + 1 are
  // decoded between the two halves of step kk's MMAs (double-buffered in
  // a_frag), so the decode's dependency chains hide behind tensor work, and
  // their raw planes and the B fragments are loaded earlier still. The kb
  // loop is unrolled by two and per-stage state is indexed by stage parity,
  // so the next stage's planes can land while the current stage still
  // decodes from its own.
  uint32_t a_frag[2][RB][4];
  uint32_t b_frag[2][TF][2];
  // NF12 planes per stage parity: metas of all 4 atoms, (L0, L1) of atoms
  // 0-1 and of atoms 2-3.
  uint4 meta[2][RB] = {}, lo01[2][RB] = {}, lo23[2][RB] = {};
  auto group_of = [&](auto par, int rb, int kk) {
    constexpr int P = decltype(par)::value;
    const uint4& m = meta[P][rb];
    const uint4& l = kk < 2 ? lo01[P][rb] : lo23[P][rb];
    return Group{kk == 0 ? m.x : kk == 1 ? m.y : kk == 2 ? m.z : m.w, (kk & 1) ? l.z : l.x, (kk & 1) ? l.w : l.y};
  };

  // Escape entries are consecutive along K within a 16-row block. Once any
  // escaped group's index is known, `frontier` predicts later stages' starts.
  // Prefetch a 32-entry window per row block one stage ahead, with lane i
  // holding entry start + i, and patch groups by shuffle. Groups outside
  // the window or before the first sighting use direct loads. Bitmasks
  // track validity separately because every uint32 index can be real.
  uint32_t esc_mask = 0;  // bit 4 * rb + atom: some lane's group is escaped in the current stage
  uint32_t known = 0;     // bit rb: frontier[rb] is valid
  uint32_t win_ok[2] = {0, 0};  // bit rb: win[par][rb] holds the entries from win_start[par][rb]
  uint4 win[2][RB] = {};
  uint32_t win_start[2][RB] = {}, frontier[RB] = {};

  // NF12 planes of a stage: metas + (L0, L1) of atoms 0-1 (kk = 0), or the
  // (L0, L1) plane of atoms 2-3 (kk = 2).
  auto load_planes = [&](int slot, int kk, auto par) {
    if constexpr (kNF12) {
      constexpr int P = decltype(par)::value;
      const uint32_t base = smem0 + slot * kStageBytes + w_lane;
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        if (kk == 0) {
          meta[P][rb] = lds128(base + rb * SA);
          lo01[P][rb] = lds128(base + rb * SA + kPlaneBytes);
        } else {
          lo23[P][rb] = lds128(base + rb * SA + 2 * kPlaneBytes);
        }
      }
    }
  };
  // Decode step kk into a_frag[kk & 1], patched by fix_a for NF12 escapes.
  // bf16t loads the same fragments directly.
  auto make_a = [&](int slot, int kk, auto par) {
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      if constexpr (kNF12) {
        decode_bulk(group_of(par, rb, kk), base_x4, a_frag[kk & 1][rb]);
      } else {
        const uint4 v = lds128(smem0 + slot * kStageBytes + w_lane + rb * SA + kk * kPlaneBytes);
        a_frag[kk & 1][rb][0] = v.x;
        a_frag[kk & 1][rb][1] = v.y;
        a_frag[kk & 1][rb][2] = v.z;
        a_frag[kk & 1][rb][3] = v.w;
      }
    }
  };

  // Escape census of the stage (parity par) whose first planes are loaded:
  // sets esc_mask, advances frontier, and prefetches the window of the stage
  // after it into the other parity (unless it is already there).
  auto scan = [&](int slot, auto par) {
    if constexpr (kNF12) {
      constexpr int P = decltype(par)::value, N = 1 - P;
      bool any = false;
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        const uint4& m = meta[P][rb];
        any |= (m.x == 0u) | (m.y == 0u) | (m.z == 0u) | (m.w == 0u);
      }
      esc_mask = 0;
      if (__any_sync(0xffffffffu, any)) {
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) {
          const uint4& m = meta[P][rb];
          const uint4& l = lo01[P][rb];
          uint32_t count = 0;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t bal = __ballot_sync(0xffffffffu, group_of(par, rb, j).meta == 0u);
            esc_mask |= (bal != 0u ? 1u : 0u) << (4 * rb + j);
            count += __popc(bal);
          }
          if (count != 0) {
            // The smallest escape index in this super-atom is its first one.
            const uint4 hi = lds128(smem0 + slot * kStageBytes + w_lane + rb * SA + 2 * kPlaneBytes);
            uint32_t first = 0xffffffffu;  // above every index: tables hold at most 2^32 - 1 entries
            if (m.x == 0u) first = min(first, l.x);
            if (m.y == 0u) first = min(first, l.z);
            if (m.z == 0u) first = min(first, hi.x);
            if (m.w == 0u) first = min(first, hi.z);
            frontier[rb] = __reduce_min_sync(0xffffffffu, first) + count;
            known |= 1u << rb;
          }
        }
      }
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        if (!((known >> rb) & 1u)) continue;
        if (!((win_ok[N] >> rb) & 1u) || frontier[rb] != win_start[N][rb]) {
          win_start[N][rb] = frontier[rb];
          win_ok[N] |= 1u << rb;
          // Stale entries in lanes past the table end cannot match real indices.
          if (frontier[rb] < num_escapes && static_cast<uint32_t>(lane) < num_escapes - frontier[rb])
            win[N][rb] = __ldg(escapes + frontier[rb] + lane);
        }
      }
    }
  };

  // Patch the escaped groups of step kk (stage parity par) in a_frag[kk & 1].
  // Only atoms flagged in esc_mask (warp-uniform) do any work.
  auto fix_a = [&](int kk, auto par) {
    constexpr int P = decltype(par)::value;
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      if ((esc_mask >> (4 * rb + kk)) & 1u) {
        const Group g = group_of(par, rb, kk);
        const uint32_t slot = g.lo0 - win_start[P][rb];
        const int src = static_cast<int>(slot & 31u);
        uint4 v;
        v.x = __shfl_sync(0xffffffffu, win[P][rb].x, src);
        v.y = __shfl_sync(0xffffffffu, win[P][rb].y, src);
        v.z = __shfl_sync(0xffffffffu, win[P][rb].z, src);
        v.w = __shfl_sync(0xffffffffu, win[P][rb].w, src);
        const bool miss = g.meta == 0u && (!((win_ok[P] >> rb) & 1u) || slot >= 32u);
        if (__any_sync(0xffffffffu, miss)) {
          if (miss) v = __ldg(escapes + g.lo0);
        }
        if (g.meta == 0u) {
          a_frag[kk & 1][rb][0] = v.x;
          a_frag[kk & 1][rb][1] = v.y;
          a_frag[kk & 1][rb][2] = v.z;
          a_frag[kk & 1][rb][3] = v.w;
        }
      }
    }
  };

  constexpr int kMmas = RB * TF;
  auto mma_range = [&](int kk, int begin, int end) {
#pragma unroll
    for (int i = begin; i < end; ++i)
      mma_bf16_16816(acc[i / TF][i % TF], a_frag[kk & 1][i / TF], b_frag[kk & 1][i % TF][0],
                     b_frag[kk & 1][i % TF][1]);
  };
  auto x_of = [&](int slot) { return smem0 + slot * kStageBytes + kWBytes; };

  // Four k16 steps of stage kb, with parity par and optional escape patching.
  auto stage = [&](int kb, auto esc, auto par) {
    using Nxt = std::integral_constant<int, 1 - decltype(par)::value>;
    const int slot = kb % C::STAGES;
    const int nslot = slot + 1 == C::STAGES ? 0 : slot + 1;
    const bool more = kb + 1 < nkb;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      if constexpr (decltype(esc)::value) fix_a(kk, par);
      if (kk == 3) {
        // Stage kb + 1 has landed, freeing the next refill slot.
        cp_async_wait<C::STAGES - 2>();
        __syncthreads();
        if (more) {
          load_planes(nslot, 0, Nxt{});
          bfrag.load(b_frag[0], x_of(nslot), 0);
        }
      }
      mma_range(kk, 0, kMmas / 2);
      if (kk < 3) {
        make_a(slot, kk + 1, par);
        if (kk == 0) load_planes(slot, 2, par);
        bfrag.load(b_frag[(kk + 1) & 1], x_of(slot), kk + 1);
      } else if (more) {
        make_a(nslot, 0, Nxt{});
      }
      mma_range(kk, kMmas / 2, kMmas);
      if (kk == 3 && more) scan(nslot, Nxt{});
    }
  };

  auto iteration = [&](int kb, auto par) {
    // Every warp passed its final-read barrier, so the prior slot can refill.
    const int pre = kb + C::STAGES - 1;
    if (pre < nkb) load_stage(pre % C::STAGES, pre);
    cp_async_commit();
    if (esc_mask != 0)
      stage(kb, std::true_type{}, par);
    else
      stage(kb, std::false_type{}, par);
  };

#pragma unroll
  for (int s = 0; s < C::STAGES - 1; ++s) {
    if (s < nkb) load_stage(s, s);
    cp_async_commit();
  }
  cp_async_wait<C::STAGES - 2>();
  __syncthreads();
  if (nkb > 0) {
    load_planes(0, 0, std::integral_constant<int, 0>{});
    bfrag.load(b_frag[0], x_of(0), 0);
    make_a(0, 0, std::integral_constant<int, 0>{});
    scan(0, std::integral_constant<int, 0>{});
  }
  for (int kb = 0; kb < nkb; kb += 2) {
    iteration(kb, std::integral_constant<int, 0>{});
    if (kb + 1 < nkb) iteration(kb + 1, std::integral_constant<int, 1>{});
  }
  cp_async_wait<0>();
  __syncthreads();
  finish<C>(acc, prob, smem, work);
  }
}

// Asks L2 to fetch the CTA's BN / 16 compressed super-atoms of k64 block
// kb_rel (relative to the CTA's K range); no registers, no completion.
template <class C, class Prob, class Work>
__device__ __forceinline__ void l2_prefetch_super_atoms(const Prob& prob, const Work& w, int kb_rel) {
  const int last_rb = prob.n() / 16 - 1 - w.n0 / 16;
  const int64_t kblocks = prob.kblocks();
  const uint8_t* base = prob.weights(w) + (static_cast<int64_t>(w.n0 / 16) * kblocks + w.kb_begin + kb_rel) * kSuperAtomBytes;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // One bulk prefetch per contiguous 1536-byte super-atom.
  const int rb = threadIdx.x;
  if (rb < C::BN / 16 && rb <= last_rb)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(base + static_cast<int64_t>(rb) * kblocks *
                                                                              kSuperAtomBytes),
                 "n"(kSuperAtomBytes));
#else
  constexpr int kLines = kSuperAtomBytes / 128;
  const int rb = threadIdx.x / kLines, line = threadIdx.x % kLines;
  if (rb < C::BN / 16 && rb <= last_rb)
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(base + static_cast<int64_t>(rb) * kblocks * kSuperAtomBytes +
                                                     line * 128));
#endif
}

// Shared memory: A[2] (bf16t-layout weights) and X[2] (activations). Stage s
// computes from A[s & 1] and X[s & 1]. Activations (and bf16t weights) of
// block t are fetched into slot t & 1 after the step-3 barrier of stage
// t - 2. NF12 block t never touches shared memory compressed: each thread
// loads its super-atom lanes into registers during stage t - 2 and decodes
// them into A[t & 1] during stage t - 1. Keeping compressed tiles out of
// shared memory allows 64x64 warp tiles to fit in 99 KB.

template <class C, WeightSource S, class Prob>
__global__ void __launch_bounds__(C::kThreads, 1) large_m_staged_kernel(const __grid_constant__ typename Prob::Params params) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr bool kNF12 = S == WeightSource::kNF12;
  constexpr int RB = C::RB, TF = C::TF;
  // Decode work: one (super-atom, lane) unit = 4 groups per thread and iteration.
  constexpr int kUnits = C::BN / 16 * 32;
  constexpr int kDecIters = (kUnits + C::kThreads - 1) / C::kThreads;
  // Leave enough k16 steps for decode stores and escape fetches to land
  // before the step-3 barrier.
  constexpr int kDecFirst = (kDecIters == 1 && C::BN * C::BM >= 128 * 256) ? 1 : 0;
  static_assert(kDecFirst + kDecIters <= 3, "decode must finish before the step-3 barrier");

  Prob prob(params, smem + C::template smem_bytes<S>(Loop::kStaged, false));
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int wn = warp % C::WN, wm = warp / C::WN;
  const uint32_t smem0 = smem_addr(smem);
  auto a_of = [&](int i) { return smem0 + i * C::kABytes; };
  auto x_of = [&](int i) { return smem0 + 2 * C::kABytes + i * C::kXBytes; };
  const BFragments<C> bfrag(wm, lane);
  const uint32_t a_lane = (wn * RB) * 2048 + lane * 16;
  auto dec_unit = [&](int i) { return tid + i * C::kThreads; };
  auto dec_active = [&](int i) { return kUnits % C::kThreads == 0 || dec_unit(i) < kUnits; };

  typename Prob::Work work;
  while (prob.template next<C>(work)) {
  const int nkb = work.nkb;
  const WeightLoader<C, kSuperAtomBytesBf16t> wload(prob, work);
  const auto xload = prob.template act<C>(work);
  const uint32_t base_x4 = prob.base_x4(work);
  const uint4* const escapes = prob.escapes(work);

  float acc[RB][TF][4];
  zero<C>(acc);
  uint32_t a_frag[2][RB][4];
  uint32_t b_frag[2][TF][2];
  auto load_frags = [&](int buf, int kk) {
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      const uint4 v = lds128(a_of(buf) + a_lane + rb * 2048 + kk * kPlaneBytes);
      a_frag[kk & 1][rb][0] = v.x;
      a_frag[kk & 1][rb][1] = v.y;
      a_frag[kk & 1][rb][2] = v.z;
      a_frag[kk & 1][rb][3] = v.w;
    }
    bfrag.load(b_frag[kk & 1], x_of(buf), kk);
  };

  // Unit u = tid + i * kThreads maps to lane u % 32 of row block u / 32.
  // pl[i] holds its three planes for the next decode. Escape cp.async
  // copies land in the A slot before the step-3 barrier.
  uint4 pl[kDecIters][3];
  const uint8_t* unit_src[kDecIters] = {};
  if constexpr (kNF12) {
    const int last_rb = prob.n() / 16 - 1 - work.n0 / 16;
#pragma unroll
    for (int i = 0; i < kDecIters; ++i) {
      const int u = dec_unit(i);
      const int rb = min(u / 32, last_rb);  // tail rows re-read the last block; never stored
      unit_src[i] = prob.weights(work) +
                    ((static_cast<int64_t>(work.n0 / 16 + rb) * prob.kblocks() + work.kb_begin) * kSuperAtomBytes) +
                    (u % 32) * 16;
    }
  }
  auto dec_fetch = [&](int i, int kb_rel) {
    const uint8_t* src = unit_src[i] + static_cast<int64_t>(kb_rel) * kSuperAtomBytes;
#pragma unroll
    for (int k = 0; k < 3; ++k) pl[i][k] = ldg_stream(src + k * kPlaneBytes);
  };
  auto dec_store = [&](int buf, int i) {
    const int u = dec_unit(i);
    const uint32_t dst = a_of(buf) + (u / 32) * 2048 + (u % 32) * 16;
    const Group g[4] = {{pl[i][0].x, pl[i][1].x, pl[i][1].y},
                        {pl[i][0].y, pl[i][1].z, pl[i][1].w},
                        {pl[i][0].z, pl[i][2].x, pl[i][2].y},
                        {pl[i][0].w, pl[i][2].z, pl[i][2].w}};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t r[4];
      decode_bulk(g[j], base_x4, r);
      store_group(dst + j * kPlaneBytes, r, g[j].meta, escapes + g[j].lo0);
    }
  };

  // A register fetch one stage ahead does not cover DRAM latency, so the
  // compressed blocks are first pulled into L2 kPrefetch stages ahead.
  constexpr int kPrefetch = 4;
  auto prefetch = [&](int kb_rel) {
    if constexpr (kNF12) {
      if (kb_rel < nkb) l2_prefetch_super_atoms<C>(prob, work, kb_rel);
    }
  };

  // Producers run right after the barrier in step 3 of stage s: from then on
  // the slots read by stage s are free (its step-3 fragments are already in
  // registers), which gives every fetch a full stage of latency slack.
  auto produce = [&](int s) {
    if (s + 2 < nkb) {
      xload.load(x_of(s & 1), s + 2);
      if constexpr (!kNF12) wload.load(a_of(s & 1), s + 2);
    }
    cp_async_commit();
    prefetch(s + kPrefetch);
  };

  if (nkb > 0) {
#pragma unroll
    for (int t = 2; t < kPrefetch; ++t) prefetch(t);
    xload.load(x_of(0), 0);
    if constexpr (!kNF12) wload.load(a_of(0), 0);
    if (nkb > 1) {
      xload.load(x_of(1), 1);
      if constexpr (!kNF12) wload.load(a_of(1), 1);
    }
    if constexpr (kNF12) {
#pragma unroll
      for (int i = 0; i < kDecIters; ++i) {
        if (!dec_active(i)) continue;
        dec_fetch(i, 0);
        dec_store(0, i);
        dec_fetch(i, min(1, nkb - 1));
      }
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    load_frags(0, 0);
  }

  for (int s = 0; s < nkb; ++s) {
    const int buf = s & 1, nbuf = buf ^ 1;
    const bool more = s + 1 < nkb;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      if (kk == 3) {
        cp_async_wait<0>();
        __syncthreads();
        produce(s);
        if (more) load_frags(nbuf, 0);
      } else {
        load_frags(buf, kk + 1);
      }
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int tf = 0; tf < TF; ++tf)
          mma_bf16_16816(acc[rb][tf], a_frag[kk & 1][rb], b_frag[kk & 1][tf][0], b_frag[kk & 1][tf][1]);
      if constexpr (kNF12) {
        // Unconditional (no branch on `more`), so the decode shares a basic
        // block with the MMAs and the scheduler interleaves the two. In the
        // last stage it re-decodes the last block into the unused slot, and
        // the fetch past the end re-reads the last block.
        const int i = kk - kDecFirst;
        if (i >= 0 && i < kDecIters && dec_active(i)) {
          dec_store(nbuf, i);
          dec_fetch(i, min(s + 2, nkb - 1));
        }
        if (i == kDecIters - 1) cp_async_commit();
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();
  finish<C>(acc, prob, smem, work);
  }
}

// kBudget keeps only the current step's A fragments and one-step B lookahead.
// Step 3 A fragments are loaded before releasing the consumed slot.
// Decode, MMA and split reduction orders match kStaged.
template <class C, WeightSource S, class Prob>
__global__ void __launch_bounds__(C::kThreads, 1)
    large_m_budget_kernel(const __grid_constant__ typename Prob::Params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
  extern __shared__ __align__(128) uint8_t smem[];
  static_assert(C::STAGES == 2, "budget loop uses two k64 stages");
  constexpr bool kNF12 = S == WeightSource::kNF12;
  constexpr int RB = C::RB, TF = C::TF;
  constexpr int kUnits = C::BN / 16 * 32;
  constexpr int kDecIters = (kUnits + C::kThreads - 1) / C::kThreads;
  constexpr int kDecFirst = (kDecIters == 1 && C::BN * C::BM >= 128 * 256) ? 1 : 0;
  static_assert(kDecFirst + kDecIters <= 3, "decode must finish before the step-3 barrier");
  Prob prob(params, smem + C::template smem_bytes<S>(Loop::kBudget, false));
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int wn = warp % C::WN, wm = warp / C::WN;
  const uint32_t smem0 = smem_addr(smem);
  auto a_of = [&](int i) { return smem0 + i * C::kABytes; };
  auto x_of = [&](int i) { return smem0 + 2 * C::kABytes + i * C::kXBytes; };
  const BFragments<C> bfrag(wm, lane);
  const uint32_t a_lane = wn * RB * 2048 + lane * 16;
  typename Prob::Work work;
  while (prob.template next<C>(work)) {
    const int nkb = work.nkb;
    const WeightLoader<C, kSuperAtomBytesBf16t> wload(prob, work);
    const auto xload = prob.template act<C>(work);
    const uint32_t base_x4 = prob.base_x4(work);
    const uint4* const escapes = prob.escapes(work);
    float acc[RB][TF][4];
    zero<C>(acc);
    uint32_t a[RB][4], b[2][TF][2];
    uint4 pl[kDecIters][3];
    const uint8_t* unit_src[kDecIters] = {};
    if constexpr (kNF12) {
      const int last_rb = prob.n() / 16 - 1 - work.n0 / 16;
#pragma unroll
      for (int i = 0; i < kDecIters; ++i) {
        const int u = tid + i * C::kThreads;
        unit_src[i] = prob.weights(work) +
                      (static_cast<int64_t>(work.n0 / 16 + min(u / 32, last_rb)) * prob.kblocks() +
                       work.kb_begin) * kSuperAtomBytes + (u % 32) * 16;
      }
    }
    auto dec_fetch = [&](int i, int kb) {
      const uint8_t* src = unit_src[i] + static_cast<int64_t>(kb) * kSuperAtomBytes;
#pragma unroll
      for (int p = 0; p < 3; ++p) pl[i][p] = ldg_stream(src + p * kPlaneBytes);
    };
    auto dec_store = [&](int buf, int i) {
      const int u = tid + i * C::kThreads;
      const uint32_t dst = a_of(buf) + (u / 32) * 2048 + (u % 32) * 16;
      const Group g[4] = {{pl[i][0].x, pl[i][1].x, pl[i][1].y},
                          {pl[i][0].y, pl[i][1].z, pl[i][1].w},
                          {pl[i][0].z, pl[i][2].x, pl[i][2].y},
                          {pl[i][0].w, pl[i][2].z, pl[i][2].w}};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t r[4];
        decode_bulk(g[j], base_x4, r);
        store_group(dst + j * kPlaneBytes, r, g[j].meta, escapes + g[j].lo0);
      }
    };
    auto prefetch = [&](int kb) {
      if constexpr (kNF12) {
        if (kb < nkb) l2_prefetch_super_atoms<C>(prob, work, kb);
      }
    };
    auto produce = [&](int s) {
      if (s + 2 < nkb) {
        xload.load(x_of(s & 1), s + 2);
        if constexpr (!kNF12) wload.load(a_of(s & 1), s + 2);
      }
      cp_async_commit();
      prefetch(s + 4);
    };
    if (nkb > 0) {
      prefetch(2);
      prefetch(3);
      xload.load(x_of(0), 0);
      if constexpr (!kNF12) wload.load(a_of(0), 0);
      if (nkb > 1) {
        xload.load(x_of(1), 1);
        if constexpr (!kNF12) wload.load(a_of(1), 1);
      }
      if constexpr (kNF12) {
#pragma unroll
        for (int i = 0; i < kDecIters; ++i) {
          if (tid + i * C::kThreads >= kUnits) continue;
          dec_fetch(i, 0);
          dec_store(0, i);
          dec_fetch(i, min(1, nkb - 1));
        }
      }
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();
      bfrag.load(b[0], x_of(0), 0);
    }
    for (int s = 0; s < nkb; ++s) {
      const int buf = s & 1, nbuf = buf ^ 1;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) {
          const uint4 v = lds128(a_of(buf) + a_lane + rb * 2048 + kk * kPlaneBytes);
          a[rb][0] = v.x;
          a[rb][1] = v.y;
          a[rb][2] = v.z;
          a[rb][3] = v.w;
        }
        if (kk == 3) {
          cp_async_wait<0>();
          __syncthreads();
          produce(s);
          if (s + 1 < nkb) bfrag.load(b[0], x_of(nbuf), 0);
        } else {
          bfrag.load(b[(kk + 1) & 1], x_of(buf), kk + 1);
        }
#pragma unroll
        for (int rb = 0; rb < RB; ++rb)
#pragma unroll
          for (int tf = 0; tf < TF; ++tf)
            mma_bf16_16816(acc[rb][tf], a[rb], b[kk & 1][tf][0], b[kk & 1][tf][1]);
        if constexpr (kNF12) {
          // As in kStaged, the last iteration re-decodes the last block into
          // the unused slot, keeping decode and MMA in one scheduling block.
          const int i = kk - kDecFirst;
          if (i >= 0 && i < kDecIters && tid + i * C::kThreads < kUnits) {
            dec_store(nbuf, i);
            dec_fetch(i, min(s + 2, nkb - 1));
          }
          if (i == kDecIters - 1) cp_async_commit();
        }
      }
    }
    cp_async_wait<0>();
    __syncthreads();
    finish<C>(acc, prob, smem, work);
  }
#else
  __trap();  // Host dispatch excludes kBudget on SM90+.
#endif
}

// Loop::kRing: the staged loop with three slots, A[3] and X[3], for two
// stages of latency slack on every fetch. Stage s computes from slot s % 3.
// Activations (and bf16t weights) of block t are fetched into slot t % 3
// after the step-3 barrier of stage t - 3. NF12 block t is decoded into
// A[t % 3] during stage t - 2 from planes loaded during stage t - 3.
// Escape fetches commit before the producer group so the step-3 wait may
// leave the newest two groups in flight. No split-K.
template <class C, WeightSource S, class Prob>
__global__ void __launch_bounds__(C::kThreads, 1) large_m_ring_kernel(const __grid_constant__ typename Prob::Params params) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr bool kNF12 = S == WeightSource::kNF12;
  constexpr int RB = C::RB, TF = C::TF;
  constexpr int kUnits = C::BN / 16 * 32;
  constexpr int kDecIters = (kUnits + C::kThreads - 1) / C::kThreads;
  constexpr int kDecFirst = (kDecIters == 1 && C::BN * C::BM >= 128 * 256) ? 1 : 0;
  static_assert(kDecFirst + kDecIters <= 3, "decode must finish before the step-3 barrier");

  Prob prob(params, smem + C::template smem_bytes<S>(Loop::kRing, false));
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int wn = warp % C::WN, wm = warp / C::WN;
  const uint32_t smem0 = smem_addr(smem);
  auto a_of = [&](int i) { return smem0 + i * C::kABytes; };
  auto x_of = [&](int i) { return smem0 + 3 * C::kABytes + i * C::kXBytes; };
  const BFragments<C> bfrag(wm, lane);
  const uint32_t a_lane = (wn * RB) * 2048 + lane * 16;
  auto dec_unit = [&](int i) { return tid + i * C::kThreads; };
  auto dec_active = [&](int i) { return kUnits % C::kThreads == 0 || dec_unit(i) < kUnits; };

  typename Prob::Work work;
  while (prob.template next<C>(work)) {
  const int nkb = work.nkb;
  const WeightLoader<C, kSuperAtomBytesBf16t> wload(prob, work);
  const auto xload = prob.template act<C>(work);
  const uint32_t base_x4 = prob.base_x4(work);
  const uint4* const escapes = prob.escapes(work);

  float acc[RB][TF][4];
  zero<C>(acc);
  uint32_t a_frag[2][RB][4];
  uint32_t b_frag[2][TF][2];
  auto load_frags = [&](int slot, int kk) {
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      const uint4 v = lds128(a_of(slot) + a_lane + rb * 2048 + kk * kPlaneBytes);
      a_frag[kk & 1][rb][0] = v.x;
      a_frag[kk & 1][rb][1] = v.y;
      a_frag[kk & 1][rb][2] = v.z;
      a_frag[kk & 1][rb][3] = v.w;
    }
    bfrag.load(b_frag[kk & 1], x_of(slot), kk);
  };

  uint4 pl[kDecIters][3];
  const uint8_t* unit_src[kDecIters] = {};
  if constexpr (kNF12) {
    const int last_rb = prob.n() / 16 - 1 - work.n0 / 16;
#pragma unroll
    for (int i = 0; i < kDecIters; ++i) {
      const int u = dec_unit(i);
      const int rb = min(u / 32, last_rb);  // tail rows re-read the last block; never stored
      unit_src[i] = prob.weights(work) +
                    ((static_cast<int64_t>(work.n0 / 16 + rb) * prob.kblocks() + work.kb_begin) * kSuperAtomBytes) +
                    (u % 32) * 16;
    }
  }
  auto dec_fetch = [&](int i, int kb_rel) {
    const uint8_t* src = unit_src[i] + static_cast<int64_t>(kb_rel) * kSuperAtomBytes;
#pragma unroll
    for (int k = 0; k < 3; ++k) pl[i][k] = ldg_stream(src + k * kPlaneBytes);
  };
  auto dec_store = [&](int slot, int i) {
    const int u = dec_unit(i);
    const uint32_t dst = a_of(slot) + (u / 32) * 2048 + (u % 32) * 16;
    const Group g[4] = {{pl[i][0].x, pl[i][1].x, pl[i][1].y},
                        {pl[i][0].y, pl[i][1].z, pl[i][1].w},
                        {pl[i][0].z, pl[i][2].x, pl[i][2].y},
                        {pl[i][0].w, pl[i][2].z, pl[i][2].w}};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t r[4];
      decode_bulk(g[j], base_x4, r);
      store_group(dst + j * kPlaneBytes, r, g[j].meta, escapes + g[j].lo0);
    }
  };

  // Pull compressed blocks into L2 kPrefetch stages ahead of register loads.
  constexpr int kPrefetch = 4;
  auto prefetch = [&](int kb_rel) {
    if constexpr (kNF12) {
      if (kb_rel < nkb) l2_prefetch_super_atoms<C>(prob, work, kb_rel);
    }
  };

  int cur = 0, nx1 = 1, nx2 = 2;  // slots of blocks s, s + 1, s + 2
  if (nkb > 0) {
#pragma unroll
    for (int t = 3; t < kPrefetch; ++t) prefetch(t);
#pragma unroll
    for (int b = 0; b < 3; ++b) {
      if (b < nkb) {
        xload.load(x_of(b), b);
        if constexpr (!kNF12) wload.load(a_of(b), b);
      }
    }
    if constexpr (kNF12) {
#pragma unroll
      for (int i = 0; i < kDecIters; ++i) {
        if (!dec_active(i)) continue;
        dec_fetch(i, 0);
        dec_store(0, i);
        dec_fetch(i, min(1, nkb - 1));
        dec_store(1, i);
        dec_fetch(i, min(2, nkb - 1));
      }
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    load_frags(0, 0);
  }

  for (int s = 0; s < nkb; ++s) {
    const bool more = s + 1 < nkb;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      if (kk == 3) {
        // Block s + 1 fetches and escapes must have landed. Both producer and
        // escape groups for block s + 2 may remain in flight.
        cp_async_wait<kNF12 ? 2 : 1>();
        __syncthreads();
        if (s + 3 < nkb) {
          xload.load(x_of(cur), s + 3);
          if constexpr (!kNF12) wload.load(a_of(cur), s + 3);
        }
        cp_async_commit();
        prefetch(s + kPrefetch);
        if (more) load_frags(nx1, 0);
      } else {
        load_frags(cur, kk + 1);
      }
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int tf = 0; tf < TF; ++tf)
          mma_bf16_16816(acc[rb][tf], a_frag[kk & 1][rb], b_frag[kk & 1][tf][0], b_frag[kk & 1][tf][1]);
      if constexpr (kNF12) {
        // Decode block s + 2 into the slot released at the prior barrier.
        // Past the end, re-decode the last block into that unread slot.
        const int i = kk - kDecFirst;
        if (i >= 0 && i < kDecIters && dec_active(i)) {
          dec_store(nx2, i);
          dec_fetch(i, min(s + 3, nkb - 1));
        }
        if (i == kDecIters - 1) cp_async_commit();
      }
    }
    const int t = cur;
    cur = nx1;
    nx1 = nx2;
    nx2 = t;
  }
  cp_async_wait<0>();
  __syncthreads();
  finish<C>(acc, prob, smem, work);
  }
}

// Split-K without clusters (below SM90): the second launch over the split
// tiles. Sums each tile's rank partials (finish() stored them) in rank
// order, the same order as the cluster reduction, and runs the epilogue.
// gridDim.z must be 1, while `splits` is the split launch's gridDim.z.
// Shared memory holds the BM * kStagingStride BF16 epilogue tile.
// SM90+ images carry no code for this kernel.
template <class C, class Prob>
__global__ void __launch_bounds__(C::kThreads, 1)
    large_m_split_reduce_kernel(const __grid_constant__ typename Prob::Params params, int splits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __trap();
#else
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int RB = C::RB, TF = C::TF;
  Prob prob(params, nullptr);
  typename Prob::Work work;
  while (prob.template next<C>(work)) {
  const float4* src = prob.template partials<C>(work);
  float acc[RB][TF][4];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int tf = 0; tf < TF; ++tf) {
      const int i = (rb * TF + tf) * C::kThreads + threadIdx.x;
      float4 s = src[i];
      for (int r = 1; r < splits; ++r) {
        const float4 v = src[static_cast<size_t>(r) * kPartialsPerRank<C> + i];
        s.x += v.x;
        s.y += v.y;
        s.z += v.z;
        s.w += v.w;
      }
      acc[rb][tf][0] = s.x;
      acc[rb][tf][1] = s.y;
      acc[rb][tf][2] = s.z;
      acc[rb][tf][3] = s.w;
    }
  finish<C>(acc, prob, smem, work);
  __syncthreads();  // the staging tile is reused by the next tile
  }
#endif
}

}  // namespace nf12::detail::mma
