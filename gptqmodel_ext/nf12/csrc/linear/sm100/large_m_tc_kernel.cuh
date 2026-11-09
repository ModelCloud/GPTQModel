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
// tcgen05/UMMA NF12 GEMM for SM100/SM110, shared by
// linear/sm100/large_m_tc.cu and moe/sm100/large_tc.cu.
// Problem policies supply work lists, coordinates, x sources and output mapping.
//
// Persistent, one CTA per SM, optionally as CTA pairs (cta_group::2: UMMA
// M = 256 weight rows, each CTA holding 128 rows of A/D and half of the
// tokens of B), or non-persistent cluster split-K (see kClusterSplit). Per
// weight tile x BN-token tile (BN > 256: kSubs token sub-tiles of at most
// 256, one MMA each per K step on the same decoded A, see TileCfg):
//   warp 0       producer: one 3D TMA of the tile's compressed super-atoms
//                (8 blocks x 1 super-atom).
//   warp 2       x producer: one 2D TMA per sub-tile into SWIZZLE_128B.
//                Gathered MoE rows instead use cp.async from warps 2-3.
//   warp 3       second MMA issuer for dual-issue cluster split-K.
//   warp 1       MMA issuer (leader CTA only): tcgen05.mma kind::f16 with A
//                (weights) and B (tokens) from shared memory, FP32
//                accumulators in TMEM (two when they fit, so the epilogue of
//                a tile overlaps the next tile's mainloop).
//   warps 4-11   decoders: decode_bulk/decode_group yield exactly the mma.sync
//                A fragment, which stmatrix writes into a ring of decoded A
//                tiles in the same SWIZZLE_128B K-major layout.
//   warps 12-19  epilogue: tcgen05.ld -> (+ stream-K partials) -> + bias or
//                x row scale -> bf16 -> per-warp smem transpose -> TMA tile
//                stores (contiguous y rows) or 16-byte row stores
//                (scattered); cluster split-K: tcgen05.ld -> FP32 partial in
//                the workspace, reduced across the cluster by every warp
//                after the role loops.
// The bf16t source replaces decoding with plain loads. Both sources feed the
// same operands to the same MMAs.
//
// Problem policy (static members; see DenseProblem / MoeProblem):
//   kBN, kCtas                 token tile and CTA group
//   kStreamK                   dense stream-K fixup in the epilogue
//   kClusterSplit              one kc-CTA cluster per tile with kCtas = 1.
//                              CTA r takes K range r. Workspace partials
//                              are reduced through L2 in rank order.
//                              Each CTA receives one work item.
//   kBias                      epilogue adds bias[n] (Params::bias may be null)
//   kRowMaps                   SPEC.md 9.2 row maps: x rows may be gathered
//                              (by cp.async, else 2D TMA tiles of rows m0..),
//                              y rows scattered and scaled per token
//   kLockstepA                 matched TMEM-A/x rings with one stage-free commit
//   Params                     kernel parameters (fields used below: n, kb,
//                              y, ldy, ymap (CTA pairs: y tensor map, box
//                              32 x 32);
//                              dense: bias, the stream-K fields, kc)
//   Sched(p, unit, units)      next(Work&): the unit's deterministic work list
//   tile(p, w, rank) -> TcTile per-tile coordinates and decode parameters
//   y_row(p, t, c)             (dense) output row pointer of token c of tile t
//   x_gathered(p)              (kRowMaps) whether x rows are gathered
//   x_row(p, t, c), x_ptr(p, row)  (kRowMaps) x row of token c, row pointer
//   out_row(p, t, c), row_scale(p, row)  (kRowMaps) y row of token c, its
//                              FP32 multiplier (1 without row_scale)
//   y_direct(p)                (kRowMaps) y rows are the tiles' token rows
//                              (no scatter): CTA pairs store them by TMA
//                              through Params::ymap, as for dense problems
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <type_traits>

#include "internal.h"
#include "linear/sm100/large_m_tc_ptx.cuh"
#include "nf12/format.cuh"

namespace nf12::detail::tcg {

inline constexpr int kTileRows = 128;  // weight rows per CTA = TMEM lanes
inline constexpr int kBlocksPerTile = kTileRows / kSuperAtomRows;
inline constexpr int kBK = kSuperAtomCols;  // k per pipeline stage: one super-atom column
// Control warps 0-3 keep epilogue warpgroups aligned so a warp's TMEM lane
// quarter is warp % 4. Eight decoders cover the 16-row blocks, followed by
// eight epilogue warps (two per quarter).
inline constexpr int kDecodeWarps = kBlocksPerTile;
inline constexpr int kEpilogueWarps = 8;
inline constexpr int kThreads = 32 * (4 + kDecodeWarps + kEpilogueWarps);
// Per epilogue warp: [32 tokens][32 rows] bf16, rows padded 64 -> 80 bytes so
// the 8 row addresses of each stmatrix land in distinct banks.
inline constexpr int kStagingRowBytes = 80;
inline constexpr int kStagingBytesPerWarp = 32 * kStagingRowBytes;
// TMA-stored y is staged in boxes instead: [32 tokens][32 rows] bf16,
// SWIZZLE_64B, at 1024-byte boundaries inside the same region.
inline constexpr int kYBoxBytes = 32 * 64;
static_assert(kYBoxBytes <= kStagingBytesPerWarp, "y box staging");
inline constexpr int kSmemOptinBytes = 232448;  // SM100/SM110 per-block opt-in limit
inline constexpr int kMaxDevices = 64;          // per-device host caches

// Where decoded A tiles live and how loads are staged:
//  kSmem: A in shared memory (SWIZZLE_128B), two accumulators up to
//    BN = 256 (BN = 512: one, all of TMEM); separate weight / x / A rings,
//    each freed as soon as its consumer is done.
//  kTmem: A in TMEM, reducing shared-memory traffic but leaving one
//    accumulator for BN = 256.
//  kTmemCoupled: weights and x share a stage freed by the MMA. A forwarding
//    thread relays the peer decoders' completion.
enum class APath { kSmem, kTmem, kTmemCoupled };

// kExtraSmem holds problem tables. kTmemXStages limits TMEM-A x stages.
// kSplitStages > 0 selects cluster split-K rings. kLockstep merges frees
// for equal-depth TMEM-A and x rings.
template <int BN, int kCtas, WeightSource kSrc, APath kPath, int kExtraSmem = 0, int kTmemXStages = 4,
          int kSplitStages = 0, bool kLockstep = false>
struct TileCfg {
  static_assert(BN % 32 == 0 && BN >= 32 && BN <= 512, "token tile");
  static_assert(kCtas == 1 || kCtas == 2, "CTA group");
  static constexpr bool kTmemA = kPath != APath::kSmem;
  static constexpr bool kCoupled = kPath == APath::kTmemCoupled;
  static constexpr int kSuperAtom = kSrc == WeightSource::kNF12 ? kSuperAtomBytes : kSuperAtomBytesBf16t;
  static constexpr int kWBytes = kSuperAtom * kBlocksPerTile;
  // UMMA N is at most 256: a wider token tile runs kSubs MMAs of kSubN
  // tokens per K step on the same A (sub-tile j = accumulator columns and
  // tokens [j kSubN, (j + 1) kSubN)), so one decoded A tile serves BN tokens.
  static constexpr int kSubs = (BN + 255) / 256;
  static constexpr int kSubN = BN / kSubs;
  static_assert(BN % kSubs == 0 && kSubN % (16 * kCtas) == 0 && kSubN / kCtas % 8 == 0, "token sub-tile");
  static constexpr int kXRows = BN / kCtas;         // token rows of B held by each CTA
  static constexpr int kXSubRows = kSubN / kCtas;   // of which each sub-tile's (one x TMA box)
  static constexpr int kXBytes = kXRows * kBK * 2;
  static constexpr int kXSubBytes = kXSubRows * kBK * 2;
  // Decoded A tile in shared memory: 128 rows x 64 k bf16, SWIZZLE_128B K-major.
  static constexpr int kABytes = kTmemA ? 0 : kTileRows * kBK * 2;
  // Two BN-column accumulators when they fit next to the decoded A slots:
  // the epilogue can drain one tile while the next mainloop runs.
  static constexpr int kAccBufs = 2 * BN + (kTmemA ? 2 * 32 : 0) <= 512 ? 2 : 1;
  // Cluster split-K ends through workspace, without epilogue staging.
  // Equal-depth x and A rings let one commit free both buffers, reducing
  // issuer overhead at small token counts. The weight ring can run deeper.
  static constexpr bool kSplitRings = kSplitStages > 0;
  // mbarriers + the TMEM address live in static shared memory, everything
  // else is dynamic.
  static constexpr int kStaticReserve = kSplitRings ? 1024 : 512;
  static constexpr int kFixedSmem =
      1024 /*align slack*/ + (kSplitRings ? 0 : kEpilogueWarps * kStagingBytesPerWarp) + kExtraSmem;
  static constexpr int kAvail = kSmemOptinBytes - kStaticReserve - kFixedSmem;
  // Separate rings release each buffer after its own consumer. Prefer
  // (A slots, W stages) = (4, 6), then (4, 4), leaving >= 4 x stages,
  // otherwise fall back to (3, 3).
  static constexpr int xstages(int a, int w) { return (kAvail - a * kABytes - w * kWBytes) / kXBytes; }
  static constexpr bool kDeep = xstages(4, 6) >= 4;
  static constexpr bool kMid = !kDeep && xstages(4, 4) >= 4;
  // Even: the split-K decoders and MMA issuers take alternating stages, so
  // with an even depth every slot only ever holds one group's stages and each
  // waiter sees every phase of the barriers it waits on (with an odd depth a
  // group skips a slot's phase and its parity wait passes a phase early).
  static constexpr int kSplitDepth =
      std::min({kSplitStages, kAvail / (kWBytes + kXBytes + kABytes), kTmemA ? (512 - kAccBufs * BN) / 32 : 64}) &
      ~1;
  // The weight ring takes the rest (its stages are in flight from DRAM).
  static constexpr int kSplitWStages = std::min(16, (kAvail - kSplitDepth * (kXBytes + kABytes)) / kWBytes) & ~1;
  // Coupled: one stage = weights + x, as many as fit (up to 7).
  static constexpr int kCoupledStages = std::min(7, kAvail / (kWBytes + kXBytes));
  static constexpr int kBaseWStages =
      kCoupled ? kCoupledStages : kSplitRings ? kSplitWStages : kDeep ? 6 : kMid ? 4 : 3;
  static constexpr int kFreeASlots = kSplitRings ? kSplitDepth
                                     : kTmemA    ? std::min(kCoupled ? kCoupledStages : 8, (512 - kAccBufs * BN) / 32)
                                                 : kDeep || kMid ? 4 : 3;
  // Dense TMEM-A limits x lookahead independently of spare shared memory.
  // Gathered MoE rows can use a deeper x ring.
  static constexpr int kFreeXStages =
      kCoupled      ? kBaseWStages
      : kSplitRings ? kSplitDepth
                    : std::min(kTmemA ? kTmemXStages : 8, xstages(kTmemA ? 0 : kFreeASlots, kBaseWStages));
  // Lockstep TMEM-A/x rings share one free commit. The weight ring uses
  // remaining shared memory for up to 16 stages of DRAM lookahead.
  static constexpr bool kLockstepA = kLockstep && kTmemA && !kCoupled && !kSplitRings;
  static constexpr int kASlots = kLockstepA ? std::min(kFreeASlots, 4) : kFreeASlots;
  static constexpr int kXStages = kLockstepA ? kASlots : kFreeXStages;
  static constexpr int kWStages =
      kLockstepA ? std::min(16, (kAvail - kXStages * kXBytes) / kWBytes) : kBaseWStages;
  static constexpr bool kMergedFree = kSplitRings || kLockstepA;
  static_assert(kXStages >= 3 && kWStages >= 2 && kASlots >= 2, "pipeline too shallow");
  static_assert(!kMergedFree || kXStages == kASlots, "merged frees need lockstep x and A rings");
  // W full/empty per load stage, x full/empty per x stage (separate rings
  // only), A full/free per slot, 2 x 2 accumulator barriers.
  static constexpr int kNumBarriers = 2 * kWStages + (kCoupled ? 0 : 2 * kXStages) + 2 * kASlots + 4;
  static_assert(kNumBarriers * 8 + 8 <= kStaticReserve, "barrier space");
  static constexpr int kRingBytes = kXStages * kXBytes + kASlots * kABytes + kWStages * kWBytes;
  static_assert(kRingBytes % 1024 == 0, "the epilogue staging starts on a 1024-byte boundary");
  static constexpr int kSmemBytes = kFixedSmem + kRingBytes;  // dynamic
};

// A unit's work item: k-blocks [kb0, kb1) of tile `tile`.
struct Work {
  int tile, kb0, kb1;
};

// Each role evaluates its tile independently. Unused fields compile away.
struct TcTile {
  int wblock;   // weight TMA coordinate of this CTA's first 16-row block
  int blocks;   // of this CTA's 8 blocks, how many lie inside the matrix
  int row0;     // first weight row (output column) of this CTA in the matrix
  int m0;       // first token: x/y row (dense) or sorted position (MoE)
  int tokens;   // live tokens (accumulator columns), 1..BN
  bool w_once;  // the weight tile is read by no other concurrent tile
  uint32_t base_x4;     // NF12 decode: exponent base x 0x01010101
  const uint4* escapes;  // NF12 decode: escape table of this matrix
  uint32_t num_escapes;  // bounds the escape-table prefetch
};

#if defined(NF12_TC_DEVICE)
// Escape prefetches use 128-byte lines ahead of each decoder warp.
inline constexpr uint32_t kEscEntriesPerLine = 128 / 16;
inline constexpr uint32_t kEscPrefetchLines = 16;

inline constexpr int kWProducerWarp = 0;
inline constexpr int kMmaWarp = 1;
inline constexpr int kXProducerWarp = 2;
inline constexpr int kMmaWarp2 = 3;  // second MMA issuer (kDualIssue)
inline constexpr int kTmemCols = 512;  // two BN <= 256 accumulators
inline constexpr int kDecodeWarp0 = 4;
inline constexpr int kEpilogueWarp0 = kDecodeWarp0 + kDecodeWarps;
static_assert(kEpilogueWarp0 % 4 == 0, "epilogue warps must start a warpgroup");

struct Ring {
  int stage = 0;
  uint32_t phase = 0;
  template <int kStages>
  __device__ void advance() {
    if (++stage == kStages) {
      stage = 0;
      phase ^= 1;
    }
  }
};

// Stream-K publication tag of the partial for (tile, first k-block): the
// launcher zeroes the tag region (cudaMemsetAsync on the launch stream), a
// contributor publishes this nonzero value after its partial and the owner
// waits for exactly it. Correctness does not depend on prior workspace
// contents.
__device__ __forceinline__ uint64_t stream_k_tag(int tile, int kb0) {
  return ((static_cast<uint64_t>(static_cast<uint32_t>(tile)) + 1) << 32) | static_cast<uint32_t>(kb0);
}

// MMA N is 16-granular, or 32-granular for CTA pairs to keep each half of B
// 16-row aligned. Return 0 for a sub-tile with no live tokens.
template <int kSubN, int kCtas>
__device__ __forceinline__ int mma_n(int tokens, int j) {
  constexpr int g = 16 * kCtas;
  const int live = tokens - j * kSubN;
  return live <= 0 ? 0 : min(kSubN, (live + g - 1) / g * g);
}

// Sub-tiles holding live tokens: x bytes per stage are counted for those only.
template <int kSubN>
__device__ __forceinline__ int live_subs(int tokens) {
  return (tokens + kSubN - 1) / kSubN;
}

__device__ __forceinline__ void set_u4(uint32_t* r, const uint4& v) {
  r[0] = v.x;
  r[1] = v.y;
  r[2] = v.z;
  r[3] = v.w;
}

// Traverse bands of `band` tiles along one dimension, with the other
// dimension varying slowly, so concurrent units walk the same band.
__device__ __forceinline__ void band_coords(int r, int band, int along, int across, int& a, int& c) {
  const int per_band = band * across;
  const int b = r / per_band, rr = r % per_band;
  const int cur = min(band, along - b * band);
  a = b * band + rr % cur;
  c = rr / cur;
}
#endif  // NF12_TC_DEVICE

// Problems with kRowMaps keep a per-tile table in shared memory: each token's
// output row and scale, filled by the epilogue before it waits for the
// accumulator, so their loads overlap the mainloop.
template <class Prob>
constexpr int table_bytes() {
  return Prob::kRowMaps ? 8 * Prob::kBN : 0;
}
// Ring depth of the cluster split-K layout: Prob::kSplitStages where the
// policy sets it, else 0 (the default layout).
template <class Prob, class = void>
struct SplitStages {
  static constexpr int value = 0;
};
template <class Prob>
struct SplitStages<Prob, std::void_t<decltype(Prob::kSplitStages)>> {
  static constexpr int value = Prob::kSplitStages;
};
template <class Prob, WeightSource kSrc, APath kPath>
using ProbCfg = TileCfg<Prob::kBN, Prob::kCtas, kSrc, kPath, table_bytes<Prob>(), Prob::kRowMaps ? 8 : 4,
                        SplitStages<Prob>::value, Prob::kLockstepA>;

// Two warps gather x with 16 bytes per thread and row. Keep row pointers in
// registers for the tile to avoid shared-table loads inside the copy loop.
inline constexpr int kXGatherWarps = 2;

template <class Prob, WeightSource kSrc, APath kPath>
__global__ void __launch_bounds__(kThreads, 1)
    tc_gemm_kernel(const __grid_constant__ CUtensorMap xmap, const __grid_constant__ CUtensorMap wmap,
                   const __grid_constant__ typename Prob::Params p) {
  // Row-mapped (MoE) problems have neither split-K partials nor a bias.
  static_assert(!Prob::kRowMaps || (!Prob::kStreamK && !Prob::kClusterSplit && !Prob::kBias),
                "unsupported policy combination");
  static_assert(!Prob::kClusterSplit || (!Prob::kStreamK && Prob::kCtas == 1), "cluster split-K: 1-CTA units");
  static_assert(Prob::kCtas == 1 || Prob::kCtas == 2, "CTA group");
  static_assert(Prob::kBN % (16 * Prob::kCtas) == 0, "token tile");
#if defined(NF12_TC_DEVICE)
  using namespace tc;
  constexpr int BN = Prob::kBN;
  constexpr int kCtas = Prob::kCtas;
  using Cfg = ProbCfg<Prob, kSrc, kPath>;
  constexpr int SW = Cfg::kWStages;
  constexpr int SX = Cfg::kXStages;
  constexpr int R = Cfg::kASlots;
  constexpr bool kPair = kCtas == 2;
  constexpr bool kCoupled = Cfg::kCoupled;
  constexpr bool kTmemA = Cfg::kTmemA;
  // Cluster split-K epilogue warps decode alternate stages during the
  // mainloop, then drain the accumulator for their one work item.
  constexpr bool kSplitDecode = Prob::kClusterSplit;
  // Cluster split-K with room for two accumulators: warp 3 takes the odd
  // stages, and the epilogue adds the two fixed-order partial sums.
  constexpr bool kDualIssue = Prob::kClusterSplit && Cfg::kAccBufs == 2;
  static_assert(!(kCoupled && Prob::kRowMaps), "gathered x needs the separate x ring");

  extern __shared__ uint8_t smem_raw[];
  // SWIZZLE_128B tiles must sit on 1024-byte boundaries.
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* x_smem = smem;                         // SX x [kXRows][128 B] swizzled
  uint8_t* a_smem = x_smem + SX * Cfg::kXBytes;   // R x [128 rows][128 B] swizzled, decoded weights
  uint8_t* w_smem = a_smem + R * Cfg::kABytes;    // SW x [8 blocks][super-atom]
  uint8_t* staging = w_smem + SW * Cfg::kWBytes;  // per epilogue warp [32 tokens][32 rows] bf16
  // kRowMaps table: out rows [BN], then scales [BN].
  int32_t* tok_rows = reinterpret_cast<int32_t*>(staging + kEpilogueWarps * kStagingBytesPerWarp);
  float* tok_scales = reinterpret_cast<float*>(tok_rows + BN);
  // Image support probes require static shared memory, absent when the
  // tcgen05 body is compiled out.
  __shared__ alignas(8) uint64_t bars[Cfg::kNumBarriers + 1];
  // Separate w_full/w_empty track weight TMA and decoder completion.
  // x_full tracks both CTAs' TMA bytes on the leader, while x_empty tracks
  // MMA completion. Coupled rings merge x into w_full and free through MMA,
  // without x barriers.
  uint64_t* w_full = bars;
  uint64_t* w_empty = w_full + SW;
  uint64_t* x_full = w_empty + SW;
  uint64_t* x_empty = x_full + (kCoupled ? 0 : SX);
  uint64_t* a_full = x_empty + (kCoupled ? 0 : SX);  // decoders -> leader MMA (leader's barrier)
  uint64_t* a_free = a_full + R;                      // MMA commit -> decoders (both CTAs)
  uint64_t* acc_full = a_free + R;                    // MMA commit -> epilogue (both CTAs)
  uint64_t* acc_empty = acc_full + 2;                 // epilogue (both CTAs) -> leader MMA
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(acc_empty + 2);
  // kMergedFree: the x stage and the A slot of a stage are freed by one commit.
  uint64_t* x_free = Cfg::kMergedFree ? a_free : x_empty;

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int rank = kPair ? static_cast<int>(cluster_rank()) : 0;
  bool gathered = false;  // x rows gathered by the cp.async producers
  if constexpr (Prob::kRowMaps) gathered = Prob::x_gathered(p);

  // Cluster split-K uses PDL (linear/dispatch.cu contract). Global accesses
  // wait below, but L2 prefetch may run earlier because it transfers no
  // values into this CTA.
  if constexpr (Prob::kClusterSplit) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (threadIdx.x == 32 * kMmaWarp2) {
      typename Prob::Sched s(p, static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x));
      Work w0;
      s.next(w0);
      const TcTile t = Prob::tile(p, w0, 0);
      for (int kb = w0.kb0; kb < min(w0.kb1, w0.kb0 + SW); ++kb) tma_prefetch_l2_3d(&wmap, 0, kb, t.wblock);
    }
  }
  if (threadIdx.x == 0) {
    for (int s = 0; s < SW; ++s) {
      mbar_init(smem_u32(&w_full[s]), 1);
      mbar_init(smem_u32(&w_empty[s]), kCoupled ? 1 : kDecodeWarps);
    }
    if constexpr (!kCoupled) {
      // Gathered x counts one arrival per copy thread plus the peer's
      // forwarded arrival on the leader. TMA uses bytes and one arming arrival.
      const int x_arrivals = gathered ? 32 * kXGatherWarps + (kPair && rank == 0 ? 1 : 0) : 1;
      for (int s = 0; s < SX; ++s) {
        mbar_init(smem_u32(&x_full[s]), x_arrivals);
        mbar_init(smem_u32(&x_empty[s]), 1);
      }
    }
    // Coupled pairs forward one peer arrival. Separate rings receive every
    // decoder's arrival directly.
    const int a_arrivals = kCoupled ? kDecodeWarps + (kPair && rank == 0 ? 1 : 0) : kDecodeWarps * kCtas;
    for (int r = 0; r < R; ++r) {
      mbar_init(smem_u32(&a_full[r]), a_arrivals);
      mbar_init(smem_u32(&a_free[r]), 1);
    }
    for (int a = 0; a < 2; ++a) {
      mbar_init(smem_u32(&acc_full[a]), kDualIssue ? 2 : 1);
      mbar_init(smem_u32(&acc_empty[a]), kEpilogueWarps * kCtas);
    }
    mbar_fence_init();
    prefetch_tmap(&xmap);
    prefetch_tmap(&wmap);
  }
  if (warp == kMmaWarp) tmem_alloc<kTmemCols, kCtas>(smem_u32(tmem_slot));
  tc_fence_before();
  __syncthreads();
  if constexpr (kPair) cluster_sync();  // peer barriers initialized before any remote arrive
  tc_fence_after();
  const uint32_t tmem = *tmem_slot;
  if constexpr (Prob::kClusterSplit) asm volatile("griddepcontrol.wait;" ::: "memory");
  // Accumulator a occupies TMEM columns [a * BN, (a + 1) * BN); with TMEM A
  // the decoded A slots follow, 32 columns each.
  const uint32_t a_tmem = tmem + Cfg::kAccBufs * BN;

  // Barriers waited on by the leader's MMA thread live in the leader CTA.
  auto arrive_leader = [&](uint64_t* bar) {
    if constexpr (kPair)
      mbar_arrive_cluster<kCoupled>(smem_u32(bar), 0);
    else
      mbar_arrive(smem_u32(bar));
  };

  // Every role of every CTA in a unit walks the same deterministic work list.
  typename Prob::Sched sched(p, static_cast<int>(blockIdx.x) / kCtas, static_cast<int>(gridDim.x) / kCtas);
  Work w;

  if (warp == kWProducerWarp) {
    if (lane == 0) {
      const uint64_t once = policy_evict_first(), reused = policy_evict_last();
      Ring ring;
      while (sched.next(w)) {
        const TcTile t = Prob::tile(p, w, rank);
        const uint64_t policy = t.w_once ? once : reused;
        // Each CTA of a pair feeds half of each sub-tile MMA's N rows.
        const int subs = live_subs<Cfg::kSubN>(t.tokens);
        for (int kb = w.kb0; kb < w.kb1; ++kb) {
          mbar_wait(smem_u32(&w_empty[ring.stage]), ring.phase ^ 1);
          // Invalid rows are zero-filled or belong to the next expert.
          // They still count toward the fixed transfer size.
          const uint32_t bar = smem_u32(&w_full[ring.stage]);
          mbar_arrive_expect_tx(bar, Cfg::kWBytes + (kCoupled ? subs * Cfg::kXSubBytes : 0));
          tma_load_3d(smem_u32(w_smem + ring.stage * Cfg::kWBytes), &wmap, 0, kb, t.wblock, bar, policy);
          if constexpr (kCoupled) {
#pragma unroll
            for (int j = 0; j < Cfg::kSubs; ++j)
              if (j == 0 || j < subs)
                tma_load_2d<1>(smem_u32(x_smem + ring.stage * Cfg::kXBytes + j * Cfg::kXSubBytes), &xmap, kb * kBK,
                               t.m0 + j * Cfg::kSubN + rank * (mma_n<Cfg::kSubN, kCtas>(t.tokens, j) / kCtas), bar,
                               policy_evict_last());
          }
          ring.advance<SW>();
        }
      }
    }
  } else if (warp == kXProducerWarp ||
             (gathered && warp > kXProducerWarp && warp < kXProducerWarp + kXGatherWarps)) {
    if (gathered) {
      if constexpr (Prob::kRowMaps) {
        // Thread xt copies 16-byte chunk xt % 8 of rows xt / 8 + 8j of this
        // CTA's halves of the token sub-tiles straight into the SWIZZLE_128B
        // layout (chunk c of row r at (c ^ r % 8) * 16, and r % 8 = xt / 8).
        // Only live tokens' rows are fetched: an accumulator column depends on
        // its own B row alone, and columns past the tile's tokens are never
        // stored, so whatever the stage holds there is harmless.
        constexpr int kRowsPerThread = Cfg::kXRows / 8;
        const int xt = static_cast<int>(threadIdx.x) - 32 * kXProducerWarp;
        const int chunk = xt % 8, rgrp = xt / 8;
        const uint32_t lane_off = rgrp * 128 + ((chunk ^ rgrp) << 4);
        // Every token's row is gathered by topk experts' tiles spread over the
        // whole launch: keep x in L2 against the streamed weights.
        const uint64_t policy = policy_evict_last();
        Ring ring;
        while (sched.next(w)) {
          const TcTile t = Prob::tile(p, w, rank);
          const __nv_bfloat16* src[kRowsPerThread];
#pragma unroll
          for (int j = 0; j < kRowsPerThread; ++j) {
            // Row r of sub-tile s (kXSubRows is a multiple of 8: s is static).
            constexpr int kSubRows = Cfg::kXSubRows;
            const int s = 8 * j / kSubRows, r = rgrp + 8 * j - s * kSubRows;
            const int half = mma_n<Cfg::kSubN, kCtas>(t.tokens, s) / kCtas;  // B rows of this CTA
            const int c = s * Cfg::kSubN + rank * half + r;
            src[j] = r < half && c < t.tokens ? Prob::x_ptr(p, Prob::x_row(p, t, c)) + chunk * 8 : nullptr;
          }
          for (int kb = w.kb0; kb < w.kb1; ++kb) {
            mbar_wait(smem_u32(&x_free[ring.stage]), ring.phase ^ 1);
            const uint32_t dst = smem_u32(x_smem + ring.stage * Cfg::kXBytes) + lane_off;
#pragma unroll
            for (int j = 0; j < kRowsPerThread; ++j)
              if (src[j] != nullptr) cp_async_16(dst + j * 1024, src[j] + kb * kBK, policy);
            // Completion on this barrier makes cp.async data visible to MMA
            // without a separate proxy fence.
            cp_async_mbar_arrive(smem_u32(&x_full[ring.stage]));
            ring.advance<SX>();
          }
        }
      }
    } else if (!kCoupled && lane == 0) {
      // Both CTAs' x halves complete on the leader's barrier, which the
      // leader arms with the bytes of both.
      const uint64_t policy = policy_evict_last();
      const uint32_t full0 = kPair ? map_to_rank(smem_u32(x_full), 0) : smem_u32(x_full);
      Ring ring;
      while (sched.next(w)) {
        const TcTile t = Prob::tile(p, w, rank);
        int x_row[Cfg::kSubs];
#pragma unroll
        for (int j = 0; j < Cfg::kSubs; ++j)
          x_row[j] = t.m0 + j * Cfg::kSubN + rank * (mma_n<Cfg::kSubN, kCtas>(t.tokens, j) / kCtas);
        const int subs = live_subs<Cfg::kSubN>(t.tokens);
        for (int kb = w.kb0; kb < w.kb1; ++kb) {
          mbar_wait(smem_u32(&x_free[ring.stage]), ring.phase ^ 1);
          const uint32_t bar = full0 + ring.stage * 8;
          if (rank == 0) mbar_arrive_expect_tx(smem_u32(&x_full[ring.stage]), subs * Cfg::kXSubBytes * kCtas);
#pragma unroll
          for (int j = 0; j < Cfg::kSubs; ++j)
            if (j == 0 || j < subs)
              tma_load_2d<kCtas>(smem_u32(x_smem + ring.stage * Cfg::kXBytes + j * Cfg::kXSubBytes), &xmap,
                                 kb * kBK, x_row[j], bar, policy);
          ring.advance<SX>();
        }
      }
    }
  } else if (warp == kMmaWarp || (kDualIssue && warp == kMmaWarp2)) {
    if (lane == 0 && rank == 0) {
      // Cluster split-K has one tile: issuer ip takes its alternating
      // stages into accumulator ip.
      const int ip = kDualIssue && warp == kMmaWarp2 ? 1 : 0;
      Ring xr, slot;
      for (int iter = 0; sched.next(w); ++iter) {
        const TcTile t = Prob::tile(p, w, rank);
        const int acc = iter % Cfg::kAccBufs;
        mbar_wait<kPair>(smem_u32(&acc_empty[acc]), ((iter / Cfg::kAccBufs) & 1) ^ 1);
        tc_fence_after();
        const uint32_t d = tmem + (kDualIssue ? ip : acc) * BN;
        // Only the live token columns are computed: the tail tile shrinks N
        // and skips sub-tiles without live tokens (idesc 0).
        uint32_t idesc[Cfg::kSubs];
#pragma unroll
        for (int j = 0; j < Cfg::kSubs; ++j) {
          const int n = mma_n<Cfg::kSubN, kCtas>(t.tokens, j);
          idesc[j] = n > 0 ? idesc_bf16(kTileRows * kCtas, n) : 0u;
        }
        for (int kb = w.kb0; kb < w.kb1; ++kb) {
          if (kDualIssue && (kb - w.kb0) % 2 != ip) {
            xr.advance<SX>();
            slot.advance<R>();
            continue;
          }
          const uint32_t acc0 = kb > w.kb0 + ip ? 1u : 0u;
          // Coupled: the decoders waited for the whole stage, x included.
          mbar_wait<kPair>(smem_u32(&a_full[slot.stage]), slot.phase);
          if constexpr (!kCoupled) mbar_wait<kPair>(smem_u32(&x_full[xr.stage]), xr.phase);
          tc_fence_after();
          // Each K=16 step advances the K-major smem operands by 32 bytes and
          // a TMEM A slot by 8 columns.
#pragma unroll
          for (int j = 0; j < Cfg::kSubs; ++j) {
            if (j > 0 && idesc[j] == 0u) continue;
            const uint32_t dj = d + j * Cfg::kSubN;
            const uint64_t b =
                smem_desc_sw128(smem_u32(x_smem + xr.stage * Cfg::kXBytes + j * Cfg::kXSubBytes));
            if constexpr (kTmemA) {
              const uint32_t a = a_tmem + slot.stage * 32;
#pragma unroll
              for (int kk = 0; kk < kBK / 16; ++kk)
                tc_mma_ts<kCtas>(dj, a + kk * 8, b + kk * 2, idesc[j], kk > 0 ? 1u : acc0);
            } else {
              const uint64_t a = smem_desc_sw128(smem_u32(a_smem + slot.stage * Cfg::kABytes));
#pragma unroll
              for (int kk = 0; kk < kBK / 16; ++kk)
                tc_mma_ss<kCtas>(dj, a + kk * 2, b + kk * 2, idesc[j], kk > 0 ? 1u : acc0);
            }
          }
          if constexpr (!Cfg::kMergedFree)
            tc_commit<kCtas>(smem_u32(kCoupled ? &w_empty[xr.stage] : &x_empty[xr.stage]));
          tc_commit<kCtas>(smem_u32(&a_free[slot.stage]));
          xr.advance<SX>();
          slot.advance<R>();
        }
        tc_commit<kCtas>(smem_u32(&acc_full[acc]));
      }
    } else if (kCoupled && kPair && lane == 0) {
      // Coupled pairs: forward "this slot's A rows (and its stage's x half)
      // are ready" from the peer's decoders to the leader, once per stage.
      Ring slot;
      while (sched.next(w)) {
        for (int kb = w.kb0; kb < w.kb1; ++kb) {
          mbar_wait(smem_u32(&a_full[slot.stage]), slot.phase);
          mbar_arrive_cluster<true>(smem_u32(&a_full[slot.stage]), 0);
          slot.advance<R>();
        }
      }
    } else if (gathered && kPair && lane == 0) {
      // cp.async arrives only locally, so forward peer x completion to the leader.
      Ring xr;
      while (sched.next(w)) {
        for (int kb = w.kb0; kb < w.kb1; ++kb) {
          mbar_wait(smem_u32(&x_full[xr.stage]), xr.phase);
          mbar_arrive_cluster<true>(smem_u32(&x_full[xr.stage]), 0);
          xr.advance<SX>();
        }
      }
    }
  } else if (warp >= kDecodeWarp0 && (kSplitDecode || warp < kEpilogueWarp0)) {
    // One decoder warp per 16-row block of the tile. Its decoded registers
    // are the mma.sync A fragments of the block's four 16x16 atoms.
    //  - TMEM A: tcgen05.st.16x128b stores exactly that fragment layout; a
    //    warp can only reach its TMEM lane quarter (warp % 4 = rows 32q..),
    //    so warp (q, half) owns block 2q + half.
    //  - smem A: atom j's {R0, R1, R2, R3} are the 8x8 matrices (rows 0-7 |
    //    8-15) x (k chunk 2j | 2j+1), which stmatrix writes as 16-byte rows
    //    into the SWIZZLE_128B layout (chunk c of row r at (c ^ r % 8) * 16).
    // kSplitDecode: the epilogue warps decode too, the two groups of 8 taking
    // alternating stages, before they run the epilogue.
    const int q = warp % 4, half = (warp - kDecodeWarp0) / 4 % 2;
    const int parity = kSplitDecode ? (warp - kDecodeWarp0) / 8 : 0;
    const int blk = 2 * q + half;
    const int mi = lane / 8, rr = lane % 8;  // stmatrix: matrix mi, row rr
    const uint32_t row_off = blk * 2048 + (mi & 1) * 1024 + rr * 128;
    const uint32_t lane_off = static_cast<uint32_t>(16 * blk) << 16;  // TMEM lanes of the block
    Ring wr, slot;
    while (sched.next(w)) {
      const TcTile t = Prob::tile(p, w, rank);
      // Rows past n (partial last tile) keep stale A rows; their D rows are never stored.
      const bool live = blk < t.blocks;
      for (int kb = w.kb0; kb < w.kb1; ++kb) {
        if (kSplitDecode && (kb - w.kb0) % 2 != parity) {
          wr.advance<SW>();
          slot.advance<R>();
          continue;
        }
        mbar_wait(smem_u32(&w_full[wr.stage]), wr.phase);
        mbar_wait(smem_u32(&a_free[slot.stage]), slot.phase ^ 1);
        uint32_t r[16];
        if (live) {
          const uint32_t sa = smem_u32(w_smem + wr.stage * Cfg::kWBytes + blk * Cfg::kSuperAtom) + 16 * lane;
          if constexpr (kSrc == WeightSource::kNF12) {
            const uint4 p0 = lds128(sa);
            const uint4 p1 = lds128(sa + kPlaneBytes);
            const uint4 p2 = lds128(sa + 2 * kPlaneBytes);
            const Group g[4] = {{p0.x, p1.x, p1.y}, {p0.y, p1.z, p1.w}, {p0.z, p2.x, p2.y}, {p0.w, p2.z, p2.w}};
#pragma unroll
            for (int j = 0; j < 4; ++j) decode_bulk(g[j], t.base_x4, *reinterpret_cast<uint32_t(*)[4]>(&r[4 * j]));
            // One warp-uniform test for all four groups keeps the common
            // path branch-free.
            const bool esc = (p0.x == 0u) | (p0.y == 0u) | (p0.z == 0u) | (p0.w == 0u);
            if (__builtin_expect(__any_sync(0xffffffffu, esc), 0)) {
              uint4 v[4];
              uint32_t next = 0;
#pragma unroll
              for (int j = 0; j < 4; ++j)
                if (g[j].meta == 0u) v[j] = __ldg(t.escapes + g[j].lo0);
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                if (g[j].meta == 0u) {
                  set_u4(&r[4 * j], v[j]);
                  next = max(next, g[j].lo0 + 1);
                }
              }
              // Flat group order makes the next super-atom's escapes follow
              // this one's last entry. Prefetch ahead for later decode stages.
              const uint32_t ahead = __reduce_max_sync(0xffffffffu, next) + kEscEntriesPerLine * lane;
              if (lane < kEscPrefetchLines && ahead < t.num_escapes) prefetch_l1(t.escapes + ahead);
            }
          } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const uint4 v = lds128(sa + j * kPlaneBytes);
              r[4 * j + 0] = v.x;
              r[4 * j + 1] = v.y;
              r[4 * j + 2] = v.z;
              r[4 * j + 3] = v.w;
            }
          }
        }
        if (live) {
          if constexpr (kTmemA) {
            tmem_st_16x128b_x8(a_tmem + lane_off + slot.stage * 32, r);
            tc_wait_st();
          } else {
            const uint32_t a_row = smem_u32(a_smem + slot.stage * Cfg::kABytes) + row_off;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const uint32_t chunk = 2 * j + (mi >> 1);
              stmatrix_x4(a_row + ((chunk ^ rr) << 4), r[4 * j], r[4 * j + 1], r[4 * j + 2], r[4 * j + 3]);
            }
            fence_proxy_async_smem();
          }
        }
        if constexpr (kTmemA) tc_fence_before();
        // The A store consumes loaded registers, completing each lane's
        // shared loads before slot release. Earlier release could race the
        // next TMA. Coupled stages also hold x, so MMA commit frees them and
        // peer decoders signal the forwarding thread.
        __syncwarp();
        if (lane == 0) {
          if constexpr (kCoupled) {
            mbar_arrive(smem_u32(&a_full[slot.stage]));
          } else {
            mbar_arrive(smem_u32(&w_empty[wr.stage]));
            arrive_leader(&a_full[slot.stage]);
          }
        }
        wr.advance<SW>();
        slot.advance<R>();
      }
    }
  }
  if (warp >= kEpilogueWarp0) {
    // Two warps per TMEM lane quarter take alternating 32-token chunks.
    // tcgen05.ld yields MMA fragments, transposed by stmatrix.trans into
    // [token][row] BF16. CTA pairs use TMA for contiguous output in
    // [32 tokens][32 rows] SWIZZLE_64B boxes. Chunk c of token r sits at
    // (c ^ (r / 2) % 4) * 16, conflict-free for stmatrix.
    // Tensor-map bounds clip rows past n and m, but an MoE expert's partial
    // chunk must use 16-byte stores to avoid overwriting the next expert.
    // Single CTAs and scattered outputs use 16-byte stores throughout.
    bool tma_y = kPair;
    if constexpr (Prob::kRowMaps) tma_y = kPair && Prob::y_direct(p);
    const int q = warp % 4;
    const int ew = (warp - kEpilogueWarp0) / 4;
    const uint32_t stg = smem_u32(staging + (warp - kEpilogueWarp0) * (tma_y ? kYBoxBytes : kStagingBytesPerWarp));
    const int g = lane / 4;
    // A fresh walk of the work list (with kSplitDecode these warps already
    // walked it as decoders).
    typename Prob::Sched esched(p, static_cast<int>(blockIdx.x) / kCtas, static_cast<int>(gridDim.x) / kCtas);
    for (int iter = 0; esched.next(w); ++iter) {
      const TcTile t = Prob::tile(p, w, rank);
      const int acc = iter % Cfg::kAccBufs;
      if constexpr (Prob::kRowMaps) {
        // The tile's output rows and scales, while its mainloop still runs.
        constexpr int kThreadsE = 32 * kEpilogueWarps;
        named_bar_sync(3, kThreadsE);  // the previous tile's entries are no longer read
        for (int c = static_cast<int>(threadIdx.x) - 32 * kEpilogueWarp0; c < t.tokens; c += kThreadsE) {
          const int o = Prob::out_row(p, t, c);
          tok_rows[c] = o;
          tok_scales[c] = Prob::row_scale(p, o);
        }
        named_bar_sync(3, kThreadsE);
      }
      mbar_wait(smem_u32(&acc_full[acc]), (iter / Cfg::kAccBufs) & 1);
      tc_fence_after();

      const int tokens = t.tokens;  // accumulator column c = token t.m0 + c
      const int row0 = t.row0 + 32 * q;  // first weight row of this warp
      const uint32_t tacc = tmem + (static_cast<uint32_t>(32 * q) << 16) + acc * BN;
      if constexpr (Prob::kClusterSplit) {
        // Store FP32 partials as [token][128 rows]. Consecutive lanes write
        // consecutive rows. The cluster reduces through L2 after role loops.
        float* part = p.partials + static_cast<int64_t>(blockIdx.x) * BN * kTileRows + 32 * q + lane;
        // kDualIssue: the odd stages' accumulator follows (absent for a one-stage range).
        const bool odd = kDualIssue && w.kb1 - w.kb0 > 1;
        for (int c0 = 32 * ew; c0 < tokens; c0 += 64) {
          uint32_t v[32], v1[32];
          tmem_ld_32x32b_x32(tacc + c0, v);
          if (odd) tmem_ld_32x32b_x32(tacc + BN + c0, v1);
          tc_wait_ld();
#pragma unroll
          for (int j = 0; j < 32; ++j)
            part[(c0 + j) * kTileRows] = odd ? __uint_as_float(v[j]) + __uint_as_float(v1[j]) : __uint_as_float(v[j]);
        }
        continue;
      }
      bool contributor = false;  // publishes a stream-K partial
      int u_last = 0;            // last unit whose partial the owner adds
      if constexpr (Prob::kStreamK) {
        contributor = w.kb0 > 0;
        const bool owner_split = w.kb0 == 0 && w.kb1 < p.kb;  // finishes a split tile
        // Contributors ran their segment of this tile first, so this wait is
        // normally already satisfied.
        u_last = esched.unit();
        if (owner_split) {
          const int64_t t0 = static_cast<int64_t>(w.tile - p.dp_tiles) * p.kb;
          for (int u = esched.unit() + 1; u < esched.units() && esched.sk_begin(u) < t0 + p.kb; ++u) {
            if (!esched.contributes(u, w.tile)) continue;
            u_last = u;
            const uint64_t tag = stream_k_tag(w.tile, static_cast<int>(esched.sk_begin(u) - t0));
            while (ld_acquire64(p.tags + u * kCtas + rank) != tag) {
            }
          }
        }
      }

      // This thread's rows: row0 + 16h + 8e + g. Adding -0.0f for a missing
      // bias preserves the accumulator, including signed zero.
      float bias[2][2];
      if constexpr (Prob::kBias) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int n = row0 + 16 * h + 8 * e + g;
            bias[h][e] = !contributor && p.bias != nullptr && n < p.n ? __bfloat162float(p.bias[n]) : -0.0f;
          }
      }

      // The accumulator is released as soon as the warp's last TMEM load has
      // completed, before that chunk's stores.
      auto release_acc = [&] {
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive_leader(&acc_empty[acc]);
      };
      if (32 * ew >= tokens) release_acc();
      for (int c0 = 32 * ew; c0 < tokens; c0 += 64) {
        // Scales of this lane's accumulator columns (tokens c0 + 8i + 2(lane % 4) + j).
        float scale[4][2];
        if constexpr (Prob::kRowMaps) {
#pragma unroll
          for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              const int c = c0 + 8 * i + 2 * (lane % 4) + j;
              scale[i][j] = c < tokens ? tok_scales[c] : 1.0f;
            }
        }
        uint32_t v[2][16];
        tmem_ld_16x256b_x4(tacc + c0, v[0]);
        tmem_ld_16x256b_x4(tacc + (16u << 16) + c0, v[1]);
        tc_wait_ld(v[0], v[1]);
        if (c0 + 64 >= tokens) release_acc();
        float f[2][16];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int j = 0; j < 16; ++j) f[h][j] = __uint_as_float(v[h][j]);
        // Partials keep the register layout: [cta][chunk][q][h][lane][16].
        const int64_t poff = ((static_cast<int64_t>(c0 / 32) * 4 + q) * 2) * 512 + lane * 16;
        if constexpr (Prob::kStreamK) {
          if (contributor) {
            float* dst = p.partials + static_cast<int64_t>(blockIdx.x) * BN * kTileRows + poff;
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int i = 0; i < 4; ++i)
                *reinterpret_cast<float4*>(dst + h * 512 + 4 * i) =
                    make_float4(f[h][4 * i], f[h][4 * i + 1], f[h][4 * i + 2], f[h][4 * i + 3]);
            continue;
          }
        }
        if constexpr (Prob::kStreamK) {
          for (int u = esched.unit() + 1; u <= u_last; ++u) {
            if (!esched.contributes(u, w.tile)) continue;
            const float* src = p.partials + static_cast<int64_t>(u * kCtas + rank) * BN * kTileRows + poff;
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                const float4 s = __ldcg(reinterpret_cast<const float4*>(src + h * 512 + 4 * i));
                f[h][4 * i] += s.x;
                f[h][4 * i + 1] += s.y;
                f[h][4 * i + 2] += s.z;
                f[h][4 * i + 3] += s.w;
              }
          }
        }
        // Register 4i + r of half h: row 16h + g + 8(r >> 1), token 8i + 2(lane % 4) + (r & 1).
        uint32_t pk[2][4][2];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              float lo = f[h][4 * i + 2 * e], hi = f[h][4 * i + 2 * e + 1];
              if constexpr (Prob::kBias) {
                lo += bias[h][e];
                hi += bias[h][e];
              }
              if constexpr (Prob::kRowMaps) {
                lo *= scale[i][0];
                hi *= scale[i][1];
              }
              pk[h][i][e] = bf16x2_rn(lo, hi);
            }
        // Matrix (h, i, e) = rows 16h + 8e..+7 x tokens 8i..8i+7; lane L
        // addresses row L % 8 of matrix L / 8 of each x4 group.
        const int mi = lane / 8;
        if (tma_y) {
          // The previous chunk's TMA has read the staging.
          if (lane == 0) bulk_wait_read();
          __syncwarp();
          const int r = lane % 8;  // token row within each matrix; (8i + r) / 2 % 4 = r / 2
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int ip = 0; ip < 4; ip += 2) {
              const int i = ip + mi / 2, c = 2 * h + mi % 2;
              stmatrix_x4_trans(stg + (8 * i + r) * 64 + ((c ^ (r / 2)) << 4), pk[h][ip][0], pk[h][ip][1],
                                pk[h][ip + 1][0], pk[h][ip + 1][1]);
            }
          fence_proxy_async_smem();
          __syncwarp();
          if (!Prob::kRowMaps || c0 + 32 <= tokens) {
            if (lane == 0 && row0 < p.n) {
              tma_store_2d(&p.ymap, stg, row0, t.m0 + c0);
              bulk_commit();
            }
          } else {
#pragma unroll
            for (int it = 0; it < 4; ++it) {
              const int rr = it * 8 + lane / 4, col = (lane % 4) * 8;
              if (c0 + rr < tokens && row0 + col < p.n) {
                const uint4 val = lds128(stg + rr * 64 + (((lane % 4) ^ ((rr / 2) % 4)) << 4));
                *reinterpret_cast<uint4*>(p.y + static_cast<int64_t>(t.m0 + c0 + rr) * p.ldy + row0 + col) = val;
              }
            }
            __syncwarp();
          }
        } else {
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int ip = 0; ip < 4; ip += 2) {
              const int i = ip + mi / 2, e = mi % 2;
              stmatrix_x4_trans(stg + (8 * i + lane % 8) * kStagingRowBytes + (16 * h + 8 * e) * 2, pk[h][ip][0],
                                pk[h][ip][1], pk[h][ip + 1][0], pk[h][ip + 1][1]);
            }
          __syncwarp();
#pragma unroll
          for (int it = 0; it < 4; ++it) {
            const int r = it * 8 + lane / 4;
            const int col = (lane % 4) * 8;
            if (c0 + r < tokens && row0 + col < p.n) {
              const uint4 val = lds128(stg + r * kStagingRowBytes + col * 2);
              __nv_bfloat16* yrow;
              if constexpr (Prob::kRowMaps)
                yrow = p.y + static_cast<int64_t>(tok_rows[c0 + r]) * p.ldy;
              else
                yrow = Prob::y_row(p, t, c0 + r);
              *reinterpret_cast<uint4*>(yrow + row0 + col) = val;
            }
          }
          __syncwarp();
        }
      }
      if constexpr (Prob::kStreamK) {
        if (contributor) {
          // Every partial store precedes the tag (release).
          __threadfence();
          named_bar_sync(1, 32 * kEpilogueWarps);
          if (warp == kEpilogueWarp0 && lane == 0) st_release64(p.tags + blockIdx.x, stream_k_tag(w.tile, w.kb0));
        }
      }
    }
    // The staging is read and y written before the CTA exits.
    if (tma_y && lane == 0) bulk_wait();
  }

  tc_fence_before();
  __syncthreads();
  if constexpr (Prob::kClusterSplit) {
    cluster_sync();  // every rank's partial is in the workspace (release / acquire at cluster scope)
    typename Prob::Sched own(p, static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x));
    own.next(w);
    const TcTile t = Prob::tile(p, w, 0);
    // Rank r finishes slice r of the tile's 8-row chunks (16 per token): the
    // sum of all ranks' partials in rank order, plus bias, as bf16.
    constexpr int kChunks = kTileRows / 8;
    const int kc = p.kc, r = static_cast<int>(cluster_rank());
    const int chunks = t.tokens * kChunks;
    const float* part = p.partials + static_cast<int64_t>(blockIdx.x - r) * BN * kTileRows;
    for (int c = chunks * r / kc + static_cast<int>(threadIdx.x); c < chunks * (r + 1) / kc; c += kThreads) {
      const int tok = c / kChunks, row = 8 * (c % kChunks);
      if (t.row0 + row >= p.n) continue;
      const float4* src = reinterpret_cast<const float4*>(part + tok * kTileRows + row);
      // Written by other SMs: L2 loads (.cg), not L1. Every rank's loads are
      // issued before the first add (one L2 round trip), then summed in rank
      // order. kc <= 8 (portable cluster size).
      float4 lo[8], hi[8];
#pragma unroll
      for (int s = 0; s < 8; ++s)
        if (s < kc) {
          lo[s] = __ldcg(src + s * (BN * kTileRows / 4));
          hi[s] = __ldcg(src + s * (BN * kTileRows / 4) + 1);
        }
#pragma unroll
      for (int s = 1; s < 8; ++s)
        if (s < kc) {
          lo[0].x += lo[s].x;
          lo[0].y += lo[s].y;
          lo[0].z += lo[s].z;
          lo[0].w += lo[s].w;
          hi[0].x += hi[s].x;
          hi[0].y += hi[s].y;
          hi[0].z += hi[s].z;
          hi[0].w += hi[s].w;
        }
      const float4 a = lo[0], b = hi[0];
      float f[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      if (p.bias != nullptr) {
        const uint4 bv = *reinterpret_cast<const uint4*>(p.bias + t.row0 + row);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&bv);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 bf = __bfloat1622float2(b2[i]);
          f[2 * i] += bf.x;
          f[2 * i + 1] += bf.y;
        }
      }
      *reinterpret_cast<uint4*>(Prob::y_row(p, t, tok) + t.row0 + row) =
          make_uint4(bf16x2_rn(f[0], f[1]), bf16x2_rn(f[2], f[3]), bf16x2_rn(f[4], f[5]), bf16x2_rn(f[6], f[7]));
    }
  } else if constexpr (kPair) {
    // The peer may still multicast commits / read this CTA's smem until both
    // CTAs are done.
    cluster_sync();
  }
  if (warp == kMmaWarp) {
    tc_fence_after();
    tmem_dealloc<kTmemCols, kCtas>(tmem);
  }
#else
  __trap();  // never launched: the *_supported() probes reject images without tcgen05
#endif
}


inline PFN_cuTensorMapEncodeTiled_v12000 tensor_map_encoder() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) !=
            cudaSuccess ||
        q != cudaDriverEntryPointSuccess)
      return static_cast<PFN_cuTensorMapEncodeTiled_v12000>(nullptr);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }();
  return fn;
}

// Tensor-map strides must be < 2^40 bytes and dimensions <= 2^32.
inline constexpr uint64_t kMaxTmaStride = uint64_t(1) << 40;

// View weights as [16-row block][k super-atom][super-atom bytes as u64].
// One TMA fetches eight super-atoms across a 128-row tile, contiguous along
// K within each row block and strided between blocks.
inline bool encode_weight_map(CUtensorMap* map, const uint8_t* weight, int super_atom, int64_t kb, int64_t blocks) {
  auto encode = tensor_map_encoder();
  if (encode == nullptr || static_cast<uint64_t>(kb) * super_atom >= kMaxTmaStride ||
      static_cast<uint64_t>(blocks) > (uint64_t(1) << 32))
    return false;
  const cuuint64_t dims[3] = {static_cast<cuuint64_t>(super_atom / 8), static_cast<cuuint64_t>(kb),
                              static_cast<cuuint64_t>(blocks)};
  const cuuint64_t strides[2] = {static_cast<cuuint64_t>(super_atom), static_cast<cuuint64_t>(kb) * super_atom};
  const cuuint32_t box[3] = {static_cast<cuuint32_t>(super_atom / 8), 1, kBlocksPerTile};
  const cuuint32_t estr[3] = {1, 1, 1};
  return encode(map, CU_TENSOR_MAP_DATA_TYPE_UINT64, 3, const_cast<uint8_t*>(weight), dims, strides, box, estr,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// y as [rows][n] bf16 for the epilogue's TMA stores: boxes of 32 tokens x 32
// weight rows, SWIZZLE_64B.
inline bool encode_y_map(CUtensorMap* map, __nv_bfloat16* y, int64_t n, int64_t rows, int64_t ldy) {
  auto encode = tensor_map_encoder();
  if (encode == nullptr || static_cast<uint64_t>(ldy) * 2 >= kMaxTmaStride ||
      static_cast<uint64_t>(rows) > (uint64_t(1) << 32) || static_cast<uint64_t>(n) > (uint64_t(1) << 32))
    return false;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(n), static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(ldy) * 2};
  const cuuint32_t box[2] = {32, 32};
  const cuuint32_t estr[2] = {1, 1};
  return encode(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, y, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// x as [rows][k] bf16 in the SWIZZLE_128B K-major layout UMMA reads: boxes of
// `box_rows` x 64.
inline bool encode_x_map(CUtensorMap* map, const __nv_bfloat16* x, int64_t k, int64_t rows, int64_t ldx,
                         int box_rows) {
  auto encode = tensor_map_encoder();
  if (encode == nullptr || static_cast<uint64_t>(ldx) * 2 >= kMaxTmaStride ||
      static_cast<uint64_t>(rows) > (uint64_t(1) << 32))
    return false;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(k), static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(ldx) * 2};
  const cuuint32_t box[2] = {kBK, static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t estr[2] = {1, 1};
  return encode(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<__nv_bfloat16*>(x), dims, strides, box, estr,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// Require SM100/103/110 and static shared memory in the loaded kernel body.
// Baseline images can match binaryVersion while containing only a trap.
// Cache the image check per device.
template <auto kKernel>
bool kernel_image_supported(const DeviceInfo& dev) {
  if (dev.sm_arch != 100 && dev.sm_arch != 103 && dev.sm_arch != 110) return false;
  static std::atomic<int8_t> cache[kMaxDevices];  // 0 unknown, 1 yes, -1 no
  const bool cacheable = dev.device >= 0 && dev.device < kMaxDevices;
  if (cacheable) {
    const int8_t c = cache[dev.device].load(std::memory_order_relaxed);
    if (c != 0) return c > 0;
  }
  cudaFuncAttributes attr{};
  bool ok = false;
  if (cudaFuncGetAttributes(&attr, kKernel) == cudaSuccess)
    ok = attr.binaryVersion == dev.sm_arch && attr.sharedSizeBytes > 0;
  else
    cudaGetLastError();  // clear the non-sticky lookup error we caused
  if (cacheable) cache[dev.device].store(ok ? 1 : -1, std::memory_order_relaxed);
  return ok;
}

// Dynamic shared memory is a per-device attribute. Cache only successful grants.
template <auto kKernel, int kSmem>
cudaError_t grant_smem(const DeviceInfo& dev) {
  static std::atomic<bool> granted[kMaxDevices];
  const int d = dev.device;
  if (d >= 0 && d < kMaxDevices && granted[d].load(std::memory_order_acquire)) return cudaSuccess;
  const cudaError_t err = cudaFuncSetAttribute(kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (err == cudaSuccess && d >= 0 && d < kMaxDevices) granted[d].store(true, std::memory_order_release);
  return err;
}

// `attr` holds two entries: the cluster shape (when cluster > 1) and, with
// `pdl`, programmatic stream serialization.
template <int kSmem>
cudaLaunchConfig_t launch_config(int grid, int cluster, cudaStream_t stream, cudaLaunchAttribute* attr,
                                 bool pdl = false) {
  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(grid);
  lc.blockDim = dim3(kThreads);
  lc.dynamicSmemBytes = kSmem;
  lc.stream = stream;
  lc.attrs = attr;
  if (cluster > 1) {
    attr[lc.numAttrs].id = cudaLaunchAttributeClusterDimension;
    attr[lc.numAttrs].val.clusterDim.x = cluster;
    attr[lc.numAttrs].val.clusterDim.y = 1;
    attr[lc.numAttrs].val.clusterDim.z = 1;
    ++lc.numAttrs;
  }
  if (pdl) {
    attr[lc.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[lc.numAttrs].val.programmaticStreamSerializationAllowed = 1;
    ++lc.numAttrs;
  }
  return lc;
}

// PDL requires griddepcontrol.wait before global access (linear/dispatch.cu).
template <auto kKernel, int kSmem, class Params>
cudaError_t launch_kernel(const DeviceInfo& dev, int grid, int cluster, cudaStream_t stream, const CUtensorMap& xmap,
                          const CUtensorMap& wmap, const Params& p, bool pdl = false) {
  const cudaError_t err = grant_smem<kKernel, kSmem>(dev);
  if (err != cudaSuccess) return err;
  cudaLaunchAttribute attr[2];
  const cudaLaunchConfig_t lc = launch_config<kSmem>(grid, cluster, stream, attr, pdl);
  return cudaLaunchKernelEx(&lc, kKernel, xmap, wmap, p);
}

// CTAs of `kKernel` resident at once in clusters of `cluster` CTAs (clusters
// must fit inside one GPC), 0 if the query fails. Not cached.
template <auto kKernel, int kSmem>
int resident_ctas(const DeviceInfo& dev, int cluster) {
  if (grant_smem<kKernel, kSmem>(dev) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  cudaLaunchAttribute attr[2];
  const cudaLaunchConfig_t lc = launch_config<kSmem>(cluster, cluster, nullptr, attr);
  int clusters = 0;
  if (cudaOccupancyMaxActiveClusters(&clusters, kKernel, &lc) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return clusters * cluster;
}

}  // namespace nf12::detail::tcg
