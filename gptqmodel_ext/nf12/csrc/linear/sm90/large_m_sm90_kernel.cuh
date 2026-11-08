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
// SM90 (Hopper) large-M NF12 GEMM kernel: warp-specialized, TMA-fed,
// persistent wgmma kernel (swap-AB: weights are the A operand, tokens B),
// shared by linear/sm90/large_m_sm90.cu and moe/sm90/large_sm90.cu.
//
// One CTA per SM, three warpgroups. Warpgroup 0 gives up registers
// (setmaxnreg) and runs the data movement:
//   warp 0     weight producer: one 3D TMA per stage fetches WK consecutive
//              compressed super-atoms (k64 blocks) of each of the CTA's BN / 16
//              row blocks (contiguous along K per 16-row block) into a ring
//              of SW stages.
//   warp 1     activation producer: fills a separate ring of SX stages,
//              SWIZZLE_128B K-major layout (the problem decides how: one 2D
//              TMA tile per stage, or rows gathered by warps 1-3).
// Warpgroups 1-2 (cooperative: both on the same BN x BM tile, each on half
// its weight rows) run wgmma.mma_async m64nBMk16 with A from registers
// ("RS"). The wgmma A fragment of warp q of a warpgroup is exactly the
// mma.sync m16n8k16 A fragment of its 16 rows, and a decoded NF12 group is
// that fragment (SPEC.md 4.1), so each lane decodes its own groups straight
// into A registers without shuffles or shared-memory round trips. Tensor
// cores read B through a swizzled shared-memory descriptor. Each consumer
// warp owns RB = BN / 128 contiguous 16-row blocks. Block i is its portion
// of the warpgroup's i-th m64 wgmma, so its accumulator matches the
// mma.sync C layout used by the problem epilogues.
//
// Consumers copy a k-block's compressed planes into registers a k-block
// ahead where the registers allow (accumulators of at most 64 registers),
// else one k16 step before it starts (a weight slot is released once its
// last k-block is copied), and decode each k16 step into one of four A
// register sets while up to three earlier steps' wgmmas are in flight.
//
// A lane starts loading its first escaped group per row block one k-block
// ahead, into registers when kAhead allows or otherwise into a fixed shared
// slot with cp.async. Further escaped groups load at decode time. The
// warp-uniform escape branch contains no wgmma, so every warp of a warpgroup
// issues the same wgmma sequence.
//
// The bf16t source runs the identical pipeline (ring depths may differ, they
// do not affect arithmetic) with the stage's four fragments in registers and
// the decode replaced by a copy, so both sources feed the same operands
// to the same wgmma sequence.
//
// A Problem policy (template parameter `Prob`) owns the work list, the
// operands of each tile and the epilogue (interface as in
// linear/sm120/large_m_sm120_kernel.cuh). Cfg fixes WN = 8 and WM = 1.
// kXTensorMap selects whether x_load uses xmap or a zeroed map.
// kXWarps producer warps run x_load with lane = their thread index
// 0..32 * kXWarps - 1. x_tile runs on warp 1 alone. No zfill_src().
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

#include "linear/common.cuh"
#include "linear/sm90/large_m_sm90_ptx.cuh"
#include "nf12/format.cuh"

namespace nf12::detail::sm90k {

inline constexpr int kConsumerWarps = 8;
inline constexpr int kThreads = 32 * (4 + kConsumerWarps);
inline constexpr int kSmemBudget = 232448;  // SM90 per-block opt-in limit (227 KiB)
inline constexpr int kBarrierBytes = 512;

// BN = 128 or 256 weight rows use one or two m64 wgmmas per warpgroup
// and k16 step. BM is wgmma N. SX counts x stages, kExtra is problem shared
// memory and WK counts k-blocks per weight stage. The weight ring takes the rest.
template <int BN_, int BM_, int SX_, WeightSource S, int kExtra_, int WK_ = 1>
struct Cfg {
  static constexpr int BN = BN_, BM = BM_, WN = kConsumerWarps, WM = 1, SX = SX_;
  static constexpr int kExtra = kExtra_;
  // A weight stage fetches WK consecutive super-atoms per row block in one
  // TMA box, amortizing transfer overhead across adjacent k-blocks.
  static constexpr int WK = WK_;
  static_assert(WK == 1 || WK == 2, "k-blocks per weight stage");
  static constexpr bool kNF12 = S == WeightSource::kNF12;
  static_assert(BN == 128 || BN == 256, "one or two m64 wgmmas per warpgroup");
  static_assert(BM % 16 == 0 && BM >= 16 && BM <= 256, "wgmma N");
  static_assert(kExtra % 16 == 0, "problem area alignment");
  static constexpr int RB = BN / 128;  // 16-row blocks per consumer warp
  static constexpr int kBlocks = BN / 16;
  static constexpr int SA = kNF12 ? kSuperAtomBytes : kSuperAtomBytesBf16t;
  static constexpr int kWBytes = kBlocks * SA * WK;
  // Round to 1024-byte multiples to align every SWIZZLE_128B stage.
  static constexpr int kXBytes = (BM * 128 + 1023) / 1024 * 1024;
  // With at most 64 accumulator registers, keep two k-blocks' planes and
  // escapes in registers. Otherwise keep one plane set and store escapes
  // in shared [consumer warp][2 k-blocks][RB][32 lanes] slots of 16 bytes.
  static constexpr bool kAhead = RB * BM <= 128;
  static constexpr int kSlotBytes = kNF12 && !kAhead ? kConsumerWarps * 2 * RB * 32 * 16 : 0;
  static constexpr int SW =
      std::min(8, (kSmemBudget - kBarrierBytes - SX * kXBytes - kSlotBytes - kExtra) / kWBytes);
  static_assert(SW >= 2, "weight ring");
  static_assert(SX >= 3, "a stage's activations are released one stage after its last wgmma is issued");
  static_assert(2 * SW + 2 * SX <= kBarrierBytes / 8, "barrier space");
  static constexpr int kSmemBytes = SX * kXBytes + SW * kWBytes + kSlotBytes + kExtra + kBarrierBytes;
};

#if defined(NF12_SM90_DEVICE)
constexpr int kProducerRegs = 40;
constexpr int kConsumerRegs = 232;  // (65536 - 128 * 40) / 256, rounded down to a multiple of 8

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

__device__ __forceinline__ uint32_t word(const uint4& v, int j) {
  return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
}
#endif  // NF12_SM90_DEVICE

template <int BN, int BM, int SX, WeightSource S, class Prob, int WK = 1>
__global__ void __launch_bounds__(kThreads, 1)
    large_m_sm90_kernel(const __grid_constant__ CUtensorMap xmap, const __grid_constant__ CUtensorMap wmap,
                        const __grid_constant__ typename Prob::Params p) {
  using C = Cfg<BN, BM, SX, S, Prob::kExtraBytes, WK>;
#if defined(NF12_SM90_DEVICE)
  static_assert(C::kSmemBytes > 0 && Prob::kXArrivals > 0 && (Prob::kXWarp || Prob::kXArrivals == 1) &&
                (Prob::kXWarp || Prob::kXWarps == 1) && Prob::kXWarps >= 1 && Prob::kXWarps <= 3);
  using namespace sm90;
  constexpr int SW = C::SW;
  constexpr int RB = C::RB;
  constexpr int TF = BM / 8;  // 8-token accumulator fragments
  constexpr int SA = C::SA;

  // SWIZZLE_128B requires 1024-byte alignment. No static shared memory
  // precedes this dynamic window.
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t x_base = smem_u32(smem);
  const uint32_t w_base = x_base + SX * C::kXBytes;
  const uint32_t slots = w_base + SW * C::kWBytes;  // NF12 escape slots without kAhead
  const uint32_t extra = slots + C::kSlotBytes;     // the problem's area
  const uint32_t bars = extra + C::kExtra;
  auto w_full = [&](int s) { return bars + 8 * s; };                   // W TMA -> consumers (tx)
  auto w_empty = [&](int s) { return bars + 8 * (SW + s); };           // consumers -> W producer
  auto x_full = [&](int s) { return bars + 8 * (2 * SW + s); };        // x producer -> consumers
  auto x_empty = [&](int s) { return bars + 8 * (2 * SW + SX + s); };  // consumers -> x producer

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  if (threadIdx.x == 0) {
    if (x_base & 1023u) __trap();
    for (int s = 0; s < SW; ++s) {
      mbar_init(w_full(s), 1);
      mbar_init(w_empty(s), kConsumerWarps);
    }
    for (int s = 0; s < SX; ++s) {
      mbar_init(x_full(s), Prob::kXArrivals);
      mbar_init(x_empty(s), kConsumerWarps);
    }
    Prob::init(extra);
    mbar_fence_init();
    if constexpr (Prob::kXTensorMap) prefetch_tmap(&xmap);
    prefetch_tmap(&wmap);
  }
  __syncthreads();
  // PDL permits prologue overlap, but all global accesses wait for the prior
  // kernel's completion and visibility (linear/dispatch.cu contract).
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");

  Prob prob(p, smem + (extra - x_base), extra);
  typename Prob::Work w{};

  if (warp < 4) {
    reg_dealloc<kProducerRegs>();
    if (warp == 0) {
      if (lane == 0) {
        const uint64_t evict_first = policy_evict_first(), evict_last = policy_evict_last();
        Ring r;
        while (prob.template next<C>(w)) {
          const uint64_t policy = prob.w_streamed_once(w) ? evict_first : evict_last;
          const int block = prob.template w_block<C>(w);
          // A segment's last stage may run WK - 1 k-blocks past kb1: still
          // in the matrix's rows (read, unused) or zero-filled past K.
          for (int kb = w.kb0; kb < w.kb1; kb += WK) {
            mbar_wait(w_empty(r.stage), r.phase ^ 1);
            // Row blocks past n are zero-filled (or the next matrix's) and still counted.
            mbar_arrive_expect_tx(w_full(r.stage), C::kWBytes);
            tma_load_3d(w_base + r.stage * C::kWBytes, &wmap, 0, kb, block, w_full(r.stage), policy);
            r.advance<SW>();
          }
        }
      }
    } else if (warp <= Prob::kXWarps) {
      if (Prob::kXWarp || lane == 0) {
        constexpr int kXThreads = 32 * Prob::kXWarps;
        const uint64_t policy = policy_evict_last();  // x is re-read by every n-tile
        Ring r;
        while (prob.template next<C>(w)) {
          // With several gathering warps, the tile's rows are resolved once
          // they are all done with the previous tile's.
          if constexpr (Prob::kXWarps > 1) named_bar_sync(2, kXThreads);
          if (warp == 1) prob.template x_tile<C>(w, lane);
          if constexpr (Prob::kXWarps > 1) named_bar_sync(2, kXThreads);
          for (int kb = w.kb0; kb < w.kb1; ++kb) {
            mbar_wait(x_empty(r.stage), r.phase ^ 1);
            prob.template x_load<C>(&xmap, x_base + r.stage * C::kXBytes, kb, w, x_full(r.stage), policy,
                                    (warp - 1) * 32 + lane);
            r.advance<SX>();
          }
        }
      }
    }
    return;
  }

  reg_alloc<kConsumerRegs>();
  const int cw = warp - 4;
  // A k-block's A operand registers: NF12 its three compressed planes, bf16t
  // its four fragments. With kAhead the next k-block's are copied during the
  // running one's first step, else during its last step (`raw` index 0 only).
  constexpr bool kAhead = C::kAhead;
  constexpr int kRaw = C::kNF12 ? 3 : 4;
  uint4 raw[kAhead ? 2 : 1][RB][kRaw];
  // Two escape sets track the current and next k-blocks. Per-row-block q
  // stores the escaped-group mask in bits 0-3, first group in bits 4-5 and
  // prefetched-entry validity in bit 6. ev holds that entry with kAhead,
  // otherwise it lives in the lane's shared slot. any is warp-uniform.
  struct Esc {
    uint32_t ev[RB][4];
    uint32_t q[RB];
    bool any;
  };
  Esc es[2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      es[i].q[rb] = 0;
#pragma unroll
      for (int c = 0; c < 4; ++c) es[i].ev[rb][c] = 0;
    }
  // Without kAhead: the lane's escape slot of row block rb for escape set i.
  auto slot = [&](int i, int rb) { return slots + (((cw * 2 + i) * RB + rb) * 32 + lane) * 16; };
  // bf16t steps pass their fragments through an XOR with this runtime zero
  // (gridDim.y is 1): a plain copy would let the wgmmas read `raw` itself,
  // which a later k-block reloads while they are in flight.
  const uint32_t zero = gridDim.y - 1u;
  float acc[RB][TF][4];
  uint32_t afr[4][RB][4];  // A fragments of the k16 steps in flight
  // rw/hw identify the next weight k-block to open. Without kAhead, rs/hs
  // track the escape scan. rx identifies the active x stage.
  Ring rw, rs, rx;
  int hw = 0, hs = 0;
  auto w_lane = [&](const Ring& r, int h) {
    return w_base + r.stage * C::kWBytes + (cw * RB * WK + h) * SA + lane * 16;
  };

  while (prob.template next<C>(w)) {
    const int valid_rb = min(RB, (prob.n() - (w.nt * BN + cw * (BN / kConsumerWarps))) / 16);
#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
      for (int tf = 0; tf < TF; ++tf)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[rb][tf][c] = 0.f;

    // Scan the next k-block after copying its planes, or from its landed
    // weight slot without kAhead. Prefetch each lane's first escape per row
    // block into registers or one cp.async group. bf16t retains a per-k-block
    // runtime-false vote to prevent ptxas serialization (decode_step).
    auto find_escapes = [&](int i, int b, bool last) {
      Esc& e = es[i];
      if constexpr (C::kNF12) {
        uint32_t ws = 0;
        if constexpr (!kAhead) {
          if (hs == 0) mbar_wait(w_full(rs.stage), rs.phase);
          ws = w_lane(rs, hs);
          if (++hs == WK || last) {
            rs.advance<SW>();
            hs = 0;
          }
        }
        uint32_t any = 0;
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) {
          const uint4 m = kAhead ? raw[b][rb][0] : lds128(ws + rb * WK * SA);
          // Rows past n are never stored: their groups are not patched.
          const uint32_t q = rb < valid_rb ? static_cast<uint32_t>(m.x == 0u) | static_cast<uint32_t>(m.y == 0u) << 1 |
                                                 static_cast<uint32_t>(m.z == 0u) << 2 |
                                                 static_cast<uint32_t>(m.w == 0u) << 3
                                           : 0u;
          e.q[rb] = q;
          any |= q;
        }
        e.any = __any_sync(0xffffffffu, any != 0u);
        if (e.any) {
#pragma unroll
          for (int rb = 0; rb < RB; ++rb) {
            const uint32_t q = e.q[rb];
            const uint32_t j = static_cast<uint32_t>(__ffs(q) - 1) & 3u;
            // L0 of group j (plane 1 + j / 2, word 2 (j & 1) of the lane's 16
            // bytes), selected by value: a runtime-indexed register array
            // would go to local memory.
            uint32_t idx;
            if constexpr (kAhead) {
              const uint4(&r)[kRaw] = raw[b][rb];
              idx = j == 0u ? r[1].x : j == 1u ? r[1].z : j == 2u ? r[2].x : r[2].z;
            } else {
              idx = lds32(ws + rb * WK * SA + (1 + (j >> 1)) * kPlaneBytes + (j & 1u) * 8);
            }
            const bool ok = q != 0u && idx < prob.num_escapes(w);
            if constexpr (kAhead) {
              ldg128_if(e.ev[rb], prob.escapes(w) + idx, ok);
            } else if (ok) {
              cp_async_16_zfill(slot(i, rb), prob.escapes(w) + idx, 16);
            }
            e.q[rb] = q | j << 4 | static_cast<uint32_t>(ok) << 6;
          }
        }
        if constexpr (!kAhead) cp_async_commit_group();
      } else {
        e.any = __any_sync(0xffffffffu, false);
      }
    };
    // Waits for the next k-block's weights, copies its planes into raw[b]
    // and, after its stage's last k-block of the segment (`last`), releases
    // the weight slot.
    auto open = [&](int b, bool last) {
      if (hw == 0) mbar_wait(w_full(rw.stage), rw.phase);
      const uint32_t ws = w_lane(rw, hw);
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int j = 0; j < kRaw; ++j) raw[b][rb][j] = lds128(ws + rb * WK * SA + j * kPlaneBytes);
      if (++hw == WK || last) {
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(w_empty(rw.stage));
        rw.advance<SW>();
        hw = 0;
      }
    };
    // Decodes step kk of the k-block in raw[b] into afr[kk] and, if the
    // warp's rows hold escaped groups (escape set i), patches the lane's
    // escaped ones: the first per row block from `ev` or its slot, any other
    // from the table now. Runs while earlier steps' wgmmas are in flight.
    auto decode_step = [&](int b, int i, int kk) {
      const Esc& e = es[i];
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int c = 0; c < 4; ++c) reg_fence(afr[kk][rb][c]);
      // The k-block's slot fills have landed: they are its escape set's
      // group, the only one outstanding when its first step decodes (the
      // next k-block's is committed during its second step).
      if constexpr (C::kNF12 && !kAhead)
        if (kk == 0 && e.any) cp_async_wait_group<0>();
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        const uint4(&r)[kRaw] = raw[b][rb];
        if constexpr (C::kNF12) {
          const uint4& l = kk < 2 ? r[1] : r[2];
          const Group g{word(r[0], kk), (kk & 1) ? l.z : l.x, (kk & 1) ? l.w : l.y};
          decode_bulk(g, prob.base_x4(w), afr[kk][rb]);
        } else {
          afr[kk][rb][0] = r[kk].x ^ zero;
          afr[kk][rb][1] = r[kk].y ^ zero;
          afr[kk][rb][2] = r[kk].z ^ zero;
          afr[kk][rb][3] = r[kk].w ^ zero;
        }
        // Preserve this runtime-false bf16t branch. Without it, ptxas
        // serializes wgmmas for either source (C7513, input-register definitions).
        if (e.any) {
          const uint32_t q = e.q[rb];
          const bool esc = (q >> kk) & 1u;
          const bool first = ((q >> 4) & 3u) == static_cast<uint32_t>(kk);
          const bool pre = esc && first && ((q >> 6) & 1u);
          if constexpr (kAhead) {
#pragma unroll
            for (int c = 0; c < 4; ++c) afr[kk][rb][c] = pre ? e.ev[rb][c] : afr[kk][rb][c];
          } else {
            lds128_if(afr[kk][rb], slot(i, rb), pre);
          }
          const uint32_t idx = kk == 0 ? r[1].x : kk == 1 ? r[1].z : kk == 2 ? r[2].x : r[2].z;
          ldg128_if(afr[kk][rb], prob.escapes(w) + idx, esc && !first && idx < prob.num_escapes(w));
        }
      }
    };

    // The tile's first k-block. Without kAhead its escapes are found before
    // it opens: opening may release the slot that holds its L0 words.
    if constexpr (kAhead) {
      open(0, w.kb0 + 1 == w.kb1);
      find_escapes(0, 0, false);
    } else {
      find_escapes(0, 0, w.kb0 + 1 == w.kb1);
      open(0, w.kb0 + 1 == w.kb1);
    }
    decode_step(0, 0, 0);
    Ring rx_prev;  // activation stage of the previous k-block of this tile
    // With kAhead, open the next k-block in step 0 and scan it in step 1.
    // Otherwise open it in step 3. Decode its first step before advancing.
    auto run = [&](int cur, int nxt, int kb) {
      const int rc = kAhead ? cur : 0, rn = kAhead ? nxt : 0;  // raw indices
      const bool more = kb + 1 < w.kb1;
      mbar_wait(x_full(rx.stage), rx.phase);
      const uint32_t xs = x_base + rx.stage * C::kXBytes;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        // Every operand of this step's wgmmas, constants included, is
        // materialized before wgmma.fence: ptxas serializes all wgmmas of a
        // kernel in which a non-wgmma instruction defines a wgmma input
        // between the fence and the wgmma.
        uint64_t desc = desc_sw128(xs + kk * 32);
        uint32_t accumulate = 1;
        reg_fence(desc);
        reg_fence(accumulate);
#pragma unroll
        for (int rb = 0; rb < RB; ++rb)
#pragma unroll
          for (int c = 0; c < 4; ++c) reg_fence(afr[kk][rb][c]);
        wgmma_fence();
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) Wgmma<BM>::mma(acc[rb], afr[kk][rb], desc, accumulate);
        wgmma_commit();
        // At most this step and the one before in flight: the last reader of
        // the next step's A registers (three steps back) has completed, so
        // the next step decodes while these two run.
        wgmma_wait<2>();
        if (kk == 1 && kb > w.kb0) {
          // The previous k-block's wgmmas have all completed: release its
          // activation stage.
          __syncwarp();
          if (lane == 0) mbar_arrive(x_empty(rx_prev.stage));
        }
        if (kk < 3) {
          decode_step(rc, cur, kk + 1);
          if constexpr (kAhead)
            if (kk == 0 && more) open(rn, kb + 2 == w.kb1);
          if (kk == 1 && more) find_escapes(nxt, rn, kb + 2 == w.kb1);
        } else {
          rx_prev = rx;
          rx.advance<SX>();
          if (more) {
            if constexpr (!kAhead) open(rn, kb + 2 == w.kb1);
            decode_step(rn, nxt, 0);
          }
        }
      }
    };
    for (int kb = w.kb0; kb < w.kb1; kb += 2) {
      run(0, 1, kb);
      if (kb + 1 < w.kb1) run(1, 0, kb + 1);
    }
    wgmma_wait<0>();
#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
      for (int tf = 0; tf < TF; ++tf)
#pragma unroll
        for (int c = 0; c < 4; ++c) reg_fence(acc[rb][tf][c]);
    __syncwarp();
    if (lane == 0) mbar_arrive(x_empty(rx_prev.stage));
    prob.template epilogue<C, RB, TF>(acc, w, cw, lane, valid_rb);
  }
#else
  static_assert(C::kSmemBytes > 0);
  __trap();  // never launched: the *_supported() probes reject images without the SM90 body
#endif
}

}  // namespace nf12::detail::sm90k
