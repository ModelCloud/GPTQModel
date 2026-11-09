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
// SM120 (RTX Blackwell) large-M NF12 GEMM kernel: warp-specialized, TMA-fed,
// persistent mma.sync kernel (swap-AB: weights are the A operand, tokens B),
// shared by linear/sm120/large_m_sm120.cu and moe/sm120/large_sm120.cu.
//
// One CTA per SM, 12 warps. Warpgroup 0 gives up registers (setmaxnreg) and
// runs the data movement:
//   warp 0     weight producer: one 3D TMA per k64 stage fetches the BN / 16
//              compressed super-atoms of the CTA's weight rows (contiguous
//              along K per 16-row block) into a ring of SW stages.
//   warp 1     activation producer: fills a separate ring of SX stages,
//              SWIZZLE_128B layout (the problem decides how: one 2D TMA
//              tile per stage, or gathered rows).
//   warps 2-3  escape fetchers (NF12): scan each landed weight stage for
//              escaped groups, cp.async their table entries into a pool and
//              publish each group's pool slot in a rank table, about a stage
//              before the consumers reach it.
// Warpgroups 1-2 (8 consumer warps, 232 registers each) hold a stage's
// compressed planes in registers (so the weight slot is released as soon as
// a stage opens), decode them straight into mma A fragments (decode_bulk) and
// run mma.sync m16n8k16 against ldmatrix B fragments. Shared memory holds
// weights at 1.5 bytes each, never decoded. A stage with escapes in a warp's
// rows runs a variant that overwrites those groups' fragments from the pool.
//
// The bf16t source runs the identical pipeline (ring depths may differ, they
// do not affect arithmetic) with the decode replaced by plain loads, so both
// sources feed the same operands to the same MMA sequence.
//
// All threads construct one Problem after griddepcontrol.wait and before
// roles split, permitting CTA-wide barriers in its constructor. Each role
// must walk the same deterministic next() list. Interface:
//   using Params;  using Work  (int nt, kb0, kb1 plus its own fields)
//   static constexpr int kExtraBytes;   shared memory of the problem
//   static constexpr bool kXWarp;   the activation producer runs on the whole warp (else lane 0)
//   static constexpr int kXArrivals;  arrivals completing an activation stage (with its tx bytes)
//   static void init(uint32_t extra)        thread 0, before the CTA barrier: problem mbarriers
//   Prob(const Params&, uint8_t* extra, uint32_t extra_addr)
//   template <class C> bool next(Work&)
//   template <class C> int w_block(w)    weight TMA row block
//   bool w_streamed_once(w)               tile's weights are read by no other tile: L2 evict-first
//   template <class C> x_tile(w, lane); x_load(xmap, dst, kb, w, bar, policy, lane)
//   base_x4(w), escapes(w), num_escapes(w): the tile's decode operands
//   n(); zfill_src()
//   template <class C, int RB, int TF> epilogue(acc, w, cw, lane, valid_rb)
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

#include "linear/common.cuh"
#include "linear/sm120/large_m_sm120_ptx.cuh"
#include "nf12/format.cuh"

namespace nf12::detail::sm120k {

inline constexpr int kConsumerWarps = 8;
inline constexpr int kThreads = 32 * (4 + kConsumerWarps);
inline constexpr int kSmemBudget = 101376;  // SM120 per-block opt-in limit
inline constexpr int kBarrierBytes = 512;

// Rank bytes use 0xfe for pool overflow and 0xff for non-escaped groups.
inline constexpr uint32_t kRankOverflow = 0xfeu;
inline constexpr uint32_t kRankNone = 0xffu;

// BN x BM tiles use WN x (8 / WN) consumer warps and SX activation stages.
// kExtra reserves problem shared memory. The weight ring takes the rest.
template <int BN_, int BM_, int WN_, int SX_, WeightSource S, int kExtra_>
struct Cfg {
  static constexpr int BN = BN_, BM = BM_, WN = WN_, WM = kConsumerWarps / WN_, SX = SX_;
  static constexpr int kExtra = kExtra_;
  static constexpr bool kNF12 = S == WeightSource::kNF12;
  static_assert(WN * WM == kConsumerWarps && BN % (16 * WN) == 0 && BM % (16 * WM) == 0, "warp tiling");
  static_assert(kExtra % 16 == 0, "problem area alignment");
  static constexpr int kBlocks = BN / 16;
  static constexpr int SA = kNF12 ? kSuperAtomBytes : kSuperAtomBytesBf16t;
  static constexpr int kWBytes = kBlocks * SA;
  static constexpr int kXBytes = BM * 128;  // multiple of 1024: SWIZZLE_128B tiles stay aligned
  // Escape areas (NF12), one per stage in flight: a rank table (one word per
  // super-atom lane, byte j = pool slot of that lane's atom-j group,
  // kRankNone if not escaped, kRankOverflow past the pool) and a pool of
  // entries. Each escape warp fills its own half of the pool (kEscCap entries
  // per super-atom it scans, shared among them, so a dense super-atom can use
  // a sparse neighbour's room). There are SW + kEscLag areas. Escape warps
  // release a weight slot only after the area kEscLag stages earlier completes,
  // keeping kEscLag stages of entries in flight. The ring is sized with a
  // minimal pool, then pools take leftover shared memory up to 32 entries per
  // super-atom and below kRankOverflow per half. Consumers load overflow
  // entries directly from the table.
  static constexpr int kEscLag = 1;
  static constexpr int kRankBytes = kBlocks * 32 * 4;
  static constexpr int kMinCap = 4, kMaxCap = 32;
  static constexpr int kFree = kSmemBudget - kBarrierBytes - SX * kXBytes - kExtra;
  static constexpr int kMinEsc = kNF12 ? kRankBytes + kBlocks * kMinCap * 16 : 0;
  static constexpr int SW = std::min(8, (kFree - kEscLag * kMinEsc) / (kWBytes + kMinEsc));
  static constexpr int SE = kNF12 ? SW + kEscLag : 0;  // escape areas
  static constexpr int kEscCap =
      std::min({kMaxCap, (kFree - SW * kWBytes - SE * kRankBytes) / std::max(1, SE * kBlocks * 16),
                (static_cast<int>(kRankOverflow) - 1) / (kBlocks / 2)});
  static constexpr int kHalfCap = kBlocks / 2 * kEscCap;  // pool entries per escape warp
  static constexpr int kEscBytes = kNF12 ? kRankBytes + 2 * kHalfCap * 16 : 0;
  static_assert(kBlocks % 2 == 0, "the two escape warps scan alternate super-atoms");
  static_assert(SW >= 2 && (!kNF12 || kEscCap >= kMinCap), "weight ring too shallow");
  static_assert(2 * SW + SE + 2 * SX <= kBarrierBytes / 8, "barrier space");
  static constexpr int kSmemBytes = SX * kXBytes + SW * kWBytes + SE * kEscBytes + kExtra + kBarrierBytes;
};

#if defined(NF12_SM120_DEVICE)
constexpr int kEscWarps = 2;  // warps 2-3: escape fetchers, splitting a stage's super-atoms
constexpr int kProducerRegs = 40;
constexpr int kConsumerRegs = 232;  // (384 * 168 - 128 * 40) / 256: the whole register file

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
#endif  // NF12_SM120_DEVICE

template <int BN, int BM, int WN, int SX, WeightSource S, class Prob>
__global__ void __launch_bounds__(kThreads, 1)
    large_m_sm120_kernel(const __grid_constant__ CUtensorMap xmap, const __grid_constant__ CUtensorMap wmap,
                         const __grid_constant__ typename Prob::Params p) {
  using C = Cfg<BN, BM, WN, SX, S, Prob::kExtraBytes>;
#if defined(NF12_SM120_DEVICE)
  using namespace sm120;
  constexpr int SW = C::SW;
  constexpr int TN = BN / C::WN;  // warp tile: weight rows
  constexpr int TM = BM / C::WM;  // warp tile: tokens
  constexpr int RB = TN / 16;     // 16-row blocks per warp
  constexpr int TF = TM / 8;      // 8-token MMA fragments per warp
  constexpr int TG = TM / 16;     // ldmatrix.x4 per k16 step
  constexpr int SA = C::SA;

  // SWIZZLE_128B tiles require 1024-byte alignment. No static shared memory
  // precedes this dynamic window.
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t x_base = smem_u32(smem);
  const uint32_t w_base = x_base + SX * C::kXBytes;
  constexpr int SE = C::SE;
  const uint32_t esc_base = w_base + SW * C::kWBytes;  // per escape area: rank table, then pool
  const uint32_t extra = esc_base + SE * C::kEscBytes;  // the problem's area
  const uint32_t bars = extra + C::kExtra;
  auto w_full = [&](int s) { return bars + 8 * s; };          // W TMA -> escape warps, consumers (tx)
  auto w_empty = [&](int s) { return bars + 8 * (SW + s); };  // consumers (+ escape warps) -> W producer
  auto x_full = [&](int s) { return bars + 8 * (2 * SW + s); };       // x TMA -> consumers (tx)
  auto x_empty = [&](int s) { return bars + 8 * (2 * SW + SX + s); };  // consumers -> x producer
  auto esc_full = [&](int e) { return bars + 8 * (2 * SW + 2 * SX + e); };  // escape area e -> consumers

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  if (threadIdx.x == 0) {
    if (x_base & 1023u) __trap();
    for (int s = 0; s < SW; ++s) {
      mbar_init(w_full(s), 1);
      mbar_init(w_empty(s), kConsumerWarps + (C::kNF12 ? kEscWarps : 0));
    }
    for (int e = 0; e < SE; ++e) mbar_init(esc_full(e), kEscWarps * (32 + 1));  // cp.asyncs + rank table, per warp
    for (int s = 0; s < SX; ++s) {
      mbar_init(x_full(s), Prob::kXArrivals);
      mbar_init(x_empty(s), kConsumerWarps);
    }
    Prob::init(extra);
    mbar_fence_init();
    prefetch_tmap(&xmap);
    prefetch_tmap(&wmap);
  }
  __syncthreads();
  // PDL allows prologue overlap but all global accesses wait for the prior
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
          for (int kb = w.kb0; kb < w.kb1; ++kb) {
            mbar_wait(w_empty(r.stage), r.phase ^ 1);
            // Row blocks past n are zero-filled (or the next matrix's) and still counted.
            mbar_arrive_expect_tx(w_full(r.stage), C::kWBytes);
            tma_load_3d(w_base + r.stage * C::kWBytes, &wmap, 0, kb, block, w_full(r.stage), policy);
            r.advance<SW>();
          }
        }
      }
    } else if (warp == 1) {
      if (Prob::kXWarp || lane == 0) {
        const uint64_t policy = policy_evict_last();  // x is re-read by every n-tile
        Ring r;
        while (prob.template next<C>(w)) {
          prob.template x_tile<C>(w, lane);
          for (int kb = w.kb0; kb < w.kb1; ++kb) {
            mbar_wait(x_empty(r.stage), r.phase ^ 1);
            prob.template x_load<C>(&xmap, x_base + r.stage * C::kXBytes, kb, w, x_full(r.stage), policy, lane);
            r.advance<SX>();
          }
        }
      }
    } else if (warp >= 2) {
      if constexpr (C::kNF12) {
        // Escape fetcher. Escape indices follow flat group order, so the rank of
        // an escaped group among the escaped groups of its super-atom (atom-major,
        // then lane) is also its offset from the super-atom's first index. The
        // ranks double as pool slots and are published in the rank table.
        const uint32_t lt = (1u << lane) - 1u;
        // Escape area of each of the last kEscLag stages (oldest first).
        Ring r, re, lagged[C::kEscLag];
        int pending = 0;
        // This warp shares a pool half across super-atoms b = warp - 2,
        // warp, ... with slots below kRankOverflow.
        const uint32_t half = static_cast<uint32_t>(warp - 2) * C::kHalfCap * 16;
        constexpr uint32_t kHalfCap = C::kHalfCap;
        while (prob.template next<C>(w)) {
          const int valid = min(C::kBlocks, prob.n() / 16 - w.nt * C::kBlocks);
          for (int kb = w.kb0; kb < w.kb1; ++kb) {
            mbar_wait(w_full(r.stage), r.phase);
            const uint32_t slot = w_base + r.stage * C::kWBytes + lane * 16;
            const uint32_t ranks = esc_base + re.stage * C::kEscBytes;
            const uint32_t pool = ranks + C::kRankBytes + half;
            uint32_t used = 0;  // pool entries of this warp taken so far in this stage
            // Load all planes before the dependent scan to avoid serializing
            // shared loads with escape votes.
            for (int b = warp - 2; b < valid; b += kEscWarps) {
              const uint4 m = lds128(slot + b * SA);
              const uint4 l01 = lds128(slot + b * SA + kPlaneBytes);
              const uint4 l23 = lds128(slot + b * SA + 2 * kPlaneBytes);
              const bool e[4] = {m.x == 0u, m.y == 0u, m.z == 0u, m.w == 0u};
              const uint32_t esc_idx[4] = {l01.x, l01.z, l23.x, l23.z};  // L0 of atoms 0-3
              uint32_t rk = 0xffffffffu;
              if (__any_sync(0xffffffffu, e[0] | e[1] | e[2] | e[3])) {
                rk = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                  const uint32_t bal = __ballot_sync(0xffffffffu, e[j]);
                  const uint32_t slot_j = used + __popc(bal & lt);
                  const bool pooled = slot_j < kHalfCap;
                  rk |= (!e[j] ? kRankNone : pooled ? slot_j : kRankOverflow) << (8 * j);
                  if (e[j] && pooled) {
                    const uint32_t idx = esc_idx[j];
                    // src-size 0 reads nothing. zfill_src() supplies a valid
                    // address for out-of-table indices.
                    const bool ok = idx < prob.num_escapes(w);
                    const void* src = ok ? static_cast<const void*>(prob.escapes(w) + idx) : prob.zfill_src();
                    cp_async_16_zfill(pool + slot_j * 16, src, ok ? 16 : 0);
                  }
                  used += __popc(bal);
                }
              }
              st_shared32(ranks + (b * 32 + lane) * 4, rk);
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(esc_full(re.stage));  // releases the rank table
            cp_async_mbar_arrive_noinc(esc_full(re.stage));
            // A consumer without escapes in a stage skips its esc_full wait, so
            // the first one it waits on next may be the same area's following
            // phase, and a parity wait cannot tell "previous phase still open"
            // from "done". Every phase of an area must therefore complete before
            // a consumer can reach its next one (stage + SE). Stage s + SE lands
            // only after this warp released the slot of stage s + SE - SW =
            // s + kEscLag, which it does only once the area of stage s is
            // complete. Waiting kEscLag stages back (not 1) keeps that many
            // stages of escape fetches in flight.
            if (pending == C::kEscLag) {
              mbar_wait(esc_full(lagged[0].stage), lagged[0].phase);
#pragma unroll
              for (int i = 0; i + 1 < C::kEscLag; ++i) lagged[i] = lagged[i + 1];
              --pending;
            }
            lagged[pending++] = re;
            fence_proxy_async_smem();
            __syncwarp();
            if (lane == 0) mbar_arrive(w_empty(r.stage));
            r.advance<SW>();
            re.advance<SE>();
          }
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
      }
    }
    return;
  }

  reg_alloc<kConsumerRegs>();
  const int cw = warp - 4;
  const int wn = cw % WN, wm = cw / WN;
  // ldmatrix.x4 lane address: matrices (tok 0-7, k lo), (tok 0-7, k hi),
  // (tok 8-15, k lo), (tok 8-15, k hi) of each 16-token group of the warp,
  // chunks XOR-swizzled by row (SWIZZLE_128B).
  const uint32_t b_row = (wm * TM + (lane & 7) + ((lane >> 4) << 3)) * 128;
  const int b_hi = (lane >> 3) & 1, b_swz = lane & 7;
  float acc[RB][TF][4];
  // Token-fragment-major MMAs consume half the B fragments before the other
  // half. Refill each consumed half with the next step, using one buffer.
  uint32_t bfrag[TF][2];
  auto load_b = [&](int half, uint32_t xs, int kk) {
    const uint32_t a = xs + b_row + (((2 * kk + b_hi) ^ b_swz) * 16);
#pragma unroll
    for (int tg = half * TG / 2; tg < (half + 1) * TG / 2; ++tg) {
      uint32_t r[4];
      ldmatrix_x4(r, a + tg * 16 * 128);
      bfrag[2 * tg][0] = r[0];
      bfrag[2 * tg][1] = r[1];
      bfrag[2 * tg + 1][0] = r[2];
      bfrag[2 * tg + 1][1] = r[3];
    }
  };
  auto mma_half = [&](const uint32_t (&a)[RB][4], int half) {
#pragma unroll
    for (int tf = half * TF / 2; tf < (half + 1) * TF / 2; ++tf)
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) mma_16816(acc[rb][tf], a[rb], bfrag[tf][0], bfrag[tf][1]);
  };

  // NF12 loads all compressed planes (12 registers per row block) when a
  // stage opens, releasing the weight slot early for escape-fetch slack.
  // bf16t loads A fragments one k16 step ahead and releases at stage end.
  // Open the next stage during the current stage's final step and run its
  // escape census after the MMAs, avoiding a load/vote/branch chain at entry.
  uint4 meta[RB] = {}, lo01[RB] = {}, lo23[RB] = {};
  uint32_t afr[2][RB][4];
  Ring rw, rx, re;  // weight stage, activation stage, escape area
  auto w_lane = [&]() { return w_base + rw.stage * C::kWBytes + wn * RB * SA + lane * 16; };
  auto x_slot = [&]() { return x_base + rx.stage * C::kXBytes; };
  auto load_a = [&](int buf, uint32_t ws, int kk) {  // bf16t
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      const uint4 v = lds128(ws + rb * SA + kk * kPlaneBytes);
      afr[buf][rb][0] = v.x;
      afr[buf][rb][1] = v.y;
      afr[buf][rb][2] = v.z;
      afr[buf][rb][3] = v.w;
    }
  };
  // NF12 loads all A operands and releases the weight slot. bf16t loads
  // only the first step's A fragments.
  auto open_stage = [&]() {
    mbar_wait(w_full(rw.stage), rw.phase);
    mbar_wait(x_full(rx.stage), rx.phase);
    const uint32_t ws = w_lane();
    if constexpr (C::kNF12) {
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        meta[rb] = lds128(ws + rb * SA);
        lo01[rb] = lds128(ws + rb * SA + kPlaneBytes);
        lo23[rb] = lds128(ws + rb * SA + 2 * kPlaneBytes);
      }
      fence_proxy_async_smem();
      __syncwarp();
      if (lane == 0) mbar_arrive(w_empty(rw.stage));
    } else {
      load_a(0, ws, 0);
    }
  };
  // Escape census of the opened stage (warp-uniform): does any of the
  // warp's row blocks hold an escaped group?
  auto census = [&](int valid_rb) {
    bool any = false;
    if constexpr (C::kNF12) {
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
        any |= rb < valid_rb && ((meta[rb].x == 0u) | (meta[rb].y == 0u) | (meta[rb].z == 0u) | (meta[rb].w == 0u));
    }
    return __any_sync(0xffffffffu, any);
  };

  // Consume an opened k64 stage with the first B half already loaded.
  // Release its slots and optionally open the next stage. Constant kEsc
  // selects predicated escape patches, staying in one basic block unless
  // the pool overflowed.
  auto run_stage = [&](const typename Prob::Work& w, bool kEsc, bool more, int valid_rb) {
    const uint32_t ws = w_lane(), xs = x_slot();
    const uint32_t ranks = esc_base + re.stage * C::kEscBytes + wn * RB * 32 * 4 + lane * 4;
    const uint32_t pool = esc_base + re.stage * C::kEscBytes + C::kRankBytes;
    // Fetch step kk + 1 replacements while kk decodes, keeping escape latency
    // out of the decode-to-MMA chain without buffering all four steps.
    uint32_t rk[RB] = {}, ev[RB][4];
    auto fetch = [&](int rb, int kk) {
      const uint32_t r = (rk[rb] >> (8 * kk)) & 0xffu;
      const uint32_t idx = kk == 0 ? lo01[rb].x : kk == 1 ? lo01[rb].z : kk == 2 ? lo23[rb].x : lo23[rb].z;
      // Super-atom wn * RB + rb was scanned by escape warp (its index % 2).
      const uint32_t half = static_cast<uint32_t>((wn * RB + rb) % kEscWarps) * C::kHalfCap;
#pragma unroll
      for (int c = 0; c < 4; ++c) ev[rb][c] = 0u;
      lds128_if(ev[rb], pool + (half + r) * 16, r < static_cast<uint32_t>(C::kHalfCap));
      ldg128_if(ev[rb], prob.escapes(w) + idx, (r == kRankOverflow) & (idx < prob.num_escapes(w)));
    };
    if constexpr (C::kNF12) {
      if (kEsc) {
        mbar_wait(esc_full(re.stage), re.phase);
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) {
          rk[rb] = rb < valid_rb ? lds32(ranks + rb * 32 * 4) : 0xffffffffu;  // rows past n: never stored
          fetch(rb, 0);
        }
      }
    }
    load_b(1, xs, 0);
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t a[RB][4];
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        if constexpr (C::kNF12) {
          const uint4& l = kk < 2 ? lo01[rb] : lo23[rb];
          const Group g{word(meta[rb], kk), (kk & 1) ? l.z : l.x, (kk & 1) ? l.w : l.y};
          decode_bulk(g, prob.base_x4(w), a[rb]);
          if (kEsc) {
            const bool e = ((rk[rb] >> (8 * kk)) & 0xffu) != kRankNone;
#pragma unroll
            for (int c = 0; c < 4; ++c) a[rb][c] = e ? ev[rb][c] : a[rb][c];
            if (kk < 3) fetch(rb, kk + 1);
          }
        } else {
#pragma unroll
          for (int c = 0; c < 4; ++c) a[rb][c] = afr[kk & 1][rb][c];
        }
      }
      if (kk < 3) {
        if constexpr (!C::kNF12) load_a((kk + 1) & 1, ws, kk + 1);
      } else {
        // Every read of this stage's slots has been issued.
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) {
          if constexpr (!C::kNF12) mbar_arrive(w_empty(rw.stage));
          mbar_arrive(x_empty(rx.stage));
        }
        rw.advance<SW>();
        rx.advance<SX>();
        if constexpr (C::kNF12) re.advance<SE>();
        if (more) open_stage();
      }
      mma_half(a, 0);
      if (kk < 3) {
        load_b(0, xs, kk + 1);
      } else if (more) {
        load_b(0, x_slot(), 0);
      }
      mma_half(a, 1);
      if (kk < 3) load_b(1, xs, kk + 1);
    }
  };

  while (prob.template next<C>(w)) {
    const int valid_rb = min(RB, (prob.n() - (w.nt * BN + wn * TN)) / 16);
#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
      for (int tf = 0; tf < TF; ++tf)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[rb][tf][c] = 0.f;

    open_stage();
    load_b(0, x_slot(), 0);
    bool esc = census(valid_rb);
    for (int kb = w.kb0; kb < w.kb1; ++kb) {
      const bool more = kb + 1 < w.kb1;
      if (esc)
        run_stage(w, true, more, valid_rb);
      else
        run_stage(w, false, more, valid_rb);
      if (more) esc = census(valid_rb);
    }
    prob.template epilogue<C, RB, TF>(acc, w, cw, lane, valid_rb);
  }
#else
  static_assert(C::kSmemBytes > 0 && Prob::kXArrivals > 0 && (Prob::kXWarp || Prob::kXArrivals == 1));
  __trap();  // never launched: the *_supported() probes reject images without the SM120 body
#endif
}

}  // namespace nf12::detail::sm120k
