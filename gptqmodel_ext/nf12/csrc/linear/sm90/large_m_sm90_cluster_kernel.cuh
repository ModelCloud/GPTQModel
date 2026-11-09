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
// SM90 (Hopper) large-M NF12 GEMM kernel with shared decode: a cluster of CL
// CTAs along the token dimension works on one 128-row weight tile (CL
// consecutive BM-token tiles), and each CTA decodes 1/CL of every weight
// stage into its own shared memory and, by DSMEM bulk copies, into its
// peers'. Decode work per CTA and FLOP drops by CL. The decoded tile feeds
// shared-memory-sourced ("SS") wgmmas. Weights are A and tokens B, sharing
// the Problem interface of linear/sm90/large_m_sm90_kernel.cuh.
// Cfg fixes BN = 128, WN = 8 and WM = 1. Prob::x_load uses the tensor map.
//
// One CTA per SM, three warpgroups. Warpgroup 0 (registers trimmed by
// setmaxnreg) provides four decoder warps. Decoder d owns stages s where
// s % 4 == d, leaving four stage times per decode. For its stage it
//   - waits for the stage's compressed weights: this CTA's share, 8 / CL
//     16-row blocks (super-atoms), fetched by one 3D TMA that the same
//     decoder issued SW stages earlier into the same slot of the weight ring
//     (SW % 4 == 0, so slots never change hands and need no empty barrier);
//   - reads the share's meta planes and, if a group is escaped, starts the
//     loads of all escaped groups' table entries (out-of-table indices:
//     zero), so their latency overlaps the rest of the stage;
//   - once the whole cluster has released the stage slot, issues the
//     stage's activation TMA (BM tokens, SWIZZLE_128B K-major), decodes the
//     share a block at a time into registers and stores each group with
//     stmatrix into the local A stage, overwrites the escaped groups'
//     fragments with their entries, and bulk-copies the share to the same
//     offset of every peer's A stage, completing as transaction bytes on the
//     peer's s_full barrier;
//   - refills its weight slot with stage s + SW.
// Warpgroups 1-2: consumer warpgroup g runs wgmma m64nBMk16 on weight rows
// 64g..64g+63 with A and B from shared memory, keeping one stage's wgmmas in
// flight while the next stage's are issued.
//
// CL = 1 decodes narrow dense tiles cooperatively (BM = 64 or 128).
// Each producer warp decodes two row blocks per stage, and four TMAs fill
// disjoint weight slices. Warp 0 also issues x TMA. Consumers group four
// shared-A wgmmas while producers prepare later stages, avoiding decode
// interleaving in consumer issue without a permanent decoded-weight cache.
//
// A stage holds BM activation rows (128 bytes each, SWIZZLE_128B) and the
// decoded weights (128 rows x 64 k, BF16) laid out as [k16 step][16-row
// block][2 x 2 core matrices], K-major without swizzle: 8x8 core matrices of
// 128 contiguous bytes, 128 bytes between the two along K (LBO), 256 bytes
// between 8-row groups (SBO). Consumer warpgroup g's operand of step j
// starts at block 4g. A lane's decoded group is its m16n8k16 A fragment
// (SPEC.md 4.1), so one stmatrix.x4 stores its four core matrices.
// A CTA's share is one run of 8 / CL blocks per k16 step.
//
// Synchronization per stage slot: s_empty in every CTA counts one arrival
// per consumer warp of the whole cluster (each consumer warp arrives on the
// slot's s_empty in all CTAs once its wgmmas of the slot are done), so the
// decoder that passes its own s_empty may write the slot in every CTA.
// s_full counts the activation TMA's arrival and the local decoder's (after
// its stores and a proxy fence), plus the activation bytes and the peers'
// bulk-copy bytes. Every CTA of a cluster runs the same tile and k-block
// sequence (the Problem schedules clusters), so the rings of all CTAs
// advance in lockstep.
//
// The bf16t source runs the identical pipeline with the decode replaced by a
// fragment load, so both sources write the same A stages and the same
// wgmma sequence consumes them.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

#include "linear/sm90/large_m_sm90_kernel.cuh"
#include "linear/sm90/large_m_sm90_ptx.cuh"
#include "nf12/format.cuh"

namespace nf12::detail::sm90k {

inline constexpr int kDecoders = 4;

// BM tokens per CTA, CL CTAs per cluster and ST decoded/activation stages.
// The compressed ring takes the rest in multiples of kDecoders slots so
// each slot remains owned by decoder s % kDecoders.
template <int BM_, int CL_, int ST_, WeightSource S, int kExtra_>
struct ClusterCfg {
  static constexpr int BN = 128, BM = BM_, WN = kConsumerWarps, WM = 1, ST = ST_, SX = ST_, CL = CL_;
  static constexpr int kExtra = kExtra_;
  static constexpr bool kNF12 = S == WeightSource::kNF12;
  static_assert(CL == 1 || CL == 2 || CL == 4, "cluster size");
  static_assert(CL == 1 ? (BM == 64 || BM == 128) : BM == 256, "consumer wgmma N");
  static_assert(kExtra % 16 == 0, "problem area alignment");
  static constexpr int kBlocks = BN / 16;            // 16-row blocks per tile
  static constexpr int kShareBlocks = kBlocks / CL;  // blocks decoded by one CTA
  static constexpr int kSuper = kNF12 ? kSuperAtomBytes : kSuperAtomBytesBf16t;
  static constexpr int kWBytes = kShareBlocks * kSuper;
  static constexpr int kXBytes = BM * 128;
  static constexpr int kAStepBytes = kBlocks * 512;  // one k16 step of the decoded tile
  static constexpr int kABytes = 4 * kAStepBytes;    // one decoded stage
  static constexpr int kShareBytes = kShareBlocks * 4 * 512;  // decoded bytes of a share
  static constexpr int SW =
      std::min(8, (kSmemBudget - kBarrierBytes - ST * (kXBytes + kABytes) - kExtra) / kWBytes) / kDecoders * kDecoders;
  static_assert(ST % kDecoders == 0 && SW >= kDecoders, "ring slots are owned by one decoder each");
  static_assert(SW + 2 * ST <= kBarrierBytes / 8, "barrier space");
  static_assert(kWBytes % 128 == 0, "TMA destination alignment");
  static constexpr int kSmemBytes = ST * (kXBytes + kABytes) + SW * kWBytes + kExtra + kBarrierBytes;
};

#if defined(NF12_SM90_DEVICE)
// setmaxnreg moves registers within the CTA's launch allocation (384 threads
// x 168): the consumers' gain equals the decoders' release.
constexpr int kDecoderRegs = 152;
constexpr int kSharedConsumerRegs = 176;  // 256 * (176 - 168) == 128 * (168 - 152)
#endif

template <int BM, int CL, int ST, WeightSource S, class Prob>
__global__ void __launch_bounds__(kThreads, 1)
    large_m_sm90_cluster_kernel(const __grid_constant__ CUtensorMap xmap, const __grid_constant__ CUtensorMap wmap,
                                const __grid_constant__ typename Prob::Params p) {
  using C = ClusterCfg<BM, CL, ST, S, Prob::kExtraBytes>;
#if defined(NF12_SM90_DEVICE)
  using namespace sm90;
  static_assert(Prob::kXTensorMap && !Prob::kXWarp, "activations by one tensor-map load per stage");
  constexpr int SW = C::SW;
  constexpr int TF = BM / 8;
  constexpr bool kCooperative = CL == 1;
  constexpr int NB = kCooperative ? C::kBlocks / kDecoders : C::kShareBlocks;

  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t x_base = smem_u32(smem);
  const uint32_t a_base = x_base + ST * C::kXBytes;
  const uint32_t w_base = a_base + ST * C::kABytes;
  const uint32_t extra = w_base + SW * C::kWBytes;
  const uint32_t bars = extra + C::kExtra;
  auto w_full = [&](int s) { return bars + 8 * s; };                // weight TMA -> its decoder (tx)
  auto s_full = [&](int s) { return bars + 8 * (SW + s); };         // x TMA, decoders (+ peers' tx) -> consumers
  auto s_empty = [&](int s) { return bars + 8 * (SW + ST + s); };  // cluster's consumers -> decoders

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const uint32_t rank = cluster_ctarank();

  if (threadIdx.x == 0) {
    if (x_base & 1023u) __trap();
    for (int s = 0; s < SW; ++s) mbar_init(w_full(s), kCooperative ? kDecoders : 1);
    for (int s = 0; s < ST; ++s) {
      mbar_init(s_full(s), Prob::kXArrivals + (kCooperative ? kDecoders : 1));
      mbar_init(s_empty(s), kConsumerWarps * CL);
    }
    Prob::init(extra);
    mbar_fence_init();
    prefetch_tmap(&xmap);
    prefetch_tmap(&wmap);
  }
  // Peers arrive on and copy into this CTA's barriers and stages.
  cluster_sync();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");

  Prob prob(p, smem + (extra - x_base), extra);
  typename Prob::Work w{};

  if (warp < kDecoders) {
    reg_dealloc<kDecoderRegs>();
    const int d = warp;
    const int b0 = (kCooperative ? d : static_cast<int>(rank)) * NB;
    const uint64_t evict_first = policy_evict_first(), evict_last = policy_evict_last();
    // stmatrix row address of lane l: row l % 8 of core matrix l / 8
    // (fragment registers R0..R3 = rows 0-7/8-15 x k 0-7/8-15).
    const int cm = lane >> 3;
    const uint32_t st_lane = static_cast<uint32_t>(cm == 1 ? 256 : cm == 2 ? 128 : cm * 128) + 16u * (lane & 7);
    constexpr uint32_t kRun = NB * 512;  // the share's bytes per k16 step

    // Copy the Problem for the ahead TMA walk. Reconstructing it may invoke
    // CTA-wide barriers that this decoder warp cannot execute alone.
    Prob ahead = prob;
    typename Prob::Work wa{};
    int kba = 0;
    bool ahead_ok = ahead.template next<C>(wa);
    if (ahead_ok) kba = wa.kb0;
    auto ahead_step = [&]() {
      if (++kba < wa.kb1) return true;
      if (!ahead.template next<C>(wa)) return false;
      kba = wa.kb0;
      return true;
    };
    auto load_w = [&](int slot) {  // the ahead stage into weight slot `slot` (lane 0)
      const uint64_t policy = ahead.w_streamed_once(wa) ? evict_first : evict_last;
      // Row blocks past n are zero-filled (or the next matrix's) and still counted.
      mbar_arrive_expect_tx(w_full(slot), NB * C::kSuper);
      const uint32_t dst = w_base + slot * C::kWBytes + (kCooperative ? b0 * C::kSuper : 0);
      tma_load_3d(dst, &wmap, 0, kba, ahead.template w_block<C>(wa) + b0, w_full(slot), policy);
    };
    for (int t = 0; t < SW && ahead_ok; ++t) {
      if ((kCooperative || t % kDecoders == d) && lane == 0) load_w(t);
      ahead_ok = ahead_step();
    }

    Ring rw, rs;
    int st = 0;  // stage index, from 0 over the whole work list
    while (prob.template next<C>(w)) {
      const int valid = min(NB, max(0, (prob.n() - (w.nt * C::BN + b0 * 16)) / 16));
      const uint32_t base_x4 = prob.base_x4(w);
      const uint4* escapes = prob.escapes(w);
      const uint32_t num_escapes = prob.num_escapes(w);
      for (int kb = w.kb0; kb < w.kb1; ++kb, ++st, rw.advance<SW>(), rs.advance<ST>()) {
        if (kCooperative || st % kDecoders == d) {
          mbar_wait(w_full(rw.stage), rw.phase);
          const uint32_t ws = w_base + rw.stage * C::kWBytes + (kCooperative ? b0 * C::kSuper : 0) + lane * 16;
          // NF12: escapes first, so the table loads overlap the stage's decode.
          uint4 meta[C::kNF12 ? NB : 1];
          uint32_t ev[C::kNF12 ? NB : 1][4][4];
          bool esc = false;
          if constexpr (C::kNF12) {
            bool any = false;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
              meta[i] = lds128(ws + i * C::kSuper);
              any |= i < valid && ((meta[i].x == 0u) | (meta[i].y == 0u) | (meta[i].z == 0u) | (meta[i].w == 0u));
            }
            esc = __any_sync(0xffffffffu, any);
            if (__builtin_expect(esc, 0)) {
#pragma unroll
              for (int i = 0; i < NB; ++i) {
                const uint4 l01 = lds128(ws + i * C::kSuper + kPlaneBytes);
                const uint4 l23 = lds128(ws + i * C::kSuper + 2 * kPlaneBytes);
                const uint32_t idx[4] = {l01.x, l01.z, l23.x, l23.z};  // L0 of atoms 0-3
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                  const bool e = i < valid && word(meta[i], j) == 0u;
#pragma unroll
                  for (int c = 0; c < 4; ++c) ev[i][j][c] = 0u;
                  ldg128_if(ev[i][j], escapes + idx[j], e & (idx[j] < num_escapes));
                }
              }
            }
          }

          // Write the stage slot in every CTA once the whole cluster released it.
          mbar_wait(s_empty(rs.stage), rs.phase ^ 1);
          // The warp's lanes call x_load (Problems issue TMA from lane 0).
          if (!kCooperative || d == 0)
            prob.template x_load<C>(&xmap, x_base + rs.stage * C::kXBytes, kb, w, s_full(rs.stage), evict_last, lane);
          // Atom (block i of the share, step j) in the A stage.
          const uint32_t as = a_base + rs.stage * C::kABytes + b0 * 512;
          auto atom = [&](int i, int j) { return as + j * C::kAStepBytes + i * 512; };
#pragma unroll
          for (int i = 0; i < NB; ++i) {
            // Load the block's two NF12 L planes or four bf16t fragments.
            uint4 pl[C::kNF12 ? 2 : 4];
#pragma unroll
            for (int q = 0; q < (C::kNF12 ? 2 : 4); ++q)
              pl[q] = lds128(ws + i * C::kSuper + (C::kNF12 ? q + 1 : q) * kPlaneBytes);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              uint32_t r[4];
              if constexpr (C::kNF12) {
                const uint4& l = pl[j / 2];
                decode_bulk(Group{word(meta[i], j), (j & 1) ? l.z : l.x, (j & 1) ? l.w : l.y}, base_x4, r);
              } else {
                r[0] = pl[j].x;
                r[1] = pl[j].y;
                r[2] = pl[j].z;
                r[3] = pl[j].w;
              }
              stmatrix_x4(atom(i, j) + st_lane, r);
            }
          }
          if constexpr (C::kNF12) {
            if (__builtin_expect(esc, 0)) {
              // Overwrite the escaped groups' fragments (lane l's register c
              // sits in core matrix c at row l / 4, column pair l % 4).
              __syncwarp();  // after every lane's stmatrix
              const uint32_t frag_lane = static_cast<uint32_t>((lane >> 2) * 16 + (lane & 3) * 4);
#pragma unroll
              for (int i = 0; i < NB; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                  if (i < valid && word(meta[i], j) == 0u) {
                    const uint32_t a = atom(i, j) + frag_lane;
                    st_shared32(a, ev[i][j][0]);        // R0: rows 0-7, k 0-7
                    st_shared32(a + 256, ev[i][j][1]);  // R1: rows 8-15, k 0-7
                    st_shared32(a + 128, ev[i][j][2]);  // R2: rows 0-7, k 8-15
                    st_shared32(a + 384, ev[i][j][3]);  // R3: rows 8-15, k 8-15
                  }
                }
            }
          }
          // The proxy fence orders the warp's generic-proxy accesses before
          // the async-proxy ones that follow: the stores before the wgmma and
          // bulk-copy reads, the weight slot's loads before its refill.
          fence_proxy_async_smem();
          __syncwarp();
          if (lane == 0) {
            for (uint32_t q = 1; q < CL; ++q) {
              const uint32_t peer = (rank + q) % CL;
              const uint32_t bar = mapa(s_full(rs.stage), peer);
#pragma unroll
              for (int j = 0; j < 4; ++j) bulk_copy_to_cluster(mapa(atom(0, j), peer), atom(0, j), kRun, bar);
            }
            mbar_arrive_expect_tx(s_full(rs.stage), (CL - 1) * C::kShareBytes);
            if (ahead_ok) load_w(rw.stage);  // stage st + SW, also this decoder's
          }
        }
        if (ahead_ok) ahead_ok = ahead_step();
      }
    }
  } else {
    reg_alloc<kSharedConsumerRegs>();
    const int cw = warp - kDecoders;
    const int wg = cw / 4;
    float acc[1][TF][4];
    Ring rs;
    // Releases a stage slot once the warp's wgmmas of it are done: in every
    // CTA of the cluster (lane q arrives on CTA q's s_empty).
    auto release = [&](const Ring& r) {
      __syncwarp();
      if (lane < CL) mbar_arrive_remote(mapa(s_empty(r.stage), lane));
    };
    while (prob.template next<C>(w)) {
      const int valid_rb = min(1, (prob.n() - (w.nt * C::BN + cw * 16)) / 16);
#pragma unroll
      for (int tf = 0; tf < TF; ++tf)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[0][tf][c] = 0.f;
      Ring rs_prev;
      for (int kb = w.kb0; kb < w.kb1; ++kb) {
        mbar_wait(s_full(rs.stage), rs.phase);
        const uint32_t as = a_base + rs.stage * C::kABytes + wg * 4 * 512;  // the warpgroup's 4 blocks
        const uint32_t xs = x_base + rs.stage * C::kXBytes;
        // Every wgmma operand is materialized before wgmma.fence (see the
        // register-A kernel).
        uint64_t da[4], db[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          da[j] = desc_interleave(as + j * C::kAStepBytes, 128, 256);
          db[j] = desc_sw128(xs + j * 32);
          reg_fence(da[j]);
          reg_fence(db[j]);
        }
        uint32_t accumulate = 1;
        reg_fence(accumulate);
        wgmma_fence();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if constexpr (BM == 64) wgmma_ss_n64(acc[0], da[j], db[j], accumulate);
          else if constexpr (BM == 128) wgmma_ss_n128(acc[0], da[j], db[j], accumulate);
          else wgmma_ss_n256(acc[0], da[j], db[j], accumulate);
        }
        wgmma_commit();
        // The previous stage's wgmmas have completed: release its slot.
        wgmma_wait<1>();
        if (kb > w.kb0) release(rs_prev);
        rs_prev = rs;
        rs.advance<ST>();
      }
      wgmma_wait<0>();
#pragma unroll
      for (int tf = 0; tf < TF; ++tf)
#pragma unroll
        for (int c = 0; c < 4; ++c) reg_fence(acc[0][tf][c]);
      // Keep the last slot through epilogue. Its activation area can become
      // scratch only after all consumer warps finish their wgmmas. The proxy
      // fence orders epilogue accesses before the slot's next activation TMA.
      prob.template epilogue<C, 1, TF>(acc, w, cw, lane, valid_rb, smem + rs_prev.stage * C::kXBytes);
      fence_proxy_async_smem();
      release(rs_prev);
    }
  }
  // No CTA leaves while a peer may still arrive on or copy into it.
  __syncwarp();
  cluster_sync();
#else
  static_assert(C::kSmemBytes > 0);
  __trap();  // never launched: the *_supported() probes reject images without the SM90 body
#endif
}

}  // namespace nf12::detail::sm90k
