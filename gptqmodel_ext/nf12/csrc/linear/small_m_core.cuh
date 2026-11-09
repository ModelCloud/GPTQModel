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
// The streaming core of the small-M kernels, shared by the dense small-M
// linear (linear/small_m.cu) and the MoE decode kernel (moe/small.cu).
//
// A CTA of WARPS warps streams `cnt` consecutive super-atoms per warp (one
// 16-row block each, all warps walking the same k positions) against up to
// 8 * MT tokens of x, and the `kc` CTAs of a thread-block cluster split K and
// reduce their FP32 partials through distributed shared memory in rank
// order, so no workspace, atomics or second kernel are needed and the result
// is deterministic. Without clusters (SM8x) K is split inside the CTA
// instead: KP warps share a row block and interleave its positions, and
// their partials are summed in shared memory in part order.
//
// Each warp's register ring crosses x-slice boundaries to overlap weight
// loads with x restaging. x is staged row-major with a 16-byte row pad and
// read by ldmatrix as the MMA B operand. Rows >= m alias row m-1 and are
// discarded at the store, so no zero fill is needed.
#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include <cstdint>

#include "linear/common.cuh"

namespace nf12::detail {

// Two super-atoms keep 24 KB in flight per 8-warp CTA without deeper rings'
// register-pressure cost.
inline constexpr int kSmallMDepth = 2;

// Each shared-memory stage is one cp.async.bulk of kSmallMBulkAtoms
// consecutive super-atoms per warp, completed through an mbarrier. This
// shifts in-flight weight storage from registers to shared memory.
inline constexpr int kSmallMBulkAtoms = 2;
inline constexpr int kSmallMBulkStages = 2;

// x slices hold whole weight steps: register-ring steps of kSmallMDepth
// super-atoms, shared-memory stages of kSmallMBulkAtoms.
NF12_HD constexpr int small_m_step(int stages) { return stages > 0 ? kSmallMBulkAtoms : kSmallMDepth; }

// Ring slots followed by one mbarrier per slot. Callers place the ring at
// the 128-byte aligned start of dynamic shared memory for bulk-copy alignment.
NF12_HD constexpr int64_t small_m_ring_bytes(int stages, int warps, int super_atom_bytes) {
  return int64_t{stages} * warps * (kSmallMBulkAtoms * super_atom_bytes + 8);
}

// FP32 partials are stored [token][row]; a stride of 4 mod 32 words makes the
// fragment scatter (lanes differ in row g and token 2t) bank-conflict free.
NF12_HD constexpr int small_m_partial_stride(int warps) { return 16 * warps + 4; }

#if defined(__CUDACC__)

template <bool Retain>
__device__ __forceinline__ uint4 small_m_weight_load(const void* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
  if constexpr (Retain) {
    // Coherent L2 loads retain Ada weights after preceding cache pressure.
    uint4 r;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
  }
  return ldg_stream(p);
#else
  return ldg_stream(p);
#endif
}

__device__ __forceinline__ uint4 ld_shared_v4(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
  return v;
}

// Streams one CTA's work and hands each finished output pair to `store`.
//   kStages  0: register ring of kSmallMDepth super-atoms per warp
//            (ld.global); > 0: shared-memory ring of kStages stages per warp
//            (cp.async.bulk, see kSmallMBulkAtoms). `slice` must then be a
//            multiple of kSmallMBulkAtoms (small_m_step).
//   KP       K parts per CTA (register ring only): the WARPS warps are
//            RW = WARPS / KP row-block warps x KP parts; warp w streams row
//            block w % RW at the CTA positions u = p, p + KP, p + 2 KP, ...
//            of its part p = w / RW, and the parts' FP32 partials are summed
//            in part order before the cluster reduction. For devices without
//            clusters (SM8x), where it is the only way to split K. `slice`
//            must be a multiple of KP * small_m_step(kStages).
//   smem     dynamic shared memory: x slots (2 * m * xstride, or one slot
//            when slice covers cnt), later reused for the partials
//            (KP * m * small_m_partial_stride(RW) floats).
//   wring    (kStages > 0) 128-byte aligned shared memory of
//            small_m_ring_bytes(kStages, WARPS, super-atom bytes), apart
//            from `smem`.
//   wp       this warp's first super-atom (CTA position p), already offset
//            by lane * 16.
//   cnt      super-atom positions of the CTA (CTA-uniform); cnt >= 1.
//   base_x4, escapes  the matrix's NF12 decode parameters (unused for bf16t).
//   m        tokens, 1 <= m <= 8 * MT.
//   xrow(r)  pointer to x row r at the CTA's first k column.
//   prep()   runs after first weight loads and before staging x. If it writes
//            values read by xrow(), it must end with a CTA barrier.
//   store(tok, rp, s)  output column pair rp (columns 2rp, 2rp+1 of the
//            CTA's 16 * RW rows) of token tok, FP32 sums of all parts and
//            ranks in order. Called once per (tok, rp) across the cluster.
template <WeightSource S, int MT, int WARPS, int kStages, int KP = 1, bool Retain = false,
          class XRow, class Prep, class Store>
__device__ __forceinline__ void small_m_core(uint8_t* smem, uint8_t* wring, const uint8_t* wp, int cnt,
                                             uint32_t base_x4, const uint4* __restrict__ escapes, int m, int slice,
                                             int xstride, int kc, int rank, XRow xrow, Prep prep, Store store) {
  using T = WeightTraits<S>;
  constexpr bool kBulk = kStages > 0;
  constexpr int kDepth = kSmallMDepth;
  constexpr int kAtoms = kSmallMBulkAtoms;
  constexpr int kStageBytes = kAtoms * T::kSuperAtomBytes;
  static_assert(KP >= 1 && WARPS % KP == 0 && (KP == 1 || !kBulk), "K parts need the register ring");
  constexpr int RW = WARPS / KP;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  // K part and row-block warp of this warp; its positions p + j * KP, j < wcnt.
  const int kpart = KP == 1 ? 0 : warp / RW;
  const int rw = KP == 1 ? warp : warp % RW;
  const int wcnt = KP == 1 ? cnt : (cnt > kpart ? (cnt - kpart + KP - 1) / KP : 0);

  // Register ring (unused with the shared-memory ring).
  uint4 ring[kDepth][T::kLoads];
  // Shared-memory ring: stage j of the warp's run (super-atoms
  // [j * kAtoms, j * kAtoms + kAtoms)) goes to slot j % kSlots, whose
  // barrier (after all slots) completes once per fill.
  constexpr int kSlots = kBulk ? kStages : 1;
  const uint32_t wslots = smem_addr(wring) + warp * kSlots * kStageBytes;
  const uint32_t wbars = smem_addr(wring) + WARPS * kSlots * kStageBytes + warp * kSlots * 8;
  const int nstages = (cnt + kAtoms - 1) / kAtoms;
  auto fill = [&](int j) {
    const int atoms = min(kAtoms, cnt - j * kAtoms);
    bulk_load_stream(wslots + (j % kSlots) * kStageBytes, wp - lane * 16 + static_cast<size_t>(j) * kStageBytes,
                     atoms * T::kSuperAtomBytes, wbars + (j % kSlots) * 8);
  };
  if constexpr (kBulk) {
    if (lane == 0) {
      for (int s = 0; s < kSlots; ++s) ring_bar_init(wbars + s * 8, 1);
      ring_bar_fence_init();
      for (int j = 0; j < kSlots && j < nstages; ++j) fill(j);
    }
    __syncwarp();
  } else {
#pragma unroll
    for (int d = 0; d < kDepth; ++d) {
      if (d < wcnt) {
#pragma unroll
        for (int l = 0; l < T::kLoads; ++l)
          ring[d][l] = small_m_weight_load<Retain>(wp + d * (KP * T::kSuperAtomBytes) + l * kPlaneBytes);
      }
    }
  }
  prep();

  const int nslices = (cnt + slice - 1) / slice;
  const uint32_t slot_bytes = static_cast<uint32_t>(m * xstride);
  const uint32_t sbase = smem_addr(smem);

  auto stage_x = [&](int s) {
    const int u0 = s * slice;
    const int row_chunks = min(slice, cnt - u0) * 8;  // 16-byte chunks per token row
    const uint32_t dst = sbase + (s & 1) * slot_bytes;
    for (int i = threadIdx.x; i < m * row_chunks; i += WARPS * 32) {
      const int r = i / row_chunks;
      const int c = i - r * row_chunks;
      cp_async_16(dst + r * xstride + c * 16, xrow(r) + static_cast<size_t>(u0) * 64 + c * 8);
    }
    cp_async_commit();
  };
  stage_x(0);
  if (nslices > 1) stage_x(1);

  // ldmatrix.x4 row addresses: lanes 8q..8q+7 feed matrix q = 16-byte chunk q
  // of a 64-byte half super-atom (b0, b1 of atom 2h, then of atom 2h+1).
  uint32_t xaddr[MT];
#pragma unroll
  for (int nt = 0; nt < MT; ++nt)
    xaddr[nt] = sbase + min(nt * 8 + (lane & 7), m - 1) * xstride + (lane >> 3) * 16;

  // Independent accumulator chains hide MMA dependency latency at small MT.
  // Atom j uses chain j % kChains, and final summation uses fixed chain order.
  constexpr int kChains = kBulk ? (MT == 1 ? 4 : MT == 2 ? 2 : 1) : 1;
  float acc[kChains][MT][4];
#pragma unroll
  for (int c = 0; c < kChains; ++c)
#pragma unroll
    for (int nt = 0; nt < MT; ++nt) acc[c][nt][0] = acc[c][nt][1] = acc[c][nt][2] = acc[c][nt][3] = 0.f;

  // One super-atom's four A fragments against the MT token tiles of x at
  // byte offset xoff of the staged slices.
  auto mma_tiles = [&](const uint32_t (&a)[4][4], uint32_t xoff) {
#pragma unroll
    for (int nt = 0; nt < MT; ++nt) {
      uint32_t b[8];
      ldmatrix_x4(*reinterpret_cast<uint32_t(*)[4]>(&b[0]), xaddr[nt] + xoff);
      ldmatrix_x4(*reinterpret_cast<uint32_t(*)[4]>(&b[4]), xaddr[nt] + xoff + 64);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma_bf16_16816(acc[j % kChains][nt], a[j], b[2 * j], b[2 * j + 1]);
    }
  };

  for (int s = 0; s < nslices; ++s) {
    if (s + 1 < nslices)
      cp_async_wait<1>();
    else
      cp_async_wait<0>();
    __syncthreads();
    const uint32_t xslot = (s & 1) * slot_bytes;
    const int u0 = s * slice;
    const int u1 = min(cnt, u0 + slice);
    if constexpr (kBulk) {
      // u0 is a multiple of kAtoms, so each step is one whole stage.
      for (int u = u0; u < u1; u += kAtoms) {
        const int j = u / kAtoms;
        ring_bar_wait(wbars + (j % kSlots) * 8, (j / kSlots) & 1);
        const uint32_t src = wslots + (j % kSlots) * kStageBytes + lane * 16;
        const int atoms = min(kAtoms, u1 - u);
        // The next super-atom's planes load while this one decodes.
        uint4 cur[T::kLoads], nxt[T::kLoads];
#pragma unroll
        for (int l = 0; l < T::kLoads; ++l) cur[l] = ld_shared_v4(src + l * kPlaneBytes);
#pragma unroll
        for (int c = 0; c < kAtoms; ++c) {
          if (c < atoms) {
            if (c + 1 < atoms) {
#pragma unroll
              for (int l = 0; l < T::kLoads; ++l)
                nxt[l] = ld_shared_v4(src + (c + 1) * T::kSuperAtomBytes + l * kPlaneBytes);
            }
            uint32_t a[4][4];
            T::fragments(cur, base_x4, escapes, a);
            mma_tiles(a, xslot + (u + c - u0) * 128);
#pragma unroll
            for (int l = 0; l < T::kLoads; ++l) cur[l] = nxt[l];
          }
        }
        // Every lane's loads of the slot have been consumed by its decode, so
        // the slot can be refilled.
        __syncwarp();
        if (lane == 0 && j + kSlots < nstages) fill(j + kSlots);
      }
    } else {
      // Warp-local steps j: position kpart + j * KP. u0 is a multiple of
      // KP * kDepth, so j0 is a multiple of kDepth and ring[d] holds step j0 + d.
      const int j0 = u0 / KP;
      const int j1 = (u1 - kpart + KP - 1) / KP;
      for (int j = j0; j < j1; j += kDepth) {
#pragma unroll
        for (int d = 0; d < kDepth; ++d) {
          if (j + d < j1) {
            uint32_t a[4][4];
            T::fragments(ring[d], base_x4, escapes, a);
            if (j + d + kDepth < wcnt) {
              const uint8_t* src = wp + static_cast<size_t>(j + d + kDepth) * (KP * T::kSuperAtomBytes);
#pragma unroll
              for (int l = 0; l < T::kLoads; ++l) ring[d][l] = small_m_weight_load<Retain>(src + l * kPlaneBytes);
            }
            // Duplicated instead of calling mma_tiles to avoid extra registers
            // and reduced SM90 occupancy.
            const uint32_t xoff = xslot + (kpart + (j + d) * KP - u0) * 128;
#pragma unroll
            for (int nt = 0; nt < MT; ++nt) {
              uint32_t b[8];
              ldmatrix_x4(*reinterpret_cast<uint32_t(*)[4]>(&b[0]), xaddr[nt] + xoff);
              ldmatrix_x4(*reinterpret_cast<uint32_t(*)[4]>(&b[4]), xaddr[nt] + xoff + 64);
              mma_bf16_16816(acc[0][nt], a[0], b[0], b[1]);
              mma_bf16_16816(acc[0][nt], a[1], b[2], b[3]);
              mma_bf16_16816(acc[0][nt], a[2], b[4], b[5]);
              mma_bf16_16816(acc[0][nt], a[3], b[6], b[7]);
            }
          }
        }
      }
    }
    if (s + 2 < nslices) {
      __syncthreads();  // every warp is done reading this slot
      stage_x(s + 2);
    }
  }

  // Partials -> shared [token][row] (x is dead). C fragment: c0, c1 = row g,
  // tokens 2t, 2t+1; c2, c3 = row g + 8.
#pragma unroll
  for (int c = 1; c < kChains; ++c)
#pragma unroll
    for (int nt = 0; nt < MT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[0][nt][i] += acc[c][nt][i];
  __syncthreads();
  constexpr int RS = small_m_partial_stride(RW);
  float* part = reinterpret_cast<float*>(smem);
  {
    const int g = lane >> 2;
    const int t = lane & 3;
    const int row = rw * 16 + g;
    float* mine = part + kpart * m * RS;
#pragma unroll
    for (int nt = 0; nt < MT; ++nt) {
      const int tok = nt * 8 + 2 * t;
      if (tok < m) {
        mine[tok * RS + row] = acc[0][nt][0];
        mine[tok * RS + row + 8] = acc[0][nt][2];
      }
      if (tok + 1 < m) {
        mine[(tok + 1) * RS + row] = acc[0][nt][1];
        mine[(tok + 1) * RS + row + 8] = acc[0][nt][3];
      }
    }
  }
  if constexpr (KP > 1) {
    // Sum the parts into part 0, in part order.
    __syncthreads();
    for (int e = threadIdx.x; e < m * RW * 16; e += WARPS * 32) {
      const int off = e / (RW * 16) * RS + e % (RW * 16);
      float s = part[off];
#pragma unroll
      for (int p = 1; p < KP; ++p) s += part[p * m * RS + off];
      part[off] = s;
    }
  }

  // Each cluster rank finalizes a contiguous share of the CTA's outputs,
  // summing the partials of ranks 0..kc-1 in order.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  namespace cg = cooperative_groups;
  if (kc > 1)
    cg::this_cluster().sync();
  else
#endif
    __syncthreads();

  constexpr int kPairs = RW * 8;  // bf16x2 outputs per token
  const int total = m * kPairs;
  const int e0 = rank * total / kc;
  const int e1 = (rank + 1) * total / kc;
  for (int e = e0 + static_cast<int>(threadIdx.x); e < e1; e += WARPS * 32) {
    const int tok = e / kPairs;
    const int rp = e - tok * kPairs;
    const int off = tok * RS + 2 * rp;
    float2 s;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (kc > 1) {
      // Issue all remote loads before adding to overlap DSMEM latency.
      // The sum still runs in rank order.
      const auto cluster = cg::this_cluster();
      float2 v[8];
#pragma unroll
      for (int q = 0; q < 8; ++q)
        if (q < kc) v[q] = *reinterpret_cast<const float2*>(cluster.map_shared_rank(part, q) + off);
      s = v[0];
#pragma unroll
      for (int q = 1; q < 8; ++q) {
        if (q < kc) {
          s.x += v[q].x;
          s.y += v[q].y;
        }
      }
    } else
#endif
      s = *reinterpret_cast<const float2*>(part + off);
    store(tok, rp, s);
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // Keep this CTA's partials alive until every rank has read them.
  if (kc > 1) cg::this_cluster().sync();
#endif
}

#endif  // __CUDACC__

}  // namespace nf12::detail
