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
// Shared PTX wrappers and weight traits mapping one super-atom to four MMA A fragments.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include <nf12/format.cuh>

#include "internal.h"

namespace nf12::detail {

// D += A * B, mma.sync m16n8k16 BF16 -> FP32. a = A fragment (4 regs),
// b = B fragment (2 regs), d = C/D fragment (4 floats).
__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t saddr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(saddr));
}

// Streaming weights bypass L1 allocation to limit cache pollution. volatile
// keeps the loads at their software-pipeline positions.
__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ void cp_async_16(uint32_t saddr, const void* gptr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr), "l"(gptr));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Shared-memory mbarrier and one-dimensional bulk copy (SM90+), for the
// small-M shared-memory weight ring. On older targets they trap: the host
// never selects that ring there (small_m_bulk_stages).
__device__ __forceinline__ void ring_bar_init(uint32_t bar, uint32_t count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
#else
  __trap();
#endif
}
__device__ __forceinline__ void ring_bar_fence_init() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void ring_bar_wait(uint32_t bar, uint32_t parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "WAIT_LOOP:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra WAIT_DONE;\n"
      "bra WAIT_LOOP;\n"
      "WAIT_DONE:\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
#else
  __trap();
#endif
}
// Copies `bytes` (a multiple of 16) from 16-byte aligned global memory into
// this CTA's shared memory. `bar` is armed for `bytes` and signals completion.
// One-use weights use the L2 evict-first policy.
__device__ __forceinline__ void bulk_load_stream(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(
          dst),
      "l"(src), "r"(bytes), "r"(bar), "l"(policy)
      : "memory");
#else
  __trap();
#endif
}

// Grouped rasterization of a tile index over an n_tiles x m_tiles grid:
// consecutive tiles walk `group` n-tiles for one m-tile before moving on, so
// co-resident CTAs share weight and activation tiles in L2.
__device__ __forceinline__ void tile_raster(int tile, int n_tiles, int m_tiles, int group, int& nt, int& mt) {
  const int per_group = group * m_tiles;
  const int gid = tile / per_group;
  const int first_n = gid * group;
  const int gsize = min(n_tiles - first_n, group);
  const int local = tile - gid * per_group;
  nt = first_n + local % gsize;
  mt = local / gsize;
}

// Per-source view of one super-atom (16 rows x 64 k) as seen by one lane:
// kLoads 16-byte plane loads at `kPlaneBytes` strides, then `fragments`
// produces the A fragment of atom j (k columns 16j..16j+15) for all four j.
// Both sources yield the same registers for the same weights, so WeightTraits
// kernels produce matching outputs.
template <WeightSource S>
struct WeightTraits;

template <>
struct WeightTraits<WeightSource::kNF12> {
  static constexpr int kLoads = 3;
  static constexpr int kSuperAtomBytes = nf12::kSuperAtomBytes;

  // plane 0 = metas of atoms 0..3; plane 1 = (L0, L1) of atoms 0, 1;
  // plane 2 = (L0, L1) of atoms 2, 3. Must be called by the full warp: the
  // escape test is one warp vote per super-atom rather than one per group.
  __device__ __forceinline__ static void fragments(const uint4 (&p)[kLoads], uint32_t base_x4,
                                                   const uint4* __restrict__ escapes, uint32_t (&a)[4][4]) {
    const Group g0{p[0].x, p[1].x, p[1].y};
    const Group g1{p[0].y, p[1].z, p[1].w};
    const Group g2{p[0].z, p[2].x, p[2].y};
    const Group g3{p[0].w, p[2].z, p[2].w};
    decode_bulk(g0, base_x4, a[0]);
    decode_bulk(g1, base_x4, a[1]);
    decode_bulk(g2, base_x4, a[2]);
    decode_bulk(g3, base_x4, a[3]);
    const bool esc = (p[0].x == 0u) | (p[0].y == 0u) | (p[0].z == 0u) | (p[0].w == 0u);
    if (__any_sync(0xffffffffu, esc)) {
      if (esc) {
        decode_group_any_lane(g0, base_x4, escapes, a[0]);
        decode_group_any_lane(g1, base_x4, escapes, a[1]);
        decode_group_any_lane(g2, base_x4, escapes, a[2]);
        decode_group_any_lane(g3, base_x4, escapes, a[3]);
      }
    }
  }
};

template <>
struct WeightTraits<WeightSource::kBF16T> {
  static constexpr int kLoads = 4;
  static constexpr int kSuperAtomBytes = kSuperAtomBytesBf16t;

  __device__ __forceinline__ static void fragments(const uint4 (&p)[kLoads], uint32_t, const uint4* __restrict__,
                                                   uint32_t (&a)[4][4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      a[j][0] = p[j].x;
      a[j][1] = p[j].y;
      a[j][2] = p[j].z;
      a[j][3] = p[j].w;
    }
  }
};

}  // namespace nf12::detail
