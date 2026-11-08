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
// tcgen05 PTX wrappers, exposed only in architecture-specific images.
// Other targets leave NF12_TC_DEVICE undefined.
#pragma once

#include <cstdint>

#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) &&                                   \
    (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || __CUDA_ARCH_FAMILY_SPECIFIC__ == 1030 ||                    \
     __CUDA_ARCH_FAMILY_SPECIFIC__ == 1100)
#define NF12_TC_DEVICE 1
#endif

#if defined(NF12_TC_DEVICE)

namespace nf12::detail::tc {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void prefetch_l1(const void* p) {
  asm volatile("prefetch.global.L1 [%0];" ::"l"(p));
}

__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
  return v;
}

// Each register i holds matrix i's MMA fragment (row lane/4, columns
// 2(lane%4)..+1). stmatrix.trans writes transposed row j to lane 8i + j's address.
__device__ __forceinline__ void stmatrix_x4_trans(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2,
                                                  uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r0),
               "r"(r1), "r"(r2), "r"(r3)
               : "memory");
}

// Round-to-nearest-even of two floats into packed bf16 (lo in bits 0-15).
__device__ __forceinline__ uint32_t bf16x2_rn(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}


__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

// Shared::cluster address of the same smem offset in CTA `rank` of the cluster.
__device__ __forceinline__ uint32_t map_to_rank(uint32_t addr, uint32_t rank) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(addr), "r"(rank));
  return remote;
}

// Arrive at the same barrier offset in CTA `rank`. kRelease selects
// release.cluster semantics, retained for the coupled-ring pipeline.
// Otherwise use the lower-overhead CTA-scope arrival.
template <bool kRelease>
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t bar, uint32_t rank) {
  if constexpr (kRelease)
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(map_to_rank(bar, rank))
                 : "memory");
  else
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(map_to_rank(bar, rank)) : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

// Suspend until phase completion instead of spinning, which can lower SM
// clocks on power-limited parts.
template <bool kCluster = false>
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  constexpr uint32_t kSuspendNs = 1000000;
  if constexpr (kCluster) {
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1, %2;\n"
        "@P1 bra WAIT_DONE;\n"
        "bra WAIT_LOOP;\n"
        "WAIT_DONE:\n"
        "}\n" ::"r"(bar),
        "r"(parity), "n"(kSuspendNs)
        : "memory");
  } else {
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
        "@P1 bra WAIT_DONE;\n"
        "bra WAIT_LOOP;\n"
        "WAIT_DONE:\n"
        "}\n" ::"r"(bar),
        "r"(parity), "n"(kSuspendNs)
        : "memory");
  }
}


__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}


__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}

__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}

// 3D tensor-map tile load (coordinates innermost first).
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const void* tmap, int32_t c0, int32_t c1, int32_t c2,
                                            uint32_t bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
      "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "l"(policy)
      : "memory");
}

// 3D tensor-map tile prefetch into L2 (no shared memory, no barrier).
__device__ __forceinline__ void tma_prefetch_l2_3d(const void* tmap, int32_t c0, int32_t c1, int32_t c2) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(tmap), "r"(c0), "r"(c1),
               "r"(c2)
               : "memory");
}

// 2D tensor-map tile load (coordinates innermost first) into this CTA's
// shared memory. With kCtas == 2, `bar` may be a shared::cluster address of
// the barrier in either CTA of the pair (completion goes to that CTA).
template <int kCtas>
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const void* tmap, int32_t c0, int32_t c1, uint32_t bar,
                                            uint64_t policy) {
  if constexpr (kCtas == 2)
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
        ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
        "l"(tmap), "r"(c0), "r"(c1), "r"(bar), "l"(policy)
        : "memory");
  else
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
        "l"(tmap), "r"(c0), "r"(c1), "r"(bar), "l"(policy)
        : "memory");
}

// TMA stores omit out-of-bounds elements and join this thread's bulk async group.
__device__ __forceinline__ void tma_store_2d(const void* tmap, uint32_t src, int32_t c0, int32_t c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(tmap), "r"(c0),
               "r"(c1), "r"(src)
               : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
// Waits until this thread's committed bulk groups have finished reading
// shared memory (the source may be overwritten).
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
// Waits until this thread's committed bulk groups have completed.
__device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }

// 16-byte global -> shared copy through L2 only (cp.async.cg).
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, uint64_t policy) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "l"(policy)
               : "memory");
}

// Arrive once prior cp.async copies complete, using the initialized arrival count.
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void prefetch_tmap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}


__device__ __forceinline__ void st_release64(uint64_t* p, uint64_t v) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint64_t ld_acquire64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void named_bar_sync(uint32_t id, uint32_t threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}


template <uint32_t kCols, int kCtas>
__device__ __forceinline__ void tmem_alloc(uint32_t dst_smem) {
  if constexpr (kCtas == 2) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_smem), "n"(kCols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
  } else {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_smem), "n"(kCols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
}

template <uint32_t kCols, int kCtas>
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr) {
  if constexpr (kCtas == 2)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(taddr), "n"(kCols) : "memory");
  else
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "n"(kCols) : "memory");
}

__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

// Makes this thread's generic-proxy shared-memory writes visible to the async
// proxy (tcgen05.mma operand reads) once the thread later synchronizes.
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Each register i holds matrix i's MMA fragment (row lane/4, columns
// 2(lane%4)..+1). stmatrix writes its 16-byte row j to lane 8i + j's address.
__device__ __forceinline__ void stmatrix_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r0), "r"(r1),
               "r"(r2), "r"(r3)
               : "memory");
}

// Arrives once on `bar` when every previously issued tcgen05.mma of this
// thread has completed (including its operand reads). For a CTA pair the
// arrival is multicast to the barrier at the same offset in both CTAs.
template <int kCtas>
__device__ __forceinline__ void tc_commit(uint32_t bar) {
  if constexpr (kCtas == 2)
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
            bar),
        "h"(uint16_t(3))
        : "memory");
  else
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar)
                 : "memory");
}

// D[tmem] (+)= A[tmem] * B[smem desc]^T, kind::f16. With a CTA pair, A/D rows
// 0-127 live in the leader's TMEM and 128-255 in the peer's (same address).
template <int kCtas>
__device__ __forceinline__ void tc_mma_ts(uint32_t d, uint32_t a, uint64_t b_desc, uint32_t idesc,
                                          uint32_t accumulate) {
  if constexpr (kCtas == 2)
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}\n" ::"r"(d),
        "r"(a), "l"(b_desc), "r"(idesc), "r"(accumulate)
        : "memory");
  else
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n" ::"r"(d),
        "r"(a), "l"(b_desc), "r"(idesc), "r"(accumulate)
        : "memory");
}

// 16 TMEM lanes x 32 columns. Register 4j+i of lane l lands at
// (row l/4 + 8*(i&1), column 8j + 4*(i>>1) + l%4): exactly the mma.sync
// m16n8k16 A fragment {a0, a1, a2, a3} of atom j, which decode produces.
__device__ __forceinline__ void tmem_st_16x128b_x8(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.16x128b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15, %16};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
      : "memory");
}


// D[tmem] (+)= A[smem desc] * B[smem desc]^T, BF16 -> FP32.
// CTA pairs hold A/D rows 0-127 in the leader and 128-255 in the peer at
// equal shared/TMEM offsets. Each CTA holds half of B's N rows.
template <int kCtas>
__device__ __forceinline__ void tc_mma_ss(uint32_t d, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
                                          uint32_t accumulate) {
  if constexpr (kCtas == 2)
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(d),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate)
        : "memory");
  else
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(d),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate)
        : "memory");
}

// 16 TMEM lanes x 32 columns. Register 4i+r of lane l holds
// (row l/4 + 8*(r>>1), column 8i + 2*(l%4) + (r&1)): the mma.sync fragment
// layout with rows = TMEM lanes.
__device__ __forceinline__ void tmem_ld_16x256b_x4(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr)
      : "memory");
}

// Tie both load-result arrays to the wait so the compiler cannot use them
// before tcgen05.wait::ld completes.
__device__ __forceinline__ void tc_wait_ld(uint32_t (&a)[16], uint32_t (&b)[16]) {
  asm volatile("tcgen05.wait::ld.sync.aligned;"
               : "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3]), "+r"(a[4]), "+r"(a[5]), "+r"(a[6]), "+r"(a[7]),
                 "+r"(a[8]), "+r"(a[9]), "+r"(a[10]), "+r"(a[11]), "+r"(a[12]), "+r"(a[13]), "+r"(a[14]), "+r"(a[15])
               :
               : "memory");
  asm volatile(""
               : "+r"(b[0]), "+r"(b[1]), "+r"(b[2]), "+r"(b[3]), "+r"(b[4]), "+r"(b[5]), "+r"(b[6]), "+r"(b[7]),
                 "+r"(b[8]), "+r"(b[9]), "+r"(b[10]), "+r"(b[11]), "+r"(b[12]), "+r"(b[13]), "+r"(b[14]), "+r"(b[15])
               :
               : "memory");
}

// 32 TMEM lanes x 32 columns: lane l of the warp gets its lane's columns 0..31.
__device__ __forceinline__ void tmem_ld_32x32b_x32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr)
      : "memory");
}

// Shared-memory matrix descriptor: K-major, SWIZZLE_128B (128-byte rows of 64
// bf16, 8-row core groups 1024 bytes apart). LBO is unused for swizzled
// K-major operands whose per-instruction K fits in one swizzle row.
__device__ __forceinline__ uint64_t smem_desc_sw128(uint32_t addr) {
  return (uint64_t((addr >> 4) & 0x3fffu)) | (uint64_t(1) << 16) | (uint64_t(1024 >> 4) << 32) |
         (uint64_t(1) << 46) | (uint64_t(2) << 61);
}

// Instruction descriptor, kind::f16: FP32 D, BF16 A and B, both K-major.
__device__ __forceinline__ uint32_t idesc_bf16(uint32_t m, uint32_t n) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((n >> 3) << 17) | ((m >> 4) << 24);
}

}  // namespace nf12::detail::tc

#endif  // NF12_TC_DEVICE
