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
// SM120 PTX wrappers. setmaxnreg requires sm_120a/sm_120f/sm_121a.
// Other targets leave NF12_SM120_DEVICE undefined and expose no wrappers.
#pragma once

#include <cstdint>

#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
    (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200 || __CUDA_ARCH_FAMILY_SPECIFIC__ == 1210)
#define NF12_SM120_DEVICE 1
#endif

#if defined(NF12_SM120_DEVICE)

namespace nf12::detail::sm120 {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
  return v;
}

__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr));
  return v;
}

__device__ __forceinline__ void st_shared32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}

// Predicated 16-byte shared/global loads preserve a[0..3] when pred == 0
// without splitting the surrounding MMA basic block.
__device__ __forceinline__ void lds128_if(uint32_t (&a)[4], uint32_t addr, uint32_t pred) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %5, 0;\n@p ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n}\n"
      : "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
      : "r"(addr), "r"(pred));
}

__device__ __forceinline__ void ldg128_if(uint32_t (&a)[4], const void* ptr, uint32_t pred) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %5, 0;\n@p ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];\n}\n"
      : "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
      : "l"(ptr), "r"(pred));
}

// Non-volatile MMA has no side effects, permitting decode/MMA interleaving.
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t movmatrix_trans(uint32_t v) {
  uint32_t r;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(r) : "r"(v));
  return r;
}

// Round-to-nearest-even of two floats into packed bf16 (lo in bits 0-15).
__device__ __forceinline__ uint32_t bf16x2_rn(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

template <uint32_t kRegs>
__device__ __forceinline__ void reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(kRegs));
}
template <uint32_t kRegs>
__device__ __forceinline__ void reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kRegs));
}


__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

// Order prior generic-proxy shared accesses before async-proxy accesses.
// An mbarrier arrive alone does not prevent in-flight loads from reading a
// released slot after the next TMA overwrites it.
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Arrive on `bar` once all of this thread's prior cp.async copies have
// landed. noinc: the arrival is one of the barrier's initialized count.
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ bool mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_try_wait(bar, parity)) {
  }
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

__device__ __forceinline__ void prefetch_tmap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

// Tensor-map tile loads (coordinates innermost first) completing on `bar`.
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const void* tmap, int32_t c0, int32_t c1, uint32_t bar,
                                            uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
      "l"(tmap), "r"(c0), "r"(c1), "r"(bar), "l"(policy)
      : "memory");
}

__device__ __forceinline__ void tma_load_3d(uint32_t dst, const void* tmap, int32_t c0, int32_t c1, int32_t c2,
                                            uint32_t bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
      "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "l"(policy)
      : "memory");
}

// Four rows (r0..r3, each one box of the 2D tensor map, whose box height must
// be 1) of a 2D tensor gathered into consecutive box-sized slots at `dst`,
// completing on `bar`. Swizzling follows the shared-memory address, so four
// 128-byte rows at a 512-byte aligned `dst` land exactly as rows 4i..4i+3 of
// a SWIZZLE_128B tile would.
__device__ __forceinline__ void tma_gather4(uint32_t dst, const void* tmap, int32_t c0, int32_t r0, int32_t r1,
                                            int32_t r2, int32_t r3, uint32_t bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;" ::"r"(dst),
      "l"(tmap), "r"(c0), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(bar), "l"(policy)
      : "memory");
}

// 16-byte cp.async with an L2 eviction policy.
__device__ __forceinline__ void cp_async_16_hint(uint32_t dst, const void* src, uint64_t policy) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "l"(policy)
               : "memory");
}

// src_bytes = 0 zero-fills 16 bytes without reading `src`.
__device__ __forceinline__ void cp_async_16_zfill(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
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

}  // namespace nf12::detail::sm120

#endif  // NF12_SM120_DEVICE
