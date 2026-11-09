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
// SM90 PTX wrappers. wgmma and setmaxnreg require sm_90a. Other targets
// leave NF12_SM90_DEVICE undefined and expose no wrappers.
#pragma once

#include <cstdint>

#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define NF12_SM90_DEVICE 1
#endif

#if defined(NF12_SM90_DEVICE)

namespace nf12::detail::sm90 {

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
// without splitting the surrounding basic block.
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

// Orders this thread's prior generic-proxy shared-memory reads before later
// async-proxy writes (TMA) into a slot released by a following arrive.
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Arrive on `bar` once all of this thread's prior cp.async copies have
// landed. noinc: the arrival is one of the barrier's initialized count.
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

// The suspend-time hint parks waiting warps rather than polling and
// consuming issue slots shared with consumers.
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nwait:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n"
      "@!p bra wait;\n}\n" ::"r"(bar),
      "r"(parity), "r"(0x989680)
      : "memory");
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

// 16-byte cp.async with an L2 eviction policy.
__device__ __forceinline__ void cp_async_16_hint(uint32_t dst, const void* src, uint64_t policy) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "l"(policy)
               : "memory");
}

// src_bytes = 0 zero-fills 16 bytes without reading `src`.
__device__ __forceinline__ void cp_async_16_zfill(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}

__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
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


// Orders the warpgroup's prior register writes (A fragments, accumulators)
// before the following wgmma.mma_async reads them.
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}

// Pin values here to prevent accumulator reads moving before completion
// waits or A-fragment writes moving before waits that release their registers.
__device__ __forceinline__ void reg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void reg_fence(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void reg_fence(uint64_t& r) { asm volatile("" : "+l"(r)::"memory"); }

// Shared-memory matrix descriptor of a K-major SWIZZLE_128B operand tile
// (rows of 128 bytes, 8-row swizzle atoms 1024 bytes apart): start address
// bits 0-13, leading byte offset 16 (unused by K-major swizzled layouts),
// stride byte offset 1024 (between 8-row atoms), layout type 1 = 128B swizzle.
// `addr` may sit 32/64/96 bytes into a 1024-byte aligned atom for a k16 step.
// The base offset field stays 0.
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) {
  return (static_cast<uint64_t>(1) << 62) | (static_cast<uint64_t>(1024 >> 4) << 32) |
         (static_cast<uint64_t>(16 >> 4) << 16) | static_cast<uint64_t>((addr & 0x3ffffu) >> 4);
}

// D[64 x N] = A[64 x 16] * B[16 x N] + (scale_d ? D : 0), BF16 -> FP32.
// Warp q supplies A rows 16q..16q+15 as one m16k16 mma.sync fragment.
// B uses the K-major shared-memory descriptor. d[j] is warp q's mma.sync
// C fragment for columns 8j..8j+7.
template <int N>
struct Wgmma;

template <>
struct Wgmma<32> {
  __device__ __forceinline__ static void mma(float (&d)[4][4], const uint32_t (&a)[4], uint64_t desc, uint32_t scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
        "{%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};

template <>
struct Wgmma<64> {
  __device__ __forceinline__ static void mma(float (&d)[8][4], const uint32_t (&a)[4], uint64_t desc, uint32_t scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};

template <>
struct Wgmma<96> {
  __device__ __forceinline__ static void mma(float (&d)[12][4], const uint32_t (&a)[4], uint64_t desc, uint32_t scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %53, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, "
        "{%48, %49, %50, %51}, %52, p, 1, 1, 0;\n}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
          "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};

template <>
struct Wgmma<128> {
  __device__ __forceinline__ static void mma(float (&d)[16][4], const uint32_t (&a)[4], uint64_t desc, uint32_t scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
          "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
          "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
          "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};

template <>
struct Wgmma<192> {
  __device__ __forceinline__ static void mma(float (&d)[24][4], const uint32_t (&a)[4], uint64_t desc, uint32_t scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %101, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, "
        "{%96, %97, %98, %99}, %100, p, 1, 1, 0;\n}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
          "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
          "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
          "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]),
          "+f"(d[16][0]), "+f"(d[16][1]), "+f"(d[16][2]), "+f"(d[16][3]), "+f"(d[17][0]), "+f"(d[17][1]), "+f"(d[17][2]), "+f"(d[17][3]),
          "+f"(d[18][0]), "+f"(d[18][1]), "+f"(d[18][2]), "+f"(d[18][3]), "+f"(d[19][0]), "+f"(d[19][1]), "+f"(d[19][2]), "+f"(d[19][3]),
          "+f"(d[20][0]), "+f"(d[20][1]), "+f"(d[20][2]), "+f"(d[20][3]), "+f"(d[21][0]), "+f"(d[21][1]), "+f"(d[21][2]), "+f"(d[21][3]),
          "+f"(d[22][0]), "+f"(d[22][1]), "+f"(d[22][2]), "+f"(d[22][3]), "+f"(d[23][0]), "+f"(d[23][1]), "+f"(d[23][2]), "+f"(d[23][3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};

template <>
struct Wgmma<256> {
  __device__ __forceinline__ static void mma(float (&d)[32][4], const uint32_t (&a)[4], uint64_t desc, uint32_t scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
        "{%128, %129, %130, %131}, %132, p, 1, 1, 0;\n}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
          "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
          "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
          "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]),
          "+f"(d[16][0]), "+f"(d[16][1]), "+f"(d[16][2]), "+f"(d[16][3]), "+f"(d[17][0]), "+f"(d[17][1]), "+f"(d[17][2]), "+f"(d[17][3]),
          "+f"(d[18][0]), "+f"(d[18][1]), "+f"(d[18][2]), "+f"(d[18][3]), "+f"(d[19][0]), "+f"(d[19][1]), "+f"(d[19][2]), "+f"(d[19][3]),
          "+f"(d[20][0]), "+f"(d[20][1]), "+f"(d[20][2]), "+f"(d[20][3]), "+f"(d[21][0]), "+f"(d[21][1]), "+f"(d[21][2]), "+f"(d[21][3]),
          "+f"(d[22][0]), "+f"(d[22][1]), "+f"(d[22][2]), "+f"(d[22][3]), "+f"(d[23][0]), "+f"(d[23][1]), "+f"(d[23][2]), "+f"(d[23][3]),
          "+f"(d[24][0]), "+f"(d[24][1]), "+f"(d[24][2]), "+f"(d[24][3]), "+f"(d[25][0]), "+f"(d[25][1]), "+f"(d[25][2]), "+f"(d[25][3]),
          "+f"(d[26][0]), "+f"(d[26][1]), "+f"(d[26][2]), "+f"(d[26][3]), "+f"(d[27][0]), "+f"(d[27][1]), "+f"(d[27][2]), "+f"(d[27][3]),
          "+f"(d[28][0]), "+f"(d[28][1]), "+f"(d[28][2]), "+f"(d[28][3]), "+f"(d[29][0]), "+f"(d[29][1]), "+f"(d[29][2]), "+f"(d[29][3]),
          "+f"(d[30][0]), "+f"(d[30][1]), "+f"(d[30][2]), "+f"(d[30][3]), "+f"(d[31][0]), "+f"(d[31][1]), "+f"(d[31][2]), "+f"(d[31][3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};


__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
// Cluster index and cluster count along x (a launch without clusters has
// one-CTA clusters: blockIdx.x and gridDim.x).
__device__ __forceinline__ uint32_t cluster_id_x() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t nclusters_x() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t cluster_nctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r));
  return r;
}

// The shared::cluster address of shared::cta address `addr` in CTA `rank`.
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}

// Arrive on another CTA's cluster-addressed mbarrier with release.cta
// semantics. The caller's wgmma reads of the released stage must be complete.
__device__ __forceinline__ void mbar_arrive_remote(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(bar) : "memory");
}

// 16 bytes to any cluster CTA's shared memory (shared::cluster address, see mapa).
__device__ __forceinline__ void st_cluster128(uint32_t addr, const float (&v)[4]) {
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "f"(v[0]), "f"(v[1]), "f"(v[2]),
               "f"(v[3])
               : "memory");
}

// Cluster-wide barrier ordering all prior memory operations before later ones.
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Bulk copy of `bytes` (multiple of 16) from this CTA's shared memory to
// shared::cluster address `dst` (another CTA's), completing as transaction
// bytes on mbarrier `bar` (shared::cluster address in the destination CTA).
__device__ __forceinline__ void bulk_copy_to_cluster(uint32_t dst, uint32_t src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "r"(src), "r"(bytes), "r"(bar)
               : "memory");
}

// Four 8x8 b16 matrices from MMA fragments. r[i] in lane l holds matrix i's
// row l / 4, columns 2 (l % 4) and +1. Lane l supplies the address of row
// l % 8 in matrix l / 8.
__device__ __forceinline__ void stmatrix_x4(uint32_t addr, const uint32_t (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r[0]), "r"(r[1]),
               "r"(r[2]), "r"(r[3])
               : "memory");
}

// Shared-memory matrix descriptor of a K-major operand without swizzle:
// 8x8 core matrices of 128 contiguous bytes (rows of 16 bytes), `lbo` bytes
// between the two core matrices of a k16 step along K, `sbo` bytes between
// 8-row groups.
__device__ __forceinline__ uint64_t desc_interleave(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (static_cast<uint64_t>(sbo >> 4) << 32) | (static_cast<uint64_t>(lbo >> 4) << 16) |
         static_cast<uint64_t>((addr & 0x3ffffu) >> 4);
}

// D[64 x 256] (+)= A[64 x 16] * B[16 x 256], both from shared-memory
// descriptors, with D layout as Wgmma<256>.
__device__ __forceinline__ void wgmma_ss_n256(float (&d)[32][4], uint64_t desc_a, uint64_t desc_b, uint32_t scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
      "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
      "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
      "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
      "%128, %129, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
        "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
        "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
        "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]),
        "+f"(d[16][0]), "+f"(d[16][1]), "+f"(d[16][2]), "+f"(d[16][3]), "+f"(d[17][0]), "+f"(d[17][1]), "+f"(d[17][2]), "+f"(d[17][3]),
        "+f"(d[18][0]), "+f"(d[18][1]), "+f"(d[18][2]), "+f"(d[18][3]), "+f"(d[19][0]), "+f"(d[19][1]), "+f"(d[19][2]), "+f"(d[19][3]),
        "+f"(d[20][0]), "+f"(d[20][1]), "+f"(d[20][2]), "+f"(d[20][3]), "+f"(d[21][0]), "+f"(d[21][1]), "+f"(d[21][2]), "+f"(d[21][3]),
        "+f"(d[22][0]), "+f"(d[22][1]), "+f"(d[22][2]), "+f"(d[22][3]), "+f"(d[23][0]), "+f"(d[23][1]), "+f"(d[23][2]), "+f"(d[23][3]),
        "+f"(d[24][0]), "+f"(d[24][1]), "+f"(d[24][2]), "+f"(d[24][3]), "+f"(d[25][0]), "+f"(d[25][1]), "+f"(d[25][2]), "+f"(d[25][3]),
        "+f"(d[26][0]), "+f"(d[26][1]), "+f"(d[26][2]), "+f"(d[26][3]), "+f"(d[27][0]), "+f"(d[27][1]), "+f"(d[27][2]), "+f"(d[27][3]),
        "+f"(d[28][0]), "+f"(d[28][1]), "+f"(d[28][2]), "+f"(d[28][3]), "+f"(d[29][0]), "+f"(d[29][1]), "+f"(d[29][2]), "+f"(d[29][3]),
        "+f"(d[30][0]), "+f"(d[30][1]), "+f"(d[30][2]), "+f"(d[30][3]), "+f"(d[31][0]), "+f"(d[31][1]), "+f"(d[31][2]), "+f"(d[31][3])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

// Narrow shared-A tiles use the same accumulator layout and operand majors.
__device__ __forceinline__ void wgmma_ss_n64(float (&d)[8][4], uint64_t desc_a, uint64_t desc_b, uint32_t scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

__device__ __forceinline__ void wgmma_ss_n128(float (&d)[16][4], uint64_t desc_a, uint64_t desc_b, uint32_t scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "%64, %65, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
        "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
        "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
        "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

}  // namespace nf12::detail::sm90

#endif  // NF12_SM90_DEVICE
