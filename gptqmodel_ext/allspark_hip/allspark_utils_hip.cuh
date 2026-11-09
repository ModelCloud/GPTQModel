// SPDX-FileCopyrightText: 2025 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// HIP/RDNA4 (gfx1200/gfx1201) device helpers for the AllSpark W8A16 GEMM.
//
// This header is intentionally free of torch/ATen includes: the ROCm pip SDK
// used to build this extension ships no thrust headers, so any translation unit
// that includes torch must stay host-only (see allspark_torch_hip.cpp).
//
// Porting notes (vs the CUDA Ampere kernel in gptqmodel_ext/allspark):
//   * cp.async          -> plain global->register loads + ds_write (gfx12 has no
//                          async copy to LDS; gfx1250 does, see hook below)
//   * ldmatrix          -> plain 16B LDS reads; the repack stores the exact WMMA
//                          fragment order so no lane permutation is needed
//   * mma.sync m16n8k16 -> v_wmma_f32_16x16x16_{f16,bf16}_w32_gfx12
//   * prmt.b32          -> v_perm_b32 (__builtin_amdgcn_perm); NOTE the AMD
//                          selector semantics are reversed vs NVIDIA
//   * __CUDA_ARCH__     -> __HIP_DEVICE_COMPILE__ + __gfx1200__/__gfx1201__
//
// The CUDA kernel guarded its entire body with `#if __CUDA_ARCH__ >= 800`.
// Under hipcc `__CUDA_ARCH__` is *undefined*, so those guards silently compile
// to empty kernels. That failure mode is rejected at compile time below and
// again at runtime (sentinel word written by the kernel, see allspark_hip.cu).

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <cstdint>

namespace allspark_hip {

// ---------------------------------------------------------------------------
// Architecture gate
// ---------------------------------------------------------------------------
// Only the RDNA4 wave32 WMMA path is implemented. gfx1250 exposes a different
// (wave32 but 16x16x32-capable) instruction set and is deliberately not wired
// up yet. It still has to compile so a family-wide `PYTORCH_ROCM_ARCH` build
// succeeds: the kernel body is a trap stub there and the host refuses to
// launch it (`allspark_hip_supported` accepts gfx1200/gfx1201 only).
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1200__) && \
    !defined(__gfx1201__) && !defined(__gfx1250__)
#error \
    "allspark_hip: only RDNA4 (gfx1200/gfx1201/gfx1250) targets are supported"
#endif

#if defined(__gfx1200__) || defined(__gfx1201__)
#define ALLSPARK_HIP_HAS_WMMA_GFX12 1
#else
#define ALLSPARK_HIP_HAS_WMMA_GFX12 0
#endif

// Wavefront size. RDNA4 (gfx1200/1201/1250) is wave32 only.
constexpr int kWarpSize = 32;

// 16x16x16 WMMA tile: every lane owns 8 A elements, 8 B elements and 8 C
// elements.
constexpr int kFragElems = 8;

typedef _Float16 f16_t;
typedef __bf16 bf16_t;

typedef f16_t v8f16 __attribute__((ext_vector_type(8)));
typedef bf16_t v8bf16 __attribute__((ext_vector_type(8)));
typedef float v8f32 __attribute__((ext_vector_type(8)));

// ---------------------------------------------------------------------------
// Kernel parameters (shape kept compatible with the CUDA AllSpark struct)
// ---------------------------------------------------------------------------
struct alignas(16) BlockTileSplitkParams {
  int Mtile;
  int Ntile;
  int SplitK;
  bool EnableFuse;
};

template <typename FType, typename QType>
struct SM8x_GEMM_W8A16_Splitk_Params {
  const FType* A_ptr;
  const QType* B_ptr;
  const FType* B_scale_ptr;
  const FType* B_zero_ptr;
  FType* C_ptr;
  int M;
  int N;
  int K;
  int N32;  // N padded to a multiple of 32 (packed-weight column count)
  int SplitK;
  // Scale/zero are per-output-channel (one group over all of K), so there is no
  // group index to track: the kernel reads B_scale_ptr[n] / B_zero_ptr[n] once.
  FType* C_split_ptr;       // non-fused split-K workspace (low precision)
  float* C_tmp_ptr;         // split-K fp32 partials (non-fused reduce)
  float* C32_ptr;           // when set, write fp32 accumulators here instead
  uint32_t* red_count_ptr;  // fused split-K counter (unused here)
};

// ---------------------------------------------------------------------------
// uint8 -> float16 / bfloat16 dequantisation
// ---------------------------------------------------------------------------
// GPTQ 8-bit codes are stored biased by 128. The CUDA kernel builds a half2 by
// splicing 0x64 into the exponent byte and subtracting 0x6480 (= 1024 + 128),
// which turns four `prmt.b32` into four cheap half2 ops.
//
// v_perm_b32 selector semantics (re-verified on gfx1201 hardware):
// the selector is a packed word with ONE FULL BYTE per output byte (not a
// 4-bit nibble as on NVIDIA prmt). For output byte i:
//   s = (selector >> (8*i)) & 0xFF
//   s in 0..3 -> src1.byte[s]
//   s in 4..7 -> src0.byte[s-4]
//   otherwise -> 0xFF (sign/out-of-range fill)
// So the two constants below expand to [a0, 0x64, a1, 0x64] / [a2, 0x64, a3,
// 0x64]; a nibble-based transliteration of the CUDA prmt constants would be
// wrong (it silently produced 0x0000ff00 instead of 0x4B0000a0).
__device__ __forceinline__ uint32_t perm_b32(uint32_t src0, uint32_t src1,
                                             uint32_t sel) {
  return __builtin_amdgcn_perm(src0, src1, sel);
}

union U32Half2 {
  uint32_t u;
  __half2 h2;
};

union U32BF162 {
  uint32_t u;
  __hip_bfloat162 b2;
};

// 4 uint8 -> 4 half, biased by 128. Returns two packed half2 words.
__device__ __forceinline__ void cvt_8bx4_to_16bx4_bias128(
    const uint32_t& idata, uint32_t& o0, uint32_t& o1) {
  constexpr uint32_t kSelLo = 0x00050004u;  // [a0, 0x64, a1, 0x64]
  constexpr uint32_t kSelHi = 0x00070006u;  // [a2, 0x64, a3, 0x64]
  constexpr uint32_t kMagic = 0x64806480u;  // half2(1152, 1152)

  U32Half2 v0, v1, m;
  m.u = kMagic;
  v0.u = perm_b32(idata, 0x64u, kSelLo);
  v1.u = perm_b32(idata, 0x64u, kSelHi);
  v0.h2 = __hsub2(v0.h2, m.h2);
  v1.h2 = __hsub2(v1.h2, m.h2);
  o0 = v0.u;
  o1 = v1.u;
}

// Pack two fp32 bit patterns into two bf16 (round to nearest even).
//
// gfx1200/gfx1201 have no v_cvt_pk_bf16_* instruction (the assembler rejects
// it), and __float2bfloat16() emits a NaN-aware compare + select per value,
// which cost ~200 extra ALU instructions across the kernel and made bf16 ~28%
// slower than fp16 on gfx1201 in the L2-resident case.
//
// The inputs here are always finite: (2^23 + code) * scale. So the NaN select
// is dead work and plain integer RNE is exact. Two adds produce the rounded
// high halves; one v_perm_b32 then gathers [a.b2, a.b3, b.b2, b.b3] into the
// packed bf16x2 word.
__device__ __forceinline__ uint32_t cvt_pk_bf16_rne(uint32_t u0, uint32_t u1) {
  const uint32_t a = u0 + 0x7FFFu + ((u0 >> 16) & 1u);
  const uint32_t b = u1 + 0x7FFFu + ((u1 >> 16) & 1u);
  // v_perm_b32: out byte i from src1.byte[s] if s in 0..3, else src0.byte[s-4].
  // Selector bytes low->high: 0x06 (src0.b2), 0x07 (src0.b3), 0x02 (src1.b2),
  // 0x03 (src1.b3) -> [a.b2, a.b3, b.b2, b.b3].
  return perm_b32(a, b, 0x03020706u);
}

// 4 uint8 -> 4 bfloat16, multiplied by `scale` in fp32 and rounded to nearest
// bf16. Doing the scale in fp32 halves the visible error versus the CUDA
// kernel's bf16 `__hmul2` (bf16 has only 8 mantissa bits, so this matters for
// the 2e-3 acceptance gate).
//
// `zero` is the dequantisation zero point **code** (AWQ), not a bias-128
// offset: (raw - 2^23 - zero) * scale is exactly (code - zero) * scale, and
// the constant 2^23 + zero stays exactly representable in fp32 for any 8-bit
// code, so this is one subtract plus one multiply per element with no
// cancellation loss.
constexpr float kTwoPow23 = 8388608.0f;
__device__ __forceinline__ void cvt_8bx4_to_16bx4_bf16_scaled(
    const uint32_t& idata, float scale, float zero, uint32_t& o0,
    uint32_t& o1) {
  // 0x4B000000 has bytes {00,00,00,4B}. Selector bytes, low to high:
  //   4+i (src0.byte i), 0 (src1.b0 = 00), 1 (src1.b1 = 00), 3 (src1.b3 = 4B)
  constexpr uint32_t kSel[4] = {0x03010004u, 0x03010005u, 0x03010006u,
                                0x03010007u};
  const float bias = kTwoPow23 + zero;
  float f[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t bits = perm_b32(idata, 0x4B000000u, kSel[i]);
    f[i] = (__uint_as_float(bits) - bias) * scale;
  }
  o0 = cvt_pk_bf16_rne(__float_as_uint(f[0]), __float_as_uint(f[1]));
  o1 = cvt_pk_bf16_rne(__float_as_uint(f[2]), __float_as_uint(f[3]));
}

// Back-compat shim: GPTQ 8-bit neutral zero point is 128.
__device__ __forceinline__ void cvt_8bx4_to_16bx4_bias128_bf16_scaled(
    const uint32_t& idata, float scale, uint32_t& o0, uint32_t& o1) {
  cvt_8bx4_to_16bx4_bf16_scaled(idata, scale, 128.0f, o0, o1);
}

// ---------------------------------------------------------------------------
// WMMA wrappers
// ---------------------------------------------------------------------------
// gfx1200/gfx1201: v_wmma_f32_16x16x16_{f16,bf16}_w32_gfx12 (3-argument form).
// Fragment layout discovered empirically on gfx1201 (probe/wmma_w8a16):
//   A row-distributed : lane l owns A[m = l%16][k = (l/16)*8 + j]
//   B col-distributed : lane l owns B[n = l%16][k = (l/16)*8 + j]
//   C column-distributed: lane l owns C[m = (l/16)*8 + j][n = l%16]
// This differs from CUDA m16n8k16 and is why the repack below is re-derived
// rather than copied.
__device__ __forceinline__ v8f32 wmma_f16(v8f16 a, v8f16 b, v8f32 c) {
#if ALLSPARK_HIP_HAS_WMMA_GFX12
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
  // gfx1250 has no wmma-128b-insts. The host never launches this path; trap
  // rather than return a wrong result if a caller bypasses the arch gate.
  (void)a;
  (void)b;
  __builtin_trap();
  return c;
#endif
}

__device__ __forceinline__ v8f32 wmma_bf16(v8bf16 a, v8bf16 b, v8f32 c) {
#if ALLSPARK_HIP_HAS_WMMA_GFX12
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#else
  (void)a;
  (void)b;
  __builtin_trap();
  return c;
#endif
}

}  // namespace allspark_hip
