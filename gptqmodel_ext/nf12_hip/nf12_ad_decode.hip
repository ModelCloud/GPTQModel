// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12-AD (NyoomFloat12) decode for AMD RDNA4 (gfx1200 / gfx1201 / gfx1250).
//
// Algorithm: "NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit
// Weight Compression", Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).
// Together AI: https://www.together.ai/
//
// Original NF12 reference implementation and CUDA kernels:
//   https://github.com/dphnAI/nf12
//   Copyright (c) 2026 AlpinDale and the nf12 contributors.
//   Licensed under the Apache License, Version 2.0.
//
// This HIP port was written for GPT-QModel (ModelCloud.ai) and consumes the
// same NF12-AD byte stream as the CUDA kernels (docs/SPEC.md in
// gptqmodel_ext/nf12). It lets NF12-AD checkpoints run on AMD RDNA4 GPUs
// without an NVIDIA card or the CUDA toolchain.
//
// Design
// ------
// RDNA4 has no `mma.sync.m16n8k16`, and wgmma/tcgen05/TMA do not exist there
// either. Rather than invent a different accumulation order and break SPEC.md
// 6.2 (NF12 output must be identical to the bf16t twin per configuration),
// this port decodes NF12 -> exact BF16 with the same super-atom walk the CUDA
// decoder uses and hands the result to rocBLAS. That is SPEC.md 6.5's
// "decode-then-GEMM" route: decode is exact, so the result equals the same
// BF16 GEMM on the exactly decoded weights. `nf12_ad_decode` output is
// bit-identical to the CUDA `nf12.decode` output.
// ---------------------------------------------------------------------------
#include "nf12_ad_format_hip.cuh"

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace nf12_hip {
namespace {

// ---------------------------------------------------------------------------
// Decode kernel: NF12 -> BF16 (exact)
// ---------------------------------------------------------------------------
constexpr int kDecodeSuperAtomsPerWarp = 2;

// Interleave warps along K so their row-major stores form contiguous segments.
template <int kPerWarp>
__device__ __forceinline__ int64_t warp_super_atom(int i) {
  return (int64_t(blockIdx.x) * kPerWarp + i) * kWarpsPerBlock + (threadIdx.x / kWarpSize);
}

template <int kPerWarp>
int64_t grid_blocks(int64_t num_sa) {
  const int64_t per_block = int64_t(kPerWarp) * kWarpsPerBlock;
  return (num_sa + per_block - 1) / per_block;
}

// Largest token count the fused kernel handles. Above this, decoding the weight
// once and running a real GEMM wins (SPEC.md 6.5), so callers fall back.
constexpr int kFusedMaxM = 16;

template <int kPerWarp>
__device__ __forceinline__ void decode_rows(const uint4* __restrict__ packed,
                                            const uint4* __restrict__ escapes,
                                            uint32_t base_x4, int64_t num_sa,
                                            int64_t k_blocks,
                                            __hip_bfloat16* __restrict__ out, int64_t ldo) {
  const int lane = threadIdx.x % kWarpSize;
  uint4 planes[kPerWarp][3];
#pragma unroll
  for (int i = 0; i < kPerWarp; ++i) {
    const int64_t sa = warp_super_atom<kPerWarp>(i);
    if (sa < num_sa) {
      const uint4* src = packed + sa * kSuperAtomVecs + lane;
#pragma unroll
      for (int p = 0; p < 3; ++p) planes[i][p] = src[p * kPlaneVecs];
    }
  }
#pragma unroll
  for (int i = 0; i < kPerWarp; ++i) {
    const int64_t sa = warp_super_atom<kPerWarp>(i);
    if (sa >= num_sa) break;
    const uint4& m = planes[i][0];
    const uint4& l01 = planes[i][1];
    const uint4& l23 = planes[i][2];
    AtomRegs r;
    decode_group(Group{m.x, l01.x, l01.y}, base_x4, escapes, r[0]);
    decode_group(Group{m.y, l01.z, l01.w}, base_x4, escapes, r[1]);
    decode_group(Group{m.z, l23.x, l23.y}, base_x4, escapes, r[2]);
    decode_group(Group{m.w, l23.z, l23.w}, base_x4, escapes, r[3]);
    RowTile x;
    atoms_to_tile(r, x);
    // Undo the encoder's quad transpose to return to row-major.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int p = 0; p < 2; ++p) quad_transpose(x[h][p], lane);
    }
    store_row_tile(out, row_tile_offset(super_atom(sa, k_blocks), ldo, lane), ldo, x);
  }
}

template <int kPerWarp>
__global__ void __launch_bounds__(kThreadsPerBlock)
    decode_kernel(const uint4* __restrict__ packed, const uint4* __restrict__ escapes,
                  uint32_t base_x4, int64_t num_sa, int64_t k_blocks,
                  __hip_bfloat16* __restrict__ out, int64_t ldo) {
  decode_rows<kPerWarp>(packed, escapes, base_x4, num_sa, k_blocks, out, ldo);
}


// ---------------------------------------------------------------------------
// Fused small-M NF12-AD x BF16 GEMM
// ---------------------------------------------------------------------------
// Decode-then-GEMM moves 24 MiB (packed) + 32 MiB (dense write) + 32 MiB
// (dense read) = 88 MiB. A BF16 GEMM only needs 32 MiB. Nothing that
// materialises the dense weight can win at small M, so this kernel never
// builds it: the decoded super-atom stays in registers and is reduced against
// the activation tile in place.
//
// Ownership. Lane l of a warp holds super-atom row pair (g, g+8) with
// g = l >> 2 and t = l & 3. Its eight slots cover, per the SPEC 4.1 table,
// rows {g, g+8} x cols {2t, 2t+1, 2t+8, 2t+9} of each of the four atoms, i.e.
// 64 columns for 2 rows. Those 64 columns are exactly the super-atom's K span,
// so ONE warp owns the full dot product for its 2 rows and reduces with two
// xor-shuffles over t. No transpose, no shared-memory staging.
//
// Numerics (SPEC 6.2/6.3): FP32 accumulation, BF16 round-to-nearest on write.
// Each output element is summed in a fixed order -- the four j atom groups,
// then the t reduction -- so results are deterministic and match the
// decode-then-GEMM route to within the documented atol/rtol bound.
//
// Split-K inside the CTA, mirroring the CUDA small-M family (SPEC.md 7:
// "the small-M kernel splits K inside the CTA (warps summed in part order)").
//
// One CTA owns one 16-row output block. Its kWarpsPerBlock warps each walk a
// contiguous K slice, accumulate in FP32, then a single pass sums the per-warp
// partials in fixed warp order. Reducing inside the CTA keeps the whole
// reduction deterministic (SPEC 6.4) and costs one barrier instead of a
// second kernel launch.
//
// Why split K at all: with one warp owning the full K span a 4096-row matrix
// exposes only N/16 = 256 warps, each serialising 64 super-atoms. Splitting K
// eight ways multiplies the available parallelism by eight, which is what the
// RDNA4 part was missing.
//
// Decoded super-atoms are still reused across every token in flight, so the
// packed weight is read once per launch rather than once per token.
template <int kM>
__global__ void __launch_bounds__(kThreadsPerBlock)
    linear_kernel(const __hip_bfloat16* __restrict__ x, const uint4* __restrict__ packed,
                  const uint4* __restrict__ escapes, uint32_t base_x4, int64_t n,
                  int64_t k, __hip_bfloat16* __restrict__ out,
                  const __hip_bfloat16* __restrict__ bias) {
  // The warp-partial plane is sized for the tokens actually in flight, except
  // at kM == 1 where it is deliberately over-allocated. That kernel carries two
  // accumulators and no decode reuse, so left alone it runs eight CTAs per CU
  // and 256 concurrent packed streams, which thrashes the 64 MiB LLC:
  // 4096x4096 m=1 measured 28.6 us at eight CTAs per CU against 20.5 us at two,
  // and 2048x8192 28.8 us against 18.5 us. 32 KiB of shared memory is the
  // occupancy throttle; at kM >= 2 the plane is genuinely used and not padded.
  constexpr int kPartTokens = (kM == 1) ? 64 : kM;
  __shared__ float part[kWarpsPerBlock][kPartTokens][kSuperAtomRows];

  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int64_t n0 = int64_t(blockIdx.x) * kSuperAtomRows;
  const int64_t k_blocks = k / kSuperAtomCols;

  // Contiguous, equal K slices; the last warp takes any remainder.
  const int64_t kb_per = (k_blocks + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t kb_begin = int64_t(warp) * kb_per;
  int64_t kb_end = kb_begin + kb_per;
  if (kb_end > k_blocks) kb_end = k_blocks;

  // Every warp's slot is read by the final reduction, but a warp whose K slice
  // is empty (few super-atoms along K) never publishes. Zero first so those
  // slots contribute exactly nothing instead of uninitialised shared memory.
  for (int64_t idx = threadIdx.x; idx < kWarpsPerBlock * kM * kSuperAtomRows;
       idx += kThreadsPerBlock) {
    (&part[0][0][0])[idx] = 0.0f;
  }
  __syncthreads();

  float acc[kM][2];
#pragma unroll
  for (int mi = 0; mi < kM; ++mi) {
    acc[mi][0] = 0.0f;
    acc[mi][1] = 0.0f;
  }

  for (int64_t kb = kb_begin; kb < kb_end; ++kb) {
    const int64_t sa = (n0 / kSuperAtomRows) * k_blocks + kb;
    const uint4* src = packed + sa * kSuperAtomVecs + lane;
    uint4 planes[3];
#pragma unroll
    for (int p = 0; p < 3; ++p) planes[p] = src[p * kPlaneVecs];

    AtomRegs regs;
    decode_group(Group{planes[0].x, planes[1].x, planes[1].y}, base_x4, escapes, regs[0]);
    decode_group(Group{planes[0].y, planes[1].z, planes[1].w}, base_x4, escapes, regs[1]);
    decode_group(Group{planes[0].z, planes[2].x, planes[2].y}, base_x4, escapes, regs[2]);
    decode_group(Group{planes[0].w, planes[2].z, planes[2].w}, base_x4, escapes, regs[3]);

    // regs[j][i] = R_i of atom j (SPEC 4.1):
    //   R0 -> row g,   cols 2t, 2t+1      R1 -> row g+8, cols 2t, 2t+1
    //   R2 -> row g,   cols 2t+8, 2t+9    R3 -> row g+8, cols 2t+8, 2t+9
    // Packed BF16 dot, one v_dot2_f32_bf16 per two columns. This is the RDNA4
    // counterpart of the CUDA mma bf16 path: without it the inner loop is
    // scalar FMA and the kernel is compute-bound instead of bandwidth-bound.
    // (__builtin_amdgcn_fdot2 is FP16, not BF16 -- using it here silently
    // produced garbage, so the explicit _f32_bf16 spelling matters.)
    const __hip_bfloat162* r = reinterpret_cast<const __hip_bfloat162*>(&regs[0][0]);

    // x is [M, K]; the super-atom covers 64 columns starting at kb*64.
    const __hip_bfloat162* xrow =
        reinterpret_cast<const __hip_bfloat162*>(x + kb * kSuperAtomCols);

    for (int mi = 0; mi < kM; ++mi) {
      const __hip_bfloat162* xp = xrow + mi * (k / 2) + t;
      const __hip_bfloat162 c0 = xp[0];    // cols 2t, 2t+1
      const __hip_bfloat162 c8 = xp[4];    // cols 2t+8, 2t+9
      const __hip_bfloat162 c16 = xp[8];
      const __hip_bfloat162 c24 = xp[12];
      const __hip_bfloat162 c32 = xp[16];
      const __hip_bfloat162 c40 = xp[20];
      const __hip_bfloat162 c48 = xp[24];
      const __hip_bfloat162 c56 = xp[28];

      // r[4*j + i] is R_i of atom j: row g uses i = 0 and 2, row g+8 uses
      // i = 1 and 3.
      acc[mi][0] = __builtin_amdgcn_fdot2_f32_bf16(c0, r[0], acc[mi][0], false);
      acc[mi][0] = __builtin_amdgcn_fdot2_f32_bf16(c8, r[2], acc[mi][0], false);
      acc[mi][1] = __builtin_amdgcn_fdot2_f32_bf16(c0, r[1], acc[mi][1], false);
      acc[mi][1] = __builtin_amdgcn_fdot2_f32_bf16(c8, r[3], acc[mi][1], false);

      acc[mi][0] = __builtin_amdgcn_fdot2_f32_bf16(c16, r[4], acc[mi][0], false);
      acc[mi][0] = __builtin_amdgcn_fdot2_f32_bf16(c24, r[6], acc[mi][0], false);
      acc[mi][1] = __builtin_amdgcn_fdot2_f32_bf16(c16, r[5], acc[mi][1], false);
      acc[mi][1] = __builtin_amdgcn_fdot2_f32_bf16(c24, r[7], acc[mi][1], false);

      acc[mi][0] = __builtin_amdgcn_fdot2_f32_bf16(c32, r[8], acc[mi][0], false);
      acc[mi][0] = __builtin_amdgcn_fdot2_f32_bf16(c40, r[10], acc[mi][0], false);
      acc[mi][1] = __builtin_amdgcn_fdot2_f32_bf16(c32, r[9], acc[mi][1], false);
      acc[mi][1] = __builtin_amdgcn_fdot2_f32_bf16(c40, r[11], acc[mi][1], false);

      acc[mi][0] = __builtin_amdgcn_fdot2_f32_bf16(c48, r[12], acc[mi][0], false);
      acc[mi][0] = __builtin_amdgcn_fdot2_f32_bf16(c56, r[14], acc[mi][0], false);
      acc[mi][1] = __builtin_amdgcn_fdot2_f32_bf16(c48, r[13], acc[mi][1], false);
      acc[mi][1] = __builtin_amdgcn_fdot2_f32_bf16(c56, r[15], acc[mi][1], false);
    }
  }

  // Collapse the four t lanes of this quad (fixed xor order 1 then 2), then
  // publish the two owned rows for every token to shared memory.
  for (int mi = 0; mi < kM; ++mi) {
    float hi = acc[mi][0];
    float lo = acc[mi][1];
    hi += __shfl_xor(hi, 1);
    hi += __shfl_xor(hi, 2);
    lo += __shfl_xor(lo, 1);
    lo += __shfl_xor(lo, 2);
    if (t == 0) {
      part[warp][mi][g] = hi;
      part[warp][mi][g + kSuperAtomRows / 2] = lo;
    }
  }
  __syncthreads();

  // Sum the warp partials in fixed warp order so the result is deterministic.
  const int64_t total = int64_t(kM) * kSuperAtomRows;
  for (int64_t idx = threadIdx.x; idx < total; idx += kThreadsPerBlock) {
    const int64_t mi = idx / kSuperAtomRows;
    const int64_t row = idx % kSuperAtomRows;
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) sum += part[w][mi][row];
    // Bias joins the FP32 reduction, so the result is rounded to BF16 once.
    // Adding it to the already-rounded BF16 output instead double-rounds:
    // on gfx1201 that measured 1.00 ULP / 1.7e-3 RMS relative error versus
    // 0.5 ULP / 7e-5 here, and the CUDA kernel also accumulates bias in FP32
    // before writing BF16 (csrc/linear/small_m.cu).
    if (bias != nullptr) sum += __bfloat162float(bias[n0 + row]);
    out[mi * n + n0 + row] = __float2bfloat16(sum);
  }
}

}  // namespace

}  // namespace nf12_hip

using namespace nf12_hip;

extern "C" {

// Decode a whole [N, K] NF12-AD matrix to row-major BF16.
hipError_t nf12_ad_decode(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                          int64_t n, int64_t k, void* out, int64_t ldo,
                          hipStream_t stream) {
  __hip_bfloat16* out_bf16 = reinterpret_cast<__hip_bfloat16*>(out);
  if (exp_base > uint32_t(kMaxExpBase)) return hipErrorInvalidValue;
  const int64_t num_sa = (n / kSuperAtomRows) * (k / kSuperAtomCols);
  const int64_t blocks = grid_blocks<kDecodeSuperAtomsPerWarp>(num_sa);
  if (blocks == 0) return hipSuccess;
  if (blocks > 0x7fffffffLL) return hipErrorInvalidValue;
  decode_kernel<kDecodeSuperAtomsPerWarp>
      <<<static_cast<unsigned>(blocks), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const uint4*>(packed), escapes, exp_base_x4(exp_base), num_sa,
          k / kSuperAtomCols, out_bf16, ldo);
  return hipGetLastError();
}

// Self-test: a real kernel launch stamps a sentinel so Python can prove that
// device code was compiled AND executed. A host-to-device memcpy would not prove
// either, and hipcc leaves __CUDA_ARCH__ undefined, so a stale architecture
// guard would silently emit an empty kernel -- this catches that failure mode.
namespace {
__global__ void selftest_kernel(uint32_t* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *out = 0x4E463132u;  // "NF12"
}
}  // namespace

hipError_t nf12_ad_selftest(uint32_t* out, hipStream_t stream) {
  selftest_kernel<<<1, 32, 0, stream>>>(out);
  return hipGetLastError();
}


// Fused small-M linear (decode + GEMM in one pass, no dense materialisation).
// `out` is [M, N] row-major BF16.
int nf12_ad_linear_fused_small_m_supported(int64_t m) { return m >= 1 && m <= kFusedMaxM; }


hipError_t nf12_ad_linear_fused(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                                const void* x, void* out, int64_t n, int64_t k,
                                int64_t m, const void* bias, hipStream_t stream) {
  if (exp_base > uint32_t(kMaxExpBase)) return hipErrorInvalidValue;
  if (m < 1 || m > kFusedMaxM) return hipErrorInvalidValue;

  // One CTA per 16-row output block: the block itself splits K across its
  // warps, so the grid does not divide by kWarpsPerBlock.
  const int64_t blocks = n / kSuperAtomRows;
  if (blocks == 0) return hipSuccess;
  if (blocks > 0x7fffffffLL) return hipErrorInvalidValue;

  // The token count is a template parameter so `acc` and the warp-partial
  // plane are compile-time sized: with a runtime `m` the accumulator array is
  // dynamically indexed, which keeps it out of VGPRs, and the shared plane is
  // sized for 16 tokens even when only one is in flight.
#define NF12_AD_FUSED_LAUNCH(K)                                                                \
  linear_kernel<K><<<static_cast<unsigned>(blocks), kThreadsPerBlock, 0, stream>>>(            \
      reinterpret_cast<const __hip_bfloat16*>(x), reinterpret_cast<const uint4*>(packed),       \
      escapes, exp_base_x4(exp_base), n, k, reinterpret_cast<__hip_bfloat16*>(out),            \
      reinterpret_cast<const __hip_bfloat16*>(bias))
  switch (m) {
    case 1: NF12_AD_FUSED_LAUNCH(1); break;
    case 2: NF12_AD_FUSED_LAUNCH(2); break;
    case 3: NF12_AD_FUSED_LAUNCH(3); break;
    case 4: NF12_AD_FUSED_LAUNCH(4); break;
    case 5: NF12_AD_FUSED_LAUNCH(5); break;
    case 6: NF12_AD_FUSED_LAUNCH(6); break;
    case 7: NF12_AD_FUSED_LAUNCH(7); break;
    case 8: NF12_AD_FUSED_LAUNCH(8); break;
    case 9: NF12_AD_FUSED_LAUNCH(9); break;
    case 10: NF12_AD_FUSED_LAUNCH(10); break;
    case 11: NF12_AD_FUSED_LAUNCH(11); break;
    case 12: NF12_AD_FUSED_LAUNCH(12); break;
    case 13: NF12_AD_FUSED_LAUNCH(13); break;
    case 14: NF12_AD_FUSED_LAUNCH(14); break;
    case 15: NF12_AD_FUSED_LAUNCH(15); break;
    default: NF12_AD_FUSED_LAUNCH(16); break;
  }
#undef NF12_AD_FUSED_LAUNCH
  return hipGetLastError();
}

}  // extern "C"
