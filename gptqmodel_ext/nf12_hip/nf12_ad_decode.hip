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

}  // extern "C"
