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
// Issue loads for several super-atoms before use to overlap memory latency.
// Warp accesses span 512 tiled bytes or 64 contiguous row-major bytes per row.
// The C API validates 16-byte bases and strides divisible by 8.
#include <cstdint>

#include "codec/codec.h"
#include "codec/tile.cuh"
#include "internal.h"

namespace nf12::detail {
namespace {

using codec::AtomRegs;
using codec::kThreadsPerBlock;
using codec::kWarpsPerBlock;
using codec::RowTile;

constexpr int kDecodeSuperAtomsPerWarp = 2;
constexpr int kPackSuperAtomsPerWarp = 2;
constexpr int kSuperAtomVecs = kSuperAtomBytes / 16;
constexpr int kSuperAtomVecsBf16t = kSuperAtomBytesBf16t / 16;
constexpr int kPlaneVecs = kPlaneBytes / 16;

// Interleave warps along K so their row-major accesses form contiguous segments.
template <int kPerWarp>
__device__ __forceinline__ int64_t warp_super_atom(int i) {
  return (int64_t(blockIdx.x) * kPerWarp + i) * kWarpsPerBlock + (threadIdx.x >> 5);
}

template <int kPerWarp>
int64_t grid_blocks(int64_t num_sa) {
  const int64_t per_block = int64_t(kPerWarp) * kWarpsPerBlock;
  return (num_sa + per_block - 1) / per_block;
}

template <int kPerWarp>
__device__ __forceinline__ void decode_rows(const uint4* __restrict__ packed, const uint4* __restrict__ escapes,
                                            uint32_t base_x4, int64_t num_sa, int64_t k_blocks,
                                            __nv_bfloat16* __restrict__ out, int64_t ldo) {
  const int lane = threadIdx.x & 31;
  uint4 planes[kPerWarp][3];
#pragma unroll
  for (int i = 0; i < kPerWarp; ++i) {
    const int64_t sa = warp_super_atom<kPerWarp>(i);
    if (sa < num_sa) {
      const uint4* src = packed + sa * kSuperAtomVecs + lane;
#pragma unroll
      for (int p = 0; p < 3; ++p) planes[i][p] = __ldcs(src + p * kPlaneVecs);
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
    codec::atoms_to_tile(r, x);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int p = 0; p < 2; ++p) codec::quad_transpose(x[h][p], lane);
    }
    codec::store_row_tile(out, codec::row_tile_offset(codec::super_atom(sa, k_blocks), ldo, lane), ldo, x);
  }
}

template <int kPerWarp>
__global__ void __launch_bounds__(kThreadsPerBlock)
    decode_kernel(const uint4* __restrict__ packed, const uint4* __restrict__ escapes, uint32_t base_x4,
                  int64_t num_sa, int64_t k_blocks, __nv_bfloat16* __restrict__ out, int64_t ldo) {
  decode_rows<kPerWarp>(packed, escapes, base_x4, num_sa, k_blocks, out, ldo);
}

template <int kPerWarp>
cudaError_t launch_decode(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                          int64_t num_sa, int64_t k_blocks, __nv_bfloat16* out, int64_t ldo,
                          cudaStream_t stream) {
  const int64_t blocks = grid_blocks<kPerWarp>(num_sa);
  if (blocks > codec::kMaxGridX) return cudaErrorInvalidValue;
  decode_kernel<kPerWarp><<<dim3(static_cast<unsigned>(blocks)), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const uint4*>(packed), escapes, exp_base_x4(exp_base), num_sa, k_blocks, out, ldo);
  return cudaGetLastError();
}

// Match moe_linear's clamped, monotone offsets (moe/schedule.cuh).
// Expert e owns [max(off'[0..e]), max(off'[0..e+1])).
// Each warp reduces the prefix over at most kMoeMaxExperts.
__device__ __forceinline__ bool moe_expert_active(const int64_t* __restrict__ offsets, int e,
                                                  int64_t max_assignments) {
  const int lane = threadIdx.x & 31;
  int64_t start = 0;
  for (int i = lane; i <= e; i += 32) start = max(start, min(max(__ldg(offsets + i), int64_t(0)), max_assignments));
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) start = max(start, __shfl_xor_sync(codec::kFullMask, start, d));
  const int64_t end = min(max(__ldg(offsets + e + 1), int64_t(0)), max_assignments);
  return end > start;
}

// Escape indices and exp_base are expert-local (SPEC.md 9.1).
template <int kPerWarp>
__global__ void __launch_bounds__(kThreadsPerBlock)
    moe_decode_kernel(const uint4* __restrict__ packed, int64_t expert_vecs, const uint4* __restrict__ escapes,
                      const int64_t* __restrict__ escape_offsets, const uint8_t* __restrict__ exp_base,
                      const int64_t* __restrict__ expert_offsets, int64_t max_assignments, int64_t num_sa,
                      int64_t k_blocks, __nv_bfloat16* __restrict__ out, int64_t expert_stride, int64_t ldo) {
  const int e = static_cast<int>(blockIdx.y);
  if (expert_offsets != nullptr && !moe_expert_active(expert_offsets, e, max_assignments)) return;
  const uint4* esc = escapes != nullptr ? escapes + __ldg(escape_offsets + e) : nullptr;
  decode_rows<kPerWarp>(packed + e * expert_vecs, esc, exp_base_x4(__ldg(exp_base + e)), num_sa, k_blocks,
                        out + e * expert_stride, ldo);
}

// The bf16t super-atom stores atom j's decoded registers at plane j (SPEC.md
// 4.4), so decoding to it needs no transpose.
template <int kPerWarp>
__global__ void __launch_bounds__(kThreadsPerBlock)
    decode_bf16t_kernel(const uint4* __restrict__ packed, const uint4* __restrict__ escapes, uint32_t base_x4,
                        int64_t num_sa, uint4* __restrict__ out) {
  const int lane = threadIdx.x & 31;
  uint4 planes[kPerWarp][3];
#pragma unroll
  for (int i = 0; i < kPerWarp; ++i) {
    const int64_t sa = warp_super_atom<kPerWarp>(i);
    if (sa < num_sa) {
      const uint4* src = packed + sa * kSuperAtomVecs + lane;
#pragma unroll
      for (int p = 0; p < 3; ++p) planes[i][p] = __ldcs(src + p * kPlaneVecs);
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
    uint4* dst = out + sa * kSuperAtomVecsBf16t + lane;
#pragma unroll
    for (int j = 0; j < kAtomsPerSuperAtom; ++j) dst[j * kPlaneVecs] = make_uint4(r[j][0], r[j][1], r[j][2], r[j][3]);
  }
}

template <int kPerWarp>
__global__ void __launch_bounds__(kThreadsPerBlock)
    pack_bf16t_kernel(const __nv_bfloat16* __restrict__ w, int64_t ldw, int64_t num_sa, int64_t k_blocks,
                      uint4* __restrict__ out) {
  const int lane = threadIdx.x & 31;
  RowTile x[kPerWarp];
#pragma unroll
  for (int i = 0; i < kPerWarp; ++i) {
    const int64_t sa = warp_super_atom<kPerWarp>(i);
    if (sa < num_sa) codec::load_row_tile(w, codec::row_tile_offset(codec::super_atom(sa, k_blocks), ldw, lane), ldw, x[i]);
  }
#pragma unroll
  for (int i = 0; i < kPerWarp; ++i) {
    const int64_t sa = warp_super_atom<kPerWarp>(i);
    if (sa >= num_sa) break;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int p = 0; p < 2; ++p) codec::quad_transpose(x[i][h][p], lane);
    }
    AtomRegs r;
    codec::tile_to_atoms(x[i], r);
    uint4* dst = out + sa * kSuperAtomVecsBf16t + lane;
#pragma unroll
    for (int j = 0; j < kAtomsPerSuperAtom; ++j) __stcs(dst + j * kPlaneVecs, make_uint4(r[j][0], r[j][1], r[j][2], r[j][3]));
  }
}

}

cudaError_t decode(const uint8_t* packed, const uint4* escapes, uint32_t exp_base, int64_t n, int64_t k,
                   __nv_bfloat16* out, int64_t ldo, cudaStream_t stream) {
  if (exp_base > kMaxExpBase) return cudaErrorInvalidValue;
  const int64_t num_sa = (n / kSuperAtomRows) * (k / kSuperAtomCols);
  if (num_sa == 0) return cudaSuccess;
  // The compressor's 512x5120 SM80 prefill decode is latency-bound with two
  // super-atoms per warp. A single super-atom measured faster on PG506-230.
  if (n == 512 && k == 5120) {
    int device = 0;
    if (cudaError_t err = cudaGetDevice(&device); err != cudaSuccess) return err;
    if (device_info(device).sm_arch == 80)
      return launch_decode<1>(packed, escapes, exp_base, num_sa, k / kSuperAtomCols, out, ldo, stream);
  }
  return launch_decode<2>(packed, escapes, exp_base, num_sa, k / kSuperAtomCols, out, ldo, stream);
}

cudaError_t moe_decode(const uint8_t* packed, const uint4* escapes, const int64_t* escape_offsets,
                       const uint8_t* exp_base, int64_t num_experts, int64_t n, int64_t k,
                       const int64_t* expert_offsets, int64_t max_assignments, int64_t row0, int64_t rows,
                       __nv_bfloat16* out, int64_t ldo, cudaStream_t stream) {
  const int64_t num_sa = (rows / kSuperAtomRows) * (k / kSuperAtomCols);
  const int64_t blocks = grid_blocks<kDecodeSuperAtomsPerWarp>(num_sa);
  if (blocks == 0) return cudaSuccess;
  if (blocks > codec::kMaxGridX || num_experts > 65535) return cudaErrorInvalidValue;
  moe_decode_kernel<kDecodeSuperAtomsPerWarp>
      <<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(num_experts)), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const uint4*>(packed + row0 * k * 3 / 2), n * k * 3 / 2 / 16, escapes, escape_offsets,
          exp_base, expert_offsets, max_assignments, num_sa, k / kSuperAtomCols, out, rows * ldo, ldo);
  return cudaGetLastError();
}

cudaError_t decode_bf16t(const uint8_t* packed, const uint4* escapes, uint32_t exp_base, int64_t n, int64_t k,
                         uint8_t* out, cudaStream_t stream) {
  if (exp_base > kMaxExpBase) return cudaErrorInvalidValue;
  const int64_t num_sa = (n / kSuperAtomRows) * (k / kSuperAtomCols);
  const int64_t blocks = grid_blocks<kDecodeSuperAtomsPerWarp>(num_sa);
  if (blocks == 0) return cudaSuccess;
  if (blocks > codec::kMaxGridX) return cudaErrorInvalidValue;
  decode_bf16t_kernel<kDecodeSuperAtomsPerWarp><<<dim3(static_cast<unsigned>(blocks)), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const uint4*>(packed), escapes, exp_base_x4(exp_base), num_sa, reinterpret_cast<uint4*>(out));
  return cudaGetLastError();
}

cudaError_t pack_bf16t(const __nv_bfloat16* w, int64_t n, int64_t k, int64_t ldw, uint8_t* out,
                       cudaStream_t stream) {
  const int64_t num_sa = (n / kSuperAtomRows) * (k / kSuperAtomCols);
  const int64_t blocks = grid_blocks<kPackSuperAtomsPerWarp>(num_sa);
  if (blocks == 0) return cudaSuccess;
  if (blocks > codec::kMaxGridX) return cudaErrorInvalidValue;
  pack_bf16t_kernel<kPackSuperAtomsPerWarp><<<dim3(static_cast<unsigned>(blocks)), kThreadsPerBlock, 0, stream>>>(
      w, ldw, num_sa, k / kSuperAtomCols, reinterpret_cast<uint4*>(out));
  return cudaGetLastError();
}

}
