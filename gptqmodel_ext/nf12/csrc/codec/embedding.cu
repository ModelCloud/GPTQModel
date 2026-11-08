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
// Row r occupies lanes 4*(r%8) + t, t in [0,4), of each super-atom.
// Their plane segments are contiguous, so decoding a row reads 3*64 bytes
// from the 1536-byte super-atom. Each group also contains the paired row r^8.
// Keep R0/R2 for r%16 < 8, R1/R3 otherwise, then quad-transpose to write
// 128 contiguous output bytes per (id, super-atom).
// A CTA packs multiple IDs when the row has fewer super-atoms than its quads.
// Invalid IDs write +0.0 without reading the table.
#include <cstdint>

#include "codec/codec.h"
#include "codec/tile.cuh"
#include "internal.h"

namespace nf12::detail {
namespace {

constexpr int kSuperAtomVecs = kSuperAtomBytes / 16;
constexpr int kPlaneVecs = kPlaneBytes / 16;

struct EmbeddingParams {
  const uint4* packed;
  const uint4* escapes;  // may be null when num_escapes == 0
  uint32_t num_escapes;
  uint32_t base_x4;
  int64_t n;
  int64_t k_blocks;        // K / 64 super-atoms per row
  int64_t num_ids;
  __nv_bfloat16* out;
  int64_t ldo;
  uint32_t ids_per_block;
  uint32_t kb_span;
  uint32_t kb_chunks;
};

// upper selects slots s2/s3/s6/s7, otherwise s0/s1/s4/s5.
// Reject out-of-range escape indices before dereferencing the table.
__device__ __forceinline__ uint2 decode_row_half(const Group& g, uint32_t base_x4, const uint4* __restrict__ escapes,
                                                 uint32_t num_escapes, bool upper) {
  uint32_t r[4];
  decode_bulk(g, base_x4, r);
  if (g.meta == 0u && g.lo0 < num_escapes) {
    const uint4 v = __ldg(escapes + g.lo0);
    r[0] = v.x;
    r[1] = v.y;
    r[2] = v.z;
    r[3] = v.w;
  }
  return upper ? make_uint2(r[1], r[3]) : make_uint2(r[0], r[2]);
}

template <int Quads, typename Id>
__global__ void __launch_bounds__(4 * Quads) embedding_kernel(EmbeddingParams p, const Id* __restrict__ ids) {
  const int lane = threadIdx.x & 31;
  const uint32_t quad = threadIdx.x >> 2;
  const uint32_t id_block = blockIdx.x / p.kb_chunks;
  const uint32_t chunk = blockIdx.x - id_block * p.kb_chunks;
  const uint32_t local = quad / p.kb_span;
  const int64_t kb = int64_t(chunk) * Quads + (quad - local * p.kb_span);
  const int64_t i = int64_t(id_block) * p.ids_per_block + local;
  const bool active = local < p.ids_per_block && kb < p.k_blocks && i < p.num_ids;

  // v0: column pairs t of output chunks 0-3, v1: of chunks 4-7 (8 columns each).
  uint4 v0 = make_uint4(0u, 0u, 0u, 0u);
  uint4 v1 = v0;
  if (active) {
    const int64_t row = static_cast<int64_t>(ids[i]);
    if (row >= 0 && row < p.n) {
      const int g = static_cast<int>(row & 7);
      const bool upper = (row & 8) != 0;
      const uint4* src = p.packed + ((row >> 4) * p.k_blocks + kb) * kSuperAtomVecs + 4 * g + (lane & 3);
      const uint4 m = __ldg(src);
      const uint4 l01 = __ldg(src + kPlaneVecs);
      const uint4 l23 = __ldg(src + 2 * kPlaneVecs);
      // Atom j's register pairs map to output chunks 2j and 2j+1.
      const uint2 a0 = decode_row_half(Group{m.x, l01.x, l01.y}, p.base_x4, p.escapes, p.num_escapes, upper);
      const uint2 a1 = decode_row_half(Group{m.y, l01.z, l01.w}, p.base_x4, p.escapes, p.num_escapes, upper);
      const uint2 a2 = decode_row_half(Group{m.z, l23.x, l23.y}, p.base_x4, p.escapes, p.num_escapes, upper);
      const uint2 a3 = decode_row_half(Group{m.w, l23.z, l23.w}, p.base_x4, p.escapes, p.num_escapes, upper);
      v0 = make_uint4(a0.x, a0.y, a1.x, a1.y);
      v1 = make_uint4(a2.x, a2.y, a3.x, a3.y);
    }
  }
  // All lanes, including inactive quads, must participate in the shuffles.
  codec::quad_transpose(v0, lane);
  codec::quad_transpose(v1, lane);
  if (active) {
    uint4* dst = reinterpret_cast<uint4*>(p.out + i * p.ldo + kb * kSuperAtomCols) + (lane & 3);
    dst[0] = v0;
    dst[4] = v1;
  }
}

template <int Quads, typename Id>
cudaError_t launch_embedding(EmbeddingParams p, const Id* ids, cudaStream_t stream) {
  const int64_t kb_chunks = (p.k_blocks + Quads - 1) / Quads;
  p.ids_per_block = p.k_blocks < Quads ? static_cast<uint32_t>(Quads / p.k_blocks) : 1u;
  p.kb_span = static_cast<uint32_t>(p.k_blocks < Quads ? p.k_blocks : Quads);
  const int64_t id_blocks = (p.num_ids + p.ids_per_block - 1) / p.ids_per_block;
  if (kb_chunks > codec::kMaxGridX / id_blocks) return cudaErrorInvalidValue;
  p.kb_chunks = static_cast<uint32_t>(kb_chunks);
  const dim3 grid(static_cast<unsigned>(id_blocks * kb_chunks));
  embedding_kernel<Quads><<<grid, 4 * Quads, 0, stream>>>(p, ids);
  return cudaGetLastError();
}

}

cudaError_t embedding(const uint8_t* packed, const uint4* escapes, uint32_t num_escapes, uint32_t exp_base, int64_t n,
                      int64_t k, const int32_t* ids32, const int64_t* ids64, int64_t num_ids, __nv_bfloat16* out,
                      int64_t ldo, cudaStream_t stream) {
  if (exp_base > kMaxExpBase || (ids32 == nullptr) == (ids64 == nullptr)) return cudaErrorInvalidValue;
  if (num_ids <= 0) return cudaSuccess;
  EmbeddingParams p{};
  p.packed = reinterpret_cast<const uint4*>(packed);
  p.escapes = escapes;
  p.num_escapes = escapes != nullptr ? num_escapes : 0;
  p.base_x4 = exp_base_x4(exp_base);
  p.n = n;
  p.k_blocks = k / kSuperAtomCols;
  p.num_ids = num_ids;
  p.out = out;
  p.ldo = ldo;
  // Larger CTAs improve cold, many-ID lookups on SM80 and SM100. Keep the
  // original geometry for small lookups and outside the measured width band.
  if (num_ids >= 4096 && k >= 1024 && k <= 4096) {
    int device = 0;
    if (cudaError_t err = cudaGetDevice(&device); err != cudaSuccess) return err;
    const int arch = device_info(device).sm_arch;
    if (arch == 80 || arch == 100) {
      return ids32 != nullptr ? launch_embedding<16>(p, ids32, stream) : launch_embedding<16>(p, ids64, stream);
    }
  }
  return ids32 != nullptr ? launch_embedding<8>(p, ids32, stream) : launch_embedding<8>(p, ids64, stream);
}

}
