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
// The C API validates shapes, 16-byte bases, ldw % 8 == 0 and workspace size.
#include <algorithm>
#include <cstdint>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

#include "codec/codec.h"
#include "codec/tile.cuh"
#include "internal.h"

namespace nf12::detail {
namespace {

using codec::AtomRegs;
using codec::kThreadsPerBlock;
using codec::kWarpsPerBlock;

constexpr int kHistBins = 128;  // B in [0, 120], padded
constexpr int kNumHists = 3;    // interval starts, interval ends, sentinel collisions
constexpr size_t kAnalyzeWorkspaceBytes = sizeof(unsigned long long) * kNumHists * kHistBins;
constexpr int kAnalyzeSuperAtomsPerWarp = 16;
// Fixed units make escape-index order independent of the launch grid.
constexpr int kUnitSuperAtoms = 8;
constexpr int kScanThreads = 256;
constexpr int kScanItemsPerThread = 8;

int64_t num_super_atoms(int64_t n, int64_t k) { return (n / kSuperAtomRows) * (k / kSuperAtomCols); }
int64_t num_units(int64_t num_sa) { return (num_sa + kUnitSuperAtoms - 1) / kUnitSuperAtoms; }

// Per-warp histograms limit same-address atomic contention.
// Shared bins fit 16*8*128 groups per block. Global counts need 64 bits.
__global__ void __launch_bounds__(kThreadsPerBlock)
    analyze_kernel(const __nv_bfloat16* __restrict__ w, int64_t ldw, int64_t num_sa, int64_t k_blocks,
                   unsigned long long* __restrict__ hist) {
  __shared__ uint32_t sh[kWarpsPerBlock][kNumHists][kHistBins];
  uint32_t* flat = &sh[0][0][0];
  for (int i = threadIdx.x; i < kWarpsPerBlock * kNumHists * kHistBins; i += kThreadsPerBlock) flat[i] = 0;
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  uint32_t(&mine)[kNumHists][kHistBins] = sh[warp];
  const int64_t sa0 = int64_t(blockIdx.x) * kAnalyzeSuperAtomsPerWarp * kWarpsPerBlock + warp;
  for (int i = 0; i < kAnalyzeSuperAtomsPerWarp; ++i) {
    const int64_t sa = sa0 + int64_t(i) * kWarpsPerBlock;
    if (sa >= num_sa) break;
    AtomRegs r;
    codec::load_atoms(w, ldw, codec::super_atom(sa, k_blocks), lane, r);
#pragma unroll
    for (int j = 0; j < kAtomsPerSuperAtom; ++j) {
      const codec::GroupStats st = codec::group_stats(r[j]);
      const uint32_t lo = st.hmax > 7u ? st.hmax - 7u : 0u;
      const uint32_t hi = min(st.hmin, uint32_t(kMaxExpBase));
      if (lo <= hi) {
        atomicAdd(&mine[0][lo], 1u);
        atomicAdd(&mine[1][hi], 1u);
      }
      if (st.all_positive && st.hmin == st.hmax && st.hmin <= uint32_t(kMaxExpBase)) atomicAdd(&mine[2][st.hmin], 1u);
    }
  }
  __syncthreads();

  for (int bin = threadIdx.x; bin < kNumHists * kHistBins; bin += kThreadsPerBlock) {
    uint32_t sum = 0;
#pragma unroll
    for (int wi = 0; wi < kWarpsPerBlock; ++wi) sum += (&sh[wi][0][0])[bin];
    if (sum != 0) atomicAdd(&hist[bin], static_cast<unsigned long long>(sum));
  }
}

// Each unit has at most kUnitSuperAtoms * 128 escapes.
__global__ void __launch_bounds__(kThreadsPerBlock)
    count_kernel(const __nv_bfloat16* __restrict__ w, int64_t ldw, int64_t num_sa, int64_t k_blocks,
                 uint32_t base_x4, uint64_t* __restrict__ unit_counts, int64_t n_units) {
  const int lane = threadIdx.x & 31;
  const int64_t unit = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
  if (unit >= n_units) return;
  uint32_t count = 0;
  for (int i = 0; i < kUnitSuperAtoms; ++i) {
    const int64_t sa = unit * kUnitSuperAtoms + i;
    if (sa >= num_sa) break;
    AtomRegs r;
    codec::load_atoms(w, ldw, codec::super_atom(sa, k_blocks), lane, r);
#pragma unroll
    for (int j = 0; j < kAtomsPerSuperAtom; ++j) {
      Group g;
      count += __popc(__ballot_sync(codec::kFullMask, !codec::encode_group_fast(r[j], base_x4, g)));
    }
  }
  if (lane == 0) unit_counts[unit] = count;
}

__global__ void __launch_bounds__(kScanThreads) scan_kernel(uint64_t* __restrict__ v, int64_t n) {
  using Load = cub::BlockLoad<uint64_t, kScanThreads, kScanItemsPerThread, cub::BLOCK_LOAD_TRANSPOSE>;
  using Scan = cub::BlockScan<uint64_t, kScanThreads>;
  using Store = cub::BlockStore<uint64_t, kScanThreads, kScanItemsPerThread, cub::BLOCK_STORE_TRANSPOSE>;
  __shared__ union {
    Load::TempStorage load;
    Scan::TempStorage scan;
    Store::TempStorage store;
  } scratch;
  uint64_t carry = 0;
  int64_t offset = 0;
  while (offset < n) {
    const int valid = static_cast<int>(min(int64_t(kScanThreads * kScanItemsPerThread), n - offset));
    uint64_t items[kScanItemsPerThread];
    // Read the whole tile before in-place writes; zero padding preserves its sum.
    Load(scratch.load).Load(v + offset, items, valid, uint64_t(0));
    __syncthreads();
    uint64_t aggregate;
    Scan(scratch.scan).ExclusiveSum(items, items, aggregate);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kScanItemsPerThread; ++i) items[i] += carry;
    carry += aggregate;
    Store(scratch.store).Store(v + offset, items, valid);
    __syncthreads();
    // Bounded advancement avoids overflowing the final 64-bit tile offset.
    offset += valid;
  }
  if (threadIdx.x == 0) v[n] = carry;
}

// Host validation guarantees that escape indices fit the 32-bit L0 word.
__global__ void __launch_bounds__(kThreadsPerBlock)
    write_kernel(const __nv_bfloat16* __restrict__ w, int64_t ldw, int64_t num_sa, int64_t k_blocks,
                 uint32_t base_x4, const uint64_t* __restrict__ unit_offsets, int64_t n_units,
                 uint4* __restrict__ packed, uint4* __restrict__ escapes) {
  const int lane = threadIdx.x & 31;
  const int64_t unit = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
  if (unit >= n_units) return;
  const uint32_t lt_mask = (1u << lane) - 1u;
  uint32_t next = static_cast<uint32_t>(unit_offsets[unit]);
  for (int i = 0; i < kUnitSuperAtoms; ++i) {
    const int64_t sa = unit * kUnitSuperAtoms + i;
    if (sa >= num_sa) break;
    AtomRegs r;
    codec::load_atoms(w, ldw, codec::super_atom(sa, k_blocks), lane, r);
    Group g[kAtomsPerSuperAtom];
    // Escape indices follow flat group id order: atom-major, then lane.
#pragma unroll
    for (int j = 0; j < kAtomsPerSuperAtom; ++j) {
      const bool esc = !codec::encode_group_fast(r[j], base_x4, g[j]);
      const uint32_t ballot = __ballot_sync(codec::kFullMask, esc);
      if (esc) {
        const uint32_t idx = next + __popc(ballot & lt_mask);
        escapes[idx] = make_uint4(r[j][0], r[j][1], r[j][2], r[j][3]);
        g[j] = Group{0u, idx, 0u};
      }
      next += __popc(ballot);
    }
    uint4* dst = packed + sa * (kSuperAtomBytes / 16) + lane;
    dst[0] = make_uint4(g[0].meta, g[1].meta, g[2].meta, g[3].meta);
    dst[kPlaneBytes / 16] = make_uint4(g[0].lo0, g[0].lo1, g[1].lo0, g[1].lo1);
    dst[2 * kPlaneBytes / 16] = make_uint4(g[2].lo0, g[2].lo1, g[3].lo0, g[3].lo1);
  }
}

}

size_t encode_workspace_bytes(int64_t n, int64_t k) {
  const size_t unit_bytes = sizeof(uint64_t) * size_t(num_units(num_super_atoms(n, k)) + 1);
  return std::max(kAnalyzeWorkspaceBytes, (unit_bytes + 15) & ~size_t(15));
}

cudaError_t encode_analyze(const __nv_bfloat16* w, int64_t n, int64_t k, int64_t ldw, void* workspace,
                           size_t /*workspace_bytes*/, uint8_t* exp_base_out, uint32_t* num_escapes_out,
                           cudaStream_t stream) {
  const int64_t num_sa = num_super_atoms(n, k);
  auto* hist = static_cast<unsigned long long*>(workspace);
  cudaError_t err = cudaMemsetAsync(hist, 0, kAnalyzeWorkspaceBytes, stream);
  if (err != cudaSuccess) return err;
  const int64_t sa_per_block = int64_t(kAnalyzeSuperAtomsPerWarp) * kWarpsPerBlock;
  const int64_t blocks = (num_sa + sa_per_block - 1) / sa_per_block;
  if (blocks > codec::kMaxGridX) return cudaErrorInvalidValue;
  if (blocks > 0) {
    analyze_kernel<<<dim3(static_cast<unsigned>(blocks)), kThreadsPerBlock, 0, stream>>>(w, ldw, num_sa,
                                                                                         k / kSuperAtomCols, hist);
    if ((err = cudaGetLastError()) != cudaSuccess) return err;
  }
  unsigned long long h[kNumHists][kHistBins];
  if ((err = cudaMemcpyAsync(h, hist, sizeof(h), cudaMemcpyDeviceToHost, stream)) != cudaSuccess) return err;
  if ((err = cudaStreamSynchronize(stream)) != cudaSuccess) return err;

  // escapes(B) = groups - intervals_containing(B) + sentinel_collisions(B).
  const unsigned long long groups = static_cast<unsigned long long>(n) * static_cast<unsigned long long>(k) / 8;
  unsigned long long covered = 0, best = ~0ull;
  int best_b = 0;
  for (int b = 0; b <= kMaxExpBase; ++b) {
    covered += h[0][b];
    const unsigned long long esc = groups - covered + h[2][b];
    if (esc < best) {
      best = esc;
      best_b = b;
    }
    covered -= h[1][b];
  }
  // Escape indices are stored in a 32-bit L0 word and counted in a uint32_t.
  if (best > UINT32_MAX) return cudaErrorInvalidValue;
  *exp_base_out = static_cast<uint8_t>(best_b);
  *num_escapes_out = static_cast<uint32_t>(best);
  return cudaSuccess;
}

cudaError_t encode(const __nv_bfloat16* w, int64_t n, int64_t k, int64_t ldw, uint8_t exp_base, uint8_t* packed,
                   uint4* escapes, uint32_t escapes_capacity, void* workspace, size_t /*workspace_bytes*/,
                   cudaStream_t stream) {
  if (exp_base > kMaxExpBase) return cudaErrorInvalidValue;
  const int64_t num_sa = num_super_atoms(n, k);
  const int64_t n_units = num_units(num_sa);
  if (n_units == 0) return cudaSuccess;
  auto* units = static_cast<uint64_t*>(workspace);
  const uint32_t base_x4 = exp_base_x4(exp_base);
  const int64_t k_blocks = k / kSuperAtomCols;
  const int64_t blocks64 = (n_units + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (blocks64 > codec::kMaxGridX) return cudaErrorInvalidValue;
  const auto blocks = static_cast<unsigned>(blocks64);
  count_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(w, ldw, num_sa, k_blocks, base_x4, units, n_units);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  if ((err = codec::escape_offsets_scan(units, n_units, stream)) != cudaSuccess) return err;
  // Check capacity and 32-bit index bounds before writing packed/escape outputs.
  uint64_t total;
  if ((err = cudaMemcpyAsync(&total, units + n_units, sizeof(total), cudaMemcpyDeviceToHost, stream)) != cudaSuccess)
    return err;
  if ((err = cudaStreamSynchronize(stream)) != cudaSuccess) return err;
  if (total > escapes_capacity || total > UINT32_MAX) return cudaErrorInvalidValue;
  write_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(w, ldw, num_sa, k_blocks, base_x4, units, n_units,
                                                         reinterpret_cast<uint4*>(packed), escapes);
  return cudaGetLastError();
}

}

namespace nf12::codec {

cudaError_t escape_offsets_scan(uint64_t* v, int64_t n, cudaStream_t stream) {
  detail::scan_kernel<<<1, detail::kScanThreads, 0, stream>>>(v, n);
  return cudaGetLastError();
}

}
