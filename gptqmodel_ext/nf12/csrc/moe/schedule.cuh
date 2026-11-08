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
// Expert counts exist only on the device. Each expert's sorted positions form
// consecutive tiles of at most bm rows, numbered expert-major in row order.
// moe_mtile_prefix() builds an in-CTA index. moe_schedule() writes the same
// tiles to a global list for persistent kernels.
//
// Clamp offsets to [0, max_assignments] and take their running maximum so
// malformed offsets cannot duplicate positions, exceed max_assignments, or
// produce more than moe_max_mtiles() tiles.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include <nf12/format.cuh>

namespace nf12::detail {

// Experts supported by the shared-memory scan (dispatch validates E against it).
inline constexpr int kMoeMaxExperts = 1024;

// Shared scratch for moe_mtile_prefix: prefix[E + 1], start[E + 1], scan scratch[32].
// Expert e owns [start[e], start[e + 1]) after clamping and the running maximum.
NF12_HD constexpr int moe_schedule_smem_ints(int num_experts) { return 2 * num_experts + 34; }

// Avoid count + bm - 1 overflow for 0 <= count <= INT32_MAX and bm >= 1.
NF12_HD constexpr int moe_ceil_div(int count, int bm) { return count / bm + (count % bm != 0); }

// Bounds sum_e ceil(c_e / bm) with at most min(E, A) nonempty experts
// and sum_e c_e <= A, where A = max_assignments.
NF12_HD int64_t moe_max_mtiles(int64_t max_assignments, int64_t num_experts, int bm) {
  if (max_assignments <= 0) return 0;
  const int64_t nonempty = num_experts < max_assignments ? num_experts : max_assignments;
  const int64_t bound = (max_assignments + bm - 1) / bm + nonempty - 1;
  return bound < max_assignments ? bound : max_assignments;
}

// Each 16-byte entry fits one vector load.
struct MoeTile {
  int32_t expert;
  int32_t row_begin;  // sorted position of the tile's first row
  int32_t rows;       // 1..bm
  int32_t expert_tile;  // index of this tile within its expert
};

// Global list layout in workspace: int32 count at byte 0, tiles from byte 16
// (moe_max_mtiles() slots), then int32 row_end[E]: expert e's clamped,
// monotone end offset (its rows are [row_end[e-1] or start, row_end[e])).
inline constexpr size_t kMoeScheduleHeaderBytes = 16;
NF12_HD size_t moe_schedule_row_end_offset(int64_t max_assignments, int64_t num_experts, int bm) {
  return kMoeScheduleHeaderBytes + static_cast<size_t>(moe_max_mtiles(max_assignments, num_experts, bm)) * 16;
}
size_t moe_schedule_bytes(int64_t max_assignments, int64_t num_experts, int bm);
// workspace requires moe_schedule_bytes() bytes and 16-byte alignment.
// On SM90+, dependent PDL kernels must execute griddepcontrol.wait before
// reading the list.
cudaError_t moe_schedule(const int64_t* expert_offsets, int64_t num_experts, int64_t max_assignments, int bm,
                         void* workspace, cudaStream_t stream);

#if defined(__CUDACC__)

__device__ __forceinline__ const int32_t* moe_schedule_count(const void* workspace) {
  return static_cast<const int32_t*>(workspace);
}
__device__ __forceinline__ const MoeTile* moe_schedule_tiles(const void* workspace) {
  return reinterpret_cast<const MoeTile*>(static_cast<const uint8_t*>(workspace) + kMoeScheduleHeaderBytes);
}
__device__ __forceinline__ const int32_t* moe_schedule_row_end(const void* workspace, int64_t max_assignments,
                                                               int64_t num_experts, int bm) {
  return reinterpret_cast<const int32_t*>(static_cast<const uint8_t*>(workspace) +
                                          moe_schedule_row_end_offset(max_assignments, num_experts, bm));
}

// Op must be associative and commutative with the supplied identity.
// total receives the reduction. Call sync() before reusing scratch[32].
template <class Op, class Sync>
__device__ __forceinline__ int moe_block_exclusive_scan(int v, int identity, Op op, int* scratch, int tid,
                                                        int nthreads, Sync sync, int& total) {
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = nthreads >> 5;
  int incl = v;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int t = __shfl_up_sync(0xffffffffu, incl, d);
    if (lane >= d) incl = op(t, incl);
  }
  if (lane == 31) scratch[warp] = incl;
  sync();
  if (warp == 0) {
    int w = lane < nwarps ? scratch[lane] : identity;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int t = __shfl_up_sync(0xffffffffu, w, d);
      if (lane >= d) w = op(t, w);
    }
    if (lane < nwarps) scratch[lane] = w;  // inclusive over warps
  }
  sync();
  int ex = __shfl_up_sync(0xffffffffu, incl, 1);
  if (lane == 0) ex = identity;
  if (warp > 0) ex = op(scratch[warp - 1], ex);
  total = scratch[nwarps - 1];
  return ex;
}

// smem requires moe_schedule_smem_ints(E) ints. Participants must be whole warps
// numbered tid = 0..nthreads-1 with tid % 32 == lane and nthreads <= 1024.
// sync() must cover exactly those threads. Returns the tile total after sync().
template <class Sync>
__device__ int moe_mtile_prefix(const int64_t* __restrict__ expert_offsets, int num_experts, int bm,
                                int64_t max_assignments, int* smem, int tid, int nthreads, Sync sync) {
  const int E = num_experts;
  int* prefix = smem;
  int* start = smem + E + 1;
  int* scratch = start + E + 1;
  const auto vmax = [](int a, int b) { return a > b ? a : b; };
  const auto vadd = [](int a, int b) { return a + b; };
  const int per = (E + nthreads) / nthreads;
  const int i0 = min(E + 1, tid * per);
  const int i1 = min(E + 1, i0 + per);
  int run = 0;
  for (int i = i0; i < i1; ++i) {
    const int64_t v = __ldg(expert_offsets + i);
    run = vmax(run, static_cast<int>(v < 0 ? 0 : v > max_assignments ? max_assignments : v));
    start[i] = run;
  }
  int unused;
  const int before = moe_block_exclusive_scan(run, 0, vmax, scratch, tid, nthreads, sync, unused);
  for (int i = i0; i < i1; ++i) start[i] = vmax(start[i], before);
  sync();
  const int e1 = min(i1, E);
  int local = 0;
  for (int e = i0; e < e1; ++e) {
    prefix[e] = local;  // thread-local exclusive prefix, rebased below
    local += moe_ceil_div(start[e + 1] - start[e], bm);
  }
  int total;
  const int base = moe_block_exclusive_scan(local, 0, vadd, scratch, tid, nthreads, sync, total);
  for (int e = i0; e < e1; ++e) prefix[e] += base;
  if (tid == 0) prefix[E] = total;
  sync();
  return total;
}

__device__ __forceinline__ int moe_mtile_prefix(const int64_t* __restrict__ expert_offsets, int num_experts, int bm,
                                                int64_t max_assignments, int* smem) {
  return moe_mtile_prefix(expert_offsets, num_experts, bm, max_assignments, smem, static_cast<int>(threadIdx.x),
                          static_cast<int>(blockDim.x), [] { __syncthreads(); });
}

// Expert owning tile `mtile` (0 <= mtile < total): the largest e with
// prefix[e] <= mtile, which skips empty experts.
__device__ __forceinline__ int moe_mtile_expert(const int* smem, int num_experts, int mtile) {
  int lo = 0;
  int hi = num_experts;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (smem[mid] <= mtile)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

// Requires block-uniform mtile in [0, total) immediately after a whole-block
// moe_mtile_prefix(). Exactly one expert owns it and publishes its index.
__device__ __forceinline__ int moe_mtile_expert_block(int* smem, int num_experts, int mtile) {
  int* slot = smem + 2 * (num_experts + 1);  // scratch[0], free once the prefix is done
  for (int e = static_cast<int>(threadIdx.x); e < num_experts; e += static_cast<int>(blockDim.x))
    if (smem[e] <= mtile && mtile < smem[e + 1]) *slot = e;
  __syncthreads();
  return *slot;
}

// Tile `t` (0-based within non-empty expert e): first sorted position and row count.
__device__ __forceinline__ void moe_mtile_rows(const int* smem, int num_experts, int e, int t, int bm, int& row0,
                                               int& rows) {
  const int* start = smem + num_experts + 1;
  row0 = start[e] + t * bm;
  rows = min(bm, start[e + 1] - row0);
}

__device__ __forceinline__ void moe_mtile_find(const int* smem, int num_experts, int bm, int mtile, int& e, int& row0,
                                               int& rows) {
  e = moe_mtile_expert(smem, num_experts, mtile);
  moe_mtile_rows(smem, num_experts, e, mtile - smem[e], bm, row0, rows);
}

#endif  // __CUDACC__

}  // namespace nf12::detail
