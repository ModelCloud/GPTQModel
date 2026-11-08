// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12-AD (NyoomFloat12) fused decode+GEMM using RDNA4 matrix cores.
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
// Why this kernel exists
// ----------------------
// The small-M kernel is bandwidth-bound and wins up to 16 tokens. Above that
// the decode-then-rocBLAS route moves 24 MiB packed + 32 MiB dense write +
// 32 MiB dense read, and lands at 0.36-0.59x of a plain BF16 GEMM. This path
// never materialises the weight: each warp decodes one 16x64 super-atom into
// shared memory, then feeds `v_wmma_f32_16x16x16_bf16` matrix cores for every
// token tile in flight, so the packed bytes are read exactly once.
//
// Numerics (SPEC.md 6.2/6.3)
// --------------------------
// Decode is the same bit-exact NF12 decode as `nf12_ad_decode`, so the
// multiplied values are identical to the decode-then-GEMM route. Accumulation
// is FP32 in the matrix core (not BF16), and bias joins the FP32 accumulator
// before the single BF16 rounding, matching the CUDA small-M epilogue. Each
// output element is the sum of a fixed K walk, so results are deterministic.
//
// WMMA fragment layout
// --------------------
// `v_wmma_f32_16x16x16_bf16` wave32 takes 8 BF16 A, 8 BF16 B and 8 FP32 C/D
// per lane. The map was determined empirically on gfx1201 (one-hot probes,
// then asserted against a CPU reference in the README) and is:
//
//   A reg (l,i) = A[ l % 16 ][ kb + i + 8 * (l / 16) ]
//   B reg (l,i) = B[ i + 8 * (l / 16) ][ colbase + l % 16 ]
//   D reg (l,i) = D[ i + 8 * (l / 16) ][ colbase + l % 16 ]
//
// With B = W^T, B reg (l,i) = W[ colbase + l%16 ][ kb + i + 8*(l/16) ], i.e.
// the SAME 16-byte row fragment as A. That is why the decoded super-atom is
// staged row-major in shared memory: one 16-byte load then serves as B for
// every token tile.
//
// gfx1250 has no `wmma-128b-insts`, so this file compiles to an empty
// translation unit there and the caller keeps the decode-then-GEMM route.
// ---------------------------------------------------------------------------
#include "nf12_ad_format_hip.cuh"

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace nf12_hip {
namespace {

// Token tiles per CTA, templated. Each tile is 16 tokens, so the CTA covers
// kMt*16 tokens. The count is chosen by the launcher from M alone:
//
//   * Every instantiated tile runs its full K walk even when the caller has
//     fewer tokens than the CTA covers, so a fixed 8-tile kernel made M=32 pay
//     for 8 tiles' worth of matrix-core work. Templating lets M=32 run 2.
//   * The tile count is a COMPILE-TIME bound. A runtime `n_tiles` loop defeats
//     `#pragma unroll` on the accumulator arrays, spills acc[] to local memory
//     and measured 3-5x slower at every M. That is why this is a template and
//     not a parameter.
//
// 8 x 16 = 128 tokens is the largest instantiation: each warp then keeps 8
// accumulator fragments (64 VGPRs). Larger values spill.
constexpr int kMaxTiles = 8;
constexpr int kMaxTokens = kMaxTiles * 16;

// Maximum K split used by the launcher. Each split walks a disjoint slice of
// the K walk, so this multiplies the CTA count without duplicating any read.
constexpr int kMaxSplits = 8;

// kMt is the number of 16-token tiles this instantiation covers.
template <int kMt, int kSplits>
__global__ void __launch_bounds__(kThreadsPerBlock)
    wmma_kernel(const __hip_bfloat16* __restrict__ x, const uint4* __restrict__ packed,
                const uint4* __restrict__ escapes, uint32_t base_x4, int64_t n, int64_t k,
                int64_t m, __hip_bfloat16* __restrict__ out, const __hip_bfloat16* __restrict__ bias,
                float* __restrict__ partial) {
#if !NF12_AD_HAS_WMMA
  // gfx1250 has no wmma-128b-insts. The body still has to exist so the host
  // and device passes agree on the symbol; the dispatcher never selects it.
  (void)x; (void)packed; (void)escapes; (void)base_x4; (void)n; (void)k; (void)m;
  (void)out; (void)bias; (void)partial;
  if (blockIdx.x == 0xffffffffu && threadIdx.x == 0xffffffffu) out[0] = __float2bfloat16(0.f);
#else
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int64_t n0 = (int64_t(blockIdx.x) * kWarpsPerBlock + warp) * kSuperAtomRows;
  if (n0 >= n) return;

  // This CTA owns tokens [m0, m0 + kMt*16). kMt is chosen from M so one block
  // covers the caller's whole M and the packed weight is read exactly once.
  const int64_t m0 = int64_t(blockIdx.y) * (kMt * 16);
  const int64_t m_local = min(int64_t(kMt * 16), m - m0);
  if (m_local <= 0) return;

  const int64_t k_blocks = k / kSuperAtomCols;

  // ---- split-K -------------------------------------------------------------
  // At N=4096 there are only 32 CTAs for 32 CUs (8 warps each), which leaves
  // HBM latency almost fully exposed and measured 0.49-0.63x of a plain BF16
  // GEMM. Splitting K across grid.z gives each split its own DISJOINT k slice,
  // so the packed weight and x are still read exactly once in total: only the
  // CTA count multiplies. Measured on gfx1201, that took m=32 from 65 -> 38 us
  // and m=64 from 75 -> 49 us at N=4096.
  //
  // It only pays when the CTA count is below a few waves. At N=11008 the same
  // split over-subscribes the 32 CUs and loses (146 -> 182 us), so the
  // launcher picks the split from the CTA count, not from M alone.
  //
  // Slices are whole super-atoms (64 k each) and contiguous, so each split
  // still walks a valid prefix of the plane layout.
  const int64_t per = (k_blocks + kSplits - 1) / kSplits;
  const int64_t kb_lo = int64_t(blockIdx.z) * per;
  const int64_t kb_hi = min(k_blocks, kb_lo + per);
  // Two buffers per warp. Decoding into the idle buffer while the busy one is
  // consumed removes the barrier between the store and the first matrix-core
  // read, so the decode ALU overlaps the wmma issue instead of serialising with
  // it. Measured 1-4% faster and bit-identical (2 x 8 x 2 KiB = 32 KiB per CTA,
  // which still admits four CTAs per 128 KiB CU).
  __shared__ __align__(16) __hip_bfloat16 tile[2][kWarpsPerBlock][kSuperAtomRows * kSuperAtomCols];

  // Output column this lane owns, and its bias, constant across the K walk.
  const int col = int(n0) + (lane & 15);
  const float bias_v = (bias != nullptr) ? __bfloat162float(bias[col]) : 0.0f;

  // Row base of each 16-token tile. A D register maps to tile row 8*(l/16)+i,
  // so the fragment loader wants a tile ORIGIN here, not a row.
  //
  // A partial tail tile cannot read past `x`, so it is clamped to the last
  // full window (m - 16). That makes it recompute rows an earlier tile already
  // covered; to keep every output written exactly once, each tile owns the
  // disjoint global range [t*16, min((t+1)*16, m)) and drops the rest.
  int64_t tile_row[kMt];
  int64_t own_lo[kMt], own_hi[kMt];
#pragma unroll
  for (int t = 0; t < kMt; ++t) {
    const int64_t lo = m0 + int64_t(t) * 16;
    own_lo[t] = lo;
    own_hi[t] = min(lo + 16, m);
    tile_row[t] = min(lo, m - 16);
    if (tile_row[t] < 0) tile_row[t] = 0;
  }

  WmmaD acc[kMt];
#pragma unroll
  for (int t = 0; t < kMt; ++t) {
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[t][i] = 0.0f;
  }

  __hip_bfloat16* const sh0 = &tile[0][warp][0];
  __hip_bfloat16* const sh1 = &tile[1][warp][0];

  // Load one super-atom's three packed planes for this lane.
  auto load_planes = [&](int64_t kb, uint4 (&planes)[3]) {
    const int64_t sa = (n0 / kSuperAtomRows) * k_blocks + kb;
    const uint4* src = packed + sa * kSuperAtomVecs + lane;
#pragma unroll
    for (int p = 0; p < 3; ++p) planes[p] = src[p * kPlaneVecs];
  };

  // Decode 48 bytes of planes into the row-major 16x64 shared tile.
  auto decode_into = [&](const uint4 (&planes)[3], __hip_bfloat16* dst) {
    AtomRegs r;
    decode_group(Group{planes[0].x, planes[1].x, planes[1].y}, base_x4, escapes, r[0]);
    decode_group(Group{planes[0].y, planes[1].z, planes[1].w}, base_x4, escapes, r[1]);
    decode_group(Group{planes[0].z, planes[2].x, planes[2].y}, base_x4, escapes, r[2]);
    decode_group(Group{planes[0].w, planes[2].z, planes[2].w}, base_x4, escapes, r[3]);

    RowTile xr;
    atoms_to_tile(r, xr);
    // Undo the encoder's quad transpose to return to row-major.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int p = 0; p < 2; ++p) quad_transpose(xr[h][p], lane);
    }
    // Tile-local offset: `sh` holds ONE super-atom, so the n-block field is 0
    // (passing the absolute super-atom would write past the 16x64 tile).
    store_row_tile(dst, row_tile_offset(SuperAtom{0, 0}, kSuperAtomCols, lane), kSuperAtomCols, xr);
  };

  // Prefetch the next super-atom's packed bytes into registers before the
  // current one is consumed, so the three plane loads overlap the decode and
  // the matrix-core work.
  uint4 planes[3], next[3];
  bool have_planes = kb_lo < kb_hi;
  int buf = 0;
  if (have_planes) {
    load_planes(kb_lo, planes);
    decode_into(planes, sh0);
    __syncwarp();
  }

  for (int64_t kb = kb_lo; kb < kb_hi; ++kb) {
    __hip_bfloat16* const sh = (buf == 0) ? sh0 : sh1;
    // Decode the NEXT super-atom into the other buffer while this one is
    // consumed, so its ALU work overlaps the matrix-core issue below.
    if (kb + 1 < kb_hi) {
      load_planes(kb + 1, next);
      decode_into(next, (buf == 0) ? sh1 : sh0);
    }

    // 4 k-chunks of 16, each feeding one B fragment
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const WmmaA b = wmma_row_frag(sh + 16 * j, kSuperAtomCols, lane);
      // A depends on both the token tile (t) and the k-chunk (j), so it cannot
      // be hoisted out of either loop: each (t, j) pair is a distinct 16-byte
      // row fragment. What IS shared is B, which is loaded once per j above and
      // reused by every tile. The A loads hit L1 because the four j-chunks for
      // a row cover consecutive 16-byte spans of the same two cache lines.
#pragma unroll
      for (int t = 0; t < kMt; ++t) {
        // `wmma_row_frag` supplies the tile's own row ((lane & 15)) itself, so
        // the base passed here is the tile origin only. Adding the lane row
        // here as well applied it twice and produced garbage for every m > 16.
        const WmmaA a = wmma_row_frag(x + tile_row[t] * k + kb * kSuperAtomCols + 16 * j, k, lane);
        acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc[t]);
      }
    }
    __syncwarp();
    buf ^= 1;
  }

  // ---- epilogue ------------------------------------------------------------
  // D reg (l,i) = out[ 8*(l/16)+i ][ colbase + l%16 ], relative to the tile
  // origin. A clamped tail tile recomputed rows an earlier tile already
  // covered, so filtering on this tile's own disjoint range keeps every output
  // written exactly once.
  //
  // With split-K each split owns a partial sum, so the FP32 contributions go to
  // a per-split plane and a small reduce kernel adds and rounds ONCE. Using
  // atomicAdd here instead measured 2.6-3.8x slower (contended FP32 atomics on
  // the whole output).
  float* const part = (kSplits == 1) ? nullptr : partial + int64_t(blockIdx.z) * m * n;
#pragma unroll
  for (int t = 0; t < kMt; ++t) {
    const int64_t origin = m0 + tile_row[t];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int64_t row = origin + 8 * (lane >> 4) + i;
      if (row < own_lo[t] || row >= own_hi[t]) continue;
      const float v = acc[t][i] + bias_v;
      if (kSplits == 1) {
        out[row * n + col] = __float2bfloat16(v);
      } else {
        part[row * n + col] = v;
      }
    }
  }
#endif  // NF12_AD_HAS_WMMA
}

// Sum the per-split FP32 planes and round to BF16 exactly once, so the result
// is identical to a single-pass FP32 accumulation followed by one rounding
// (splitting K only changes the association order of the FP32 adds).
__global__ void wmma_reduce_kernel(const float* __restrict__ partial, __hip_bfloat16* __restrict__ out,
                                   int64_t total, int splits) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += int64_t(gridDim.x) * blockDim.x) {
    float s = partial[i];
    for (int j = 1; j < splits; ++j) s += partial[int64_t(j) * total + i];
    out[i] = __float2bfloat16(s);
  }
}

}  // namespace
}  // namespace nf12_hip

using namespace nf12_hip;

extern "C" {

// Largest M this route takes. Each grid.y block decodes the weight for its own
// M slice, so above the crossover the extra packed reads lose to
// decode-then-rocBLAS, which decodes once and runs a full GEMM. 128 tokens is
// the measured crossover on gfx1201 (see the README table); beyond it the
// dispatcher keeps the rocBLAS route.
int nf12_ad_wmma_max_m() { return kMaxTokens; }

// `__gfx1201__` is defined only while compiling DEVICE code, so a host function
// cannot use it: guarding on it silently reports "unsupported" and the route is
// never taken. Ask the runtime instead. gfx1250 has no wmma-128b-insts, so it
// stays on the decode-then-GEMM route even though the file compiles there.
static int wmma_arch_ok(int device) {
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, device) != hipSuccess) return 0;
  const char* arch = props.gcnArchName;
  return (arch[0] == 'g' && arch[1] == 'f' && arch[2] == 'x' && arch[3] == '1' &&
          arch[4] == '2' && arch[5] == '0' && (arch[6] == '0' || arch[6] == '1'));
}

int nf12_ad_wmma_supported(int64_t m) {
  if (m < 17 || m > kMaxTokens) return 0;
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) return 0;
  return wmma_arch_ok(device);
}

// Split workspace, cached per device.
//
// It must NOT be allocated per call: hipMalloc/hipFree are synchronising, and
// doing them around every launch measured +110 us at m=32 (147 us instead of
// 38 us), which completely erased the win split-K buys. The buffer is a plain
// process-wide scratch pointer that never escapes this file, grows on demand,
// and is sized once per (splits, m, n) shape seen.
static float* g_partial = nullptr;
static size_t g_partial_bytes = 0;

static hipError_t ensure_partial(size_t bytes) {
  if (bytes <= g_partial_bytes) return hipSuccess;
  if (g_partial != nullptr) {
    if (hipFree(g_partial) != hipSuccess) return hipErrorOutOfMemory;
    g_partial = nullptr;
    g_partial_bytes = 0;
  }
  if (hipMalloc(&g_partial, bytes) != hipSuccess) {
    g_partial = nullptr;
    return hipErrorOutOfMemory;
  }
  g_partial_bytes = bytes;
  return hipSuccess;
}

// Split count for a given problem. Measured on gfx1201 (see README):
//   * Below ~4 waves of CTAs (a wave is one CTA per CU, 32 CUs) the kernel is
//     latency-starved, and splitting K is nearly free because each split still
//     reads a DISJOINT k slice.
//   * At or above ~4 waves the machine is already saturated and the extra
//     partial planes and reduce pass cost more than the parallelism buys.
// So the split is chosen from the CTA count only.
static int pick_splits(int64_t blocks_x, int64_t m) {
  // gfx1201 has 32 CUs, so 32 CTAs is one wave. Only the single-wave case is
  // latency-starved enough for split-K to pay. Measured at N=11008 (86 CTAs,
  // ~2.7 waves) split-K LOSES: 146 us unsplit vs 182 us at split 2. So the
  // rule is "split only when the launch is under ~1.2 waves".
  constexpr int64_t kOneWave = 32;
  if (blocks_x > kOneWave + kOneWave / 4) return 1;
  return (m <= 64) ? 4 : 2;
}

// `out` is [m, n] row-major BF16; `x` is [m, k].
//
// Tiles per CTA are chosen so the CTA covers the caller's M with as few
// instantiated tiles as possible: a 32-token call runs a 2-tile kernel instead
// of an 8-tile one. Both cover the same tokens; the 2-tile version just skips
// matrix-core work (and shared-memory traffic) that the tail tiles would have
// thrown away. Every tile count in [1, kMaxTiles] is instantiated so the choice
// stays a compile-time bound inside the kernel.
hipError_t nf12_ad_linear_wmma(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                              const void* x, void* out, int64_t n, int64_t k, int64_t m,
                              const void* bias, hipStream_t stream) {
  if (exp_base > uint32_t(kMaxExpBase)) return hipErrorInvalidValue;
  if (m < 17 || m > nf12_ad_wmma_max_m()) return hipErrorInvalidValue;
  if (n % kSuperAtomRows != 0) return hipErrorInvalidValue;

  const int64_t n_blocks = (n / kSuperAtomRows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t m_blocks = (m + kMaxTokens - 1) / kMaxTokens;
  if (n_blocks == 0 || m_blocks == 0) return hipSuccess;
  if (n_blocks > 0x7fffLL || m_blocks > 0x7fffLL) return hipErrorInvalidValue;

  const int tiles = static_cast<int>((m + 15) / 16);
  const int splits = pick_splits(n_blocks, m);

  float* partial = nullptr;
  if (splits > 1) {
    const size_t bytes = size_t(splits) * size_t(m) * size_t(n) * sizeof(float);
    const hipError_t alloc = ensure_partial(bytes);
    if (alloc != hipSuccess) return alloc;
    partial = g_partial;
  }

  const dim3 grid(static_cast<unsigned>(n_blocks), static_cast<unsigned>(m_blocks),
                  static_cast<unsigned>(splits));

#define NF12_AD_LAUNCH(T, S)                                                             \
  wmma_kernel<T, S><<<grid, kThreadsPerBlock, 0, stream>>>(                              \
      reinterpret_cast<const __hip_bfloat16*>(x), reinterpret_cast<const uint4*>(packed), \
      escapes, exp_base_x4(exp_base), n, k, m, reinterpret_cast<__hip_bfloat16*>(out),    \
      reinterpret_cast<const __hip_bfloat16*>(bias), partial)

  // (tile count, split count) pairs the dispatcher can reach. Split 1 needs no
  // reduce pass; every other case reduces from a partial plane.
#define NF12_AD_DISPATCH(T)                     \
  do {                                          \
    if (splits == 1) {                          \
      NF12_AD_LAUNCH(T, 1);                     \
    } else if (splits == 2) {                   \
      NF12_AD_LAUNCH(T, 2);                     \
    } else {                                    \
      NF12_AD_LAUNCH(T, 4);                     \
    }                                           \
  } while (0)

  switch (tiles) {
    case 1: NF12_AD_DISPATCH(1); break;
    case 2: NF12_AD_DISPATCH(2); break;
    case 3: NF12_AD_DISPATCH(3); break;
    case 4: NF12_AD_DISPATCH(4); break;
    case 5: NF12_AD_DISPATCH(5); break;
    case 6: NF12_AD_DISPATCH(6); break;
    case 7: NF12_AD_DISPATCH(7); break;
    default: NF12_AD_DISPATCH(kMaxTiles); break;
  }
#undef NF12_AD_DISPATCH
#undef NF12_AD_LAUNCH

  hipError_t status = hipGetLastError();
  if (status == hipSuccess && splits > 1) {
    const int64_t total = m * n;
    const int reduce_blocks = static_cast<int>((total + 255) / 256);
    wmma_reduce_kernel<<<reduce_blocks, 256, 0, stream>>>(
        partial, reinterpret_cast<__hip_bfloat16*>(out), total, splits);
    status = hipGetLastError();
  }
  return status;
}

}  // extern "C"
