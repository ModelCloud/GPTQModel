// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12-AD (NyoomFloat12) fused decode+GEMM using RDNA4 matrix cores.
//
// Credits: NyoomFloat12 algorithm by Liberman, Fang, Zhang, Dao and Fu
// (Together AI, 2026), https://www.together.ai/. NF12 format, reference codec
// and CUDA kernels by AlpinDale, https://github.com/dphnAI/nf12 (Apache-2.0).
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

// Shared-memory row stride for the decoded super-atom. A plain 64 (128 bytes)
// makes every lane in a row group hit the same LDS banks: the 16x64 store is
// 8-way conflicted and the B-fragment load 16-way. Eight bf16 of padding
// (144 bytes) turns 144/4 = 36 banks-per-row, so consecutive rows rotate by
// four banks.
//
// The padding costs 4 KiB of LDS per CTA (36 vs 32 KiB), which drops the
// resident CTAs per CU from 4 to 3. That is free while the whole launch fits
// in three CTAs per CU and a loss beyond it, so the launcher picks kPad from
// the grid size (n_blocks * m_blocks <= 96) rather than paying it always.
constexpr int kLdPadded = kSuperAtomCols + 8;

// Maximum K split used by the launcher. Each split walks a disjoint slice of
// the K walk, so this multiplies the CTA count without duplicating any read.
constexpr int kMaxSplits = 8;

// CTAs resident on all 32 CUs at one per CU: one wave on this part.
constexpr int64_t kCtasPerWave = 32;

// kMt is the number of 16-token tiles this instantiation covers.
template <int kMt, int kSplits, bool kDbl, bool kPad>
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
  constexpr int kLd = kPad ? kLdPadded : kSuperAtomCols;
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
  // Two buffers per warp when kDbl. Decoding into the idle buffer while the
  // busy one is consumed removes the barrier between the store and the first
  // matrix-core read. It is not free: the second buffer costs 16 KiB of LDS and
  // ~20 VGPRs (measured 169-188 vs 67-167), which lowers the resident CTA count
  // per CU, so the launcher enables it only where that still pays (kMt <= 6).
  __shared__ __align__(16) __hip_bfloat16 tile[kDbl ? 2 : 1][kWarpsPerBlock]
                                                 [kSuperAtomRows * kLd];

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
  // Tile t covers rows [m0 + 16t, m0 + 16t + 16). A partial tail tile clamps
  // its load origin to the last full window so it cannot read past x; the
  // epilogue filters on the tile's own disjoint range. Computing these inline
  // instead of in three per-tile arrays keeps ~48 int64 values out of the VGPR
  // budget; at kMt=8 that is the difference between 167 and ~127 VGPRs.

  WmmaD acc[kMt];
#pragma unroll
  for (int t = 0; t < kMt; ++t) {
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[t][i] = 0.0f;
  }

  __hip_bfloat16* const sh0 = &tile[0][warp][0];
  __hip_bfloat16* const sh1 = &tile[kDbl ? 1 : 0][warp][0];

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
    // Scatter the dwords straight into the row-major tile instead of running
    // quad_transpose first. quad_transpose is a 4x4 dword transpose inside
    // each 16-byte chunk: it costs 16 ds_bpermute_b32 + 32 selects per
    // super-atom, and every ds_bpermute occupies the same LDS crossbar as the
    // store and the B-fragment load.
    //
    // The transpose is self-inverse, so undoing it on the address is
    // equivalent: dword d of lane l's chunk (h,q) belongs to dword (lane&3) of
    // chunk (d + 4q) in row (lane>>2) + 8h. That is 16 STS.32 whose bank is
    // 4*(lane>>2) + (lane&3) + const, i.e. all 32 banks exactly once.
    const int64_t row0 = int64_t(lane >> 2);
    const int64_t col0 = int64_t(lane & 3) * 2;  // dword position, in bf16
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const uint4 v = xr[h][q];
        const uint32_t* const w = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          *reinterpret_cast<uint32_t*>(
              dst + (row0 + 8 * h) * kLd + (d + 4 * q) * 8 + col0) = w[d];
        }
      }
    }
  };

  // 4 k-chunks of 16, each feeding one B fragment. A depends on both the token
  // tile (t) and the k-chunk (j), so it cannot be hoisted out of either loop:
  // each (t, j) pair is a distinct 16-byte row fragment. What IS shared is B,
  // which is loaded once per j and reused by every tile. The A loads hit L1
  // because the four j-chunks for a row cover consecutive 16-byte spans of the
  // same two cache lines.
  auto do_mma = [&](__hip_bfloat16* sh, int64_t kb) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const WmmaA b = wmma_row_frag(sh + 16 * j, kLd, lane);
#pragma unroll
      for (int t = 0; t < kMt; ++t) {
        // `wmma_row_frag` supplies the tile's own row ((lane & 15)) itself, so
        // the base passed here is the tile origin only. Adding the lane row
        // here as well applied it twice and produced garbage for every m > 16.
        const int64_t tr = max(int64_t(0), min(m0 + int64_t(t) * 16, m - 16));
        const WmmaA a = wmma_row_frag(x + tr * k + kb * kSuperAtomCols + 16 * j, k, lane);
        acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc[t]);
      }
    }
  };

  // Prefetch the next super-atom's packed bytes into registers before the
  // current one is consumed, so the three plane loads overlap the decode and
  // the matrix-core work.
  uint4 planes[3], next[3];
  const bool have_planes = kb_lo < kb_hi;

  if (kDbl) {
    // Decode the NEXT super-atom into the idle buffer while this one is
    // consumed, so the store no longer waits on the barrier before the first
    // matrix-core read. Bit-identical to the single-buffer path (checked by
    // checksum on gfx1201); 6% faster at 4096x4096/m=64 and 8-24% at
    // 11008x4096 for m=64..96.
    int buf = 0;
    if (have_planes) {
      load_planes(kb_lo, planes);
      decode_into(planes, sh0);
      __syncwarp();
    }
    for (int64_t kb = kb_lo; kb < kb_hi; ++kb) {
      __hip_bfloat16* const sh = (buf == 0) ? sh0 : sh1;
      // Issue the next super-atom's loads BEFORE the matrix-core work and
      // decode afterwards, so the L3 latency is covered by the mma instead of
      // stalling the decode. The decode writes the buffer do_mma is not
      // reading, so no extra barrier is needed.
      if (kb + 1 < kb_hi) load_planes(kb + 1, next);
      do_mma(sh, kb);
      if (kb + 1 < kb_hi) decode_into(next, (buf == 0) ? sh1 : sh0);
      __syncwarp();
      buf ^= 1;
    }
  } else {
    // Single buffer: the store has to wait for the barrier, so the decode does
    // not overlap the matrix-core issue. Kept for kMt >= 7, where the second
    // buffer pushed VGPRs to 188 and halved resident CTAs per CU (m=128 at
    // N=11008 regressed 184 -> 289 us before this split).
    if (have_planes) load_planes(kb_lo, planes);
    for (int64_t kb = kb_lo; kb < kb_hi; ++kb) {
      if (kb + 1 < kb_hi) load_planes(kb + 1, next);
      decode_into(planes, sh0);
      __syncwarp();
      do_mma(sh0, kb);
      __syncwarp();
      if (kb + 1 < kb_hi) {
#pragma unroll
        for (int p = 0; p < 3; ++p) planes[p] = next[p];
      }
    }
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
    const int64_t lo = m0 + int64_t(t) * 16;
    const int64_t hi = min(lo + 16, m);
    // Absolute tile origin, clamped for a partial tail. This is the row the
    // accumulator fragment starts at; `m0 + origin` double-counted m0 and only
    // worked while m_blocks == 1 (m <= 128).
    const int64_t origin = max(int64_t(0), min(lo, m - 16));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int64_t row = origin + 8 * (lane >> 4) + i;
      if (row < lo || row >= hi) continue;
      // With split-K each split writes its own FP32 partial plane and the
      // reduce kernel sums them, so bias must be added exactly once across
      // the splits (adding it in every split multiplied it by kSplits).
      const float v = acc[t][i] + ((kSplits == 1 || blockIdx.z == 0) ? bias_v : 0.0f);
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
// decode-then-rocBLAS, which decodes once and runs a full GEMM. 256 tokens is
// the measured limit on gfx1201 (see the README table); beyond it the
// dispatcher keeps the rocBLAS route.
int nf12_ad_wmma_max_m() { return 2 * kMaxTokens; }

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
  if (m < 16 || m > 2 * kMaxTokens) return 0;
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) return 0;
  return wmma_arch_ok(device);
}

// Split count for a given problem. A split reads a DISJOINT k slice, so the
// packed weight is never read twice; the cost is the per-split FP32 partial
// plane plus the reduce pass.
//
// The constants below are measured on gfx1201 (best split at m=16 / m=128),
// on the k x n grid of bench/common.py. blocks_x is n/16/8 (one warp per
// super-atom row, eight warps per CTA) and k_blocks is k/64:
//
//   4096x4096    32 CTAs,  64 k-blocks   s4 / s2
//   4096x14336  112 CTAs,  64 k-blocks   s1 / s1
//   14336x4096   32 CTAs, 224 k-blocks   s1 / s1
//   4096x28672  224 CTAs,  64 k-blocks   s4 / s1
//   5120x25600  200 CTAs,  80 k-blocks   s4 / s1
//   25600x5120   40 CTAs, 400 k-blocks   s1 / s1
//   8192x8192    64 CTAs, 128 k-blocks   s1 / s1
//   8192x57344  448 CTAs, 128 k-blocks   s2 / s2
//   2048x8192    64 CTAs,  32 k-blocks   s1 / s1
//   896x4864     38 CTAs,  14 k-blocks   s1 / s2
static int pick_splits(int64_t blocks_x, int64_t m, int64_t k_blocks) {
  constexpr int64_t kOneWave = kCtasPerWave;
  // Narrow N: 7 CTAs is under a quarter wave, and every extra CTA is free
  // because the splits read disjoint k slices.
  if (blocks_x * 4 < kOneWave) return 8;
  // Very wide N: the weight is HBM-resident and a 2-way split hides the extra
  // latency (8192x57344 m=128: 2598 -> 2418 us). It costs 0.3% at m=16.
  if (blocks_x >= 8 * kOneWave) return 2;
  if (blocks_x > kOneWave + kOneWave / 4) {
    // Wide but short K still benefits from more CTAs at small M
    // (5120x25600 m=16: 371 -> 360 us); a long K does not.
    if (m <= 64 && k_blocks < kOneWave * 4 && blocks_x >= 4 * kOneWave) return 4;
    return 1;
  }
  // One wave or less. Split only when there is enough K to give every split a
  // real slice: 4096x4096 (64 k-blocks) goes 27.3 -> 25.7 us at m=16, while
  // 896x4864 (14 k-blocks) is faster unsplit because the per-CTA prologue
  // dominates once each split only walks three super-atoms.
  //
  // Four ways pays up to m=32 and not beyond. At kMt <= 2 the VGPR budget
  // leaves room for the extra CTAs (4096x4096 m=32: 30.1 -> 27.1 us), but from
  // kMt = 4 the ~135 VGPRs already cap a CU at three CTAs, so the fourth split
  // only queues (m=64: 38.2 -> 43.3 us).
  if (k_blocks >= kOneWave && k_blocks < kOneWave * 4) return (m <= 32) ? 4 : 2;
  if (m > 64 && k_blocks < kOneWave) return 2;
  return 1;
}

struct WmmaLaunchPlan {
  int tiles;
  int splits;
  int64_t n_blocks;
  int64_t m_blocks;
};

static WmmaLaunchPlan wmma_launch_plan(int64_t n, int64_t k, int64_t m) {
  const int64_t n_blocks = (n / kSuperAtomRows + kWarpsPerBlock - 1) / kWarpsPerBlock;

  const int full_tiles = static_cast<int>((m + 15) / 16);
  // At kMt=8 the 8x8 FP32 accumulator costs 64 VGPRs and the kernel lands at
  // 167 VGPRs, which fits about one CTA per CU. Halving the tile count and
  // splitting M across CTAs doubles the packed read but restores the occupancy,
  // and that trade only pays when the second read is cheap. Two cases qualify:
  //
  //   * long K, where the matrix-core work amortises the extra read
  //     (8192x8192 m=128 0.68 -> 0.88x, 14336x4096 0.79 -> 0.83x);
  //   * a narrow N, where the doubled grid is still only ~2.5 waves
  //     (4096x4096 m=128: 67.2 -> 55.0 us, 896x4864 m=128: 30.2 -> 20.6 us).
  //
  // 2048x8192 has 64 column blocks, so splitting M makes 128 CTAs *and* reads
  // the weight twice, which measured 58.6 -> 85.2 us; it stays on 8 tiles.
  // Wide N (>64 blocks) is excluded outright, and above m=128 there are already
  // two M-blocks so the launch is not starved and splitting only re-reads.
  const int64_t k_blocks = k / kSuperAtomCols;
  const bool split_m = full_tiles > 4 && full_tiles <= 8 && n_blocks <= 64 &&
                       (k_blocks >= 128 || n_blocks <= 40);
  const int tiles = split_m ? 4 : full_tiles;
  // The dispatcher clamps anything above kMaxTiles to the kMaxTiles kernel, so
  // the tokens one CTA actually covers is min(tiles, kMaxTiles)*16. Deriving
  // m_blocks from the UNCLAMPED tile count dropped the second M block at
  // m=256 (tiles=16 -> m_blocks=1, but only 8 tiles are launched).
  const int64_t launch_tiles = (tiles < kMaxTiles) ? tiles : kMaxTiles;
  const int64_t m_blocks = (m + launch_tiles * 16 - 1) / (launch_tiles * 16);
  return {tiles, pick_splits(n_blocks * m_blocks, m, k_blocks), n_blocks, m_blocks};
}

// The Torch binding allocates this many bytes from its stream-aware caching
// allocator, so two in-flight calls never share a split-K partial plane.
size_t nf12_ad_wmma_workspace_bytes(int64_t n, int64_t k, int64_t m) {
  const auto plan = wmma_launch_plan(n, k, m);
  return plan.splits > 1 ? size_t(plan.splits) * size_t(m) * size_t(n) * sizeof(float) : 0;
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
                              const void* bias, void* workspace, size_t workspace_bytes,
                              hipStream_t stream) {
  if (exp_base > uint32_t(kMaxExpBase)) return hipErrorInvalidValue;
  if (m < 16 || m > nf12_ad_wmma_max_m()) return hipErrorInvalidValue;
  if (n % kSuperAtomRows != 0) return hipErrorInvalidValue;

  const auto plan = wmma_launch_plan(n, k, m);
  const int64_t n_blocks = plan.n_blocks;
  const int64_t m_blocks = plan.m_blocks;
  if (n_blocks == 0 || m_blocks == 0) return hipSuccess;
  if (n_blocks > 0x7fffLL || m_blocks > 0x7fffLL) return hipErrorInvalidValue;
  const int tiles = plan.tiles;
  const int splits = plan.splits;
  const size_t needed = nf12_ad_wmma_workspace_bytes(n, k, m);
  if (needed > 0 && (workspace == nullptr || workspace_bytes < needed)) return hipErrorInvalidValue;
  float* partial = static_cast<float*>(workspace);

  const dim3 grid(static_cast<unsigned>(n_blocks), static_cast<unsigned>(m_blocks),
                  static_cast<unsigned>(splits));

#define NF12_AD_LAUNCH(T, S, D, P)                                                          \
  wmma_kernel<T, S, D, P><<<grid, kThreadsPerBlock, 0, stream>>>(                           \
      reinterpret_cast<const __hip_bfloat16*>(x), reinterpret_cast<const uint4*>(packed), \
      escapes, exp_base_x4(exp_base), n, k, m, reinterpret_cast<__hip_bfloat16*>(out),    \
      reinterpret_cast<const __hip_bfloat16*>(bias), partial)

  // (tile count, split count) pairs the dispatcher can reach. Split 1 needs no
  // reduce pass; every other case reduces from a partial plane.
#define NF12_AD_DISPATCH(T, D, P)               \
  do {                                          \
    if (splits == 1) {                          \
      NF12_AD_LAUNCH(T, 1, D, P);               \
    } else if (splits == 2) {                   \
      NF12_AD_LAUNCH(T, 2, D, P);               \
    } else if (splits == 4) {                   \
      NF12_AD_LAUNCH(T, 4, D, P);               \
    } else {                                    \
      NF12_AD_LAUNCH(T, 8, D, P);               \
    }                                           \
  } while (0)

  // Double buffering only pays while the extra LDS and VGPRs still leave room
  // for more than one CTA per CU; above six tiles they do not (m=128 at
  // N=11008 regressed 184 -> 289 us). Each tile count therefore instantiates
  // exactly one of the two variants, so the kernel count is unchanged.
  //
  // The padded shared stride (36 KiB/CTA instead of 32) costs one resident CTA
  // per CU. That is free while the whole launch fits under three CTAs per CU
  // and a loss beyond it, so the launcher picks the variant from the grid size.
  const bool pad = n_blocks * m_blocks <= 96;
  // Double buffering is only worth its shared memory where shared memory is the
  // binding occupancy limit, and that is only true for the smallest tile counts.
  // A padded CTA is 18 KiB single-buffered and 36 KiB double-buffered against
  // 128 KiB per CU, so it holds 7 or 3 CTAs. At kMt <= 2 the accumulator is
  // small enough that the VGPR budget allows well over three CTAs, so the LDS
  // footprint decides; from kMt = 3 up the ~135 VGPRs already cap the CU at
  // three CTAs and dropping the buffer buys nothing (25600x5120 m=128 measured
  // 447 us buffered against 479 us not).
  //
  // Where LDS does bind, dropping the buffer only helps in a narrow band: the
  // launch must be starved enough that one extra resident CTA matters, but not
  // so starved that the buffer's own latency hiding is the bigger win.
  // 4096x4096 m=16 is 128 CTAs = four waves against a cap of three, and the
  // buffer costs 28.4 -> 23.5 us. The wide shapes are 14-28 waves deep, where
  // the cap does not bite at all and removing the buffer only loses
  // (4096x28672 m=16 measured 1.14 -> 1.08x). 2048x8192 m=16 is two waves, the
  // three still fit, and the buffer is worth 30.5 -> 25.9 us.
  const int64_t ctas = n_blocks * m_blocks * splits;
  const int lds_kib = pad ? 36 : 32;
  const int64_t waves = (ctas + kCtasPerWave - 1) / kCtasPerWave;
  const bool lds_starved = waves > 128 / lds_kib && waves <= 4;
  const bool dbl = (tiles <= 6) && (tiles > 2 || !lds_starved);
  // Only the two smallest tile counts get the runtime buffer choice; from three
  // tiles up the buffer is unconditional, so those stay single-instantiation.
#define NF12_AD_TILES(T, D)                                                     \
  do {                                                                          \
    if (pad) {                                                                  \
      NF12_AD_DISPATCH(T, D, true);                                             \
    } else {                                                                    \
      NF12_AD_DISPATCH(T, D, false);                                            \
    }                                                                           \
  } while (0)
#define NF12_AD_TILES_DYN(T)                                                    \
  do {                                                                          \
    if (dbl) {                                                                  \
      NF12_AD_TILES(T, true);                                                   \
    } else {                                                                    \
      NF12_AD_TILES(T, false);                                                  \
    }                                                                           \
  } while (0)
  switch (tiles) {
    case 1: NF12_AD_TILES_DYN(1); break;
    case 2: NF12_AD_TILES_DYN(2); break;
    case 3: NF12_AD_TILES(3, true); break;
    case 4: NF12_AD_TILES(4, true); break;
    case 5: NF12_AD_TILES(5, true); break;
    case 6: NF12_AD_TILES(6, true); break;
    case 7: NF12_AD_TILES(7, false); break;
    default: NF12_AD_TILES(kMaxTiles, false); break;
  }
#undef NF12_AD_TILES_DYN
#undef NF12_AD_TILES
#undef NF12_AD_DISPATCH
#undef NF12_AD_LAUNCH

  hipError_t status = hipGetLastError();
  if (status == hipSuccess && splits > 1) {
    const int64_t total = m * n;
    // Cap the natural (total+255)/256 grid at 128 blocks: at exactly 256
    // blocks gfx1201 hits a pathological 14 us case (measured 3.3 us at 64,
    // 128 and 512). Small outputs keep their natural, smaller grid.
    int rb = static_cast<int>((total + 255) / 256);
    if (rb > 128) rb = 128;
    const int reduce_blocks = rb;
    wmma_reduce_kernel<<<reduce_blocks, 256, 0, stream>>>(
        partial, reinterpret_cast<__hip_bfloat16*>(out), total, splits);
    status = hipGetLastError();
  }
  return status;
}

}  // extern "C"
