// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12-AD (NyoomFloat12) encode for AMD RDNA4 (gfx1200 / gfx1201 / gfx1250).
//
// Credits: NyoomFloat12 algorithm by Liberman, Fang, Zhang, Dao and Fu
// (Together AI, 2026), https://www.together.ai/. NF12 format, reference codec
// and CUDA kernels by AlpinDale, https://github.com/dphnAI/nf12 (Apache-2.0).
//
// It is a direct transliteration of the vendored CUDA encoder
// (gptqmodel_ext/nf12/csrc/codec/encode.cu) and produces BYTE-IDENTICAL
// output: the same exp_base, the same packed bytes and the same escape order.
// It replaces the pure-Torch ``nf12_codec.encode`` fallback that previously ran
// this stage on the CPU (~691 ms for a 4096x4096 matrix).
//
// Port notes
// ----------
// The CUDA pipeline uses three CUB block primitives (BlockLoad/BlockScan/
// BlockStore). The ROCm SDK here ships no thrust/CUB, so scan_kernel is
// replaced by a hand-written block-wide exclusive scan with the same
// semantics. Everything else maps 1:1:
//
//   analyze_kernel  -> 3 per-warp histograms -> exp_base selection
//   count_kernel    -> per-unit escape counts
//   scan_kernel     -> exclusive prefix sum over unit counts (hand-written)
//   write_kernel    -> packed planes + escape table
//
// Numerical contract: ``exp_base`` must be chosen exactly as Python's
// ``choose_exp_base`` (minimum escapes, ties to the SMALLEST base), and the
// escape indices must follow flat group id order. The validator asserts
// bytes_match / escapes_match / exp_base against the Torch oracle.
// ---------------------------------------------------------------------------
#include "nf12_ad_format_hip.cuh"

#include <algorithm>
#include <functional>
#include <cstdint>

namespace nf12_hip {
namespace {

using nf12_hip::kAtomsPerSuperAtom;
using nf12_hip::kMaxExpBase;
using nf12_hip::kPlaneBytes;
using nf12_hip::kSuperAtomBytes;
using nf12_hip::kSuperAtomVecs;
using nf12_hip::kThreadsPerBlock;

using nf12_hip::kWarpsPerBlock;

constexpr int kHistBins = 128;  // B in [0, 120], padded
constexpr int kNumHists = 3;    // interval starts, interval ends, sentinel collisions
constexpr size_t kAnalyzeWorkspaceBytes = sizeof(unsigned long long) * kNumHists * kHistBins;
// Short batches expose more waves for router weights and cache-resident
// matrices. Two-super-atom units keep neighboring warps close along K while
// limiting the serial scan's cost on large weights. Units remain contiguous
// in flat super-atom order, so changing their size cannot reorder escapes.
constexpr int kAnalyzeSuperAtomsPerWarp = 4;
constexpr int kUnitSuperAtoms = 2;
constexpr int kScanThreads = 256;
constexpr int kScanItemsPerThread = 8;
constexpr int64_t kMaxGridX = 0x7fffffffLL;

int64_t num_super_atoms(int64_t n, int64_t k) {
  return (n / kSuperAtomRows) * (k / kSuperAtomCols);
}
int64_t num_units(int64_t num_sa) { return (num_sa + kUnitSuperAtoms - 1) / kUnitSuperAtoms; }

// ---------------------------------------------------------------------------
// analyze: histogram the feasible base interval per group
// ---------------------------------------------------------------------------
// Per-warp histograms limit same-address atomic contention. Shared bins fit
// kAnalyzeSuperAtomsPerWarp * 128 groups per warp. Global counts need 64 bits.
__global__ void __launch_bounds__(kThreadsPerBlock)
    analyze_kernel(const __hip_bfloat16* __restrict__ w, int64_t ldw, int64_t num_sa,
                   int64_t k_blocks, unsigned long long* __restrict__ hist) {
  __shared__ uint32_t sh[kWarpsPerBlock][kNumHists][kHistBins];
  uint32_t* flat = &sh[0][0][0];
  for (int i = threadIdx.x; i < kWarpsPerBlock * kNumHists * kHistBins; i += kThreadsPerBlock)
    flat[i] = 0;
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  uint32_t(&mine)[kNumHists][kHistBins] = sh[warp];
  const int64_t sa0 = int64_t(blockIdx.x) * kAnalyzeSuperAtomsPerWarp * kWarpsPerBlock + warp;
  for (int i = 0; i < kAnalyzeSuperAtomsPerWarp; ++i) {
    const int64_t sa = sa0 + int64_t(i) * kWarpsPerBlock;
    if (sa >= num_sa) break;
    AtomRegs r;
    load_atoms(w, ldw, super_atom(sa, k_blocks), lane, r);
#pragma unroll
    for (int j = 0; j < kAtomsPerSuperAtom; ++j) {
      const GroupStats st = group_stats(r[j]);
      const uint32_t lo = st.hmax > 7u ? st.hmax - 7u : 0u;
      const uint32_t hi = min(st.hmin, uint32_t(kMaxExpBase));
      if (lo <= hi) {
        atomicAdd(&mine[0][lo], 1u);
        atomicAdd(&mine[1][hi], 1u);
      }
      if (st.all_positive && st.hmin == st.hmax && st.hmin <= uint32_t(kMaxExpBase))
        atomicAdd(&mine[2][st.hmin], 1u);
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

// ---------------------------------------------------------------------------
// count: escapes per unit
// ---------------------------------------------------------------------------
// Each unit has at most kUnitSuperAtoms * 128 escapes.
__global__ void __launch_bounds__(kThreadsPerBlock)
    count_kernel(const __hip_bfloat16* __restrict__ w, int64_t ldw, int64_t num_sa,
                 int64_t k_blocks, uint32_t base_x4, uint64_t* __restrict__ unit_counts,
                 int64_t n_units) {
  const int lane = threadIdx.x & 31;
  const int64_t unit = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
  if (unit >= n_units) return;
  uint32_t count = 0;
  for (int i = 0; i < kUnitSuperAtoms; ++i) {
    const int64_t sa = unit * kUnitSuperAtoms + i;
    if (sa >= num_sa) break;
    AtomRegs r;
    load_atoms(w, ldw, super_atom(sa, k_blocks), lane, r);
#pragma unroll
    for (int j = 0; j < kAtomsPerSuperAtom; ++j) {
      Group g;
      count += __popc(__builtin_amdgcn_ballot_w32(!encode_group_fast(r[j], base_x4, g)));
    }
  }
  if (lane == 0) unit_counts[unit] = count;
}

// ---------------------------------------------------------------------------
// scan: exclusive prefix sum over unit counts, total in v[n]
// ---------------------------------------------------------------------------
// Hand-written replacement for cub::BlockScan. One block walks the array in
// tiles of kScanThreads * kScanItemsPerThread, carrying the running total.
// Each tile is read into registers BEFORE any write (the scan is in place and
// the tail must see the original values), zero padding preserves the sum, and
// the carry is advanced by the tile aggregate so the final v[n] is exact.
__global__ void __launch_bounds__(kScanThreads) scan_kernel(uint64_t* __restrict__ v, int64_t n) {
  __shared__ uint64_t warp_sums[kScanThreads / 32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  uint64_t carry = 0;
  int64_t offset = 0;
  while (offset < n) {
    const int valid = static_cast<int>(min(int64_t(kScanThreads * kScanItemsPerThread), n - offset));
    uint64_t items[kScanItemsPerThread];
    const int base = tid * kScanItemsPerThread;
#pragma unroll
    for (int i = 0; i < kScanItemsPerThread; ++i)
      items[i] = (base + i < valid) ? v[offset + base + i] : uint64_t(0);
    __syncthreads();

    // Sequential scan within this thread's items.
#pragma unroll
    for (int i = 1; i < kScanItemsPerThread; ++i) items[i] += items[i - 1];
    const uint64_t thread_total = items[kScanItemsPerThread - 1];

    // Warp-level exclusive scan of the per-thread totals.
    uint64_t inclusive = thread_total;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const uint64_t other = __shfl_up(inclusive, d);
      if (lane >= d) inclusive += other;
    }
    const uint64_t thread_prefix = inclusive - thread_total;
    if (lane == 31) warp_sums[warp] = inclusive;
    __syncthreads();

    uint64_t warp_prefix = 0;
    for (int wi = 0; wi < warp; ++wi) warp_prefix += warp_sums[wi];
    const uint64_t tile_total = [&] {
      uint64_t t = 0;
      for (int wi = 0; wi < kScanThreads / 32; ++wi) t += warp_sums[wi];
      return t;
    }();
    __syncthreads();

    const uint64_t thread_prefix_total = carry + warp_prefix + thread_prefix;
#pragma unroll
    for (int i = 0; i < kScanItemsPerThread; ++i) {
      if (base + i < valid) v[offset + base + i] = thread_prefix_total + (i ? items[i - 1] : uint64_t(0));
    }
    carry += tile_total;
    offset += valid;
    __syncthreads();
  }
  if (tid == 0) v[n] = carry;
}

// ---------------------------------------------------------------------------
// write: packed planes + escape table
// ---------------------------------------------------------------------------
// Host validation guarantees that escape indices fit the 32-bit L0 word.
__global__ void __launch_bounds__(kThreadsPerBlock)
    write_kernel(const __hip_bfloat16* __restrict__ w, int64_t ldw, int64_t num_sa,
                 int64_t k_blocks, uint32_t base_x4, const uint64_t* __restrict__ unit_offsets,
                 int64_t n_units, uint4* __restrict__ packed, uint4* __restrict__ escapes) {
  const int lane = threadIdx.x & 31;
  const int64_t unit = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
  if (unit >= n_units) return;
  const uint32_t lt_mask = (1u << lane) - 1u;
  uint32_t next = static_cast<uint32_t>(unit_offsets[unit]);
  for (int i = 0; i < kUnitSuperAtoms; ++i) {
    const int64_t sa = unit * kUnitSuperAtoms + i;
    if (sa >= num_sa) break;
    AtomRegs r;
    load_atoms(w, ldw, super_atom(sa, k_blocks), lane, r);
    Group g[kAtomsPerSuperAtom];
    // Escape indices follow flat group id order: atom-major, then lane.
#pragma unroll
    for (int j = 0; j < kAtomsPerSuperAtom; ++j) {
      const bool esc = !encode_group_fast(r[j], base_x4, g[j]);
      const uint32_t ballot = __builtin_amdgcn_ballot_w32(esc);
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

}  // namespace
}  // namespace nf12_hip

using namespace nf12_hip;
namespace {

int64_t g_num_super_atoms(int64_t n, int64_t k) { return num_super_atoms(n, k); }
int64_t g_num_units(int64_t n, int64_t k) { return num_units(num_super_atoms(n, k)); }

}  // namespace

extern "C" {

size_t nf12_ad_encode_workspace_bytes(int64_t n, int64_t k) {
  const size_t unit_bytes = sizeof(uint64_t) * size_t(g_num_units(n, k) + 1);
  return std::max(kAnalyzeWorkspaceBytes, (unit_bytes + 15) & ~size_t(15));
}

hipError_t nf12_ad_encode_analyze(const void* w_in, int64_t n, int64_t k, int64_t ldw, void* workspace,
                                  size_t workspace_bytes, uint8_t* exp_base_out,
                                  uint32_t* num_escapes_out, hipStream_t stream) {
  // SPEC.md 4 geometry. Without this a k that is not a multiple of 64 is
  // truncated by k / kSuperAtomCols and the encoder silently writes a matrix
  // covering only part of the caller's K, which then reads back as garbage.
  if (n <= 0 || k <= 0 || n % kSuperAtomRows != 0 || k % kSuperAtomCols != 0 || ldw < k) {
    return hipErrorInvalidValue;
  }
  // The workspace is the caller's buffer: the histogram below is memset and
  // then written by the analyze kernel, so a short or null buffer must be
  // rejected before any device write.
  if (exp_base_out == nullptr || num_escapes_out == nullptr || workspace == nullptr ||
      workspace_bytes < nf12_ad_encode_workspace_bytes(n, k)) {
    return hipErrorInvalidValue;
  }
  const auto* w = reinterpret_cast<const __hip_bfloat16*>(w_in);
  const int64_t num_sa = g_num_super_atoms(n, k);
  auto* hist = static_cast<unsigned long long*>(workspace);
  hipError_t err = hipMemsetAsync(hist, 0, kAnalyzeWorkspaceBytes, stream);
  if (err != hipSuccess) return err;
  const int64_t sa_per_block = int64_t(kAnalyzeSuperAtomsPerWarp) * kWarpsPerBlock;
  const int64_t blocks = (num_sa + sa_per_block - 1) / sa_per_block;
  if (blocks > kMaxGridX) return hipErrorInvalidValue;
  if (blocks > 0) {
    analyze_kernel<<<static_cast<unsigned>(blocks), kThreadsPerBlock, 0, stream>>>(
        w, ldw, num_sa, k / kSuperAtomCols, hist);
    if ((err = hipGetLastError()) != hipSuccess) return err;
  }
  unsigned long long h[kNumHists][kHistBins];
  if ((err = hipMemcpyAsync(h, hist, sizeof(h), hipMemcpyDeviceToHost, stream)) != hipSuccess)
    return err;
  if ((err = hipStreamSynchronize(stream)) != hipSuccess) return err;

  // escapes(B) = groups - intervals_containing(B) + sentinel_collisions(B).
  const unsigned long long groups =
      static_cast<unsigned long long>(n) * static_cast<unsigned long long>(k) / 8;
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
  if (best > 0xffffffffull) return hipErrorInvalidValue;
  *exp_base_out = static_cast<uint8_t>(best_b);
  *num_escapes_out = static_cast<uint32_t>(best);
  return hipSuccess;
}

hipError_t nf12_ad_encode(const void* w_in, int64_t n, int64_t k, int64_t ldw, uint8_t exp_base,
                          uint8_t* packed, uint4* escapes, uint32_t escapes_capacity,
                          void* workspace, size_t workspace_bytes, hipStream_t stream) {
  if (n <= 0 || k <= 0 || n % kSuperAtomRows != 0 || k % kSuperAtomCols != 0 || ldw < k) {
    return hipErrorInvalidValue;
  }
  const auto* w = reinterpret_cast<const __hip_bfloat16*>(w_in);
  if (exp_base > kMaxExpBase) return hipErrorInvalidValue;
  if (packed == nullptr || workspace == nullptr ||
      workspace_bytes < nf12_ad_encode_workspace_bytes(n, k)) {
    return hipErrorInvalidValue;
  }
  if (escapes_capacity > 0 && escapes == nullptr) return hipErrorInvalidValue;
  const int64_t num_sa = g_num_super_atoms(n, k);
  const int64_t n_units = g_num_units(n, k);
  if (n_units == 0) return hipSuccess;
  auto* units = static_cast<uint64_t*>(workspace);
  const uint32_t base_x4 = exp_base_x4(exp_base);
  const int64_t k_blocks = k / kSuperAtomCols;
  const int64_t blocks64 = (n_units + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (blocks64 > kMaxGridX) return hipErrorInvalidValue;
  const auto blocks = static_cast<unsigned>(blocks64);
  count_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(w, ldw, num_sa, k_blocks, base_x4, units,
                                                        n_units);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) return err;
  scan_kernel<<<1, kScanThreads, 0, stream>>>(units, n_units);
  if ((err = hipGetLastError()) != hipSuccess) return err;
  // Check capacity and 32-bit index bounds before writing packed/escape outputs.
  uint64_t total;
  if ((err = hipMemcpyAsync(&total, units + n_units, sizeof(total), hipMemcpyDeviceToHost, stream)) !=
      hipSuccess)
    return err;
  if ((err = hipStreamSynchronize(stream)) != hipSuccess) return err;
  if (total > escapes_capacity || total > 0xffffffffull) return hipErrorInvalidValue;
  write_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(w, ldw, num_sa, k_blocks, base_x4, units,
                                                        n_units, reinterpret_cast<uint4*>(packed),
                                                        escapes);
  return hipGetLastError();
}

}  // extern "C"
