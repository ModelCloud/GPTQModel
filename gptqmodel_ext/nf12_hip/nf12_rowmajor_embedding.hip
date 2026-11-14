// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// Row-major NF12 embedding transform and lookup for AMD RDNA4.
//
// Matrix kernels keep canonical NF12's 16-row x 64-column super-atoms.
// Embedding lookup instead stores one 96-byte record for every row and
// 64-column block: 32 metadata bytes followed by 64 low-mantissa bytes.
// The transform selects one row half from each canonical paired-row group
// and materializes collision escapes so every BF16 bit remains exact.
// ---------------------------------------------------------------------------
#include "nf12_format_hip.cuh"
#include "rowmajor_format.cuh"

#include <cstdint>
#include <limits>

namespace nf12_hip {
namespace {

constexpr int kRowBlockBytes = 96;
constexpr int kRowMetaBytes = 32;
constexpr int kPackThreads = 128;

__global__ void count_row_collisions(const uint8_t* __restrict__ source,
                                     int32_t* __restrict__ counts) {
  const int64_t tile = blockIdx.x;
  const int tid = threadIdx.x;
  const uint32_t meta =
      reinterpret_cast<const uint32_t*>(source + tile * kSuperAtomBytes)[tid];
  const int count = nf12_rowmajor::collision_count(meta);
  if (count != 0) {
    atomicAdd(counts + tile, count);
  }
}

__global__ void pack_rowmajor(const uint8_t* __restrict__ source,
                              int64_t original_escapes, int64_t k_blocks,
                              uint32_t base,
                              const int64_t* __restrict__ prefix,
                              uint8_t* __restrict__ dest,
                              uint64_t* __restrict__ dest_escapes,
                              int32_t* __restrict__ invalid_escape) {
  const int64_t tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid >> 2;
  const int atom = tid & 3;
  const uint8_t* src = source + tile * kSuperAtomBytes;
  const uint32_t meta = reinterpret_cast<const uint32_t*>(src)[tid];
  const int plane = atom < 2 ? 1 : 2;
  const int lo_index = lane * 4 + 2 * (atom & 1);
  const uint32_t* lows_plane =
      reinterpret_cast<const uint32_t*>(src + plane * kPlaneBytes);
  const uint32_t lo0 = lows_plane[lo_index];
  const uint32_t lo1 = lows_plane[lo_index + 1];
  const uint16_t lower_meta = nf12_rowmajor::selected_meta(meta, 0);
  const uint16_t upper_meta = nf12_rowmajor::selected_meta(meta, 1);
  const bool lower_collision = meta != 0u && lower_meta == 0u;
  const bool upper_collision = meta != 0u && upper_meta == 0u;

  int warp_prefix = nf12_rowmajor::collision_count(meta);
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset *= 2) {
    const int other = __shfl_up(warp_prefix, offset);
    if ((tid & (kWarpSize - 1)) >= offset) {
      warp_prefix += other;
    }
  }
  __shared__ int warp_totals[4];
  if ((tid & (kWarpSize - 1)) == kWarpSize - 1) {
    warp_totals[tid / kWarpSize] = warp_prefix;
  }
  __syncthreads();

  int prior_warps = 0;
#pragma unroll
  for (int warp = 0; warp < 4; ++warp) {
    if (warp < tid / kWarpSize) {
      prior_warps += warp_totals[warp];
    }
  }
  const int count = nf12_rowmajor::collision_count(meta);
  const int64_t prior_tiles = tile != 0 ? prefix[tile - 1] : 0;
  const uint32_t first_collision = static_cast<uint32_t>(
      2 * original_escapes + prior_tiles + prior_warps + warp_prefix - count);
  const int64_t block = tile / k_blocks;
  const int64_t kb = tile - block * k_blocks;
  const int ri = lane >> 2;
  const int group = 4 * atom + (lane & 3);

  if (meta == 0u && lo0 >= original_escapes) {
    atomicExch(invalid_escape, 1);
  }

#pragma unroll
  for (int upper = 0; upper < 2; ++upper) {
    const uint16_t selected = upper != 0 ? upper_meta : lower_meta;
    const uint32_t normal_lows = nf12_rowmajor::selected_lows(lo0, lo1, upper);
    const bool collision = upper != 0 ? upper_collision : lower_collision;
    const uint32_t escape_index =
        first_collision + (upper != 0 ? static_cast<int>(lower_collision) : 0);
    const uint32_t out_lows =
        meta == 0u ? static_cast<uint32_t>(lo0 * 2 + upper)
                   : collision ? escape_index : normal_lows;
    const int64_t row = block * kSuperAtomRows + ri + upper * 8;
    uint8_t* dst =
        dest + (row * k_blocks + kb) * kRowBlockBytes;
    reinterpret_cast<uint16_t*>(dst)[group] = selected;
    reinterpret_cast<uint32_t*>(dst + kRowMetaBytes)[group] = out_lows;
    if (collision) {
      dest_escapes[escape_index] = nf12_rowmajor::collision_escape(normal_lows, base);
    }
  }
}

__global__ void split_row_escapes(const uint4* __restrict__ source,
                                  int64_t original_count,
                                  uint64_t* __restrict__ dest) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= original_count) return;
  const uint4 bits = source[index];
  dest[2 * index] =
      static_cast<uint64_t>(bits.x) | (static_cast<uint64_t>(bits.z) << 32);
  dest[2 * index + 1] =
      static_cast<uint64_t>(bits.y) | (static_cast<uint64_t>(bits.w) << 32);
}

template <typename Id, int IdsPerCTA, int Threads>
__global__ void rowmajor_lookup_kernel(
    const Id* __restrict__ ids,
    const uint8_t* __restrict__ packed,
    const uint64_t* __restrict__ escapes,
    uint32_t num_escapes, uint32_t base,
    int64_t n, int64_t k_blocks, int64_t num_ids,
    uint4* __restrict__ out) {
  constexpr int kLanesPerId = 4;
  constexpr int kGroupsPerId = Threads / kLanesPerId / IdsPerCTA;
  const int lane = threadIdx.x & (kLanesPerId - 1);
  const int group = threadIdx.x / kLanesPerId;
  const int64_t id_pos =
      static_cast<int64_t>(blockIdx.x) * IdsPerCTA +
      group / kGroupsPerId;
  const int64_t kb =
      static_cast<int64_t>(blockIdx.y) * kGroupsPerId +
      group % kGroupsPerId;
  if (id_pos >= num_ids || kb >= k_blocks) return;

  const int64_t row = static_cast<int64_t>(ids[id_pos]);
  uint2 meta = make_uint2(0u, 0u);
  uint4 lows = make_uint4(0u, 0u, 0u, 0u);
  if (row >= 0 && row < n) {
    const uint8_t* src =
        packed + (row * k_blocks + kb) * kRowBlockBytes;
    meta = *reinterpret_cast<const uint2*>(src + lane * 8);
    lows = *reinterpret_cast<const uint4*>(src + kRowMetaBytes + lane * 16);
  }
  const bool valid = row >= 0 && row < n;
  const uint64_t meta_bits =
      static_cast<uint64_t>(meta.x) | (static_cast<uint64_t>(meta.y) << 32);
  const nf12_rowmajor::Quad q = valid
      ? nf12_rowmajor::decode_quad(meta_bits, lows, escapes, num_escapes, base)
      : nf12_rowmajor::Quad{};

  uint4* dst = reinterpret_cast<uint4*>(
      reinterpret_cast<__hip_bfloat16*>(out) +
      id_pos * k_blocks * kSuperAtomCols + kb * kSuperAtomCols);
  dst[2 * lane] = make_uint4(q.a.lo, q.b.lo, q.c.lo, q.d.lo);
  dst[2 * lane + 1] = make_uint4(q.a.hi, q.b.hi, q.c.hi, q.d.hi);
}

template <typename Id, int IdsPerCTA, int Threads>
hipError_t launch_lookup(const uint8_t* packed,
                         const uint64_t* escapes,
                         uint32_t num_escapes,
                         uint32_t exp_base,
                         int64_t n, int64_t k,
                         const Id* ids, int64_t num_ids,
                         void* out, hipStream_t stream) {
  constexpr int kLanesPerId = 4;
  constexpr int kGroupsPerId = Threads / kLanesPerId / IdsPerCTA;
  const int64_t k_blocks = k / kSuperAtomCols;
  const int64_t id_blocks = (num_ids + IdsPerCTA - 1) / IdsPerCTA;
  const int64_t kb_chunks =
      (k_blocks + kGroupsPerId - 1) / kGroupsPerId;
  if (id_blocks > std::numeric_limits<uint32_t>::max() ||
      kb_chunks > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  const dim3 grid(static_cast<uint32_t>(id_blocks),
                  static_cast<uint32_t>(kb_chunks));
  rowmajor_lookup_kernel<Id, IdsPerCTA, Threads><<<grid, Threads, 0, stream>>>(
      ids, packed, escapes, num_escapes, exp_base_x4(exp_base),
      n, k_blocks, num_ids, reinterpret_cast<uint4*>(out));
  return hipGetLastError();
}

template <typename Id>
hipError_t launch_lookup_any(const uint8_t* packed,
                             const uint64_t* escapes,
                             uint32_t num_escapes,
                             uint32_t exp_base,
                             int64_t n, int64_t k,
                             const Id* ids, int64_t num_ids,
                             void* out, hipStream_t stream) {
  return num_ids < 8
      ? launch_lookup<Id, 1, 128>(packed, escapes, num_escapes, exp_base,
                                  n, k, ids, num_ids, out, stream)
      : launch_lookup<Id, 8, 256>(packed, escapes, num_escapes, exp_base,
                                  n, k, ids, num_ids, out, stream);
}

}  // namespace
}  // namespace nf12_hip

extern "C" hipError_t nf12_rowmajor_count(
    const uint8_t* source, int64_t n, int64_t k,
    int32_t* counts, hipStream_t stream) {
  if (source == nullptr || counts == nullptr ||
      n <= 0 || n % nf12_hip::kSuperAtomRows != 0 ||
      k <= 0 || k % nf12_hip::kSuperAtomCols != 0) {
    return hipErrorInvalidValue;
  }
  const int64_t tiles =
      (n / nf12_hip::kSuperAtomRows) *
      (k / nf12_hip::kSuperAtomCols);
  if (tiles > std::numeric_limits<int32_t>::max()) {
    return hipErrorInvalidValue;
  }
  nf12_hip::count_row_collisions<<<static_cast<uint32_t>(tiles),
                                    nf12_hip::kPackThreads, 0, stream>>>(
      source, counts);
  return hipGetLastError();
}

extern "C" hipError_t nf12_rowmajor_pack(
    const uint8_t* source, const uint4* original_escapes,
    int64_t original_count, int64_t n, int64_t k,
    uint32_t exp_base, const int64_t* prefix,
    uint8_t* dest, uint64_t* dest_escapes,
    int32_t* invalid_escape, hipStream_t stream) {
  if (source == nullptr || prefix == nullptr || dest == nullptr ||
      invalid_escape == nullptr ||
      (original_count != 0 && original_escapes == nullptr) ||
      n <= 0 || n % nf12_hip::kSuperAtomRows != 0 ||
      k <= 0 || k % nf12_hip::kSuperAtomCols != 0 ||
      exp_base > nf12_hip::kMaxExpBase || original_count < 0) {
    return hipErrorInvalidValue;
  }
  const int64_t num_tiles =
      (n / nf12_hip::kSuperAtomRows) *
      (k / nf12_hip::kSuperAtomCols);
  if (num_tiles > std::numeric_limits<int32_t>::max()) {
    return hipErrorInvalidValue;
  }
  nf12_hip::pack_rowmajor<<<static_cast<uint32_t>(num_tiles),
                            nf12_hip::kPackThreads, 0, stream>>>(
      source, original_count, k / nf12_hip::kSuperAtomCols,
      exp_base, prefix, dest, dest_escapes, invalid_escape);
  if (hipError_t error = hipGetLastError(); error != hipSuccess) {
    return error;
  }
  if (original_count != 0) {
    const uint32_t blocks = static_cast<uint32_t>(
        (original_count + 255) / 256);
    nf12_hip::split_row_escapes<<<blocks, 256, 0, stream>>>(
        original_escapes, original_count, dest_escapes);
  }
  return hipGetLastError();
}

extern "C" hipError_t nf12_rowmajor_lookup(
    const void* ids, const uint8_t* packed,
    const uint64_t* escapes, uint32_t num_escapes,
    uint32_t exp_base, int64_t n, int64_t k,
    int64_t num_ids, void* out, int id_bytes,
    hipStream_t stream) {
  if (ids == nullptr || packed == nullptr || out == nullptr ||
      n <= 0 || k <= 0 || k % 64 != 0 ||
      exp_base > nf12_hip::kMaxExpBase ||
      num_ids < 0 || (id_bytes != 4 && id_bytes != 8)) {
    return hipErrorInvalidValue;
  }
  if (num_ids == 0) return hipSuccess;
  if (id_bytes == 4) {
    return nf12_hip::launch_lookup_any(
        packed, escapes, num_escapes, exp_base, n, k,
        static_cast<const int32_t*>(ids), num_ids, out, stream);
  }
  return nf12_hip::launch_lookup_any(
      packed, escapes, num_escapes,
      exp_base, n, k, static_cast<const int64_t*>(ids),
      num_ids, out, stream);
}
