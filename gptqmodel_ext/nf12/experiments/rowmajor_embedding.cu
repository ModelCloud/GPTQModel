// SPDX-License-Identifier: Apache-2.0
// Experimental packed-only NF12 row-major embedding. Each 64-output row block
// holds 32 metadata bytes and 64 mantissa bytes; escapes hold four BF16 bits.
#include <cuda_runtime.h>
#include <cstdint>

namespace {

struct Pair { uint32_t lo, hi; };

__device__ __forceinline__ Pair decode(uint16_t meta, uint32_t lows,
                                       const uint64_t* __restrict__ escapes,
                                       int64_t num_escapes, uint32_t base) {
  if (meta == 0) {
    if (lows < num_escapes) {
      const uint64_t v = __ldg(escapes + lows);
      return Pair{static_cast<uint32_t>(v), static_cast<uint32_t>(v >> 32)};
    }
    return Pair{0, 0};
  }
  const uint32_t m0 = meta & 0xffu, m2 = meta >> 8;
  const uint32_t h0 = ((m0 & 0x87u) + base) & 0xffu;
  const uint32_t h1 = ((((m0 << 1) & 0x80u) | ((m0 >> 3) & 7u)) + base) & 0xffu;
  const uint32_t h2 = ((m2 & 0x87u) + base) & 0xffu;
  const uint32_t h3 = ((((m2 << 1) & 0x80u) | ((m2 >> 3) & 7u)) + base) & 0xffu;
  const uint32_t lo = ((h0 << 8) | (lows & 0xffu)) |
                      (((h1 << 8) | ((lows >> 8) & 0xffu)) << 16);
  const uint32_t hi = ((h2 << 8) | ((lows >> 16) & 0xffu)) |
                      (((h3 << 8) | ((lows >> 24) & 0xffu)) << 16);
  return Pair{lo, hi};
}

template <typename T, bool CacheGlobal>
__device__ __forceinline__ T load_packed(const T* src) {
  if constexpr (CacheGlobal) return __ldcg(src);
  return __ldg(src);
}

template <bool CacheGlobal>
__device__ __forceinline__ void store_output(uint4* dst, uint4 value) {
  if constexpr (CacheGlobal) __stcg(dst, value);
  else *dst = value;
}

// 0: prior 8 lanes, 1: 4 lanes, 2: 4 lanes with streaming stores,
// 3: 4 lanes with L2-only packed loads, 4: both L2-only loads/stores,
// 5: 2 lanes, 6: 2 lanes with streaming stores.
template <typename Id, int IdsPerCTA, int Mode>
__global__ void rowmajor_kernel(const Id* __restrict__ ids,
                                const uint8_t* __restrict__ packed,
                                const uint64_t* __restrict__ escapes,
                                int64_t num_escapes, uint32_t base,
                                int64_t n, int64_t k, int64_t m,
                                uint64_t* __restrict__ out) {
  constexpr int lanes_per_block = Mode == 0 ? 8 : (Mode >= 5 ? 2 : 4);
  const int groups_per_id = (blockDim.x / lanes_per_block) / IdsPerCTA;
  const int group = threadIdx.x / lanes_per_block;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * IdsPerCTA + group / groups_per_id;
  const int64_t kb = static_cast<int64_t>(blockIdx.y) * groups_per_id + group % groups_per_id;
  if (i >= m || kb >= k / 64) return;
  const int lane = threadIdx.x & (lanes_per_block - 1);
  const int64_t row = static_cast<int64_t>(ids[i]);
  const bool valid = row >= 0 && row < n;
  if constexpr (Mode == 0) {
    uint64_t first = 0, second = 0;
    if (valid) {
      const uint8_t* src = packed + (row * (k / 64) + kb) * 96;
      const uint32_t mm = __ldg(reinterpret_cast<const uint32_t*>(src) + lane);
      const uint64_t ll = __ldg(reinterpret_cast<const uint64_t*>(src + 32) + lane);
      const Pair a = decode(static_cast<uint16_t>(mm), static_cast<uint32_t>(ll),
                            escapes, num_escapes, base);
      const Pair b = decode(static_cast<uint16_t>(mm >> 16), static_cast<uint32_t>(ll >> 32),
                            escapes, num_escapes, base);
      first = static_cast<uint64_t>(a.lo) | (static_cast<uint64_t>(b.lo) << 32);
      second = static_cast<uint64_t>(a.hi) | (static_cast<uint64_t>(b.hi) << 32);
    }
    uint64_t* dst = out + (i * k + kb * 64) / 4;
    const int off = (lane / 2) * 4 + (lane & 1);
    dst[off] = first;
    dst[off + 2] = second;
  } else {
    const uint8_t* src = valid ? packed + (row * (k / 64) + kb) * 96 : nullptr;
    uint4* dst = reinterpret_cast<uint4*>(out + (i * k + kb * 64) / 4);
    constexpr int atoms_per_lane = Mode >= 5 ? 2 : 1;
#pragma unroll
    for (int atom = 0; atom < atoms_per_lane; ++atom) {
      const int atom_index = lane * atoms_per_lane + atom;
      uint2 mm2 = make_uint2(0, 0);
      uint4 low = make_uint4(0, 0, 0, 0);
      if (valid) {
        mm2 = load_packed<uint2, Mode == 3 || Mode == 4>(
            reinterpret_cast<const uint2*>(src) + atom_index);
        low = load_packed<uint4, Mode == 3 || Mode == 4>(
            reinterpret_cast<const uint4*>(src + 32) + atom_index);
      }
      const uint64_t mm = static_cast<uint64_t>(mm2.x) | (static_cast<uint64_t>(mm2.y) << 32);
      const Pair a = valid ? decode(static_cast<uint16_t>(mm), low.x, escapes, num_escapes, base) : Pair{0, 0};
      const Pair b = valid ? decode(static_cast<uint16_t>(mm >> 16), low.y, escapes, num_escapes, base) : Pair{0, 0};
      const Pair c = valid ? decode(static_cast<uint16_t>(mm >> 32), low.z, escapes, num_escapes, base) : Pair{0, 0};
      const Pair d = valid ? decode(static_cast<uint16_t>(mm >> 48), low.w, escapes, num_escapes, base) : Pair{0, 0};
      store_output<Mode == 2 || Mode == 4 || Mode == 6>(
          dst + 2 * atom_index, make_uint4(a.lo, b.lo, c.lo, d.lo));
      store_output<Mode == 2 || Mode == 4 || Mode == 6>(
          dst + 2 * atom_index + 1, make_uint4(a.hi, b.hi, c.hi, d.hi));
    }
  }
}

// The transform maps one canonical 1536-byte super-atom to sixteen 96-byte
// row records. Each of 128 threads owns one canonical eight-value group.
// Count only selected-half zero-sentinel collisions; original escapes already
// have indices in the canonical table.
__device__ __forceinline__ int collision_count(uint32_t meta) {
  if (meta == 0) return 0;
  const bool lower = (meta & 0x00ff00ffu) == 0;
  const bool upper = (meta & 0xff00ff00u) == 0;
  return int(lower) + int(upper);
}

__global__ void count_row_collisions(const uint8_t* __restrict__ source,
                                     int32_t* __restrict__ counts) {
  const int tile = blockIdx.x;
  const int tid = threadIdx.x;
  const uint32_t* meta = reinterpret_cast<const uint32_t*>(source + int64_t(tile) * 1536);
  const int count = collision_count(meta[tid]);
  const int warp_sum = __reduce_add_sync(0xffffffffu, count);
  __shared__ int warp_counts[4];
  if ((tid & 31) == 0) warp_counts[tid >> 5] = warp_sum;
  __syncthreads();
  if (tid == 0) counts[tile] = warp_counts[0] + warp_counts[1] + warp_counts[2] + warp_counts[3];
}

__device__ __forceinline__ uint16_t selected_meta(uint32_t meta, int upper) {
  return uint16_t(((meta >> (8 * upper)) & 255u) |
                  (((meta >> (16 + 8 * upper)) & 255u) << 8));
}

__device__ __forceinline__ uint32_t selected_lows(uint32_t lo0, uint32_t lo1, int upper) {
  return ((lo0 >> (16 * upper)) & 65535u) | (((lo1 >> (16 * upper)) & 65535u) << 16);
}

__device__ __forceinline__ uint64_t collision_escape(uint32_t lows, uint32_t base) {
  const uint64_t high = uint64_t(base) << 8;
  return (high | (lows & 255u)) |
         ((high | ((lows >> 8) & 255u)) << 16) |
         ((high | ((lows >> 16) & 255u)) << 32) |
         ((high | ((lows >> 24) & 255u)) << 48);
}

__global__ void pack_rowmajor(const uint8_t* __restrict__ source,
                              int64_t original_escapes, int64_t k_blocks,
                              uint32_t base, const int64_t* __restrict__ prefix,
                              uint8_t* __restrict__ dest,
                              uint64_t* __restrict__ dest_escapes,
                              int32_t* __restrict__ invalid_escape) {
  const int tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid >> 2;
  const int atom = tid & 3;
  const uint8_t* src = source + int64_t(tile) * 1536;
  const uint32_t meta = reinterpret_cast<const uint32_t*>(src)[tid];
  const int plane = atom < 2 ? 1 : 2;
  const int lo_index = lane * 4 + 2 * (atom & 1);
  const uint32_t* lows_plane = reinterpret_cast<const uint32_t*>(src + plane * 512);
  const uint32_t lo0 = lows_plane[lo_index];
  const uint32_t lo1 = lows_plane[lo_index + 1];
  const uint16_t lower_meta = selected_meta(meta, 0);
  const uint16_t upper_meta = selected_meta(meta, 1);
  const bool lower_collision = meta != 0 && lower_meta == 0;
  const bool upper_collision = meta != 0 && upper_meta == 0;
  const int count = int(lower_collision) + int(upper_collision);

  // A 128-thread block-wide prefix gives stable, compact collision indices
  // without a full-weight flag tensor or atomic allocation order.
  int warp_prefix = count;
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    const int other = __shfl_up_sync(0xffffffffu, warp_prefix, offset);
    if ((tid & 31) >= offset) warp_prefix += other;
  }
  __shared__ int warp_totals[4];
  if ((tid & 31) == 31) warp_totals[tid >> 5] = warp_prefix;
  __syncthreads();
  int prior_warps = 0;
#pragma unroll
  for (int warp = 0; warp < 4; ++warp) {
    if (warp < (tid >> 5)) prior_warps += warp_totals[warp];
  }
  const int64_t prior_tiles = tile ? prefix[tile - 1] : 0;
  const uint32_t first_collision = uint32_t(2 * original_escapes + prior_tiles +
                                            prior_warps + warp_prefix - count);
  const int64_t block = tile / k_blocks;
  const int64_t kb = tile - block * k_blocks;
  const int ri = lane >> 2;
  const int group = 4 * atom + (lane & 3);
  if (meta == 0 && lo0 >= original_escapes) atomicExch(invalid_escape, 1);
#pragma unroll
  for (int upper = 0; upper < 2; ++upper) {
    const uint16_t selected = upper ? upper_meta : lower_meta;
    const uint32_t normal_lows = selected_lows(lo0, lo1, upper);
    const bool collision = upper ? upper_collision : lower_collision;
    const uint32_t escape_index = first_collision + (upper ? int(lower_collision) : 0);
    const uint32_t out_lows = meta == 0 ? uint32_t(lo0 * 2 + upper) :
                              collision ? escape_index : normal_lows;
    const int64_t row = block * 16 + ri + upper * 8;
    uint8_t* dst = dest + (row * k_blocks + kb) * 96;
    reinterpret_cast<uint16_t*>(dst)[group] = selected;
    reinterpret_cast<uint32_t*>(dst + 32)[group] = out_lows;
    if (collision) dest_escapes[escape_index] = collision_escape(normal_lows, base);
  }
}

__global__ void split_row_escapes(const uint4* __restrict__ source,
                                  int64_t original_count,
                                  uint64_t* __restrict__ dest) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= original_count) return;
  const uint4 bits = source[index];
  dest[2 * index] = uint64_t(bits.x) | (uint64_t(bits.z) << 32);
  dest[2 * index + 1] = uint64_t(bits.y) | (uint64_t(bits.w) << 32);
}

}  // namespace

extern "C" int nf12_rowmajor_count(const uint8_t* source, int64_t n, int64_t k,
                                   int32_t* counts, cudaStream_t stream) {
  if (!source || !counts || n <= 0 || n % 16 || k <= 0 || k % 64)
    return cudaErrorInvalidValue;
  const int64_t tiles = (n / 16) * (k / 64);
  if (tiles > 0x7fffffffu) return cudaErrorInvalidValue;
  count_row_collisions<<<static_cast<unsigned>(tiles), 128, 0, stream>>>(source, counts);
  return cudaGetLastError();
}

extern "C" int nf12_rowmajor_pack(const uint8_t* source,
                                  const uint4* original_escapes, int64_t original_count,
                                  int64_t n, int64_t k, uint32_t base,
                                  const int64_t* prefix, uint8_t* dest,
                                  uint64_t* dest_escapes, int32_t* invalid_escape,
                                  cudaStream_t stream) {
  if (!source || !prefix || !dest || (!dest_escapes && original_count) || !invalid_escape ||
      n <= 0 || n % 16 || k <= 0 || k % 64 || base > 120 ||
      original_count < 0 || (original_count && !original_escapes))
    return cudaErrorInvalidValue;
  const int64_t tiles = (n / 16) * (k / 64);
  if (tiles > 0x7fffffffu) return cudaErrorInvalidValue;
  pack_rowmajor<<<static_cast<unsigned>(tiles), 128, 0, stream>>>(
      source, original_count, k / 64, base, prefix, dest, dest_escapes, invalid_escape);
  if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) return err;
  if (original_count) {
    split_row_escapes<<<static_cast<unsigned>((original_count + 255) / 256), 256, 0, stream>>>(
        original_escapes, original_count, dest_escapes);
  }
  return cudaGetLastError();
}

template <typename Id, int Mode>
void launch_kernel(const Id* ids, const uint8_t* packed, const uint64_t* escapes,
                   int64_t num_escapes, uint32_t base, int64_t n, int64_t k,
                   int64_t m, uint64_t* out, int threads, int ids_per_cta,
                   cudaStream_t stream) {
#define LAUNCH(PACK) { \
  constexpr int lanes_per_block = Mode == 0 ? 8 : (Mode >= 5 ? 2 : 4); \
  const int groups_per_id = (threads / lanes_per_block) / PACK; \
  const dim3 grid(static_cast<unsigned>((m + PACK - 1) / PACK), \
                  static_cast<unsigned>((k / 64 + groups_per_id - 1) / groups_per_id)); \
  rowmajor_kernel<Id, PACK, Mode><<<grid, threads, 0, stream>>>( \
      ids, packed, escapes, num_escapes, base, n, k, m, out); \
  break; \
}
  switch (ids_per_cta) {
    case 1: LAUNCH(1)
    case 2: LAUNCH(2)
    case 4: LAUNCH(4)
    case 8: LAUNCH(8)
    case 16: LAUNCH(16)
  }
#undef LAUNCH
}

extern "C" int nf12_rowmajor_lookup(const void* ids, const uint8_t* packed,
                                    const uint64_t* escapes, int64_t num_escapes,
                                    int64_t n, int64_t k, uint32_t base,
                                    int64_t m, uint64_t* out, int id_bytes,
                                    int threads, int ids_per_cta, int mode,
                                    cudaStream_t stream) {
  if (!ids || !packed || !out || n <= 0 || k <= 0 || k % 64 || base > 120 ||
      m < 0 || (id_bytes != 4 && id_bytes != 8) ||
      (threads != 64 && threads != 128 && threads != 256 && threads != 512) ||
      (ids_per_cta != 1 && ids_per_cta != 2 && ids_per_cta != 4 &&
       ids_per_cta != 8 && ids_per_cta != 16) ||
      mode < 0 || mode > 6 || ids_per_cta > threads / (mode == 0 ? 8 : (mode >= 5 ? 2 : 4)))
    return cudaErrorInvalidValue;
  if (m == 0) return cudaSuccess;
#define DISPATCH_MODE(ID) \
  switch (mode) { \
    case 0: launch_kernel<ID, 0>(static_cast<const ID*>(ids), packed, escapes, num_escapes, base, n, k, m, out, threads, ids_per_cta, stream); break; \
    case 1: launch_kernel<ID, 1>(static_cast<const ID*>(ids), packed, escapes, num_escapes, base, n, k, m, out, threads, ids_per_cta, stream); break; \
    case 2: launch_kernel<ID, 2>(static_cast<const ID*>(ids), packed, escapes, num_escapes, base, n, k, m, out, threads, ids_per_cta, stream); break; \
    case 3: launch_kernel<ID, 3>(static_cast<const ID*>(ids), packed, escapes, num_escapes, base, n, k, m, out, threads, ids_per_cta, stream); break; \
    case 4: launch_kernel<ID, 4>(static_cast<const ID*>(ids), packed, escapes, num_escapes, base, n, k, m, out, threads, ids_per_cta, stream); break; \
    case 5: launch_kernel<ID, 5>(static_cast<const ID*>(ids), packed, escapes, num_escapes, base, n, k, m, out, threads, ids_per_cta, stream); break; \
    case 6: launch_kernel<ID, 6>(static_cast<const ID*>(ids), packed, escapes, num_escapes, base, n, k, m, out, threads, ids_per_cta, stream); break; \
  }
  if (id_bytes == 4) { DISPATCH_MODE(int32_t) }
  else { DISPATCH_MODE(int64_t) }
#undef DISPATCH_MODE
  return cudaGetLastError();
}
