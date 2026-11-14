// SPDX-License-Identifier: Apache-2.0
// Packed-only NF12 row-major embedding. Each 64-output row block holds 32
// metadata bytes and 64 mantissa bytes; escapes hold four BF16 bits.
#include <cuda_runtime.h>
#include <cstdint>

#include "../common/rowmajor_pack.cuh"

namespace {

// Production lookup geometry: four lanes per 64-column row block, one atom
// per lane, streaming stores. The experimental lane/store variants were
// removed after the production binding settled on this schedule.
template <typename Id, int IdsPerCTA>
__global__ void rowmajor_kernel(const Id* __restrict__ ids,
                                const uint8_t* __restrict__ packed,
                                const uint64_t* __restrict__ escapes,
                                int64_t num_escapes, uint32_t base,
                                int64_t n, int64_t k, int64_t m,
                                uint64_t* __restrict__ out) {
  constexpr int kLanesPerBlock = 4;
  const int groups_per_id = (blockDim.x / kLanesPerBlock) / IdsPerCTA;
  const int group = threadIdx.x / kLanesPerBlock;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * IdsPerCTA + group / groups_per_id;
  const int64_t kb = static_cast<int64_t>(blockIdx.y) * groups_per_id + group % groups_per_id;
  if (i >= m || kb >= k / 64) return;
  const int lane = threadIdx.x & (kLanesPerBlock - 1);
  const int64_t row = static_cast<int64_t>(ids[i]);
  const bool valid = row >= 0 && row < n;
  const uint8_t* src = valid ? packed + (row * (k / 64) + kb) * 96 : nullptr;
  uint2 mm2 = make_uint2(0, 0);
  uint4 low = make_uint4(0, 0, 0, 0);
  if (valid) {
    mm2 = __ldg(reinterpret_cast<const uint2*>(src) + lane);
    low = __ldg(reinterpret_cast<const uint4*>(src + 32) + lane);
  }
  const uint64_t mm = static_cast<uint64_t>(mm2.x) | (static_cast<uint64_t>(mm2.y) << 32);
  const nf12_rowmajor::Quad q = valid
      ? nf12_rowmajor::decode_quad(mm, low, escapes, num_escapes, base)
      : nf12_rowmajor::Quad{};
  uint4* dst = reinterpret_cast<uint4*>(out + (i * k + kb * 64) / 4) + 2 * lane;
  __stcg(dst, make_uint4(q.a.lo, q.b.lo, q.c.lo, q.d.lo));
  __stcg(dst + 1, make_uint4(q.a.hi, q.b.hi, q.c.hi, q.d.hi));
}

// A one-token decode launch has only one ID. Keep one CTA resident and walk
// the K blocks instead of creating a second grid dimension. The 160-thread
// geometry is four-lane decode groups with three groups left over for the
// streaming stores measured on SM89 and SM120.
template <typename Id>
__global__ void __launch_bounds__(160) rowmajor_one_kernel(
    const Id* __restrict__ ids, const uint8_t* __restrict__ packed,
    const uint64_t* __restrict__ escapes, int64_t num_escapes, uint32_t base,
    int64_t n, int64_t k_blocks, uint64_t* __restrict__ out) {
  constexpr int kLanesPerBlock = 4;
  constexpr int kGroups = 160 / kLanesPerBlock;
  const int lane = threadIdx.x & (kLanesPerBlock - 1);
  const int group = threadIdx.x / kLanesPerBlock;
  const int64_t row = static_cast<int64_t>(ids[0]);
  const bool valid = row >= 0 && row < n;
  for (int64_t kb = group; kb < k_blocks; kb += kGroups) {
    uint2 mm2 = make_uint2(0, 0);
    uint4 low = make_uint4(0, 0, 0, 0);
    if (valid) {
      const uint8_t* src = packed + (row * k_blocks + kb) * 96;
      mm2 = __ldg(reinterpret_cast<const uint2*>(src) + lane);
      low = __ldg(reinterpret_cast<const uint4*>(src + 32) + lane);
    }
    const uint64_t mm =
        static_cast<uint64_t>(mm2.x) | (static_cast<uint64_t>(mm2.y) << 32);
    const nf12_rowmajor::Quad q = valid
        ? nf12_rowmajor::decode_quad(mm, low, escapes, num_escapes, base)
        : nf12_rowmajor::Quad{};
    uint4* dst =
        reinterpret_cast<uint4*>(out + (kb * 64) / 4) + 2 * lane;
    __stcg(dst, make_uint4(q.a.lo, q.b.lo, q.c.lo, q.d.lo));
    __stcg(dst + 1, make_uint4(q.a.hi, q.b.hi, q.c.hi, q.d.hi));
  }
}

}  // namespace

extern "C" int nf12_rowmajor_count(const uint8_t* source, int64_t n, int64_t k,
                                   int32_t* counts, cudaStream_t stream) {
  if (!source || !counts || n <= 0 || n % 16 || k <= 0 || k % 64)
    return cudaErrorInvalidValue;
  const int64_t tiles = (n / 16) * (k / 64);
  if (tiles > 0x7fffffffu) return cudaErrorInvalidValue;
  nf12_rowmajor::count_collisions<<<static_cast<unsigned>(tiles), 128, 0, stream>>>(source, counts);
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
  nf12_rowmajor::pack<<<static_cast<unsigned>(tiles), 128, 0, stream>>>(
      source, original_count, k / 64, base, prefix, dest, dest_escapes, invalid_escape);
  if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) return err;
  if (original_count) {
    nf12_rowmajor::split_escapes<<<static_cast<unsigned>((original_count + 255) / 256), 256, 0, stream>>>(
        original_escapes, original_count, dest_escapes);
  }
  return cudaGetLastError();
}

template <typename Id>
void launch_kernel(const Id* ids, const uint8_t* packed, const uint64_t* escapes,
                   int64_t num_escapes, uint32_t base, int64_t n, int64_t k,
                   int64_t m, uint64_t* out, cudaStream_t stream) {
  constexpr int kThreads = 256;
  constexpr int kIdsPerCta = 8;
  constexpr int kGroupsPerId = (kThreads / 4) / kIdsPerCta;
  const dim3 grid(static_cast<unsigned>((m + kIdsPerCta - 1) / kIdsPerCta),
                  static_cast<unsigned>((k / 64 + kGroupsPerId - 1) / kGroupsPerId));
  rowmajor_kernel<Id, kIdsPerCta><<<grid, kThreads, 0, stream>>>(
      ids, packed, escapes, num_escapes, base, n, k, m, out);
}

extern "C" int nf12_rowmajor_lookup(const void* ids, const uint8_t* packed,
                                    const uint64_t* escapes, int64_t num_escapes,
                                    int64_t n, int64_t k, uint32_t base,
                                    int64_t m, uint64_t* out, int id_bytes,
                                    cudaStream_t stream) {
  if (!ids || !packed || !out || n <= 0 || k <= 0 || k % 64 || base > 120 ||
      m < 0 || (id_bytes != 4 && id_bytes != 8))
    return cudaErrorInvalidValue;
  if (m == 0) return cudaSuccess;
  if (m == 1 && k <= 2048) {
    if (id_bytes == 4) {
      rowmajor_one_kernel<int32_t><<<1, 160, 0, stream>>>(
          static_cast<const int32_t*>(ids), packed, escapes, num_escapes, base,
          n, k / 64, out);
    } else {
      rowmajor_one_kernel<int64_t><<<1, 160, 0, stream>>>(
          static_cast<const int64_t*>(ids), packed, escapes, num_escapes, base,
          n, k / 64, out);
    }
    return cudaGetLastError();
  }
  if (id_bytes == 4) {
    launch_kernel<int32_t>(static_cast<const int32_t*>(ids), packed, escapes,
                           num_escapes, base, n, k, m, out, stream);
  } else {
    launch_kernel<int64_t>(static_cast<const int64_t*>(ids), packed, escapes,
                           num_escapes, base, n, k, m, out, stream);
  }
  return cudaGetLastError();
}
