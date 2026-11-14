// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12 embedding lookup for AMD RDNA4. Decodes only requested 64-column row
// slices, directly from the packed table to BF16 output. The layout and the
// zero-row behavior for invalid IDs match the vendored CUDA embedding kernel.
// ---------------------------------------------------------------------------
#include "nf12_format_hip.cuh"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace nf12_hip {
namespace {

// One group holds four 32-bit words of BF16 data for two paired rows. A
// selected row needs R0/R2 (lower half) or R1/R3 (upper half). An invalid
// escape index is treated as zero instead of reading beyond the table.
__device__ __forceinline__ uint2 decode_row_half_fast(
    const Group& group, uint32_t base_x4, bool upper) {
  const uint32_t m = group.meta;
  const uint32_t a = (m & 0x87878787u) + base_x4;
  const uint32_t b = (((m << 1) & 0x80808080u) | ((m >> 3) & 0x07070707u)) + base_x4;
  const uint32_t p0 = prmt<0x5140u>(a, b);
  const uint32_t p1 = prmt<0x7362u>(a, b);
  const uint32_t select = upper ? 0x03070206u : 0x01050004u;
  return make_uint2(__builtin_amdgcn_perm(group.lo0, p0, select),
                    __builtin_amdgcn_perm(group.lo1, p1, select));
}

__device__ __forceinline__ uint2 decode_row_half(
    const Group& group, uint32_t base_x4,
    const uint4* __restrict__ escapes, uint32_t num_escapes, bool upper) {
  uint2 r;
  if (group.meta == 0u) {
    if (group.lo0 >= num_escapes || escapes == nullptr) return make_uint2(0u, 0u);
    const uint4 escaped = escapes[group.lo0];
    r = upper ? make_uint2(escaped.y, escaped.w) : make_uint2(escaped.x, escaped.z);
  } else {
    r = decode_row_half_fast(group, base_x4, upper);
  }
  return r;
}

template <int Quads, int IdsPerCTA, bool SmallIndex, typename Id>
__global__ void __launch_bounds__(Quads * 4)
embedding_kernel(const uint4* __restrict__ packed,
                 const uint4* __restrict__ escapes,
                 uint32_t num_escapes, uint32_t base_x4,
                 std::conditional_t<SmallIndex, uint32_t, int64_t> n,
                 std::conditional_t<SmallIndex, uint32_t, int64_t> k_blocks,
                 const Id* __restrict__ ids,
                 std::conditional_t<SmallIndex, uint32_t, int64_t> num_ids,
                 __hip_bfloat16* __restrict__ out,
                 std::conditional_t<SmallIndex, uint32_t, int64_t> k) {
  using Index = std::conditional_t<SmallIndex, uint32_t, int64_t>;
  const int lane = threadIdx.x & 31;
  const int t = lane & 3;
  constexpr int kQuadsPerId = Quads / IdsPerCTA;
  const Index id_pos = Index(blockIdx.x) * IdsPerCTA + (threadIdx.x >> 2) / kQuadsPerId;
  const Index kb = Index(blockIdx.y) * kQuadsPerId + (threadIdx.x >> 2) % kQuadsPerId;
  const bool active = id_pos < num_ids && kb < k_blocks;

  uint4 first = make_uint4(0u, 0u, 0u, 0u);
  uint4 second = first;
  if (active) {
    const int64_t row_value = static_cast<int64_t>(ids[id_pos]);
    if (row_value >= 0 && row_value < n) {
      const Index row = static_cast<Index>(row_value);
      const int g = static_cast<int>(row & 7);
      const bool upper = (row & 8) != 0;
      const Index sa = (row >> 4) * k_blocks + kb;
      const uint4* src = packed + int64_t(sa) * kSuperAtomVecs + 4 * g + t;
      const uint4 meta = src[0];
      const uint4 low01 = src[kPlaneVecs];
      const uint4 low23 = src[2 * kPlaneVecs];
      const Group g0{meta.x, low01.x, low01.y};
      const Group g1{meta.y, low01.z, low01.w};
      const Group g2{meta.z, low23.x, low23.y};
      const Group g3{meta.w, low23.z, low23.w};
      const bool escaped = (meta.x == 0u) | (meta.y == 0u) | (meta.z == 0u) | (meta.w == 0u);
      uint2 a0, a1, a2, a3;
      if (__any(escaped)) {
        a0 = decode_row_half(g0, base_x4, escapes, num_escapes, upper);
        a1 = decode_row_half(g1, base_x4, escapes, num_escapes, upper);
        a2 = decode_row_half(g2, base_x4, escapes, num_escapes, upper);
        a3 = decode_row_half(g3, base_x4, escapes, num_escapes, upper);
      } else {
        a0 = decode_row_half_fast(g0, base_x4, upper);
        a1 = decode_row_half_fast(g1, base_x4, upper);
        a2 = decode_row_half_fast(g2, base_x4, upper);
        a3 = decode_row_half_fast(g3, base_x4, upper);
      }
      first = make_uint4(a0.x, a0.y, a1.x, a1.y);
      second = make_uint4(a2.x, a2.y, a3.x, a3.y);
    }
  }

  // The four lanes in a quad write adjacent 16-byte vectors after the same
  // self-inverse transpose used by the CUDA codec. Even inactive lanes must
  // participate in their quad's shuffles.
  quad_transpose(first, lane);
  quad_transpose(second, lane);
  if (active) {
    uint4* dst = reinterpret_cast<uint4*>(
        out + int64_t(id_pos) * k + int64_t(kb) * kSuperAtomCols) + t;
    dst[0] = first;
    dst[4] = second;
  }
}

template <int Quads, int IdsPerCTA, bool SmallIndex, typename Id>
hipError_t launch_embedding_typed(const uint8_t* packed, const uint4* escapes,
                                  uint32_t num_escapes, uint32_t exp_base,
                                  int64_t n, int64_t k, const Id* ids,
                                  int64_t num_ids, void* out, hipStream_t stream) {
  const int64_t k_blocks = k / kSuperAtomCols;
  constexpr int kQuadsPerId = Quads / IdsPerCTA;
  const int64_t kb_chunks = (k_blocks + kQuadsPerId - 1) / kQuadsPerId;
  const int64_t id_blocks = (num_ids + IdsPerCTA - 1) / IdsPerCTA;
  if (id_blocks > std::numeric_limits<uint32_t>::max() ||
      kb_chunks > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  const dim3 grid(static_cast<uint32_t>(id_blocks), static_cast<uint32_t>(kb_chunks));
  embedding_kernel<Quads, IdsPerCTA, SmallIndex><<<grid, Quads * 4, 0, stream>>>(
      reinterpret_cast<const uint4*>(packed), escapes, num_escapes,
      exp_base_x4(exp_base), n, k_blocks, ids, num_ids,
      reinterpret_cast<__hip_bfloat16*>(out), k);
  return hipGetLastError();
}

template <int Quads, int IdsPerCTA, typename Id>
hipError_t launch_embedding(const uint8_t* packed, const uint4* escapes,
                            uint32_t num_escapes, uint32_t exp_base,
                            int64_t n, int64_t k, const Id* ids,
                            int64_t num_ids, void* out, hipStream_t stream) {
  if (packed == nullptr || ids == nullptr || out == nullptr ||
      n <= 0 || k <= 0 || n % kSuperAtomRows != 0 ||
      k % kSuperAtomCols != 0 || exp_base > kMaxExpBase ||
      num_ids < 0) return hipErrorInvalidValue;
  if (num_ids == 0) return hipSuccess;

  const int64_t k_blocks = k / kSuperAtomCols;
  if (num_ids > std::numeric_limits<int32_t>::max() ||
      n > std::numeric_limits<uint32_t>::max() ||
      k > std::numeric_limits<uint32_t>::max() ||
      (n / kSuperAtomRows) * k_blocks > std::numeric_limits<uint32_t>::max()) {
    return launch_embedding_typed<Quads, IdsPerCTA, false>(
        packed, escapes, num_escapes, exp_base, n, k, ids, num_ids, out, stream);
  }
  return launch_embedding_typed<Quads, IdsPerCTA, true>(
      packed, escapes, num_escapes, exp_base, n, k, ids, num_ids, out, stream);
}

template <typename Id>
hipError_t launch_embedding_any(const uint8_t* packed, const uint4* escapes,
                                uint32_t num_escapes, uint32_t exp_base,
                                int64_t n, int64_t k, const Id* ids,
                                int64_t num_ids, void* out, hipStream_t stream) {
  return num_ids == 1
      ? launch_embedding<256, 1>(packed, escapes, num_escapes, exp_base, n, k,
                                 ids, num_ids, out, stream)
      : launch_embedding<32, 16>(packed, escapes, num_escapes, exp_base, n, k,
                                 ids, num_ids, out, stream);
}

}  // namespace
}  // namespace nf12_hip

extern "C" hipError_t nf12_embedding(
    const uint8_t* packed, const uint4* escapes, uint32_t num_escapes,
    uint32_t exp_base, int64_t n, int64_t k,
    const int32_t* ids32, const int64_t* ids64, int64_t num_ids,
    void* out, hipStream_t stream) {
  if ((ids32 == nullptr) == (ids64 == nullptr)) return hipErrorInvalidValue;
  if (ids32 != nullptr) {
    return nf12_hip::launch_embedding_any(
        packed, escapes, num_escapes, exp_base, n, k, ids32, num_ids, out, stream);
  }
  return nf12_hip::launch_embedding_any(
      packed, escapes, num_escapes, exp_base, n, k, ids64, num_ids, out, stream);
}
