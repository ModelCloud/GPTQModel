// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// BF16 -> bf16t twin packing (SPEC.md 4.4) for RDNA4.
// Port of csrc/codec/decode.cu's pack_bf16t_kernel using the shared
// format_hip.cuh row-tile helpers.

#include "nf12_ad_format_hip.cuh"

namespace nf12_hip {
namespace {

constexpr int kPackSuperAtomsPerWarp = 2;

template <int kPerWarp>
__global__ void __launch_bounds__(kThreadsPerBlock)
    pack_bf16t_kernel(const __hip_bfloat16* __restrict__ w, int64_t ldw, int64_t num_sa,
                      int64_t k_blocks, uint4* __restrict__ out) {
  const int lane = threadIdx.x % kWarpSize;
  RowTile x[kPerWarp];
#pragma unroll
  for (int i = 0; i < kPerWarp; ++i) {
    const int64_t sa = warp_super_atom<kPerWarp>(i);
    if (sa < num_sa) {
      load_row_tile(w, row_tile_offset(super_atom(sa, k_blocks), ldw, lane), ldw, x[i]);
    }
  }
#pragma unroll
  for (int i = 0; i < kPerWarp; ++i) {
    const int64_t sa = warp_super_atom<kPerWarp>(i);
    if (sa >= num_sa) break;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int p = 0; p < 2; ++p) quad_transpose(x[i][h][p], lane);
    }
    AtomRegs r;
    tile_to_atoms(x[i], r);
    uint4* dst = out + sa * kSuperAtomVecsBf16t + lane;
#pragma unroll
    for (int j = 0; j < kAtomsPerSuperAtom; ++j) {
      dst[j * kPlaneVecs] = make_uint4(r[j][0], r[j][1], r[j][2], r[j][3]);
    }
  }
}

}  // namespace

hipError_t pack_bf16t(const void* w_in, int64_t n, int64_t k, int64_t ldw, uint8_t* out,
                      hipStream_t stream) {
  const int64_t num_sa = (n / kSuperAtomRows) * (k / kSuperAtomCols);
  const int64_t blocks = grid_blocks<kPackSuperAtomsPerWarp>(num_sa);
  if (blocks == 0) return hipSuccess;
  if (blocks > 0x7fffffff) return hipErrorInvalidValue;
  pack_bf16t_kernel<kPackSuperAtomsPerWarp>
      <<<static_cast<unsigned>(blocks), kThreadsPerBlock, 0, stream>>>(
          static_cast<const __hip_bfloat16*>(w_in), ldw, num_sa, k / kSuperAtomCols,
          reinterpret_cast<uint4*>(out));
  return hipGetLastError();
}

}  // namespace nf12_hip
