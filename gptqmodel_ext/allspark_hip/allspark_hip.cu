// SPDX-FileCopyrightText: 2025 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// AllSpark W8A16 fused GEMM for AMD RDNA4 (gfx1200 / gfx1201).
//
// Structure follows the CUDA kernel in gptqmodel_ext/allspark:
//   * split-K over the K dimension, non-fused fp32 partials + reduce kernel
//   * a tile parameter struct (BlockTileSplitkParams)
//   * per-N scale / zero-point applied to the dequantised weight fragment
//   * identical torch op signatures (see allspark_torch_hip.cpp)
//
// What had to be rewritten (see allspark_utils_hip.cuh for the full table):
//   * cp.async -> global->register->ds_write staging, NSTAGE deep
//   * ldmatrix  -> plain 16B LDS reads (the repack already emits fragment order)
//   * mma.sync  -> v_wmma_f32_16x16x16_{f16,bf16}_w32_gfx12
//   * prmt      -> v_perm_b32
//   * cuBLAS    -> not used; M > 16 falls back to an explicit dequant + matmul
//                  in Python (temporary, never cached)
//
// gfx1250 hook: gfx1250 has `global_load_async_to_lds_b128` +
// `s_waitcnt vmcnt`, so the staging below can become a true cp.async-style
// pipeline there. The staging function is isolated (`stage_async_hook`) and
// marked, but only the synchronous gfx12 path is implemented.

#include "allspark_utils_hip.cuh"

#include <cstddef>
#include <cstdint>

namespace allspark_hip {

// ---------------------------------------------------------------------------
// Tile configuration
// ---------------------------------------------------------------------------
constexpr int kBlock = 128;          // 4 wave32 warps
constexpr int kWarps = kBlock / 32;
constexpr int kNPerWarp = 32;        // one (n-tile pair) = 2 x 16 columns
constexpr int kNtile = kWarps * kNPerWarp;  // 128 columns per block
constexpr int kKTile = 16;           // k per WMMA step
constexpr int kNStage = 4;           // LDS staging depth

constexpr int kBytesPerPairTile = 512;  // 32 n x 16 k uint8, fragment ordered
constexpr int kSmemB = kNtile * kKTile; // 2048 bytes per stage
constexpr int kSmemA = 16 * 16;         // 256 elements per stage

// Sentinel written by the kernel; the host verifies it once at load time so a
// silently-empty kernel body (the `#if __CUDA_ARCH__` failure mode) is caught.
constexpr uint32_t kRanMagic = 0x41535350u;  // "ASSP"

// ---------------------------------------------------------------------------
// Fragment helpers
// ---------------------------------------------------------------------------
template <typename FType>
struct WmmaTraits;

template <>
struct WmmaTraits<f16_t> {
  using vec_t = v8f16;
  static __device__ __forceinline__ v8f32 mm(vec_t a, vec_t b, v8f32 c) {
    return wmma_f16(a, b, c);
  }
};

template <>
struct WmmaTraits<bf16_t> {
  using vec_t = v8bf16;
  static __device__ __forceinline__ v8f32 mm(vec_t a, vec_t b, v8f32 c) {
    return wmma_bf16(a, b, c);
  }
};

union U4Frag16 {
  uint4 u;
  v8f16 v;
};

union U4FragBF {
  uint4 u;
  v8bf16 v;
};

// Dequantise 8 uint8 codes (two packed 4-byte words) into one 16x16x16 B
// fragment, applying the per-N zero point and scale.
//
// `zero` is the zero point **code** in the quantised domain (AWQ qzeros),
// so the weight is exactly (code - zero) * scale. GPTQ 8-bit is the neutral
// case zero = 128, which is also the value the kernel uses when HAS_ZP is
// false. Note this differs from the CUDA AllSpark kernel, which subtracts the
// zero point after the bias-128 conversion and therefore needs a
// pre-adjusted (zero - 128) input; the HIP entry point takes the code as-is.
template <typename FType, bool HAS_ZP>
__device__ __forceinline__ void dequant_pair(uint32_t w0, uint32_t w1,
                                             float scale, float zero,
                                             typename WmmaTraits<FType>::vec_t& out) {
  uint32_t o[4];
  if constexpr (__is_same(FType, f16_t)) {
    cvt_8bx4_to_16bx4_bias128(w0, o[0], o[1]);
    cvt_8bx4_to_16bx4_bias128(w1, o[2], o[3]);
    const __half2 s2 = __half2half2(__float2half_rn(scale));
    // cvt() already removed 128, so fold the remaining offset in here:
    //   (code - 128) - (zero - 128) == code - zero.
    const __half2 z2 = __half2half2(__float2half_rn(zero - 128.0f));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      U32Half2 t;
      t.u = o[i];
      if constexpr (HAS_ZP) t.h2 = __hsub2(t.h2, z2);
      t.h2 = __hmul2(t.h2, s2);
      o[i] = t.u;
    }
    U4Frag16 f;
    f.u = make_uint4(o[0], o[1], o[2], o[3]);
    out = f.v;
  } else {
    // The zero point is applied before the scale inside the fp32 conversion,
    // so bf16 sees (code - zero) * scale with no double rounding.
    cvt_8bx4_to_16bx4_bf16_scaled(w0, scale, HAS_ZP ? zero : 128.0f, o[0], o[1]);
    cvt_8bx4_to_16bx4_bf16_scaled(w1, scale, HAS_ZP ? zero : 128.0f, o[2], o[3]);
    U4FragBF f;
    f.u = make_uint4(o[0], o[1], o[2], o[3]);
    out = f.v;
  }
}

// ---------------------------------------------------------------------------
// GEMM kernel
// ---------------------------------------------------------------------------
template <typename FType, bool HAS_ZP>
__global__ __launch_bounds__(kBlock) void allspark_w8a16_kernel(
    SM8x_GEMM_W8A16_Splitk_Params<FType, uint8_t> p, uint32_t* ran) {
  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;
  const int warp = tid / kWarpSize;

  const int m0 = blockIdx.x * 16;
  const int n0 = blockIdx.y * kNtile;
  const int split = blockIdx.z;

  const int kt_total = p.K / kKTile;
  const int kt_per_split = (kt_total + p.SplitK - 1) / p.SplitK;
  const int kt_begin = split * kt_per_split;
  const int kt_end = min(kt_total, kt_begin + kt_per_split);
  const int nkt = kt_end - kt_begin;
  if (nkt <= 0) return;

  if (ran != nullptr && tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 &&
      blockIdx.z == 0) {
    *ran = kRanMagic;
  }

  __shared__ uint8_t sB[kNStage][kSmemB];
  __shared__ FType sA[kNStage][kSmemA];

  // This block owns n in [n0, n0 + kNtile). Warp w owns the 32-column pair
  // (n0 + 32*w). The repack stores each (pair, kt) tile as 512 contiguous
  // bytes in WMMA fragment order, so lane l reads bytes [l*16, l*16+16).
  const int pair = n0 / kNPerWarp + warp;
  const int npairs = p.N32 / kNPerWarp;
  const bool pair_ok = pair < npairs;
  const uint8_t* bptr =
      p.B_ptr + static_cast<size_t>(pair) * kt_total * kBytesPerPairTile +
      lane * 16;

  // Per-lane output columns: n-tile pair column offset is lane%16, and the two
  // 16-column tiles of the pair are 16 apart.
  const int n_a = n0 + warp * kNPerWarp + (lane % 16);
  const int n_b = n_a + 16;

  // Scale/zero are per-output-channel and live in natural N order. GPTQ 8-bit
  // has scale only; AWQ 8-bit adds a zero point. Both are one row per N (one
  // group over all of K), so they are loaded once here and reused for every
  // k-tile: no per-tile reload and no group arithmetic in the k-loop.
  const float scale_a = static_cast<float>(p.B_scale_ptr[n_a]);
  const float scale_b = static_cast<float>(p.B_scale_ptr[n_b]);
  const float zero_a = HAS_ZP ? static_cast<float>(p.B_zero_ptr[n_a]) : 128.0f;
  const float zero_b = HAS_ZP ? static_cast<float>(p.B_zero_ptr[n_b]) : 128.0f;

  const int a_row = m0 + (lane % 16);
  const int a_koff = (lane / 16) * 8;
  const bool a_ok = a_row < p.M;

  const int a_thd_m = tid / 2;   // threads 0..31 stage the 16x16 A tile
  const int a_thd_k = tid % 2;
  const bool a_stage_ok = a_thd_m < 16 && (m0 + a_thd_m) < p.M;

  auto load_b = [&](int kt) -> uint4 {
    if (!pair_ok) return make_uint4(0, 0, 0, 0);
    return *reinterpret_cast<const uint4*>(bptr + static_cast<size_t>(kt) * kBytesPerPairTile);
  };
  auto load_a = [&](int kt) -> uint4 {
    if (!a_stage_ok) return make_uint4(0, 0, 0, 0);
    const FType* src = p.A_ptr + static_cast<size_t>(m0 + a_thd_m) * p.K +
                       kt * kKTile + a_thd_k * 8;
    return *reinterpret_cast<const uint4*>(src);
  };

  // ---- prologue: fill kNStage-1 stages --------------------------------
#pragma unroll
  for (int s = 0; s < kNStage - 1; ++s) {
    const int kt = kt_begin + s;
    const uint4 b = (kt < kt_end) ? load_b(kt) : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(sB[s] + warp * kBytesPerPairTile + lane * 16) = b;
    if (tid < 32) {
      const uint4 a = (kt < kt_end) ? load_a(kt) : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(sA[s] + a_thd_m * 16 + a_thd_k * 8) = a;
    }
  }
  __syncthreads();

  typename WmmaTraits<FType>::vec_t acc_a, acc_b;
  v8f32 c0, c1;
#pragma unroll
  for (int j = 0; j < kFragElems; ++j) {
    c0[j] = 0.0f;
    c1[j] = 0.0f;
  }

  const FType* a_frag_src = sA[0] + (lane % 16) * 16 + (lane / 16) * 8;

  for (int i = 0; i < nkt; ++i) {
    const int slot = i % kNStage;
    const int pf = kt_begin + i + (kNStage - 1);
    // Issue the global loads first so they overlap the WMMA of this stage.
    const uint4 nb = (pf < kt_end) ? load_b(pf) : make_uint4(0, 0, 0, 0);
    const uint4 na = (pf < kt_end && tid < 32) ? load_a(pf) : make_uint4(0, 0, 0, 0);

    // ---- compute the stage staged kNStage-1 iterations ago -------------

    U4Frag16 a4;
    a4.u = *reinterpret_cast<const uint4*>(a_frag_src + slot * kSmemA);
    U4Frag16 b4;
    b4.u = *reinterpret_cast<const uint4*>(sB[slot] + warp * kBytesPerPairTile +
                                          lane * 16);

    typename WmmaTraits<FType>::vec_t bv;
    if constexpr (__is_same(FType, f16_t)) {
      dequant_pair<FType, HAS_ZP>(b4.u.x, b4.u.y, scale_a, zero_a, bv);
      acc_a = a4.v;
      c0 = WmmaTraits<FType>::mm(acc_a, bv, c0);
      dequant_pair<FType, HAS_ZP>(b4.u.z, b4.u.w, scale_b, zero_b, bv);
      c1 = WmmaTraits<FType>::mm(acc_a, bv, c1);
    } else {
      U4FragBF ab;
      ab.u = a4.u;
      dequant_pair<FType, HAS_ZP>(b4.u.x, b4.u.y, scale_a, zero_a, bv);
      acc_b = ab.v;
      c0 = WmmaTraits<FType>::mm(acc_b, bv, c0);
      dequant_pair<FType, HAS_ZP>(b4.u.z, b4.u.w, scale_b, zero_b, bv);
      c1 = WmmaTraits<FType>::mm(acc_b, bv, c1);
    }

    // ---- commit the prefetched stage -----------------------------------
    if (pf < kt_end) {
      const int wslot = (i + kNStage - 1) % kNStage;
      *reinterpret_cast<uint4*>(sB[wslot] + warp * kBytesPerPairTile + lane * 16) = nb;
      if (tid < 32) {
        *reinterpret_cast<uint4*>(sA[wslot] + a_thd_m * 16 + a_thd_k * 8) = na;
      }
    }
    __syncthreads();
  }

  // ---- epilogue ---------------------------------------------------------
  // C is column-distributed: lane l owns rows (l/16)*8 + j for column l%16.
  const int c_row0 = m0 + (lane / 16) * 8;
  const int c_col_a = n_a;
  const int c_col_b = n_b;
  const bool store_split = p.SplitK > 1;
  const bool out32 = (p.C32_ptr != nullptr) && !store_split;
  const size_t split_stride = static_cast<size_t>(p.M) * p.N;

  if (store_split) {
    float* base = p.C_tmp_ptr + static_cast<size_t>(split) * split_stride;
#pragma unroll
    for (int j = 0; j < kFragElems; ++j) {
      const int row = c_row0 + j;
      if (row < p.M) {
        if (c_col_a < p.N) base[static_cast<size_t>(row) * p.N + c_col_a] = c0[j];
        if (c_col_b < p.N) base[static_cast<size_t>(row) * p.N + c_col_b] = c1[j];
      }
    }
  } else if (out32) {
#pragma unroll
    for (int j = 0; j < kFragElems; ++j) {
      const int row = c_row0 + j;
      if (row < p.M) {
        if (c_col_a < p.N) p.C32_ptr[static_cast<size_t>(row) * p.N + c_col_a] = c0[j];
        if (c_col_b < p.N) p.C32_ptr[static_cast<size_t>(row) * p.N + c_col_b] = c1[j];
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < kFragElems; ++j) {
      const int row = c_row0 + j;
      if (row < p.M) {
        if (c_col_a < p.N) {
          p.C_ptr[static_cast<size_t>(row) * p.N + c_col_a] =
              static_cast<FType>(c0[j]);
        }
        if (c_col_b < p.N) {
          p.C_ptr[static_cast<size_t>(row) * p.N + c_col_b] =
              static_cast<FType>(c1[j]);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Split-K reduce (fp32 partials -> FType)
// ---------------------------------------------------------------------------
template <typename OType>
__global__ void splitk_reduce_kernel(const float* __restrict__ csplit,
                                     OType* __restrict__ c, int total,
                                     int splitk, int plane) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  float sum = 0.0f;
  for (int z = 0; z < splitk; ++z) {
    sum += csplit[static_cast<size_t>(z) * plane + idx];
  }
  c[idx] = static_cast<OType>(sum);
}

// ---------------------------------------------------------------------------
// Repack: (K, N) uint8 + per-N scale/zero -> WMMA fragment order
// ---------------------------------------------------------------------------
// Output byte layout (identical total size to the CUDA N32K16 buffer):
//   for pair p (32 columns), k-tile kt (16 rows):
//     base = (p * (K/16) + kt) * 512
//     base + lane*16 + q*8 + j  =  B[(kt*16 + (lane/16)*8 + j) * N
//                                    + (p*32 + q*16 + lane%16)]
// i.e. lane l reads 16 contiguous bytes: 8 for column tile 2p, 8 for 2p+1.
__global__ void repack_kernel(const uint8_t* __restrict__ B,
                              uint8_t* __restrict__ Bout, int K, int N, int N32,
                              int kt_total) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t per_pair = static_cast<int64_t>(kt_total) * 64;
  const int64_t total = static_cast<int64_t>(N32 / 32) * per_pair;
  if (idx >= total) return;

  const int p = static_cast<int>(idx / per_pair);
  const int rem = static_cast<int>(idx % per_pair);
  const int kt = rem / 64;
  const int c = rem % 64;
  const int lane = c / 2;
  const int q = c % 2;

  const int n = p * 32 + q * 16 + (lane % 16);
  const int kbase = kt * 16 + (lane / 16) * 8;
  uint8_t* dst = Bout + (static_cast<int64_t>(p) * kt_total + kt) * 512 +
                 lane * 16 + q * 8;
  if (n < N) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int k = kbase + j;
      dst[j] = (k < K) ? B[static_cast<int64_t>(k) * N + n] : uint8_t(0);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) dst[j] = uint8_t(0);
  }
}

template <typename FType>
__global__ void pad_scale_kernel(const FType* __restrict__ src,
                                 FType* __restrict__ dst, int N, int N32,
                                 FType pad) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N32) return;
  dst[n] = (n < N) ? src[n] : pad;
}

// ---------------------------------------------------------------------------
// Unpack: fragment order -> (K, N) dense, used only for the M > 16 fallback.
// The result is a per-call temporary in Python; it is never cached.
// ---------------------------------------------------------------------------
template <typename FType>
__global__ void unpack_kernel(const uint8_t* __restrict__ Bq,
                              const FType* __restrict__ Bs,
                              const FType* __restrict__ Bz, FType* __restrict__ B,
                              int K, int N, int N32, int kt_total) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t per_pair = static_cast<int64_t>(kt_total) * 64;
  const int64_t total = static_cast<int64_t>(N32 / 32) * per_pair;
  if (idx >= total) return;

  const int p = static_cast<int>(idx / per_pair);
  const int rem = static_cast<int>(idx % per_pair);
  const int kt = rem / 64;
  const int c = rem % 64;
  const int lane = c / 2;
  const int q = c % 2;

  const int n = p * 32 + q * 16 + (lane % 16);
  if (n >= N) return;
  const int kbase = kt * 16 + (lane / 16) * 8;
  const uint8_t* src = Bq + (static_cast<int64_t>(p) * kt_total + kt) * 512 +
                       lane * 16 + q * 8;
  const float s = static_cast<float>(Bs[n]);
  const float z = (Bz != nullptr) ? static_cast<float>(Bz[n]) : 128.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int k = kbase + j;
    if (k < K) {
      B[static_cast<int64_t>(k) * N + n] =
          static_cast<FType>((static_cast<int>(src[j]) - z) * s);
    }
  }
}

// ---------------------------------------------------------------------------
// Host-side launchers (C linkage: no torch types cross this boundary)
// ---------------------------------------------------------------------------

}  // namespace allspark_hip

using namespace allspark_hip;

extern "C" {

size_t allspark_hip_workspace_bytes(int M, int N, int splitk) {
  if (splitk <= 1) return 0;
  return static_cast<size_t>(splitk) * M * N * sizeof(float);
}

int allspark_hip_pick_splitk(int M, int N, int N32, int K, int sm_count) {
  const int nblocks = (N32 + kNtile - 1) / kNtile;
  const int kt_total = K / kKTile;
  int best = 1;
  // Aim for ~3 blocks per CU so the tail of the launch is not a partial wave.
  for (int s = 1; s <= 8; ++s) {
    if (kt_total / s < 16) break;  // keep at least 16 k-tiles per split
    if (nblocks * s <= sm_count * 3) best = s;
  }
  return best;
}

void allspark_hip_launch(const void* A, const void* Bq, const void* Bs,
                         const void* Bz, void* C, void* ws, int M, int N,
                         int N32, int K, int splitk, int dtype, int out_fp32,
                         void* stream, uint32_t* ran) {
  hipStream_t s = reinterpret_cast<hipStream_t>(stream);
  const int kt_total = K / kKTile;
  const int kt_per_split = (kt_total + splitk - 1) / splitk;
  const int grid_z = (kt_total + kt_per_split - 1) / kt_per_split;
  dim3 grid((M + 15) / 16, (N32 + kNtile - 1) / kNtile, grid_z);
  SM8x_GEMM_W8A16_Splitk_Params<f16_t, uint8_t> pf;
  SM8x_GEMM_W8A16_Splitk_Params<__bf16, uint8_t> pb;

#define ALLSPARK_FILL(P, FT)                                                  \
  do {                                                                        \
    (P).A_ptr = reinterpret_cast<const FT*>(A);                               \
    (P).B_ptr = reinterpret_cast<const uint8_t*>(Bq);                         \
    (P).B_scale_ptr = reinterpret_cast<const FT*>(Bs);                        \
    (P).B_zero_ptr = reinterpret_cast<const FT*>(Bz);                         \
    (P).C_ptr = reinterpret_cast<FT*>(C);                                     \
    (P).M = M;                                                                \
    (P).N = N;                                                                \
    (P).K = K;                                                                \
    (P).SplitK = grid_z;                                                      \
    (P).C_split_ptr = nullptr;                                                \
    (P).C_tmp_ptr = reinterpret_cast<float*>(ws);                             \
    (P).C32_ptr = out_fp32 ? reinterpret_cast<float*>(C) : nullptr;           \
    (P).red_count_ptr = nullptr;                                              \
    (P).N32 = N32;                                                            \
  } while (0)

  if (dtype == 0) {
    ALLSPARK_FILL(pf, f16_t);
    if (Bz == nullptr) {
      allspark_w8a16_kernel<f16_t, false><<<grid, kBlock, 0, s>>>(pf, ran);
    } else {
      allspark_w8a16_kernel<f16_t, true><<<grid, kBlock, 0, s>>>(pf, ran);
    }
  } else {
    ALLSPARK_FILL(pb, __bf16);
    if (Bz == nullptr) {
      allspark_w8a16_kernel<__bf16, false><<<grid, kBlock, 0, s>>>(pb, ran);
    } else {
      allspark_w8a16_kernel<__bf16, true><<<grid, kBlock, 0, s>>>(pb, ran);
    }
  }
#undef ALLSPARK_FILL

  if (grid_z > 1) {
    const int total = M * N;
    const int block = 256;
    const int rgrid = (total + block - 1) / block;
    if (out_fp32) {
      splitk_reduce_kernel<float><<<rgrid, block, 0, s>>>(
          reinterpret_cast<const float*>(ws), reinterpret_cast<float*>(C), total,
          grid_z, total);
    } else if (dtype == 0) {
      splitk_reduce_kernel<f16_t><<<rgrid, block, 0, s>>>(
          reinterpret_cast<const float*>(ws), reinterpret_cast<f16_t*>(C), total,
          grid_z, total);
    } else {
      splitk_reduce_kernel<__bf16><<<rgrid, block, 0, s>>>(
          reinterpret_cast<const float*>(ws), reinterpret_cast<__bf16*>(C),
          total, grid_z, total);
    }
  }
}

void allspark_hip_launch_repack(const void* B, const void* Bs, const void* Bz,
                                void* Bq_out, void* Bs_out, void* Bz_out, int K,
                                int N, int N32, int dtype, void* stream) {
  hipStream_t s = reinterpret_cast<hipStream_t>(stream);
  const int kt_total = K / kKTile;
  const int64_t total = static_cast<int64_t>(N32 / 32) * kt_total * 64;
  const int block = 256;
  const int grid = static_cast<int>((total + block - 1) / block);
  const int sgrid = (N32 + block - 1) / block;

  repack_kernel<<<grid, block, 0, s>>>(reinterpret_cast<const uint8_t*>(B),
                                       reinterpret_cast<uint8_t*>(Bq_out), K, N,
                                       N32, kt_total);

  // scale / zero stay in natural N order; pad the tail so the kernel can read
  // a full 32-column pair without branching. One row per N: AWQ -1 has a single
  // group over all of K, and GPTQ 8-bit is per-channel.
  if (dtype == 0) {
    pad_scale_kernel<f16_t><<<sgrid, block, 0, s>>>(
        reinterpret_cast<const f16_t*>(Bs), reinterpret_cast<f16_t*>(Bs_out),
        N, N32, f16_t(1.0f));
    if (Bz != nullptr) {
      pad_scale_kernel<f16_t><<<sgrid, block, 0, s>>>(
          reinterpret_cast<const f16_t*>(Bz), reinterpret_cast<f16_t*>(Bz_out),
          N, N32, f16_t(0.0f));
    }
  } else {
    pad_scale_kernel<__bf16><<<sgrid, block, 0, s>>>(
        reinterpret_cast<const __bf16*>(Bs),
        reinterpret_cast<__bf16*>(Bs_out), N, N32, static_cast<__bf16>(1.0f));
    if (Bz != nullptr) {
      pad_scale_kernel<__bf16><<<sgrid, block, 0, s>>>(
          reinterpret_cast<const __bf16*>(Bz),
          reinterpret_cast<__bf16*>(Bz_out), N, N32,
          static_cast<__bf16>(0.0f));
    }
  }
}

void allspark_hip_launch_unpack(const void* Bq, const void* Bs, const void* Bz,
                                void* B, int K, int N, int N32, int dtype,
                                void* stream) {
  hipStream_t s = reinterpret_cast<hipStream_t>(stream);
  const int kt_total = K / kKTile;
  const int64_t total = static_cast<int64_t>(N32 / 32) * kt_total * 64;
  const int block = 256;
  const int grid = static_cast<int>((total + block - 1) / block);
  if (dtype == 0) {
    unpack_kernel<f16_t><<<grid, block, 0, s>>>(
        reinterpret_cast<const uint8_t*>(Bq), reinterpret_cast<const f16_t*>(Bs),
        reinterpret_cast<const f16_t*>(Bz), reinterpret_cast<f16_t*>(B), K, N, N32,
        kt_total);
  } else {
    unpack_kernel<__bf16><<<grid, block, 0, s>>>(
        reinterpret_cast<const uint8_t*>(Bq), reinterpret_cast<const __bf16*>(Bs),
        reinterpret_cast<const __bf16*>(Bz), reinterpret_cast<__bf16*>(B), K,
        N, N32, kt_total);
  }
}

}  // extern "C"
