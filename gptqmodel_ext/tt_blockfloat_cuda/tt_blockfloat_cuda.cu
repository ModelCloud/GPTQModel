// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Tenstorrent block-float (bf8_b / bf4_b) emulation kernels for CUDA.
//
// CUDA sibling of ``gptqmodel_ext/tt_blockfloat_hip``.  The storage layout and
// every rounding decision match ``gptqmodel.utils.tt_blockfloat_quant``:
//
//   value      (code - (2*levels-1)) / levels * 2**exponent
//   levels     4 (bf4_b) or 64 (bf8_b)
//   codes      packed uint8, two per byte for bf4_b, low nibble first
//   exponents  int8, one per 16 consecutive values along K
//
// The quantiser reproduces ``quantize_blockfloat`` exactly, including its
// three-candidate exponent search and its strict ``<`` tie-break, so a weight
// quantised here is bit-identical to the torch reference.

#include "tt_blockfloat_cuda.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <limits>
#include <map>
#include <math_constants.h>
#include <mutex>
#include <type_traits>
#include <utility>

namespace {

constexpr int BLOCK = 16;

__device__ __host__ constexpr int stored_mantissa_bits(int bits) {
  return bits - 2 < 1 ? 1 : bits - 2;
}

// floor(log2(x)) for a positive normal float, exactly, from the bit pattern.
__device__ __forceinline__ int floor_log2_exact(float x) {
  const unsigned bits = __float_as_uint(x);
  int e = (int)((bits >> 23) & 0xffu) - 127;
  // Subnormals: log2 is below -126; the reference clamps at FLT_MIN so this
  // cannot be reached, but keep it well defined.
  return e < -126 ? -126 : e;
}

// ---------------------------------------------------------------------------
// Element type helpers
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float bf_load(const T* p);
template <>
__device__ __forceinline__ float bf_load<float>(const float* p) {
  return *p;
}
template <>
__device__ __forceinline__ float bf_load<__half>(const __half* p) {
  return __half2float(*p);
}
template <>
__device__ __forceinline__ float bf_load<__nv_bfloat16>(const __nv_bfloat16* p) {
  return __bfloat162float(*p);
}

template <typename T>
__device__ __forceinline__ T bf_cast(float v);
template <>
__device__ __forceinline__ float bf_cast<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ __half bf_cast<__half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 bf_cast<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

// Bit pattern of a 16-bit element type, so a dequantised value can be packed
// two at a time into a shared-memory store regardless of T.
template <typename T>
__device__ __forceinline__ uint32_t bf_u16(T v) {
  union {
    T t;
    uint16_t u;
  } c;
  c.t = v;
  return (uint32_t)c.u;
}

// ---------------------------------------------------------------------------
// Grid-quantised value <-> code
// ---------------------------------------------------------------------------
template <int BITS>
__device__ __forceinline__ int code_offset() {
  return 2 * (1 << stored_mantissa_bits(BITS)) - 1;
}

template <int BITS>
__device__ __forceinline__ float dequant_value(int code, int exponent) {
  constexpr int levels = 1 << stored_mantissa_bits(BITS);
  return (float)(code - code_offset<BITS>()) * (1.0f / levels) * exp2f((float)exponent);
}

// ---------------------------------------------------------------------------
// dequant: codes + exponents -> dense
// ---------------------------------------------------------------------------
template <int BITS, typename T>
__global__ void dequant_kernel(const uint8_t* __restrict__ codes,
                               const int8_t* __restrict__ exponents, int64_t rows,
                               int64_t k, T* __restrict__ out) {
  const int64_t n = rows * k;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t r = i / k;
  const int64_t c = i - r * k;
  const int exponent = (int)exponents[r * (k / BLOCK) + (c >> 4)];
  int code;
  if constexpr (BITS == 4) {
    const uint8_t byte = codes[i >> 1];
    code = (i & 1) ? ((byte >> 4) & 0xf) : (byte & 0xf);
  } else {
    code = codes[i];
  }
  out[i] = bf_cast<T>(dequant_value<BITS>(code, exponent));
}

// Vectorised variant: one thread per 8 values.  Eight values is exactly one
// 16B store for a 16-bit element, so every store is a single fully coalesced
// transaction; the two threads that share a block's exponent each decode it
// once rather than once per element.  Codes come in as one 4B (bf4) or 8B
// (bf8) load.
template <int BITS, typename T>
__global__ __launch_bounds__(256) void dequant_vec_kernel(
    const uint8_t* __restrict__ codes, const int8_t* __restrict__ exponents,
    int64_t n8, T* __restrict__ out) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n8) return;

  constexpr int levels = 1 << stored_mantissa_bits(BITS);
  constexpr int offset = 2 * levels - 1;
  // exponent is int8: signed, and negative for blocks below 1.0.
  const float scale = exp2f((float)exponents[i >> 1]) * (1.0f / (float)levels);

  constexpr int per = BITS == 4 ? 8 : 4;      // codes per 32-bit word
  constexpr uint32_t mask = BITS == 4 ? 0xfu : 0xffu;
  constexpr int shift = BITS == 4 ? 4 : 8;

  uint32_t w[BITS == 4 ? 1 : 2];
  if constexpr (BITS == 4) {
    w[0] = reinterpret_cast<const uint32_t*>(codes)[i];
  } else {
    const uint2 d = reinterpret_cast<const uint2*>(codes)[i];
    w[0] = d.x;
    w[1] = d.y;
  }

  if constexpr (sizeof(T) == 4) {
    T* dst = out + 8 * i;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const uint32_t code = (w[kk / per] >> (shift * (kk % per))) & mask;
      dst[kk] = bf_cast<T>((float)((int)code - offset) * scale);
    }
  } else {
    uint32_t o[4];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const uint32_t code = (w[kk / per] >> (shift * (kk % per))) & mask;
      const uint32_t h = bf_u16<T>(bf_cast<T>((float)((int)code - offset) * scale));
      if (kk & 1) o[kk >> 1] |= h << 16;
      else o[kk >> 1] = h;
    }
    reinterpret_cast<uint4*>(out)[i] = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

// ---------------------------------------------------------------------------
// quantize: float32 -> codes + exponents (mirrors quantize_blockfloat)
// ---------------------------------------------------------------------------
template <int BITS>
__global__ void quantize_kernel(const float* __restrict__ x, int64_t rows, int64_t k,
                                uint8_t* __restrict__ codes,
                                int8_t* __restrict__ exponents) {
  constexpr int levels = 1 << stored_mantissa_bits(BITS);
  constexpr int offset = 2 * levels - 1;
  const int64_t per_row = k / BLOCK;
  const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= rows * per_row) return;
  const int64_t r = b / per_row;
  const int64_t blk = b - r * per_row;
  const float* p = x + r * k + blk * BLOCK;

  float block[BLOCK];
  float mag = 0.0f;
#pragma unroll
  for (int i = 0; i < BLOCK; ++i) {
    block[i] = p[i];
    mag = fmaxf(mag, fabsf(block[i]));
  }
  const float safe = fmaxf(mag, FLT_MIN);
  const int base = floor_log2_exact(safe);

  // Three candidates: base, base-1, base+1.  Strict `<` keeps the earlier one
  // on ties, matching the reference's sequential torch.where updates.
  int best_exp = base;
  float best_err = CUDART_INF_F;
  int best_code[BLOCK];
  const int deltas[3] = {0, -1, 1};
#pragma unroll
  for (int t = 0; t < 3; ++t) {
    const int ex = base + deltas[t];
    const float scale = exp2f((float)ex);
    float err = 0.0f;
    int cand[BLOCK];
#pragma unroll
    for (int i = 0; i < BLOCK; ++i) {
      const float q = rintf(block[i] / scale * (float)levels);
      int qi = (int)q;
      qi = qi > offset ? offset : (qi < -offset ? -offset : qi);
      cand[i] = qi;
      const float v = (float)qi * (1.0f / levels) * scale;
      const float d = v - block[i];
      err += d * d;
    }
    if (err < best_err) {
      best_err = err;
      best_exp = ex;
#pragma unroll
      for (int i = 0; i < BLOCK; ++i) best_code[i] = cand[i];
    }
  }

  exponents[b] = (int8_t)best_exp;
  const int64_t base_elem = (r * k) + blk * BLOCK;
  if constexpr (BITS == 4) {
    uint32_t o[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < BLOCK; ++i) {
      const uint32_t c = (uint32_t)(best_code[i] + offset);
      o[i >> 3] |= c << (4 * (i & 7));
    }
    *reinterpret_cast<uint2*>(codes + (base_elem >> 1)) = make_uint2(o[0], o[1]);
  } else {
    uint32_t o[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < BLOCK; ++i) {
      const uint32_t c = (uint32_t)(best_code[i] + offset);
      o[i >> 2] |= c << (8 * (i & 3));
    }
    *reinterpret_cast<uint4*>(codes + base_elem) = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

// ---------------------------------------------------------------------------
// decompose: an on-grid float32 matrix -> codes + exponents
// ---------------------------------------------------------------------------
template <int BITS>
__global__ void decompose_kernel(const float* __restrict__ x, int64_t rows, int64_t k,
                                 uint8_t* __restrict__ codes,
                                 int8_t* __restrict__ exponents,
                                 int32_t* __restrict__ failed) {
  constexpr int levels = 1 << stored_mantissa_bits(BITS);
  constexpr int offset = 2 * levels - 1;
  const int64_t per_row = k / BLOCK;
  const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= rows * per_row) return;
  const int64_t r = b / per_row;
  const int64_t blk = b - r * per_row;
  const float* p = x + r * k + blk * BLOCK;

  float mag = 0.0f;
#pragma unroll
  for (int i = 0; i < BLOCK; ++i) mag = fmaxf(mag, fabsf(p[i]));
  const int base = floor_log2_exact(fmaxf(mag, FLT_MIN));

  // Same candidate order as decompose_blockfloat.
  const int shifts[4] = {0, 1, -1, 2};
  for (int s = 0; s < 4; ++s) {
    const int ex = base + shifts[s];
    const float scale = exp2f((float)ex);
    bool ok = true;
    int cand[BLOCK];
#pragma unroll
    for (int i = 0; i < BLOCK; ++i) {
      const float q = rintf(p[i] / scale * (float)levels);
      const int qi = (int)q;
      cand[i] = qi;
      const float v = (float)qi * (1.0f / levels) * scale;
      if (qi > offset || qi < -offset || v != p[i]) ok = false;
    }
    if (!ok) continue;
    exponents[b] = (int8_t)ex;
    const int64_t base_elem = (r * k) + blk * BLOCK;
    if constexpr (BITS == 4) {
      uint32_t o[2] = {0, 0};
#pragma unroll
      for (int i = 0; i < BLOCK; ++i) {
        const uint32_t c = (uint32_t)(cand[i] + offset);
        o[i >> 3] |= c << (4 * (i & 7));
      }
      *reinterpret_cast<uint2*>(codes + (base_elem >> 1)) = make_uint2(o[0], o[1]);
    } else {
      uint32_t o[4] = {0, 0, 0, 0};
#pragma unroll
      for (int i = 0; i < BLOCK; ++i) {
        const uint32_t c = (uint32_t)(cand[i] + offset);
        o[i >> 2] |= c << (8 * (i & 3));
      }
      *reinterpret_cast<uint4*>(codes + base_elem) = make_uint4(o[0], o[1], o[2], o[3]);
    }
    return;
  }
  atomicExch(failed, 1);
}

// ---------------------------------------------------------------------------
// embedding: gather rows by id
// ---------------------------------------------------------------------------
template <int BITS, typename T>
__global__ void embedding_kernel(const int64_t* __restrict__ ids,
                                 const uint8_t* __restrict__ codes,
                                 const int8_t* __restrict__ exponents,
                                 int64_t tokens, int64_t rows, int64_t k,
                                 T* __restrict__ out) {
  const int64_t n = tokens * k;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t t = i / k;
  const int64_t c = i - t * k;
  const int64_t r = ids[t];
  if (r < 0 || r >= rows) {
    out[i] = bf_cast<T>(0.0f);
    return;
  }
  const int exponent = (int)exponents[r * (k / BLOCK) + (c >> 4)];
  const int64_t e = r * k + c;
  int code;
  if constexpr (BITS == 4) {
    const uint8_t byte = codes[e >> 1];
    code = (e & 1) ? ((byte >> 4) & 0xf) : (byte & 0xf);
  } else {
    code = codes[e];
  }
  out[i] = bf_cast<T>(dequant_value<BITS>(code, exponent));
}

// ---------------------------------------------------------------------------
// Fused dequant + GEMM, tensor-core path (bf16 / f16 activations)
//
// 128x128x32 tile, 8 warps, cp.async for the activation tile and a
// register-prefetched dequantise for the weight tile (the weights cannot go
// through cp.async because they are still packed).  Both stages are double
// buffered so the weight fetch for k+1 overlaps the MMA for k.
//
// This is the no-cuBLAS fallback; ``launch_linear_cublas`` is the default path.
// ---------------------------------------------------------------------------
constexpr int PM = 128;
constexpr int PN = 128;
constexpr int PK = 32;
constexpr int PTHREADS = 256;
constexpr int PN_FRAG = 4;   // n-frags per warp (32 columns)
constexpr int PA_LD = PK + 8;
constexpr int PB_LD = PK + 8;

__device__ __forceinline__ uint32_t lsmem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
               "r"(sz));
}

__device__ __forceinline__ void ldm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldm_x2(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

template <typename T>
__device__ __forceinline__ void mma_m16n8k16(float* c, const uint32_t* a,
                                             const uint32_t* b);
template <>
__device__ __forceinline__ void mma_m16n8k16<__nv_bfloat16>(float* c, const uint32_t* a,
                                                            const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <>
__device__ __forceinline__ void mma_m16n8k16<__half>(float* c, const uint32_t* a,
                                                     const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Fetch one 16-value weight block into registers, dequantise it, and write it
// to the shared tile.  ``elem`` is the flat (row, k) offset of the block.
template <int BITS>
__device__ __forceinline__ void fetch_weight_block(const uint8_t* __restrict__ codes,
                                                   const int8_t* __restrict__ exponents,
                                                   int64_t elem, int64_t k,
                                                   uint32_t* packed, int* exponent) {
  *exponent = (int)exponents[(elem >> 4)];
  if constexpr (BITS == 4) {
    const uint2 v = *reinterpret_cast<const uint2*>(codes + (elem >> 1));
    packed[0] = v.x;
    packed[1] = v.y;
  } else {
    const uint4 v = *reinterpret_cast<const uint4*>(codes + elem);
    packed[0] = v.x; packed[1] = v.y; packed[2] = v.z; packed[3] = v.w;
  }
  (void)k;
}

template <int BITS, typename T, bool B_DENSE>
__global__ __launch_bounds__(PTHREADS) void linear_mma_kernel(
    const T* __restrict__ x, const uint8_t* __restrict__ codes,
    const int8_t* __restrict__ exponents, int64_t m, int64_t n, int64_t k,
    const T* __restrict__ bias, T* __restrict__ out, const T* __restrict__ b_dense) {
  __shared__ __align__(16) T As[2][PM][PA_LD];
  __shared__ __align__(16) T Bs[2][PN][PB_LD];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp >> 2;   // 0..1
  const int wn = warp & 3;    // 0..3
  const int64_t m0 = (int64_t)blockIdx.y * PM;
  const int64_t n0 = (int64_t)blockIdx.x * PN;

  float acc[4][PN_FRAG][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < PN_FRAG; ++j)
#pragma unroll
      for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.0f;

  const int64_t nk = (k + PK - 1) / PK;

  // Activation tile: 128 rows x 32 columns, 4 x 16B chunks per row.
  auto load_a = [&](int stage, int64_t kb) {
    const int64_t k0 = kb * PK;
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int idx = tid + it * PTHREADS;      // 0..511
      const int row = idx >> 2, c = idx & 3;
      const int64_t gm = m0 + row, gk = k0 + c * 8;
      cp_async16(lsmem(&As[stage][row][c * 8]), &x[gm * k + gk],
                 gm < m && gk + 8 <= k);
    }
  };

  auto load_b_dense = [&](int stage, int64_t kb) {
    const int64_t k0 = kb * PK;
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int idx = tid + it * PTHREADS;      // 0..511
      const int row = idx >> 2, c = idx & 3;
      const int64_t gn = n0 + row, gk = k0 + c * 8;
      cp_async16(lsmem(&Bs[stage][row][c * 8]), &b_dense[gn * k + gk],
                 gn < n && gk + 8 <= k);
    }
  };

  // Weight tile: 128 rows x 32 columns, 2 blocks of 16 per row.
  auto fetch_b = [&](int64_t kb, uint32_t* packed, int* ex) {
    const int64_t k0 = kb * PK;
    const int row = tid >> 1;          // 0..127
    const int cb = tid & 1;            // which 16-wide block
    const int64_t gn = n0 + row;
    const int64_t elem = gn * k + k0 + cb * BLOCK;
    if (gn < n && k0 + cb * BLOCK < k) {
      fetch_weight_block<BITS>(codes, exponents, elem, k, packed, ex);
    } else {
#pragma unroll
      for (int i = 0; i < (BITS == 4 ? 2 : 4); ++i) packed[i] = 0;
      *ex = 0;
    }
  };

  auto store_b = [&](int stage, const uint32_t* packed, int ex) {
    const int row = tid >> 1;
    const int cb = tid & 1;
    const uint32_t* w = packed;
    // One exponent per block: hoist the power of two out of the 16 datums.
    constexpr int levels = 1 << stored_mantissa_bits(BITS);
    constexpr int offset = 2 * levels - 1;
    const float scale = exp2f((float)ex) * (1.0f / (float)levels);
    uint32_t o[8];
    if constexpr (BITS == 4) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const uint32_t code = (w[i >> 3] >> (4 * (i & 7))) & 0xfu;
        const uint32_t h =
            bf_u16<T>(bf_cast<T>((float)((int)code - offset) * scale));
        if (i & 1) o[i >> 1] |= h << 16; else o[i >> 1] = h;
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const uint32_t code = (w[i >> 2] >> (8 * (i & 3))) & 0xffu;
        const uint32_t h =
            bf_u16<T>(bf_cast<T>((float)((int)code - offset) * scale));
        if (i & 1) o[i >> 1] |= h << 16; else o[i >> 1] = h;
      }
    }
    T* dst = &Bs[stage][row][cb * BLOCK];
    if constexpr (sizeof(T) == 2) {
      uint4* d = reinterpret_cast<uint4*>(dst);
      d[0] = make_uint4(o[0], o[1], o[2], o[3]);
      d[1] = make_uint4(o[4], o[5], o[6], o[7]);
    }
  };

  uint32_t pk0[BITS == 4 ? 2 : 4];
  int ex0 = 0;
  load_a(0, 0);
  if constexpr (B_DENSE) load_b_dense(0, 0);
  asm volatile("cp.async.commit_group;\n");
  if constexpr (!B_DENSE) {
    fetch_b(0, pk0, &ex0);
    store_b(0, pk0, ex0);
  }

  for (int64_t kb = 0; kb < nk; ++kb) {
    const int stage = (int)(kb & 1);
    uint32_t pk1[BITS == 4 ? 2 : 4];
    int ex1 = 0;
    if (kb + 1 < nk) {
      load_a(stage ^ 1, kb + 1);
      if constexpr (B_DENSE) load_b_dense(stage ^ 1, kb + 1);
      else fetch_b(kb + 1, pk1, &ex1);
      asm volatile("cp.async.commit_group;\n");
      asm volatile("cp.async.wait_group 1;\n");
    } else {
      asm volatile("cp.async.wait_group 0;\n");
    }
    __syncthreads();

    const int a_row = lane & 15;
    const int a_col = (lane >> 4) << 3;
    const int b_row = lane & 7;
    const int b_col = ((lane >> 3) & 1) << 3;
#pragma unroll
    for (int k16 = 0; k16 < PK; k16 += 16) {
      uint32_t afrag[4][4];
      uint32_t bfrag[PN_FRAG][2];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ldm_x4(afrag[i], lsmem(&As[stage][wm * 64 + i * 16 + a_row][k16 + a_col]));
#pragma unroll
      for (int j = 0; j < PN_FRAG; ++j)
        ldm_x2(bfrag[j], lsmem(&Bs[stage][wn * 32 + j * 8 + b_row][k16 + b_col]));
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < PN_FRAG; ++j) mma_m16n8k16<T>(acc[i][j], afrag[i], bfrag[j]);
    }

    if (kb + 1 < nk) {
      if constexpr (!B_DENSE) store_b(stage ^ 1, pk1, ex1);
    }
    __syncthreads();
  }

  const int crow = lane >> 2;
  const int ccol = (lane & 3) * 2;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < PN_FRAG; ++j) {
      const int64_t gm = m0 + wm * 64 + i * 16 + crow;
      const int64_t gn = n0 + wn * 32 + j * 8 + ccol;
      if (gn < n) {
        // The fragment covers a pair of adjacent columns, but n may be odd, so
        // the second column of the last pair is out of range.
        const bool has_hi = gn + 1 < n;
        const float b0 = bias ? bf_load<T>(&bias[gn]) : 0.0f;
        const float b1 = (bias != nullptr && has_hi) ? bf_load<T>(&bias[gn + 1]) : 0.0f;
        if (gm < m) {
          out[gm * n + gn] = bf_cast<T>(acc[i][j][0] + b0);
          if (has_hi) out[gm * n + gn + 1] = bf_cast<T>(acc[i][j][1] + b1);
        }
        if (gm + 8 < m) {
          out[(gm + 8) * n + gn] = bf_cast<T>(acc[i][j][2] + b0);
          if (has_hi) out[(gm + 8) * n + gn + 1] = bf_cast<T>(acc[i][j][3] + b1);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Fused dequant + GEMM, fp32 SIMT path
// ---------------------------------------------------------------------------
constexpr int FM = 32;
constexpr int FN = 32;
constexpr int FK = 16;

template <int BITS>
__global__ __launch_bounds__(256) void linear_f32_kernel(
    const float* __restrict__ x, const uint8_t* __restrict__ codes,
    const int8_t* __restrict__ exponents, int64_t m, int64_t n, int64_t k,
    const float* __restrict__ bias, float* __restrict__ out) {
  __shared__ float As[FM][FK + 1];
  __shared__ float Bs[FN][FK + 1];

  const int tid = threadIdx.x;
  const int64_t m0 = (int64_t)blockIdx.y * FM;
  const int64_t n0 = (int64_t)blockIdx.x * FN;
  const int row = tid >> 3;          // 0..31
  const int col0 = (tid & 7) * 4;    // 0,4,...,28

  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int64_t nk = (k + FK - 1) / FK;
  for (int64_t kb = 0; kb < nk; ++kb) {
    const int64_t k0 = kb * FK;
    for (int i = tid; i < FM * FK; i += 256) {
      const int r = i / FK, c = i - r * FK;
      const int64_t gm = m0 + r;
      As[r][c] = (gm < m && k0 + c < k) ? x[gm * k + k0 + c] : 0.0f;
    }
    for (int i = tid; i < FN * FK; i += 256) {
      const int r = i / FK, c = i - r * FK;
      const int64_t gn = n0 + r;
      if (gn < n && k0 + c < k) {
        // ``e0`` is the flat *element* index of this tile's first value; bf4
        // stores two elements per byte, so its byte offset is ``e0 / 2``.
        // k0 is a multiple of BLOCK and so is k, which keeps e0 even.
        const int64_t e0 = gn * k + k0;
        const int8_t* ep = exponents + gn * (k / BLOCK) + (k0 >> 4);
        int code;
        if constexpr (BITS == 4) {
          const uint8_t byte = codes[(e0 >> 1) + (c >> 1)];
          code = (c & 1) ? ((byte >> 4) & 0xf) : (byte & 0xf);
        } else {
          code = codes[e0 + c];
        }
        Bs[r][c] = dequant_value<BITS>(code, (int)ep[c >> 4]);
      } else {
        Bs[r][c] = 0.0f;
      }
    }
    __syncthreads();
#pragma unroll
    for (int c = 0; c < FK; ++c) {
      const float a = As[row][c];
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] += a * Bs[col0 + j][c];
    }
    __syncthreads();
  }

  const int64_t gm = m0 + row;
  if (gm >= m) return;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int64_t gn = n0 + col0 + j;
    if (gn < n) {
      const float b = bias ? bias[gn] : 0.0f;
      out[gm * n + gn] = acc[j] + b;
    }
  }
}

// ---------------------------------------------------------------------------
// selftest
// ---------------------------------------------------------------------------
__global__ void selftest_kernel(int32_t* out) { *out = 0x54544246; }  // "TTBF"

// ---------------------------------------------------------------------------
// cuBLAS entry points, resolved at run time
// ---------------------------------------------------------------------------
// A link-time ``-lcublas`` is not portable: pip CUDA wheels ship
// ``libcublas.so.13`` with no ``libcublas.so`` symlink, and the soname follows
// the CUDA major.  Every process that loads this extension already has the
// library mapped (torch depends on it), so the four entry points we need are
// resolved with dlsym and the fused kernel stays as the fallback when that
// fails.  Nothing here emits a link-time reference to cuBLAS.
using GemmExFn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                    int, int, int, const void*, const void*, cudaDataType,
                                    int, const void*, cudaDataType, int, const void*, void*,
                                    cudaDataType, int, cublasComputeType_t,
                                    cublasGemmAlgo_t);

struct CublasApi {
  cublasStatus_t (*create)(cublasHandle_t*) = nullptr;
  cublasStatus_t (*destroy)(cublasHandle_t) = nullptr;
  cublasStatus_t (*set_stream)(cublasHandle_t, cudaStream_t) = nullptr;
  GemmExFn gemm_ex = nullptr;
  bool ready = false;
};

inline const CublasApi& cublas_api() {
  static const CublasApi api = [] {
    CublasApi a;
    // Escape hatch that forces the fused fallback.  It is the only way to
    // exercise that path on a machine which does have cuBLAS, and it lets a
    // build with no cuBLAS at all run the same tests.
    if (const char* off = std::getenv("GPTQMODEL_BLOCKFLOAT_NO_CUBLAS");
        off != nullptr && off[0] != '\0' && off[0] != '0') {
      return a;
    }
    static const char* kSonames[] = {"libcublas.so.13", "libcublas.so.12",
                                     "libcublas.so.11", "libcublas.so"};
    void* lib = nullptr;
    // Prefer the copy that is already mapped into this process.
    for (const char* name : kSonames) {
      if ((lib = dlopen(name, RTLD_LAZY | RTLD_NOLOAD)) != nullptr) break;
    }
    for (const char* name : kSonames) {
      if (lib != nullptr) break;
      lib = dlopen(name, RTLD_LAZY);
    }
    if (lib == nullptr) return a;
    a.create = reinterpret_cast<decltype(a.create)>(dlsym(lib, "cublasCreate_v2"));
    a.destroy = reinterpret_cast<decltype(a.destroy)>(dlsym(lib, "cublasDestroy_v2"));
    a.set_stream = reinterpret_cast<decltype(a.set_stream)>(dlsym(lib, "cublasSetStream_v2"));
    a.gemm_ex = reinterpret_cast<GemmExFn>(dlsym(lib, "cublasGemmEx"));
    a.ready = a.create && a.destroy && a.set_stream && a.gemm_ex;
    return a;
  }();
  return api;
}

constexpr int kMaxDevices = 16;

// cuBLAS handles are not thread safe and carry a device-side workspace, so two
// streams must not share one even if the enqueue itself is serialised: the
// first stream's kernel can still be running when the second enqueues.  One
// handle per (device, stream) is the only policy that is correct for overlapping
// calls, which is what a stream-scoped forward produces.  The map is keyed by
// stream value and grows with the number of live streams, which torch keeps
// small; a cached handle stays bound to the stream the caller passes.
inline cudaError_t cublas_handle_for_stream(cudaStream_t stream, cublasHandle_t* out) {
  const CublasApi& api = cublas_api();
  if (!api.ready) return cudaErrorNotSupported;
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return e;
  if (dev < 0 || dev >= kMaxDevices) return cudaErrorInvalidDevice;
  static std::mutex mu;
  static std::map<std::pair<int, cudaStream_t>, cublasHandle_t> handles;
  std::lock_guard<std::mutex> lock(mu);
  const auto key = std::make_pair(dev, stream);
  const auto it = handles.find(key);
  if (it != handles.end()) {
    *out = it->second;
    return cudaSuccess;
  }
  cublasHandle_t handle = nullptr;
  if (api.create(&handle) != CUBLAS_STATUS_SUCCESS) return cudaErrorNotSupported;
  if (api.set_stream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
    api.destroy(handle);
    return cudaErrorNotSupported;
  }
  handles.emplace(key, handle);
  *out = handle;
  return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Stream-ordered temporary storage
// ---------------------------------------------------------------------------
// Scratch must not outlive the call that asked for it: a shared buffer lets two
// overlapping streams overwrite each other's decoded weights, and keeping one
// alive between calls pins an expanded copy of the model's weights.  Each call
// therefore allocates its own buffer and frees it on the same stream, where
// ``cudaFreeAsync`` is ordered behind the kernels that read it.
//
// The blocks come from a per-device pool rather than the default pool.  The
// default pool hands every freed block straight back to the driver, so a fresh
// ~34MB allocation there costs ~1.9ms -- more than the GEMM it feeds.  A pool
// that keeps its blocks makes the same alloc/free pair ~12us.  What it keeps is
// *reserved* memory (``cudaMemPoolAttrUsedMemCurrent`` returns to zero once the
// stream drains); no allocation is live between calls.
inline cudaError_t device_pool(cudaMemPool_t* out) {
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return e;
  if (dev < 0 || dev >= kMaxDevices) return cudaErrorInvalidDevice;
  static std::mutex mu;
  static cudaMemPool_t pools[kMaxDevices] = {};
  std::lock_guard<std::mutex> lock(mu);
  if (pools[dev] == nullptr) {
    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = dev;
    e = cudaMemPoolCreate(&pools[dev], &props);
    if (e != cudaSuccess) {
      pools[dev] = nullptr;
      return e;
    }
    // cudaMemPoolAttrReleaseThreshold is a 64-bit value; the runtime API takes
    // it by void*, so any 8-byte unsigned type is valid.
    std::uint64_t threshold = ~static_cast<std::uint64_t>(0);
    cudaMemPoolSetAttribute(pools[dev], cudaMemPoolAttrReleaseThreshold, &threshold);
  }
  *out = pools[dev];
  return cudaSuccess;
}

class TemporaryBuffer {
 public:
  TemporaryBuffer(size_t bytes, cudaStream_t stream) : stream_(stream) {
    cudaMemPool_t pool = nullptr;
    status_ = device_pool(&pool);
    if (status_ == cudaSuccess) {
      status_ = cudaMallocFromPoolAsync(&ptr_, bytes, pool, stream);
    } else {
      // No pool (an old driver, or a device that refuses one): correctness
      // first, so fall back to the default pool and accept the slower path.
      status_ = cudaMallocAsync(&ptr_, bytes, stream);
    }
    if (status_ != cudaSuccess) ptr_ = nullptr;
  }

  ~TemporaryBuffer() {
    if (ptr_ != nullptr) cudaFreeAsync(ptr_, stream_);
  }

  TemporaryBuffer(const TemporaryBuffer&) = delete;
  TemporaryBuffer& operator=(const TemporaryBuffer&) = delete;

  cudaError_t status() const { return status_; }
  void* get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
  cudaStream_t stream_;
  cudaError_t status_ = cudaSuccess;
};

// Epilogue for the cuBLAS path: fp32 accumulator -> bias -> T.
template <typename T>
__global__ void epilogue_kernel(const float* __restrict__ c, const T* __restrict__ bias,
                                T* __restrict__ out, int64_t m, int64_t n) {
  const int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= n) return;
  const float b = bias != nullptr ? bf_load<T>(&bias[col]) : 0.0f;
  // gridDim.y is capped at 65535, so rows are strided instead of mapped 1:1.
  // A flattened batch/sequence dimension reaches that cap in long-context runs.
  for (int64_t row = blockIdx.y; row < m; row += gridDim.y) {
    out[row * n + col] = bf_cast<T>(c[row * n + col] + b);
  }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------
inline int grid_for(int64_t n, int threads) {
  const int64_t g = (n + threads - 1) / threads;
  return (int)(g > 0x7fffffff ? 0x7fffffff : g);
}

template <int BITS, typename T>
cudaError_t launch_dequant(const uint8_t* codes, const int8_t* exponents, int64_t rows,
                           int64_t k, void* out, cudaStream_t stream) {
  const int64_t n = rows * k;
  T* op = reinterpret_cast<T*>(out);
  // The vectorised kernel issues one wide load per block on every operand.
  if ((reinterpret_cast<uintptr_t>(codes) & 15u) == 0 &&
      (reinterpret_cast<uintptr_t>(exponents) & 15u) == 0 &&
      (reinterpret_cast<uintptr_t>(op) & 15u) == 0 && (n & 7) == 0) {
    const int64_t n8 = n / 8;
    dequant_vec_kernel<BITS, T><<<grid_for(n8, 256), 256, 0, stream>>>(
        codes, exponents, n8, op);
    return cudaGetLastError();
  }
  dequant_kernel<BITS, T><<<grid_for(n, 256), 256, 0, stream>>>(
      codes, exponents, rows, k, op);
  return cudaGetLastError();
}

template <int BITS>
cudaError_t launch_linear_f32(const void* x, const uint8_t* codes,
                              const int8_t* exponents, int64_t m, int64_t n, int64_t k,
                              const void* bias, void* out, cudaStream_t stream) {
  const dim3 grid((unsigned)((n + FN - 1) / FN), (unsigned)((m + FM - 1) / FM));
  linear_f32_kernel<BITS><<<grid, 256, 0, stream>>>(
      reinterpret_cast<const float*>(x), codes, exponents, m, n, k,
      reinterpret_cast<const float*>(bias), reinterpret_cast<float*>(out));
  return cudaGetLastError();
}

template <typename T>
constexpr int dtype_of();
template <>
constexpr int dtype_of<float>() { return BLOCKFLOAT_DTYPE_F32; }
template <>
constexpr int dtype_of<__half>() { return BLOCKFLOAT_DTYPE_F16; }
template <>
constexpr int dtype_of<__nv_bfloat16>() { return BLOCKFLOAT_DTYPE_BF16; }

template <typename T>
constexpr cudaDataType cuda_type_of();
template <>
constexpr cudaDataType cuda_type_of<float>() { return CUDA_R_32F; }
template <>
constexpr cudaDataType cuda_type_of<__half>() { return CUDA_R_16F; }
template <>
constexpr cudaDataType cuda_type_of<__nv_bfloat16>() { return CUDA_R_16BF; }

// ---------------------------------------------------------------------------
// Prefill: expand the weights once, then hand the GEMM to cuBLAS
// ---------------------------------------------------------------------------
template <int BITS, typename T>
cudaError_t launch_linear_cublas(const void* x, const uint8_t* codes,
                                 const int8_t* exponents, int64_t m, int64_t n, int64_t k,
                                 const void* bias, void* out, cudaStream_t stream) {
  cublasHandle_t handle = nullptr;
  cudaError_t e = cublas_handle_for_stream(stream, &handle);
  if (e != cudaSuccess) return e;
  const CublasApi& api = cublas_api();

  const size_t wbytes = (size_t)n * k * sizeof(T);
  const size_t cbytes = (size_t)m * n * sizeof(float);
  // Owned by this call only; the free is ordered behind the epilogue on
  // ``stream``, so no expanded weight outlives the forward pass.
  TemporaryBuffer buffer(wbytes + cbytes, stream);
  if (buffer.status() != cudaSuccess) return buffer.status();
  T* w = reinterpret_cast<T*>(buffer.get());
  float* c32 = reinterpret_cast<float*>(reinterpret_cast<char*>(buffer.get()) + wbytes);

  e = blockfloat_dequant(codes, exponents, BITS, n, k, w, dtype_of<T>(), stream);
  if (e != cudaSuccess) return e;

  // Column-major cuBLAS view of the row-major operands: C^T (n x m) =
  // W^T (n x k) * x^T (k x m), i.e. out = x @ W^T.  ``w`` is row-major n x k,
  // which is column-major k x n, so it needs OP_T (leading dim k); ``x`` is
  // row-major m x k = column-major k x m, so it is used as-is.
  const float alpha = 1.0f, beta = 0.0f;
  const cudaDataType dt = cuda_type_of<T>();
  const cublasStatus_t st =
      api.gemm_ex(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha, w,
                  dt, (int)k, x, dt, (int)k, &beta, c32, CUDA_R_32F, (int)n,
                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) {
    return st == CUBLAS_STATUS_NOT_SUPPORTED ? cudaErrorNotSupported
                                             : cudaErrorInvalidValue;
  }

  const unsigned rows = (unsigned)std::min<int64_t>(m, 65535);
  const dim3 grid((unsigned)((n + 255) / 256), rows);
  epilogue_kernel<T><<<grid, 256, 0, stream>>>(
      c32, reinterpret_cast<const T*>(bias), reinterpret_cast<T*>(out), m, n);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Fallback: fused dequant + MMA, no cuBLAS
// ---------------------------------------------------------------------------
template <int BITS, typename T>
cudaError_t launch_linear_fused(const void* x, const uint8_t* codes,
                                const int8_t* exponents, int64_t m, int64_t n, int64_t k,
                                const void* bias, void* out, cudaStream_t stream) {
  const dim3 grid((unsigned)((n + PN - 1) / PN), (unsigned)((m + PM - 1) / PM));
  const T* xp = reinterpret_cast<const T*>(x);
  const T* bp = reinterpret_cast<const T*>(bias);
  T* op = reinterpret_cast<T*>(out);
  // Up to one row tile the weights are dequantised exactly once either way, so
  // keeping them in shared memory avoids the round trip through a workspace.
  if (m <= PM) {
    linear_mma_kernel<BITS, T, false><<<grid, PTHREADS, 0, stream>>>(
        xp, codes, exponents, m, n, k, bp, op, nullptr);
    return cudaGetLastError();
  }
  TemporaryBuffer buffer((size_t)n * k * sizeof(T), stream);
  if (buffer.status() != cudaSuccess) return buffer.status();
  T* w = reinterpret_cast<T*>(buffer.get());
  cudaError_t e;
  e = blockfloat_dequant(codes, exponents, BITS, n, k, w, dtype_of<T>(), stream);
  if (e == cudaSuccess) {
    linear_mma_kernel<BITS, T, true><<<grid, PTHREADS, 0, stream>>>(
        xp, codes, exponents, m, n, k, bp, op, w);
    e = cudaGetLastError();
  }
  return e;
}

template <int BITS, typename T>
cudaError_t launch_linear_mma(const void* x, const uint8_t* codes,
                              const int8_t* exponents, int64_t m, int64_t n, int64_t k,
                              const void* bias, void* out, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) return cudaErrorInvalidValue;
  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  if (cublas_api().ready && m <= kIntMax && n <= kIntMax && k <= kIntMax) {
    cudaError_t e =
        launch_linear_cublas<BITS, T>(x, codes, exponents, m, n, k, bias, out, stream);
    if (e != cudaErrorNotSupported) return e;
  }
  return launch_linear_fused<BITS, T>(x, codes, exponents, m, n, k, bias, out, stream);
}

}  // namespace

extern "C" {

cudaError_t blockfloat_dequant(const uint8_t* codes, const int8_t* exponents, int bits,
                         int64_t rows, int64_t k, void* out, int dtype,
                         cudaStream_t stream) {
  if (k % BLOCK) return cudaErrorInvalidValue;
  if (bits == 4) {
    if (dtype == BLOCKFLOAT_DTYPE_F32) return launch_dequant<4, float>(codes, exponents, rows, k, out, stream);
    if (dtype == BLOCKFLOAT_DTYPE_F16) return launch_dequant<4, __half>(codes, exponents, rows, k, out, stream);
    if (dtype == BLOCKFLOAT_DTYPE_BF16) return launch_dequant<4, __nv_bfloat16>(codes, exponents, rows, k, out, stream);
  } else if (bits == 8) {
    if (dtype == BLOCKFLOAT_DTYPE_F32) return launch_dequant<8, float>(codes, exponents, rows, k, out, stream);
    if (dtype == BLOCKFLOAT_DTYPE_F16) return launch_dequant<8, __half>(codes, exponents, rows, k, out, stream);
    if (dtype == BLOCKFLOAT_DTYPE_BF16) return launch_dequant<8, __nv_bfloat16>(codes, exponents, rows, k, out, stream);
  }
  return cudaErrorInvalidValue;
}

cudaError_t blockfloat_quantize(const float* x, int bits, int64_t rows, int64_t k,
                          uint8_t* codes, int8_t* exponents, cudaStream_t stream) {
  if (k % BLOCK) return cudaErrorInvalidValue;
  const int64_t nb = rows * (k / BLOCK);
  if (bits == 4) {
    quantize_kernel<4><<<grid_for(nb, 256), 256, 0, stream>>>(x, rows, k, codes, exponents);
    return cudaGetLastError();
  }
  if (bits == 8) {
    quantize_kernel<8><<<grid_for(nb, 256), 256, 0, stream>>>(x, rows, k, codes, exponents);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

cudaError_t blockfloat_embedding(const int64_t* ids, const uint8_t* codes,
                           const int8_t* exponents, int bits, int64_t tokens,
                           int64_t rows, int64_t k, void* out, int dtype,
                           cudaStream_t stream) {
  if (k % BLOCK) return cudaErrorInvalidValue;
  const int64_t n = tokens * k;
#define TTBF_EMB(B, T)                                                       \
  embedding_kernel<B, T><<<grid_for(n, 256), 256, 0, stream>>>(              \
      ids, codes, exponents, tokens, rows, k, reinterpret_cast<T*>(out))
  if (bits == 4) {
    if (dtype == BLOCKFLOAT_DTYPE_F32) { TTBF_EMB(4, float); return cudaGetLastError(); }
    if (dtype == BLOCKFLOAT_DTYPE_F16) { TTBF_EMB(4, __half); return cudaGetLastError(); }
    if (dtype == BLOCKFLOAT_DTYPE_BF16) { TTBF_EMB(4, __nv_bfloat16); return cudaGetLastError(); }
  } else if (bits == 8) {
    if (dtype == BLOCKFLOAT_DTYPE_F32) { TTBF_EMB(8, float); return cudaGetLastError(); }
    if (dtype == BLOCKFLOAT_DTYPE_F16) { TTBF_EMB(8, __half); return cudaGetLastError(); }
    if (dtype == BLOCKFLOAT_DTYPE_BF16) { TTBF_EMB(8, __nv_bfloat16); return cudaGetLastError(); }
  }
#undef TTBF_EMB
  return cudaErrorInvalidValue;
}

cudaError_t blockfloat_decompose(const float* x, int bits, int64_t rows, int64_t k,
                           uint8_t* codes, int8_t* exponents, int32_t* failed,
                           cudaStream_t stream) {
  if (k % BLOCK) return cudaErrorInvalidValue;
  const int64_t nb = rows * (k / BLOCK);
  if (bits == 4) {
    decompose_kernel<4><<<grid_for(nb, 256), 256, 0, stream>>>(x, rows, k, codes, exponents, failed);
    return cudaGetLastError();
  }
  if (bits == 8) {
    decompose_kernel<8><<<grid_for(nb, 256), 256, 0, stream>>>(x, rows, k, codes, exponents, failed);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

cudaError_t blockfloat_linear(const void* x, int dtype, const uint8_t* codes,
                        const int8_t* exponents, int bits, int64_t m, int64_t n,
                        int64_t k, const void* bias, void* out, cudaStream_t stream) {
  if (k % BLOCK) return cudaErrorInvalidValue;
  if (m <= 0 || n <= 0 || k <= 0) return cudaErrorInvalidValue;
  if (bits == 4) {
    if (dtype == BLOCKFLOAT_DTYPE_F32) return launch_linear_f32<4>(x, codes, exponents, m, n, k, bias, out, stream);
    if (dtype == BLOCKFLOAT_DTYPE_F16) return launch_linear_mma<4, __half>(x, codes, exponents, m, n, k, bias, out, stream);
    if (dtype == BLOCKFLOAT_DTYPE_BF16) return launch_linear_mma<4, __nv_bfloat16>(x, codes, exponents, m, n, k, bias, out, stream);
  } else if (bits == 8) {
    if (dtype == BLOCKFLOAT_DTYPE_F32) return launch_linear_f32<8>(x, codes, exponents, m, n, k, bias, out, stream);
    if (dtype == BLOCKFLOAT_DTYPE_F16) return launch_linear_mma<8, __half>(x, codes, exponents, m, n, k, bias, out, stream);
    if (dtype == BLOCKFLOAT_DTYPE_BF16) return launch_linear_mma<8, __nv_bfloat16>(x, codes, exponents, m, n, k, bias, out, stream);
  }
  return cudaErrorInvalidValue;
}

cudaError_t blockfloat_selftest(int32_t* out, cudaStream_t stream) {
  selftest_kernel<<<1, 1, 0, stream>>>(out);
  return cudaGetLastError();
}

}  // extern "C"
