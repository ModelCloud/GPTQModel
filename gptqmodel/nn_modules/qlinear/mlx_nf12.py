# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# NF12 reference: AlpinDale and the nf12 contributors, Apache-2.0, https://github.com/dphnAI/nf12
"""Packed NF12 linear and embedding inference on MLX Metal."""

import ctypes
import sys
from functools import lru_cache
from platform import mac_ver

import mlx.core as mx
import mlx.nn as nn


_THREADS = 256
_DECODE_SLICE_BYTES = 64 << 20

# Adapted from the Apache-2.0 NF12 Metal linear and codec kernels at
# https://github.com/dphnAI/nf12/tree/58e151f536f35649098027376e3b159acb78a8fb/csrc/metal
# GPT-QModel binds the packed checkpoint tensors directly, supports FP16 and
# FP32 activations and bias, and compiles these shaders through MLX at runtime.
_DECODE_HEADER = """
constant constexpr uint kSuperAtomRows = 16;
constant constexpr uint kSuperAtomCols = 64;
constant constexpr uint kSuperAtomBytes = 1536;
constant constexpr uint kPlaneBytes = 512;

struct Group {
  uint meta, lo0, lo1;
};

// ---------------------------------------------------------------------------
// Widen either half of a packed BF16 pair exactly.
inline float bf16_lo(uint pair) { return as_type<float>(pair << 16); }
inline float bf16_hi(uint pair) { return as_type<float>(pair & 0xffff0000u); }

// ---------------------------------------------------------------------------
// Group decode (SPEC.md section 3). Byte j of a/b is hi(s(2j)) / hi(s(2j+1)).
inline uint exp_base_x4(uint exp_base) { return exp_base * 0x01010101u; }

inline void decode_bulk(Group g, uint base_x4, thread uint (&r)[4]) {
  const uint m = g.meta;
  const uchar4 a = as_type<uchar4>((m & 0x87878787u) + base_x4);
  const uchar4 b = as_type<uchar4>((((m << 1) & 0x80808080u) | ((m >> 3) & 0x07070707u)) + base_x4);
  const uchar4 l0 = as_type<uchar4>(g.lo0);
  const uchar4 l1 = as_type<uchar4>(g.lo1);
  r[0] = as_type<uint>(uchar4(l0.x, a.x, l0.y, b.x));
  r[1] = as_type<uint>(uchar4(l0.z, a.y, l0.w, b.y));
  r[2] = as_type<uint>(uchar4(l1.x, a.z, l1.y, b.z));
  r[3] = as_type<uint>(uchar4(l1.z, a.w, l1.w, b.w));
}

// Escapes need no simdgroup-uniform control flow on Metal: each lane reads its own entry.
inline void decode_group(Group g, uint base_x4, device const uint4* escapes, thread uint (&r)[4]) {
  if (g.meta == 0u) {
    const uint4 v = escapes[g.lo0];
    r[0] = v.x;
    r[1] = v.y;
    r[2] = v.z;
    r[3] = v.w;
  } else {
    decode_bulk(g, base_x4, r);
  }
}

// Stage a decoded 16 x 64 tile in BF16 pairs, with a stride of Ld elements.
template <uint Ld>
inline void stage_bf16_tile(threadgroup uint* tile, uint lane, const thread uint (&r)[4][4]) {
  threadgroup uint* w0 = tile + (lane >> 2) * (Ld / 2) + (lane & 3u);
  threadgroup uint* w1 = w0 + 8 * (Ld / 2);
  #pragma unroll
  for (uint j = 0; j < 4; ++j) {
    w0[8 * j] = r[j][0];
    w0[8 * j + 4] = r[j][2];
    w1[8 * j] = r[j][1];
    w1[8 * j + 4] = r[j][3];
  }
}

struct NF12Source {
  device const uint4* packed;   // super-atoms, 96 uint4 each
  device const uint4* escapes;  // may be unused when the weight has none
  uint base_x4;

  // All four atoms of super-atom sa for this lane.
  inline void load(ulong sa, uint lane, thread uint (&r)[4][4]) const {
    device const uint4* src = packed + sa * (kSuperAtomBytes / 16) + lane;
    const uint4 m = src[0];
    const uint4 l01 = src[kPlaneBytes / 16];
    const uint4 l23 = src[2 * kPlaneBytes / 16];
    decode_group(Group{m.x, l01.x, l01.y}, base_x4, escapes, r[0]);
    decode_group(Group{m.y, l01.z, l01.w}, base_x4, escapes, r[1]);
    decode_group(Group{m.z, l23.x, l23.y}, base_x4, escapes, r[2]);
    decode_group(Group{m.w, l23.z, l23.w}, base_x4, escapes, r[3]);
  }
};

"""

_LINEAR_HEADER = _DECODE_HEADER + """
#include <metal_simdgroup_matrix>

// Small M (GEMV-like). One simdgroup streams the super-atoms (nb, kb0..kb1) of
// a 16-row block. Lane (g, t) accumulates rows g and g + 8 over its 16 columns
// per super-atom for up to MT activation rows, whose element offsets in x are
// xoff[m]. acc0/acc1 hold rows g / g + 8 and are reduced across t afterwards.
template <uint MT, typename T>
inline void gemv_accumulate(NF12Source src, ulong sa_base, uint kb0, uint kb1, device const T* x,
                            const thread ulong (&xoff)[MT], uint lane, thread float (&acc0)[MT],
                            thread float (&acc1)[MT]) {
  const uint col = 2 * (lane & 3u);
  for (uint kb = kb0; kb < kb1; ++kb) {
    uint r[4][4];
    src.load(sa_base + kb, lane, r);
    float w0[16], w1[16];  // rows g / g + 8, columns 16j + col + {0, 1, 8, 9}
    for (uint j = 0; j < 4; ++j) {
      w0[4 * j + 0] = bf16_lo(r[j][0]);
      w0[4 * j + 1] = bf16_hi(r[j][0]);
      w0[4 * j + 2] = bf16_lo(r[j][2]);
      w0[4 * j + 3] = bf16_hi(r[j][2]);
      w1[4 * j + 0] = bf16_lo(r[j][1]);
      w1[4 * j + 1] = bf16_hi(r[j][1]);
      w1[4 * j + 2] = bf16_lo(r[j][3]);
      w1[4 * j + 3] = bf16_hi(r[j][3]);
    }
    const ulong kcol = ulong(kb) * kSuperAtomCols + col;
    for (uint m = 0; m < MT; ++m) {
      device const T* xr = x + xoff[m] + kcol;
      float a0 = acc0[m], a1 = acc1[m];
      for (uint j = 0; j < 4; ++j) {
        const float x0 = float(xr[16 * j]), x1 = float(xr[16 * j + 1]);
        const float x8 = float(xr[16 * j + 8]), x9 = float(xr[16 * j + 9]);
        a0 = fma(w0[4 * j + 0], x0, a0);
        a0 = fma(w0[4 * j + 1], x1, a0);
        a0 = fma(w0[4 * j + 2], x8, a0);
        a0 = fma(w0[4 * j + 3], x9, a0);
        a1 = fma(w1[4 * j + 0], x0, a1);
        a1 = fma(w1[4 * j + 1], x1, a1);
        a1 = fma(w1[4 * j + 2], x8, a1);
        a1 = fma(w1[4 * j + 3], x9, a1);
      }
      acc0[m] = a0;
      acc1[m] = a1;
    }
  }
}

// Sums the four t lanes of each g in a fixed order; every lane of the quad
// ends with the same bits. Must be called by the whole simdgroup.
template <uint MT>
inline void gemv_reduce_quad(thread float (&acc0)[MT], thread float (&acc1)[MT]) {
  for (uint m = 0; m < MT; ++m) {
    acc0[m] += simd_shuffle_xor(acc0[m], ushort(1));
    acc1[m] += simd_shuffle_xor(acc1[m], ushort(1));
    acc0[m] += simd_shuffle_xor(acc0[m], ushort(2));
    acc1[m] += simd_shuffle_xor(acc1[m], ushort(2));
  }
}

// ---------------------------------------------------------------------------
// Medium M (register MMA). Apple's 8x8 simdgroup_matrix fragment gives lane l
// the elements (fm, fn) and (fm, fn + 1) with
//   fm = (l / 4 & 4) + (l / 2) % 4,  fn = (l / 4 & 2) * 2 + (l % 2) * 2.
// A decoded NF12 register is also a bf16 pair (row g, columns 2t, 2t + 1) of an
// 8x8 quarter of an atom, so the A fragment of W[8h.., 16j + 8q..] is register
// R[h + 2q] of lane 4 fm + fn / 2: one shuffle per fragment, no threadgroup
// staging. B[k][m] = x[m][k] is loaded directly (rows fn, fn + 1 of the token tile).
// One simdgroup streams the super-atoms of a 16-row block for MB 8-token tiles;
// acc[h][mb] holds rows 8h..8h+7 x tokens 8mb..8mb+7.
inline ushort2 mma_coord(uint lane) {  // (fm, fn)
  const uint qid = lane / 4;
  return ushort2((qid & 4u) + (lane / 2) % 4u, (qid & 2u) * 2 + (lane % 2u) * 2);
}

template <uint MB, typename T>
inline void mma_accumulate(NF12Source src, ulong sa_base, uint kb0, uint kb1, device const T* x,
                           const thread ulong (&xrow)[2 * MB], uint lane, thread simdgroup_float8x8 (&acc)[2][MB]) {
  const ushort2 c = mma_coord(lane);
  const ushort src_lane = ushort(4 * c.x + c.y / 2);
  for (uint kb = kb0; kb < kb1; ++kb) {
    uint r[4][4];
    src.load(sa_base + kb, lane, r);
    const ulong kcol = ulong(kb) * kSuperAtomCols + c.x;
    for (uint j = 0; j < 4; ++j) {
      for (uint q = 0; q < 2; ++q) {
        simdgroup_float8x8 b[MB];
        for (uint mb = 0; mb < MB; ++mb) {
          const ulong k = kcol + 16 * j + 8 * q;
          b[mb].thread_elements()[0] = float(x[xrow[2 * mb] + k]);
          b[mb].thread_elements()[1] = float(x[xrow[2 * mb + 1] + k]);
        }
        for (uint h = 0; h < 2; ++h) {
          const uint pair = simd_shuffle(r[j][h + 2 * q], src_lane);
          simdgroup_float8x8 a;
          a.thread_elements()[0] = bf16_lo(pair);
          a.thread_elements()[1] = bf16_hi(pair);
          for (uint mb = 0; mb < MB; ++mb) simdgroup_multiply_accumulate(acc[h][mb], a, b[mb], acc[h][mb]);
        }
      }
    }
  }
}
"""

_LINEAR_SETUP = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nb = threadgroup_position_in_grid.x * BLOCKS + sg / SPLITS;
    const uint ks = sg % SPLITS;
    const uint kb0 = min(ks * KB_PER_SPLIT, uint(K_BLOCKS));
    const uint kb1 = min(kb0 + KB_PER_SPLIT, uint(K_BLOCKS));
    const uint m0 = threadgroup_position_in_grid.y * TILE_M;
    const NF12Source src{
        reinterpret_cast<device const uint4*>(packed),
        reinterpret_cast<device const uint4*>(escapes),
        exp_base_x4(exp_base[0]),
    };
"""

_GEMV_SOURCE = _LINEAR_SETUP + """
    ulong xoff[TILE_M];
    float acc0[TILE_M] = {}, acc1[TILE_M] = {};
    for (uint m = 0; m < TILE_M; ++m) xoff[m] = ulong(min(m0 + m, uint(M - 1))) * K;
    if (nb < N / 16) gemv_accumulate<TILE_M>(src, ulong(nb) * K_BLOCKS, kb0, kb1, x, xoff, lane, acc0, acc1);
    gemv_reduce_quad<TILE_M>(acc0, acc1);
    const uint row = lane >> 2;
    if (SPLITS == 1) {
        if ((lane & 3) == 0 && nb < N / 16) {
            const uint n = nb * 16 + row;
            for (uint m = 0; m < TILE_M && m0 + m < M; ++m) {
                device T* y = output + ulong(m0 + m) * N + n;
                y[0] = T(acc0[m] + float(bias[n]));
                y[8] = T(acc1[m] + float(bias[n + 8]));
            }
        }
        return;
    }
    threadgroup float part[BLOCKS * SPLITS * TILE_M * 16];
    if ((lane & 3) == 0) {
        for (uint m = 0; m < TILE_M; ++m) {
            part[(sg * TILE_M + m) * 16 + row] = acc0[m];
            part[(sg * TILE_M + m) * 16 + row + 8] = acc1[m];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_position_in_threadgroup.x; i < BLOCKS * TILE_M * 16; i += 32 * BLOCKS * SPLITS) {
        const uint block = i / (TILE_M * 16), m = i / 16 % TILE_M, col = i % 16;
        const uint n = (threadgroup_position_in_grid.x * BLOCKS + block) * 16 + col;
        if (n >= N || m0 + m >= M) continue;
        const uint offset = ((block * SPLITS) * TILE_M + m) * 16 + col;
        float sum = part[offset];
        for (uint s = 1; s < SPLITS; ++s) sum += part[offset + s * TILE_M * 16];
        output[ulong(m0 + m) * N + n] = T(sum + float(bias[n]));
    }
"""

_MMA_SOURCE = _LINEAR_SETUP + """
    constexpr uint MB = TILE_M / 8;
    const ushort2 c = mma_coord(lane);
    ulong xrow[2 * MB];
    for (uint mb = 0; mb < MB; ++mb) {
        xrow[2 * mb] = ulong(min(m0 + 8 * mb + c.y, uint(M - 1))) * K;
        xrow[2 * mb + 1] = ulong(min(m0 + 8 * mb + c.y + 1, uint(M - 1))) * K;
    }
    simdgroup_float8x8 acc[2][MB];
    for (uint h = 0; h < 2; ++h)
        for (uint mb = 0; mb < MB; ++mb) acc[h][mb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    if (nb < N / 16) mma_accumulate<MB>(src, ulong(nb) * K_BLOCKS, kb0, kb1, x, xrow, lane, acc);
    if (SPLITS == 1) {
        if (nb < N / 16) {
            for (uint h = 0; h < 2; ++h) {
                const uint n = nb * 16 + 8 * h + c.x;
                const float b = float(bias[n]);
                for (uint mb = 0; mb < MB; ++mb) {
                    const uint m = m0 + 8 * mb + c.y;
                    if (m < M) output[ulong(m) * N + n] = T(acc[h][mb].thread_elements()[0] + b);
                    if (m + 1 < M) output[ulong(m + 1) * N + n] = T(acc[h][mb].thread_elements()[1] + b);
                }
            }
        }
        return;
    }
    threadgroup float part[BLOCKS * SPLITS * 16 * TILE_M];
    threadgroup float* dst = part + sg * 16 * TILE_M;
    for (uint h = 0; h < 2; ++h)
        for (uint mb = 0; mb < MB; ++mb) simdgroup_store(acc[h][mb], dst + 8 * h * TILE_M + 8 * mb, TILE_M);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_position_in_threadgroup.x; i < BLOCKS * 16 * TILE_M; i += 32 * BLOCKS * SPLITS) {
        const uint block = i / (16 * TILE_M), col = i / TILE_M % 16, m = i % TILE_M;
        const uint n = (threadgroup_position_in_grid.x * BLOCKS + block) * 16 + col;
        if (n >= N || m0 + m >= M) continue;
        const uint offset = ((block * SPLITS) * 16 + col) * TILE_M + m;
        float sum = part[offset];
        for (uint s = 1; s < SPLITS; ++s) sum += part[offset + s * 16 * TILE_M];
        output[ulong(m0 + m) * N + n] = T(sum + float(bias[n]));
    }
"""


_TENSOR_HEADER = _DECODE_HEADER + """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

constant constexpr uint kTensorK = kSuperAtomCols;
using Extents = metal::dextents<int, 2>;
using Strides = metal::array<int, 2>;
using WeightTile = metal::tensor<threadgroup bfloat, Extents, metal::tensor_inline>;

template <uint BM, uint BN, uint KS>
constant constexpr uint kScratchWords = BN * (KS > 1 && BM > kTensorK / 2 ? BM : kTensorK / 2);
"""

_TENSOR_SOURCE = """
  const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const NF12Source src{
      reinterpret_cast<device const uint4*>(packed),
      reinterpret_cast<device const uint4*>(escapes),
      exp_base_x4(exp_base[0]),
  };
  threadgroup uint scratch[SG][kScratchWords<BM, BN, KS>];
  const uint n0 = (threadgroup_position_in_grid.x * (SG / KS) + sg / KS) * BN;
  // Move the last partial tile inside the bounds so its tensor loads stay
  // full sized. This threadgroup only stores the rows assigned to it.
  const uint first_row = threadgroup_position_in_grid.y * BM;
  const uint m0 = min(first_row, uint(M) - BM);
  const bool valid = n0 < N;
  if constexpr (KS == 1) {
    if (!valid) return;
  }
  const uint k = uint(K_BLOCKS) * kTensorK;
  using Activation = metal::tensor<device T, Extents, metal::tensor_inline>;
  Activation a(const_cast<device T*>(x) + ulong(m0) * K, Extents(k, BM),
               Strides{1, int(K)});
  WeightTile w(reinterpret_cast<threadgroup bfloat*>(scratch[sg]), Extents(kTensorK, BN),
               Strides{1, kTensorK});
  auto w_tile = w.template slice<kTensorK, BN>(0, 0);
  using ActivationTile = decltype(a.template slice<kTensorK, BM>(0, 0));
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BM, BN, kTensorK, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto acc = op.template get_destination_cooperative_tensor<ActivationTile, decltype(w_tile), float>();
  #pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

  constexpr uint kb_per_split = (K_BLOCKS + KS - 1) / KS;
  const uint kb0 = (sg % KS) * kb_per_split;
  const uint kb1 = min(kb0 + kb_per_split, uint(K_BLOCKS));
  if (valid) {
    const ulong first_atom = ulong(n0 / kSuperAtomRows) * uint(K_BLOCKS);
    for (uint kb = kb0; kb < kb1; ++kb) {
      #pragma unroll
      for (uint block = 0; block < BN / kSuperAtomRows; ++block) {
        uint regs[4][4] = {};
        if (BN == kSuperAtomRows || n0 + block * kSuperAtomRows < N)
          src.load(first_atom + ulong(block) * K_BLOCKS + kb, lane, regs);
        stage_bf16_tile<kTensorK>(scratch[sg] + block * kSuperAtomRows * (kTensorK / 2), lane, regs);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      auto a_tile = a.template slice<kTensorK, BM>(kb * kTensorK, 0);
      op.run(a_tile, w_tile, acc);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // Reuse the decoded tile for split-K partials and add them in split order.
  if constexpr (KS > 1) {
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i)
      scratch[sg][lane * acc.get_capacity() + i] = as_type<uint>(acc[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid || sg % KS != 0) return;
  }
  #pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    const auto coord = acc.get_multidimensional_index(i);
    const uint m = m0 + coord[1], n = n0 + coord[0];
    if (m < first_row || (BN > kSuperAtomRows && n >= N)) continue;
    float sum = acc[i];
    #pragma unroll
    for (uint split = 1; split < KS; ++split)
      sum += as_type<float>(scratch[sg + split][lane * acc.get_capacity() + i]);
    output[ulong(m) * N + n] = T(sum + float(bias[n]));
  }
"""

@lru_cache(maxsize=2)
def _linear_kernel(use_mma):
    return mx.fast.metal_kernel(
        name="gptqmodel_nf12_linear_mma" if use_mma else "gptqmodel_nf12_linear_gemv",
        input_names=["x", "packed", "escapes", "exp_base", "bias"],
        output_names=["output"],
        header=_LINEAR_HEADER,
        source=_MMA_SOURCE if use_mma else _GEMV_SOURCE,
    )


@lru_cache(maxsize=1)
def _decode_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_nf12_linear_decode",
        input_names=["packed", "escapes", "exp_base"],
        output_names=["weight"],
        header=_DECODE_HEADER,
        source="""
            const uint gid = thread_position_in_grid.x;
            const uint lane = gid % 32;
            const ulong sa = gid / 32;
            const NF12Source src{
                reinterpret_cast<device const uint4*>(packed),
                reinterpret_cast<device const uint4*>(escapes),
                exp_base_x4(exp_base[0]),
            };
            uint r[4][4];
            src.load(sa, lane, r);
            const ulong row = (sa / K_BLOCKS) * 16 + (lane >> 2);
            const ulong col = (sa % K_BLOCKS) * 64 + 2 * (lane & 3);
            device T* dst = weight + row * K + col;
            for (uint j = 0; j < 4; ++j) {
                for (uint i = 0; i < 4; ++i) {
                    const uint offset = 16 * j + 8 * (i >> 1) + (i & 1) * 8 * K;
                    const uint pair = r[j][i];
                    if (WIDE) {
                        *reinterpret_cast<device float2*>(dst + offset) = float2(bf16_lo(pair), bf16_hi(pair));
                    } else {
                        *reinterpret_cast<device uint*>(dst + offset) = pair;
                    }
                }
            }
        """,
    )


@lru_cache(maxsize=1)
def _tensor_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_nf12_tensor",
        input_names=["x", "packed", "escapes", "exp_base", "bias"],
        output_names=["output"],
        header=_TENSOR_HEADER,
        source=_TENSOR_SOURCE,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _has_accelerated_tensorops():
    if sys.platform != "darwin" or not mx.metal.is_available():
        return False
    if int(mac_ver()[0].split(".")[0]) < 27:
        return False
    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    metal.MTLCreateSystemDefaultDevice.argtypes = ()
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = (ctypes.c_char_p,)
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.objc_msgSend.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long)
    objc.objc_msgSend.restype = ctypes.c_bool
    objc.objc_release.argtypes = (ctypes.c_void_p,)
    objc.objc_release.restype = None
    device = metal.MTLCreateSystemDefaultDevice()
    if not device:
        return False
    try:
        return objc.objc_msgSend(device, objc.sel_registerName(b"supportsFamily:"), 1010)  # Apple10
    finally:
        objc.objc_release(device)


def _tensor_prefill(rows, n, k):
    if not (16 <= rows <= 128 and 2048 <= k <= 16384):
        return False
    if n < 4096:
        return n >= 512 and k <= 4096 and (rows in (16, 32, 64) or (n <= 1024 and rows > 64))
    if n > 32768:
        return rows == 16 or 32 <= rows <= 64
    return True


def _linear_route(rows, n, k, dtype):
    narrow_router = n < 512 and 17 <= rows <= 128 and 2048 <= k <= 8192
    # Metal and MLX can narrow FP32 matmul operands on Apple10. Router
    # logits need the full mantissa, so use the register MMA tile here.
    if dtype == mx.float32 and narrow_router:
        return "simd"
    if (dtype in (mx.bfloat16, mx.float16) and (narrow_router or _tensor_prefill(rows, n, k))
            and _has_accelerated_tensorops()):
        return "tensor"
    return "simd" if rows <= 16 else "decode"


@lru_cache(maxsize=1)
def _is_m5_ultra():
    return mx.metal.is_available() and mx.device_info(mx.gpu).get("device_name") == "Apple M5 Ultra"


def _tensor_tile(rows, n, k):
    bm = 16 if rows < 32 else 32 if rows < 64 else 64
    if _is_m5_ultra():
        if 1024 <= n <= 3072 and 3072 <= k <= 4096:
            if bm == 16 and n <= 1536:
                return bm, 32, 4, 1
            # Reuse each activation tile across more output rows. At BM=64,
            # BN=32 keeps the ordered split-K partials within 32 KiB.
            return bm, 64 if bm < 64 else 32, 4, 2
        if n >= 65536 and k == 2048:
            return (bm, 64, 1, 1) if bm < 64 else (bm, 32, 2, 1)
        if n == 4096 and rows >= 96 and k >= 8192:
            # Larger token tiles can trade some split-K parallelism for reuse.
            return (bm, 32, 4, 2) if k == 8192 else (bm, 16, 2, 2)
    return bm, 16, 4, 4 if k > 4096 else 1


def _linear_launch(rows, out_features, in_features):
    use_mma = rows > 4
    tile_rows = min(16, 1 << (rows - 1).bit_length())
    n_blocks = out_features // 16
    k_blocks = in_features // 64

    # Split K for occupancy. Each split gets at least two super-atoms, and
    # the threadgroup is limited to eight SIMD groups across output blocks.
    splits = 1
    while splits < 8 and n_blocks * splits < 2048 and k_blocks >= 4 * splits:
        splits *= 2
    blocks = min(8 // splits, n_blocks)
    grid_n = (n_blocks + blocks - 1) // blocks
    grid_m = (rows + tile_rows - 1) // tile_rows
    return use_mma, tile_rows, splits, blocks, grid_n, grid_m


class _NF12Projection(nn.Module):
    def _init_packed_weight(self, in_features, out_features, num_escapes):
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if self.in_features % 64 or self.out_features % 16:
            raise ValueError("NF12 MLX needs in_features % 64 and out_features % 16")
        self.packed = mx.zeros((self.out_features, self.in_features * 3 // 2), dtype=mx.uint8)
        self.escapes = mx.zeros((max(int(num_escapes), 1), 8), dtype=mx.int16)
        self.exp_base = mx.zeros((1,), dtype=mx.uint8)

    def _weight_inputs(self, packed):
        # Empty MLX arrays bind as constants, so use a device buffer in the
        # unreachable escape branch of weights that have no escapes.
        return packed, self.escapes if self.escapes.size else packed, self.exp_base

    def _decode(self, start=0, end=None, *, dtype=mx.bfloat16):
        """Decode a 16-row-aligned slice for the current matmul."""
        end = self.out_features if end is None else end
        rows = end - start
        return _decode_kernel()(
            inputs=self._weight_inputs(self.packed[start:end]),
            template=[
                ("T", dtype), ("K", self.in_features), ("K_BLOCKS", self.in_features // 64),
                ("WIDE", dtype == mx.float32),
            ],
            grid=(rows * self.in_features // 32, 1, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(rows, self.in_features)],
            output_dtypes=[dtype],
        )[0]

    def _fused_linear(self, x, rows, bias):
        use_mma, tile_rows, splits, blocks, grid_n, grid_m = _linear_launch(
            rows, self.out_features, self.in_features,
        )
        k_blocks = self.in_features // 64
        threads = 32 * splits * blocks
        return _linear_kernel(use_mma)(
            inputs=[x, *self._weight_inputs(self.packed), bias],
            template=[
                ("T", x.dtype), ("K", self.in_features), ("N", self.out_features), ("M", rows),
                ("K_BLOCKS", k_blocks), ("KB_PER_SPLIT", (k_blocks + splits - 1) // splits),
                ("TILE_M", tile_rows), ("SPLITS", splits), ("BLOCKS", blocks),
            ],
            grid=(threads * grid_n, grid_m, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(rows, self.out_features)],
            output_dtypes=[x.dtype],
        )[0]

    def _tensor_prefill(self, x, rows, bias):
        n, k = self.out_features, self.in_features
        bm, bn, simdgroups, splits = _tensor_tile(rows, n, k)
        tile_n = bn * (simdgroups // splits)
        threads = 32 * simdgroups
        return _tensor_kernel()(
            inputs=[x, *self._weight_inputs(self.packed), bias],
            template=[
                ("M", rows), ("N", n), ("K", k), ("K_BLOCKS", k // 64),
                ("BM", bm), ("BN", bn), ("SG", simdgroups), ("KS", splits), ("T", x.dtype),
            ],
            grid=(threads * ((n + tile_n - 1) // tile_n), (rows + bm - 1) // bm, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(rows, n)],
            output_dtypes=[x.dtype],
        )[0]

    def _decode_matmul(self, x, bias):
        # BF16 addmm reduces and adds a BF16 bias in FP32 before rounding.
        # Use FP32 operands when either input or bias needs wider precision.
        compute = mx.bfloat16 if x.dtype == bias.dtype == mx.bfloat16 else mx.float32
        element_bytes = 2 if compute == mx.bfloat16 else 4
        slice_rows = max(16, _DECODE_SLICE_BYTES // (self.in_features * element_bytes * 16) * 16)
        activations = x.astype(compute)
        parts = []
        for start in range(0, self.out_features, slice_rows):
            end = min(start + slice_rows, self.out_features)
            weight = self._decode(start, end, dtype=compute)
            parts.append(mx.addmm(bias[start:end], activations, weight.T).astype(x.dtype))
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)

    def _project(self, x, bias):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        output_shape = (*x.shape[:-1], self.out_features)
        if x.size == 0 or self.out_features == 0:
            return mx.zeros(output_shape, dtype=x.dtype)
        if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported NF12 activation dtype: {x.dtype}")
        rows = x.size // self.in_features
        x = x.reshape(rows, self.in_features)
        route = _linear_route(rows, self.out_features, self.in_features, x.dtype)
        if route == "tensor":
            out = self._tensor_prefill(x, rows, bias)
        elif route == "simd":
            out = self._fused_linear(x, rows, bias)
        else:
            out = self._decode_matmul(x, bias)
        return out.reshape(output_shape)


class MlxNF12Linear(_NF12Projection):
    """Multiply activations by a packed NF12 weight on Metal."""

    def __init__(self, *, in_features, out_features, num_escapes=0, threads=_THREADS):
        super().__init__()
        self._init_packed_weight(in_features, out_features, num_escapes)
        self.threads = int(threads)
        self.bias = mx.zeros((self.out_features,), dtype=mx.float32)
        self.freeze()

    def __call__(self, x):
        return self._project(x, self.bias)


_EMBED_SOURCE = """
    constexpr uint kRowBlockMetaBytes = 32, kRowBlockBytes = 96;
    const uint gx = thread_position_in_grid.x, gy = thread_position_in_grid.y;
    const uint kb0 = PARALLEL ? gx >> 4 : 0;
    const uint id_pos = PARALLEL ? gy : 8 * gy + (gx >> 4);
    if (id_pos >= COUNT || kb0 >= K / 64) return;
    const uint j = (gx >> 2) & 3u, t = gx & 3u;
    const long id = long(ids[id_pos]);
    const bool valid = id >= 0 && ulong(id) < N;
    const uint row = uint(id & 15), lane = 4 * (row & 7u) + t;
    const ulong row_base = (ulong(id) >> 4) * (K / 64) * (kSuperAtomBytes / 4) + 4 * lane;
    const uint lo = (1 + (j >> 1)) * (kPlaneBytes / 4) + 2 * (j & 1u);
    const uint h = row >> 3;
    const uint base_x4 = exp_base_x4(exp_base[0]);
    const device uint* words = reinterpret_cast<device const uint*>(packed);
    const device uint4* table = reinterpret_cast<device const uint4*>(escapes);
    device uint* dst = reinterpret_cast<device uint*>(output) +
        (ulong(id_pos) * K + 16 * j + 2 * t) / 2;
    const uint kb1 = PARALLEL ? kb0 + 1u : K / 64;
    for (uint kb = kb0; kb < kb1; ++kb) {
        uint left = 0, right = 0;
        if (valid) {
            uint r[4];
            if constexpr (ROW_MAJOR) {
                // One row block contains 16 metadata halves and 16 low words.
                const device uchar* tile = packed + (ulong(id) * (K / 64) + kb) * kRowBlockBytes;
                const uint group = 4 * j + t;
                const uint meta = reinterpret_cast<device const ushort*>(tile)[group];
                const uint lows = reinterpret_cast<device const uint*>(tile + kRowBlockMetaBytes)[group];
                if (meta == 0) {
                    const uint2 e = reinterpret_cast<device const uint2*>(escapes)[lows];
                    left = e.x;
                    right = e.y;
                } else {
                    decode_bulk(Group{(meta & 255u) | ((meta & 0xff00u) << 8), lows & 65535u, lows >> 16},
                                base_x4, r);
                    left = r[0];
                    right = r[2];
                }
            } else {
                const ulong base = row_base + ulong(kb) * (kSuperAtomBytes / 4);
                decode_group(Group{words[base + j], words[base + lo], words[base + lo + 1]}, base_x4, table, r);
                left = r[h];
                right = r[2 + h];
            }
        }
        dst[32 * kb] = left;
        dst[32 * kb + 4] = right;
    }
"""


@lru_cache(maxsize=1)
def _embed_kernel():
    """Packed NF12 lookup with independent tiles along the row and ID axes."""
    return mx.fast.metal_kernel(
        name="gptqmodel_nf12_embedding",
        input_names=["ids", "packed", "escapes", "exp_base"],
        output_names=["output"],
        header=_DECODE_HEADER,
        source=_EMBED_SOURCE,
    )


class MlxNF12Embedding(_NF12Projection):
    """Gather BF16 rows from an NF12 table without materializing it dense.

    ``ids`` of any shape select rows of a ``[num_embeddings, embedding_dim]``
    table; out-of-range IDs give ``+0.0`` rows, matching the Torch codec.
    """

    def __init__(self, *, num_embeddings, embedding_dim, num_escapes=0):
        super().__init__()
        self._init_packed_weight(embedding_dim, num_embeddings, num_escapes)
        self.num_embeddings = self.out_features
        self.embedding_dim = self.in_features
        self._row_major = False
        self.freeze()

    @property
    def row_major(self) -> bool:
        return self._row_major

    @classmethod
    def from_weight(cls, weight, *, tied: bool):
        """Load a canonical ``gptqmodel_nf12.NF12Weight`` for MLX inference.

        Pass ``tied=True`` when this table also serves the output projection.
        Untied tables use the row-major gather layout on M5 Ultra.
        """
        packed = weight.packed.detach().cpu().contiguous()
        escapes = weight.escapes.detach().cpu().contiguous()
        row_major = not tied and _is_m5_ultra()
        if row_major:
            from ...utils.nf12_codec import rowmajor_pack_cpu

            packed, escapes = rowmajor_pack_cpu(packed, escapes, weight.exp_base)
        n, k = weight.shape
        module = cls(num_embeddings=n, embedding_dim=k, num_escapes=len(escapes))
        module.packed = mx.array(packed.numpy())
        module.escapes = mx.array(escapes.numpy())
        module.exp_base = mx.array([weight.exp_base], dtype=mx.uint8)
        module._row_major = row_major
        return module

    def as_linear(self, x):
        """Project through the packed table for a tied output head."""
        if self.row_major:
            raise ValueError("NF12 output projection requires canonical storage. Load the embedding with tied=True.")
        return self._project(x, mx.zeros((self.num_embeddings,), dtype=mx.bfloat16))

    def __call__(self, ids):
        flat = ids.reshape(-1)
        count = flat.size
        if count == 0:
            return mx.zeros((*ids.shape, self.embedding_dim), dtype=mx.bfloat16)
        columns = self.embedding_dim // 4
        # Width tiles distribute long rows across the GPU for large prefills.
        if self.row_major:
            parallel = count >= (512 if self.embedding_dim > 2048 else 2048)
        else:
            parallel = count >= 1024 and self.embedding_dim >= 4096
        threads = 256 if parallel and self.row_major else 128
        grid = ((columns + threads - 1) // threads * threads, count, 1) if parallel else (
            threads, (count + 7) // 8, 1,
        )
        index_type = mx.int64 if flat.dtype in (mx.int64, mx.uint32, mx.uint64) else mx.int32
        output = _embed_kernel()(
            inputs=[flat.astype(index_type), *self._weight_inputs(self.packed)],
            template=[("K", self.embedding_dim), ("N", self.num_embeddings), ("COUNT", count),
                      ("PARALLEL", parallel), ("ROW_MAJOR", self.row_major)],
            grid=grid,
            threadgroup=(threads, 1, 1),
            output_shapes=[(count, self.embedding_dim)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return output.reshape((*ids.shape, self.embedding_dim))


__all__ = ["MlxNF12Embedding", "MlxNF12Linear"]
