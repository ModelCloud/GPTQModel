# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# NF12 reference: AlpinDale and the nf12 contributors, Apache-2.0, https://github.com/dphnAI/nf12
"""Packed NF12 linear and embedding inference on MLX Metal."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


_THREADS = 256
_SUPER_ATOM_BYTES = 1536
_DECODE_SLICE_BYTES = 64 << 20

# Adapted from the Apache-2.0 NF12 Metal linear and codec kernels at
# https://github.com/dphnAI/nf12/tree/97c4101541a8d92d2b3d89c910c452cfa8efc9c9/csrc/metal
# GPT-QModel binds the packed checkpoint tensors directly, supports FP16 and
# FP32 activations and bias, and compiles these shaders through MLX at runtime.
_LINEAR_HEADER = """
#include <metal_simdgroup_matrix>

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
        header=_LINEAR_HEADER,
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


def _linear_launch(rows, out_features, in_features):
    use_mma = rows > 4
    tile_rows = 1 << (rows - 1).bit_length()
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


class MlxNF12Linear(nn.Module):
    """Multiply activations by a packed NF12 weight on Metal."""

    def __init__(
        self,
        *,
        in_features,
        out_features,
        num_escapes=0,
        threads=_THREADS,
    ):
        super().__init__()
        in_features = int(in_features)
        out_features = int(out_features)
        if in_features % 64 or out_features % 16:
            raise ValueError("NF12 MLX inference needs in_features % 64 and out_features % 16")
        self.in_features = in_features
        self.out_features = out_features
        self.threads = int(threads)
        self.packed = mx.zeros((out_features, in_features * 3 // 2), dtype=mx.uint8)
        self.escapes = mx.zeros((max(int(num_escapes), 1), 8), dtype=mx.int16)
        self.exp_base = mx.zeros((1,), dtype=mx.uint8)
        # A zero bias keeps one fused kernel for biased and unbiased layers.
        self.bias = mx.zeros((out_features,), dtype=mx.float32)
        self.freeze()

    def _decode(self, start=0, end=None, *, dtype=mx.bfloat16):
        """Decode a 16-row-aligned slice for the current matmul."""
        end = self.out_features if end is None else end
        rows = end - start
        return _decode_kernel()(
            inputs=[self.packed[start:end], self.escapes, self.exp_base],
            template=[
                ("T", dtype), ("K", self.in_features), ("K_BLOCKS", self.in_features // 64),
                ("WIDE", dtype == mx.float32),
            ],
            grid=(rows * self.in_features // 32, 1, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(rows, self.in_features)],
            output_dtypes=[dtype],
        )[0]

    def _fused_linear(self, x, rows):
        use_mma, tile_rows, splits, blocks, grid_n, grid_m = _linear_launch(
            rows, self.out_features, self.in_features,
        )
        k_blocks = self.in_features // 64
        threads = 32 * splits * blocks
        return _linear_kernel(use_mma)(
            inputs=[x, self.packed, self.escapes, self.exp_base, self.bias],
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

    def _decode_matmul(self, x):
        # BF16 addmm reduces and adds a BF16 bias in FP32 before rounding.
        # Use FP32 operands when either input or bias needs wider precision.
        compute = mx.bfloat16 if x.dtype == self.bias.dtype == mx.bfloat16 else mx.float32
        element_bytes = 2 if compute == mx.bfloat16 else 4
        slice_rows = max(16, _DECODE_SLICE_BYTES // (self.in_features * element_bytes * 16) * 16)
        activations = x.astype(compute)
        parts = []
        for start in range(0, self.out_features, slice_rows):
            end = min(start + slice_rows, self.out_features)
            weight = self._decode(start, end, dtype=compute)
            parts.append(mx.addmm(self.bias[start:end], activations, weight.T).astype(x.dtype))
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)

    def __call__(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        output_shape = (*x.shape[:-1], self.out_features)
        if x.size == 0 or self.out_features == 0:
            return mx.zeros(output_shape, dtype=x.dtype)
        if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported NF12 activation dtype: {x.dtype}")
        rows = x.size // self.in_features
        x = x.reshape(rows, self.in_features)
        out = self._fused_linear(x, rows) if rows <= 16 else self._decode_matmul(x)
        return out.reshape(output_shape)


_EMBED_SOURCE = """            uint physical = thread_position_in_threadgroup.x;
            uint id_chunk = threadgroup_position_in_grid.y;
            uint slot = physical >> 4;
            uint lane = physical >> 2;
            uint j = physical & 3u;
            uint t = lane & 3u;
            uint id_pos = id_chunk * 8u + slot;
            if (id_pos >= NUM_IDS) {{
                return;
            }}
            uint k_j = 16u * j + 2u * t;
            uint out_base = id_pos * K + k_j;
            int row = ids[id_pos];
            if (row < 0 || uint(row) >= N) {{
                // Vocab-parallel shards hand out-of-range IDs and expect +0.0.
                for (uint step = 0; step < SA_ROW; ++step) {{
                    uint base = out_base + step * 64u;
                    *(device uint *)(output + base) = 0u;
                    *(device uint *)(output + base + 8u) = 0u;
                }}
                return;
            }}
            uint block = uint(row) >> 4;
            uint upper = (uint(row) >> 3) & 1u;
            uint wlane = 4u * (uint(row) & 7u) + t;
            uint meta_word = block * SA_BYTES + wlane * 4u + j;
            uint low_word = block * SA_BYTES + 128u + (j >> 1) * 128u + wlane * 4u + (j & 1u) * 2u;
            // The selected row takes the lower or upper half of the paired
            // group, so both the meta byte and the low byte move by 8 bits.
            uint meta_shift = upper * 8u;
            uint byte_shift = upper * 16u;
            uint escape_base = upper * 2u;
            uint exp_base_value = uint(exp_base[0]);
            const device uint *base = (const device uint *)packed;
            for (uint step = 0; step < SA_ROW; ++step) {{
                const device uint *sa = base + step * 384u;
                uint meta = sa[meta_word];
                uint low0 = sa[low_word];
                uint low1 = sa[low_word + 1u];
                uint m_lo = (meta >> meta_shift) & 0xFFu;
                uint m_hi = (meta >> (16u + meta_shift)) & 0xFFu;
                uint l0 = (low0 >> byte_shift) & 0xFFu;
                uint l1 = (low0 >> (byte_shift + 8u)) & 0xFFu;
                uint l2 = (low1 >> byte_shift) & 0xFFu;
                uint l3 = (low1 >> (byte_shift + 8u)) & 0xFFu;
                uint v0, v1, v2, v3;
                if (meta == 0u) {{
                    v0 = uint(uint16_t(escapes[low0 * 8u + escape_base])) << 16u;
                    v1 = uint(uint16_t(escapes[low0 * 8u + escape_base + 1u])) << 16u;
                    v2 = uint(uint16_t(escapes[low0 * 8u + escape_base + 4u])) << 16u;
                    v3 = uint(uint16_t(escapes[low0 * 8u + escape_base + 5u])) << 16u;
                }} else {{
                    uint hi_lo = m_lo & 0x87u;
                    uint hi_lo1 = ((m_lo << 1u) & 0x80u) | ((m_lo >> 3u) & 0x07u);
                    uint hi_hi = m_hi & 0x87u;
                    uint hi_hi1 = ((m_hi << 1u) & 0x80u) | ((m_hi >> 3u) & 0x07u);
                    v0 = (((hi_lo + exp_base_value) << 24u) | (l0 << 16u));
                    v1 = (((hi_lo1 + exp_base_value) << 24u) | (l1 << 16u));
                    v2 = (((hi_hi + exp_base_value) << 24u) | (l2 << 16u));
                    v3 = (((hi_hi1 + exp_base_value) << 24u) | (l3 << 16u));
                }}
                // The BF16 word is the top half of the assembled float bits.
                uint out = out_base + step * 64u;
                *(device uint *)(output + out) = (v0 >> 16u) | ((v1 >> 16u) << 16u);
                *(device uint *)(output + out + 8u) = (v2 >> 16u) | ((v3 >> 16u) << 16u);
            }}
"""


@lru_cache(maxsize=1)
def _embed_kernel():
    """Packed NF12 embedding lookup: 8 IDs per threadgroup, 4 values per thread."""
    return mx.fast.metal_kernel(
        name="gptqmodel_nf12_embedding",
        input_names=["ids", "packed", "escapes", "exp_base"],
        output_names=["output"],
        source=_EMBED_SOURCE,
    )


class MlxNF12Embedding(nn.Module):
    """Gather BF16 rows from an NF12 table without materializing it dense.

    ``ids`` of any shape select rows of a ``[num_embeddings, embedding_dim]``
    table; out-of-range IDs give ``+0.0`` rows, matching the Torch codec.
    """

    def __init__(self, *, num_embeddings, embedding_dim, num_escapes=0):
        super().__init__()
        num_embeddings = int(num_embeddings)
        embedding_dim = int(embedding_dim)
        if embedding_dim % 64 or num_embeddings % 16:
            raise ValueError("NF12 MLX embedding needs dim % 64 and num_embeddings % 16")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.packed = mx.zeros((num_embeddings, embedding_dim * 3 // 2), dtype=mx.uint8)
        self.escapes = mx.zeros((max(int(num_escapes), 1), 8), dtype=mx.int16)
        self.exp_base = mx.zeros((1,), dtype=mx.uint8)
        self.freeze()

    def __call__(self, ids):
        flat = ids.reshape(-1)
        count = flat.size
        if count == 0:
            return mx.zeros((*ids.shape, self.embedding_dim), dtype=mx.bfloat16)
        output = _embed_kernel()(
            inputs=[flat.astype(mx.int32), self.packed, self.escapes, self.exp_base],
            template=[
                ("K", self.embedding_dim), ("N", self.num_embeddings),
                ("SA_ROW", self.embedding_dim // 64),
                ("SA_BYTES", self.embedding_dim * 6),
                ("NUM_IDS", count),
            ],
            grid=(128, (count + 7) // 8, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(count, self.embedding_dim)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return output.reshape((*ids.shape, self.embedding_dim))


__all__ = ["MlxNF12Embedding", "MlxNF12Linear"]
