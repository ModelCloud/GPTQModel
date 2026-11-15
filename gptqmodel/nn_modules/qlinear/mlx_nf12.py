# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# NF12 reference: AlpinDale and the nf12 contributors, Apache-2.0, https://github.com/dphnAI/nf12
"""Packed NF12 (12-bit BF16) inference kernels for MLX Metal.

NF12 keeps the checkpoint's tiled 12-byte group stream: four meta bytes hold the
sign and the 3-bit exponent offset, eight low bytes hold the mantissa tail, and
a group whose meta bytes are all zero stores its eight BF16 words in the escape
table. This module decodes that stream on Metal and never installs a dense
weight. Three paths cover the token counts that matter on Apple silicon:

* one token: the decode is fused into a dot product,
* 2..16 tokens: a fused tile kernel decodes one 1536-byte super-atom per K step
  into threadgroup memory and runs it through the simdgroup matrix units, so the
  packed stream is read exactly once and no BF16 weight buffer is written,
* more tokens: one decode into BF16 followed by an MLX matmul, because the
  padded 16-row MMA tile would re-read the weights per token block.

NF12 is a lossless BF16 encoding, so every path reproduces the checkpoint's
exact BF16 weights; only the accumulation order differs from the Torch codec.
"""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


_THREADS = 256
_SUPER_ATOM_BYTES = 1536
# Token count through which the fused MMA tile beats a decode plus matmul.
_PREFILL_ROWS = 16


@lru_cache(maxsize=1)
def _decode_kernel():
    """Decode the packed stream into the exact BF16 weight matrix."""

    return mx.fast.metal_kernel(
        name="gptqmodel_nf12_decode",
        input_names=["packed", "escapes", "exp_base"],
        output_names=["weight"],
        source="""
            uint col = thread_position_in_grid.x;
            uint column = thread_position_in_grid.y;
            uint base = (column >> 4) * SA_BYTES + (col >> 6) * 1536u;
            uint lane = 4u * (column & 7u) + ((col >> 1) & 3u);
            uint j = (col >> 4) & 3u;
            uint meta_offset = base + lane * 16u + j * 4u;
            uint low_offset = base + 512u + (j >> 1) * 512u + lane * 16u + (j & 1u) * 8u;
            uint m0 = packed[meta_offset];
            uint m1 = packed[meta_offset + 1u];
            uint m2 = packed[meta_offset + 2u];
            uint m3 = packed[meta_offset + 3u];
            uint slot = 4u * ((col >> 3) & 1u) + 2u * ((column >> 3) & 1u) + (col & 1u);
            uint bits;
            if ((m0 | m1 | m2 | m3) == 0u) {
                uint index = uint(packed[low_offset]) | (uint(packed[low_offset + 1u]) << 8)
                    | (uint(packed[low_offset + 2u]) << 16) | (uint(packed[low_offset + 3u]) << 24);
                bits = uint(uint16_t(escapes[index * 8u + slot]));
            } else {
                uint meta = slot < 2u ? m0 : (slot < 4u ? m1 : (slot < 6u ? m2 : m3));
                uint high = (slot & 1u) == 0u ? (meta & 0x87u)
                    : (((meta << 1u) & 0x80u) | ((meta >> 3u) & 0x07u));
                high += uint(exp_base[0]);
                bits = (high << 8u) | uint(packed[low_offset + slot]);
            }
            weight[column * K + col] = bfloat16_t(as_type<float>(bits << 16));
        """,
    )


@lru_cache(maxsize=1)
def _prefill_kernel():
    """Decode one super-atom per K step and multiply it on the matrix units.

    A super-atom is exactly 16 weight rows by 64 K values, so a threadgroup that
    owns 16 output columns reads each packed byte once per token block. The
    decoded tile is staged as ``[k][out]`` in threadgroup memory, which is the
    operand order ``simdgroup_multiply_accumulate`` expects for ``x @ W.T``.
    """

    return mx.fast.metal_kernel(
        name="gptqmodel_nf12_prefill",
        input_names=["x", "packed", "escapes", "exp_base"],
        output_names=["output"],
        header="#include <metal_simdgroup_matrix>\n",
        source="""
            uint physical = thread_position_in_threadgroup.x;
            uint simdgroup = physical >> 5;
            uint output_tile = threadgroup_position_in_grid.y;
            threadgroup float tile_weight[64 * TILE_COLUMNS];
            threadgroup float tile_input[16 * 64];
            simdgroup_float8x8 accumulator = make_filled_simdgroup_matrix<float, 8>(0.0f);
            uint row_base = threadgroup_position_in_grid.z * 16u;
            uint super_atom = output_tile * SA_ROW;
            uint exp_base_value = uint(exp_base[0]);

            for (uint input_tile = 0; input_tile < K_TILES; ++input_tile) {
                uint base = (super_atom + input_tile) * 1536u;
                for (uint position = physical; position < 1024u; position += THREADS) {
                    uint out_row = position & 15u;
                    uint k_local = position >> 4;
                    uint lane = 4u * (out_row & 7u) + ((k_local >> 1) & 3u);
                    uint j = (k_local >> 4) & 3u;
                    uint meta_offset = base + lane * 16u + j * 4u;
                    uint low_offset = base + 512u + (j >> 1) * 512u + lane * 16u + (j & 1u) * 8u;
                    uint m0 = packed[meta_offset];
                    uint m1 = packed[meta_offset + 1u];
                    uint m2 = packed[meta_offset + 2u];
                    uint m3 = packed[meta_offset + 3u];
                    uint slot = 4u * ((k_local >> 3) & 1u) + 2u * ((out_row >> 3) & 1u) + (k_local & 1u);
                    uint bits;
                    if ((m0 | m1 | m2 | m3) == 0u) {
                        uint index = uint(packed[low_offset]) | (uint(packed[low_offset + 1u]) << 8)
                            | (uint(packed[low_offset + 2u]) << 16) | (uint(packed[low_offset + 3u]) << 24);
                        bits = uint(uint16_t(escapes[index * 8u + slot]));
                    } else {
                        uint meta = slot < 2u ? m0 : (slot < 4u ? m1 : (slot < 6u ? m2 : m3));
                        uint high = (slot & 1u) == 0u ? (meta & 0x87u)
                            : (((meta << 1u) & 0x80u) | ((meta >> 3u) & 0x07u));
                        high += exp_base_value;
                        bits = (high << 8u) | uint(packed[low_offset + slot]);
                    }
                    tile_weight[k_local * TILE_COLUMNS + out_row] = as_type<float>(bits << 16);
                }
                for (uint position = physical; position < 1024u; position += THREADS) {
                    uint token = position >> 6;
                    uint k_local = position & 63u;
                    tile_input[position] = row_base + token < ROWS
                        ? float(x[(row_base + token) * K + input_tile * 64u + k_local])
                        : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (simdgroup < SIMD_GROUPS) {
                    uint input_row = (simdgroup / COLUMN_GROUPS) * 512u;
                    uint weight_column = (simdgroup % COLUMN_GROUPS) * 8u;
                    simdgroup_float8x8 input_fragment;
                    simdgroup_float8x8 weight_fragment;
                    for (uint part = 0; part < 8u; ++part) {
                        simdgroup_load(input_fragment, tile_input + input_row + part * 8u, 64u);
                        simdgroup_load(
                            weight_fragment,
                            tile_weight + part * 8u * TILE_COLUMNS + weight_column,
                            TILE_COLUMNS
                        );
                        simdgroup_multiply_accumulate(
                            accumulator, input_fragment, weight_fragment, accumulator
                        );
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (simdgroup < SIMD_GROUPS) {
                simdgroup_store(
                    accumulator,
                    output + (row_base + simdgroup / COLUMN_GROUPS * 8u) * N
                        + output_tile * TILE_COLUMNS
                        + simdgroup % COLUMN_GROUPS * 8u,
                    N
                );
            }
        """,
    )


@lru_cache(maxsize=3)
def _gemv_kernel(output_dtype):
    """Decode and dot one activation row in a single Metal dispatch."""

    output_types = {
        mx.float16: ("fp16", "half"),
        mx.bfloat16: ("bf16", "bfloat16_t"),
        mx.float32: ("fp32", "float"),
    }
    try:
        output_suffix, output_type = output_types[output_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported NF12 activation dtype: {output_dtype}") from exc
    return mx.fast.metal_kernel(
        name=f"gptqmodel_nf12_gemv_{output_suffix}",
        input_names=["x", "packed", "escapes", "exp_base", "bias"],
        output_names=["output"],
        source="""
            uint lane = thread_position_in_threadgroup.x;
            uint column = threadgroup_position_in_grid.y;
            uint row_base = (column >> 4) * SA_BYTES;
            uint lane_base = 4u * (column & 7u);
            uint rh = (column >> 3) & 1u;
            uint exp_base_value = uint(exp_base[0]);
            float sum = 0.0f;
            for (uint tile = lane; tile < TILES; tile += THREADS) {
                uint kb = tile >> 2;
                uint j = tile & 3u;
                uint k0 = tile * 16u;
                uint base = row_base + kb * 1536u;
                uint meta_base = base + j * 4u;
                uint low_base = base + 512u + (j >> 1) * 512u + (j & 1u) * 8u;
                for (uint t = 0; t < 4u; ++t) {
                    uint offset = (lane_base + t) * 16u;
                    uint meta_offset = meta_base + offset;
                    uint low_offset = low_base + offset;
                    uint m0 = packed[meta_offset];
                    uint m1 = packed[meta_offset + 1u];
                    uint m2 = packed[meta_offset + 2u];
                    uint m3 = packed[meta_offset + 3u];
                    bool escaped = (m0 | m1 | m2 | m3) == 0u;
                    uint escape_index = 0u;
                    if (escaped) {
                        escape_index = uint(packed[low_offset]) | (uint(packed[low_offset + 1u]) << 8)
                            | (uint(packed[low_offset + 2u]) << 16) | (uint(packed[low_offset + 3u]) << 24);
                    }
                    for (uint ch = 0; ch < 2u; ++ch) {
                        for (uint p = 0; p < 2u; ++p) {
                            uint slot = 4u * ch + 2u * rh + p;
                            uint bits;
                            if (escaped) {
                                bits = uint(uint16_t(escapes[escape_index * 8u + slot]));
                            } else {
                                uint meta = slot < 2u ? m0 : (slot < 4u ? m1 : (slot < 6u ? m2 : m3));
                                uint high = (slot & 1u) == 0u ? (meta & 0x87u)
                                    : (((meta << 1u) & 0x80u) | ((meta >> 3u) & 0x07u));
                                high += exp_base_value;
                                bits = (high << 8u) | uint(packed[low_offset + slot]);
                            }
                            uint k = k0 + 8u * ch + 2u * t + p;
                            sum = metal::fma(float(x[k]), as_type<float>(bits << 16), sum);
                        }
                    }
                }
            }
            float reduced = simd_sum(sum);
            threadgroup float partials[GROUPS];
            if ((lane & 31u) == 0u) partials[lane >> 5] = reduced;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                float total = 0.0f;
                for (uint index = 0u; index < GROUPS; ++index) total += partials[index];
                output[column] = OUTPUT_TYPE(total + bias[column]);
            }
        """.replace("OUTPUT_TYPE", output_type),
    )


class MlxNF12Linear(nn.Module):
    """Multiply activations by the packed NF12 weight without dense expansion."""

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

    def _decode(self):
        return _decode_kernel()(
            inputs=[self.packed, self.escapes, self.exp_base],
            template=[("K", self.in_features), ("SA_BYTES", self.in_features // 64 * _SUPER_ATOM_BYTES)],
            grid=(self.in_features, self.out_features, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(self.out_features, self.in_features)],
            output_dtypes=[mx.bfloat16],
        )[0]

    def _prefill(self, x, rows):
        """Fused decode-plus-MMA tile for 2..16 tokens."""
        tile_columns = 16
        threads = tile_columns * 8
        padded = ((rows + 15) // 16) * 16
        return _prefill_kernel()(
            inputs=[x, self.packed, self.escapes, self.exp_base],
            template=[
                ("K", self.in_features), ("N", self.out_features),
                ("ROWS", rows), ("K_TILES", self.in_features // 64),
                ("SA_ROW", self.in_features // 64),
                ("THREADS", threads), ("TILE_COLUMNS", tile_columns),
                # The tile is 16 token rows, so the 8x8 grid has two row
                # groups; extra threads only join the decode and load loops.
                ("COLUMN_GROUPS", tile_columns // 8),
                ("SIMD_GROUPS", (tile_columns // 8) * 2),
            ],
            grid=(threads, self.out_features // tile_columns, padded // 16),
            threadgroup=(threads, 1, 1),
            output_shapes=[(padded, self.out_features)],
            output_dtypes=[mx.float32],
        )[0][:rows]

    def _gemv(self, x):
        threads = self.threads
        return _gemv_kernel(x.dtype)(
            inputs=[x, self.packed, self.escapes, self.exp_base, self.bias],
            template=[
                ("K", self.in_features),
                ("TILES", self.in_features // 16),
                ("THREADS", threads),
                ("GROUPS", threads // 32),
                ("SA_BYTES", self.in_features // 64 * _SUPER_ATOM_BYTES),
            ],
            grid=(threads, self.out_features, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(self.out_features,)],
            output_dtypes=[x.dtype],
        )[0]

    def __call__(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        output_shape = (*x.shape[:-1], self.out_features)
        if x.size == 0:
            return mx.zeros(output_shape, dtype=x.dtype)
        if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported NF12 activation dtype: {x.dtype}")
        rows = x.size // self.in_features
        if rows == 1:
            # One dispatch: the decode never leaves the dot product.
            return self._gemv(x.reshape(self.in_features)).reshape(output_shape)
        if rows <= _PREFILL_ROWS:
            # 2..16 tokens: the packed stream is read once, straight into the
            # matrix units, instead of round-tripping a dense BF16 weight.
            output = self._prefill(x.reshape(rows, self.in_features), rows)
            return (output + self.bias).astype(x.dtype).reshape(output_shape)
        # NF12 weights are BF16, so the multiply stays in BF16 for FP16/BF16
        # activations; FP32 activations widen the exact decode instead of
        # rounding it down.
        compute = mx.float32 if x.dtype == mx.float32 else mx.bfloat16
        weight = self._decode()
        if compute != mx.bfloat16:
            weight = weight.astype(compute)
        output = mx.matmul(x.astype(compute), weight.T) + self.bias
        return output.astype(x.dtype).reshape(output_shape)


__all__ = ["MlxNF12Linear"]
