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
# Split-K keeps the 16-column tile grid wide enough to fill the GPU; the
# reduce pass it adds costs one pass over the activation-sized output.
_TARGET_TILES = 1024
_MAX_SPLITS = 8


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


_PREFILL_SOURCE = """
            uint physical = thread_position_in_threadgroup.x;
            uint simdgroup = physical >> 5;
            uint column_block = threadgroup_position_in_grid.y;
            uint split = threadgroup_position_in_grid.z / ROW_BLOCKS;
            uint row_base = (threadgroup_position_in_grid.z % ROW_BLOCKS) * 16u;
            threadgroup uint words[384];
            threadgroup float tile_weight[64 * TILE_COLUMNS];
            threadgroup float tile_input[16 * 64];
            threadgroup float tile_out[16 * TILE_COLUMNS];
            simdgroup_float8x8 accumulator = make_filled_simdgroup_matrix<float, 8>(0.0f);
            uint exp_base_value = uint(exp_base[0]);

            for (uint super_atom = split; super_atom < SA_ROW; super_atom += SPLITS) {
                // Super-atoms are contiguous 1536-byte blocks in the packed
                // stream; a 16-row block starts SA_BYTES words in.
                uint base = column_block * SA_BYTES + super_atom * 384u;
                for (uint index = physical; index < 384u; index += THREADS) {
                    words[index] = ((const device uint *)packed)[base + index];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // The decode is driven by the layout: one (lane, j) group is
                // three shared words that expand to eight BF16 values. The
                // slots are unrolled so every shift and mask is a constant.
                for (uint group = physical; group < 128u; group += THREADS) {
                    uint lane = group >> 2;
                    uint j = group & 3u;
                    uint meta = words[lane * 4u + j];
                    uint low0 = words[128u + (j >> 1) * 128u + lane * 4u + (j & 1u) * 2u];
                    uint low1 = words[129u + (j >> 1) * 128u + lane * 4u + (j & 1u) * 2u];
                    uint k_j = 16u * j + 2u * (lane & 3u);
                    uint row_rh = lane >> 2;
                    if (meta == 0u) {
                        tile_weight[(k_j + 0u) * TILE_COLUMNS + row_rh] = as_type<float>(uint(uint16_t(escapes[low0 * 8u + 0u])) << 16u);
                        tile_weight[(k_j + 1u) * TILE_COLUMNS + row_rh] = as_type<float>(uint(uint16_t(escapes[low0 * 8u + 1u])) << 16u);
                        tile_weight[(k_j + 0u) * TILE_COLUMNS + row_rh + 8u] = as_type<float>(uint(uint16_t(escapes[low0 * 8u + 2u])) << 16u);
                        tile_weight[(k_j + 1u) * TILE_COLUMNS + row_rh + 8u] = as_type<float>(uint(uint16_t(escapes[low0 * 8u + 3u])) << 16u);
                        tile_weight[(k_j + 8u) * TILE_COLUMNS + row_rh] = as_type<float>(uint(uint16_t(escapes[low0 * 8u + 4u])) << 16u);
                        tile_weight[(k_j + 9u) * TILE_COLUMNS + row_rh] = as_type<float>(uint(uint16_t(escapes[low0 * 8u + 5u])) << 16u);
                        tile_weight[(k_j + 8u) * TILE_COLUMNS + row_rh + 8u] = as_type<float>(uint(uint16_t(escapes[low0 * 8u + 6u])) << 16u);
                        tile_weight[(k_j + 9u) * TILE_COLUMNS + row_rh + 8u] = as_type<float>(uint(uint16_t(escapes[low0 * 8u + 7u])) << 16u);
                    } else {
                        uint m0 = meta & 0xFFu;
                        uint m1 = (meta >> 8u) & 0xFFu;
                        uint m2 = (meta >> 16u) & 0xFFu;
                        uint m3 = (meta >> 24u) & 0xFFu;
                        tile_weight[(k_j + 0u) * TILE_COLUMNS + row_rh] = as_type<float>(((((m0 & 0x87u) + exp_base_value) << 8u) | (low0 & 0xFFu)) << 16u);
                        tile_weight[(k_j + 1u) * TILE_COLUMNS + row_rh] = as_type<float>(((((((m0 << 1u) & 0x80u) | ((m0 >> 3u) & 0x07u)) + exp_base_value) << 8u) | ((low0 >> 8u) & 0xFFu)) << 16u);
                        tile_weight[(k_j + 0u) * TILE_COLUMNS + row_rh + 8u] = as_type<float>(((((m1 & 0x87u) + exp_base_value) << 8u) | ((low0 >> 16u) & 0xFFu)) << 16u);
                        tile_weight[(k_j + 1u) * TILE_COLUMNS + row_rh + 8u] = as_type<float>(((((((m1 << 1u) & 0x80u) | ((m1 >> 3u) & 0x07u)) + exp_base_value) << 8u) | ((low0 >> 24u) & 0xFFu)) << 16u);
                        tile_weight[(k_j + 8u) * TILE_COLUMNS + row_rh] = as_type<float>(((((m2 & 0x87u) + exp_base_value) << 8u) | (low1 & 0xFFu)) << 16u);
                        tile_weight[(k_j + 9u) * TILE_COLUMNS + row_rh] = as_type<float>(((((((m2 << 1u) & 0x80u) | ((m2 >> 3u) & 0x07u)) + exp_base_value) << 8u) | ((low1 >> 8u) & 0xFFu)) << 16u);
                        tile_weight[(k_j + 8u) * TILE_COLUMNS + row_rh + 8u] = as_type<float>(((((m3 & 0x87u) + exp_base_value) << 8u) | ((low1 >> 16u) & 0xFFu)) << 16u);
                        tile_weight[(k_j + 9u) * TILE_COLUMNS + row_rh + 8u] = as_type<float>(((((((m3 << 1u) & 0x80u) | ((m3 >> 3u) & 0x07u)) + exp_base_value) << 8u) | ((low1 >> 24u) & 0xFFu)) << 16u);
                    }
                }
                for (uint position = physical; position < 1024u; position += THREADS) {
                    uint token = position >> 6;
                    uint k_local = position & 63u;
                    tile_input[position] = row_base + token < ROWS
                        ? float(x[(row_base + token) * K + super_atom * 64u + k_local])
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
                    tile_out + (simdgroup / COLUMN_GROUPS) * 8u * TILE_COLUMNS
                        + (simdgroup % COLUMN_GROUPS) * 8u,
                    TILE_COLUMNS
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // The accumulator is only reachable through threadgroup memory, so
            // the bias and the single output rounding happen here.
            for (uint position = physical; position < 256u; position += THREADS) {
                uint r = position >> 4;
                uint c = position & 15u;
                uint row = row_base + r;
                if (row >= ROWS) continue;
                uint column = column_block * TILE_COLUMNS + c;
                float value = tile_out[position];
                if (SPLITS == 1) {
                    output[row * N + column] = OUTPUT_TYPE(value + bias[column]);
                } else {
                    output[split * ROWS * N + row * N + column] = OUTPUT_TYPE(value);
                }
            }
"""


@lru_cache(maxsize=3)
def _prefill_kernel(output_dtype):
    """Decode a super-atom per K step and multiply it on the matrix units.

    A super-atom is exactly 16 weight rows by 64 K values in 1536 bytes, so the
    threadgroup that owns those 16 output columns reads it with coalesced
    "'``uint32`` loads and decodes it out of threadgroup memory. Reading the packed
    bytes in place instead issues one 1-byte load per meta byte, which costs four
    memory transactions for every word it touches.

    The decode is driven by the layout itself: one "'``(lane, j)`` group is three
    contiguous words that expand to eight BF16 values, so 128 threads decode a
    whole super-atom with three shared loads each. The result is staged as
    `[k][out]`, the operand order `simdgroup_multiply_accumulate` expects for
    `x @ W.T`, so no dense BF16 weight is ever written.
    """

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
        name=f"gptqmodel_nf12_prefill_{output_suffix}",
        input_names=["x", "packed", "escapes", "exp_base", "bias"],
        output_names=["output"],
        header="#include <metal_simdgroup_matrix>\n",
        source=_PREFILL_SOURCE.replace("OUTPUT_TYPE", output_type),
    )


@lru_cache(maxsize=3)
def _reduce_kernel(output_dtype):
    """Sum the split-K partials, add the bias, and round once."""

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
        name=f"gptqmodel_nf12_reduce_{output_suffix}",
        input_names=["partial", "bias"],
        output_names=["output"],
        source="""
            uint index = thread_position_in_grid.x;
            uint column = index % N;
            float total = 0.0f;
            for (uint split = 0; split < SPLITS; ++split) {
                total += partial[split * ROWS_N + index];
            }
            output[index] = OUTPUT_TYPE(total + bias[column]);
        """.replace("OUTPUT_TYPE", output_type),
    )


_GEMV16_SOURCE = """
            uint physical = thread_position_in_threadgroup.x;
            uint column_block = threadgroup_position_in_grid.y;
            uint split = threadgroup_position_in_grid.z;
            // PHASES K-streams share one threadgroup so the column-block grid is
            // not the only source of parallelism.
            uint phase = physical >> 7;
            uint tid = physical & 127u;
            threadgroup uint words[384];
            threadgroup float reduce[16 * THREADS];
            uint lane = tid >> 2;
            uint j = tid & 3u;
            float sum_low = 0.0f;
            float sum_high = 0.0f;
            uint exp_base_value = uint(exp_base[0]);

            for (uint step = phase; step < SA_ROW; step += PHASES) {
                uint base = column_block * SA_BYTES + step * 384u;
                for (uint index = tid; index < 384u; index += 128u) {
                    words[index] = ((const device uint *)packed)[base + index];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                uint meta = words[lane * 4u + j];
                uint low0 = words[128u + (j >> 1) * 128u + lane * 4u + (j & 1u) * 2u];
                uint low1 = words[129u + (j >> 1) * 128u + lane * 4u + (j & 1u) * 2u];
                uint k_j = step * 64u + 16u * j + 2u * (lane & 3u);
                // A group whose four meta bytes are all zero carries its eight
                // BF16 words in the escape table, indexed by low0.
                if (meta == 0u) {
                    sum_low = metal::fma(float(x[k_j + 0u]), as_type<float>(uint(uint16_t(escapes[low0 * 8u + 0u])) << 16u), sum_low);
                    sum_low = metal::fma(float(x[k_j + 1u]), as_type<float>(uint(uint16_t(escapes[low0 * 8u + 1u])) << 16u), sum_low);
                    sum_high = metal::fma(float(x[k_j + 0u]), as_type<float>(uint(uint16_t(escapes[low0 * 8u + 2u])) << 16u), sum_high);
                    sum_high = metal::fma(float(x[k_j + 1u]), as_type<float>(uint(uint16_t(escapes[low0 * 8u + 3u])) << 16u), sum_high);
                    sum_low = metal::fma(float(x[k_j + 8u]), as_type<float>(uint(uint16_t(escapes[low0 * 8u + 4u])) << 16u), sum_low);
                    sum_low = metal::fma(float(x[k_j + 9u]), as_type<float>(uint(uint16_t(escapes[low0 * 8u + 5u])) << 16u), sum_low);
                    sum_high = metal::fma(float(x[k_j + 8u]), as_type<float>(uint(uint16_t(escapes[low0 * 8u + 6u])) << 16u), sum_high);
                    sum_high = metal::fma(float(x[k_j + 9u]), as_type<float>(uint(uint16_t(escapes[low0 * 8u + 7u])) << 16u), sum_high);                } else {
                    uint m0 = meta & 0xFFu;
                    uint m1 = (meta >> 8u) & 0xFFu;
                    uint m2 = (meta >> 16u) & 0xFFu;
                    uint m3 = (meta >> 24u) & 0xFFu;
                    sum_low = metal::fma(float(x[k_j + 0u]), as_type<float>(((((m0 & 0x87u) + exp_base_value) << 8u) | (low0 & 0xFFu)) << 16u), sum_low);
                    sum_low = metal::fma(float(x[k_j + 1u]), as_type<float>(((((((m0 << 1u) & 0x80u) | ((m0 >> 3u) & 0x07u)) + exp_base_value) << 8u) | ((low0 >> 8u) & 0xFFu)) << 16u), sum_low);
                    sum_high = metal::fma(float(x[k_j + 0u]), as_type<float>(((((m1 & 0x87u) + exp_base_value) << 8u) | ((low0 >> 16u) & 0xFFu)) << 16u), sum_high);
                    sum_high = metal::fma(float(x[k_j + 1u]), as_type<float>(((((((m1 << 1u) & 0x80u) | ((m1 >> 3u) & 0x07u)) + exp_base_value) << 8u) | ((low0 >> 24u) & 0xFFu)) << 16u), sum_high);
                    sum_low = metal::fma(float(x[k_j + 8u]), as_type<float>(((((m2 & 0x87u) + exp_base_value) << 8u) | (low1 & 0xFFu)) << 16u), sum_low);
                    sum_low = metal::fma(float(x[k_j + 9u]), as_type<float>(((((((m2 << 1u) & 0x80u) | ((m2 >> 3u) & 0x07u)) + exp_base_value) << 8u) | ((low1 >> 8u) & 0xFFu)) << 16u), sum_low);
                    sum_high = metal::fma(float(x[k_j + 8u]), as_type<float>(((((m3 & 0x87u) + exp_base_value) << 8u) | ((low1 >> 16u) & 0xFFu)) << 16u), sum_high);
                    sum_high = metal::fma(float(x[k_j + 9u]), as_type<float>(((((((m3 << 1u) & 0x80u) | ((m3 >> 3u) & 0x07u)) + exp_base_value) << 8u) | ((low1 >> 24u) & 0xFFu)) << 16u), sum_high);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            reduce[(lane >> 2) * THREADS + physical] = sum_low;
            reduce[((lane >> 2) + 8u) * THREADS + physical] = sum_high;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (physical < 16u) {
                float total = 0.0f;
                for (uint index = 0u; index < THREADS; ++index) {
                    total += reduce[physical * THREADS + index];
                }
                uint column = column_block * 16u + physical;
                if (SPLITS == 1) {
                    output[column] = OUTPUT_TYPE(total + bias[column]);
                } else {
                    output[split * N + column] = OUTPUT_TYPE(total);
                }
            }
"""


@lru_cache(maxsize=3)
def _gemv16_kernel(output_dtype):
    """Single-token GEMV over 16 output columns per threadgroup.

    A super-atom is 16 rows by 64 K in 1536 contiguous bytes, so one
    threadgroup stages it with coalesced "'``uint32`` loads and then decodes one
    "'``(lane, j)`` group per thread: three shared reads expand to eight BF16
    values, each multiplied against one activation value. The two row sums a
    thread owns are reduced once at the end instead of once per super-atom, so
    the loop carries no per-step reduction.

    Reading the packed bytes in place instead issues a separate 1-byte load per
    meta byte, which costs a memory transaction for every byte it touches. The
    eight slots of a group are fully unrolled, so every shift, mask, and
    low-byte index is a compile-time constant and the escape test is hoisted
    out of the value loop.
    """

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
        name=f"gptqmodel_nf12_gemv16_{output_suffix}",
        input_names=["x", "packed", "escapes", "exp_base", "bias"],
        output_names=["output"],
        source=_GEMV16_SOURCE.replace("OUTPUT_TYPE", output_type),
    )


def _tile_template(layer, rows, padded, splits, threads, tile_columns):
    """Template constants for the fused tile kernel."""
    return [
        ("K", layer.in_features), ("N", layer.out_features),
        ("ROWS", rows), ("PADDED", padded),
        ("SA_ROW", layer.in_features // 64),
        # Words per 16-row block: 16 rows of 1.5 bytes per weight.
        ("SA_BYTES", layer.in_features * 6),
        ("SPLITS", splits), ("ROW_BLOCKS", padded // 16),
        ("THREADS", threads), ("TILE_COLUMNS", tile_columns),
        # The tile is 16 token rows, so the 8x8 grid has two row groups; extra
        # threads only join the decode and load loops.
        ("COLUMN_GROUPS", tile_columns // 8),
        ("SIMD_GROUPS", (tile_columns // 8) * 2),
    ]


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
        # A single row uses the GEMV: it reads the packed bytes once and needs
        # no second dispatch, which is the whole cost at that size.
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

    def _splits(self, rows):
        """Split K only when the 16-column tile grid would starve the GPU."""
        blocks = (self.out_features // 16) * ((rows + 15) // 16)
        super_atoms = self.in_features // 64
        splits = 1
        while (
            blocks * splits * 2 <= _TARGET_TILES
            and splits * 2 <= super_atoms
            and splits < _MAX_SPLITS
        ):
            splits *= 2
        return splits

    def _prefill(self, x, rows):
        """Fused decode-plus-MMA tile for 2..16 tokens."""
        tile_columns = 16
        threads = tile_columns * 8
        padded = ((rows + 15) // 16) * 16
        splits = self._splits(rows)
        # MLX takes the grid in threads: x spans one threadgroup, y and z index
        # the threadgroups.
        grid = (threads, self.out_features // tile_columns, splits * (padded // 16))
        kernel = _prefill_kernel(x.dtype)
        if splits == 1:
            return kernel(
                inputs=[x, self.packed, self.escapes, self.exp_base, self.bias],
                template=_tile_template(self, rows, padded, splits, threads, tile_columns),
                grid=grid,
                threadgroup=(threads, 1, 1),
                output_shapes=[(padded, self.out_features)],
                output_dtypes=[x.dtype],
            )[0][:rows]
        partial = _prefill_kernel(mx.float32)(
            inputs=[x, self.packed, self.escapes, self.exp_base, self.bias],
            template=_tile_template(self, rows, padded, splits, threads, tile_columns),
            grid=grid,
            threadgroup=(threads, 1, 1),
            output_shapes=[(splits, rows, self.out_features)],
            output_dtypes=[mx.float32],
        )[0]
        rows_n = rows * self.out_features
        return _reduce_kernel(x.dtype)(
            inputs=[partial, self.bias],
            template=[("N", self.out_features), ("SPLITS", splits), ("ROWS_N", rows_n)],
            grid=(rows_n, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, self.out_features)],
            output_dtypes=[x.dtype],
        )[0]

    def _gemv16(self, x):
        """Single-token path: 16 output columns per threadgroup.

        PHASES spreads K inside one threadgroup, so the GEMV needs no split-K
        pass and therefore no second dispatch.
        """
        threads = 128
        return _gemv16_kernel(x.dtype)(
            inputs=[x.reshape(self.in_features), self.packed, self.escapes,
                    self.exp_base, self.bias],
            template=[
                ("K", self.in_features), ("N", self.out_features),
                ("SA_ROW", self.in_features // 64),
                # Words per 16-row block: 16 rows of 1.5 bytes per weight.
                ("SA_BYTES", self.in_features * 6),
                ("SPLITS", 1), ("THREADS", threads), ("PHASES", 1),
            ],
            grid=(threads, self.out_features // 16, 1),
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
            return self._gemv16(x).reshape(output_shape)
        if rows <= _PREFILL_ROWS:
            # 2..16 tokens: the packed stream is read once, straight into the
            # matrix units, instead of round-tripping a dense BF16 weight.
            return self._prefill(x.reshape(rows, self.in_features), rows).reshape(output_shape)
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
