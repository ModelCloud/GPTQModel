# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# NF12 reference: AlpinDale and the nf12 contributors, Apache-2.0, https://github.com/dphnAI/nf12
"""Packed NF12 (12-bit BF16) inference kernels for MLX Metal.

NF12 keeps the checkpoint's tiled 12-byte group stream: four meta bytes hold the
sign and the 3-bit exponent offset, eight low bytes hold the mantissa tail, and
a group whose meta bytes are all zero stores its eight BF16 words in the escape
table. This module decodes that stream on Metal and never installs a dense
weight. Three paths cover the token counts that matter on Apple silicon:

* 1..15 tokens: the decode is fused into the dot product. One threadgroup owns
  16 output columns, reads each 1536-byte super-atom once with coalesced
  ``uint32`` loads, and keeps one accumulator per token row, so no dense BF16
  weight buffer is written and no per-super-atom barrier is paid,
* exactly 16 tokens on a wide output: a fused tile decodes one super-atom per K
  step into threadgroup memory and runs it through the simdgroup matrix units,
  which wins back the FMA cost the dot product pays at 16 rows,
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
# At exactly 16 tokens the tile's matrix units win back the FMA cost, but only
# once the output is wide enough for its 16-column grid to fill the GPU.
_TILE_MIN_OUT_FEATURES = 65536
# Split-K keeps the 16-column tile grid wide enough to fill the GPU; the
# reduce pass it adds costs one pass over the activation-sized output.
_TARGET_TILES = 1024
_MAX_SPLITS = 8


def _row_blocks(rows, out_features):
    """Adjacent 16-column blocks per threadgroup for the dot-product path.

    Every output column needs the same activation values, so decoding several
    blocks per threadgroup divides the per-step activation loads. The price is
    ``rows * blocks`` accumulator pairs per thread, so the wide token counts
    keep fewer blocks and stay out of a register spill, and a short output grid
    keeps one block so the GPU still has a full wave of threadgroups. The block
    count must divide the 16-column block count: the launch covers exactly
    ``out_features // (16 * blocks)`` groups, so a non-divisor would floor the
    grid and leave the trailing columns unwritten.
    """
    if rows > 12:
        return 1
    cap = 4 if rows <= 4 else 2
    # One wave is 32 threadgroups; a shorter grid costs more than the shared
    # activation loads save, so the block count shrinks with the output width.
    blocks = min(cap, out_features // 512)
    blocks = 1 if blocks < 2 else (2 if blocks < 4 else 4)
    column_blocks = out_features // 16
    while blocks > 1 and column_blocks % blocks:
        blocks //= 2
    return blocks


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
            for (uint position = physical; position < 16u * TILE_COLUMNS; position += THREADS) {
                uint r = position / TILE_COLUMNS;
                uint c = position % TILE_COLUMNS;
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


def _rows_body(rows, blocks):
    """Unrolled slot decode plus one FMA per token row, over ``blocks`` columns.

    The decode builds each BF16 word from its meta nibble and low byte with the
    fewest integer ops that fit: the exponent byte is shifted straight to the
    top of the word, the low byte is masked into place, and the loop prefetches
    the next super-atom so the load latency overlaps the current step's ALU.

    A threadgroup owns ``blocks`` adjacent 16-column blocks and decodes them
    from the same activation values, because ``x`` does not depend on the output
    column. That is what makes the wide token counts fast: at eight rows the
    per-step activation loads outnumber the packed loads, and sharing them
    across blocks removes most of that traffic without touching the weight
    stream.
    """

    def slots(escaped, pad, block):
        out = []
        for s in range(8):
            ch, rh, p = s >> 2, (s >> 1) & 1, s & 1
            low = f"{'low0' if s < 4 else 'low1'}_{block}"
            byte = s & 3
            k = f"k_j + {8 * ch + p}u"
            # (byte << 16) written as a masked shift of the source word.
            shift = {0: "<< 16u", 1: "<< 8u", 2: "", 3: ">> 8u"}[byte]
            if escaped:
                # The escape table is indexed by the group's first low word.
                bits = f"uint(uint16_t(escapes[low0_{block} * 8u + {s}u])) << 16u"
            else:
                meta = f"m{s >> 1}"
                high = (f"({meta} & 0x87u)" if p == 0
                        else f"((({meta} << 1u) & 0x80u) | (({meta} >> 3u) & 0x07u))")
                bits = (
                    f"((({high} + exp_base_value) << 24u)"
                    f" | (({low} {shift}) & 0x00FF0000u))"
                )
            out.append(f"{pad}weight = as_type<float>({bits});")
            for r in range(rows):
                out.append(
                    f"{pad}sum_{block}_{r}_{rh} = metal::fma("
                    f"float(x[{r} * K + {k}]), weight, sum_{block}_{r}_{rh});"
                )
        return "\n".join(out)

    decls = "\n".join(
        f"            float sum_{b}_{r}_{rh} = 0.0f;"
        for b in range(blocks) for r in range(rows) for rh in (0, 1)
    )
    pre = ""
    prefetch = ""
    advance = ""
    for b in range(blocks):
        off = "" if b == 0 else f" + {b}u * SA_BYTES"
        pre += f"""
            uint meta_{b} = base[meta_word{off}];
            uint low0_{b} = base[low_word{off}];
            uint low1_{b} = base[low_word + 1u{off}];"""
        prefetch += f"""
                uint next_meta_{b} = next[meta_word{off}];
                uint next_low0_{b} = next[low_word{off}];
                uint next_low1_{b} = next[low_word + 1u{off}];"""
        advance += f"""
                meta_{b} = next_meta_{b};
                low0_{b} = next_low0_{b};
                low1_{b} = next_low1_{b};"""
    decode = ""
    for b in range(blocks):
        decode += f"""
                {{
                    uint meta = meta_{b};
                    // A group whose four meta bytes are all zero carries its
                    // eight BF16 words in the escape table.
                    if (meta == 0u) {{
{slots(True, "                        ", b)}
                    }} else {{
                        uint m0 = meta & 0xFFu;
                        uint m1 = (meta >> 8u) & 0xFFu;
                        uint m2 = (meta >> 16u) & 0xFFu;
                        uint m3 = (meta >> 24u) & 0xFFu;
{slots(False, "                        ", b)}
                    }}
                }}"""
    loop = f"""
{decls}

            // Double-buffer the super-atom stream: the next step's words are in
            // flight while this step decodes and accumulates.
            const device uint *base = (const device uint *)packed;{pre}
            for (uint step = 0; step < SA_ROW; ++step) {{
                const device uint *next = base + min(step + 1u, SA_ROW - 1u) * 384u;{prefetch}
                uint k_j = step * 64u + 16u * j + 2u * (lane & 3u);{decode}{advance}
            }}
"""
    tail = ""
    for b in range(blocks):
        for r in range(rows):
            tail += f"""
            // A butterfly over the 16 threads that share a column halves the
            // tail four times, then the surviving lane writes both weight rows.
            float v0_{b}_{r} = sum_{b}_{r}_0;
            float v1_{b}_{r} = sum_{b}_{r}_1;
            v0_{b}_{r} += simd_shuffle_xor(v0_{b}_{r}, 8u);
            v0_{b}_{r} += simd_shuffle_xor(v0_{b}_{r}, 4u);
            v0_{b}_{r} += simd_shuffle_xor(v0_{b}_{r}, 2u);
            v0_{b}_{r} += simd_shuffle_xor(v0_{b}_{r}, 1u);
            v1_{b}_{r} += simd_shuffle_xor(v1_{b}_{r}, 8u);
            v1_{b}_{r} += simd_shuffle_xor(v1_{b}_{r}, 4u);
            v1_{b}_{r} += simd_shuffle_xor(v1_{b}_{r}, 2u);
            v1_{b}_{r} += simd_shuffle_xor(v1_{b}_{r}, 1u);
            if ((physical & 15u) == 0u) {{
                uint column = (column_block + {b}u) * 16u + (physical >> 4);
                output[{r} * N + column] = OUTPUT_TYPE(v0_{b}_{r} + bias[column]);
                output[{r} * N + column + 8u] = OUTPUT_TYPE(v1_{b}_{r} + bias[column + 8u]);
            }}
"""
    return loop + tail


_ROWS_SOURCE_HEAD = """            uint physical = thread_position_in_threadgroup.x;
            uint column_block = threadgroup_position_in_grid.y * BLOCKS;
            uint lane = physical >> 2;
            uint j = physical & 3u;
            float weight = 0.0f;
            uint exp_base_value = uint(exp_base[0]);
            // Consecutive threads read consecutive words, so a group's three
            // loads coalesce without staging the super-atom in shared memory.
            uint meta_word = column_block * SA_BYTES + lane * 4u + j;
            uint low_word = column_block * SA_BYTES + 128u + (j >> 1) * 128u + lane * 4u + (j & 1u) * 2u;
__BODY__"""


def _rows_source(rows, blocks):
    """Build the packed-decode GEMV body for one token count and block count.

    Each thread owns one ``(lane, j)`` group, which is three words that expand
    to eight BF16 values covering two weight rows. The eight slots are unrolled
    so every shift and mask is a compile-time constant, the escape test is
    hoisted out of the value loop, and one accumulator per token row keeps the
    packed read amortized across the whole block.
    """

    return _ROWS_SOURCE_HEAD.replace("__BODY__", _rows_body(rows, blocks))


@lru_cache(maxsize=64)
def _rows_kernel(rows, blocks, output_dtype):
    """Decode-plus-dot kernel for 1..16 tokens, ``16 * blocks`` columns per group."""

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
        name=f"gptqmodel_nf12_rows{rows}x{blocks}_{output_suffix}",
        input_names=["x", "packed", "escapes", "exp_base", "bias"],
        output_names=["output"],
        source=_rows_source(rows, blocks).replace("OUTPUT_TYPE", output_type),
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
        """Fused decode-plus-MMA tile for exactly 16 tokens on a wide output."""
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

    def _rows(self, x, rows):
        """Packed-decode dot product for 1..16 tokens.

        One threadgroup reads each 1536-byte super-atom exactly once and keeps
        one accumulator per token row, so a wider block costs FMAs but no extra
        weight traffic. It also owns several adjacent 16-column blocks, because
        the activation values are shared by every output column: decoding four
        blocks from one set of activation loads is what makes the wide token
        counts fast. That beats the MMA tile up to fifteen rows on this GPU,
        where the tile's per-super-atom barriers cost more than the matrix units
        save.
        """
        threads = 128
        blocks = _row_blocks(rows, self.out_features)
        return _rows_kernel(rows, blocks, x.dtype)(
            inputs=[x.reshape(rows, self.in_features), self.packed, self.escapes,
                    self.exp_base, self.bias],
            template=[
                ("K", self.in_features), ("N", self.out_features),
                ("ROWS", rows), ("SA_ROW", self.in_features // 64),
                # Words per 16-row block: 16 rows of 1.5 bytes per weight.
                ("SA_BYTES", self.in_features * 6),
                ("BLOCKS", blocks),
            ],
            grid=(threads, self.out_features // (16 * blocks), 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(rows, self.out_features)],
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
        if rows <= _PREFILL_ROWS and not (
            rows == _PREFILL_ROWS and self.out_features >= _TILE_MIN_OUT_FEATURES
        ):
            # One dispatch: the decode never leaves the dot product. At 1..15
            # tokens this beats the MMA tile at every width measured on this
            # GPU, because the tile pays a barrier per super-atom and the dot
            # product does not; only a full 16-row tile on a wide output has
            # enough matrix-unit work to win back that cost.
            return self._rows(x, rows).reshape(output_shape)
        if rows <= _PREFILL_ROWS:
            # A full 16-row tile on a wide output: the packed stream is read
            # once, straight into the matrix units, instead of round-tripping a
            # dense BF16 weight.
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
