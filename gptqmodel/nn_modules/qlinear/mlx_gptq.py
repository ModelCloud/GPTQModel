# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPTQ format and packing: ModelCloud.ai, Apache-2.0, https://github.com/ModelCloud/GPTQModel
"""GPTQ MLX inference kernels that preserve FP16/BF16 activation dtype."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from ...quantization.mlx_splitk_dispatch import quantized_matmul as _dispatch_matmul
from .mlx_group16 import MlxGroup16Linear


def _uses_packed_prefill(output_dtype, rows, input_dims, output_dims):
    return (
        output_dtype == mx.bfloat16
        and input_dims >= 5120
        and output_dims <= 1024
        and (1 < rows <= 8 or rows in (12, 16))
    )


# Decode columns per warp for wide projections.  A warp owns `COLUMNS` columns so
# each activation load feeds several fused multiply-adds, which removes the
# per-column activation reload that dominates the one-column-per-warp layout.
_PACKED_COLUMNS = 4
_PACKED_COLUMN_WARPS = 8
_PACKED_COLUMNS_MIN_OUTPUT = 5120


def _group16_columns_source(output_type, word_wide):
    """Build the wide-decode MSL body shared by the GPTQ and AWQ kernels.

    `word_wide` is true only when a code can straddle a 32-bit packed word,
    which happens exactly when `bits` does not divide 32 (3, 5 and 6 bits).
    Every other width extracts the code from a single word.
    """
    if word_wide:
        extract = """
                ulong window = ulong(packed[c][word]);
                if (word + 1 < WORDS) window |= ulong(packed[c][word + 1]) << 32;
                uint code = uint((window >> shift) & MASK);"""
    else:
        extract = """
                uint code = (packed[c][word] >> shift) & MASK;"""
    return """
            uint lane = thread_position_in_threadgroup.x & 31u;
            uint warp = thread_position_in_threadgroup.x >> 5;
            uint column_base =
                (threadgroup_position_in_grid.y * WARPS + warp) * COLUMNS;
            float sums[COLUMNS];
            uint offsets[COLUMNS];
            for (uint c = 0; c < COLUMNS; ++c) {
                sums[c] = 0.0f;
                uint column = column_base + c;
                offsets[c] = column < N ? column * (K * BITS / 32) : 0u;
            }

            for (uint group = lane; group < K / 16; group += 32) {
                uint start_bit = group * 16 * BITS;
                uint word_base = start_bit >> 5;
                uint first_shift = start_bit & 31u;
                uint packed[COLUMNS][4];
                float group_scales[COLUMNS];
                float group_biases[COLUMNS];
                for (uint c = 0; c < COLUMNS; ++c) {
                    uint column = column_base + c;
                    for (uint word = 0; word < WORDS; ++word)
                        packed[c][word] = column < N
                            ? weight[offsets[c] + word_base + word] : 0u;
                    uint scale_index = column < N
                        ? column * (K / 32) + (group >> 1) : 0u;
                    group_scales[c] = (group & 1u)
                        ? scales_odd[scale_index] : scales_even[scale_index];
                    group_biases[c] = (group & 1u)
                        ? biases_odd[scale_index] : biases_even[scale_index];
                }
                for (uint index = 0; index < 16; ++index) {
                    float value = float(x[group * 16 + index]);
                    uint bit = first_shift + index * BITS;
                    uint word = bit >> 5;
                    uint shift = bit & 31u;
                    for (uint c = 0; c < COLUMNS; ++c) {EXTRACT
                        sums[c] = metal::fma(
                            value,
                            metal::fma(float(code), group_scales[c], group_biases[c]),
                            sums[c]);
                    }
                }
            }

            for (uint c = 0; c < COLUMNS; ++c) {
                uint column = column_base + c;
                float reduced = simd_sum(sums[c]);
                if (lane == 0 && column < N)
                    output[column] = OUTPUT_TYPE(reduced + bias[column]);
            }
        """.replace("EXTRACT", extract).replace("OUTPUT_TYPE", output_type)


@lru_cache(maxsize=12)
def _group16_columns_kernel(name, output_dtype, word_wide):
    output_types = {
        mx.float16: ("fp16", "half"),
        mx.bfloat16: ("bf16", "bfloat16_t"),
        mx.float32: ("fp32", "float"),
    }
    try:
        output_suffix, output_type = output_types[output_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported group-16 activation dtype: {output_dtype}") from exc
    return mx.fast.metal_kernel(
        name=f"gptqmodel_{name}_group16_columns_{output_suffix}",
        input_names=[
            "x", "weight", "scales_even", "scales_odd",
            "biases_even", "biases_odd", "bias",
        ],
        output_names=["output"],
        source=_group16_columns_source(output_type, word_wide),
    )


def _uses_packed_columns(output_dims):
    return output_dims >= _PACKED_COLUMNS_MIN_OUTPUT


def _packed_columns_launch(kernel, inputs, input_dims, output_dims, bits, dtype):
    """Launch the wide-decode kernel and return its output tensor."""
    return kernel(
        inputs=inputs,
        template=[
            ("K", input_dims), ("N", output_dims), ("BITS", bits),
            ("MASK", (1 << bits) - 1), ("WORDS", (bits + 1) // 2),
            ("WARPS", _PACKED_COLUMN_WARPS), ("COLUMNS", _PACKED_COLUMNS),
        ],
        grid=(
            32 * _PACKED_COLUMN_WARPS,
            (output_dims + _PACKED_COLUMN_WARPS * _PACKED_COLUMNS - 1)
            // (_PACKED_COLUMN_WARPS * _PACKED_COLUMNS),
            1,
        ),
        threadgroup=(32 * _PACKED_COLUMN_WARPS, 1, 1),
        output_shapes=[(1, output_dims)], output_dtypes=[dtype],
    )[0]


@lru_cache(maxsize=6)
def _gptq_group16_kernel(output_dtype, tiled=False):
    """Multiply packed GPTQ codes with exact 16-value affine groups."""
    output_types = {
        mx.float16: ("fp16", "half"),
        mx.bfloat16: ("bf16", "bfloat16_t"),
        mx.float32: ("fp32", "float"),
    }
    try:
        output_suffix, output_type = output_types[output_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported GPTQ group-16 activation dtype: {output_dtype}") from exc

    if tiled:
        output_suffix += "_prefill"
        row_setup = """
            uint row_base = threadgroup_position_in_grid.z * RTILE;
            threadgroup float partials[RTILE * GROUPS];
            float sums[RTILE];
            for (uint r = 0; r < RTILE; ++r) sums[r] = 0.0f;
        """
        accumulate = """
            uint k = group * 16 + index;
            for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r)
                sums[r] = metal::fma(float(x[(row_base + r) * K + k]), value, sums[r]);
        """
        reduction = """
            for (uint r = 0; r < RTILE; ++r) {
                float reduced = simd_sum(sums[r]);
                if ((lane & 31u) == 0)
                    partials[r * GROUPS + (lane >> 5)] = reduced;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r) {
                float reduced = lane < GROUPS ? partials[r * GROUPS + lane] : 0.0f;
                reduced = simd_sum(reduced);
                if (lane == 0)
                    output[(row_base + r) * N + column] = OUTPUT_TYPE(reduced + bias[column]);
            }
        """
    else:
        row_setup = """
            threadgroup float partials[8];
            float sum = 0.0f;
        """
        accumulate = """
            sum = metal::fma(float(x[group * 16 + index]), value, sum);
        """
        reduction = """
            float reduced = simd_sum(sum);
            if ((lane & 31u) == 0)
                partials[lane >> 5] = reduced;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            reduced = lane < GROUPS ? partials[lane] : 0.0f;
            reduced = simd_sum(reduced);
            if (lane == 0)
                output[column] = OUTPUT_TYPE(reduced + bias[column]);
        """

    return mx.fast.metal_kernel(
        name=f"gptqmodel_gptq_group16_matmul_{output_suffix}",
        input_names=[
            "x", "weight", "scales_even", "scales_odd",
            "biases_even", "biases_odd", "bias",
        ],
        output_names=["output"],
        source="""
            uint lane = thread_position_in_threadgroup.x;
            uint column = threadgroup_position_in_grid.y;
            ROW_SETUP
            uint weight_offset = column * (K * BITS / 32);

            for (uint group = lane; group < K / 16; group += THREADS) {
                uint start_bit = group * 16 * BITS;
                uint word_base = weight_offset + (start_bit >> 5);
                uint first_shift = start_bit & 31u;
                uint packed[4];
                for (uint word = 0; word < WORDS; ++word)
                    packed[word] = weight[word_base + word];

                uint scale_index = column * (K / 32) + (group >> 1);
                float scale = (group & 1u)
                    ? scales_odd[scale_index] : scales_even[scale_index];
                float offset = (group & 1u)
                    ? biases_odd[scale_index] : biases_even[scale_index];
                for (uint index = 0; index < 16; ++index) {
                    uint bit = first_shift + index * BITS;
                    uint word = bit >> 5;
                    uint shift = bit & 31u;
                    ulong window = ulong(packed[word]);
                    if (word + 1 < WORDS)
                        window |= ulong(packed[word + 1]) << 32;
                    uint code = uint((window >> shift) & MASK);
                    float value = metal::fma(float(code), scale, offset);
                    ACCUMULATE
                }
            }

            REDUCTION
        """.replace("ROW_SETUP", row_setup)
        .replace("ACCUMULATE", accumulate)
        .replace("REDUCTION", reduction)
        .replace("OUTPUT_TYPE", output_type),
    )


class MlxGPTQGroup16Linear(MlxGroup16Linear):
    """Execute GPTQ group-16 decode in one packed Metal matrix product."""

    def __init__(self, input_dims, output_dims, bits, bias=False):
        super().__init__(input_dims, output_dims, bits=bits, bias=bias)
        if not bias:
            self.zero_bias = (mx.zeros((output_dims,), dtype=mx.float32),)

    def _packed_prefill(self, x, rows, row_tile, output_dtype=None):
        output_dims = self.weight.shape[0]
        output_dtype = x.dtype if output_dtype is None else output_dtype
        bias = self.bias if "bias" in self else self.zero_bias[0]
        return _gptq_group16_kernel(output_dtype, tiled=True)(
            inputs=[
                x, self.weight, self.scales_even, self.scales_odd,
                self.biases_even, self.biases_odd, bias,
            ],
            template=[
                ("K", self.input_dims), ("N", output_dims),
                ("BITS", self.bits), ("MASK", (1 << self.bits) - 1),
                ("WORDS", (self.bits + 1) // 2), ("ROWS", rows),
                ("RTILE", row_tile), ("THREADS", 32), ("GROUPS", 1),
            ],
            grid=(32, output_dims, (rows + row_tile - 1) // row_tile),
            threadgroup=(32, 1, 1),
            output_shapes=[(rows, output_dims)], output_dtypes=[output_dtype],
        )[0]

    def __call__(self, x):
        if x.shape[-1] != self.input_dims:
            raise ValueError(f"expected input width {self.input_dims}, got {x.shape[-1]}")
        output_dims = self.weight.shape[0]
        output_shape = (*x.shape[:-1], output_dims)
        if x.size == 0:
            return mx.zeros(output_shape, dtype=x.dtype)
        rows = x.size // self.input_dims
        if rows != 1:
            if not _uses_packed_prefill(
                x.dtype, rows, self.input_dims, output_dims,
            ):
                return super().__call__(x)
            output = self._packed_prefill(x, rows, min(rows, 4))
            return output.reshape(output_shape)
        bias = self.bias if "bias" in self else self.zero_bias[0]
        if _uses_packed_columns(output_dims):
            output = _packed_columns_launch(
                _group16_columns_kernel(
                    "gptq", x.dtype, 32 % self.bits != 0,
                ),
                [
                    x, self.weight, self.scales_even, self.scales_odd,
                    self.biases_even, self.biases_odd, bias,
                ],
                self.input_dims, output_dims, self.bits, x.dtype,
            )
            return output.reshape(output_shape)
        threads = 128 if self.input_dims >= 8192 or output_dims >= 16384 else 64
        output = _gptq_group16_kernel(x.dtype)(
            inputs=[
                x, self.weight, self.scales_even, self.scales_odd,
                self.biases_even, self.biases_odd, bias,
            ],
            template=[
                ("K", self.input_dims), ("BITS", self.bits),
                ("MASK", (1 << self.bits) - 1), ("WORDS", (self.bits + 1) // 2),
                ("THREADS", threads), ("GROUPS", threads // 32),
            ],
            grid=(threads, output_dims, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(1, output_dims)], output_dtypes=[x.dtype],
        )[0]
        return output.reshape(output_shape)


def _mlx_forward(linear, x):
    """Apply the wrapped quantized affine layer, avoiding MLX's split-K pass."""
    out = _dispatch_matmul(
        x,
        linear.weight,
        linear.scales,
        linear.get("biases"),
        group_size=linear.group_size,
        bits=linear.bits,
        mode=linear.mode,
    )
    if "bias" in linear:
        out = out + linear["bias"]
    return out.astype(x.dtype)


class MlxGPTQLinear(nn.Module):
    """Run MLX affine GPTQ matmul and round its output to the input dtype."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.freeze()

    def __call__(self, x):
        return _mlx_forward(self.linear, x)
