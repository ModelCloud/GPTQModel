# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# AWQ format reference: ModelCloud.ai, Apache-2.0, GPT-QModel AWQ packers.
"""AWQ MLX inference kernels that preserve FP16/BF16 activation dtype."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .mlx_group16 import MlxGroup16Linear


@lru_cache(maxsize=6)
def _awq_group16_kernel(output_dtype, tiled=False):
    """Multiply packed AWQ 4-bit codes with exact 16-value affine groups."""
    output_types = {
        mx.float16: ("fp16", "half"),
        mx.bfloat16: ("bf16", "bfloat16_t"),
        mx.float32: ("fp32", "float"),
    }
    try:
        output_suffix, output_type = output_types[output_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported AWQ group-16 activation dtype: {output_dtype}") from exc

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
        name=f"gptqmodel_awq_group16_matmul_{output_suffix}",
        input_names=[
            "x", "weight", "scales_even", "scales_odd",
            "biases_even", "biases_odd", "bias",
        ],
        output_names=["output"],
        source="""
            uint lane = thread_position_in_threadgroup.x;
            uint column = threadgroup_position_in_grid.y;
            ROW_SETUP
            uint weight_offset = column * (K / 8);

            for (uint group = lane; group < K / 16; group += THREADS) {
                uint word_base = weight_offset + group * 2;
                ulong packed = ulong(weight[word_base]) | (ulong(weight[word_base + 1]) << 32);
                uint scale_index = column * (K / 32) + (group >> 1);
                float scale = (group & 1u)
                    ? scales_odd[scale_index] : scales_even[scale_index];
                float offset = (group & 1u)
                    ? biases_odd[scale_index] : biases_even[scale_index];
                for (uint index = 0; index < 16; ++index) {
                    uint code = uint((packed >> (index * 4)) & 15u);
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


class MlxAWQGroup16Linear(MlxGroup16Linear):
    """Execute AWQ group-16 decode in one packed Metal matrix product."""

    def __init__(self, input_dims, output_dims, bias=False):
        super().__init__(input_dims, output_dims, bits=4, bias=bias)
        if not bias:
            self.zero_bias = (mx.zeros((output_dims,), dtype=mx.float32),)

    def _packed_prefill(self, x, rows, row_tile, output_dtype=None):
        output_dims = self.weight.shape[0]
        output_dtype = x.dtype if output_dtype is None else output_dtype
        bias = self.bias if "bias" in self else self.zero_bias[0]
        return _awq_group16_kernel(output_dtype, tiled=True)(
            inputs=[
                x, self.weight, self.scales_even, self.scales_odd,
                self.biases_even, self.biases_odd, bias,
            ],
            template=[
                ("K", self.input_dims), ("N", output_dims), ("ROWS", rows),
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
        # The packed row tile wins for small-output BF16 prefill only.
        prefill = (
            x.dtype == mx.bfloat16
            and 1 < rows <= 16
            and output_dims <= 2048
        )
        if rows != 1:
            if not prefill:
                return super().__call__(x)
            output = self._packed_prefill(x, rows, min(rows, 4))
            return output.reshape(output_shape)
        threads = 128 if self.input_dims >= 8192 or output_dims >= 16384 else 64
        bias = self.bias if "bias" in self else self.zero_bias[0]
        output = _awq_group16_kernel(x.dtype)(
            inputs=[
                x, self.weight, self.scales_even, self.scales_odd,
                self.biases_even, self.biases_odd, bias,
            ],
            template=[
                ("K", self.input_dims), ("THREADS", threads),
                ("GROUPS", threads // 32),
            ],
            grid=(threads, output_dims, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(1, output_dims)], output_dtypes=[x.dtype],
        )[0]
        return output.reshape(output_shape)


class MlxAWQLinear(nn.Module):
    """Run MLX affine AWQ matmul and round its output to the input dtype."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.freeze()

    def __call__(self, x):
        return self.linear(x).astype(x.dtype)
