# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# MLX quantized matmul: Apple Inc., MIT, https://github.com/ml-explore/mlx
# GPTQ format and packing: ModelCloud.ai, Apache-2.0, https://github.com/ModelCloud/GPTQModel
"""GPTQ MLX inference kernels that preserve FP16/BF16 activation dtype."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .mlx_group16 import MlxGroup16Linear


@lru_cache(maxsize=1)
def _gptq_group16_kernel():
    """Multiply packed GPTQ codes with exact 16-value affine groups."""
    return mx.fast.metal_kernel(
        name="gptqmodel_gptq_group16_matmul",
        input_names=[
            "x", "weight", "scales_even", "scales_odd",
            "biases_even", "biases_odd", "bias",
        ],
        output_names=["output"],
        source="""
            uint lane = thread_position_in_threadgroup.x;
            uint column = threadgroup_position_in_grid.y;
            threadgroup float partials[8];
            float sum = 0.0f;
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
                    sum = metal::fma(float(x[group * 16 + index]), value, sum);
                }
            }

            float reduced = simd_sum(sum);
            if ((lane & 31u) == 0)
                partials[lane >> 5] = reduced;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            reduced = lane < GROUPS ? partials[lane] : 0.0f;
            reduced = simd_sum(reduced);
            if (lane == 0)
                output[column] = reduced + bias[column];
        """,
    )


class MlxGPTQGroup16Linear(MlxGroup16Linear):
    """Execute GPTQ group-16 decode in one packed Metal matrix product."""

    def __init__(self, input_dims, output_dims, bits, bias=False):
        super().__init__(input_dims, output_dims, bits=bits, bias=bias)
        if not bias:
            self.zero_bias = (mx.zeros((output_dims,), dtype=mx.float32),)

    def __call__(self, x):
        if x.shape[-1] != self.input_dims:
            raise ValueError(f"expected input width {self.input_dims}, got {x.shape[-1]}")
        output_dims = self.weight.shape[0]
        output_shape = (*x.shape[:-1], output_dims)
        if x.size == 0:
            return mx.zeros(output_shape, dtype=x.dtype)
        rows = x.size // self.input_dims
        if rows != 1:
            return super().__call__(x)
        threads = 128 if self.input_dims >= 8192 or output_dims >= 16384 else 64
        bias = self.bias if "bias" in self else self.zero_bias[0]
        output = _gptq_group16_kernel()(
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
            output_shapes=[(1, output_dims)], output_dtypes=[mx.float32],
        )[0]
        return output.reshape(output_shape).astype(x.dtype)


class MlxGPTQLinear(nn.Module):
    """Run MLX affine GPTQ matmul and round its output to the input dtype."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.freeze()

    def __call__(self, x):
        return self.linear(x).astype(x.dtype)
