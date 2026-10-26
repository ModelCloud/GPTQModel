# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
"""GGUF MLX inference wrappers that preserve FP16/BF16 activation dtype."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .mlx_group16 import MlxGroup16Linear


@lru_cache(maxsize=3)
def _q6_k_kernel(output_dtype):
    """Multiply packed Q6_K codes with their exact 16-value affine groups."""
    output_types = {
        mx.float16: ("fp16", "half"),
        mx.bfloat16: ("bf16", "bfloat16_t"),
        mx.float32: ("fp32", "float"),
    }
    try:
        output_suffix, output_type = output_types[output_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported GGUF Q6_K activation dtype: {output_dtype}") from exc

    return mx.fast.metal_kernel(
        name=f"gptqmodel_gguf_q6_k_matmul_{output_suffix}",
        input_names=[
            "x", "weight", "scales_even", "scales_odd",
            "biases_even", "biases_odd", "bias",
        ],
        output_names=["output"],
        source="""
            uint lane = thread_position_in_threadgroup.x;
            uint column = threadgroup_position_in_grid.y;
            uint row_base = threadgroup_position_in_grid.z * RTILE;
            threadgroup float partials[RTILE * GROUPS];
            float sums[RTILE];
            for (uint r = 0; r < RTILE; ++r) sums[r] = 0.0f;

            uint weight_offset = column * (K * 6 / 32);
            for (uint group = lane; group < K / 16; group += THREADS) {
                uint word_index = weight_offset + group * 3;
                ulong lower = ulong(weight[word_index]) | (ulong(weight[word_index + 1]) << 32);
                ulong upper = ulong(weight[word_index + 1]) | (ulong(weight[word_index + 2]) << 32);
                uint scale_index = column * (K / 32) + (group >> 1);
                float scale = (group & 1) ? scales_odd[scale_index] : scales_even[scale_index];
                float offset = (group & 1) ? biases_odd[scale_index] : biases_even[scale_index];
                for (uint index = 0; index < 16; ++index) {
                    uint bit = index * 6;
                    uint code = uint((bit < 32 ? lower >> bit : upper >> (bit - 32)) & 63);
                    float value = metal::fma(float(code), scale, offset);
                    uint k = group * 16 + index;
                    for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r)
                        sums[r] = metal::fma(float(x[(row_base + r) * K + k]), value, sums[r]);
                }
            }
            for (uint r = 0; r < RTILE; ++r) {
                float reduced = simd_sum(sums[r]);
                if ((lane & 31) == 0) partials[r * GROUPS + (lane >> 5)] = reduced;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r) {
                float reduced = lane < GROUPS ? partials[r * GROUPS + lane] : 0.0f;
                reduced = simd_sum(reduced);
                if (lane == 0)
                    output[(row_base + r) * N + column] = OUTPUT_TYPE(reduced + bias[column]);
            }
        """.replace("OUTPUT_TYPE", output_type),
    )


class MlxGGUFQ6KLinear(MlxGroup16Linear):
    """Execute GGUF Q6_K's packed 6-bit matrix product in one Metal pass."""

    def __init__(self, input_dims, output_dims, bias=False):
        super().__init__(input_dims, output_dims, bits=6, bias=bias)
        if not bias:
            self.zero_bias = (mx.zeros((output_dims,), dtype=mx.float32),)

    def _packed_matmul(self, x, rows, row_tile, threads, output_dtype=None):
        output_dims = self.weight.shape[0]
        output_dtype = x.dtype if output_dtype is None else output_dtype
        bias = self.bias if "bias" in self else self.zero_bias[0]
        return _q6_k_kernel(output_dtype)(
            inputs=[
                x, self.weight, self.scales_even, self.scales_odd,
                self.biases_even, self.biases_odd, bias,
            ],
            template=[
                ("K", self.input_dims), ("N", output_dims), ("ROWS", rows),
                ("RTILE", row_tile), ("THREADS", threads), ("GROUPS", threads // 32),
            ],
            grid=(threads, output_dims, (rows + row_tile - 1) // row_tile),
            threadgroup=(threads, 1, 1),
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
        small_prefill = (
            x.dtype == mx.bfloat16
            and 1 < rows <= 16
            and output_dims <= 2048
        )
        if rows != 1 and not small_prefill:
            return super().__call__(x)
        if small_prefill:
            row_tile = min(rows, 4)
            threads = 32
        elif self.input_dims >= 8192 or output_dims <= 2048:
            row_tile = 1
            threads = 128
        elif output_dims >= 16384:
            row_tile = 1
            threads = 32
        else:
            row_tile = 1
            threads = 64
        output = self._packed_matmul(x, rows, row_tile, threads)
        return output.reshape(output_shape)


class MlxGGUFLinear(nn.Module):
    """Run MLX quantized GGUF matmul and cast its output to the input dtype."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.freeze()

    def __call__(self, x):
        return self.linear(x).astype(x.dtype)
