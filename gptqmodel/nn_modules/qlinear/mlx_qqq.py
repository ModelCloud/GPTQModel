# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# QQQ arithmetic reference: https://github.com/vllm-project/vllm
# MLX quantized matmul: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""MLX packed 4/8-bit QQQ matmul with fused dynamic activation quantization."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


@lru_cache(maxsize=1)
def _dynamic_quant_kernel():
    """Reduce and quantize each token in one Metal dispatch."""
    return mx.fast.metal_kernel(
        name="gptqmodel_qqq_dynamic_quant",
        input_names=["x"],
        output_names=["quantized", "scales"],
        source="""
            uint row = threadgroup_position_in_grid.y;
            uint lane = thread_position_in_threadgroup.x;
            threadgroup float maxima[8];
            float local_max = 0.0f;
            for (uint col = lane; col < K; col += 256) {
                local_max = metal::max(local_max, metal::abs(float(x[row * K + col])));
            }
            float simd_maximum = simd_max(local_max);
            if ((lane & 31) == 0) {
                maxima[lane / 32] = simd_maximum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float group_maximum = lane < 8 ? maxima[lane] : 0.0f;
            group_maximum = simd_max(group_maximum);
            if (lane == 0) {
                maxima[0] = group_maximum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // The source QQQ runtime divides a half-precision maximum by 127
            // before promoting the resulting scale to float32.
            float scale = float(half(maxima[0] / 127.0f));
            if (lane == 0) {
                scales[row] = scale;
            }
            for (uint col = lane; col < K; col += 256) {
                float value = scale == 0.0f ? 0.0f :
                    metal::rint(float(x[row * K + col]) / scale);
                quantized[row * K + col] = metal::clamp(value, -128.0f, 127.0f);
            }
        """,
    )


def _dynamic_quant(x):
    width = x.shape[-1]
    rows = x.size // width
    return _dynamic_quant_kernel()(
        inputs=[x], template=[("K", width)],
        grid=(256, rows, 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (*x.shape[:-1], 1)],
        output_dtypes=[mx.float32, mx.float32],
    )


class MlxQQQLinear(nn.Module):
    """Preserve QQQ's per-token input scale and per-channel output scale."""

    def __init__(self, linear, channel_scale, bias=None):
        super().__init__()
        self.linear = linear
        self.channel_scale = mx.array(channel_scale)
        if bias is not None:
            self.bias = mx.array(bias)
        self.freeze()

    def __call__(self, x):
        original_dtype = x.dtype
        if x.size == 0:
            return mx.zeros((*x.shape[:-1], self.linear.weight.shape[0]), dtype=original_dtype)
        x = x.astype(mx.float16)
        quantized, input_scale = _dynamic_quant(x)
        output = self.linear(quantized) * input_scale * self.channel_scale
        output = output.astype(mx.float16)
        if "bias" in self:
            output = output + self.bias
        return output.astype(original_dtype)
