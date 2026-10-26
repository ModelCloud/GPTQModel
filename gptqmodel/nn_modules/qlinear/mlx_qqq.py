# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# QQQ arithmetic reference: vLLM contributors, Apache-2.0, https://github.com/vllm-project/vllm
"""MLX packed 4/8-bit QQQ matmul with fused dynamic activation quantization."""

from functools import lru_cache

import mlx.core as mx
from mlx import nn


@lru_cache(maxsize=2)
def _dynamic_quant_kernel(threads):
    """Reduce and quantize each token in one Metal dispatch."""
    groups = threads // 32
    return mx.fast.metal_kernel(
        name=(
            "gptqmodel_qqq_dynamic_quant"
            if threads == 256 else f"gptqmodel_qqq_dynamic_quant_{threads}"
        ),
        input_names=["x"],
        output_names=["quantized", "scales"],
        source="""
            uint row = threadgroup_position_in_grid.y;
            uint lane = thread_position_in_threadgroup.x;
            threadgroup float maxima[MAX_GROUPS];
            float local_max = 0.0f;
            for (uint col = lane; col < K; col += THREAD_COUNT) {
                local_max = metal::max(local_max, metal::abs(float(x[row * K + col])));
            }
            float simd_maximum = simd_max(local_max);
            if ((lane & 31) == 0) {
                maxima[lane / 32] = simd_maximum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float group_maximum = lane < GROUP_COUNT ? maxima[lane] : 0.0f;
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
            for (uint col = lane; col < K; col += THREAD_COUNT) {
                float value = scale == 0.0f ? 0.0f :
                    metal::rint(float(x[row * K + col]) / scale);
                quantized[row * K + col] = metal::clamp(value, -128.0f, 127.0f);
            }
        """.replace("MAX_GROUPS", str(groups))
        .replace("THREAD_COUNT", str(threads))
        .replace("GROUP_COUNT", str(groups)),
    )


def _dynamic_quant(x):
    width = x.shape[-1]
    rows = x.size // width
    threads = 512 if width == 17408 else 256
    return _dynamic_quant_kernel(threads)(
        inputs=[x],
        template=[("K", width)],
        grid=(threads, rows, 1), threadgroup=(threads, 1, 1),
        output_shapes=[x.shape, (*x.shape[:-1], 1)],
        output_dtypes=[mx.float32, mx.float32],
    )


@lru_cache(maxsize=3)
def _finalize_kernel(output_dtype):
    """Apply QQQ output scales, FP16 rounding, bias, and dtype storage."""
    output_types = {
        mx.float16: ("fp16", "half"),
        mx.bfloat16: ("bf16", "bfloat16_t"),
        mx.float32: ("fp32", "float"),
    }
    try:
        output_suffix, output_type = output_types[output_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported QQQ activation dtype: {output_dtype}") from exc
    return mx.fast.metal_kernel(
        name=f"gptqmodel_qqq_finalize_{output_suffix}",
        input_names=["raw", "input_scale", "channel_scale", "bias"],
        output_names=["output"],
        source="""
            uint index = thread_position_in_grid.x;
            uint column = index % N;
            uint row = index / N;
            half value = half(
                float(raw[index]) * input_scale[row] * channel_scale[column]
            );
            if (HAS_BIAS) value = half(value + half(bias[column]));
            output[index] = OUTPUT_TYPE(value);
        """.replace("OUTPUT_TYPE", output_type),
    )


class MlxQQQLinear(nn.Module):
    """Preserve QQQ's per-token input scale and per-channel output scale."""

    def __init__(self, linear, channel_scale, bias=None):
        super().__init__()
        self.linear = linear
        self.channel_scale = mx.array(channel_scale)
        if bias is not None:
            self.bias = mx.array(bias)
        else:
            self.zero_bias = (mx.zeros((linear.weight.shape[0],), dtype=mx.float16),)
        self.freeze()

    def __call__(self, x):
        original_dtype = x.dtype
        if x.size == 0:
            return mx.zeros((*x.shape[:-1], self.linear.weight.shape[0]), dtype=original_dtype)
        x = x.astype(mx.float16)
        quantized, input_scale = _dynamic_quant(x)
        raw = self.linear(quantized)
        rows = x.size // x.shape[-1]
        if rows == 1:
            output_dims = self.linear.weight.shape[0]
            return _finalize_kernel(original_dtype)(
                inputs=[
                    raw,
                    input_scale,
                    self.channel_scale,
                    self.bias if "bias" in self else self.zero_bias[0],
                ],
                template=[("N", output_dims), ("HAS_BIAS", "bias" in self)],
                grid=(rows * output_dims, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(*x.shape[:-1], output_dims)],
                output_dtypes=[original_dtype],
            )[0]
        output = raw * input_scale * self.channel_scale
        output = output.astype(mx.float16)
        if "bias" in self:
            output = output + self.bias
        return output.astype(original_dtype)
