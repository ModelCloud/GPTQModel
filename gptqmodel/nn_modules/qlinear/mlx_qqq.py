# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# QQQ arithmetic reference: https://github.com/vllm-project/vllm
# MLX quantized matmul: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""MLX compressed INT8 matmul with QQQ dynamic activation quantization."""

import mlx.core as mx
import mlx.nn as nn


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
        x = x.astype(mx.float16)
        input_scale = (mx.max(mx.abs(x), axis=-1, keepdims=True) / 127).astype(mx.float32)
        # Zero rows follow the Torch reference: division by zero yields NaN.
        quantized = mx.clip(mx.round(x.astype(mx.float32) / input_scale), -128, 127)
        output = self.linear(quantized) * input_scale * self.channel_scale
        output = output.astype(mx.float16)
        if "bias" in self:
            output = output + self.bias
        return output.astype(original_dtype)
