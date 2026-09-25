# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# FP8 E4M3 layout: PyTorch contributors, BSD-3-Clause, https://github.com/pytorch/pytorch
# MXFP8 matmul: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""Exact E4M3 weight transfer to MLX's native MXFP8 matrix multiplication."""

import mlx.core as mx
import mlx.nn as nn


class MlxFP8Linear(nn.Module):
    """Apply checkpoint scales in FP32 and preserve the activation dtype."""

    def __init__(self, in_features, out_features, output_scale, bias=None):
        super().__init__()
        self.linear = nn.QuantizedLinear(
            in_features, out_features, bias=False, group_size=32, bits=8, mode="mxfp8",
        )
        self.output_scale = mx.array(output_scale)
        if bias is not None:
            self.bias = mx.array(bias)
        self.freeze()

    def _unscaled_dot(self, x):
        # The unscaled E4M3 dot product can exceed FP16 even when the final
        # checkpoint output is finite.
        return self.linear(x.astype(mx.float32))

    def __call__(self, x):
        result = self._unscaled_dot(x) * self.output_scale
        if "bias" in self:
            result = result + self.bias
        return result.astype(x.dtype)


class MlxFP8DenseLinear(nn.Module):
    """Keep the activation dtype for FP8 formats decoded to FP16 weights."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.freeze()

    def __call__(self, x):
        return self.linear(x).astype(x.dtype)
