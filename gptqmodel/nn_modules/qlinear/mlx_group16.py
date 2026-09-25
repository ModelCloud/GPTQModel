# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# MLX quantized matmul interface: Apple Inc., MIT, mlx.core.quantized_matmul.
"""Exact group-16 GPTQ/AWQ inference using two MLX affine matmuls."""

import mlx.core as mx
import mlx.nn as nn


class MlxGroup16Linear(nn.Module):
    """Apply separate group-16 scales to alternating halves of MLX groups."""

    def __init__(self, input_dims, output_dims, bits, bias=False):
        super().__init__()
        if input_dims % 32 or bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError("Group-16 MLX inference needs 32-aligned inputs and affine-supported bits")
        self.input_dims = input_dims
        self.bits = bits
        self.weight = mx.zeros((output_dims, input_dims * bits // 32), dtype=mx.uint32)
        for name in ("scales_even", "scales_odd", "biases_even", "biases_odd"):
            setattr(self, name, mx.zeros((output_dims, input_dims // 32), dtype=mx.float32))
        if bias:
            self.bias = mx.zeros((output_dims,), dtype=mx.float16)
        self.freeze()

    def __call__(self, x):
        first_half = mx.arange(self.input_dims) % 32 < 16
        even = mx.quantized_matmul(
            mx.where(first_half, x, 0), self.weight, self.scales_even, self.biases_even,
            group_size=32, bits=self.bits,
        )
        odd = mx.quantized_matmul(
            mx.where(first_half, 0, x), self.weight, self.scales_odd, self.biases_odd,
            group_size=32, bits=self.bits,
        )
        result = even + odd
        if "bias" in self:
            result += self.bias
        return result
