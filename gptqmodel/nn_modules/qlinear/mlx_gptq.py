# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# MLX quantized matmul: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""GPTQ MLX inference wrapper that preserves FP16/BF16 activation dtype."""

import mlx.nn as nn


class MlxGPTQLinear(nn.Module):
    """Run MLX affine GPTQ matmul and round its output to the input dtype."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.freeze()

    def __call__(self, x):
        return self.linear(x).astype(x.dtype)
