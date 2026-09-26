# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# MLX quantized matmul: Apple Inc., MIT, https://github.com/ml-explore/mlx
# AWQ format reference: ModelCloud.ai, Apache-2.0, GPT-QModel AWQ packers.
"""AWQ MLX inference wrapper that preserves FP16/BF16 activation dtype."""

import mlx.nn as nn


class MlxAWQLinear(nn.Module):
    """Run MLX affine AWQ matmul and round its output to the input dtype."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.freeze()

    def __call__(self, x):
        return self.linear(x).astype(x.dtype)
