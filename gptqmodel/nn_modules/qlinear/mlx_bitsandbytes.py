# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# MLX dense matmul: Apple Inc., MIT, https://github.com/ml-explore/mlx
# bitsandbytes: Tim Dettmers et al., MIT, https://github.com/bitsandbytes-foundation/bitsandbytes
"""BitsAndBytes MLX dense fallback that preserves FP16/BF16 activation dtype."""

import mlx.nn as nn


class MlxBitsAndBytesLinear(nn.Module):
    """Run decoded BitsAndBytes weights through MLX dense matmul."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.freeze()

    def __call__(self, x):
        return self.linear(x).astype(x.dtype)
