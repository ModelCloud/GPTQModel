# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 format: TurboDerp and ExLlamaV3 contributors, MIT, https://github.com/turboderp-org/exllamav3
"""EXL3 MLX dense fallback that preserves FP16/BF16 activation dtype."""

import mlx.nn as nn


class MlxEXL3Linear(nn.Module):
    """Run once-decoded EXL3 weights through MLX dense matmul."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.freeze()

    def __call__(self, x):
        return self.linear(x).astype(x.dtype)
