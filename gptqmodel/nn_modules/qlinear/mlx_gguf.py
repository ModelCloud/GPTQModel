# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# MLX quantized matmul: Apple Inc., MIT, https://github.com/ml-explore/mlx
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
"""GGUF MLX inference wrapper that preserves FP16/BF16 activation dtype."""

import mlx.nn as nn


class MlxGGUFLinear(nn.Module):
    """Run MLX quantized GGUF matmul and cast its output to the input dtype."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.freeze()

    def __call__(self, x):
        return self.linear(x).astype(x.dtype)
