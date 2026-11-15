# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Standalone quantization-method names shared by package-init shims.

Kept out of ``gptqmodel.quantization.config`` so ``gptqmodel/__init__.py`` can
import it before the quantization package finishes initializing.
"""

# GPT-QModel's packed GGUF tensors are not readable by transformers' own
# `GgufHfQuantizer`, which owns the `gguf` name and expects a `.gguf` file.
# Publish a non-colliding name so an HF loader fails loudly instead of
# random-initializing every linear layer. Existing checkpoints that carry
# `quant_method: "gguf"` (plus `method: "gguf"`) keep loading in GPT-QModel.
GGUF_EXPORT_QUANT_METHOD = "gptqmodel_gguf"
