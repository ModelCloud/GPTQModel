# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Standalone quantization-method names shared by package-init shims.

Kept out of ``gptqmodel.quantization.config`` so ``gptqmodel/__init__.py`` can
import it before the quantization package finishes initializing.
"""

from typing import Tuple


# GPT-QModel's packed GGUF tensors are not readable by transformers' own
# `GgufHfQuantizer`, which owns the `gguf` name and expects a `.gguf` file.
# Publish a non-colliding name so an HF loader fails loudly instead of
# random-initializing every linear layer. Existing checkpoints that carry
# `quant_method: "gguf"` (plus `method: "gguf"`) keep loading in GPT-QModel.
GGUF_EXPORT_QUANT_METHOD = "gptqmodel_gguf"

# The single table of `quant_method` names GPT-QModel publishes for layouts
# that plain `transformers` does not implement. `transformers` only warns
# ("Unknown quantization type ... skip the quantization") and then
# random-initializes every linear layer, so package import registers a refusing
# quantizer for each name here and the writer warns at save time. Add a method
# here, not at a call site, when it becomes GPT-QModel-only.
GPTQMODEL_ONLY_QUANT_METHODS: Tuple[str, ...] = (
    "exl3",
    "qqq",
    "paroquant",
    "macaq",
    "nf12",
    "tt-blockfloat",
)
