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

# Other published `quant_method` names that transformers does not implement.
# In transformers 5.x an unknown name only emits "Unknown quantization type"
# and returns a model whose packed weights were randomly initialized, so
# `gptqmodel/__init__.py` registers a refusing quantizer for every entry here.
# Keep this as one table: parallel fixes extend the same mapping.
# Values are the human-readable format labels used in the refusal message.
REFUSED_HF_QUANT_METHODS = {
    "exl3": "EXL3",
    "macaq": "MACaQ MXFP4",
    "paroquant": "ParoQuant",
    "qqq": "QQQ",
    "nf12": "NF12",
    "tt-blockfloat": "Tenstorrent block-float",
}
