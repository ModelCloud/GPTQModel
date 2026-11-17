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

#: How to load a checkpoint whose ``quant_method`` is in
#: ``REFUSED_HF_QUANT_METHODS`` (or the packed GGUF marker).
#:
#: Only loaders that actually read these layouts belong here. The vLLM bridge
#: accepts ``FORMAT.GPTQ``/``FORMAT.GEMM`` only, and
#: ``docs/support-matrix.md`` marks vLLM ✗ for every entry below, so pointing at
#: it sent users into a second failure (#4676).
REFUSED_HF_QUANT_METHOD_REMEDIES = {
    "exl3": "`GPTQModel.load` with the ExLlamaV3 backend",
    "macaq": "`GPTQModel.load`",
    "paroquant": "`GPTQModel.load`",
    "qqq": "`GPTQModel.load`",
    "nf12": "`GPTQModel.load`",
    "tt-blockfloat": "`GPTQModel.load`",
    GGUF_EXPORT_QUANT_METHOD: "`GPTQModel.load`",
}

#: ``docs/support-matrix.md`` "Feature" cell backing each remedy, so a test can
#: prove the suggested runtime is the one that matrix marks ✅.
REFUSED_HF_QUANT_METHOD_SUPPORT_ROWS = {
    "exl3": "Exllama V3 / EXL3",
    "macaq": "MACaQ (MXFP4 W4A4)",
    "paroquant": "ParoQuant",
    "qqq": "QQQ",
    "nf12": "NF12",
}


def refused_hf_load_remedy(method: str) -> str:
    """Loader instruction for a refused ``quant_method`` (never vLLM)."""

    return REFUSED_HF_QUANT_METHOD_REMEDIES.get(method, "`GPTQModel.load`")
