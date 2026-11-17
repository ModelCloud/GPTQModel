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

# Checkpoints written before the rename store this method and format name. It
# is accepted on load so those checkpoints keep working; new checkpoints only
# ever publish ``paro``.
LEGACY_PARO_QUANT_METHOD = "paroquant"

# Other published `quant_method` names that transformers does not implement.
# In transformers 5.x an unknown name only emits "Unknown quantization type"
# and returns a model whose packed weights were randomly initialized, so
# `gptqmodel/__init__.py` registers a refusing quantizer for every entry here.
# Keep this as one table: parallel fixes extend the same mapping.
# Values are the human-readable format labels used in the refusal message.
REFUSED_HF_QUANT_METHODS = {
    # GPT-QModel's BitsAndBytes layout is not HF bnb's: it stores
    # `weight_scb`/`weight_absmax`/`weight_quant_map`/`weight_nested_*`, so
    # adding `load_in_8bit` would not make it loadable. transformers 5 has only
    # `bitsandbytes_4bit`/`bitsandbytes_8bit` and picks one from
    # `load_in_4bit`/`load_in_8bit`, so the plain name must fail closed instead
    # of returning a bf16 `nn.Linear` full of int8 codes (#4664).
    "bitsandbytes": "BitsAndBytes",
    "exl3": "EXL3",
    "macaq": "MACaQ MXFP4",
    "paro": "Paro",
    LEGACY_PARO_QUANT_METHOD: "Paro",
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
    "bitsandbytes": "`GPTQModel.load`",
    "exl3": "`GPTQModel.load` with the ExLlamaV3 backend",
    "macaq": "`GPTQModel.load`",
    "paro": "`GPTQModel.load`",
    LEGACY_PARO_QUANT_METHOD: "`GPTQModel.load`",
    "qqq": "`GPTQModel.load`",
    "nf12": "`GPTQModel.load`",
    "tt-blockfloat": "`GPTQModel.load`",
    GGUF_EXPORT_QUANT_METHOD: "`GPTQModel.load`",
}

#: ``docs/support-matrix.md`` "Feature" cell backing each remedy, so a test can
#: prove the suggested runtime is the one that matrix marks ✅.
REFUSED_HF_QUANT_METHOD_SUPPORT_ROWS = {
    "bitsandbytes": "BitsAndBytes",
    "exl3": "Exllama V3 / EXL3",
    "macaq": "MACaQ (MXFP4 W4A4)",
    "paro": "Paro",
    LEGACY_PARO_QUANT_METHOD: "Paro",
    "qqq": "QQQ",
    "nf12": "NF12",
}


def refused_hf_load_remedy(method: str) -> str:
    """Loader instruction for a refused ``quant_method`` (never vLLM)."""

    return REFUSED_HF_QUANT_METHOD_REMEDIES.get(method, "`GPTQModel.load`")
