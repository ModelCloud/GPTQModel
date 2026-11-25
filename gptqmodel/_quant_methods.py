# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Standalone quantization-method names shared by config and save warnings.

Kept out of ``gptqmodel.quantization.config`` so it imports without the
quantization package.
"""

# GPT-QModel's packed GGUF tensors are not readable by transformers' own
# `GgufHfQuantizer`, which owns the `gguf` name and expects a `.gguf` file.
# Publish a non-colliding name so transformers' GGUF quantizer never claims it.
# Existing checkpoints that carry
# `quant_method: "gguf"` (plus `method: "gguf"`) keep loading in GPT-QModel.
GGUF_EXPORT_QUANT_METHOD = "gptqmodel_gguf"

#: Methods that reuse another method's checkpoint layout publish that layout's
#: ``quant_method`` name instead of their own. ADMM-Q writes ordinary
#: GPTQ-packed tensors, so plain transformers, vLLM and SGLang read the
#: checkpoint as GPTQ; the internal method is preserved separately in ``meta``.
EXPORTED_QUANT_METHOD_NAMES = {
    "admmq": "gptq",
}

# Checkpoints written by GPT-QModel's removed FP8 exporter (between #5181 and
# #5242) publish this method. It is accepted on load and normalized to
# `METHOD.FP8` so those checkpoints keep working, but new checkpoints never
# publish it: FP8 is load-only and the standard `fp8` name belongs to
# transformers' `FineGrainedFP8`.
LEGACY_FP8_QUANT_METHOD = "gptqmodel_fp8"

# transformers' `FineGrainedFP8` reads `weight_scale_inv` as a multiplier.
# Every GPT-QModel FP8 export stores the inverse convention, and released
# checkpoints up to v7.11.2 publish the standard `fp8` name with
# `weight_scale_semantics: "inverse"`. Refuse that combination as well; no
# production FP8 producer writes it.
FP8_QUANT_METHOD = "fp8"
INVERSE_WEIGHT_SCALE_SEMANTICS = "inverse"


# Checkpoints written before the rename store this method and format name. It
# is accepted on load so those checkpoints keep working; new checkpoints only
# ever publish ``paro``.
LEGACY_PARO_QUANT_METHOD = "paroquant"

# Other published `quant_method` names that transformers does not implement.
# In transformers 5.x an unknown name only emits "Unknown quantization type"
# and returns a model whose packed weights were randomly initialized. GPT-QModel
# never registers into transformers, so the save path warns for every entry here
# and these checkpoints load only through `GPTQModel.load`.
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
    LEGACY_FP8_QUANT_METHOD: "FP8",
    "macaq": "MACaQ MXFP4",
    "cptc": "CPTC",
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
    LEGACY_FP8_QUANT_METHOD: "`GPTQModel.load`",
    "macaq": "`GPTQModel.load`",
    "cptc": "`GPTQModel.load`",
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
    LEGACY_FP8_QUANT_METHOD: "FP8",
    "macaq": "MACaQ (MXFP4 W4A4)",
    "cptc": "[CPTC](cptc.md)",
    "paro": "Paro",
    LEGACY_PARO_QUANT_METHOD: "Paro",
    "qqq": "QQQ",
    "nf12": "NF12",
}


def _config_field(config, name: str):
    """Read one quantization-config field from a dict or config object."""

    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def is_legacy_inverse_fp8_config(config) -> bool:
    """Whether a standard ``fp8`` config declares GPT-QModel's inverse scale.

    transformers' ``FineGrainedFP8`` treats ``weight_scale_inv`` as a
    multiplier, so a checkpoint with this marker cannot be loaded by plain
    transformers without silently decoding the wrong weights. The marker is
    only written by GPT-QModel's removed FP8 exporter.
    """

    if _config_field(config, "quant_method") != FP8_QUANT_METHOD:
        return False
    semantics = _config_field(config, "weight_scale_semantics")
    return (
        isinstance(semantics, str)
        and semantics.strip().lower() == INVERSE_WEIGHT_SCALE_SEMANTICS
    )


def refused_hf_load_remedy(method: str) -> str:
    """Loader instruction for a refused ``quant_method`` (never vLLM)."""

    return REFUSED_HF_QUANT_METHOD_REMEDIES.get(method, "`GPTQModel.load`")
