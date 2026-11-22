# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPT-QModel's BitsAndBytes layout is GPTQModel.load-only (#4664).

Transformers 5 has no quantizer named ``bitsandbytes``; it only registers
``bitsandbytes_4bit``/``bitsandbytes_8bit`` and picks one from
``load_in_4bit``/``load_in_8bit``. GPT-QModel writes ``quant_method:
"bitsandbytes"`` with neither flag, so plain HF warns ``Unknown quantization
type`` and loads the int8 codes into a bf16 ``nn.Linear``. GPT-QModel never
registers into transformers, so the save path warns and names
``GPTQModel.load`` instead.
"""

import pytest


pytestmark = [pytest.mark.cpu]


def test_bitsandbytes_is_a_refused_hf_method():
    from gptqmodel._quant_methods import (
        REFUSED_HF_QUANT_METHOD_REMEDIES,
        REFUSED_HF_QUANT_METHOD_SUPPORT_ROWS,
        REFUSED_HF_QUANT_METHODS,
    )

    assert REFUSED_HF_QUANT_METHODS["bitsandbytes"] == "BitsAndBytes"
    assert REFUSED_HF_QUANT_METHOD_REMEDIES["bitsandbytes"] == "`GPTQModel.load`"
    assert REFUSED_HF_QUANT_METHOD_SUPPORT_ROWS["bitsandbytes"] == "BitsAndBytes"


def test_save_warning_labels_a_bitsandbytes_checkpoint():
    from gptqmodel.models.writer import hf_compat_warnings
    from gptqmodel.quantization import FORMAT, BitsAndBytesConfig

    warnings = hf_compat_warnings(BitsAndBytesConfig(bits=8), FORMAT.BITSANDBYTES)

    assert len(warnings) == 1
    assert "bitsandbytes" in warnings[0]
    assert "random-initializes" in warnings[0]
    assert "GPTQModel.load" in warnings[0]


def test_every_saveable_method_is_hf_readable_or_warned():
    """The contract the table exists for, independent of its current entries."""

    from transformers.quantizers import auto as hf_auto

    from gptqmodel._quant_methods import (
        GGUF_EXPORT_QUANT_METHOD,
        REFUSED_HF_QUANT_METHODS,
    )
    from gptqmodel.quantization import METHOD

    for method in METHOD:
        # GGUF is published under a non-colliding alias; the plain `gguf`
        # name belongs to transformers' own loader. FP8 is load-only and keeps
        # the standard `fp8` name so transformers' `FineGrainedFP8` reads it.
        names = [method.value]
        if method is METHOD.GGUF:
            names.append(GGUF_EXPORT_QUANT_METHOD)
        for name in names:
            if name in REFUSED_HF_QUANT_METHODS or name == GGUF_EXPORT_QUANT_METHOD:
                # Both carry a save-time warning naming `GPTQModel.load`.
                continue
            quantizer = hf_auto.AUTO_QUANTIZER_MAPPING.get(name)
            assert quantizer is not None, (
                f"`{name}` is neither warned about by GPT-QModel nor readable by transformers; "
                "plain HF would silently load garbage"
            )
