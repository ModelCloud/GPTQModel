# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Plain transformers must refuse GPT-QModel's BitsAndBytes layout (#4664).

Transformers 5 has no quantizer named ``bitsandbytes``; it only registers
``bitsandbytes_4bit``/``bitsandbytes_8bit`` and picks one from
``load_in_4bit``/``load_in_8bit``. GPT-QModel writes ``quant_method:
"bitsandbytes"`` with neither flag, so before the refusal was registered HF
warned ``Unknown quantization type`` and loaded the int8 codes into a bf16
``nn.Linear`` (nll 1.67 -> 15.57) with no error.
"""

import json

import pytest


pytestmark = [pytest.mark.cpu]


def _write_bnb_checkpoint(path, bits=8):
    from transformers import LlamaConfig

    from gptqmodel.quantization import BitsAndBytesConfig

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    payload = config.to_dict()
    payload["quantization_config"] = BitsAndBytesConfig(bits=bits).to_dict()
    path.mkdir()
    (path / "config.json").write_text(json.dumps(payload))
    return payload


def test_bitsandbytes_is_a_refused_hf_method():
    from gptqmodel._quant_methods import (
        REFUSED_HF_QUANT_METHOD_REMEDIES,
        REFUSED_HF_QUANT_METHOD_SUPPORT_ROWS,
        REFUSED_HF_QUANT_METHODS,
    )

    assert REFUSED_HF_QUANT_METHODS["bitsandbytes"] == "BitsAndBytes"
    assert REFUSED_HF_QUANT_METHOD_REMEDIES["bitsandbytes"] == "`GPTQModel.load`"
    assert REFUSED_HF_QUANT_METHOD_SUPPORT_ROWS["bitsandbytes"] == "BitsAndBytes"


def test_refusing_quantizer_is_registered_for_the_plain_name():
    from transformers.quantizers import auto as hf_auto

    from gptqmodel.utils.hf_refusing_quantizer import (
        GPTQModelRefusingHfQuantizer,
        GPTQModelRefusingQuantizationConfig,
    )

    assert hf_auto.AUTO_QUANTIZER_MAPPING["bitsandbytes"] is GPTQModelRefusingHfQuantizer
    assert (
        hf_auto.AUTO_QUANTIZATION_CONFIG_MAPPING["bitsandbytes"]
        is GPTQModelRefusingQuantizationConfig
    )


def test_refusal_names_a_loader_that_reads_the_layout():
    from types import SimpleNamespace

    from gptqmodel.utils.hf_refusing_quantizer import GPTQModelRefusingHfQuantizer

    with pytest.raises(ValueError) as error:
        GPTQModelRefusingHfQuantizer(SimpleNamespace(quant_method="bitsandbytes"))

    message = str(error.value)
    assert "cannot be loaded by `transformers`" in message
    assert "GPTQModel.load" in message
    assert "vLLM" not in message


def test_hf_bitsandbytes_config_is_not_refused():
    """transformers' own bnb config shares the plain name and must keep loading.

    #4706 refused every ``quant_method == "bitsandbytes"``, which is also what
    ``transformers.BitsAndBytesConfig`` reports, so importing GPT-QModel broke
    every HF bnb load (QLoRA, any Hub bnb checkpoint) in that process (#4731).
    """

    from transformers import BitsAndBytesConfig
    from transformers.quantizers.auto import AutoHfQuantizer

    four_bit = AutoHfQuantizer.from_config(BitsAndBytesConfig(load_in_4bit=True))
    assert type(four_bit).__name__ == "Bnb4BitHfQuantizer"

    eight_bit = AutoHfQuantizer.from_config(BitsAndBytesConfig(load_in_8bit=True))
    assert type(eight_bit).__name__ == "Bnb8BitHfQuantizer"


@pytest.mark.parametrize("bits", [4, 8])
def test_hf_saved_bitsandbytes_config_dict_is_not_refused(bits):
    """The dict an HF bnb ``save_pretrained`` writes must resolve to HF's config.

    It carries ``quant_method: "bitsandbytes"`` plus the matching ``load_in_*``
    flag, which ``AutoQuantizationConfig.from_dict`` turns into a real
    ``BitsAndBytesConfig`` before ``from_config`` appends ``_4bit``/``_8bit``.
    """

    from transformers import BitsAndBytesConfig
    from transformers.quantizers.auto import AutoHfQuantizer, AutoQuantizationConfig

    saved = {
        "quant_method": "bitsandbytes",
        "load_in_4bit": bits == 4,
        "load_in_8bit": bits == 8,
    }
    assert isinstance(AutoQuantizationConfig.from_dict(saved), BitsAndBytesConfig)

    quantizer = AutoHfQuantizer.from_config(saved)
    assert type(quantizer).__name__ == f"Bnb{bits}BitHfQuantizer"


@pytest.mark.parametrize("bits", [4, 8])
def test_gptqmodel_bitsandbytes_layout_is_still_refused(bits):
    """Both GPT-QModel bnb layouts (``bits: 4`` and ``bits: 8``) still fail closed."""

    from transformers.quantizers.auto import AutoHfQuantizer

    from gptqmodel.quantization import BitsAndBytesConfig as GPTQModelBnB

    config = GPTQModelBnB(bits=bits)
    assert "load_in_4bit" not in config.to_dict() and "load_in_8bit" not in config.to_dict()

    with pytest.raises(ValueError, match="cannot be loaded by `transformers`"):
        AutoHfQuantizer.from_config(config)
    with pytest.raises(ValueError, match="cannot be loaded by `transformers`"):
        AutoHfQuantizer.from_config(config.to_dict())


@pytest.mark.parametrize("bits", [4, 8])
def test_from_pretrained_refuses_a_bitsandbytes_checkpoint(tmp_path, bits):
    from transformers import AutoModelForCausalLM

    _write_bnb_checkpoint(tmp_path / "bnb", bits=bits)

    with pytest.raises(ValueError) as error:
        AutoModelForCausalLM.from_pretrained(str(tmp_path / "bnb"), local_files_only=True)

    assert "cannot be loaded by `transformers`" in str(error.value)


def test_save_warning_labels_a_bitsandbytes_checkpoint():
    from gptqmodel.models.writer import hf_compat_warnings
    from gptqmodel.quantization import FORMAT, BitsAndBytesConfig

    warnings = hf_compat_warnings(BitsAndBytesConfig(bits=8), FORMAT.BITSANDBYTES)

    assert len(warnings) == 1
    assert "bitsandbytes" in warnings[0]
    assert "random-initializes" in warnings[0]
    assert "GPTQModel.load" in warnings[0]


def test_every_saveable_method_is_hf_readable_or_refused():
    """The contract the table exists for, independent of its current entries."""

    from transformers.quantizers import auto as hf_auto

    from gptqmodel._quant_methods import GGUF_EXPORT_QUANT_METHOD, REFUSED_HF_QUANT_METHODS
    from gptqmodel.quantization import METHOD
    from gptqmodel.utils.hf_refusing_quantizer import GPTQModelRefusingHfQuantizer

    for method in METHOD:
        # GGUF is published under the non-colliding `gptqmodel_gguf` alias;
        # the plain `gguf` name belongs to transformers' file-based loader.
        name = GGUF_EXPORT_QUANT_METHOD if method is METHOD.GGUF else method.value
        if name in REFUSED_HF_QUANT_METHODS:
            continue
        quantizer = hf_auto.AUTO_QUANTIZER_MAPPING.get(name)
        assert quantizer is not None, (
            f"`{name}` is neither refused by GPT-QModel nor readable by transformers; "
            "plain HF would silently load garbage"
        )
        assert quantizer is not GPTQModelRefusingHfQuantizer, name
