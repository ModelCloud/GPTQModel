# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Refusal messages must name a loader that can read the layout (#4676).

Both the transformers refusal and the save-time compatibility warning used to
say "load them with ``GPTQModel.load`` or vLLM". The vLLM bridge accepts only
``FORMAT.GPTQ``/``FORMAT.GEMM``, and ``docs/support-matrix.md`` marks vLLM x for
EXL3, MACaQ, Paro, QQQ, NF12 and GGUF, so following that advice hit a second
failure. The remedy is now per-method and checked against that matrix.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gptqmodel._quant_methods import (
    GGUF_EXPORT_QUANT_METHOD,
    REFUSED_HF_QUANT_METHOD_REMEDIES,
    REFUSED_HF_QUANT_METHOD_SUPPORT_ROWS,
    REFUSED_HF_QUANT_METHODS,
    refused_hf_load_remedy,
)
from gptqmodel.models.writer import hf_compat_warnings
from gptqmodel.quantization import METHOD
from gptqmodel.utils.hf_gguf_quantizer import GPTQModelGGUFHfQuantizer
from gptqmodel.utils.hf_refusing_quantizer import GPTQModelRefusingHfQuantizer


pytestmark = [pytest.mark.cpu]

MATRIX = Path(__file__).resolve().parents[1] / "docs" / "support-matrix.md"
SUPPORTED = "\u2705"


def _matrix_row(label):
    for line in MATRIX.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == label:
            return cells
    raise AssertionError(f"no support-matrix row for {label!r}")


def test_every_refused_method_has_a_remedy():
    assert set(REFUSED_HF_QUANT_METHODS) <= set(REFUSED_HF_QUANT_METHOD_REMEDIES)
    assert GGUF_EXPORT_QUANT_METHOD in REFUSED_HF_QUANT_METHOD_REMEDIES


@pytest.mark.parametrize("method", sorted(REFUSED_HF_QUANT_METHOD_REMEDIES))
def test_remedy_never_points_at_a_runtime_that_cannot_read_the_layout(method):
    remedy = refused_hf_load_remedy(method)

    assert "GPTQModel.load" in remedy
    assert "vllm" not in remedy.lower()


@pytest.mark.parametrize("method", sorted(REFUSED_HF_QUANT_METHOD_SUPPORT_ROWS))
def test_remedy_runtime_is_the_one_the_support_matrix_marks_supported(method):
    row = _matrix_row(REFUSED_HF_QUANT_METHOD_SUPPORT_ROWS[method])

    # Feature | GPT-QModel | Transformers | vLLM | SGLang | Lora Training
    assert row[1] == SUPPORTED, f"{row[0]} is not marked supported by GPT-QModel"
    assert row[3] == "x", f"{row[0]} is marked loadable by vLLM"


@pytest.mark.parametrize("method", sorted(REFUSED_HF_QUANT_METHOD_REMEDIES))
def test_refusal_names_a_working_loader(method):
    with pytest.raises(ValueError) as error:
        GPTQModelRefusingHfQuantizer(SimpleNamespace(quant_method=method))

    message = str(error.value)
    assert "cannot be loaded by `transformers`" in message
    assert "vLLM" not in message
    assert refused_hf_load_remedy(method) in message


def test_no_refusal_suggests_vllm():
    """Behavioral half of the contract, independent of the remedy table."""

    for method in REFUSED_HF_QUANT_METHODS:
        with pytest.raises(ValueError) as error:
            GPTQModelRefusingHfQuantizer(SimpleNamespace(quant_method=method))

        assert "vLLM" not in str(error.value), method


def test_packed_gguf_refusal_names_a_working_loader():
    with pytest.raises(ValueError) as error:
        GPTQModelGGUFHfQuantizer(SimpleNamespace())

    message = str(error.value)
    assert "GPTQModel.load" in message
    assert "vLLM" not in message


@pytest.mark.parametrize(
    "method",
    [member for member in METHOD if member.value in REFUSED_HF_QUANT_METHOD_REMEDIES],
    ids=lambda member: member.value,
)
def test_save_time_warning_names_a_working_loader(method):
    warnings = hf_compat_warnings(
        SimpleNamespace(method=method, lm_head=False), runtime_format=None
    )
    refused = [message for message in warnings if "GPT-QModel-only layout" in message]

    assert len(refused) == 1
    assert "vLLM" not in refused[0]
    assert refused_hf_load_remedy(method.value) in refused[0]


def test_packed_gguf_save_time_warning_names_a_working_loader():
    warnings = hf_compat_warnings(
        SimpleNamespace(method=METHOD.GGUF, lm_head=False), runtime_format=None
    )
    gguf = [message for message in warnings if "Packed GGUF" in message]

    assert len(gguf) == 1
    assert "vLLM" not in gguf[0]
    assert "`GPTQModel.load`" in gguf[0]
