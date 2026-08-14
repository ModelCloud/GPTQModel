# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from typing import get_args, get_type_hints

import pytest

from gptqmodel.quantization import QVQConfig
from gptqmodel.quantization.config import (
    FORMAT,
    GGUFBits,
    _normalize_quant_bits,
    quant_bits_width,
    serialize_quant_bits,
)


def test_quant_bit_helper_annotations_cover_every_runtime_type():
    normalize_hints = get_type_hints(_normalize_quant_bits)
    serialize_hints = get_type_hints(serialize_quant_bits)
    width_hints = get_type_hints(quant_bits_width)

    expected_inputs = {int, float, str, GGUFBits}
    assert set(get_args(normalize_hints["bits"])) == expected_inputs
    assert set(get_args(serialize_hints["bits"])) == expected_inputs
    assert set(get_args(width_hints["bits"])) == expected_inputs
    assert set(get_args(normalize_hints["return"])) == {int, float, GGUFBits}
    assert set(get_args(serialize_hints["return"])) == {int, float, str}


def test_quant_bit_helpers_preserve_integer_and_qvq_fractional_runtime_types():
    assert _normalize_quant_bits(4) == 4
    assert type(_normalize_quant_bits(4)) is int
    assert _normalize_quant_bits(4.0) == 4
    assert type(_normalize_quant_bits(4.0)) is int
    assert _normalize_quant_bits(2.5, format_value=FORMAT.QVQ) == 2.5
    assert type(_normalize_quant_bits(2.5, format_value=FORMAT.QVQ)) is float
    assert serialize_quant_bits(4) == 4
    assert serialize_quant_bits(2.5) == 2.5


def test_qvq_bit_choice_validation_uses_exact_rate_not_floor():
    qcfg = QVQConfig(bits=2.5, offload_to_disk=False)

    assert qcfg._bits_in_choices([2.5]) is True
    assert qcfg._bits_in_choices([2]) is False
    with pytest.raises(ValueError, match="GGUF bit encodings"):
        _normalize_quant_bits(GGUFBits.from_alias("q4_k_m"), format_value=FORMAT.QVQ)
