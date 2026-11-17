# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Rejection messages must name the feature the checkpoint actually declares.

A checkpoint that asks for KV-cache quantization used to be refused with
"does not support loading of activation quantized models", which sends the user
looking at their activation config. The early config check and the ModelOpt
runtime check also disagreed about the ``num_bits`` boundary: any non-empty
scheme was rejected in one place, only sub-16-bit schemes in the other.
"""

import types

import pytest

from gptqmodel.models.auto import _is_supported_quantization_config
from gptqmodel.utils.quant_metadata import (
    UnsupportedMetadata,
    declares_reduction,
    describe_unsupported_metadata,
    find_unsupported_metadata,
)


def _config(quantization_config):
    return types.SimpleNamespace(quantization_config=quantization_config)


def test_kv_cache_scheme_is_reported_as_kv_cache_not_activation():
    payload = {"quant_method": "modelopt", "kv_cache_scheme": {"num_bits": 8, "type": "float"}}

    with pytest.raises(ValueError) as excinfo:
        _is_supported_quantization_config(_config(payload))

    message = str(excinfo.value)
    assert "quantize the KV cache" in message
    assert "kv_cache_scheme" in message
    assert "8 bits" in message
    assert "activation quantized models" not in message


def test_activation_quantization_keeps_its_own_message():
    payload = {
        "quant_method": "modelopt",
        "config_groups": {"group_0": {"input_activations": {"num_bits": 4}}},
    }

    with pytest.raises(ValueError, match="activation quantized models") as excinfo:
        _is_supported_quantization_config(_config(payload))

    assert "KV cache" not in str(excinfo.value)


def test_sixteen_bit_schemes_do_not_block_loading():
    """A scheme that does not reduce precision is a no-op, not a blocker."""
    payload = {"quant_method": "modelopt", "kv_cache_scheme": {"num_bits": 16, "type": "float"}}
    assert find_unsupported_metadata(payload) is None


def test_scheme_without_a_width_is_treated_as_quantized():
    payload = {"kv_cache_scheme": {"type": "float"}}
    found = find_unsupported_metadata(payload)
    assert found is not None
    assert found.kind == "kv_cache"
    assert found.num_bits is None


def test_empty_payloads_are_not_rejected():
    assert find_unsupported_metadata(None) is None
    assert find_unsupported_metadata({}) is None
    assert find_unsupported_metadata({"kv_cache_scheme": None}) is None
    assert find_unsupported_metadata({"kv_cache_scheme": {}}) is None


def test_activation_in_a_config_group_wins_over_the_kv_cache_field():
    """Activation quantization is the larger blocker, so it is reported first."""
    payload = {
        "config_groups": {"group_0": {"input_activations": {"num_bits": 4}}},
        "kv_cache_scheme": {"num_bits": 8},
    }
    found = find_unsupported_metadata(payload)
    assert found.kind == "activation"


@pytest.mark.parametrize(
    "scheme,expected",
    [
        (None, False),
        ({}, False),
        ({"num_bits": 4}, True),
        ({"num_bits": 8}, True),
        ({"num_bits": 16}, False),
        ({"num_bits": 32}, False),
        ({"type": "float"}, True),
        ({"num_bits": "8"}, True),
        ({"num_bits": True}, True),
    ],
)
def test_declares_reduction_covers_the_boundary(scheme, expected):
    assert declares_reduction(scheme) is expected


def test_checkpoint_scales_message_is_distinct():
    message = describe_unsupported_metadata(
        UnsupportedMetadata(field="checkpoint_scales", kind="checkpoint_scales")
    )
    assert "pre-quantized activation scales" in message
    assert "KV cache" not in message


def test_kv_cache_message_offers_a_way_forward():
    message = describe_unsupported_metadata(
        UnsupportedMetadata(field="kv_cache_scheme", kind="kv_cache", num_bits=8)
    )
    assert "weight-only variant" in message
    assert "serving runtime" in message


def test_modelopt_runtime_reports_the_kv_cache_kind():
    """The runtime check must classify, not just reject."""

    from types import SimpleNamespace

    from gptqmodel.models.base import BaseQModel

    stub = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                quantization_config={"quant_method": "modelopt",
                                     "kv_cache_scheme": {"num_bits": 8, "type": "float"}},
            )
        )
    )

    found = BaseQModel._modelopt_unsupported_metadata(stub)
    assert found is not None
    assert found.kind == "kv_cache"
    assert found.num_bits == 8
