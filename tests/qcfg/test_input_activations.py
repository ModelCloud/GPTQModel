# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import AWQConfig, GPTQConfig, InputActivationQuantConfig, QuantizeConfig


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_gptq_input_activations_roundtrip_through_meta_payload():
    cfg = QuantizeConfig(
        input_activations={
            "num_bits": 8,
            "type": "float",
            "format": "e4m3",
            "strategy": "tensor",
            "dynamic": True,
            "implementation": "reference",
        }
    )

    assert isinstance(cfg, GPTQConfig)
    assert isinstance(cfg.input_activations, InputActivationQuantConfig)
    assert cfg.input_activations.format == "float8_e4m3fn"

    payload = cfg.to_dict()
    assert "input_activations" not in payload
    assert payload["meta"]["input_activations"]["format"] == "float8_e4m3fn"

    restored = QuantizeConfig.from_quant_config(payload)
    assert isinstance(restored, GPTQConfig)
    assert isinstance(restored.input_activations, InputActivationQuantConfig)
    assert restored.input_activations.format == "float8_e4m3fn"
    assert restored.quant_linear_init_kwargs()["input_activations"]["format"] == "float8_e4m3fn"


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_awq_input_activations_roundtrip_through_meta_payload():
    cfg = QuantizeConfig(
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        input_activations={
            "num_bits": 8,
            "type": "float",
            "format": "e4m3",
            "strategy": "tensor",
            "dynamic": True,
            "implementation": "reference",
        },
    )

    assert isinstance(cfg, AWQConfig)
    assert isinstance(cfg.input_activations, InputActivationQuantConfig)
    assert cfg.input_activations.format == "float8_e4m3fn"

    payload = cfg.to_dict()
    assert "input_activations" not in payload
    assert payload["meta"]["input_activations"]["format"] == "float8_e4m3fn"

    restored = QuantizeConfig.from_quant_config(payload)
    assert isinstance(restored, AWQConfig)
    assert isinstance(restored.input_activations, InputActivationQuantConfig)
    assert restored.input_activations.format == "float8_e4m3fn"
    assert restored.quant_linear_init_kwargs()["input_activations"]["format"] == "float8_e4m3fn"
