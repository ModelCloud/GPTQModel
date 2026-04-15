# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import AWQConfig, GPTQConfig, InputActivationQuantConfig, QuantizeConfig


def _canonical_input_activations(**overrides):
    payload = {
        "dtype": "float8_e4m3fn",
        "strategy": "tensor",
        "dynamic": False,
        "symmetric": True,
    }
    payload.update(overrides)
    return payload


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_gptq_input_activations_roundtrip_through_runtime_payload():
    cfg = QuantizeConfig(
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        input_activations=_canonical_input_activations(dtype="e4m3"),
    )

    assert isinstance(cfg, GPTQConfig)
    assert isinstance(cfg.input_activations, InputActivationQuantConfig)
    assert cfg.input_activations.dtype == "float8_e4m3fn"

    payload = cfg.to_dict()
    assert payload["input_activations"]["dtype"] == "float8_e4m3fn"
    assert "input_activations" not in payload["meta"]

    restored = QuantizeConfig.from_quant_config(payload)
    assert isinstance(restored, GPTQConfig)
    assert isinstance(restored.input_activations, InputActivationQuantConfig)
    assert restored.input_activations.dtype == "float8_e4m3fn"
    assert restored.quant_linear_init_kwargs()["input_activations"]["dtype"] == "float8_e4m3fn"


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_awq_input_activations_roundtrip_through_runtime_payload():
    cfg = QuantizeConfig(
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        input_activations=_canonical_input_activations(dtype="e4m3"),
    )

    assert isinstance(cfg, AWQConfig)
    assert isinstance(cfg.input_activations, InputActivationQuantConfig)
    assert cfg.input_activations.dtype == "float8_e4m3fn"

    payload = cfg.to_dict()
    assert payload["input_activations"]["dtype"] == "float8_e4m3fn"
    assert "input_activations" not in payload["meta"]

    restored = QuantizeConfig.from_quant_config(payload)
    assert isinstance(restored, AWQConfig)
    assert isinstance(restored.input_activations, InputActivationQuantConfig)
    assert restored.input_activations.dtype == "float8_e4m3fn"
    assert restored.quant_linear_init_kwargs()["input_activations"]["dtype"] == "float8_e4m3fn"


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_awq_input_activations_roundtrip_preserves_canonical_runtime_payload():
    cfg = QuantizeConfig(
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        input_activations=_canonical_input_activations(),
    )

    assert isinstance(cfg, AWQConfig)
    assert isinstance(cfg.input_activations, InputActivationQuantConfig)
    assert cfg.input_activations.dtype == "float8_e4m3fn"
    assert cfg.input_activations.dynamic is False
    assert cfg.input_activations.symmetric is True


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_input_activations_surface_defaults_to_awq_w4a8_path():
    cfg = QuantizeConfig(
        input_activations=_canonical_input_activations(),
    )

    assert isinstance(cfg, AWQConfig)
    assert cfg.quant_method == METHOD.AWQ
    assert cfg.format == FORMAT.GEMM


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_awq_input_activations_payload_roundtrips_without_meta_alias():
    cfg = QuantizeConfig(
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        input_activations=_canonical_input_activations(),
    )

    assert isinstance(cfg.input_activations, InputActivationQuantConfig)
    assert cfg.input_activations.dtype == "float8_e4m3fn"
    assert cfg.input_activations.dynamic is False
    assert cfg.input_activations.symmetric is True
    payload = cfg.to_dict()
    assert payload["input_activations"]["dtype"] == "float8_e4m3fn"
    assert payload["input_activations"]["symmetric"] is True
    assert "input_activations" not in payload["meta"]


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_activation_alias_is_rejected():
    with pytest.raises(ValueError, match="`activation` is no longer supported"):
        QuantizeConfig(
            activation={
                "dtype": "float8_e4m3fn",
            },
        )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_awq_input_activations_surface_rejects_non_gemm_format():
    with pytest.raises(ValueError, match="format` must be `gemm`"):
        QuantizeConfig(
            quant_method=METHOD.AWQ,
            format=FORMAT.MARLIN,
            input_activations=_canonical_input_activations(),
        )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_input_activations_rejects_removed_method_key():
    with pytest.raises(ValueError, match="remove legacy keys `method`"):
        QuantizeConfig(
            quant_method=METHOD.AWQ,
            format=FORMAT.GEMM,
            input_activations={
                "method": "fp8",
                "dtype": "float8_e4m3fn",
            },
        )


def test_input_activations_rejects_removed_implementation_key():
    with pytest.raises(ValueError, match="remove legacy keys `implementation`"):
        QuantizeConfig(
            quant_method=METHOD.AWQ,
            format=FORMAT.GEMM,
            input_activations={
                "implementation": "reference",
                "dtype": "float8_e4m3fn",
            },
        )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_input_activations_rejects_removed_scales_key():
    with pytest.raises(ValueError, match="no longer accepts `scales`"):
        QuantizeConfig(
            quant_method=METHOD.AWQ,
            format=FORMAT.GEMM,
            input_activations={
                **_canonical_input_activations(),
                "scales": "static",
            },
        )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_input_activations_in_meta_is_rejected():
    with pytest.raises(ValueError, match="meta.input_activations"):
        QuantizeConfig.from_quant_config(
            {
                "bits": 4,
                "dynamic": None,
                "group_size": 128,
                "desc_act": False,
                "sym": True,
                "checkpoint_format": "gemm",
                "quant_method": "awq",
                "meta": {
                    "input_activations": {
                        "dtype": "float8_e4m3fn",
                        "strategy": "tensor",
                        "dynamic": False,
                        "symmetric": True,
                    }
                },
            }
        )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
def test_legacy_input_activation_payload_still_normalizes_to_dtype_schema():
    cfg = QuantizeConfig(
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        input_activations={
            "type": "float",
            "bits": 8,
            "format": "e4m3",
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
        },
    )

    assert cfg.input_activations.to_dict() == {
        "dtype": "float8_e4m3fn",
        "strategy": "token",
        "dynamic": True,
        "symmetric": True,
    }
