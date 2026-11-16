# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Config serialization must survive construction, reload, and re-save.

Issue #4052: ``EXL3Config`` bypassed the base ``__post_init__`` normalisation,
so ``auto_dtype_cast``, ``telemetry``, and ``mse`` stayed raw strings/dicts and
``to_dict`` crashed. The same audit found ``static_groups`` written to
``meta`` but never read back, and bool fields coerced with ``bool()``.
"""

import pytest

from gptqmodel.quantization.config import (
    FORMAT,
    AutoDtypeCast,
    AWQConfig,
    BitsAndBytesConfig,
    EXL3Config,
    FP8Config,
    GGUFConfig,
    GPTQConfig,
    MSEConfig,
    NF12Config,
    ParoConfig,
    QQQConfig,
    QuantizeConfig,
    RTNConfig,
    TelemetryConfig,
    TTBlockfloatConfig,
    VPTQConfig,
)


CONFIG_CLASSES = (
    GPTQConfig,
    AWQConfig,
    ParoConfig,
    QQQConfig,
    VPTQConfig,
    NF12Config,
    TTBlockfloatConfig,
    FP8Config,
    BitsAndBytesConfig,
    EXL3Config,
    RTNConfig,
    GGUFConfig,
)


@pytest.mark.parametrize("config_cls", CONFIG_CLASSES, ids=lambda cls: cls.__name__)
def test_every_concrete_config_normalizes_serialized_fields(config_cls):
    config = config_cls(
        auto_dtype_cast="off",
        telemetry={"device": True},
        mse=2.4,
    )

    assert config.auto_dtype_cast is AutoDtypeCast.OFF
    assert isinstance(config.telemetry, TelemetryConfig)
    assert config.telemetry.device is True
    assert isinstance(config.mse, MSEConfig)
    assert config.mse.norm == 2.4


def test_exl3_config_round_trips_serialized_fields():
    config = EXL3Config(
        bits=3,
        auto_dtype_cast="off",
        telemetry=TelemetryConfig(device=True),
        mse=2.4,
    )

    payload = config.to_dict()
    assert payload["auto_dtype_cast"] == "off"
    assert payload["meta"]["telemetry"] == {"device": True}

    reloaded = EXL3Config.from_quant_config(payload)
    assert reloaded.auto_dtype_cast is AutoDtypeCast.OFF
    assert isinstance(reloaded.telemetry, TelemetryConfig)
    assert isinstance(reloaded.mse, MSEConfig)
    assert reloaded.to_dict() == payload


@pytest.mark.parametrize(
    ("format_value", "expected_quant_method"),
    [
        (FORMAT.GPTQ, "gptq"),
        (FORMAT.GPTQ_V2, "gptq"),
        (FORMAT.GEMM, "awq"),
        (FORMAT.GEMV, "awq"),
        (FORMAT.GEMV_FAST, "awq"),
        (FORMAT.LLM_AWQ, "awq"),
    ],
)
def test_rtn_config_serializes_export_quant_method(format_value, expected_quant_method):
    """RTN's AWQ export formats must not be labelled `quant_method: gptq`."""

    from transformers.quantizers import AutoQuantizationConfig

    config = RTNConfig(bits=4, group_size=128, format=format_value)
    payload = config.to_dict()

    assert payload["method"] == "gptq"
    assert payload["quant_method"] == expected_quant_method

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, RTNConfig)
    assert reloaded.export_quant_method().value == expected_quant_method
    # transformers must route the serialized payload to the matching loader.
    assert type(AutoQuantizationConfig.from_dict(payload)).__name__ == (
        "AwqConfig" if expected_quant_method == "awq" else "GPTQConfig"
    )


def test_exl3_config_rejects_invalid_auto_dtype_cast():
    with pytest.raises(ValueError, match="auto_dtype_cast"):
        EXL3Config(auto_dtype_cast="bogus")


def test_static_groups_written_to_meta_reloads():
    config = GPTQConfig(bits=4, static_groups=True)
    # The writer records this field in `meta`, not as a top-level config key.
    config.meta_set("static_groups", True)

    payload = config.to_dict()
    assert payload["meta"]["static_groups"] is True

    reloaded = GPTQConfig.from_quant_config(payload)
    assert reloaded.static_groups is True
    assert GPTQConfig.from_quant_config(GPTQConfig(bits=4).to_dict()).static_groups is False


@pytest.mark.parametrize("config_cls", CONFIG_CLASSES, ids=lambda cls: cls.__name__)
def test_true_sequential_round_trips_for_every_concrete_config(config_cls):
    """#4382: the shared serializer must not drop `true_sequential`."""

    config = config_cls(true_sequential=False)

    payload = config.to_dict()

    assert payload["meta"]["true_sequential"] is False
    assert QuantizeConfig(**payload).true_sequential is False
    assert QuantizeConfig.from_quant_config(payload).true_sequential is False


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("desc_act", "False"),
        ("desc_act", 1),
        ("sym", "False"),
        ("sym", 1),
    ),
)
def test_bool_fields_reject_non_bool_values(field_name, value):
    with pytest.raises(TypeError, match=field_name):
        GPTQConfig(**{field_name: value})


def test_bool_fields_still_accept_bools():
    config = GPTQConfig(desc_act=False, sym=False)
    assert config.desc_act is False
    assert config.sym is False
