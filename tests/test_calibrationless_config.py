# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.quantization.config import BaseQuantizeConfig, GPTQQuantizeConfig, QuantizeConfig, RTNQuantizeConfig, SmoothMAD


def test_quantize_config_calibrationless_round_trip():
    smooth = SmoothMAD(k=1.75)
    cfg = RTNQuantizeConfig(
        bits=4,
        group_size=128,
        smooth=smooth,
    )

    assert cfg.uses_calibrationless_lifecycle() is True
    assert cfg.requires_calibration_dataset() is False
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.smooth.k == pytest.approx(1.75)

    payload = cfg.to_dict()
    assert "method" not in payload["meta"]["calibrationless"]
    assert payload["meta"]["calibrationless"]["smooth"]["type"] == "mad"
    assert payload["meta"]["calibrationless"]["smooth"]["k"] == pytest.approx(1.75)

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, RTNQuantizeConfig)
    assert isinstance(reloaded.smooth, SmoothMAD)
    assert reloaded.smooth.k == pytest.approx(1.75)
    assert reloaded.uses_calibrationless_lifecycle() is True
    assert reloaded.requires_calibration_dataset() is False


def test_rtn_quantize_config_defaults_to_direct_smoother():
    cfg = RTNQuantizeConfig(bits=4, group_size=128)

    assert isinstance(cfg, BaseQuantizeConfig)
    assert not isinstance(cfg, GPTQQuantizeConfig)
    assert cfg.uses_calibrationless_lifecycle() is True
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.export_quant_method() is not None


def test_rtn_quantize_config_supports_awq_export_round_trip():
    smooth = SmoothMAD(k=1.5)
    cfg = RTNQuantizeConfig(
        bits=4,
        group_size=128,
        format="gemm",
        smooth=smooth,
    )

    assert cfg.format == "gemm"
    assert cfg.export_quant_method().value == "awq"

    payload = cfg.to_dict()
    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded, RTNQuantizeConfig)
    assert reloaded.format == cfg.format
    assert reloaded.export_quant_method() == cfg.export_quant_method()
    assert isinstance(reloaded.smooth, SmoothMAD)
    assert reloaded.smooth.k == pytest.approx(1.5)


def test_legacy_calibrationless_payload_still_dispatches_to_rtn():
    cfg = QuantizeConfig(
        bits=4,
        group_size=128,
        calibrationless={
            "method": "rtn",
            "smooth": {"type": "mad", "k": 2.0},
        },
    )

    assert isinstance(cfg, RTNQuantizeConfig)
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.smooth.k == pytest.approx(2.0)
