# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.quantization.config import QuantizeConfig, RTNQuantizeConfig, SmoothMAD


def test_quantize_config_calibrationless_round_trip():
    cfg = RTNQuantizeConfig(
        bits=4,
        group_size=128,
        smooth={"type": "mad", "k": 1.75},
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

    assert isinstance(cfg, QuantizeConfig)
    assert cfg.uses_calibrationless_lifecycle() is True
    assert isinstance(cfg.smooth, SmoothMAD)


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
