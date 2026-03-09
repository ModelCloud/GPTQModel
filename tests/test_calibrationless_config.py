# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.quantization.config import (
    CalibrationlessConfig,
    CalibrationlessMethod,
    QuantizeConfig,
    SmoothMAD,
)


def test_quantize_config_calibrationless_round_trip():
    cfg = QuantizeConfig(
        bits=4,
        group_size=128,
        calibrationless={
            "method": "rtn",
            "smooth": {"type": "mad", "k": 1.75},
        },
    )

    assert cfg.uses_calibrationless_lifecycle() is True
    assert cfg.requires_calibration_dataset() is False
    assert isinstance(cfg.calibrationless, CalibrationlessConfig)
    assert cfg.calibrationless.method == CalibrationlessMethod.RTN
    assert isinstance(cfg.calibrationless.smooth, SmoothMAD)
    assert cfg.calibrationless.smooth.k == pytest.approx(1.75)

    payload = cfg.to_dict()
    assert payload["meta"]["calibrationless"]["method"] == "rtn"
    assert payload["meta"]["calibrationless"]["smooth"]["type"] == "mad"
    assert payload["meta"]["calibrationless"]["smooth"]["k"] == pytest.approx(1.75)

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded.calibrationless, CalibrationlessConfig)
    assert reloaded.calibrationless.method == CalibrationlessMethod.RTN
    assert isinstance(reloaded.calibrationless.smooth, SmoothMAD)
    assert reloaded.calibrationless.smooth.k == pytest.approx(1.75)
    assert reloaded.uses_calibrationless_lifecycle() is True
    assert reloaded.requires_calibration_dataset() is False
