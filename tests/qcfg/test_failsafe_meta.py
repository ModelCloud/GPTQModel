# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from gptqmodel.quantization.config import FailSafe, QuantizeConfig, SmoothMAD


def test_quantize_config_serializes_failsafe_in_meta():
    cfg = QuantizeConfig()
    payload = cfg.to_dict()

    assert "failsafe" not in payload
    assert "meta" in payload
    assert "failsafe" in payload["meta"]

    meta_failsafe = payload["meta"]["failsafe"]
    assert meta_failsafe["strategy"] == cfg.failsafe.strategy.value
    assert meta_failsafe["threshold"] == cfg.failsafe.threshold
    assert meta_failsafe["smooth"]["type"] == "mad"
    assert meta_failsafe["smooth"]["k"] == cfg.failsafe.smooth.k


def test_quantize_config_reads_failsafe_from_meta():
    cfg = QuantizeConfig()
    payload = cfg.to_dict()

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded.failsafe, FailSafe)
    assert reloaded.failsafe.strategy == cfg.failsafe.strategy
    assert reloaded.failsafe.threshold == cfg.failsafe.threshold
    assert isinstance(reloaded.failsafe.smooth, SmoothMAD)
    assert reloaded.failsafe.smooth.k == cfg.failsafe.smooth.k
