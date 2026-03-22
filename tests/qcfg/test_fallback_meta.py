# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from gptqmodel.quantization.config import Fallback, QuantizeConfig, SmoothMAD


def test_quantize_config_serializes_default_fallback_in_meta_without_smoother():
    cfg = QuantizeConfig()
    payload = cfg.to_dict()

    assert "fallback" not in payload
    assert "meta" in payload
    assert "fallback" in payload["meta"]

    meta_fallback = payload["meta"]["fallback"]
    assert meta_fallback["strategy"] == cfg.fallback.strategy.value
    assert meta_fallback["threshold"] == cfg.fallback.threshold
    assert meta_fallback["smooth"] is None


def test_quantize_config_reads_default_fallback_from_meta_without_smoother():
    cfg = QuantizeConfig()
    payload = cfg.to_dict()

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded.fallback, Fallback)
    assert reloaded.fallback.strategy == cfg.fallback.strategy
    assert reloaded.fallback.threshold == cfg.fallback.threshold
    assert reloaded.fallback.smooth is None


def test_quantize_config_round_trips_explicit_fallback_smoother():
    cfg = QuantizeConfig(fallback=Fallback(smooth=SmoothMAD(k=1.75)))
    payload = cfg.to_dict()

    meta_fallback = payload["meta"]["fallback"]
    assert meta_fallback["smooth"]["type"] == "mad"
    assert meta_fallback["smooth"]["k"] == 1.75

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded.fallback.smooth, SmoothMAD)
    assert reloaded.fallback.smooth.k == cfg.fallback.smooth.k
