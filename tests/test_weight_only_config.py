# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.quantization.config import BaseQuantizeConfig, GPTQQuantizeConfig, QuantizeConfig, RTNQuantizeConfig, SmoothMAD


def test_quantize_config_weight_only_round_trip():
    smooth = SmoothMAD(k=1.75)
    cfg = RTNQuantizeConfig(
        bits=4,
        group_size=128,
        smooth=smooth,
    )

    assert cfg.uses_weight_only_lifecycle() is True
    assert cfg.requires_calibration_dataset() is False
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.smooth.k == pytest.approx(1.75)

    payload = cfg.to_dict()
    assert "method" not in payload["meta"]["weight_only"]
    assert payload["meta"]["weight_only"]["smooth"]["type"] == "mad"
    assert payload["meta"]["weight_only"]["smooth"]["k"] == pytest.approx(1.75)

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, RTNQuantizeConfig)
    assert isinstance(reloaded.smooth, SmoothMAD)
    assert reloaded.smooth.k == pytest.approx(1.75)
    assert reloaded.uses_weight_only_lifecycle() is True
    assert reloaded.requires_calibration_dataset() is False


def test_rtn_quantize_config_defaults_to_no_smoother():
    cfg = RTNQuantizeConfig(bits=4, group_size=128)

    assert isinstance(cfg, BaseQuantizeConfig)
    assert not isinstance(cfg, GPTQQuantizeConfig)
    assert cfg.uses_weight_only_lifecycle() is True
    assert cfg.smooth is None
    assert cfg.export_quant_method() is not None

    payload = cfg.to_dict()
    assert payload["meta"]["weight_only"]["smooth"] is None

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, RTNQuantizeConfig)
    assert reloaded.smooth is None


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


def test_rtn_quantize_config_supports_gguf_export_round_trip():
    smooth = SmoothMAD(k=1.25)
    cfg = RTNQuantizeConfig(
        bits=4,
        group_size=128,
        format="gguf",
        smooth=smooth,
    )

    assert cfg.format == "gguf"
    assert cfg.export_quant_method().value == "gptq"

    payload = cfg.to_dict()
    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded, RTNQuantizeConfig)
    assert reloaded.format == cfg.format
    assert reloaded.export_quant_method() == cfg.export_quant_method()
    assert isinstance(reloaded.smooth, SmoothMAD)
    assert reloaded.smooth.k == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("bits", "gguf_qtype"),
    [
        (4, "q4_k_m"),
        (5, "q5_k_s"),
        (5, "q5_k_m"),
        (6, "q6_k"),
    ],
)
def test_rtn_quantize_config_supports_gguf_qtype_round_trip(bits: int, gguf_qtype: str):
    cfg = RTNQuantizeConfig(
        bits=bits,
        group_size=128,
        format="gguf",
        gguf_qtype=gguf_qtype,
        smooth=SmoothMAD(k=1.25),
    )

    payload = cfg.to_dict()
    assert payload["meta"]["weight_only"]["gguf_qtype"] == gguf_qtype

    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded, RTNQuantizeConfig)
    assert reloaded.format == "gguf"
    assert reloaded.bits == bits
    assert reloaded.gguf_qtype == gguf_qtype


def test_weight_only_payload_dispatches_to_rtn():
    cfg = QuantizeConfig(
        bits=4,
        group_size=128,
        weight_only={
            "method": "rtn",
            "smooth": {"type": "mad", "k": 2.0},
        },
    )

    assert isinstance(cfg, RTNQuantizeConfig)
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.smooth.k == pytest.approx(2.0)


def test_weight_only_payload_dispatches_to_rtn_gguf():
    cfg = QuantizeConfig(
        bits=4,
        group_size=128,
        weight_only={
            "method": "gguf",
            "smooth": {"type": "mad", "k": 1.5},
        },
    )

    assert isinstance(cfg, RTNQuantizeConfig)
    assert cfg.format == "gguf"
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.smooth.k == pytest.approx(1.5)


def test_weight_only_payload_dispatches_to_rtn_gguf_with_qtype():
    cfg = QuantizeConfig(
        bits=6,
        group_size=128,
        weight_only={
            "method": "gguf",
            "gguf_qtype": "q6_k",
        },
    )

    assert isinstance(cfg, RTNQuantizeConfig)
    assert cfg.format == "gguf"
    assert cfg.bits == 6
    assert cfg.gguf_qtype == "q6_k"
