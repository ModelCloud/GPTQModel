# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.quantization.config import (
    BaseQuantizeConfig,
    GGUFQuantizeConfig,
    GGUFBits,
    GPTQQuantizeConfig,
    QuantizeConfig,
    RTNQuantizeConfig,
    SmoothMAD,
    SmootherConfig,
)


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


def test_gguf_quantize_config_round_trip():
    smooth = SmoothMAD(k=1.25)
    cfg = GGUFQuantizeConfig(
        bits=4,
        smoother=smooth,
    )

    assert cfg.format == "gguf"
    assert cfg.uses_weight_only_lifecycle() is True
    assert cfg.requires_calibration_dataset() is False
    assert isinstance(cfg.bits, GGUFBits)
    assert cfg.bits == "q4_0"
    assert cfg.bits.bits == 4
    assert cfg.bits.version == "q"
    assert cfg.bits.variant == "0"
    assert cfg.bits.quality is None
    assert cfg.group_size == -1
    assert cfg.desc_act is False
    assert cfg.export_quant_method().value == "gptq"
    assert isinstance(cfg.smoother, SmootherConfig)
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.smooth.k == pytest.approx(1.25)

    payload = cfg.to_dict()
    assert payload["bits"] == "q4_0"
    assert "group_size" not in payload
    assert "desc_act" not in payload
    assert "quant_method" not in payload
    assert "pack_dtype" not in payload
    assert "weight_only" not in payload["meta"]
    assert payload["meta"]["pre_filters"][0]["code"] == "smoother"
    assert payload["meta"]["pre_filters"][0]["smooth"]["type"] == "mad"
    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded, GGUFQuantizeConfig)
    assert reloaded.format == cfg.format
    assert reloaded.bits == "q4_0"
    assert reloaded.export_quant_method() == cfg.export_quant_method()
    assert isinstance(reloaded.smooth, SmoothMAD)
    assert reloaded.smooth.k == pytest.approx(1.25)


def test_gguf_quantize_config_hides_non_gguf_constructor_args():
    with pytest.raises(TypeError):
        GGUFQuantizeConfig(bits="q4_k_m", group_size=128)

    with pytest.raises(TypeError):
        GGUFQuantizeConfig(bits="q4_k_m", desc_act=True)


def test_gguf_quantize_config_registers_smoother_prefilter():
    cfg = GGUFQuantizeConfig(
        bits="q4_k_m",
        pre_filters=[SmootherConfig(smooth=SmoothMAD(k=1.9))],
    )

    assert isinstance(cfg.smoother, SmootherConfig)
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.smooth.k == pytest.approx(1.9)
    assert len(cfg.pre_filters) == 1
    assert cfg.pre_filters[0].code == "smoother"


@pytest.mark.parametrize(
    ("bits", "bit_width", "variant", "quality"),
    [
        ("q4_k_m", 4, "k", "m"),
        ("q5_k_s", 5, "k", "s"),
        ("q5_k_m", 5, "k", "m"),
        ("q6_k", 6, "k", None),
    ],
)
def test_rtn_quantize_config_supports_gguf_bits_round_trip(
    bits: str,
    bit_width: int,
    variant: str,
    quality: str | None,
):
    cfg = GGUFQuantizeConfig(
        bits=bits,
        smoother=SmoothMAD(k=1.25),
    )

    payload = cfg.to_dict()
    assert payload["bits"] == bits
    assert cfg.bits.bits == bit_width
    assert cfg.bits.version == "q"
    assert cfg.bits.variant == variant
    assert cfg.bits.quality == quality

    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded, GGUFQuantizeConfig)
    assert reloaded.format == "gguf"
    assert reloaded.bits == bits
    assert reloaded.bits.bits == bit_width
    assert reloaded.bits.version == "q"
    assert reloaded.bits.variant == variant
    assert reloaded.bits.quality == quality


def test_rtn_quantize_config_supports_structured_gguf_bits_round_trip():
    cfg = GGUFQuantizeConfig(
        bits=GGUFBits(bits=4, version="q", variant="k", quality="s"),
        smoother=SmoothMAD(k=1.25),
    )

    assert isinstance(cfg.bits, GGUFBits)
    assert cfg.bits == "q4_k_s"
    assert cfg.bits.bits == 4
    assert cfg.bits.version == "q"
    assert cfg.bits.variant == "k"
    assert cfg.bits.quality == "s"

    payload = cfg.to_dict()
    assert payload["bits"] == "q4_k_s"

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert reloaded.bits == "q4_k_s"
    assert isinstance(reloaded.bits, GGUFBits)


def test_gguf_bits_string_parser_round_trip():
    bits = GGUFBits.from_string("q4_k_s")

    assert isinstance(bits, GGUFBits)
    assert bits.bits == 4
    assert bits.version == "q"
    assert bits.variant == "k"
    assert bits.quality == "s"
    assert bits.to_string() == "q4_k_s"
    assert str(bits) == "q4_k_s"
    assert int(bits) == 4


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
        weight_only={
            "method": "gguf",
            "smooth": {"type": "mad", "k": 1.5},
        },
    )

    assert isinstance(cfg, GGUFQuantizeConfig)
    assert cfg.format == "gguf"
    assert cfg.bits == "q4_0"
    assert cfg.bits.bits == 4
    assert cfg.bits.version == "q"
    assert cfg.bits.variant == "0"
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.smooth.k == pytest.approx(1.5)


def test_weight_only_payload_dispatches_to_rtn_gguf_with_qtype():
    cfg = QuantizeConfig(
        bits="q6_k",
        weight_only={
            "method": "gguf",
        },
    )

    assert isinstance(cfg, GGUFQuantizeConfig)
    assert cfg.format == "gguf"
    assert cfg.bits == "q6_k"
    assert cfg.bits.bits == 6
    assert cfg.bits.version == "q"
    assert cfg.bits.variant == "k"
    assert cfg.bits.quality is None


def test_weight_only_payload_dispatches_legacy_gguf_qtype_to_bits():
    cfg = QuantizeConfig(
        bits=6,
        weight_only={
            "method": "gguf",
            "gguf_qtype": "q6_k",
        },
    )

    assert isinstance(cfg, GGUFQuantizeConfig)
    assert cfg.format == "gguf"
    assert cfg.bits == "q6_k"
    assert cfg.bits.bits == 6
    assert cfg.bits.version == "q"
    assert cfg.bits.variant == "k"
