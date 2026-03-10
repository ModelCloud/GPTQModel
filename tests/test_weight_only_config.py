# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.quantization.config import (
    BaseQuantizeConfig,
    FORMAT,
    GGUFConfig,
    GGUFQuantizeConfig,
    GGUFBits,
    GPTQQuantizeConfig,
    METHOD,
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
    cfg = GGUFConfig(
        bits=4,
        smoother=smooth,
    )

    assert cfg.format == "q_0"
    assert cfg.checkpoint_format == FORMAT.GGUF
    assert cfg.uses_weight_only_lifecycle() is True
    assert cfg.requires_calibration_dataset() is False
    assert cfg.bits == 4
    assert isinstance(cfg.runtime_bits, GGUFBits)
    assert cfg.runtime_bits == "q4_0"
    assert cfg.runtime_bits.bits == 4
    assert cfg.runtime_bits.version == "q"
    assert cfg.runtime_bits.variant == "0"
    assert cfg.runtime_bits.quality is None
    assert cfg.group_size == -1
    assert cfg.desc_act is False
    assert cfg.quant_method == METHOD.GGUF
    assert cfg.export_quant_method() == METHOD.GGUF
    assert isinstance(cfg.smoother, SmootherConfig)
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.smooth.k == pytest.approx(1.25)

    payload = cfg.to_dict()
    assert payload["bits"] == 4
    assert payload["format"] == "q_0"
    assert payload["checkpoint_format"] == "gguf"
    assert "group_size" not in payload
    assert "desc_act" not in payload
    assert "quant_method" not in payload
    assert "pack_dtype" not in payload
    assert "weight_only" not in payload["meta"]
    assert payload["meta"]["pre_filters"][0]["code"] == "smoother"
    assert payload["meta"]["pre_filters"][0]["smooth"]["type"] == "mad"
    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded, GGUFConfig)
    assert reloaded.format == cfg.format
    assert reloaded.bits == 4
    assert reloaded.runtime_bits == "q4_0"
    assert reloaded.quant_method == METHOD.GGUF
    assert reloaded.export_quant_method() == cfg.export_quant_method()
    assert isinstance(reloaded.smooth, SmoothMAD)
    assert reloaded.smooth.k == pytest.approx(1.25)


def test_gguf_quantize_config_hides_non_gguf_constructor_args():
    with pytest.raises(TypeError):
        GGUFConfig(bits=4, format="q_k_m", group_size=128)

    with pytest.raises(TypeError):
        GGUFConfig(bits=4, format="q_k_m", desc_act=True)


def test_gguf_quantize_config_registers_smoother_prefilter():
    cfg = GGUFConfig(
        bits=4,
        format="q_k_m",
        pre_filters=[SmootherConfig(smooth=SmoothMAD(k=1.9))],
    )

    assert isinstance(cfg.smoother, SmootherConfig)
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.smooth.k == pytest.approx(1.9)
    assert len(cfg.pre_filters) == 1
    assert cfg.pre_filters[0].code == "smoother"


@pytest.mark.parametrize(
    ("bits", "format", "bit_width", "variant", "quality"),
    [
        (4, "q_k_m", 4, "k", "m"),
        (5, "q_k_s", 5, "k", "s"),
        (5, "q_k_m", 5, "k", "m"),
        (6, "q_k", 6, "k", None),
    ],
)
def test_rtn_quantize_config_supports_gguf_bits_round_trip(
    bits: int,
    format: str,
    bit_width: int,
    variant: str,
    quality: str | None,
):
    cfg = GGUFConfig(
        bits=bits,
        format=format,
        smoother=SmoothMAD(k=1.25),
    )

    payload = cfg.to_dict()
    assert payload["bits"] == bit_width
    assert payload["format"] == format
    assert cfg.bits == bit_width
    assert cfg.runtime_bits.bits == bit_width
    assert cfg.runtime_bits.version == "q"
    assert cfg.runtime_bits.variant == variant
    assert cfg.runtime_bits.quality == quality

    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded, GGUFConfig)
    assert reloaded.checkpoint_format == FORMAT.GGUF
    assert reloaded.format == format
    assert reloaded.bits == bit_width
    assert reloaded.runtime_bits.bits == bit_width
    assert reloaded.runtime_bits.version == "q"
    assert reloaded.runtime_bits.variant == variant
    assert reloaded.runtime_bits.quality == quality


def test_rtn_quantize_config_supports_structured_gguf_bits_round_trip():
    cfg = GGUFConfig(
        bits=GGUFBits(bits=4, version="q", variant="k", quality="s"),
        smoother=SmoothMAD(k=1.25),
    )

    assert cfg.bits == 4
    assert cfg.format == "q_k_s"
    assert isinstance(cfg.runtime_bits, GGUFBits)
    assert cfg.runtime_bits == "q4_k_s"
    assert cfg.runtime_bits.bits == 4
    assert cfg.runtime_bits.version == "q"
    assert cfg.runtime_bits.variant == "k"
    assert cfg.runtime_bits.quality == "s"

    payload = cfg.to_dict()
    assert payload["bits"] == 4
    assert payload["format"] == "q_k_s"

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert reloaded.bits == 4
    assert reloaded.format == "q_k_s"
    assert reloaded.runtime_bits == "q4_k_s"


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

    assert isinstance(cfg, GGUFConfig)
    assert cfg.format == "q_0"
    assert cfg.bits == 4
    assert cfg.runtime_bits == "q4_0"
    assert cfg.runtime_bits.bits == 4
    assert cfg.runtime_bits.version == "q"
    assert cfg.runtime_bits.variant == "0"
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.smooth.k == pytest.approx(1.5)


def test_weight_only_payload_dispatches_to_rtn_gguf_with_qtype():
    cfg = QuantizeConfig(
        bits="q6_k",
        weight_only={
            "method": "gguf",
        },
    )

    assert isinstance(cfg, GGUFConfig)
    assert cfg.format == "q_k"
    assert cfg.bits == 6
    assert cfg.runtime_bits == "q6_k"
    assert cfg.runtime_bits.bits == 6
    assert cfg.runtime_bits.version == "q"
    assert cfg.runtime_bits.variant == "k"
    assert cfg.runtime_bits.quality is None


def test_weight_only_payload_dispatches_legacy_gguf_qtype_to_bits():
    cfg = QuantizeConfig(
        bits=6,
        weight_only={
            "method": "gguf",
            "gguf_qtype": "q6_k",
        },
    )

    assert isinstance(cfg, GGUFConfig)
    assert cfg.format == "q_k"
    assert cfg.bits == 6
    assert cfg.runtime_bits == "q6_k"
    assert cfg.runtime_bits.bits == 6
    assert cfg.runtime_bits.version == "q"
    assert cfg.runtime_bits.variant == "k"
