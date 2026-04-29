# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
from dataclasses import fields
from inspect import signature

import pytest
import torch

from gptqmodel.quantization.config import (
    METHOD,
    AutoModuleDecoderConfig,
    BaseQuantizeConfig,
    BitsAndBytesConfig,
    GGUFBits,
    GGUFConfig,
    GPTQConfig,
    QuantizeConfig,
    RTNConfig,
    SmootherConfig,
    SmoothMAD,
    TensorParallelPadderConfig,
)


def test_quantize_config_weight_only_round_trip():
    smooth = SmoothMAD(k=1.75)
    cfg = RTNConfig(
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
    assert isinstance(reloaded, RTNConfig)
    assert isinstance(reloaded.smooth, SmoothMAD)
    assert reloaded.smooth.k == pytest.approx(1.75)
    assert reloaded.uses_weight_only_lifecycle() is True
    assert reloaded.requires_calibration_dataset() is False


def test_rtn_quantize_config_defaults_to_no_smoother():
    cfg = RTNConfig(bits=4, group_size=128)

    assert isinstance(cfg, BaseQuantizeConfig)
    assert not isinstance(cfg, GPTQConfig)
    assert cfg.uses_weight_only_lifecycle() is True
    assert cfg.smooth is None
    assert cfg.export_quant_method() is not None

    payload = cfg.to_dict()
    assert payload["meta"]["weight_only"]["smooth"] is None

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, RTNConfig)
    assert reloaded.smooth is None


def test_rtn_quantize_config_supports_awq_export_round_trip():
    smooth = SmoothMAD(k=1.5)
    cfg = RTNConfig(
        bits=4,
        group_size=128,
        format="gemm",
        smooth=smooth,
    )

    assert cfg.format == "gemm"
    assert cfg.export_quant_method().value == "awq"

    payload = cfg.to_dict()
    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded, RTNConfig)
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
    assert payload["method"] == "gguf"
    assert payload["quant_method"] == "gguf"
    assert payload["format"] == "q_0"
    assert payload["checkpoint_format"] == "q_0"
    assert "group_size" not in payload
    assert "desc_act" not in payload
    assert "pack_dtype" not in payload
    assert "weight_only" not in payload["meta"]
    assert payload["meta"]["preprocessors"][0]["code"] == "smoother"
    assert payload["meta"]["preprocessors"][0]["smooth"]["type"] == "mad"
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


def test_bitsandbytes_quantize_config_round_trip_4bit():
    cfg = BitsAndBytesConfig(
        bits=4,
        format="nf4",
        block_size=128,
        compress_statistics=False,
        smoother=SmoothMAD(k=1.2),
    )

    assert cfg.quant_method == METHOD.BITSANDBYTES
    assert cfg.format == "nf4"
    assert cfg.bits == 4
    assert cfg.block_size == 128
    assert cfg.compress_statistics is False
    assert cfg.uses_weight_only_lifecycle() is True

    payload = cfg.to_dict()
    assert payload["method"] == "bitsandbytes"
    assert payload["quant_method"] == "bitsandbytes"
    assert payload["format"] == "nf4"
    assert payload["checkpoint_format"] == "nf4"
    assert payload["block_size"] == 128
    assert payload["compress_statistics"] is False

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, BitsAndBytesConfig)
    assert reloaded.bits == 4
    assert reloaded.format == "nf4"
    assert reloaded.block_size == 128
    assert reloaded.compress_statistics is False
    assert isinstance(reloaded.smooth, SmoothMAD)


def test_bitsandbytes_quantize_config_round_trip_8bit():
    cfg = BitsAndBytesConfig(bits=8)

    assert cfg.quant_method == METHOD.BITSANDBYTES
    assert cfg.format == "int8"
    assert cfg.bits == 8
    assert cfg.uses_weight_only_lifecycle() is True

    payload = cfg.to_dict()
    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded, BitsAndBytesConfig)
    assert reloaded.bits == 8
    assert reloaded.format == "int8"
    assert reloaded.quant_method == METHOD.BITSANDBYTES


def test_gguf_config_registers_smoother_preprocessor():
    cfg = GGUFConfig(
        bits=4,
        format="q_k_m",
        preprocessors=[SmootherConfig(smooth=SmoothMAD(k=1.9))],
    )

    assert isinstance(cfg.smoother, SmootherConfig)
    assert isinstance(cfg.smooth, SmoothMAD)
    assert cfg.smooth.k == pytest.approx(1.9)
    assert len(cfg.preprocessors) == 1
    assert cfg.preprocessors[0].code == "smoother"


def test_gguf_config_does_not_auto_register_tensor_parallel_padder():
    cfg = GGUFConfig(bits=4, format="q_k_m")

    assert cfg.preprocessors == []


def test_gguf_config_registers_auto_module_decoder_preprocessor():
    cfg = GGUFConfig(
        bits=4,
        format="q_k_m",
        preprocessors=[
            AutoModuleDecoderConfig(target_dtype=torch.float16)
        ],
    )

    assert len(cfg.preprocessors) == 1
    decoder = cfg.preprocessors[0]
    assert isinstance(decoder, AutoModuleDecoderConfig)
    assert decoder.code == "auto_module_decoder"
    assert decoder.source_dtype == "auto"
    assert decoder.target_dtype == torch.float16

    payload = cfg.to_dict()
    assert payload["meta"]["preprocessors"][0]["code"] == "auto_module_decoder"
    assert payload["meta"]["preprocessors"][0]["target_dtype"] == "float16"

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded.preprocessors[0], AutoModuleDecoderConfig)
    assert reloaded.preprocessors[0].target_dtype == torch.float16


def test_gguf_config_registers_tensor_parallel_padder_preprocessor():
    cfg = GGUFConfig(
        bits=4,
        format="q_k_m",
        preprocessors=[TensorParallelPadderConfig()],
    )

    assert len(cfg.preprocessors) == 1
    padder = cfg.preprocessors[0]
    assert isinstance(padder, TensorParallelPadderConfig)
    assert padder.code == "tensor_parallel_padder"

    payload = cfg.to_dict()
    assert payload["meta"]["preprocessors"][0]["code"] == "tensor_parallel_padder"

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded.preprocessors[0], TensorParallelPadderConfig)


def test_auto_module_decoder_config_does_not_expose_code_as_init_field():
    decoder_fields = {field.name for field in fields(AutoModuleDecoderConfig)}

    assert "code" not in signature(AutoModuleDecoderConfig).parameters
    assert "code" not in decoder_fields
    assert AutoModuleDecoderConfig().to_dict()["code"] == "auto_module_decoder"


def test_tensor_parallel_padder_config_does_not_expose_code_as_init_field():
    padder_fields = {field.name for field in fields(TensorParallelPadderConfig)}

    assert "code" not in signature(TensorParallelPadderConfig).parameters
    assert "code" not in padder_fields
    assert TensorParallelPadderConfig().to_dict()["code"] == "tensor_parallel_padder"


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
    assert reloaded.quant_method == METHOD.GGUF
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

    assert isinstance(cfg, RTNConfig)
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
