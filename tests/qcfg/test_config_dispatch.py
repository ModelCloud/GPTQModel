# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.quantization.config import (
    FORMAT,
    METHOD,
    AWQConfig,
    BaseQuantizeConfig,
    BitsAndBytesConfig,
    FP8Config,
    GGUFBits,
    GGUFConfig,
    GPTQConfig,
    QuantizeConfig,
    RTNConfig,
    SmoothMAD,
    WeightOnlyConfig,
)
from gptqmodel.quantization.dtype import available_float8_dtype_names


def _fp8_alias_cases():
    cases = [
        ("e4m3", "float8_e4m3fn"),
        ("e5m2", "float8_e5m2"),
        ("e4m3fnuz", "float8_e4m3fnuz"),
        ("e5m2fnuz", "float8_e5m2fnuz"),
        ("e8m0", "float8_e8m0fnu"),
        ("float8_e8m0", "float8_e8m0fnu"),
    ]
    available = set(available_float8_dtype_names())
    return [(alias, expected) for alias, expected in cases if expected in available]


def test_quantize_config_dispatches_gptq_by_default():
    cfg = QuantizeConfig()

    assert isinstance(cfg, GPTQConfig)
    assert cfg.quant_method == METHOD.GPTQ
    assert cfg.format == FORMAT.GPTQ


def test_quantize_config_dispatches_awq_constructor():
    cfg = QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, sym=False)

    assert isinstance(cfg, AWQConfig)
    assert isinstance(cfg, QuantizeConfig)
    assert cfg.quant_method == METHOD.AWQ
    assert cfg.format == FORMAT.GEMM
    assert cfg.sym is False


def test_quantize_config_dispatches_awq_with_canonical_method_field():
    cfg = QuantizeConfig(method=METHOD.AWQ, format=FORMAT.GEMM, sym=False)

    assert isinstance(cfg, AWQConfig)
    assert cfg.method == METHOD.AWQ
    assert cfg.quant_method == METHOD.AWQ
    assert cfg.format == FORMAT.GEMM
    assert cfg.sym is False


def test_quantize_config_dispatches_awq_from_format_without_explicit_method():
    cfg = QuantizeConfig(format=FORMAT.GEMM, sym=False)

    assert isinstance(cfg, AWQConfig)
    assert cfg.quant_method == METHOD.AWQ
    assert cfg.format == FORMAT.GEMM
    assert cfg.sym is False


def test_quantize_config_dispatches_awq_ignoring_legacy_gptq_only_kwargs():
    cfg = QuantizeConfig(
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        sym=False,
        act_group_aware=True,
        fallback=None,
        damp_percent=0.05,
        mse=0.0,
    )

    assert isinstance(cfg, AWQConfig)
    assert cfg.quant_method == METHOD.AWQ
    assert cfg.format == FORMAT.GEMM
    assert cfg.sym is False


def test_quantize_config_rejects_is_marlin_format_constructor_arg():
    with pytest.raises(ValueError, match="is_marlin_format"):
        QuantizeConfig(is_marlin_format=True)


def test_quantize_config_rejects_is_marlin_format_in_serialized_payload():
    with pytest.raises(ValueError, match="is_marlin_format"):
        QuantizeConfig.from_quant_config(
            {
                "bits": 4,
                "is_marlin_format": True,
            }
        )


def test_quantize_config_dispatches_rtn_constructor():
    cfg = QuantizeConfig(weight_only=WeightOnlyConfig(smooth=SmoothMAD(k=2.0)))

    assert isinstance(cfg, BaseQuantizeConfig)
    assert isinstance(cfg, RTNConfig)
    assert not isinstance(cfg, GPTQConfig)
    assert cfg.uses_weight_only_lifecycle() is True
    assert cfg.smooth is not None
    assert cfg.export_quant_method() == METHOD.GPTQ


def test_quantize_config_dispatches_rtn_awq_export_constructor():
    cfg = QuantizeConfig(
        format=FORMAT.GEMM,
        weight_only=WeightOnlyConfig(smooth=SmoothMAD(k=2.0)),
    )

    assert isinstance(cfg, RTNConfig)
    assert cfg.format == FORMAT.GEMM
    assert cfg.export_quant_method() == METHOD.AWQ


def test_quantize_config_dispatches_rtn_gguf_export_constructor():
    cfg = QuantizeConfig(
        format=FORMAT.GGUF,
    )

    assert isinstance(cfg, GGUFConfig)
    assert cfg.quant_method == METHOD.GGUF
    assert cfg.format == "q_0"
    assert cfg.bits == 4
    assert isinstance(cfg.runtime_bits, GGUFBits)
    assert cfg.runtime_bits == "q4_0"
    assert cfg.runtime_bits.bits == 4
    assert cfg.runtime_bits.version == "q"
    assert cfg.runtime_bits.variant == "0"
    assert cfg.export_quant_method() == METHOD.GGUF


def test_quantize_config_dispatches_rtn_from_gguf_weight_only_method():
    cfg = QuantizeConfig(
        weight_only=WeightOnlyConfig(method="gguf", smooth=SmoothMAD(k=1.5)),
    )

    assert isinstance(cfg, GGUFConfig)
    assert cfg.quant_method == METHOD.GGUF
    assert cfg.format == "q_0"
    assert cfg.bits == 4
    assert isinstance(cfg.runtime_bits, GGUFBits)
    assert cfg.runtime_bits == "q4_0"
    assert cfg.runtime_bits.bits == 4
    assert cfg.runtime_bits.variant == "0"
    assert cfg.export_quant_method() == METHOD.GGUF


def test_quantize_config_dispatches_rtn_from_gguf_weight_only_method_preserving_qtype():
    cfg = QuantizeConfig(
        bits="q5_k_m",
        weight_only=WeightOnlyConfig(method="gguf"),
    )

    assert isinstance(cfg, GGUFConfig)
    assert cfg.quant_method == METHOD.GGUF
    assert cfg.bits == 5
    assert isinstance(cfg.runtime_bits, GGUFBits)
    assert cfg.runtime_bits == "q5_k_m"
    assert cfg.runtime_bits.bits == 5
    assert cfg.runtime_bits.version == "q"
    assert cfg.runtime_bits.variant == "k"
    assert cfg.runtime_bits.quality == "m"
    assert cfg.format == "q_k_m"
    assert cfg.export_quant_method() == METHOD.GGUF


def test_quantize_config_dispatches_fp8_constructor():
    cfg = QuantizeConfig(
        quant_method=METHOD.FP8,
        format="float8_e5m2",
        weight_scale_method="row",
    )

    assert isinstance(cfg, FP8Config)
    assert cfg.quant_method == METHOD.FP8
    assert cfg.format == "float8_e5m2"
    assert cfg.weight_scale_method == "row"
    assert cfg.uses_weight_only_lifecycle() is True


def test_quantize_config_dispatches_fp8_from_weight_only_method():
    cfg = QuantizeConfig(
        weight_only=WeightOnlyConfig(method="fp8", smooth=SmoothMAD(k=1.5)),
        weight_scale_method="block",
        weight_block_size=[128, 128],
    )

    assert isinstance(cfg, FP8Config)
    assert cfg.quant_method == METHOD.FP8
    assert cfg.format == "float8_e4m3fn"
    assert cfg.weight_scale_method == "block"
    assert cfg.weight_block_size == [128, 128]
    assert cfg.smooth is not None


@pytest.mark.parametrize(("format_value", "expected"), _fp8_alias_cases())
def test_quantize_config_normalizes_all_supported_fp8_aliases(format_value: str, expected: str):
    cfg = QuantizeConfig(
        quant_method=METHOD.FP8,
        format=format_value,
    )

    assert isinstance(cfg, FP8Config)
    assert cfg.format == expected


def test_quantize_config_dispatches_bitsandbytes_constructor():
    cfg = QuantizeConfig(
        quant_method=METHOD.BITSANDBYTES,
        bits=8,
    )

    assert isinstance(cfg, BitsAndBytesConfig)
    assert cfg.quant_method == METHOD.BITSANDBYTES
    assert cfg.format == "int8"
    assert cfg.bits == 8
    assert cfg.uses_weight_only_lifecycle() is True


def test_quantize_config_dispatches_bitsandbytes_from_weight_only_method():
    cfg = QuantizeConfig(
        bits=4,
        weight_only=WeightOnlyConfig(method="bitsandbytes", smooth=SmoothMAD(k=1.25)),
        format="nf4",
        block_size=128,
        compress_statistics=False,
    )

    assert isinstance(cfg, BitsAndBytesConfig)
    assert cfg.quant_method == METHOD.BITSANDBYTES
    assert cfg.format == "nf4"
    assert cfg.bits == 4
    assert cfg.block_size == 128
    assert cfg.compress_statistics is False
    assert cfg.smooth is not None


def test_quantize_config_dispatches_gptq_marlin_constructor():
    cfg = QuantizeConfig(quant_method=METHOD.GPTQ, format=FORMAT.MARLIN)

    assert isinstance(cfg, GPTQConfig)
    assert cfg.export_quant_method() == METHOD.GPTQ


def test_from_quant_config_dispatches_awq_and_loads_zero_point():
    cfg = QuantizeConfig.from_quant_config(
        {
            "bits": 4,
            "group_size": 128,
            "quant_method": "awq",
            "format": "gemm",
            "zero_point": True,
        }
    )

    assert isinstance(cfg, AWQConfig)
    assert cfg.sym is False
