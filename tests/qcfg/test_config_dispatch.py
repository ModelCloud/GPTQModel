# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from gptqmodel.quantization.config import (
    AWQQuantizeConfig,
    BaseQuantizeConfig,
    FORMAT,
    GPTQQuantizeConfig,
    METHOD,
    QuantizeConfig,
    RTNQuantizeConfig,
    SmoothMAD,
    WeightOnlyConfig,
)


def test_quantize_config_dispatches_gptq_by_default():
    cfg = QuantizeConfig()

    assert isinstance(cfg, GPTQQuantizeConfig)
    assert cfg.quant_method == METHOD.GPTQ
    assert cfg.format == FORMAT.GPTQ


def test_quantize_config_dispatches_awq_constructor():
    cfg = QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, sym=False)

    assert isinstance(cfg, AWQQuantizeConfig)
    assert isinstance(cfg, QuantizeConfig)
    assert cfg.quant_method == METHOD.AWQ
    assert cfg.format == FORMAT.GEMM
    assert cfg.sym is False


def test_quantize_config_dispatches_awq_from_format_without_explicit_method():
    cfg = QuantizeConfig(format=FORMAT.GEMM, sym=False)

    assert isinstance(cfg, AWQQuantizeConfig)
    assert cfg.quant_method == METHOD.AWQ
    assert cfg.format == FORMAT.GEMM
    assert cfg.sym is False


def test_quantize_config_dispatches_awq_ignoring_legacy_gptq_only_kwargs():
    cfg = QuantizeConfig(
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        sym=False,
        act_group_aware=True,
        failsafe=None,
        damp_percent=0.05,
        mse=0.0,
    )

    assert isinstance(cfg, AWQQuantizeConfig)
    assert cfg.quant_method == METHOD.AWQ
    assert cfg.format == FORMAT.GEMM
    assert cfg.sym is False


def test_quantize_config_dispatches_rtn_constructor():
    cfg = QuantizeConfig(weight_only=WeightOnlyConfig(smooth=SmoothMAD(k=2.0)))

    assert isinstance(cfg, BaseQuantizeConfig)
    assert isinstance(cfg, RTNQuantizeConfig)
    assert not isinstance(cfg, GPTQQuantizeConfig)
    assert cfg.uses_weight_only_lifecycle() is True
    assert cfg.smooth is not None
    assert cfg.export_quant_method() == METHOD.GPTQ


def test_quantize_config_dispatches_rtn_awq_export_constructor():
    cfg = QuantizeConfig(
        format=FORMAT.GEMM,
        weight_only=WeightOnlyConfig(smooth=SmoothMAD(k=2.0)),
    )

    assert isinstance(cfg, RTNQuantizeConfig)
    assert cfg.format == FORMAT.GEMM
    assert cfg.export_quant_method() == METHOD.AWQ


def test_quantize_config_dispatches_rtn_gguf_export_constructor():
    cfg = QuantizeConfig(
        format=FORMAT.GGUF,
        weight_only=WeightOnlyConfig(smooth=SmoothMAD(k=2.0)),
    )

    assert isinstance(cfg, RTNQuantizeConfig)
    assert cfg.format == FORMAT.GGUF
    assert cfg.export_quant_method() == METHOD.GPTQ


def test_quantize_config_dispatches_rtn_from_gguf_weight_only_method():
    cfg = QuantizeConfig(
        weight_only=WeightOnlyConfig(method="gguf", smooth=SmoothMAD(k=1.5)),
    )

    assert isinstance(cfg, RTNQuantizeConfig)
    assert cfg.format == FORMAT.GGUF
    assert cfg.export_quant_method() == METHOD.GPTQ


def test_quantize_config_dispatches_rtn_from_gguf_weight_only_method_preserving_qtype():
    cfg = QuantizeConfig(
        bits=5,
        weight_only=WeightOnlyConfig(method="gguf", gguf_qtype="q5_k_m"),
    )

    assert isinstance(cfg, RTNQuantizeConfig)
    assert cfg.bits == 5
    assert cfg.format == FORMAT.GGUF
    assert cfg.gguf_qtype == "q5_k_m"
    assert cfg.export_quant_method() == METHOD.GPTQ


def test_quantize_config_dispatches_gptq_marlin_constructor():
    cfg = QuantizeConfig(quant_method=METHOD.GPTQ, format=FORMAT.MARLIN)

    assert isinstance(cfg, GPTQQuantizeConfig)
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

    assert isinstance(cfg, AWQQuantizeConfig)
    assert cfg.sym is False
