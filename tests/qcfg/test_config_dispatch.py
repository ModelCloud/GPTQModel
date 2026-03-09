# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from gptqmodel.quantization.config import (
    AWQQuantizeConfig,
    BaseQuantizeConfig,
    CalibrationlessConfig,
    FORMAT,
    GPTQQuantizeConfig,
    METHOD,
    QuantizeConfig,
    RTNQuantizeConfig,
    SmoothMAD,
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


def test_quantize_config_dispatches_rtn_constructor():
    cfg = QuantizeConfig(calibrationless=CalibrationlessConfig(smooth=SmoothMAD(k=2.0)))

    assert isinstance(cfg, BaseQuantizeConfig)
    assert isinstance(cfg, RTNQuantizeConfig)
    assert not isinstance(cfg, GPTQQuantizeConfig)
    assert cfg.uses_calibrationless_lifecycle() is True
    assert cfg.smooth is not None
    assert cfg.export_quant_method() == METHOD.GPTQ


def test_quantize_config_dispatches_rtn_awq_export_constructor():
    cfg = QuantizeConfig(
        format=FORMAT.GEMM,
        calibrationless=CalibrationlessConfig(smooth=SmoothMAD(k=2.0)),
    )

    assert isinstance(cfg, RTNQuantizeConfig)
    assert cfg.format == FORMAT.GEMM
    assert cfg.export_quant_method() == METHOD.AWQ


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
