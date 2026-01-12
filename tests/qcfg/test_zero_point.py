
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig


def test_awq_zero_point_load_normalization():
    payload = {
        "bits": 4,
        "group_size": 128,
        "quant_method": "awq",
        "format": "gemm",
        "zero_point": True,
    }

    cfg = QuantizeConfig.from_quant_config(payload)

    assert cfg.sym is False


def test_awq_zero_point_overrides_sym():
    payload = {
        "bits": 4,
        "group_size": 128,
        "quant_method": "awq",
        "format": "gemm",
        "sym": True,
        "zero_point": True,
    }

    cfg = QuantizeConfig.from_quant_config(payload)

    assert cfg.sym is False


def test_awq_to_dict_uses_zero_point():
    cfg = QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, sym=False)
    payload = cfg.to_dict()

    assert payload["zero_point"] is True
    assert "sym" not in payload


def test_gptq_to_dict_uses_sym():
    cfg = QuantizeConfig(quant_method=METHOD.GPTQ, format=FORMAT.GPTQ, sym=False)
    payload = cfg.to_dict()

    assert payload["sym"] is False
    assert "zero_point" not in payload
