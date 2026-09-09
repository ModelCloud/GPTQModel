import json

import pytest

from gptqmodel.looper.qvq_processor import clone_qvq_config_for_module
from gptqmodel.quantization import GSQConfig, QuantizeConfig, QVQConfig


def test_gsq_default_and_json_roundtrip():
    assert QVQConfig().gsq is None
    assert not GSQConfig().enabled
    cfg = QVQConfig(format="qvq_v2b2_p32", gsq={"enabled": True, "modules": [r"self_attn\.[qkv]_proj$"]})
    restored = QuantizeConfig.from_quant_config(json.loads(json.dumps(cfg.to_dict())))
    assert restored.gsq == cfg.gsq
    assert restored.quant_linear_init_kwargs() == QVQConfig(format="qvq_v2b2_p32").quant_linear_init_kwargs()
    old = cfg.to_dict()
    old.pop("gsq")
    assert QuantizeConfig.from_quant_config(old).gsq is None


def test_gsq_dynamic_scope_includes_f6_nonbank_w4_modules():
    cfg = QVQConfig(format="qvq_v2b2_p32", gsq={"enabled": True, "modules": [r"self_attn\."]},
                    dynamic={r".*o_proj$": {"bits": 4, "format": "qvq"}})
    assert clone_qvq_config_for_module(cfg, "model.layers.0.self_attn.q_proj").gsq.enabled
    ordinary = clone_qvq_config_for_module(cfg, "model.layers.0.self_attn.o_proj")
    assert ordinary.bits == 4 and ordinary.gsq.enabled
    assert ordinary.bank_count == 1
    assert clone_qvq_config_for_module(cfg, "model.layers.0.mlp.gate_proj").gsq is None
    assert cfg.gsq.enabled


@pytest.mark.parametrize("bits", [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8])
def test_nonbank_gsq_config_roundtrip(bits):
    cfg = QVQConfig(bits=bits, format="qvq", gsq={"enabled": True}, offload_to_disk=False)
    assert QuantizeConfig.from_quant_config(cfg.to_dict()).gsq == cfg.gsq


@pytest.mark.parametrize("kwargs", [
    {"enabled": "false"}, {"steps": 0}, {"steps": True}, {"candidates": 1}, {"seed": -1},
    {"max_candidate_bytes": 0}, {"learning_rate": float("nan")}, {"temperature_end": 0},
    {"temperature_start": True}, {"modules": []}, {"modules": "q_proj"}, {"modules": [""]},
])
def test_gsq_invalid_controls(kwargs):
    with pytest.raises((TypeError, ValueError)):
        GSQConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"format": "qvq"}, {"rounding": "block_ldlq"},
    {"yaqa": {"spectral_refinement": True}},
    {"module_granular_replay": True},
])
def test_gsq_unsupported_combinations(kwargs):
    settings = {"format": "qvq_v2b2_p32", "gsq": {"enabled": True}} | kwargs
    with pytest.raises(ValueError, match="gsq"):
        QVQConfig(**settings)


def test_gsq_rejects_bare_boolean():
    with pytest.raises(TypeError, match="gsq"):
        QVQConfig(gsq=True)
