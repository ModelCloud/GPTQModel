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


def test_qvq_paper_schedule_has_llama_twenty_epoch_update_budget():
    cfg = GSQConfig.for_qvq_paper_schedule()
    assert cfg.enabled and cfg.steps == 1280 and cfg.qvq_relaxation_patience == 0
    assert cfg.qvq_hard_eval_interval == 64
    assert cfg.qvq_cuda_graph_updates_per_replay == 4
    assert cfg.qvq_soft_dtype == "bfloat16"
    assert cfg.qvq_coordinate_sweeps == 0
    assert GSQConfig.for_qvq_paper_schedule(num_samples=101, batch_size=64, epochs=3).steps == 6
    with pytest.raises(ValueError, match="computes steps"):
        GSQConfig.for_qvq_paper_schedule(steps=10)


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
    {"qvq_learning_rate": 0}, {"qvq_kappa_start": float("nan")}, {"qvq_weight_decay": -1},
    {"qvq_initialization_std": 0}, {"qvq_initialization_strength": -1},
    {"qvq_gumbel_samples": 0}, {"qvq_coordinate_sweeps": -1},
    {"qvq_coordinate_chunk_tiles": 0}, {"qvq_hard_eval_interval": 0},
    {"qvq_cuda_graph_updates_per_replay": 0},
    {"qvq_relaxation_patience": -1},
    {"qvq_soft_dtype": "float16"},
    {"qvq_candidate_policy": "bits"},
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


def test_low_level_qvq_rejects_scale_learning():
    import torch
    from gptqmodel.quantization.qvq import quantize_qvq_linear
    from gptqmodel.quantization.qvq_gsq import refine_trellis_fisher

    config = GSQConfig(enabled=True, learn_scales=True)
    with pytest.raises(ValueError, match="scale learning"):
        quantize_qvq_linear(torch.eye(16), torch.eye(16), bits=2.5, gsq=config)
    with pytest.raises(ValueError, match="scale learning"):
        refine_trellis_fisher(None, target=torch.eye(16), input_hessian=torch.eye(16),
                              output_hessian=torch.eye(16), config=config, bits=2.5)
