import copy
from types import SimpleNamespace

from gptqmodel.utils.hf import normalize_hf_config_compat


def test_nested_attention_rope_parameters_are_preserved():
    parameters = {
        "deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 80_000_000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 50_000.0},
    }
    config = SimpleNamespace(
        model_type="dots3_note", layer_types=list(parameters) * 2,
        rope_parameters=copy.deepcopy(parameters), rope_theta=80_000_000.0,
    )
    normalize_hf_config_compat(config)
    assert config.rope_parameters == parameters


def test_flat_legacy_rope_parameters_still_receive_defaults():
    config = SimpleNamespace(model_type="legacy", rope_parameters=None, rope_theta=50_000.0)
    normalize_hf_config_compat(config)
    assert config.rope_parameters == {"rope_type": "default", "rope_theta": 50_000.0}
