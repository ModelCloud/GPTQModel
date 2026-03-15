from transformers import GPTNeoXConfig, LlamaConfig

from gptqmodel.utils.hf import normalize_hf_config_compat


def test_normalize_hf_config_compat_backfills_llama_rope_parameters():
    config = LlamaConfig(rope_parameters=None, rope_theta=12345.0)

    normalize_hf_config_compat(config)

    assert config.rope_parameters["rope_type"] == "default"
    assert config.rope_parameters["rope_theta"] == 12345.0


def test_normalize_hf_config_compat_uses_gpt_neox_defaults():
    config = GPTNeoXConfig(rope_parameters=None)

    normalize_hf_config_compat(config)

    assert config.rope_parameters["rope_type"] == "default"
    assert config.rope_parameters["rope_theta"] == config.default_theta
    assert config.rope_parameters["partial_rotary_factor"] == 0.25
