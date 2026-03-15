from types import SimpleNamespace

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


def test_normalize_hf_config_compat_drops_default_remote_rope_scaling_dict():
    config = SimpleNamespace(rope_scaling={"rope_type": "default", "rope_theta": 10000.0})

    normalize_hf_config_compat(config, trust_remote_code=True)

    assert config.rope_scaling is None


def test_normalize_hf_config_compat_preserves_rope_parameters_after_remote_cleanup():
    config = LlamaConfig(rope_scaling={"rope_type": "default", "rope_theta": 10000.0})

    normalize_hf_config_compat(config, trust_remote_code=True)

    assert config.rope_parameters["rope_type"] == "default"
    assert config.rope_parameters["rope_theta"] == 10000.0


def test_normalize_hf_config_compat_reapplies_remote_rope_migration_after_sync():
    class SyncConfig:
        def __init__(self):
            self.rope_scaling = {"rope_type": "default", "rope_theta": 10000.0}
            self.rope_parameters = None
            self.rope_theta = 10000.0

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "rope_parameters" and isinstance(value, dict):
                mirrored = {
                    "rope_type": value.get("rope_type"),
                    "rope_theta": value.get("rope_theta"),
                }
                object.__setattr__(self, "rope_scaling", mirrored)

    config = SyncConfig()

    normalize_hf_config_compat(config, trust_remote_code=True)

    assert config.rope_parameters["rope_type"] == "default"
    assert config.rope_scaling is None
