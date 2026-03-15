from types import SimpleNamespace

from transformers import GPTNeoXConfig, LlamaConfig

from gptqmodel.utils import hf as hf_utils
from gptqmodel.utils.hf import normalize_hf_config_compat, resolve_trust_remote_code


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


def test_resolve_trust_remote_code_overrides_when_native_support_exists(monkeypatch, capsys):
    hf_utils._TRUST_REMOTE_CODE_OVERRIDE_WARNED.clear()
    monkeypatch.setattr(
        hf_utils,
        "_detect_native_transformers_causallm_support",
        lambda model_id_or_path: (True, "phi3", "Phi3ForCausalLM"),
    )

    resolved = resolve_trust_remote_code("/tmp/phi3", trust_remote_code=True)
    captured = capsys.readouterr()

    assert resolved is False
    assert "overriding trust_remote_code=True to False" in captured.out + captured.err


def test_resolve_trust_remote_code_keeps_true_without_native_support(monkeypatch, caplog):
    hf_utils._TRUST_REMOTE_CODE_OVERRIDE_WARNED.clear()
    monkeypatch.setattr(
        hf_utils,
        "_detect_native_transformers_causallm_support",
        lambda model_id_or_path: (False, None, None),
    )

    with caplog.at_level("WARNING"):
        resolved = resolve_trust_remote_code("/tmp/custom", trust_remote_code=True)

    assert resolved is True
    assert caplog.text == ""
