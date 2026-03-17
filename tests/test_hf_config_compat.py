import sys
from types import SimpleNamespace

import torch
from transformers import GPTNeoXConfig, LlamaConfig

from gptqmodel.utils import hf as hf_utils
from transformers import cache_utils

from gptqmodel.utils.hf import (
    normalize_hf_config_compat,
    prepare_remote_model_init_compat,
    resolve_trust_remote_code,
)


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


def test_normalize_hf_config_compat_restores_sliding_window_cache_alias(monkeypatch):
    monkeypatch.delattr(cache_utils, "SlidingWindowCache", raising=False)

    normalize_hf_config_compat(SimpleNamespace(), trust_remote_code=True)

    assert cache_utils.SlidingWindowCache is cache_utils.StaticCache


def test_prepare_remote_model_init_compat_patches_phi4_scalar_tensors(monkeypatch):
    calls = []

    def fake_tensor(data, *args, **kwargs):
        calls.append((data, kwargs.get("device")))
        return (data, kwargs)

    fake_torch = SimpleNamespace(tensor=fake_tensor)
    speech_module_name = "transformers_modules.fake_phi4.speech_conformer_encoder"
    speech_module = SimpleNamespace(torch=fake_torch)
    monkeypatch.setitem(sys.modules, speech_module_name, speech_module)

    dummy_cls = type("DummyPhi4MM", (), {})
    dummy_cls.__module__ = "transformers_modules.fake_phi4.modeling_phi4mm"
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_id_or_path, **kwargs: dummy_cls,
    )

    config = SimpleNamespace(
        model_type="phi4mm",
        auto_map={"AutoModelForCausalLM": "modeling_phi4mm.Phi4MMForCausalLM"},
    )

    prepare_remote_model_init_compat("/tmp/phi4mm", config)
    monkeypatch.setattr(
        "gptqmodel.utils.hf.inspect.stack",
        lambda context=0: [SimpleNamespace(filename="/tmp/speech_conformer_encoder.py", lineno=1426)],
    )
    monkeypatch.setattr(torch.utils._device, "CURRENT_DEVICE", torch.device("meta"))

    speech_module.torch.tensor(80, dtype=torch.float32)

    assert calls == [(80, "cpu")]
    assert getattr(speech_module, "_gptqmodel_scalar_tensor_meta_patch", False) is True


def test_prepare_remote_model_init_compat_promotes_phi4_positional_seed_to_meta(monkeypatch):
    seen_devices = []

    class AbsolutePositionalEncoding:
        def extend_pe(self, x):
            seen_devices.append(x.device.type)

    fake_torch = SimpleNamespace(tensor=lambda data, *args, **kwargs: (data, kwargs))
    speech_module_name = "transformers_modules.fake_phi4_meta.speech_conformer_encoder"
    speech_module = SimpleNamespace(torch=fake_torch, AbsolutePositionalEncoding=AbsolutePositionalEncoding)
    monkeypatch.setitem(sys.modules, speech_module_name, speech_module)

    dummy_cls = type("DummyPhi4MM", (), {})
    dummy_cls.__module__ = "transformers_modules.fake_phi4_meta.modeling_phi4mm"
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_id_or_path, **kwargs: dummy_cls,
    )

    config = SimpleNamespace(
        model_type="phi4mm",
        auto_map={"AutoModelForCausalLM": "modeling_phi4mm.Phi4MMForCausalLM"},
    )

    prepare_remote_model_init_compat("/tmp/phi4mm", config)
    monkeypatch.setattr(
        "gptqmodel.utils.hf.inspect.stack",
        lambda context=0: [SimpleNamespace(filename="/tmp/speech_conformer_encoder.py", lineno=895)],
    )

    AbsolutePositionalEncoding().extend_pe(torch.tensor(0.0))

    assert seen_devices == ["meta"]


def test_prepare_remote_model_init_compat_tightens_peft_awq_probe(monkeypatch):
    fake_torch = SimpleNamespace(tensor=lambda data, *args, **kwargs: (data, kwargs))
    speech_module_name = "transformers_modules.fake_phi4_awq.speech_conformer_encoder"
    speech_module = SimpleNamespace(torch=fake_torch)
    monkeypatch.setitem(sys.modules, speech_module_name, speech_module)

    dummy_cls = type("DummyPhi4MM", (), {})
    dummy_cls.__module__ = "transformers_modules.fake_phi4_awq.modeling_phi4mm"
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_id_or_path, **kwargs: dummy_cls,
    )

    config = SimpleNamespace(
        model_type="phi4mm",
        auto_map={"AutoModelForCausalLM": "modeling_phi4mm.Phi4MMForCausalLM"},
    )

    def fake_find_spec(name):
        if name == "awq.modules.linear":
            return None
        if name == "awq":
            return object()
        return None

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    prepare_remote_model_init_compat("/tmp/phi4mm", config)

    import peft.tuners.lora.awq as peft_awq

    peft_awq.is_auto_awq_available.cache_clear()
    assert peft_awq.is_auto_awq_available() is False


def test_prepare_remote_model_init_compat_adds_phi4_inner_prepare_inputs_hook(monkeypatch):
    class Phi4MMModel:
        pass

    remote_module_name = "transformers_modules.fake_phi4_inner.modeling_phi4mm"
    speech_module_name = "transformers_modules.fake_phi4_inner.speech_conformer_encoder"
    remote_module = SimpleNamespace(Phi4MMModel=Phi4MMModel)
    speech_module = SimpleNamespace(torch=SimpleNamespace(tensor=lambda data, *args, **kwargs: (data, kwargs)))

    monkeypatch.setitem(sys.modules, remote_module_name, remote_module)
    monkeypatch.setitem(sys.modules, speech_module_name, speech_module)

    dummy_cls = type("DummyPhi4MM", (), {})
    dummy_cls.__module__ = remote_module_name
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_id_or_path, **kwargs: dummy_cls,
    )

    config = SimpleNamespace(
        model_type="phi4mm",
        auto_map={"AutoModelForCausalLM": "modeling_phi4mm.Phi4MMForCausalLM"},
    )

    prepare_remote_model_init_compat("/tmp/phi4mm", config)

    model_inputs = Phi4MMModel().prepare_inputs_for_generation(input_ids="ids", past_key_values="cache")

    assert model_inputs["input_ids"] == "ids"
    assert model_inputs["past_key_values"] == "cache"


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
