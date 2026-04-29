import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest
import torch
import transformers
import transformers.generation.utils as generation_utils
from transformers import GenerationConfig, GPTNeoXConfig, LlamaConfig, cache_utils
from transformers.generation.configuration_utils import GenerationMode

from gptqmodel.utils import hf as hf_utils
from gptqmodel.utils import internal_gguf
from gptqmodel.utils.hf import (
    INTERNAL_HF_GGUF_FILE_KWARG,
    get_hf_gguf_load_kwargs,
    normalize_hf_config_compat,
    normalize_model_id_or_path_for_hf_gguf,
    normalize_torch_dtype_kwarg,
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


def test_normalize_torch_dtype_kwarg_moves_alias_to_dtype():
    kwargs = {"torch_dtype": torch.float16}

    resolved = normalize_torch_dtype_kwarg(kwargs, api_name="test")

    assert resolved is torch.float16
    assert kwargs == {"dtype": torch.float16}


def test_normalize_torch_dtype_kwarg_resolves_explicit_dtype_parameter():
    kwargs = {"torch_dtype": torch.bfloat16}

    resolved = normalize_torch_dtype_kwarg(kwargs, api_name="test", explicit_dtype="auto")

    assert resolved is torch.bfloat16
    assert kwargs == {}


def test_normalize_torch_dtype_kwarg_rejects_conflicting_values():
    kwargs = {"dtype": torch.float16, "torch_dtype": torch.bfloat16}

    with pytest.raises(ValueError, match="both `dtype` and deprecated `torch_dtype`"):
        normalize_torch_dtype_kwarg(kwargs, api_name="test")


def test_normalize_model_id_or_path_for_hf_gguf_rejects_public_kwarg():
    kwargs = {"gguf_file": "bonsai.gguf"}

    with pytest.raises(TypeError, match="does not accept `gguf_file`"):
        normalize_model_id_or_path_for_hf_gguf("/tmp/fake-model", kwargs, api_name="test")


def test_normalize_model_id_or_path_for_hf_gguf_normalizes_local_file(monkeypatch, tmp_path):
    gguf_path = tmp_path / "bonsai.gguf"
    gguf_path.write_bytes(b"GGUF")

    monkeypatch.setattr(hf_utils, "_patch_transformers_prism_gguf_compat", lambda **_kwargs: None)

    kwargs = {}
    model_root = normalize_model_id_or_path_for_hf_gguf(str(gguf_path), kwargs, api_name="test")

    assert model_root == str(tmp_path)
    assert kwargs[INTERNAL_HF_GGUF_FILE_KWARG] == "bonsai.gguf"
    assert get_hf_gguf_load_kwargs(kwargs) == {"gguf_file": "bonsai.gguf"}


def test_patch_transformers_prism_gguf_compat_registers_internal_runtime(monkeypatch):
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.utils import import_utils as hf_import_utils

    monkeypatch.delitem(sys.modules, "gguf", raising=False)
    monkeypatch.setattr(gguf_utils, "is_gguf_available", lambda *args, **kwargs: False)
    monkeypatch.setattr(hf_import_utils, "is_gguf_available", lambda *args, **kwargs: False)
    monkeypatch.setattr(hf_utils, "_transformers_has_native_prism_gguf_support", lambda: False)

    hf_utils._patch_transformers_prism_gguf_compat(api_name="test")

    assert sys.modules["gguf"] is internal_gguf
    assert gguf_utils.is_gguf_available() is True
    assert hf_import_utils.is_gguf_available() is True
    assert gguf_utils.PRISM_Q1_0_G128_NAME == hf_utils.PRISM_Q1_0_G128_NAME
    assert gguf_utils._dequantize_prism_q1_0_g128 is hf_utils._dequantize_prism_q1_0_g128


def test_patch_transformers_prism_gguf_compat_wraps_load_checkpoint_for_torch_loader(monkeypatch):
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.utils import import_utils as hf_import_utils

    calls = {"direct": 0}

    def _original_load_gguf_checkpoint(*args, **kwargs):
        return {"variant": "original"}

    def _direct_loader(**kwargs):
        calls["direct"] += 1
        return {"variant": "direct", "path": kwargs["gguf_checkpoint_path"]}

    monkeypatch.setenv("GPTQMODEL_INTERNAL_GGUF_TORCH_LOADER", "1")
    monkeypatch.delitem(sys.modules, "gguf", raising=False)
    monkeypatch.setattr(gguf_utils, "load_gguf_checkpoint", _original_load_gguf_checkpoint)
    monkeypatch.delattr(gguf_utils, "_GPTQMODEL_INTERNAL_GGUF_TORCH_LOADER_PATCHED", raising=False)
    monkeypatch.delattr(gguf_utils, "_gptqmodel_original_load_gguf_checkpoint", raising=False)
    monkeypatch.setattr(gguf_utils, "is_gguf_available", lambda *args, **kwargs: False)
    monkeypatch.setattr(hf_import_utils, "is_gguf_available", lambda *args, **kwargs: False)
    monkeypatch.setattr(hf_utils, "_transformers_has_native_prism_gguf_support", lambda: False)
    monkeypatch.setattr(hf_utils, "_load_gguf_checkpoint_torch_direct", _direct_loader)

    hf_utils._patch_transformers_prism_gguf_compat(api_name="test")
    result = gguf_utils.load_gguf_checkpoint("bonsai.gguf", return_tensors=True, model_to_load=object())

    assert calls["direct"] == 1
    assert result == {"variant": "direct", "path": "bonsai.gguf"}


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


def test_normalize_hf_config_compat_restores_hybrid_cache_alias(monkeypatch):
    monkeypatch.delattr(cache_utils, "HybridCache", raising=False)

    normalize_hf_config_compat(SimpleNamespace(), trust_remote_code=True)

    assert cache_utils.HybridCache is cache_utils.StaticCache


def test_normalize_hf_config_compat_restores_is_parallelizable_default(monkeypatch):
    monkeypatch.delattr(transformers.PreTrainedModel, "is_parallelizable", raising=False)

    normalize_hf_config_compat(SimpleNamespace(), trust_remote_code=True)

    assert transformers.PreTrainedModel.is_parallelizable is False


def test_normalize_hf_config_compat_restores_flash_attn_legacy_version_probe(monkeypatch):
    monkeypatch.delattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10", raising=False)
    monkeypatch.setattr(transformers.utils, "is_flash_attn_greater_or_equal", lambda version: version == "2.1.0")

    normalize_hf_config_compat(SimpleNamespace(), trust_remote_code=True)

    assert transformers.utils.is_flash_attn_greater_or_equal_2_10() is True


def test_normalize_hf_config_compat_restores_default_rope_init_alias(monkeypatch):
    import transformers.modeling_rope_utils as rope_utils

    monkeypatch.delitem(rope_utils.ROPE_INIT_FUNCTIONS, "default", raising=False)

    config = SimpleNamespace(
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        rope_theta=10000.0,
        hidden_size=64,
        num_attention_heads=8,
        head_dim=8,
    )

    normalize_hf_config_compat(config, trust_remote_code=True)

    inv_freq, attention_factor = rope_utils.ROPE_INIT_FUNCTIONS["default"](config, None)

    torch.testing.assert_close(inv_freq, torch.tensor([1.0, 0.1, 0.01, 0.001]))
    assert attention_factor == 1.0


def test_normalize_hf_config_compat_restores_legacy_cache_length_helpers(monkeypatch):
    monkeypatch.delattr(cache_utils.Cache, "get_max_length", raising=False)
    monkeypatch.delattr(cache_utils.Cache, "get_usable_length", raising=False)

    class DummyLayer(cache_utils.CacheLayerMixin):
        def __init__(self, seq_length, max_cache_shape):
            super().__init__()
            self._seq_length = seq_length
            self._max_cache_shape = max_cache_shape

        def lazy_initialization(self, key_states, value_states):
            self.keys = key_states
            self.values = value_states
            self.is_initialized = True

        def update(self, key_states, value_states, *args, **kwargs):
            self.lazy_initialization(key_states, value_states)
            return key_states, value_states

        def get_mask_sizes(self, query_length):
            return query_length, self._seq_length

        def get_seq_length(self):
            return self._seq_length

        def get_max_cache_shape(self):
            return self._max_cache_shape

    class DummyCache(cache_utils.Cache):
        def __init__(self, seq_length, max_cache_shape):
            self.layers = [DummyLayer(seq_length, max_cache_shape)]

    normalize_hf_config_compat(SimpleNamespace(), trust_remote_code=True)

    limited_cache = DummyCache(seq_length=8, max_cache_shape=10)
    dynamic_cache = DummyCache(seq_length=8, max_cache_shape=-1)

    assert limited_cache.get_max_length() == 10
    assert limited_cache.get_usable_length(4) == 6
    assert dynamic_cache.get_max_length() is None
    assert dynamic_cache.get_usable_length(4) == 8


def test_normalize_hf_config_compat_restores_legacy_dynamic_cache_converters(monkeypatch):
    monkeypatch.delattr(cache_utils.DynamicCache, "to_legacy_cache", raising=False)
    monkeypatch.delattr(cache_utils.DynamicCache, "from_legacy_cache", raising=False)

    normalize_hf_config_compat(SimpleNamespace(), trust_remote_code=True)

    key_states = torch.randn(1, 2, 3, 4)
    value_states = torch.randn(1, 2, 3, 4)

    cache = cache_utils.DynamicCache()
    cache.update(key_states, value_states, 0)
    legacy_cache = cache.to_legacy_cache()
    restored_cache = cache_utils.DynamicCache.from_legacy_cache(legacy_cache)

    assert len(legacy_cache) == 1
    assert torch.equal(legacy_cache[0][0], key_states)
    assert torch.equal(legacy_cache[0][1], value_states)
    assert restored_cache.get_seq_length(0) == 3
    assert torch.equal(restored_cache.layers[0].keys, key_states)
    assert torch.equal(restored_cache.layers[0].values, value_states)


def test_normalize_hf_config_compat_restores_generation_cache_mapping_alias(monkeypatch):
    monkeypatch.delattr(generation_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING", raising=False)

    normalize_hf_config_compat(SimpleNamespace(), trust_remote_code=True)

    namespace = {}
    exec("from transformers.generation.utils import NEED_SETUP_CACHE_CLASSES_MAPPING", namespace)

    assert namespace["NEED_SETUP_CACHE_CLASSES_MAPPING"] is generation_utils.NEED_SETUP_CACHE_CLASSES_MAPPING
    assert isinstance(generation_utils.NEED_SETUP_CACHE_CLASSES_MAPPING, dict)


def test_normalize_hf_config_compat_supports_legacy_custom_generation_cache_mapping():
    class DummyCustomCache:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyConfig:
        is_encoder_decoder = False
        dtype = torch.float16

        def get_text_config(self, decoder=True):
            assert decoder is True
            return self

    class DummyModel(generation_utils.GenerationMixin):
        _is_stateful = False

        def __init__(self):
            self.config = DummyConfig()
            self.dtype = torch.float16
            self.device = torch.device("cpu")

    normalize_hf_config_compat(SimpleNamespace(), trust_remote_code=True)

    original_mapping = dict(generation_utils.NEED_SETUP_CACHE_CLASSES_MAPPING)
    generation_utils.NEED_SETUP_CACHE_CLASSES_MAPPING["variable"] = DummyCustomCache
    try:
        model = DummyModel()
        generation_config = GenerationConfig(use_cache=True, num_beams=2, num_return_sequences=1)
        generation_config.cache_implementation = "variable"
        model_kwargs = {}

        model._prepare_cache_for_generation(
            generation_config,
            model_kwargs,
            GenerationMode.GREEDY_SEARCH,
            batch_size=3,
            max_cache_length=17,
        )
    finally:
        generation_utils.NEED_SETUP_CACHE_CLASSES_MAPPING.clear()
        generation_utils.NEED_SETUP_CACHE_CLASSES_MAPPING.update(original_mapping)

    assert isinstance(model_kwargs["past_key_values"], DummyCustomCache)
    assert model_kwargs["past_key_values"].kwargs["config"] is model.config
    assert model_kwargs["past_key_values"].kwargs["batch_size"] == 6
    assert model_kwargs["past_key_values"].kwargs["max_batch_size"] == 6
    assert model_kwargs["past_key_values"].kwargs["max_cache_len"] == 17
    assert model_kwargs["past_key_values"].kwargs["dtype"] == torch.float16
    assert model_kwargs["past_key_values"].kwargs["device"] == torch.device("cpu")


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


def test_prepare_remote_model_init_compat_wraps_legacy_tie_weights_signature(monkeypatch):
    calls = []

    class DummyRemoteModel:
        __module__ = "transformers_modules.fake_ovis.modeling_ovis"

        def tie_weights(self):
            calls.append("tied")

    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_id_or_path, **kwargs: DummyRemoteModel,
    )

    config = SimpleNamespace(
        model_type="ovis",
        auto_map={"AutoModelForCausalLM": "modeling_ovis.Ovis"},
    )

    prepare_remote_model_init_compat("/tmp/ovis", config)

    DummyRemoteModel().tie_weights(missing_keys={"lm_head.weight"}, recompute_mapping=False)

    assert calls == ["tied"]
    assert getattr(DummyRemoteModel, "_gptqmodel_tie_weights_kwargs_patch", False) is True


def test_prepare_remote_model_init_compat_backfills_legacy_flash_attn_flag(monkeypatch):
    class DummyRemoteBase:
        _supports_flash_attn_2 = True

    class DummyRemoteModel(DummyRemoteBase):
        __module__ = "transformers_modules.fake_bailing.modeling_bailing_moe_v2"

    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_id_or_path, **kwargs: DummyRemoteModel,
    )

    config = SimpleNamespace(
        model_type="bailing_moe",
        auto_map={"AutoModelForCausalLM": "modeling_bailing_moe_v2.BailingMoeV2ForCausalLM"},
    )

    assert "_supports_flash_attn" not in DummyRemoteBase.__dict__

    prepare_remote_model_init_compat("/tmp/ling", config)

    assert DummyRemoteBase._supports_flash_attn is True
    assert DummyRemoteModel._supports_flash_attn is True


def test_prepare_remote_model_init_compat_accepts_tokenizers_backend_for_ovis(monkeypatch):
    class DummyRemoteModel:
        __module__ = "transformers_modules.fake_ovis.modeling_ovis"

    config_module = ModuleType("transformers_modules.fake_ovis.configuration_ovis")

    class Llama3ConversationFormatter:
        support_tokenizer_types = ["PreTrainedTokenizerFast"]

    config_module.Llama3ConversationFormatter = Llama3ConversationFormatter
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, "transformers_modules.fake_ovis.modeling_ovis", ModuleType("transformers_modules.fake_ovis.modeling_ovis"))
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_id_or_path, **kwargs: DummyRemoteModel,
    )

    config = SimpleNamespace(
        model_type="ovis",
        auto_map={"AutoModelForCausalLM": "modeling_ovis.Ovis"},
    )

    prepare_remote_model_init_compat("/tmp/ovis", config)

    assert "TokenizersBackend" in Llama3ConversationFormatter.support_tokenizer_types
    assert getattr(Llama3ConversationFormatter, "_gptqmodel_tokenizer_backend_patch", False) is True


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

    peft_awq = pytest.importorskip("peft.tuners.lora.awq")

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


def test_prepare_remote_model_init_compat_defaults_phi4_forward_input_mode(monkeypatch):
    class InputMode(Enum):
        LANGUAGE = 0
        VISION = 1
        SPEECH = 2
        VISION_SPEECH = 3

    class Phi4MMForCausalLM:
        def forward(self, *args, **kwargs):
            return kwargs["input_mode"]

    remote_module_name = "transformers_modules.fake_phi4_forward.modeling_phi4mm"
    speech_module_name = "transformers_modules.fake_phi4_forward.speech_conformer_encoder"
    remote_module = SimpleNamespace(InputMode=InputMode, Phi4MMForCausalLM=Phi4MMForCausalLM)
    speech_module = SimpleNamespace(torch=SimpleNamespace(tensor=lambda data, *args, **kwargs: (data, kwargs)))

    Phi4MMForCausalLM.__module__ = remote_module_name
    monkeypatch.setitem(sys.modules, remote_module_name, remote_module)
    monkeypatch.setitem(sys.modules, speech_module_name, speech_module)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_id_or_path, **kwargs: Phi4MMForCausalLM,
    )

    config = SimpleNamespace(
        model_type="phi4mm",
        auto_map={"AutoModelForCausalLM": "modeling_phi4mm.Phi4MMForCausalLM"},
    )

    prepare_remote_model_init_compat("/tmp/phi4mm", config)

    model = Phi4MMForCausalLM()

    assert model.forward(input_ids="ids") is InputMode.LANGUAGE
    assert model.forward(input_audio_embeds="audio") is InputMode.SPEECH
    assert model.forward(input_image_embeds="image") is InputMode.VISION


def test_prepare_remote_model_init_compat_skips_input_mode_patch_without_forward(monkeypatch):
    class Phi4MMForCausalLM:
        pass

    remote_module_name = "transformers_modules.fake_phi4_no_forward.modeling_phi4mm"
    speech_module_name = "transformers_modules.fake_phi4_no_forward.speech_conformer_encoder"
    remote_module = SimpleNamespace(Phi4MMForCausalLM=Phi4MMForCausalLM)
    speech_module = SimpleNamespace(torch=SimpleNamespace(tensor=lambda data, *args, **kwargs: (data, kwargs)))

    Phi4MMForCausalLM.__module__ = remote_module_name
    monkeypatch.setitem(sys.modules, remote_module_name, remote_module)
    monkeypatch.setitem(sys.modules, speech_module_name, speech_module)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_id_or_path, **kwargs: Phi4MMForCausalLM,
    )

    config = SimpleNamespace(
        model_type="phi4mm",
        auto_map={"AutoModelForCausalLM": "modeling_phi4mm.Phi4MMForCausalLM"},
    )

    prepare_remote_model_init_compat("/tmp/phi4mm", config)

    assert not hasattr(Phi4MMForCausalLM, "_gptqmodel_input_mode_patch")


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
