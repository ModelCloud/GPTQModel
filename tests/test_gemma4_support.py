from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoConfig

from gptqmodel.models import auto
from gptqmodel.models.definitions.gemma4 import Gemma4ForConditionalGenerationGPTQ, Gemma4TextQModel


GEMMA4_VARIANTS = [
    "/monster/data/model/gemma-4-E2B",
    "/monster/data/model/gemma-4-E4B-it",
    "/monster/data/model/gemma-4-31B-it",
]


@pytest.mark.parametrize("model_path", GEMMA4_VARIANTS)
def test_gemma4_local_variants_select_multimodal_definition(model_path):
    config = AutoConfig.from_pretrained(model_path)

    assert config.model_type == "gemma4"
    assert auto.check_and_get_model_definition(model_path) is Gemma4ForConditionalGenerationGPTQ


def test_gemma4_text_model_type_selects_text_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="gemma4_text")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/gemma4-text") is Gemma4TextQModel


def test_gemma4_module_tree_keeps_optional_variant_paths_non_strict():
    layer_modules = Gemma4TextQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert Gemma4TextQModel.layer_modules_strict is False
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "per_layer_input_gate" in flat_modules
    assert "per_layer_projection" in flat_modules


def test_gemma4_multimodal_base_modules_include_per_layer_helpers():
    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity()])
            self.embed_tokens = nn.Embedding(4, 4)
            self.embed_tokens_per_layer = nn.Embedding(4, 4)
            self.per_layer_model_projection = nn.Linear(4, 4, bias=False)
            self.per_layer_projection_norm = nn.LayerNorm(4)
            self.norm = nn.LayerNorm(4)
            self.rotary_emb = nn.Identity()

    class _Gemma4Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _LanguageModel()
            self.vision_tower = nn.Identity()
            self.embed_vision = nn.Identity()
            self.audio_tower = nn.Identity()
            self.embed_audio = nn.Identity()

    class _Gemma4Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Gemma4Core()
            self.lm_head = nn.Linear(4, 4, bias=False)

    model = _Gemma4Wrapper()
    base_modules = set(Gemma4ForConditionalGenerationGPTQ.get_base_modules(model))

    assert Gemma4ForConditionalGenerationGPTQ.extract_layers_node() == ["model.language_model.layers"]
    assert "model.vision_tower" in base_modules
    assert "model.embed_vision" in base_modules
    assert "model.audio_tower" in base_modules
    assert "model.embed_audio" in base_modules
    assert "model.language_model.embed_tokens" in base_modules
    assert "model.language_model.embed_tokens_per_layer" in base_modules
    assert "model.language_model.per_layer_model_projection" in base_modules
    assert "model.language_model.per_layer_projection_norm" in base_modules


def test_gemma4_capture_preserves_per_layer_input():
    model_def = object.__new__(Gemma4ForConditionalGenerationGPTQ)
    hidden_states = torch.randn(1, 4, 8)
    per_layer_input = torch.randn(1, 4, 2)

    captured = model_def.capture_first_layer_positional_inputs(
        args=(hidden_states, per_layer_input),
        kwargs={},
        batch_device=torch.device("cpu"),
    )

    assert len(captured) == 2
    assert torch.equal(captured[0], hidden_states)
    assert torch.equal(captured[1], per_layer_input)


def test_gemma4_replay_kwargs_refresh_position_embeddings():
    class _FakeRotary(nn.Module):
        def forward(self, x, position_ids, layer_type=None):
            marker = 7.0 if layer_type == "full_attention" else 3.0
            shape = (position_ids.shape[0], position_ids.shape[1], 1)
            value = torch.full(shape, marker, dtype=x.dtype, device=x.device)
            return value, value + 1

    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = _FakeRotary()

    class _Gemma4Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _LanguageModel()

    class _Gemma4Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Gemma4Core()

    model_def = object.__new__(Gemma4ForConditionalGenerationGPTQ)
    nn.Module.__init__(model_def)
    model_def.model = _Gemma4Wrapper()

    layer = SimpleNamespace(self_attn=SimpleNamespace(layer_type="full_attention"))
    hidden_states = torch.randn(1, 4, 8)
    refreshed = model_def.prepare_layer_replay_kwargs(
        layer=layer,
        layer_input=[hidden_states],
        additional_inputs={
            "position_ids": torch.arange(4).unsqueeze(0),
            "position_embeddings": ("stale",),
        },
        target_device=torch.device("cpu"),
    )

    cos, sin = refreshed["position_embeddings"]
    assert cos.shape == (1, 4, 1)
    assert sin.shape == (1, 4, 1)
    assert torch.all(cos == 7)
    assert torch.all(sin == 8)


def test_gemma4_capture_kwargs_preserve_all_per_layer_inputs():
    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = nn.Identity()
            self._gptqmodel_cached_all_per_layer_inputs = torch.randn(1, 4, 3, 2)

    class _Gemma4Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _LanguageModel()

    class _Gemma4Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Gemma4Core()

    model_def = object.__new__(Gemma4ForConditionalGenerationGPTQ)
    nn.Module.__init__(model_def)
    model_def.model = _Gemma4Wrapper()

    captured = model_def.capture_first_layer_input_kwargs(
        args=(),
        kwargs={},
        batch_device=torch.device("cpu"),
        layer_input_kwargs={},
    )

    assert "__gptqmodel_gemma4_all_per_layer_inputs" in captured
    assert captured["__gptqmodel_gemma4_all_per_layer_inputs"].shape == (1, 4, 3, 2)


def _tiny_gemma4_causal_lm(num_kv_shared_layers):
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        sliding_window=4,
        num_kv_shared_layers=num_kv_shared_layers,
        hidden_size_per_layer_input=0,
    )
    return Gemma4ForCausalLM(config).eval()


def _text_model_def(model):
    model_def = object.__new__(Gemma4TextQModel)
    nn.Module.__init__(model_def)
    model_def.model = model
    return model_def


def _store_flags(model):
    return [layer.self_attn.store_full_length_kv for layer in model.model.layers]


def test_gemma4_disables_unused_full_length_kv_store():
    model = _tiny_gemma4_causal_lm(num_kv_shared_layers=0)

    # Transformers flags the last layer of each attention type even without a KV-shared tail.
    assert _store_flags(model) == [False, False, True, True]
    assert not any(layer.self_attn.is_kv_shared_layer for layer in model.model.layers)

    _text_model_def(model).pre_quantize_generate_hook_start()

    assert _store_flags(model) == [False, False, False, False]


def test_gemma4_unused_kv_store_no_longer_grows_captured_shared_kv_states():
    model = _tiny_gemma4_causal_lm(num_kv_shared_layers=0)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 6))

    # Cached replay reuses one captured dict per batch; before the fix every
    # storing layer leaves its full-length K/V behind in it.
    captured = {}
    with torch.no_grad():
        model(input_ids=input_ids, shared_kv_states=captured, use_cache=False)
    assert set(captured) == {"sliding_attention", "full_attention"}

    _text_model_def(model).pre_quantize_generate_hook_start()

    captured = {}
    with torch.no_grad():
        reference = model(input_ids=input_ids, use_cache=False).logits
        replayed = model(input_ids=input_ids, shared_kv_states=captured, use_cache=False).logits
    assert captured == {}
    assert torch.equal(reference, replayed)


def test_gemma4_keeps_kv_store_for_kv_shared_variants():
    # Layers 2-3 read the K/V stored by layers 0-1, so the store must stay enabled.
    model = _tiny_gemma4_causal_lm(num_kv_shared_layers=2)
    flags_before = _store_flags(model)
    assert flags_before == [True, True, False, False]

    _text_model_def(model).pre_quantize_generate_hook_start()

    assert _store_flags(model) == flags_before
