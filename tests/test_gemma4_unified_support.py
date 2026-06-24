from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.gemma4_unified import Gemma4UnifiedForConditionalGenerationGPTQ


def test_gemma4_unified_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="gemma4_unified")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/gemma4-unified") is Gemma4UnifiedForConditionalGenerationGPTQ


def test_gemma4_unified_module_tree_excludes_per_layer_input_paths():
    layer_modules = Gemma4UnifiedForConditionalGenerationGPTQ.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert Gemma4UnifiedForConditionalGenerationGPTQ.layer_modules_strict is False
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert f"self_attn.{proj}" in flat_modules
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert f"mlp.{proj}" in flat_modules
    # Unlike the per-layer-input Gemma 4 variants, gemma4_unified has no per-layer adapters.
    assert "per_layer_input_gate" not in flat_modules
    assert "per_layer_projection" not in flat_modules


def test_gemma4_unified_replay_kwargs_refresh_position_embeddings_per_layer_type():
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

    class _Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _LanguageModel()

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Core()

    model_def = object.__new__(Gemma4UnifiedForConditionalGenerationGPTQ)
    nn.Module.__init__(model_def)
    model_def.model = _Wrapper()

    hidden_states = torch.randn(1, 4, 8)
    for layer_type, marker in (("sliding_attention", 3.0), ("full_attention", 7.0)):
        layer = SimpleNamespace(self_attn=SimpleNamespace(layer_type=layer_type))
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
        assert torch.all(cos == marker)
        assert torch.all(sin == marker + 1)
