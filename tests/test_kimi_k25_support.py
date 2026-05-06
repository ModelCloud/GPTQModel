from types import SimpleNamespace

from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.kimi_k25 import KimiK25QModel


def test_kimi_k25_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="kimi_k25")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/kimi-k2.5", trust_remote_code=True) is KimiK25QModel


def test_kimi_k25_quantizes_language_model_and_keeps_multimodal_modules_in_base():
    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()

    class _KimiCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = nn.Identity()
            self.mm_projector = nn.Identity()
            self.language_model = _LanguageModel()

    class _KimiWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _KimiCore()

    base_modules = set(KimiK25QModel.get_base_modules(_KimiWrapper()))

    assert KimiK25QModel.require_trust_remote_code is True
    assert KimiK25QModel.require_load_processor is True
    assert KimiK25QModel.pre_lm_head_norm_module == "language_model.model.norm"
    assert "model.vision_tower" in base_modules
    assert "model.mm_projector" in base_modules
    assert "model.language_model" not in base_modules


def test_kimi_k25_module_tree_targets_deepseek_v3_text_backbone():
    layer_modules = KimiK25QModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=2),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert KimiK25QModel.layer_modules_strict is False
    assert KimiK25QModel.dynamic_expert_index == "n_routed_experts"
    assert KimiK25QModel.extract_layers_node() == ["language_model.model.layers"]
    assert "self_attn.q_a_proj" in flat_modules
    assert "self_attn.kv_a_proj_with_mqa" in flat_modules
    assert "self_attn.q_b_proj" in flat_modules
    assert "self_attn.kv_b_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.experts.0.gate_proj" in flat_modules
    assert "mlp.shared_experts.up_proj" in flat_modules
