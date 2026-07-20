from types import SimpleNamespace

from gptqmodel.models import auto
from gptqmodel.models.definitions.ministral3 import Ministral3GPTQ


def test_ministral3_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="ministral3")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/ministral3") is Ministral3GPTQ


def test_ministral3_module_tree_matches_text_only_layout():
    layer_modules = Ministral3GPTQ.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert Ministral3GPTQ.module_tree[:3] == ["model", "layers", "#"]
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules

