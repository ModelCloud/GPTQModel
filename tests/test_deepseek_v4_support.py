from types import SimpleNamespace

from gptqmodel.models import auto
from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel


def test_deepseek_v4_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="deepseek_v4")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/deepseek-v4") is DeepSeekV4QModel


def test_deepseek_v4_module_tree_matches_v4_attention_and_fused_experts():
    layer_modules = DeepSeekV4QModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=256),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert "self_attn.q_a_proj" in flat_modules
    assert "self_attn.q_b_proj" in flat_modules
    assert "self_attn.kv_proj" in flat_modules
    assert "self_attn.o_b_proj" in flat_modules
    # grouped projection must stay native and should not be part of quant blocks
    assert "self_attn.o_a_proj" not in flat_modules
    assert "mlp.experts.99.gate_proj" in flat_modules
    assert "mlp.experts.99.up_proj" in flat_modules
    assert "mlp.experts.99.down_proj" in flat_modules
    assert "mlp.shared_experts.gate_proj" in flat_modules
