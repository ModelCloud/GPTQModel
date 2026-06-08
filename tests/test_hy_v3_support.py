from types import SimpleNamespace

from gptqmodel.models import auto
from gptqmodel.models.definitions.hy_v3 import HYV3QModel

def test_hy_v3_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="hy_v3")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/hy_v3") is HYV3QModel


def test_hy_v3_module_tree_expands_dense_and_sparse_moe_paths():
    layer_modules = HYV3QModel.simple_layer_modules(
        model_config=SimpleNamespace(num_experts=3),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}
    first_expert_block = next(i for i, block in enumerate(layer_modules) if "mlp.experts.0.gate_proj" in block)
    shared_block = next(i for i, block in enumerate(layer_modules) if "mlp.shared_experts.gate_proj" in block)

    assert HYV3QModel.layer_modules_strict is False
    assert HYV3QModel.dynamic_expert_index == "num_experts"
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "self_attn.q_norm" not in flat_modules
    assert "self_attn.k_norm" not in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules
    assert "mlp.shared_experts.gate_proj" in flat_modules
    assert "mlp.shared_experts.up_proj" in flat_modules
    assert "mlp.shared_experts.down_proj" in flat_modules
    assert "mlp.experts.0.gate_proj" in flat_modules
    assert "mlp.experts.1.up_proj" in flat_modules
    assert "mlp.experts.2.down_proj" in flat_modules
    assert "mlp.gate" not in flat_modules
    assert "mlp.e_score_correction_bias" not in flat_modules
    assert first_expert_block < shared_block