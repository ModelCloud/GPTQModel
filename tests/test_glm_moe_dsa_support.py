from types import SimpleNamespace

from defuser import convert_model
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

from gptqmodel.models import auto
from gptqmodel.models.definitions.glm_moe_dsa import GlmMoeDsaQModel


def _tiny_glm_moe_dsa_config(num_hidden_layers: int = 4) -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_glm_moe_dsa_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="glm_moe_dsa")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/glm-5.1") is GlmMoeDsaQModel


def test_glm_moe_dsa_module_tree_expands_dense_and_sparse_paths():
    layer_modules = GlmMoeDsaQModel.simple_layer_modules(
        model_config=_tiny_glm_moe_dsa_config(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert GlmMoeDsaQModel.layer_modules_strict is False
    assert "self_attn.q_a_proj" in flat_modules
    assert "self_attn.kv_a_proj_with_mqa" in flat_modules
    assert "self_attn.indexer.wk" in flat_modules
    assert "self_attn.indexer.wq_b" in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.experts.0.gate_proj" in flat_modules
    assert "mlp.experts.0.up_proj" in flat_modules
    assert "mlp.experts.0.down_proj" in flat_modules
    assert "mlp.shared_experts.gate_proj" in flat_modules
    assert "mlp.shared_experts.up_proj" in flat_modules
    assert "mlp.shared_experts.down_proj" in flat_modules


def test_glm_moe_dsa_tiny_model_matches_definition():
    model = GlmMoeDsaForCausalLM(_tiny_glm_moe_dsa_config())
    convert_model(model, cleanup_original=False)

    dense_layer = model.model.layers[0]
    moe_layer = model.model.layers[3]

    assert hasattr(dense_layer.self_attn, "q_a_proj")
    assert hasattr(dense_layer.self_attn, "kv_a_proj_with_mqa")
    assert hasattr(dense_layer.self_attn.indexer, "wk")
    assert hasattr(dense_layer.self_attn.indexer, "wq_b")
    assert hasattr(dense_layer.mlp, "gate_proj")
    assert hasattr(dense_layer.mlp, "up_proj")
    assert hasattr(dense_layer.mlp, "down_proj")
    assert not hasattr(dense_layer.mlp, "experts")

    assert hasattr(moe_layer.mlp, "gate")
    assert hasattr(moe_layer.mlp, "experts")
    assert hasattr(moe_layer.mlp, "shared_experts")
    assert len([name for name, _ in moe_layer.mlp.experts.named_children() if name.isdigit()]) == model.config.n_routed_experts

    expert0 = getattr(moe_layer.mlp.experts, "0")
    assert hasattr(expert0, "gate_proj")
    assert hasattr(expert0, "up_proj")
    assert hasattr(expert0, "down_proj")
