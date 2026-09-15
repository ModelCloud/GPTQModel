from types import SimpleNamespace

import defuser
import torch
from transformers import AutoModelForCausalLM
from transformers.models.deepseek_v32.configuration_deepseek_v32 import (
    DeepseekV32Config,
)

from gptqmodel.models import auto
from gptqmodel.models.definitions.deepseek_v32 import DeepSeekV32QModel
from gptqmodel.utils.model import find_modules


def _tiny_config() -> DeepseekV32Config:
    config = DeepseekV32Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=4,
        n_group=1,
        topk_group=1,
        max_position_embeddings=32,
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    return config


def test_deepseek_v32_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="deepseek_v32")

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config
    )

    assert auto.check_and_get_model_definition("deepseek-v32-fixture") is DeepSeekV32QModel


def test_deepseek_v32_module_tree_covers_dsa_dense_and_moe_paths():
    config = _tiny_config()
    layer_modules = DeepSeekV32QModel.simple_layer_modules(
        model_config=config,
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert {
        "self_attn.q_a_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.kv_b_proj",
        "self_attn.indexer.wk",
        "self_attn.indexer.wq_b",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp.experts.0.gate_proj",
        "mlp.experts.1.up_proj",
        "mlp.shared_experts.down_proj",
    } <= flat_modules
    assert "self_attn.indexer.weights_proj" not in flat_modules

    full_modules = {
        name
        for block in DeepSeekV32QModel.full_layer_modules(model_config=config)
        for name in block
    }
    assert "self_attn.indexer.weights_proj:!" in full_modules
    assert DeepSeekV32QModel.out_of_model_tensors == {"prefixes": ["model.layers.61"]}


def test_deepseek_v32_defusion_preserves_forward_and_exposes_quantizable_experts():
    torch.manual_seed(0)
    config = _tiny_config()
    model = AutoModelForCausalLM.from_config(
        config, dtype=torch.float32, trust_remote_code=False
    ).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits

    assert defuser.convert_model(model, cleanup_original=False) is True

    with torch.no_grad():
        actual = model(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)

    modules = find_modules(model)
    assert "model.layers.0.self_attn.indexer.weights_proj" in modules
    assert "model.layers.3.mlp.experts.0.gate_proj" in modules
    assert "model.layers.3.mlp.experts.0.up_proj" in modules
    assert "model.layers.3.mlp.experts.0.down_proj" in modules
    assert "model.layers.3.mlp.shared_experts.gate_proj" in modules

    layer_modules = DeepSeekV32QModel.simple_layer_modules(
        model_config=config,
        quantize_config=SimpleNamespace(dynamic=None),
    )
    suffixes = {name for block in layer_modules for name in block}
    matched = {
        name for name in modules if any(name.endswith(suffix) for suffix in suffixes)
    }
    assert any(name.endswith("self_attn.indexer.wk") for name in matched)
    assert any(name.endswith("mlp.experts.0.gate_proj") for name in matched)
    assert not any(name.endswith("self_attn.indexer.weights_proj") for name in matched)
