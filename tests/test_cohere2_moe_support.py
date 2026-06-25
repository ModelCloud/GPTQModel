# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from gptqmodel.models import auto
from gptqmodel.models.definitions.cohere2_moe import Cohere2MoeQModel


def test_cohere2_moe_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="cohere2_moe")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/cohere2_moe") is Cohere2MoeQModel


def test_cohere2_moe_module_tree_expands_dense_and_sparse_moe_paths():
    layer_modules = Cohere2MoeQModel.simple_layer_modules(
        model_config=SimpleNamespace(num_experts=3),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert Cohere2MoeQModel.layer_modules_strict is False
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "post_attention_layernorm" not in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules
    assert "mlp.experts.0.gate_proj" in flat_modules
    assert "mlp.experts.1.up_proj" in flat_modules
    assert "mlp.experts.2.down_proj" in flat_modules
    assert "mlp.gate" not in flat_modules


def test_cohere2_moe_consumes_defuser_for_fused_expert_conversion():
    modeling_cohere2_moe = pytest.importorskip("transformers.models.cohere2_moe.modeling_cohere2_moe")

    from defuser import convert_model
    from defuser.model_registry import MODEL_CONFIG

    assert "cohere2_moe" in MODEL_CONFIG

    config = modeling_cohere2_moe.Cohere2MoeConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=8,
        prefix_dense_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=3,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
    )
    model = modeling_cohere2_moe.Cohere2MoeForCausalLM(config)
    experts = model.model.layers[1].mlp.experts
    assert hasattr(experts, "gate_up_proj")

    assert convert_model(model, cleanup_original=False) is True
    experts = model.model.layers[1].mlp.experts
    assert not hasattr(experts, "gate_up_proj")
    assert hasattr(experts, "0")
    assert hasattr(experts[0], "gate_proj")
    assert hasattr(experts[0], "up_proj")
    assert hasattr(experts[0], "down_proj")
