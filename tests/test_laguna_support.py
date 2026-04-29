# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from gptqmodel.models import auto
from gptqmodel.models.definitions.laguna import LagunaQModel


def test_laguna_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="laguna")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/laguna") is LagunaQModel


def test_laguna_module_tree_expands_dense_attention_and_sparse_moe_paths():
    layer_modules = LagunaQModel.simple_layer_modules(
        model_config=SimpleNamespace(num_experts=3),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert LagunaQModel.layer_modules_strict is False
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.g_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
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


def test_laguna_consumes_defuser_for_fused_expert_conversion():
    modeling_laguna = pytest.importorskip("transformers.models.laguna.modeling_laguna")

    from defuser.model_registry import MODEL_CONFIG

    assert "laguna" in MODEL_CONFIG

    config = modeling_laguna.LagunaConfig(
        num_hidden_layers=2,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_attention_heads_per_layer=[2, 2],
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        num_experts=3,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        mlp_layer_types=["dense", "sparse"],
        layer_types=["full_attention", "full_attention"],
    )
    model = modeling_laguna.LagunaForCausalLM(config)
    experts = model.model.layers[1].mlp.experts
    assert hasattr(experts, "gate_up_proj")

    from defuser import convert_model

    assert convert_model(model, cleanup_original=False) is True
    experts = model.model.layers[1].mlp.experts
    assert not hasattr(experts, "gate_up_proj")
    assert hasattr(experts, "0")
    assert hasattr(experts[0], "gate_proj")
    assert hasattr(experts[0], "up_proj")
    assert hasattr(experts[0], "down_proj")
