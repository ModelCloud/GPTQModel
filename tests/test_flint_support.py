# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.models import auto
from gptqmodel.models.definitions.flint import FlintQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.utils.model import find_modules


def test_flint_model_definition_is_registered(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(model_type="flint"))
    assert auto.check_and_get_model_definition("flint-fixture") is FlintQModel


def test_flint_model_tree_covers_experts_and_attention_sinks():
    layer = FlintQModel.module_tree[3]
    assert "softmax_offset:!" in layer["self_attn"]
    assert "experts" in layer["mlp:moe"]
    assert "shared_expert" in layer["mlp:moe"]


def test_flint_defuser_splits_stacked_experts_and_preserves_forward():
    modeling_flint = pytest.importorskip("transformers.models.flint.modeling_flint")
    config = modeling_flint.FlintConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[0],
        max_position_embeddings=32,
    )
    model = modeling_flint.FlintForCausalLM(config).eval()
    experts = model.model.layers[1].mlp.experts
    hidden_states = torch.randn(5, config.hidden_size)
    top_k_indices = torch.tensor([[0, 1], [2, 3], [1, 2], [3, 0], [2, 1]])
    top_k_weights = torch.full((5, 2), 0.5)

    with torch.no_grad():
        expected = experts(hidden_states, top_k_indices, top_k_weights)

    assert _convert_model_with_defuser(FlintQModel, model, cleanup_original=False)

    with torch.no_grad():
        actual = experts(hidden_states, top_k_indices, top_k_weights)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    modules = find_modules(model)
    assert "model.layers.1.mlp.experts.0.gate_proj" in modules
    assert "model.layers.1.mlp.experts.0.up_proj" in modules
    assert "model.layers.1.mlp.experts.0.down_proj" in modules
