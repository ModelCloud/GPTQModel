# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from gptqmodel.models import auto
from gptqmodel.models.definitions.granitemoe import GraniteMoeQModel
from gptqmodel.models.definitions.granitemoehybrid import GraniteMoeHybridQModel


def test_granitemoe_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="granitemoe")

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: fake_config,
    )

    assert auto.check_and_get_model_definition("/tmp/power-moe") is GraniteMoeQModel
    assert auto.MODEL_MAP["granitemoe"] is GraniteMoeQModel


def test_granitemoe_module_tree_targets_defused_expert_projections():
    tree = GraniteMoeQModel.module_tree[-1]
    moe_tree = tree["block_sparse_moe:moe:?"]

    assert GraniteMoeQModel.dynamic_expert_index == "num_local_experts"
    assert GraniteMoeQModel.pre_lm_head_norm_module == "model.norm"
    assert moe_tree["experts"]["#"] == ("gate_proj:0", "up_proj:0", "down_proj:1")
    assert "input_linear" not in moe_tree
    assert "output_linear" not in moe_tree


def _expert_modules(definition):
    layer_modules = definition.simple_layer_modules(
        model_config=SimpleNamespace(num_local_experts=2),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    return {name for block in layer_modules for name in block}


def test_granitemoe_targets_defused_gate_up_down_expert_projections():
    flat_modules = _expert_modules(GraniteMoeQModel)

    assert "block_sparse_moe.experts.0.gate_proj" in flat_modules
    assert "block_sparse_moe.experts.0.up_proj" in flat_modules
    assert "block_sparse_moe.experts.0.down_proj" in flat_modules
    assert not any("input_linear" in name or "output_linear" in name for name in flat_modules)
    assert not any(name.endswith((".w1", ".w2", ".w3")) for name in flat_modules)


def test_granitemoehybrid_targets_defused_gate_up_down_expert_projections():
    flat_modules = _expert_modules(GraniteMoeHybridQModel)

    assert "block_sparse_moe.experts.0.gate_proj" in flat_modules
    assert "block_sparse_moe.experts.0.up_proj" in flat_modules
    assert "block_sparse_moe.experts.0.down_proj" in flat_modules
    assert not any(name.endswith((".w1", ".w2", ".w3")) for name in flat_modules)

    # Shared MLP and Mamba projections keep their own (correct) leaf names.
    assert "shared_mlp.input_linear" in flat_modules
    assert "shared_mlp.output_linear" in flat_modules
    for name in ("mamba.in_proj", "mamba.out_proj", "self_attn.q_proj", "self_attn.o_proj"):
        assert name in flat_modules
