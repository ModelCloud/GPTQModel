# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import get_module_by_name_prefix, get_module_by_name_suffix


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.mlp = nn.Module()
        layer.mlp.gate_proj = nn.Linear(2, 2)
        layer.mlp.up_proj = nn.Linear(2, 2)
        layer.mlp.down_proj = nn.Linear(2, 2)
        layer.mlp_extra = nn.Linear(2, 2)


def test_prefix_resolution_rejects_sibling_prefix():
    model = _ToyModel()
    module, matched = get_module_by_name_prefix(model, "model.layers.0.mlp")
    assert module is model.model.layers[0].mlp
    assert matched == "model.layers.0.mlp"


def test_prefix_resolution_prefers_the_exact_container():
    root = nn.Module()
    root.a = nn.Module()
    root.a.one = nn.Linear(1, 1)
    root.a.two = nn.Linear(1, 1)
    module, matched = get_module_by_name_prefix(root, "a")
    assert module is root.a
    assert matched == "a"


def test_prefix_resolution_does_not_match_similarly_named_siblings():
    model = _ToyModel()
    module, _ = get_module_by_name_prefix(model, "model.layers.0.mlp")
    assert module is model.model.layers[0].mlp
    module, _ = get_module_by_name_prefix(model, "model.layers.0.mlp.extra")
    assert module is None


def test_suffix_resolution_requires_segment_boundary():
    root = nn.Module()
    root.alpha = nn.Module()
    root.alpha.proj = nn.Linear(1, 1)
    assert get_module_by_name_suffix(root, "proj") is root.alpha.proj
    assert get_module_by_name_suffix(root, "alpha.pro") is None


def test_module_plan_resolves_roles_without_live_name_heuristics():
    plan = ModulePlan.from_tree([
        "model",
        "layers",
        "#",
        {
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "mlp:moe": {
                "experts": {"#": ("gate_proj:0", "up_proj:0", "down_proj:1")},
                "shared_mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ])
    assert plan.roles("mlp.experts.0.gate_proj") == ("gate",)
    assert plan.roles("mlp.experts.0.up_proj") == ("up",)
    assert plan.roles("mlp.experts.0.down_proj") == ("down",)
    assert plan.is_moe("mlp.experts.0.gate_proj")
    assert not plan.is_moe("mlp_extra.gate_proj")
    assert plan.expert_container_leaf("mlp") == "experts"
    assert plan.shared_container_leaf("mlp") == "shared_mlp"


def test_module_plan_shared_containers_do_not_depend_on_expert_spelling():
    plan = ModulePlan.from_tree([
        "model", "layers", "#",
        {
            "mlp:moe": {
                "experts": {"#": ("w1:0:gate", "w3:0:up", "w2:1:down")},
                "parallel_branch": ("w1:0:gate", "w3:0:up", "w2:1:down"),
            },
        },
    ])
    assert plan.shared_expert_containers() == ("mlp.parallel_branch",)
    assert plan.shared_container_leaf("mlp") == "parallel_branch"


def test_module_plan_uses_explicit_projection_tags_for_renamed_leaves():
    plan = ModulePlan.from_tree([
        "model", "layers", "#",
        {"mlp:moe": {"experts": {"#": ("w1:0:gate", "w3:0:up", "w2:1:down")}}},
    ])
    assert plan.roles("mlp.experts.3.w1") == ("gate",)
    assert plan.roles("mlp.experts.3.w3") == ("up",)
    assert plan.roles("mlp.experts.3.w2") == ("down",)


def test_is_moe_module_uses_tree_relative_paths_only():
    class ToyDefinition(BaseQModel):
        module_tree = [
            "model", "layers", "#",
            {"mlp:moe": {"experts": {"#": ("gate_proj:0",)}}},
        ]

    assert ToyDefinition.is_moe_module("model.layers.2.mlp.experts.0.gate_proj")
    assert ToyDefinition.is_moe_module("model.layers.2.mlp")
    assert not ToyDefinition.is_moe_module("model.layers.2.mlp_extra")


def test_unsupported_tree_fails_closed_without_auto_detection():
    class UndefinedTree(BaseQModel):
        module_tree = None

    class Config:
        model_type = "not_registered"

    class Model:
        config = Config()

    assert UndefinedTree._resolve_effective_module_tree(Model(), object()) is None
