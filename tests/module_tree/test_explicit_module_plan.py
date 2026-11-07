# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import io
import json

import pytest
import torch.nn as nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.hymba import HymbaQModel
from gptqmodel.models.module_plan import ModulePlan, _model_module_plan
from gptqmodel.models.moe_lifecycle import MoELifecycleHooks
from gptqmodel.models.moe_lifecycle import _module_plan as _hook_module_plan
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


def test_expert_queries_reject_undeclared_descendants():
    plan = ModulePlan.from_tree([
        "model", "layers", "#",
        {"mlp:moe": {"experts": {"#": ("gate_proj:0",)}}},
    ])
    assert plan.is_expert("mlp.experts.3.gate_proj")
    assert plan.expert_root("mlp.experts.3.gate_proj") == "mlp.experts.3"
    assert plan.find("mlp.experts.3.undeclared") is None
    assert not plan.is_expert("mlp.experts.3.undeclared")
    assert plan.expert_root("mlp.experts.3.undeclared") is None


def test_fixed_index_expert_definition_is_not_a_shared_expert():
    plan = ModulePlan.from_tree(HymbaQModel.module_tree)
    assert plan.expert_containers() == ("moe.experts",)
    assert plan.shared_expert_containers() == ()
    assert plan.expert_container_leaf("moe") == "experts"
    assert plan.is_expert("moe.experts.0.gate_proj")
    assert plan.expert_root("moe.experts.0.down_proj") == "moe.experts.0"
    assert not plan.is_expert("moe.experts.1.gate_proj")

    block = nn.Module()
    block.experts = nn.ModuleList([nn.Linear(2, 2)])
    assert MoELifecycleHooks().get_experts_module(block, HymbaQModel) is block.experts


def test_unbound_moe_hook_compiles_definition_plan_instead_of_using_method():
    class ToyDefinition(BaseQModel):
        module_tree = [
            "model", "layers", "#",
            {"mlp:moe": {
                "experts": {"#": ("gate_proj:0",)},
                "shared_experts": ("gate_proj:0",),
            }},
        ]

    block = nn.Module()
    block.experts = nn.ModuleList([nn.Linear(2, 2)])
    block.shared_experts = nn.Module()
    block.shared_experts.gate_proj = nn.Linear(2, 2)
    hooks = MoELifecycleHooks()
    assert isinstance(_hook_module_plan(ToyDefinition), ModulePlan)
    assert hooks.get_experts_module(block, ToyDefinition) is block.experts
    assert hooks.get_shared_experts_module(block, ToyDefinition) is block.shared_experts

    compiled = ModulePlan.from_tree(ToyDefinition.module_tree)

    class LiveModel:
        def module_plan(self):
            return compiled

    assert _hook_module_plan(LiveModel()) is compiled


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


def _render_plan() -> ModulePlan:
    return ModulePlan.from_tree([
        "model", "layers", "#",
        {
            "self_attn:!": (
                "q_proj:0:q:in=x",
                "k_proj:0:k:in=x",
                "v_proj:0:v",
                "o_proj:1:o",
            ),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {"#": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down")},
                "shared_experts": (
                    "gate_proj:0:gate",
                    "up_proj:0:up",
                    "down_proj:1:down",
                ),
            },
        },
    ])


def test_module_plan_renders_a_tui_tree():
    plan = _render_plan()

    rendered = plan.render(color=False)
    assert "ModulePlan: 16 specs | 14 quantized | 2 excluded | 0 capture-only" in rendered
    assert "MoE roots: mlp" in rendered
    assert "├─ self_attn [container] [excluded]" in rendered
    assert "q_proj [quantize] [group=0] [in=x] [roles=q]" in rendered
    assert "{expert_index} [container] [expert]" in rendered
    assert "shared_experts [container] [shared-expert]" in rendered
    assert "… 4 descendant path(s)" in plan.render(color=False, max_depth=1, show_summary=False)


def test_module_plan_print_and_extraction_payload():
    plan = _render_plan()

    output = io.StringIO()
    plan.print(file=output, color=False)
    assert output.getvalue() == plan.render(color=False) + "\n"

    payload = plan.to_dict()
    assert payload["summary"] == {
        "spec_count": 16,
        "quantized_count": 14,
        "excluded_count": 2,
        "capture_only_count": 0,
        "moe_roots": ["mlp"],
        "expert_containers": ["mlp.experts"],
        "shared_expert_containers": ["mlp.shared_experts"],
    }
    q_proj = next(spec for spec in payload["specs"] if spec["template"] == "self_attn.q_proj")
    assert q_proj == {
        "template": "self_attn.q_proj",
        "parent": "self_attn",
        "flags": ["0", "q", "in=x"],
        "roles": ["q"],
        "quantize": True,
        "capture_only": False,
        "moe": False,
        "container": False,
        "expert_placeholder": False,
    }
    experts = next(spec for spec in payload["specs"] if spec["template"] == "mlp.experts")
    assert experts["container"] is True
    assert json.loads(plan.to_json()) == payload


def test_model_plan_returns_the_compiled_plan(capsys):
    plan = _render_plan()
    model = object.__new__(BaseQModel)
    model._module_plan = plan

    assert model.module_plan() is plan
    model.module_plan().print(color=False)
    assert "ModulePlan: 16 specs" in capsys.readouterr().out


def test_module_plan_is_exported_from_public_package_roots():
    from gptqmodel import ModulePlan as public_module_plan
    from gptqmodel.models import ModulePlan as models_module_plan
    from gptqmodel.models import ModuleSpec as models_module_spec

    assert public_module_plan is ModulePlan
    assert models_module_plan is ModulePlan
    assert models_module_spec is not None


def test_model_plan_fails_loudly_without_a_compiled_plan():
    model = object.__new__(BaseQModel)

    with pytest.raises(RuntimeError, match="module plan is not compiled yet"):
        model.module_plan()


def test_internal_consumers_resolve_the_plan_through_the_public_method():
    plan = _render_plan()

    class _Model:
        _module_plan = object()

        def module_plan(self):
            return plan

    assert _model_module_plan(_Model()) is plan
    assert _model_module_plan(object()) is None
