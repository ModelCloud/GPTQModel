# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Unit tests for shared-input metadata derived from ``module_tree`` (no model forward)."""

import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models import BaseQModel
from gptqmodel.models._const import EXPERT_INDEX_PLACEHOLDER
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.deepseek_v3 import DeepSeekV3QModel
from gptqmodel.models.definitions.glm4_moe import GLM4MoEGPTQ
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.definitions.qwen3_5_moe import Qwen3_5_MoeQModel
from gptqmodel.models.definitions.qwen3_moe import Qwen3MoeQModel
from gptqmodel.models.shared_input import (
    SharedInputGroup,
    SharedInputPlan,
    build_shared_input_plan,
    collect_leaf_specs,
    parse_shared_input_tag,
    probe_shared_inputs,
    resolve_leaf_spec,
)
from gptqmodel.utils.structure import LazyTurtle


QC = SimpleNamespace(dynamic=None)


def _plan(model_cls, model_config=None):
    return model_cls.shared_input_plan(model_config, QC)


def _groups(plan):
    return {g.key: g.modules for g in plan.groups}


# --------------------------------------------------------------------------- #
# Phase 1: flag parsing
# --------------------------------------------------------------------------- #


class TestFlagParsing:
    def test_absent(self):
        assert parse_shared_input_tag([]) is None
        assert parse_shared_input_tag(["0", "!", "q", "moe"]) is None

    def test_present(self):
        assert parse_shared_input_tag(["0", "in=x"]) == "x"
        assert parse_shared_input_tag(["in=kv_a", "1", "k", "v"]) == "kv_a"

    def test_empty_tag_rejected(self):
        with pytest.raises(ValueError):
            parse_shared_input_tag(["in="])

    def test_role_flags_are_not_input_tags(self):
        # ":q", ":k", ":v", ":gate" etc. are free-form semantic labels and must not be
        # mistaken for shared-input tags.
        assert parse_shared_input_tag(["1", "q"]) is None
        assert parse_shared_input_tag(["0", "k", "v"]) is None

    def test_existing_flag_parser_keeps_in_tag_as_plain_flag(self):
        aliases, flags = LazyTurtle._parse_module_spec("q_b_proj:1:q:in=q_a")
        assert aliases == ("q_b_proj",)
        assert flags == ("1", "q", "in=q_a")

    def test_base_qmodel_flag_parser_keeps_in_tag(self):
        _, flags = BaseQModel._parse_module_flags("q_b_proj:1:q:in=q_a")
        assert "in=q_a" in flags


# --------------------------------------------------------------------------- #
# Phase 2: leaf spec collection from module_tree
# --------------------------------------------------------------------------- #


class TestLeafSpecs:
    tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1", "extra:1:in=z"),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "experts": {"#": ("gate_proj:0", "up_proj:0", "down_proj:1")},
                "shared_experts": {
                    "gate_proj": ("gate_proj:0",),
                    "up_proj": ("up_proj:0",),
                    "down_proj": ("down_proj:1",),
                },
            },
        },
    ]

    def test_collects_flags_and_quantize_state(self):
        specs = collect_leaf_specs(self.tree)
        assert specs["self_attn.q_proj"].subset_tag == 0
        assert specs["self_attn.q_proj"].input_tag is None
        assert specs["self_attn.o_proj"].subset_tag == 1
        assert specs["self_attn.extra"].input_tag == "z"
        assert specs["input_layernorm"].quantize is False
        assert specs["mlp.gate"].quantize is False

    def test_expert_placeholder_template(self):
        specs = collect_leaf_specs(self.tree)
        template = f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj"
        assert template in specs
        assert resolve_leaf_spec(specs, "mlp.experts.7.gate_proj") is specs[template]
        assert resolve_leaf_spec(specs, "mlp.experts.7.gate_proj_x") is None

    def test_leaf_repeating_parent_name_maps_to_parent(self):
        # `"shared_experts": {"gate_proj": ("gate_proj:0",)}` is the same module as
        # `shared_experts.gate_proj` (matches BaseQModel._build_layer_modules_for_tree).
        specs = collect_leaf_specs(self.tree)
        assert "mlp.shared_experts.gate_proj" in specs
        assert "mlp.shared_experts.gate_proj.gate_proj" not in specs
        assert specs["mlp.shared_experts.down_proj"].subset_tag == 1

    def test_resolve_unknown(self):
        assert resolve_leaf_spec(collect_leaf_specs(self.tree), "nope") is None


# --------------------------------------------------------------------------- #
# Phase 3: plan derivation (default + explicit grouping)
# --------------------------------------------------------------------------- #


class _Def(BaseQModel):
    layer_type = "Layer"
    dynamic_expert_index = None


def _make_def(tree):
    return type("TmpQModel", (_Def,), {"module_tree": tree})


class TestPlanDerivation:
    def test_default_groups_by_parent_and_subset(self):
        plan = _plan(LlamaQModel)
        assert _groups(plan) == {
            "self_attn:0": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            "self_attn:1": ("self_attn.o_proj",),
            "mlp:2": ("mlp.gate_proj", "mlp.up_proj"),
            "mlp:3": ("mlp.down_proj",),
        }
        assert plan.leaders == ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj")
        assert plan.dedup_count == 3
        assert plan.leader_for("self_attn.v_proj") == "self_attn.q_proj"
        assert plan.followers_of("self_attn.q_proj") == ("self_attn.k_proj", "self_attn.v_proj")
        assert plan.followers_of("self_attn.k_proj") == ()
        assert plan.shares_input("mlp.gate_proj", "mlp.up_proj")
        assert not plan.shares_input("mlp.gate_proj", "mlp.down_proj")
        assert not plan.shares_input("self_attn.q_proj", "mlp.gate_proj")

    def test_subset_indices_follow_layer_modules_order(self):
        plan = _plan(LlamaQModel)
        g = plan.group_for("mlp.up_proj")
        assert g.subset_indices == (2, 2)
        assert g.subset_index_of("mlp.up_proj") == 2
        assert not g.spans_subsets

    def test_explicit_tag_splits_same_subset(self):
        tree = ["model", "layers", "#", {"self_attn": ("a:0", "b:0:in=x", "c:0:in=y", "d:0:in=x")}]
        plan = _plan(_make_def(tree))
        assert _groups(plan) == {
            "self_attn:0": ("self_attn.a",),
            "self_attn:in=x": ("self_attn.b", "self_attn.d"),
            "self_attn:in=y": ("self_attn.c",),
        }
        assert plan.group_for("self_attn.b").explicit
        assert not plan.group_for("self_attn.a").explicit

    def test_explicit_tag_spans_subsets(self):
        tree = ["model", "layers", "#", {"attn": ("qkv:0:in=x", "z:1:in=x", "o:2")}]
        plan = _plan(_make_def(tree))
        g = plan.group_for("attn.qkv")
        assert g.modules == ("attn.qkv", "attn.z")
        assert g.subset_indices == (0, 1)
        assert g.spans_subsets
        assert _groups(plan.for_subset(1)) == {"attn:in=x": ("attn.z",)}
        assert _groups(plan.for_subset(2)) == {"attn:2": ("attn.o",)}

    def test_default_grouping_follows_expanded_subset_block_not_leaf_digit(self):
        # OPT-style: `"fc1": ("fc1",), "fc2": ("fc2",)` both carry digit 0 but land in
        # separate subset blocks, so they must not be assumed to share an input.
        tree = ["model", "layers", "#", {"attn": ("q:0", "o:1"), "fc1": ("fc1",), "fc2": ("fc2",)}]
        plan = _plan(_make_def(tree))
        assert _groups(plan) == {"attn:0": ("attn.q",), "attn:1": ("attn.o",), ":2": ("fc1",), ":3": ("fc2",)}

    def test_dict_key_digit_offsets_split_children(self):
        tree = [
            "model",
            "layers",
            "#",
            {"mixer": {"": ("in_proj:0",), "fc1:2": ("fc1:0",), "fc2:3": ("fc2:0",)}},
        ]
        plan = _plan(_make_def(tree))
        assert not plan.shares_input("mixer.in_proj", "mixer.fc1")
        assert not plan.shares_input("mixer.fc1", "mixer.fc2")

    def test_non_quantized_and_capture_only_excluded(self):
        tree = [
            "model",
            "layers",
            "#",
            {"attn": ("norm:0:!", "q:0", "k:0", "probe:0:?")},
        ]
        plan = _plan(_make_def(tree))
        assert _groups(plan) == {"attn:0": ("attn.q", "attn.k")}

    def test_different_parents_never_share(self):
        tree = ["model", "layers", "#", {"a": ("x:0",), "b": ("x:0",)}]
        plan = _plan(_make_def(tree))
        assert _groups(plan) == {"a:0": ("a.x",), "b:1": ("b.x",)}

    def test_in_tag_does_not_change_layer_modules(self):
        plain = _make_def(["model", "layers", "#", {"attn": ("q:0", "k:0", "v:0", "o:1")}])
        tagged = _make_def(["model", "layers", "#", {"attn": ("q:0:in=x", "k:0:in=x", "v:0:in=y", "o:1")}])
        assert plain.simple_layer_modules(None, QC) == tagged.simple_layer_modules(None, QC)
        assert plain.full_layer_modules(None, QC, True) == tagged.full_layer_modules(None, QC, True)
        assert _groups(_plan(plain)) != _groups(_plan(tagged))

    def test_build_from_raw_layer_modules_ignores_flagged_entries(self):
        tree = ["model", "layers", "#", {"attn": ("q:0", "k:0", "n:0:!")}]
        plan = build_shared_input_plan(tree, [["attn.q", "attn.k", "attn.n:!"]])
        assert _groups(plan) == {"attn:0": ("attn.q", "attn.k")}

    def test_duplicate_module_names_across_blocks_counted_once(self):
        tree = ["model", "layers", "#", {"attn": ("q:0", "k:0")}]
        plan = build_shared_input_plan(tree, [["attn.q", "attn.k"], ["attn.q"]])
        assert plan.modules == ("attn.q", "attn.k")

    def test_plan_rejects_module_in_two_groups(self):
        g = SharedInputGroup(key="a:0", parent="a", modules=("a.x",), subset_indices=(0,))
        with pytest.raises(ValueError):
            SharedInputPlan(groups=(g, g))

    def test_with_prefix_and_filter(self):
        plan = _plan(LlamaQModel).with_prefix("model.layers.3")
        assert plan.leader_for("model.layers.3.self_attn.k_proj") == "model.layers.3.self_attn.q_proj"
        assert plan.group_for("model.layers.3.self_attn.k_proj").parent == "model.layers.3.self_attn"
        filtered = plan.filter_modules({"model.layers.3.self_attn.k_proj", "model.layers.3.mlp.down_proj"})
        assert _groups(filtered) == {
            "model.layers.3.self_attn:0": ("model.layers.3.self_attn.k_proj",),
            "model.layers.3.mlp:3": ("model.layers.3.mlp.down_proj",),
        }
        assert plan.with_prefix("") is plan

    def test_deterministic_and_thread_safe(self):


        results = []

        def worker():
            results.append(_groups(_plan(LlamaQModel)))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r == results[0] for r in results)


# --------------------------------------------------------------------------- #
# Phase 4: MoE expansion and real model definitions
# --------------------------------------------------------------------------- #


class TestModelDefinitions:
    def test_qwen3_moe_experts_grouped_per_expert(self):
        cfg = SimpleNamespace(num_experts=3)
        plan = _plan(Qwen3MoeQModel, cfg)
        groups = _groups(plan)
        for i in range(3):
            assert plan.shares_input(f"mlp.experts.{i}.gate_proj", f"mlp.experts.{i}.up_proj")
            assert plan.group_for(f"mlp.experts.{i}.down_proj").modules == (f"mlp.experts.{i}.down_proj",)
            assert plan.group_for(f"mlp.experts.{i}.gate_proj").parent == f"mlp.experts.{i}"
        assert not plan.shares_input("mlp.experts.0.gate_proj", "mlp.experts.1.gate_proj")
        assert "mlp.gate" not in plan.modules
        assert len(groups) == 2 + 2 * 3

    def test_unexpanded_template_plan_keeps_placeholder(self):
        plan = _plan(Qwen3MoeQModel)
        tmpl = f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}"
        assert plan.shares_input(f"{tmpl}.gate_proj", f"{tmpl}.up_proj")

    def test_deepseek_v3_mla_explicit_tags(self):
        plan = _plan(DeepSeekV3QModel, SimpleNamespace(n_routed_experts=2))
        groups = _groups(plan)
        assert groups["self_attn:0"] == ("self_attn.q_proj", "self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa")
        assert groups["self_attn:in=q_a"] == ("self_attn.q_b_proj",)
        assert groups["self_attn:in=kv_a"] == ("self_attn.kv_b_proj",)
        assert not plan.shares_input("self_attn.q_b_proj", "self_attn.kv_b_proj")
        assert plan.shares_input("mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj")
        assert not plan.shares_input("mlp.shared_experts.gate_proj", "mlp.shared_experts.down_proj")

    def test_glm4_moe_nested_shared_experts_follow_subset_blocks(self):
        # `"shared_experts": {"gate_proj": ("gate_proj:0",), ...}` places each child in its own
        # subset block, so the conservative default keeps them apart.
        plan = _plan(GLM4MoEGPTQ, SimpleNamespace(n_routed_experts=2))
        g = plan.group_for("mlp.shared_experts.gate_proj")
        assert g.modules == ("mlp.shared_experts.gate_proj",)
        assert not plan.shares_input("mlp.shared_experts.gate_proj", "mlp.shared_experts.down_proj")
        assert plan.shares_input("mlp.experts.0.gate_proj", "mlp.experts.0.up_proj")

    def test_qwen3_5_moe_shared_expert_and_linear_attn(self):
        plan = _plan(Qwen3_5_MoeQModel, SimpleNamespace(num_experts=2))
        assert plan.shares_input("mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj")
        assert plan.shares_input("mlp.experts.0.gate_proj", "mlp.experts.0.up_proj")
        assert not plan.shares_input("mlp.shared_expert.gate_proj", "mlp.experts.0.gate_proj")
        assert "linear_attn.in_proj_qkvz" in plan.modules or "linear_attn.in_proj_qkv" in plan.modules

    @pytest.mark.parametrize("model_type", sorted(MODEL_MAP))
    def test_every_definition_yields_consistent_plan(self, model_type):
        model_cls = MODEL_MAP[model_type]
        plan = model_cls.shared_input_plan(None, QC)
        layer_modules = model_cls.simple_layer_modules(None, QC)
        quantizable = []
        for block in layer_modules:
            for raw in block:
                if ":!" in raw or ":?" in raw:
                    continue
                name = raw.split(":", 1)[0]
                if name not in quantizable:
                    quantizable.append(name)
        assert set(plan.modules) == set(quantizable)
        for g in plan.groups:
            assert len(g.modules) == len(g.subset_indices)
            for m in g.modules:
                assert m.rsplit(".", 1)[0] == g.parent if "." in m else g.parent == ""
            if g.explicit:
                assert ":in=" in g.key
            else:
                # Default groups never mix subset blocks; only explicit tags may span them.
                assert len(set(g.subset_indices)) == 1, g


# --------------------------------------------------------------------------- #
# Phase 5: probe semantics on synthetic layers (no transformers dependency)
# --------------------------------------------------------------------------- #


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

    def forward(self, x):
        return self.o(self.q(x) + self.k(x) + self.v(x))


class _Layer(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = _Attn(d)

    def forward(self, x):
        return self.attn(self.norm(x))


def _group(key, parent, modules, subsets):
    return SharedInputGroup(key=key, parent=parent, modules=tuple(modules), subset_indices=tuple(subsets))


class TestProbe:
    def _run(self, plan, layer=None, x=None):
        layer = layer or _Layer()
        x = x if x is not None else torch.randn(2, 5, 8)
        return probe_shared_inputs(layer, plan, lambda: layer(x))

    def test_correct_plan_verifies(self):
        plan = SharedInputPlan(
            groups=(
                _group("attn:0", "attn", ["attn.q", "attn.k", "attn.v"], [0, 0, 0]),
                _group("attn:1", "attn", ["attn.o"], [1]),
            )
        )
        report = self._run(plan)
        assert report.ok
        assert report.verified == ("attn:0", "attn:1")
        assert report.unverified == ()
        assert report.missing_modules == ()
        assert report.call_counts == {"attn.q": 1, "attn.k": 1, "attn.v": 1, "attn.o": 1}

    def test_wrong_grouping_reports_mismatch(self):
        plan = SharedInputPlan(groups=(_group("attn:0", "attn", ["attn.q", "attn.o"], [0, 0]),))
        report = self._run(plan)
        assert not report.ok
        assert len(report.mismatches) == 1
        m = report.mismatches[0]
        assert (m.leader, m.module) == ("attn.q", "attn.o")
        assert "values differ" in m.reason
        assert "MISMATCH" in report.describe()

    def test_shape_mismatch_reason(self):
        class L(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 4)
                self.b = nn.Linear(4, 4)

            def forward(self, x):
                return self.b(self.a(x))

        layer = L()
        plan = SharedInputPlan(groups=(_group("g", "", ["a", "b"], [0, 0]),))
        report = probe_shared_inputs(layer, plan, lambda: layer(torch.randn(3, 8)))
        assert "shape" in report.mismatches[0].reason

    def test_dtype_mismatch_reason(self):
        class L(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 4)
                self.b = nn.Linear(8, 4).to(torch.float64)

            def forward(self, x):
                return self.a(x).sum() + self.b(x.double()).sum()

        layer = L()
        plan = SharedInputPlan(groups=(_group("g", "", ["a", "b"], [0, 0]),))
        report = probe_shared_inputs(layer, plan, lambda: layer(torch.randn(3, 8)))
        assert "dtype" in report.mismatches[0].reason

    def test_undeclared_shared_input_detected(self):
        # q and k are split into different groups but receive the same tensor.
        plan = SharedInputPlan(
            groups=(
                _group("attn:in=a", "attn", ["attn.q"], [0]),
                _group("attn:in=b", "attn", ["attn.k"], [0]),
            )
        )
        report = self._run(plan)
        assert not report.ok
        assert report.undeclared == (("attn.q", "attn.k"),)
        assert "UNDECLARED" in report.describe()

    def test_undeclared_not_flagged_across_subsets(self):
        # Same input but declared in different subsets -> not a same-subset dedup miss.
        plan = SharedInputPlan(
            groups=(
                _group("attn:0", "attn", ["attn.q"], [0]),
                _group("attn:1", "attn", ["attn.k"], [1]),
            )
        )
        assert self._run(plan).undeclared == ()

    def test_missing_and_uncalled_modules(self):
        class L(nn.Module):
            def __init__(self):
                super().__init__()
                self.used = nn.Linear(8, 8)
                self.idle = nn.Linear(8, 8)

            def forward(self, x):
                return self.used(x)

        layer = L()
        plan = SharedInputPlan(
            groups=(
                _group("u", "", ["used"], [0]),
                _group("i", "", ["idle"], [1]),
                _group("m", "", ["ghost"], [2]),
            )
        )
        report = probe_shared_inputs(layer, plan, lambda: layer(torch.randn(2, 8)))
        assert report.ok
        assert report.verified == ("u",)
        assert report.unverified == ("i",)
        assert report.missing_modules == ("ghost",)
        assert report.call_counts["idle"] == 0

    def test_partial_group_call_uses_called_module_as_reference(self):
        class L(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 8)
                self.b = nn.Linear(8, 8)

            def forward(self, x):
                return self.b(x)

        layer = L()
        plan = SharedInputPlan(groups=(_group("g", "", ["a", "b"], [0, 0]),))
        report = probe_shared_inputs(layer, plan, lambda: layer(torch.randn(2, 8)))
        assert not report.ok
        assert report.mismatches[0].leader == "b"
        assert "call count" in report.mismatches[0].reason

    def test_multiple_calls_per_module_all_compared(self):
        layer = _Layer()
        plan = SharedInputPlan(groups=(_group("attn:0", "attn", ["attn.q", "attn.k"], [0, 0]),))
        xs = [torch.randn(1, 3, 8) for _ in range(3)]

        def fwd():
            for x in xs:
                layer(x)

        report = probe_shared_inputs(layer, plan, fwd)
        assert report.ok
        assert report.call_counts == {"attn.q": 3, "attn.k": 3}

    def test_kwarg_tensor_input_captured(self):
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 8)

            def forward(self, *, hidden_states):
                return self.lin(hidden_states)

        class L(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Inner()
                self.b = Inner()

            def forward(self, x):
                return self.a(hidden_states=x) + self.b(hidden_states=x)

        layer = L()
        plan = SharedInputPlan(groups=(_group("g", "", ["a", "b"], [0, 0]),))
        report = probe_shared_inputs(layer, plan, lambda: layer(torch.randn(2, 8)))
        assert report.ok and report.verified == ("g",)

    def test_hooks_removed_after_probe(self):
        layer = _Layer()
        plan = SharedInputPlan(groups=(_group("attn:0", "attn", ["attn.q", "attn.k"], [0, 0]),))
        probe_shared_inputs(layer, plan, lambda: layer(torch.randn(1, 2, 8)))
        assert not layer.attn.q._forward_pre_hooks
        assert not layer.attn.k._forward_pre_hooks

    def test_hooks_removed_when_forward_raises(self):
        layer = _Layer()
        plan = SharedInputPlan(groups=(_group("attn:0", "attn", ["attn.q"], [0]),))

        def boom():
            raise RuntimeError("x")

        with pytest.raises(RuntimeError):
            probe_shared_inputs(layer, plan, boom)
        assert not layer.attn.q._forward_pre_hooks
