# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Immutable composition and actual tree-consumer contracts for fusion policy."""

import copy
import json
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from gptqmodel import FusableConfig, LlamaFusableConfig
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.fusable import require_fusable_config
from tests.module_tree.test_llama_fusion import _wrapper


def _policy(config, layer=0, family="qkv", overrides=None):
    return config.policy_for(family, layer, parent="self_attn", full_parent=f"model.layers.{layer}.self_attn",
                             overrides=overrides)


def test_target_first_builder_and_direct_constructor_are_immutable():
    builder = FusableConfig("qwen2")
    base = LlamaFusableConfig()
    merged = builder.merge(base)
    assert builder.model_types == merged.model_types == ("qwen2",)
    assert builder.profiles == () and base.model_types == ("llama",)
    assert merged.profiles == base.profiles and merged.profiles[0] is base.profiles[0]
    assert merged.validate() is merged and copy.deepcopy(merged) is merged
    assert FusableConfig(("a", "b"), profiles=base.profiles).profiles_for("b") == base.profiles
    with pytest.raises(FrozenInstanceError):
        merged.profiles = ()
    with pytest.raises(TypeError):
        merged.overrides["qkv"] = False
    with pytest.raises(TypeError):
        merged._profile_origins["qkv"] = ("changed",)
    assert not any(hasattr(merged, name) for name in ("derive", "for_model", "inherit"))


def test_empty_builder_is_rejected_at_definition_binding_and_before_model_mutation(monkeypatch):
    model, _ = _wrapper()
    original = model.model.model.layers[0].self_attn.forward
    config = FusableConfig("llama")
    monkeypatch.setattr(LlamaQModel, "fusable", config)
    with pytest.raises(ValueError, match="at least one FusableProfile"):
        model.fuse()
    assert model._fusion_plan is None and model.model.model.layers[0].self_attn.forward == original
    with pytest.raises(ValueError, match="at least one FusableProfile"):
        config.profiles_for("llama")


def test_multiple_sources_and_diamond_deduplicate_without_mutating_sources():
    base = LlamaFusableConfig(overrides={"qkv": False})
    left = FusableConfig("left").merge(base)
    right = FusableConfig("right").merge(base)
    diamond = FusableConfig("target").merge(left, right)
    chained = FusableConfig("target").merge(left).merge(right)
    assert diamond.to_dict() == chained.to_dict()
    assert diamond.model_types == ("target",) and diamond.profiles == base.profiles
    assert diamond.profile_origins("qkv") == ("llama",)
    assert diamond.validate().overrides["qkv"] is False
    assert base.to_dict()["conflicts"] == {"profiles": [], "policy": []}
    reversed_sources = FusableConfig("target").merge(right, left)
    assert reversed_sources.profiles_for("target") == diamond.profiles


def test_profile_selection_composes_disjoint_sources_and_only_matching_policy():
    base = LlamaFusableConfig(overrides={"qkv": False, "gate_up": True})
    selected = FusableConfig("custom").merge(base, profiles=("qkv",))
    assert [p.name for p in selected.profiles] == ["qkv"] and dict(selected.overrides) == {"qkv": False}
    mlp = FusableConfig("mlp").profile("gate_up", roles=("gate", "up"))
    combined = selected.merge(mlp)
    assert combined.validate().profiles == base.profiles
    assert combined.profile_origins("qkv") == ("llama",)
    assert combined.profile_origins("gate_up") == ("mlp",)
    assert mlp.profiles[0] is combined.profiles[1]


def test_structural_conflict_requires_explicit_replacement():
    full = LlamaFusableConfig()
    other = FusableConfig("other").profile("qkv", roles=("q", "k"))
    conflict = FusableConfig("target").merge(full, other)
    for config in (conflict, FusableConfig("target").merge(other, full)):
        with pytest.raises(ValueError, match="Conflicting fusion profile.*replace_profile"):
            config.validate()
        assert config.to_dict()["conflicts"]["profiles"]
    fixed = conflict.replace_profile("qkv", roles=("q", "k", "v"))
    assert fixed.validate().profiles == full.profiles
    assert fixed.profile_origins("qkv") == ("target",)
    assert not fixed.to_dict()["conflicts"]["profiles"]
    assert fixed.merge(other).validate().profiles == fixed.profiles
    with pytest.raises(ValueError, match="already exists"):
        fixed.profile("qkv", roles=("q", "k"))
    with pytest.raises(ValueError, match="Unknown fusion profile"):
        fixed.replace_profile("missing", roles=("q", "k"))


def test_policy_conflicts_need_explicit_target_override_in_either_source_order():
    disabled = LlamaFusableConfig(overrides={"qkv": False})
    enabled = LlamaFusableConfig(overrides={"qkv": True})
    for sources in ((disabled, enabled), (enabled, disabled)):
        conflict = FusableConfig("target").merge(*sources)
        with pytest.raises(ValueError, match="Conflicting fusion policy.*override"):
            conflict.validate()
        fixed = conflict.override("qkv", enabled=True)
        assert fixed.validate().overrides["qkv"] is True
        assert not fixed.to_dict()["conflicts"]["policy"]
        before = FusableConfig("target").override("qkv", enabled=True).merge(*sources)
        assert before.validate().overrides["qkv"] is True
    assert disabled.overrides["qkv"] is False


def test_scoped_policies_merge_by_layer_and_reject_overlapping_local_conflicts():
    left = FusableConfig("left").merge(LlamaFusableConfig()).override("qkv", enabled=False, layers=(0, 1))
    right = FusableConfig("right").merge(LlamaFusableConfig()).override("qkv", enabled=True, layers=(1, 2))
    conflict = FusableConfig("target").merge(left, right)
    with pytest.raises(ValueError, match="Conflicting fusion policy"):
        conflict.validate()
    fixed = conflict.override("qkv", enabled=False, layers=(1,))
    fixed.validate()
    assert [_policy(fixed, i)[0] for i in range(4)] == [False, False, True, True]
    with pytest.raises(ValueError, match="Conflicting local fusion policy"):
        fixed.override("qkv", enabled=True, layers=(3, 1))
    assert _policy(fixed, 3)[0] is True  # A rejected override did not partially mutate the configuration.
    assert fixed.override("qkv", enabled=False, layers=(1,)).to_dict() == fixed.to_dict()


def test_layer_specificity_runtime_precedence_and_provenance():
    config = (FusableConfig("target").merge(LlamaFusableConfig())
              .override("qkv", enabled=False).override("qkv", enabled=True, layers=(1,)))
    assert _policy(config, 0)[0] is False and _policy(config, 1)[0] is True
    assert "qkv[1]" in _policy(config, 1)[1]
    assert _policy(config, 1, overrides={"qkv": False}) == (False, "runtime: qkv")
    assert _policy(config, 0, overrides={"self_attn": True}) == (True, "runtime: self_attn")
    snapshot = json.loads(json.dumps(config.to_dict()))
    assert snapshot["layer_overrides"] == [{"profile": "qkv", "layer": 1, "enabled": True, "origins": ["target"]}]
    assert snapshot["profile_origins"]["qkv"] == ["llama"]


def test_alias_variants_remain_target_bound_and_can_be_selected_explicitly():
    base = LlamaFusableConfig()
    source = replace(base, model_types=("llama", "attention"), model_profiles={"attention": (base.profiles[0],)})
    aliases = FusableConfig(("llama", "attention")).merge(source)
    assert len(aliases.profiles_for("llama")) == 2 and len(aliases.profiles_for("attention")) == 1
    wrapper = FusableConfig("wrapper").merge(source, source_model_type="attention")
    assert wrapper.profiles_for("wrapper") == (base.profiles[0],)
    assert wrapper.model_types == ("wrapper",)
    with pytest.raises(ValueError, match="does not support"):
        FusableConfig("wrapper").merge(source, source_model_type="missing")


def test_unused_default_profiles_do_not_create_false_alias_conflicts():
    base = LlamaFusableConfig()
    shorter = FusableConfig("attention").profile("qkv", roles=("q", "k"))
    source = replace(base, model_types=("llama", "attention"), model_profiles={"attention": shorter.profiles})
    merged = FusableConfig("attention").merge(shorter, source)
    assert merged.validate().profiles_for("attention")[0].roles == ("q", "k")


@pytest.mark.parametrize("kwargs", [{"profiles": ()}, {"profiles": "qkv"}, {"profiles": {}},
                                    {"profiles": ("qkv", "qkv")}, {"profiles": (object(),)},
                                    {"profiles": ("unknown",)}])
def test_invalid_profile_selection_rejects(kwargs):
    with pytest.raises((TypeError, ValueError)):
        FusableConfig("target").merge(LlamaFusableConfig(), **kwargs)


@pytest.mark.parametrize("sources", [(), (object(),), (LlamaFusableConfig, )])
def test_invalid_merge_sources_reject(sources):
    with pytest.raises(TypeError, match="FusableConfig instances"):
        FusableConfig("target").merge(*sources)


def test_multiple_sources_require_separate_selection_calls():
    with pytest.raises(ValueError, match="exactly one merge source"):
        FusableConfig("target").merge(LlamaFusableConfig(), LlamaFusableConfig(), profiles=("qkv",))
    with pytest.raises(ValueError, match="at least one"):
        FusableConfig("target").merge(FusableConfig("incomplete"))


@pytest.mark.parametrize("kwargs", [{"enabled": 1}, {"enabled": False, "layers": ()},
                                    {"enabled": False, "layers": (-1,)}, {"enabled": False, "layers": (True,)},
                                    {"enabled": False, "layers": "1"}])
def test_invalid_local_override_rejects(kwargs):
    with pytest.raises((TypeError, ValueError)):
        FusableConfig("target").merge(LlamaFusableConfig()).override("qkv", **kwargs)


def test_unknown_override_profile_rejects_at_binding():
    config = FusableConfig("target").merge(LlamaFusableConfig()).override("misspelled", enabled=False)
    with pytest.raises(ValueError, match="Unknown fusion profiles"):
        config.validate()


@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_composed_layer_policy_reaches_real_tree_consumers_and_printed_plan(monkeypatch, method):
    model, _ = _wrapper(method, canonical=True, kv_heads=4)
    config = (FusableConfig("llama").merge(LlamaFusableConfig(), profiles=("qkv",))
              .profile("gate_up", roles=("gate", "up"))
              .override("qkv", enabled=False, layers=(0,)))
    monkeypatch.setattr(LlamaQModel, "fusable", config)
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        before = model.model(ids).logits
        plan = model.fuse()
        after = model.model(ids).logits
    assert len(plan.groups) == 3
    assert {g.decision.parent for g in plan.groups} == {"model.layers.0.mlp", "model.layers.1.mlp", "model.layers.1.self_attn"}
    skipped = next(d for d in plan.decisions if d.parent == "model.layers.0.self_attn")
    assert not skipped.enabled and skipped.profile_origins == ("llama",) and "qkv[0]" in skipped.policy
    assert "qkv[0]" in plan.render() and "from: llama" in plan.render()
    assert json.loads(json.dumps(plan.to_dict()))["fusable"] == config.to_dict()
    torch.testing.assert_close(after, before, rtol=2e-3, atol=2e-3)
    model.fuse(False)


def test_unresolved_merge_rejected_before_replacing_installed_fusion(monkeypatch):
    model, _ = _wrapper()
    plan = model.fuse()
    conflict = FusableConfig("llama").merge(LlamaFusableConfig(), FusableConfig("other").profile("qkv", roles=("q", "k")))
    monkeypatch.setattr(LlamaQModel, "fusable", conflict)
    with pytest.raises(ValueError, match="Conflicting fusion profile"):
        require_fusable_config(LlamaQModel, "llama")
    with pytest.raises(ValueError, match="Conflicting fusion profile"):
        model.fuse()
    assert model._fusion_plan is plan and all(group.packed is not None for group in plan.groups)
    model.fuse(False)
