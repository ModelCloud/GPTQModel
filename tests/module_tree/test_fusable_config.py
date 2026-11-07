# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Definition-owned fusion admission and extension on real model classes."""

import copy
import json
from dataclasses import FrozenInstanceError, dataclass, replace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from gptqmodel import FusableConfig, FusableProfile, LlamaFusableConfig
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.definitions.qwen2 import Qwen2QModel
from gptqmodel.models.fusable import require_fusable_config
from gptqmodel.models.fusion import FusionPlanner
from tests.module_tree.test_llama_fusion import _wrapper


class NonFusableLlamaQModel(LlamaQModel):
    fusable = None


class IndependentMlp(torch.nn.Module):
    """A real runtime parent whose names/class are absent from any allowlist."""

    def __init__(self, native):
        super().__init__()
        self.branch_alpha = native.gate_proj
        self.branch_beta = native.up_proj
        self.output = native.down_proj
        self.activation = native.act_fn

    def forward(self, x):
        return self.output(self.activation(self.branch_alpha(x)) * self.branch_beta(x))


class IndependentPrefixMlp(IndependentMlp):
    def __init__(self, native):
        super().__init__(native)
        self.prefix_calls = 0

    def forward(self, x):
        self.prefix_calls += 1
        return self.output(self.activation(self.branch_alpha(x)) * self.branch_beta(x))


@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("parent_type", [IndependentMlp, IndependentPrefixMlp])
def test_live_parent_and_renamed_projections_resolve_only_from_definition_tree(method, parent_type):
    source, _ = _wrapper(method)
    for layer in source.model.model.layers:
        layer.compute_unit = parent_type(layer.mlp).eval()
        del layer.mlp

    class RenamedDefinition(LlamaQModel):
        fusable = FusableConfig("llama").merge(LlamaFusableConfig(), profiles=("gate_up",))
        module_tree = copy.deepcopy(LlamaQModel.module_tree)

    mapping = RenamedDefinition.module_tree[-1]
    del mapping["mlp"]
    mapping["compute_unit"] = ("branch_alpha:0:gate:in=x", "branch_beta:0:up:in=x", "output:1:down")
    model = RenamedDefinition(source.model, quantized=True, quantize_config=source.quantize_config)
    inputs = torch.randn(1, 3, 64, dtype=torch.float16)
    with torch.inference_mode():
        before = [layer.compute_unit(inputs) for layer in model.model.model.layers]
        plan = model.fuse()
        after = [layer.compute_unit(inputs) for layer in model.model.model.layers]
    assert len(plan.groups) == 2
    assert all(type(group.parent) is parent_type and group.calls == 1 for group in plan.groups)
    assert all(group.decision.paths == ("compute_unit.branch_alpha", "compute_unit.branch_beta")
               for group in plan.groups)
    assert all(set(profile) == {"name", "roles"} for profile in plan.to_dict()["fusable"]["profiles"])
    for actual, expected in zip(after, before):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    if parent_type is IndependentPrefixMlp:
        group = plan.groups[0]
        assert group.parent.prefix_calls == 2
        handle = group.modules[0].register_forward_pre_hook(lambda _module, args: (args[0] + 0.25,))
        try:
            with torch.inference_mode():
                fallback = group.parent(inputs)
                assert group.parent.prefix_calls == 3  # Preamble executes once on fallback too.
                expected = group.native_forward(inputs)
            torch.testing.assert_close(fallback, expected, rtol=0, atol=0)
            assert group.calls == 1 and group.packed is None
        finally:
            handle.remove()
    model.fuse(False)


def test_llama_profile_is_immutable_and_public():
    overrides = {"qkv": False}
    config = LlamaFusableConfig(overrides=overrides)
    overrides["qkv"] = True
    assert config.overrides["qkv"] is False
    assert isinstance(config, FusableConfig)
    assert config.model_types == ("llama",)
    assert "llama" in LlamaQModel.fusable.model_types
    assert [(p.name, p.roles) for p in config.profiles] == [("qkv", ("q", "k", "v")),
                                                         ("gate_up", ("gate", "up"))]
    with pytest.raises(FrozenInstanceError):
        config.model_types = ("other",)
    with pytest.raises(TypeError):
        config.overrides["qkv"] = True
    assert copy.deepcopy(config) is config
    assert json.loads(json.dumps(config.to_dict()))["profiles"][0]["roles"] == ["q", "k", "v"]
    assert all(set(profile) == {"name", "roles"} for profile in config.to_dict()["profiles"])


@pytest.mark.parametrize("kwargs", [
    {"model_types": ()}, {"model_types": ("llama", "llama")},
    {"model_types": ("",)}, {"profiles": (object(),)},
    {"profiles": (LlamaFusableConfig().profiles[0],) * 2},
    {"overrides": {"qkv": "yes"}}, {"overrides": {"": False}}, {"overrides": []},
])
def test_invalid_configs_reject(kwargs):
    with pytest.raises((TypeError, ValueError)):
        LlamaFusableConfig(**kwargs)


@pytest.mark.parametrize("roles", [("q",), ("q", "q"), "qkv", ("", "k"), ("q", object())])
def test_invalid_profile_rejects(roles):
    with pytest.raises((TypeError, ValueError)):
        FusableProfile("qkv", roles)


def test_missing_definition_profile_rejects_without_mutation():
    model, _ = _wrapper(definition=NonFusableLlamaQModel)
    leaves = list(model.model.model.layers[0].self_attn.children())
    forwards = [module.forward for module in leaves]
    weights = [module.qweight for module in leaves]
    with pytest.raises(ValueError, match="not fusable.*fusable"):
        model.fuse()
    with pytest.raises(ValueError, match="not fusable"):
        FusionPlanner(model)
    # Admission belongs to the definition, not a runtime instance attribute.
    model.fusable = LlamaFusableConfig()
    with pytest.raises(ValueError, match="not fusable"):
        model.fuse()
    assert [module.forward for module in leaves] == forwards
    assert all(module.qweight is weight for module, weight in zip(leaves, weights))
    assert model._fusion_plan is None and not model._auto_fusion
    assert model.fuse(False) is None


def test_explicit_load_and_quantization_opt_in_reject_before_work():
    with pytest.raises(ValueError, match="not fusable"):
        _wrapper(definition=NonFusableLlamaQModel, automatic=True)
    model, _ = _wrapper(definition=NonFusableLlamaQModel, quantized=False)
    with pytest.raises(ValueError, match="not fusable"):
        model.quantize([], fuse=True)
    assert not model._auto_fusion and not model.quantized


@pytest.mark.parametrize("value", [{}, LlamaFusableConfig])
def test_definition_requires_a_config_instance(monkeypatch, value):
    model, _ = _wrapper()
    monkeypatch.setattr(LlamaQModel, "fusable", value)
    with pytest.raises(TypeError, match="FusableConfig instance"):
        model.fuse()


def test_failed_admission_leaves_existing_fusion_installed(monkeypatch):
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    packed = plan.groups[0].packed
    monkeypatch.setattr(LlamaQModel, "fusable", None)
    with pytest.raises(ValueError, match="not fusable"):
        model.fuse()
    assert model._fusion_plan is plan and plan.groups[0].packed is packed
    model.fuse(False)
    assert model._fusion_plan is None and plan.groups[0].packed is None


def test_unsupported_model_policy_rejects_before_removing_existing_fusion(monkeypatch):
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    packed = plan.groups[0].packed
    bad = FusableConfig("other").merge(LlamaFusableConfig())
    monkeypatch.setattr(LlamaQModel, "fusable", bad)
    with pytest.raises(ValueError, match="does not support model_type"):
        model.fuse()
    assert model._fusion_plan is plan and plan.groups[0].packed is packed
    model.fuse(False)


def test_inherited_config_can_restrict_profiles_and_override_paths():
    @dataclass(frozen=True)
    class GateOnlyConfig(LlamaFusableConfig):
        profiles: tuple[FusableProfile, ...] = (LlamaFusableConfig().profiles[1],)

    class GateOnlyLlamaQModel(LlamaQModel):
        fusable = GateOnlyConfig(overrides={"model.layers.0.mlp": False})

    model, _ = _wrapper(definition=GateOnlyLlamaQModel)
    plan = model.fuse()
    assert [group.decision.parent for group in plan.groups] == ["model.layers.1.mlp"]
    assert all(d.family == "gate_up" for d in plan.decisions)
    assert plan.to_dict()["fusable"] == GateOnlyLlamaQModel.fusable.to_dict()
    assert len(LlamaQModel.fusable.profiles) == 2
    assert not LlamaQModel.fusable.overrides


def test_overlapping_profiles_reject_before_installing_execution():
    default = LlamaFusableConfig()

    class ConflictingLlamaQModel(LlamaQModel):
        fusable = replace(default, profiles=(*default.profiles, replace(default.profiles[0], name="other_qkv")))

    model, _ = _wrapper(definition=ConflictingLlamaQModel)
    parent = model.model.model.layers[0].self_attn
    forward = parent.forward
    qweight = parent.q_proj.qweight
    with pytest.raises(ValueError, match="Multiple fusion profiles match"):
        model.fuse()
    assert model._fusion_plan is None
    assert parent.forward == forward and parent.q_proj.qweight is qweight


def test_extended_llama_config_consumes_real_qwen2_tree_and_parents():
    source, _ = _wrapper()
    config = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=64)
    config._attn_implementation = "eager"
    actual = Qwen2ForCausalLM(config).half().eval()
    for original, layer in zip(source.model.model.layers, actual.model.layers):
        for path in (spec.template for spec in source.module_plan().specs if spec.quantize and spec.roles):
            linear = layer.get_submodule(path)
            module = original.get_submodule(path)
            if linear.bias is None:
                module.bias = None
            parent, leaf = path.rsplit(".", 1)
            setattr(layer.get_submodule(parent), leaf, module)
    assert require_fusable_config(Qwen2QModel, "qwen2") is Qwen2QModel.fusable

    class UnsupportedQwen2QModel(Qwen2QModel):
        fusable = LlamaFusableConfig()

    unsupported = UnsupportedQwen2QModel(actual, quantized=True, quantize_config=source.quantize_config)
    with pytest.raises(ValueError, match="does not support model_type='qwen2'"):
        unsupported.fuse()

    @dataclass(frozen=True)
    class QwenFusableConfig(LlamaFusableConfig):
        model_types: tuple[str, ...] = ("qwen2",)

    class OptInQwen2QModel(Qwen2QModel):
        fusable = QwenFusableConfig()

    model = OptInQwen2QModel(actual, quantized=True, quantize_config=source.quantize_config)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = model.model(ids).logits
        plan = model.fuse()
        after = model.model(ids).logits
    assert len(plan.groups) == 4
    assert {group.decision.family for group in plan.groups} == {"qkv", "gate_up"}
    torch.testing.assert_close(after, before, rtol=2e-3, atol=2e-3)
    model.fuse(False)
    with torch.no_grad():
        torch.testing.assert_close(model.model(ids).logits, before, rtol=0, atol=0)
    assert require_fusable_config(OptInQwen2QModel, "qwen2") is OptInQwen2QModel.fusable


def test_alias_role_profiles_are_immutable_and_need_no_architecture_imports():
    other = FusableProfile("qkv", ("q", "k", "v"))
    variants = {"other": [other]}
    config = replace(LlamaFusableConfig(), model_types=("llama", "other"), model_profiles=variants)
    variants["other"].clear()

    class AliasedDefinition:
        fusable = config

    assert require_fusable_config(AliasedDefinition, "llama") is config
    assert config.profiles_for("other") == (other,)
    assert require_fusable_config(AliasedDefinition, "other") is config
    with pytest.raises(TypeError):
        config.model_profiles["other"] = ()
    assert FusableConfig("wrapper").merge(config, source_model_type="other").profiles == (other,)
    assert not FusableConfig("wrapper").merge(config, source_model_type="other").model_profiles
    with pytest.raises(ValueError, match="does not support"):
        FusableConfig("wrapper").merge(config, source_model_type="typo")
    with pytest.raises(ValueError, match="undeclared"):
        replace(config, model_profiles={"typo": (other,)})
    with pytest.raises(ValueError, match="distinct"):
        replace(config, model_profiles={"other": (other, other)})
    assert copy.deepcopy(config) is config
    assert json.loads(json.dumps(config.to_dict()))["model_profiles"]["other"][0]["name"] == "qkv"


def test_planner_resolves_alternative_and_duplicate_decoder_prefixes(monkeypatch):
    model, _ = _wrapper()
    monkeypatch.setattr(LlamaQModel, "extract_layers_node", classmethod(
        lambda cls: ["absent.layers", "model.layers", "model.layers"]))
    plan = model.fuse()
    assert len(plan.groups) == 4
    assert {group.decision.parent for group in plan.groups} == {
        "model.layers.0.self_attn", "model.layers.0.mlp", "model.layers.1.self_attn", "model.layers.1.mlp",
    }
    model.fuse(False)
