# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Definition-owned fusion admission and extension on real model classes."""

import copy
import json
from dataclasses import FrozenInstanceError, dataclass, replace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP

from gptqmodel import FusableConfig, FusableProfile, LlamaFusableConfig
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.definitions.qwen2 import Qwen2QModel
from gptqmodel.models.fusable import require_fusable_config
from gptqmodel.models.fusion import FusionPlanner
from tests.module_tree.test_llama_fusion import _wrapper


class NonFusableLlamaQModel(LlamaQModel):
    fusable_config = None


def test_llama_profile_is_immutable_and_public():
    overrides = {"qkv": False}
    config = LlamaFusableConfig(overrides=overrides)
    overrides["qkv"] = True
    assert config.overrides["qkv"] is False
    assert isinstance(config, FusableConfig)
    assert LlamaQModel.fusable_config.model_types == ("llama",)
    assert [(p.name, p.roles) for p in config.profiles] == [("qkv", ("q", "k", "v")),
                                                         ("gate_up", ("gate", "up"))]
    with pytest.raises(FrozenInstanceError):
        config.model_types = ("other",)
    with pytest.raises(TypeError):
        config.overrides["qkv"] = True
    assert copy.deepcopy(config) is config
    assert json.loads(json.dumps(config.to_dict()))["profiles"][0]["roles"] == ["q", "k", "v"]
    for profile in config.profiles:
        assert len(profile.resolve_parent_types()) == 1


@pytest.mark.parametrize("kwargs", [
    {"model_types": ()}, {"model_types": "llama"}, {"model_types": ("llama", "llama")},
    {"model_types": ("",)}, {"profiles": ()}, {"profiles": (object(),)},
    {"profiles": (LlamaFusableConfig().profiles[0],) * 2},
    {"overrides": {"qkv": "yes"}}, {"overrides": {"": False}}, {"overrides": []},
])
def test_invalid_configs_reject(kwargs):
    with pytest.raises((TypeError, ValueError)):
        LlamaFusableConfig(**kwargs)


@pytest.mark.parametrize("roles,parents", [
    (("q",), (Qwen2Attention,)), (("q", "q"), (Qwen2Attention,)),
    ("qkv", (Qwen2Attention,)), (("q", "k"), ()),
    (("q", "k"), ("not_dotted",)), (("q", "k"), "some.Parent"),
])
def test_invalid_profile_rejects(roles, parents):
    with pytest.raises((TypeError, ValueError)):
        FusableProfile("qkv", roles, parents)


def test_missing_definition_profile_rejects_without_mutation():
    model, _ = _wrapper(definition=NonFusableLlamaQModel)
    leaves = list(model.model.model.layers[0].self_attn.children())
    forwards = [module.forward for module in leaves]
    weights = [module.qweight for module in leaves]
    with pytest.raises(ValueError, match="not fusable.*fusable_config"):
        model.fuse()
    with pytest.raises(ValueError, match="not fusable"):
        FusionPlanner(model)
    # Admission belongs to the definition, not a runtime instance attribute.
    model.fusable_config = LlamaFusableConfig()
    with pytest.raises(ValueError, match="not fusable"):
        model.fuse()
    assert [module.forward for module in leaves] == forwards
    assert all(module.qweight is weight for module, weight in zip(leaves, weights))
    assert model.fusion_plan() is None and not model._auto_fusion
    assert model.fuse(False) is model


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
    monkeypatch.setattr(LlamaQModel, "fusable_config", value)
    with pytest.raises(TypeError, match="FusableConfig instance"):
        model.fuse()


def test_failed_admission_leaves_existing_fusion_installed(monkeypatch):
    model, _ = _wrapper(automatic=True)
    plan = model.fusion_plan()
    packed = plan.groups[0].packed
    monkeypatch.setattr(LlamaQModel, "fusable_config", None)
    with pytest.raises(ValueError, match="not fusable"):
        model.fuse()
    assert model.fusion_plan() is plan and plan.groups[0].packed is packed
    model.fuse(False)
    assert model.fusion_plan() is None and plan.groups[0].packed is None


def test_bad_executor_profile_rejects_before_removing_existing_fusion(monkeypatch):
    model, _ = _wrapper(automatic=True)
    plan = model.fusion_plan()
    packed = plan.groups[0].packed
    bad = replace(LlamaFusableConfig(), profiles=(FusableProfile("qkv", ("q", "k", "v"), ("builtins.len",)),))
    monkeypatch.setattr(LlamaQModel, "fusable_config", bad)
    with pytest.raises(TypeError, match="parent must resolve to a class"):
        model.fuse()
    assert model.fusion_plan() is plan and plan.groups[0].packed is packed
    model.fuse(False)


def test_inherited_config_can_restrict_profiles_and_override_paths():
    @dataclass(frozen=True)
    class GateOnlyConfig(LlamaFusableConfig):
        profiles: tuple[FusableProfile, ...] = (LlamaFusableConfig().profiles[1],)

    class GateOnlyLlamaQModel(LlamaQModel):
        fusable_config = GateOnlyConfig(overrides={"model.layers.0.mlp": False})

    model, _ = _wrapper(definition=GateOnlyLlamaQModel)
    model.fuse()
    plan = model.fusion_plan()
    assert [group.decision.parent for group in plan.groups] == ["model.layers.1.mlp"]
    assert all(d.family == "gate_up" for d in plan.decisions)
    assert plan.to_dict()["fusable_config"] == GateOnlyLlamaQModel.fusable_config.to_dict()
    assert len(LlamaQModel.fusable_config.profiles) == 2
    assert not LlamaQModel.fusable_config.overrides


def test_overlapping_profiles_reject_before_installing_execution():
    default = LlamaFusableConfig()

    class ConflictingLlamaQModel(LlamaQModel):
        fusable_config = replace(default, profiles=(*default.profiles, replace(default.profiles[0], name="other_qkv")))

    model, _ = _wrapper(definition=ConflictingLlamaQModel)
    parent = model.model.model.layers[0].self_attn
    forward = parent.forward
    qweight = parent.q_proj.qweight
    with pytest.raises(ValueError, match="Multiple fusion profiles match"):
        model.fuse()
    assert model.fusion_plan() is None
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
    unsupported = Qwen2QModel(actual, quantized=True, quantize_config=source.quantize_config)
    with pytest.raises(ValueError, match="does not support model_type='qwen2'"):
        unsupported.fuse()

    @dataclass(frozen=True)
    class QwenFusableConfig(LlamaFusableConfig):
        model_types: tuple[str, ...] = ("qwen2",)
        profiles: tuple[FusableProfile, ...] = (
            replace(LlamaFusableConfig().profiles[0], parent_types=(Qwen2Attention,)),
            replace(LlamaFusableConfig().profiles[1], parent_types=(Qwen2MLP,)),
        )

    class OptInQwen2QModel(Qwen2QModel):
        fusable_config = QwenFusableConfig()

    model = OptInQwen2QModel(actual, quantized=True, quantize_config=source.quantize_config)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = model.model(ids).logits
        model.fuse()
        after = model.model(ids).logits
    assert len(model.fusion_plan().groups) == 4
    assert {group.decision.family for group in model.fusion_plan().groups} == {"qkv", "gate_up"}
    torch.testing.assert_close(after, before, rtol=2e-3, atol=2e-3)
    model.fuse(False)
    with torch.no_grad():
        torch.testing.assert_close(model.model(ids).logits, before, rtol=0, atol=0)
    assert require_fusable_config(OptInQwen2QModel, "qwen2") is OptInQwen2QModel.fusable_config
