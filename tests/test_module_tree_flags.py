# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for module-tree role flags used by the looper and processors."""

import re
from types import SimpleNamespace

import pytest
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.laguna import LagunaQModel
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks
from gptqmodel.quantization.config import (
    ExpertsRoutingBypass,
    MoEConfig,
    MoERoutingConfig,
    QuantizeConfig,
)
from gptqmodel.utils.model import find_modules


class _NoOpProcessor:
    def preprocess(self, named_module, fallback=None, **kwargs):
        return None

    def is_skipped(self, named_module):
        return False


def _routed_expert_replay_declarations(node):
    declarations = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                parts = key.split(":")
                aliases = parts[0].split("|")
                if "experts" in aliases and "routed" in parts:
                    declarations.append(
                        (key, [part for part in parts[1:] if part.startswith("expert_") and "=" in part])
                    )
            declarations.extend(_routed_expert_replay_declarations(value))
    elif isinstance(node, (list, tuple)):
        for value in node:
            declarations.extend(_routed_expert_replay_declarations(value))
    return declarations


def test_llama_build_layer_modules_caches_module_tree_flags():
    """build_layer_modules populates role flags without emitting them in names."""

    blocks = LlamaQModel.build_layer_modules(LlamaQModel.module_tree)
    qkv_block = next(b for b in blocks if "self_attn.q_proj" in b)
    assert qkv_block == ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]

    assert LlamaQModel.get_module_tree_flags("self_attn.q_proj") == frozenset({"q"})
    assert LlamaQModel.get_module_tree_flags("self_attn.k_proj") == frozenset({"k"})
    assert LlamaQModel.get_module_tree_flags("self_attn.v_proj") == frozenset({"v"})
    assert LlamaQModel.get_module_tree_flags("self_attn.o_proj") == frozenset({"o"})
    assert LlamaQModel.get_module_tree_flags("mlp.gate_proj") == frozenset({"gate"})
    assert LlamaQModel.get_module_tree_flags("mlp.up_proj") == frozenset({"up"})
    assert LlamaQModel.get_module_tree_flags("mlp.down_proj") == frozenset({"down"})
    assert LlamaQModel.get_module_tree_flags("nonexistent.path") == frozenset()


def test_laguna_moe_expansion_caches_role_flags():
    """MoE placeholder expansion copies role flags to each concrete expert path."""

    class Cfg:
        num_experts = 4

    qcfg = SimpleNamespace(dynamic=None)
    simple = LagunaQModel.simple_layer_modules(Cfg(), qcfg)

    gate_block = next(b for b in simple if "mlp.experts.0.gate_proj" in b)
    assert "mlp.experts.0.gate_proj" in gate_block

    assert LagunaQModel.get_module_tree_flags("mlp.experts.0.gate_proj") == frozenset(
        {"expert_activation=experts.act_fn", "gate", "routed"}
    )
    assert LagunaQModel.get_module_tree_flags("mlp.experts.2.up_proj") == frozenset(
        {"expert_activation=experts.act_fn", "routed", "up"}
    )
    assert LagunaQModel.get_module_tree_flags("mlp.experts.3.down_proj") == frozenset(
        {"down", "expert_activation=experts.act_fn", "routed"}
    )
    assert LagunaQModel.get_module_tree_flags("mlp.shared_expert.gate_proj") == frozenset({"gate", "shared"})
    assert LagunaQModel.get_module_tree_flags("mlp.shared_experts.up_proj") == frozenset({"up", "shared"})
    assert LagunaQModel.get_module_tree_expert_group("mlp.experts.2.up_proj") == "mlp.experts.2"
    assert LagunaQModel.get_module_tree_expert_group("mlp.shared_expert.gate_proj") == "mlp.shared_expert"


def test_all_moe_module_trees_declare_exact_expert_replay():
    """Every routed expert tree names its exact activation, gate, or forward owner."""

    checked_classes = set()
    for model_class in MODEL_MAP.values():
        if model_class in checked_classes or getattr(model_class, "moe_lifecycle_hooks", None) is None:
            continue
        checked_classes.add(model_class)
        routed_nodes = _routed_expert_replay_declarations(model_class.module_tree)
        assert routed_nodes, f"{model_class.__name__} has lifecycle hooks but no routed expert tree"
        for node, declarations in routed_nodes:
            assert len(declarations) == 1, f"{model_class.__name__} {node} has replay declarations {declarations}"


def test_all_model_tree_expert_flags_are_normalized():
    """Routed/shared roles imply MoE and never coexist with a redundant moe role."""

    checked_classes = set()
    for model_class in MODEL_MAP.values():
        if model_class in checked_classes or model_class.module_tree is None:
            continue
        checked_classes.add(model_class)
        model_class.build_layer_modules(model_class.module_tree, include_capture_only=True)
        for block in model_class.full_layer_modules(include_capture_only=True):
            for raw_name in block:
                name, _ = model_class._parse_module_flags(raw_name)
                flags = model_class.get_module_tree_flags(name)
                assert not ({"routed", "shared"} <= flags), (model_class.__name__, name, flags)
                assert not ("moe" in flags and flags & {"routed", "shared"}), (
                    model_class.__name__,
                    name,
                    flags,
                )


def test_build_layer_modules_direct_expert_placeholder():
    """A direct expert placeholder inherits only explicitly declared structural roles."""

    class ExplicitRoutedTreeQModel(BaseQModel):
        """Test model whose expert role is explicitly declared in its module tree."""

    tree = ["model", "layers", "#", {"mlp": {"experts:routed": {"#": "#"}}}]
    blocks = ExplicitRoutedTreeQModel._build_layer_modules_for_tree(tree)
    assert blocks == [["mlp.experts.{expert_index}"]]
    assert ExplicitRoutedTreeQModel.get_module_tree_flags("mlp.experts.{expert_index}") == frozenset({"routed"})
    assert ExplicitRoutedTreeQModel.get_module_tree_expert_group(
        "mlp.experts.{expert_index}"
    ) == "mlp.experts.{expert_index}"


def test_build_layer_modules_does_not_infer_expert_roles_from_names():
    """Names such as experts/shared_experts must not manufacture structural flags."""

    class UntaggedExpertTreeQModel(BaseQModel):
        """Test model with deliberately untagged expert-like module names."""

    tree = [
        "model",
        "layers",
        "#",
        {
            "mlp:moe": {
                "experts": {"#": ("gate_proj:gate",)},
                "shared_experts": ("up_proj:up",),
            }
        },
    ]
    UntaggedExpertTreeQModel._build_layer_modules_for_tree(tree)

    assert UntaggedExpertTreeQModel.get_module_tree_flags(
        "mlp.experts.{expert_index}.gate_proj"
    ) == frozenset({"gate", "moe"})
    assert UntaggedExpertTreeQModel.get_module_tree_flags("mlp.shared_experts.up_proj") == frozenset(
        {"moe", "up"}
    )


def test_arbitrary_names_use_explicit_expert_roles_and_groups():
    """Expert semantics remain exact when no path contains conventional MoE words."""

    class ArbitraryNamesQModel(BaseQModel):
        """Test model whose semantic roles cannot be inferred from its names."""

    tree = [
        "model", "layers", "#",
        {
            "ffn:moe": {
                "specialists:routed": {"#": ("left:gate", "right:up", "reduce:down")},
                "always_on:shared": ("left:gate", "right:up", "reduce:down"),
            },
            "experts": ("gate_proj:gate",),
        },
    ]
    ArbitraryNamesQModel._build_layer_modules_for_tree(tree)

    routed_path = "ffn.specialists.{expert_index}.left"
    assert ArbitraryNamesQModel.get_module_tree_flags(routed_path) == frozenset({"gate", "routed"})
    assert ArbitraryNamesQModel.get_module_tree_expert_group(routed_path) == "ffn.specialists.{expert_index}"
    assert ArbitraryNamesQModel.get_module_tree_flags("ffn.always_on.reduce") == frozenset({"down", "shared"})
    assert ArbitraryNamesQModel.get_module_tree_expert_group("ffn.always_on.reduce") == "ffn.always_on"
    assert ArbitraryNamesQModel.get_module_tree_flags("experts.gate_proj") == frozenset({"gate"})
    assert ArbitraryNamesQModel.get_module_tree_expert_group("experts.gate_proj") is None

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = ArbitraryNamesQModel
    assert looper._extract_moe_group_key(routed_path) == "ffn.specialists.{expert_index}"
    assert looper._extract_moe_group_key("experts.gate_proj") is None


def test_module_tree_rejects_ambiguous_expert_roles():
    """One node cannot be both a routed and shared expert container."""

    class AmbiguousExpertQModel(BaseQModel):
        """Deliberately invalid test declaration."""

    tree = ["model", "layers", "#", {"ffn:moe": {"specialists:routed:shared": ("left:gate",)}}]
    with pytest.raises(ValueError, match="both routed and shared"):
        AmbiguousExpertQModel._build_layer_modules_for_tree(tree)


def test_module_looper_create_named_modules_sets_module_tree_flags():
    """create_named_modules stores the role flags on NamedModule.state."""

    cfg = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
    )
    model = LlamaForCausalLM(cfg)
    layer = model.model.layers[0]
    full = find_modules(layer)

    qcfg = SimpleNamespace(dynamic=None)
    blocks = LlamaQModel.simple_layer_modules(cfg, qcfg)
    qkv_block = next(b for b in blocks if "self_attn.q_proj" in b)

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = LlamaQModel
    looper.input_embeddings_name = "model.embed_tokens"
    looper.output_embeddings_name = "lm_head"

    subset = looper.create_named_modules(
        module=layer,
        full=full,
        is_lm_head_module=False,
        layer_index=0,
        layers_prefix="model.layers",
        names=qkv_block,
        processor=_NoOpProcessor(),
        fallback=None,
        layer_module=None,
    )

    assert set(subset) == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
    assert subset["self_attn.q_proj"].state["module_tree_flags"] == frozenset({"q"})
    assert subset["self_attn.k_proj"].state["module_tree_flags"] == frozenset({"k"})
    assert subset["self_attn.v_proj"].state["module_tree_flags"] == frozenset({"v"})


def test_module_looper_create_named_modules_skips_not_quantize_flags():
    """Tokens with the ``!`` flag are skipped and not added to the subset."""

    cfg = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
    )
    model = LlamaForCausalLM(cfg)
    layer = model.model.layers[0]
    full = find_modules(layer)

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = LlamaQModel
    looper.input_embeddings_name = "model.embed_tokens"
    looper.output_embeddings_name = "lm_head"

    subset = looper.create_named_modules(
        module=layer,
        full=full,
        is_lm_head_module=False,
        layer_index=0,
        layers_prefix="model.layers",
        names=["self_attn.q_proj", "self_attn.o_proj:!"],
        processor=_NoOpProcessor(),
        fallback=None,
        layer_module=None,
    )

    assert set(subset) == {"self_attn.q_proj"}


def test_module_looper_create_named_modules_handles_capture_only_and_errors():
    """Capture-only tokens are resolved by prefix; missing modules raise when strict."""

    cfg = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
    )
    model = LlamaForCausalLM(cfg)
    layer = model.model.layers[0]
    full = find_modules(layer)

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = LlamaQModel
    looper.input_embeddings_name = "model.embed_tokens"
    looper.output_embeddings_name = "lm_head"

    # Capture-only token resolves a non-leaf module by prefix.
    subset = looper.create_named_modules(
        module=layer,
        full=full,
        is_lm_head_module=False,
        layer_index=0,
        layers_prefix="model.layers",
        names=["self_attn.q_proj", "self_attn:?"],
        processor=_NoOpProcessor(),
        fallback=None,
        layer_module=None,
    )
    assert "self_attn" in subset
    assert subset["self_attn"].state["capture_only"] is True

    # Missing module name raises because LlamaQModel is strict by default.
    with pytest.raises(ValueError, match=re.escape("layer module item `nonexistent` not found")):
        looper.create_named_modules(
            module=layer,
            full=full,
            is_lm_head_module=False,
            layer_index=0,
            layers_prefix="model.layers",
            names=["nonexistent"],
            processor=_NoOpProcessor(),
            fallback=None,
            layer_module=None,
        )

    # Non-string tokens take the ``name, flags = token, []`` path and then raise.
    with pytest.raises(ValueError, match="layer module item"):
        looper.create_named_modules(
            module=layer,
            full=full,
            is_lm_head_module=False,
            layer_index=0,
            layers_prefix="model.layers",
            names=[nn.Linear(4, 4)],
            processor=_NoOpProcessor(),
            fallback=None,
            layer_module=None,
        )


def test_module_looper_init_assigns_gptq_model_to_processors():
    """ModuleLooper wires every processor's ``gptq_model`` reference during init."""

    cfg = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
    )
    llama = LlamaForCausalLM(cfg)
    gptq_model = object.__new__(LlamaQModel)
    nn.Module.__init__(gptq_model)
    gptq_model.model = llama
    gptq_model.quantize_config = QuantizeConfig(bits=4, group_size=128, device="cpu")
    gptq_model.dynamic_expert_index = None

    processor = LoopProcessor.__new__(LoopProcessor)
    processor.qcfg = gptq_model.quantize_config

    looper = ModuleLooper(gptq_model, [processor])
    assert processor.gptq_model is gptq_model
    assert looper.gptq_model is gptq_model


def test_loop_processor_module_tree_helpers():
    """LoopProcessor exposes generic MoE/module-tree helpers used by all processors."""

    processor = LoopProcessor.__new__(LoopProcessor)

    # _is_bypass_moe_routing
    processor.qcfg = QuantizeConfig(bits=4, group_size=8, sym=True, desc_act=False)
    assert processor._is_bypass_moe_routing() is False

    processor.qcfg = QuantizeConfig(
        bits=4,
        group_size=8,
        sym=True,
        desc_act=False,
        moe=MoEConfig(routing=MoERoutingConfig()),
    )
    assert processor._is_bypass_moe_routing() is False

    processor.qcfg = QuantizeConfig(
        bits=4,
        group_size=8,
        sym=True,
        desc_act=False,
        moe=MoEConfig(routing=ExpertsRoutingBypass()),
    )
    assert processor._is_bypass_moe_routing() is True

    # _module_expert_isolation_key
    processor.gptq_model = None
    assert processor._module_expert_isolation_key("mlp.experts.0.gate_proj") is None

    processor.gptq_model = SimpleNamespace(moe_lifecycle_hooks=GateUpDownMoELifecycleHooks())
    assert processor._module_expert_isolation_key("mlp.experts.0.gate_proj") is None
    assert processor._module_expert_isolation_key("mlp.shared_experts.0.gate_proj") is None
    assert processor._module_expert_isolation_key("mlp.gate_proj") is None

    routed = NamedModule(nn.Linear(4, 4), name="mlp.specialists.0.proj_a", full_name="routed", layer_index=0)
    routed.state["module_tree_flags"] = frozenset({"gate", "routed"})
    routed.state["module_tree_expert_group"] = "mlp.specialists.0"
    processor.tasks = {routed.name: SimpleNamespace(_named_module=routed)}
    assert processor._module_expert_isolation_key(routed.name) == "mlp.specialists.0"

    # _module_tree_flags
    module = nn.Linear(4, 4)
    named = NamedModule(module, name="a", full_name="b", layer_index=0)
    named.state["module_tree_flags"] = frozenset({"down"})
    assert processor._module_tree_flags(SimpleNamespace(_named_module=named)) == frozenset({"down"})
    assert processor._module_tree_flags(SimpleNamespace(_named_module="not_named")) == frozenset()
    assert processor._module_tree_flags(SimpleNamespace()) == frozenset()

    # _module_is_expert_down_proj
    processor.gptq_model = None
    assert processor._module_is_expert_down_proj("mlp.experts.0.down_proj") is False

    processor.gptq_model = SimpleNamespace(moe_lifecycle_hooks=GateUpDownMoELifecycleHooks())
    assert processor._module_is_expert_down_proj("mlp.experts.0.down_proj") is False
    assert processor._module_is_expert_down_proj("mlp.experts.0.gate_proj") is False

    down = NamedModule(nn.Linear(4, 4), name="mlp.specialists.0.proj_c", full_name="down", layer_index=0)
    down.state["module_tree_flags"] = frozenset({"down", "routed"})
    down.state["module_tree_expert_group"] = "mlp.specialists.0"
    processor.tasks = {down.name: SimpleNamespace(_named_module=down)}
    assert processor._module_is_expert_down_proj(down.name) is True


def test_module_is_moe_related_uses_only_named_module_tree_flags():
    """MoE membership is derived from module_tree flags, not name patterns."""

    processor = LoopProcessor.__new__(LoopProcessor)
    processor.gptq_model = None

    module = nn.Linear(4, 4)
    named = NamedModule(module, name="a", full_name="b", layer_index=0)
    named.state["module_tree_flags"] = frozenset({"moe"})

    # When the wrapped module is in the processor's task map, use its flags.
    processor.tasks = {"model.layers.0.mlp.gate": SimpleNamespace(_named_module=named)}
    assert processor._module_is_moe_related("model.layers.0.mlp.gate") is True

    # A dense path (no moe/routed/shared flag) is not treated as MoE.
    named.state["module_tree_flags"] = frozenset({"gate"})
    assert processor._module_is_moe_related("model.layers.0.mlp.gate_proj") is False

    # Names never manufacture membership when task metadata is absent.
    processor.tasks = {}
    processor.gptq_model = object.__new__(LagunaQModel)
    assert processor._module_is_moe_related("model.layers.0.mlp.gate") is False
    assert processor._module_is_moe_related("model.layers.0.mlp.experts.0.gate_proj") is False
    assert processor._module_is_moe_related("model.layers.0.self_attn.q_proj") is False

    # A model without module_tree MoE flags returns False.
    processor.gptq_model = object.__new__(LlamaQModel)
    assert processor._module_is_moe_related("model.layers.0.mlp.gate_proj") is False
