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
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.laguna import LagunaQModel
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks
from gptqmodel.quantization.config import (
    BaseMoERouting,
    ExpertsRoutingBypass,
    MoEConfig,
    QuantizeConfig,
)
from gptqmodel.utils.model import find_modules


class _NoOpProcessor:
    def preprocess(self, named_module, fallback=None, **kwargs):
        return None

    def is_skipped(self, named_module):
        return False


def test_llama_build_layer_modules_caches_module_tree_flags():
    """build_layer_modules populates role flags without emitting them in names."""

    blocks = LlamaQModel.build_layer_modules(LlamaQModel.module_tree)
    qkv_block = next(b for b in blocks if "self_attn.q_proj" in b)
    assert qkv_block == ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]

    assert LlamaQModel.get_module_tree_flags("self_attn.q_proj") == frozenset({"q"})
    assert LlamaQModel.get_module_tree_flags("self_attn.k_proj") == frozenset({"k"})
    assert LlamaQModel.get_module_tree_flags("self_attn.v_proj") == frozenset({"v"})
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

    assert LagunaQModel.get_module_tree_flags("mlp.experts.0.gate_proj") == frozenset({"gate", "moe", "routed"})
    assert LagunaQModel.get_module_tree_flags("mlp.experts.2.up_proj") == frozenset({"up", "moe", "routed"})
    assert LagunaQModel.get_module_tree_flags("mlp.experts.3.down_proj") == frozenset({"down", "moe", "routed"})
    assert LagunaQModel.get_module_tree_flags("mlp.shared_expert.gate_proj") == frozenset({"gate", "moe", "shared"})
    assert LagunaQModel.get_module_tree_flags("mlp.shared_experts.up_proj") == frozenset({"up", "moe", "shared"})


def test_build_layer_modules_direct_expert_placeholder():
    """A tree with ``experts: {"#": "#"}`` uses the template parent directly."""

    tree = ["model", "layers", "#", {"mlp": {"experts": {"#": "#"}}}]
    blocks = BaseQModel._build_layer_modules_for_tree(tree)
    assert blocks == [["mlp.experts.{expert_index}"]]
    assert BaseQModel.get_module_tree_flags("mlp.experts.{expert_index}") == frozenset({"routed"})


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
        moe=MoEConfig(routing=BaseMoERouting()),
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
    assert processor._module_expert_isolation_key("mlp.experts.0.gate_proj") == ("experts", 0)
    assert processor._module_expert_isolation_key("mlp.shared_experts.0.gate_proj") is None
    assert processor._module_expert_isolation_key("mlp.gate_proj") is None

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
    assert processor._module_is_expert_down_proj("mlp.experts.0.down_proj") is True
    assert processor._module_is_expert_down_proj("mlp.experts.0.gate_proj") is False


def test_module_is_moe_related_uses_module_tree_flags_and_fallback():
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

    # When no tasks are present, fall back to the model definition.
    processor.tasks = {}
    processor.gptq_model = object.__new__(LagunaQModel)
    assert processor._module_is_moe_related("model.layers.0.mlp.gate") is True
    assert processor._module_is_moe_related("model.layers.0.mlp.experts.0.gate_proj") is True
    assert processor._module_is_moe_related("model.layers.0.self_attn.q_proj") is False

    # A model without module_tree MoE flags returns False.
    processor.gptq_model = object.__new__(LlamaQModel)
    assert processor._module_is_moe_related("model.layers.0.mlp.gate_proj") is False
