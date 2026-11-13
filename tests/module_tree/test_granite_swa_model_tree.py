# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-model module-tree coverage for Granite SWA and Granite MoE SWA.

Both definitions are built from the installed Transformers model classes on the
``meta`` device, converted through the registered Defuser pipeline, and then
checked against the same plan consumers used by quantization.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch
import transformers

from gptqmodel.models._const import EXPERT_INDEX_PLACEHOLDER
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


QC = SimpleNamespace(dynamic=None)
MODEL_TYPES = ("granite_swa", "granitemoe_swa")
MOE_ROOT = "block_sparse_moe"
EXPERTS = "block_sparse_moe.experts"


def _config(model_type: str):
    config = transformers.AutoConfig.for_model(model_type)
    config.vocab_size = 128
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.pad_token_id = None
    config.bos_token_id = None
    config.eos_token_id = None
    config.layer_types = ["full_attention", "sliding_attention"]
    config.sliding_window = 16
    if hasattr(config, "num_local_experts"):
        config.num_local_experts = 4
        config.num_experts_per_tok = 2
    if hasattr(config, "shared_intermediate_size"):
        config.shared_intermediate_size = 64
    return config


@pytest.fixture(scope="module", params=MODEL_TYPES)
def case(request):
    model_type = request.param
    config = _config(model_type)
    definition = MODEL_MAP[model_type]

    with torch.device("meta"):
        model = definition.loader.from_config(config)
    _convert_model_with_defuser(definition, model, cleanup_original=False)

    layers, layer_names = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert len(layers) == config.num_hidden_layers
    assert layer_names[0] == f"{definition.extract_layers_node()[0]}.0"

    specs = tuple(
        sorted({name.split(":", 1)[0] for block in definition.simple_layer_modules(config, QC) for name in block})
    )
    resolved = set()
    for layer in layers:
        resolved.update(find_modules(layer))

    return SimpleNamespace(
        model_type=model_type,
        config=config,
        definition=definition,
        model=model,
        layers=layers,
        specs=specs,
        resolved=resolved,
        plan=ModulePlan.from_tree(definition.module_tree),
    )


def test_quantizable_specs_resolve_on_the_real_model(case):
    for spec in case.specs:
        pattern = re.compile("^" + re.escape(spec).replace(re.escape(EXPERT_INDEX_PLACEHOLDER), r"\d+") + "$")
        assert any(pattern.match(name) for name in case.resolved), (
            case.model_type,
            spec,
        )


def test_every_checkpoint_weight_owner_is_declared(case):
    for layer in case.layers:
        modules = {name for name, _ in layer.named_modules() if name}
        for weight_path in layer.state_dict():
            owner = max(
                (
                    name
                    for name in modules
                    if weight_path.startswith(f"{name}.") and "." not in weight_path[len(name) + 1 :]
                ),
                key=len,
                default="",
            )
            assert owner, (case.model_type, weight_path)
            assert case.plan.find(owner) is not None, (
                case.model_type,
                owner,
            )


def test_nonquantized_owners_carry_explicit_tags(case):
    for name in ("input_layernorm", "post_attention_layernorm"):
        spec = case.plan.find(name)
        assert spec is not None and not spec.quantize, (case.model_type, name)

    if case.model_type == "granitemoe_swa":
        router = case.plan.find(f"{MOE_ROOT}.router")
        assert router is not None and not router.quantize

    # The learnable attention sink is a direct parameter on the quantizable
    # attention container, so it is never replaced by a quantized linear.
    for layer in case.layers:
        assert isinstance(layer.self_attn.sinks, torch.nn.Parameter)
        assert not isinstance(layer.self_attn.sinks, torch.nn.Linear)


def test_moe_experts_are_defused_into_quantizable_linears(case):
    if case.model_type != "granitemoe_swa":
        assert case.plan.moe_roots() == ()
        return

    for layer in case.layers:
        experts = getattr(layer, MOE_ROOT).experts
        assert not any(param.dim() == 3 for param in experts.parameters())
        for expert_idx in range(case.config.num_local_experts):
            expert = experts._modules[str(expert_idx)]
            assert isinstance(expert.gate_proj, torch.nn.Linear)
            assert isinstance(expert.up_proj, torch.nn.Linear)
            assert isinstance(expert.down_proj, torch.nn.Linear)


def test_roles_and_moe_families_reach_the_consumers(case):
    plan = case.plan
    assert plan.roles("self_attn.q_proj") == ("q",)
    assert plan.roles("self_attn.k_proj") == ("k",)
    assert plan.roles("self_attn.v_proj") == ("v",)
    assert plan.roles("self_attn.o_proj") == ("o",)

    if case.model_type == "granitemoe_swa":
        assert tuple(spec.template for spec in plan.moe_roots()) == (MOE_ROOT,)
        assert plan.expert_container_leaf(MOE_ROOT) == "experts"
        assert plan.is_expert(f"{EXPERTS}.0.gate_proj")
        assert plan.expert_root(f"{EXPERTS}.3.down_proj") == f"{EXPERTS}.3"
        assert plan.same_moe_family(f"{EXPERTS}.3.gate_proj", f"{EXPERTS}.3.down_proj")
        assert not plan.same_moe_family(f"{EXPERTS}.3.gate_proj", f"{EXPERTS}.4.down_proj")
        assert plan.find("shared_mlp.input_linear").quantize
        assert plan.find("shared_mlp.output_linear").quantize
    else:
        assert plan.roles("mlp.gate_proj") == ("gate",)
        assert plan.roles("mlp.up_proj") == ("up",)
        assert plan.roles("mlp.down_proj") == ("down",)


def test_undeclared_descendants_are_rejected(case):
    assert case.plan.find("self_attn.unknown_proj") is None
    assert case.plan.find("self_attn.sinks") is None
    assert case.plan.find("mlp.unknown_proj") is None
    if case.model_type == "granitemoe_swa":
        assert case.plan.find(f"{EXPERTS}.0.unknown_proj") is None
