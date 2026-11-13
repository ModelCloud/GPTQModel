# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-model module-tree coverage for Granite SWA and Granite MoE SWA.

Both definitions are built from the installed Transformers model classes on the
``meta`` device, converted through the registered Defuser pipeline, and then
checked against the same plan consumers used by quantization.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import transformers

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


QC = SimpleNamespace(dynamic=None)
MODEL_CASES = (
    pytest.param(("granite_swa", None), id="granite_swa"),
    pytest.param(("granitemoe_swa", 64), id="granitemoe_swa-shared"),
    pytest.param(("granitemoe_swa", 0), id="granitemoe_swa-no-shared"),
)
MOE_ROOT = "block_sparse_moe"
EXPERTS = "block_sparse_moe.experts"


def _config(model_type: str, shared_intermediate_size: int | None = None):
    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "pad_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
        "layer_types": ["full_attention", "sliding_attention"],
        "layer_rope_theta": [0, 10000],
        "sliding_window": 16,
    }
    if model_type == "granitemoe_swa":
        kwargs.update(
            num_local_experts=4,
            num_experts_per_tok=2,
            shared_intermediate_size=shared_intermediate_size,
        )
    return transformers.AutoConfig.for_model(model_type, **kwargs)


@pytest.fixture(scope="module", params=MODEL_CASES)
def case(request):
    model_type, shared_intermediate_size = request.param
    config = _config(model_type, shared_intermediate_size)
    definition = MODEL_MAP[model_type]

    with torch.device("meta"):
        model = definition.loader.from_config(config)
    _convert_model_with_defuser(definition, model, cleanup_original=False)

    wrapper = definition(model=model, quantized=False, quantize_config=None)

    layers, layer_names = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert len(layers) == config.num_hidden_layers
    assert layer_names[0] == f"{definition.extract_layers_node()[0]}.0"

    specs = tuple(
        sorted({name.split(":", 1)[0] for block in definition.simple_layer_modules(config, QC) for name in block})
    )
    return SimpleNamespace(
        model_type=model_type,
        config=config,
        definition=definition,
        model=model,
        wrapper=wrapper,
        layers=layers,
        specs=specs,
        plan=wrapper.module_plan(),
    )


def test_quantizable_specs_resolve_on_the_real_model(case):
    specs = set(case.specs)
    if case.model_type == "granitemoe_swa" and case.config.shared_intermediate_size == 0:
        assert all(layer.shared_mlp is None for layer in case.layers)
        specs.difference_update({"shared_mlp.input_linear", "shared_mlp.output_linear"})
    for layer in case.layers:
        assert specs == set(find_modules(layer)), case.model_type


def test_every_checkpoint_weight_owner_is_declared(case):
    for layer in case.layers:
        modules = {name for name, _ in layer.named_modules() if name}
        quantizable = find_modules(layer)
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
            spec = case.plan.find(owner)
            assert spec is not None, (case.model_type, weight_path, owner)
            if spec.quantize:
                assert owner in case.specs, (case.model_type, weight_path, owner)
                assert owner in quantizable, (case.model_type, weight_path, owner)
            else:
                assert {"!", "?"}.intersection(spec.flags), (case.model_type, weight_path, spec)


def test_nonquantized_owners_carry_explicit_tags(case):
    for name in ("input_layernorm", "self_attn", "post_attention_layernorm"):
        spec = case.plan.find(name)
        assert spec is not None and not spec.quantize, (case.model_type, name)

    if case.model_type == "granitemoe_swa":
        router = case.plan.find(f"{MOE_ROOT}.router")
        assert router is not None and not router.quantize

    # The direct sink owner is excluded while its projection children remain
    # quantizable. A container declaration alone does not establish that policy.
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


def test_moe_plan_reaches_live_hooks_and_looper_grouping(case):
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = case.wrapper
    weight_only = WeightOnlyLooper.__new__(WeightOnlyLooper)
    weight_only.gptq_model = case.wrapper

    if case.model_type != "granitemoe_swa":
        assert not case.plan.moe_roots()
        return

    hooks = case.wrapper.moe_lifecycle_hooks
    assert hooks.module_plan is case.plan
    for layer in case.layers:
        block = hooks.get_moe_block(layer, case.wrapper)
        assert block is layer.block_sparse_moe
        assert hooks.get_experts_module(block, case.wrapper) is block.experts
        # The shared MLP is a decoder sibling, outside the routed MoE root.
        assert hooks.get_shared_experts_module(block, case.wrapper) is None
        for index in range(case.config.num_local_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"{EXPERTS}.{index}.{projection}"
                expected = f"{EXPERTS}.{index}"
                assert looper._extract_moe_group_key(name) == expected
                assert weight_only._extract_moe_group_key(name) == expected
                assert looper._extract_moe_group_key(name + ".unknown") is None
    assert looper._extract_moe_group_key("shared_mlp.input_linear") is None
