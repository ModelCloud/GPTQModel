# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-architecture module-plan coverage for the Gemma family and OLMo 2.

Gemma 2, Gemma 3 text, the Gemma 3 multimodal shell, and OLMo 2 reused the
Llama definition. Their decoder layers add checkpoint-owned RMSNorm weights
(Gemma 2/3 feed-forward norms, Gemma 3 and OLMo 2 per-head Q/K norms, OLMo 2
post-feed-forward norm) that the Llama tree does not declare. ``ModulePlan.find``
therefore returned ``None`` for those owners even though the registry contract
passed, because that contract only compares the declaration against itself.

These tests build the real Transformers classes on the ``meta`` device, run the
registered Defuser conversion, and check every checkpoint weight owner against
the compiled plan. ``llama`` stays as the control that must not grow these
modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from gptqmodel.models._const import EXPERT_INDEX_PLACEHOLDER
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


QC = SimpleNamespace(dynamic=None)


@dataclass(frozen=True)
class FamilyCase:
    model_type: str
    definition: type[BaseQModel]
    config_class: str
    layers_node: str
    # Dense checkpoint owners that the inherited Llama tree could not resolve.
    added_dense: tuple[str, ...]
    # Every decoder weight owner, as a sanity set for the real layout.
    owners: tuple[str, ...]
    # Dense norms the definition must declare with the ``:norm`` tag.
    tagged_norms: tuple[str, ...]


CASES = {
    "gemma2": FamilyCase(
        model_type="gemma2",
        definition=MODEL_MAP["gemma2"],
        config_class="Gemma2Config",
        layers_node="model.layers",
        added_dense=("pre_feedforward_layernorm", "post_feedforward_layernorm"),
        owners=(
            "input_layernorm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "post_feedforward_layernorm",
        ),
        tagged_norms=("input_layernorm", "post_attention_layernorm"),
    ),
    "gemma3_text": FamilyCase(
        model_type="gemma3_text",
        definition=MODEL_MAP["gemma3_text"],
        config_class="Gemma3TextConfig",
        layers_node="model.layers",
        added_dense=(
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ),
        owners=(
            "input_layernorm",
            "self_attn.q_norm",
            "self_attn.q_proj",
            "self_attn.k_norm",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "post_feedforward_layernorm",
        ),
        tagged_norms=("input_layernorm", "post_attention_layernorm"),
    ),
    "gemma3": FamilyCase(
        model_type="gemma3",
        definition=MODEL_MAP["gemma3"],
        config_class="Gemma3Config",
        layers_node="model.language_model.layers",
        added_dense=(
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ),
        owners=(
            "input_layernorm",
            "self_attn.q_norm",
            "self_attn.q_proj",
            "self_attn.k_norm",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "post_feedforward_layernorm",
        ),
        # The multimodal shell already declared its two layer norms as dense
        # but never tagged them `:norm`, so norm tuning stays unavailable there.
        tagged_norms=(),
    ),
    "olmo2": FamilyCase(
        model_type="olmo2",
        definition=MODEL_MAP["olmo2"],
        config_class="Olmo2Config",
        layers_node="model.layers",
        added_dense=(
            "post_feedforward_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ),
        owners=(
            "self_attn.q_proj",
            "self_attn.q_norm",
            "self_attn.k_proj",
            "self_attn.k_norm",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "post_feedforward_layernorm",
        ),
        # OLMo 2 has no pre-attention `input_layernorm`.
        tagged_norms=("post_attention_layernorm",),
    ),
}


def _quantizable_specs(definition: type[BaseQModel], config) -> tuple[str, ...]:
    blocks = definition.simple_layer_modules(model_config=config, quantize_config=QC)
    return tuple(sorted({name.split(":", 1)[0] for block in blocks for name in block}))


def _resolved_modules(layers) -> set[str]:
    resolved = set()
    for layer in layers:
        resolved.update(find_modules(layer))
    return resolved


def _missing_specs(specs: tuple[str, ...], resolved: set[str]) -> list[str]:
    missing = []
    for spec in specs:
        pattern = re.compile(
            "^" + re.escape(spec).replace(re.escape(EXPERT_INDEX_PLACEHOLDER), r"\d+") + "$"
        )
        if not any(pattern.match(name) for name in resolved):
            missing.append(spec)
    return missing


def _weight_owners(layer) -> set[str]:
    """Modules owning a checkpoint ``state_dict()`` entry, per decoder layer."""

    modules = {name for name, _ in layer.named_modules() if name}
    owners = set()
    for weight_path in layer.state_dict():
        owner = max(
            (
                name
                for name in modules
                if weight_path.startswith(f"{name}.")
                and "." not in weight_path[len(name) + 1:]
            ),
            key=len,
            default="",
        )
        assert owner, f"state entry {weight_path!r} is not owned by a module"
        owners.add(owner)
    return owners


def _normalize_owner(owner: str) -> str:
    return re.sub(r"(?<=\.)\d+(?=\.|$)", "#", owner)


def _build_meta_case(case: FamilyCase) -> SimpleNamespace:
    config = transformers.AutoConfig.for_model(case.model_type)
    definition = MODEL_MAP[case.model_type]
    with torch.device("meta"):
        model = definition.loader.from_config(config)

    _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, layer_names = get_layers_with_prefixes(model, definition.extract_layers_node())
    text_config = getattr(config, "text_config", config)
    assert layers, f"{case.model_type}: no decoder layers resolved"
    assert len(layers) == text_config.num_hidden_layers

    return SimpleNamespace(
        config=config,
        definition=definition,
        model=model,
        layers=layers,
        layer_names=layer_names,
        specs=_quantizable_specs(definition, config),
        resolved=_resolved_modules(layers),
        plan=ModulePlan.from_tree(definition.module_tree),
    )


@pytest.fixture(scope="module", params=tuple(CASES.values()), ids=lambda case: case.model_type)
def family_case(request):
    if request.param.model_type not in CONFIG_MAPPING:
        pytest.skip(f"Installed Transformers has no {request.param.model_type!r} implementation")
    return _build_meta_case(request.param)


def test_family_definitions_reuse_the_registered_classes(family_case):
    case = next(item for item in CASES.values() if item.definition is family_case.definition)

    assert MODEL_MAP[case.model_type] is case.definition
    assert type(family_case.config).__name__ == case.config_class
    assert family_case.definition.extract_layers_node() == [case.layers_node]
    assert family_case.layer_names[0] == f"{case.layers_node}.0"


def test_every_quantizable_spec_resolves_on_the_real_model(family_case):
    case = next(item for item in CASES.values() if item.definition is family_case.definition)

    missing = _missing_specs(family_case.specs, family_case.resolved)
    assert not missing, f"{case.model_type}: quantizable entries matched no decoder layer: {missing[:8]}"
    # The extra norms are dense helpers, never quantizable leaves.
    assert set(case.added_dense).isdisjoint(family_case.specs)
    assert family_case.plan.roles("self_attn.q_proj") == ("q",)
    assert family_case.plan.roles("self_attn.k_proj") == ("k",)
    assert family_case.plan.roles("self_attn.v_proj") == ("v",)
    assert family_case.plan.roles("self_attn.o_proj") == ("o",)
    assert family_case.plan.roles("mlp.gate_proj") == ("gate",)
    assert family_case.plan.roles("mlp.up_proj") == ("up",)
    assert family_case.plan.roles("mlp.down_proj") == ("down",)


def test_every_checkpoint_weight_owner_is_declared_with_the_right_tag(family_case):
    case = next(item for item in CASES.values() if item.definition is family_case.definition)
    quantized_patterns = [
        re.compile("^" + re.escape(spec).replace(re.escape(EXPERT_INDEX_PLACEHOLDER), r"\d+") + "$")
        for spec in family_case.specs
    ]
    owners = set()
    for layer in family_case.layers:
        owners.update(_weight_owners(layer))

    normalized = {_normalize_owner(owner) for owner in owners}
    assert normalized == set(case.owners), f"{case.model_type}: unexpected decoder weight owners"

    for owner in owners:
        spec = family_case.plan.find(owner)
        assert spec is not None, (
            f"{case.model_type}: checkpoint weights owned by undeclared module {owner!r}"
        )
        if not any(pattern.match(owner) for pattern in quantized_patterns):
            assert not spec.quantize, (
                f"{case.model_type}: non-quantized weight owner {owner!r} must carry "
                "an explicit `:!`/`:?` tag"
            )


def test_added_norms_are_explicitly_tagged_and_rejected_when_undeclared(family_case):
    case = next(item for item in CASES.values() if item.definition is family_case.definition)
    plan = family_case.plan

    full = {
        name.split(":", 1)[0]
        for block in family_case.definition.full_layer_modules(
            model_config=family_case.config,
            include_capture_only=True,
        )
        for name in block
    }
    for name in case.added_dense:
        spec = plan.find(name)
        assert spec is not None, f"{case.model_type}: dense owner {name!r} is undeclared"
        assert not spec.quantize, f"{case.model_type}: {name!r} must stay dense"
        assert name in full
        assert plan.find(f"{name}.undeclared") is None
        assert plan.find(f"other.{name}") is None

    assert set(family_case.definition.get_norm_modules()) == set(case.tagged_norms)


def test_dense_norm_blocks_keep_the_decoder_forward_order(family_case):
    """AWQ scaling walks blocks in order and takes the previous block as ``prev_op``.

    The extra feed-forward norm must therefore sit between attention and the MLP
    block, and the post-feed-forward norm after the MLP block. Before this fix
    both were absent, so Gemma 2's MLP scaling used ``post_attention_layernorm``
    as its previous operator even though the forward feeds the MLP from
    ``pre_feedforward_layernorm``.
    """

    case = next(item for item in CASES.values() if item.definition is family_case.definition)
    blocks = [
        tuple(name.split(":", 1)[0] for name in block)
        for block in family_case.definition.full_layer_modules(
            model_config=family_case.config,
            is_awq_quantize=True,
            include_capture_only=True,
        )
    ]
    flat = [name for block in blocks for name in block]

    gate_up = flat.index("mlp.gate_proj")
    up = flat.index("mlp.up_proj")
    down = flat.index("mlp.down_proj")
    assert (gate_up, up) == (gate_up, gate_up + 1)
    assert down > up

    if "pre_feedforward_layernorm" in case.added_dense:
        assert flat.index("pre_feedforward_layernorm") == gate_up - 1
        assert "post_attention_layernorm" in flat[:gate_up]
    if "post_feedforward_layernorm" in case.added_dense:
        assert flat.index("post_feedforward_layernorm") == down + 1
    # The norms consume projected Q/K. Preserve the order of the real attention
    # forward within the block, including the different Gemma 3/OLMo 2 layouts.
    for attention_norm in ("self_attn.q_norm", "self_attn.k_norm"):
        if attention_norm in case.added_dense:
            assert flat.index(attention_norm) < flat.index("self_attn.o_proj")
    if case.model_type in {"gemma3", "gemma3_text"}:
        assert [name for name in flat if name.startswith("self_attn.")][:5] == [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.q_norm", "self_attn.k_norm",
        ]
    elif case.model_type == "olmo2":
        assert [name for name in flat if name.startswith("self_attn.")][:5] == [
            "self_attn.q_proj", "self_attn.q_norm", "self_attn.k_proj",
            "self_attn.k_norm", "self_attn.v_proj",
        ]


def test_llama_control_does_not_declare_the_family_specific_norms():
    """The gap came from inheritance: the Llama tree must stay Llama-only."""

    llama_plan = ModulePlan.from_tree(MODEL_MAP["llama"].module_tree)
    for name in (
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "self_attn.q_norm",
        "self_attn.k_norm",
    ):
        assert llama_plan.find(name) is None, name

    # ...while each family tree resolves its own extra norms as dense layers.
    expected = {
        "gemma2": ("pre_feedforward_layernorm", "post_feedforward_layernorm"),
        "gemma3_text": (
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ),
        "gemma3": (
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ),
        "olmo2": (
            "post_feedforward_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ),
    }
    for model_type, names in expected.items():
        plan = ModulePlan.from_tree(MODEL_MAP[model_type].module_tree)
        for name in names:
            spec = plan.find(name)
            assert spec is not None and not spec.quantize, (model_type, name)
    # OLMo 2 has no pre-attention norm at all.
    assert ModulePlan.from_tree(MODEL_MAP["olmo2"].module_tree).find("input_layernorm") is None
