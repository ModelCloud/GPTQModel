# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Always-run plan contracts for every MODEL_MAP entry, including aliases.

The checked-in inventory pins the independent quantization subset planner's
ordered blocks. These are definition/compiler unit tests, not a claim that
every remote checkpoint has been instantiated. Real model layout and weight
ownership tests complement this suite (see test_flash_model_tree_consumption).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.models._const import EXPERT_INDEX_PLACEHOLDER
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.fusable import FusableConfig, require_fusable_config
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.shared_input import collect_leaf_specs


INVENTORY = json.loads((Path(__file__).parent / "fixtures/registered_module_plans.json").read_text())


def _methods(definition):
    return sorted({"gptq", *(getattr(method, "value", method) for method in (definition.module_tree_overrides or {}))})


def test_every_registered_model_has_an_explicit_plan_case():
    missing = set(MODEL_MAP) - INVENTORY["models"].keys()
    assert not missing, f"Add module-plan unit test cases for new model support: {sorted(missing)}"
    for model_type, definition in MODEL_MAP.items():
        assert INVENTORY["models"][model_type] == definition.__name__
        for method in _methods(definition):
            assert f"{definition.__name__}/{method}" in INVENTORY["cases"]


def test_every_registered_definition_has_explicit_fusion_admission():
    for model_type, definition in MODEL_MAP.items():
        config = definition.fusable
        assert config is None or isinstance(config, FusableConfig)
        if config is None or model_type not in config.model_types:
            with pytest.raises(ValueError):
                require_fusable_config(definition, model_type)
        else:
            assert require_fusable_config(definition, model_type) is config
            assert config.profiles


@pytest.mark.parametrize("model_type", sorted(MODEL_MAP))
def test_registered_model_plan_matches_quantization_consumers(model_type):
    definition = MODEL_MAP[model_type]
    for method in _methods(definition):
        expected = INVENTORY["cases"][f"{definition.__name__}/{method}"]
        tree = definition._resolve_effective_module_tree(
            SimpleNamespace(), SimpleNamespace(method=method),
        )
        if model_type == "dbrx":
            # Registry sentinel: original DBRX explicitly requires conversion.
            assert tree is None
            assert expected == {"layers": [], "blocks": []}
            model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
            with pytest.raises(ValueError, match="no explicit module_tree"):
                definition(model=model, quantized=False, quantize_config=None)
            continue

        assert tree is not None
        with definition._module_tree_context(tree):
            assert definition.extract_layers_node() == expected["layers"]
        blocks = definition.build_layer_modules(tree, include_capture_only=True)
        assert blocks == expected["blocks"], "Review changed paths/order/flags and update the explicit case"

        plan = ModulePlan.from_tree(tree)
        assert plan.specs
        with definition._module_tree_context(tree):
            for name in definition.get_norm_modules():
                assert "norm" in plan.find(name).flags
            roots = definition.get_moe_module_name() or []
            assert {spec.template for spec in plan.moe_roots()} == set(roots)
        # The shared-input planner has a separate metadata parser. A disagreement
        # here can silently route capture or reuse Hessians for the wrong input.
        for leaf in collect_leaf_specs(tree).values():
            spec = plan.find(leaf.template)
            assert spec is not None
            assert spec.quantize == leaf.quantize
            if leaf.input_tag is not None:
                assert f"in={leaf.input_tag}" in spec.flags
        wrapper = SimpleNamespace(module_plan=lambda: plan)
        looper = ModuleLooper.__new__(ModuleLooper)
        looper.gptq_model = wrapper
        weight_only = WeightOnlyLooper.__new__(WeightOnlyLooper)
        weight_only.gptq_model = wrapper

        for block in expected["blocks"]:
            for entry in block:
                path = entry.replace(":!", "").replace(":?", "")
                concrete = path.replace(EXPERT_INDEX_PLACEHOLDER, "7")
                spec = plan.find(concrete)
                assert spec is not None, (model_type, method, entry)
                assert spec.quantize == (":!" not in entry and ":?" not in entry)
                assert spec.capture_only == (":?" in entry)
                assert plan.find(concrete + ".undeclared") is None
                assert not plan.is_expert(concrete + ".undeclared")
                assert looper._extract_moe_group_key(concrete + ".undeclared") is None
                if EXPERT_INDEX_PLACEHOLDER in path:
                    prefix = path.split(EXPERT_INDEX_PLACEHOLDER, 1)[0]
                    expected_root = prefix + "7"
                    assert plan.is_expert(concrete)
                    assert looper._extract_moe_group_key(concrete) == expected_root
                    assert weight_only._extract_moe_group_key(concrete) == expected_root

        # Each layout variant must compile on its own as well as in the union.
        for variant in definition._iter_module_tree_variants(tree):
            variant_plan = ModulePlan.from_tree(variant)
            assert variant_plan.specs
            for block in definition.build_layer_modules(variant, include_capture_only=True):
                for entry in block:
                    concrete = entry.replace(":!", "").replace(":?", "").replace(EXPERT_INDEX_PLACEHOLDER, "7")
                    assert variant_plan.find(concrete) is not None

        payload = json.loads(plan.to_json())
        assert len(payload["specs"]) == len(plan.specs)
        assert "ModulePlan:" in plan.render(color=False, max_depth=2)


def test_dbrx_converted_nested_ffn_config_expert_count():
    """DBRX keeps ``moe_num_experts`` under ``ffn_config``, not on the top config."""

    import transformers

    definition = MODEL_MAP["dbrx_converted"]
    config = transformers.AutoConfig.for_model("dbrx")

    assert definition.dynamic_expert_index == "moe_num_experts"
    assert not hasattr(config, "moe_num_experts")
    assert definition.get_num_experts(config) == config.ffn_config.moe_num_experts

    class _QuantizeConfig(SimpleNamespace):
        dynamic = None

    blocks = definition.simple_layer_modules(
        model_config=config, quantize_config=_QuantizeConfig(dynamic=None)
    )
    entries = [entry for block in blocks for entry in block]

    for entry in entries:
        assert EXPERT_INDEX_PLACEHOLDER not in entry, entry

    expert_names = [entry for entry in entries if ".experts.mlp." in entry]
    assert len(expert_names) == 3 * config.ffn_config.moe_num_experts
    for index in range(config.ffn_config.moe_num_experts):
        for projection in ("w1", "v1", "w2"):
            assert f"ffn.experts.mlp.{index}.{projection}" in expert_names

    # Routers and the pre/post attention norms are checkpoint-owned but never
    # quantized; the definition must declare them explicitly.  Use the tree
    # blocks because ``simple_layer_modules`` drops every ``:!`` entry.
    declared = [
        entry
        for block in definition.build_layer_modules(
            definition.module_tree, include_capture_only=True
        )
        for entry in block
    ]
    for tagged in (
        "ffn.router.layer:!",
        "norm_attn_norm.norm_1:!",
        "norm_attn_norm.norm_2:!",
    ):
        assert tagged in declared, tagged


def test_ffn_config_does_not_shadow_a_top_level_expert_count():
    """An unrelated ``ffn_config`` must not hide a top-level cardinality.

    ``_expert_count_config`` gained an ``ffn_config`` branch for DBRX. Selecting
    that branch unconditionally would make every definition that keeps its
    expert count on the top-level config raise ``AttributeError`` as soon as a
    config also carried an ``ffn_config`` object without that field. Only a
    config that actually declares the requested field may claim ownership.
    """

    definition = MODEL_MAP["qwen3_moe"]
    field = definition.dynamic_expert_index

    # Unrelated FFN sub-config; the count stays on the top-level config.
    top_level = SimpleNamespace(ffn_config=SimpleNamespace(hidden_size=16), **{field: 8})
    assert definition._expert_count_config(top_level, field) is top_level
    assert definition.get_num_experts(top_level) == 8

    # FFN sub-config that owns the field is still preferred.
    nested = SimpleNamespace(ffn_config=SimpleNamespace(**{field: 8}))
    assert definition._expert_count_config(nested, field) is nested.ffn_config
    assert definition.get_num_experts(nested) == 8

    # When both declare it, the FFN owner keeps winning for DBRX-compatible
    # remote-code configs (this is the documented precedence).
    both = SimpleNamespace(ffn_config=SimpleNamespace(**{field: 8}), **{field: 4})
    assert definition.get_num_experts(both) == 8


@pytest.mark.parametrize("model_type", sorted(MODEL_MAP))
def test_every_expert_placeholder_expands_to_concrete_names(model_type):
    """Routed-expert placeholders must expand before quantization sees them.

    ``ModuleLooper.create_named_modules`` matches concrete names; a surviving
    ``{expert_index}`` entry can never hit and, on ``layer_modules_strict = False``
    definitions, is only recorded as a miss.  Every definition that declares the
    placeholder must therefore also declare the config field that expands it.
    """
    definition = MODEL_MAP[model_type]
    tree = definition._resolve_effective_module_tree(
        SimpleNamespace(model_type=model_type), SimpleNamespace(method="gptq"),
    )
    if tree is None:
        # Registry sentinels (original DBRX) reject quantization explicitly.
        assert model_type == "dbrx"
        return

    templates = {
        entry.replace(":!", "").replace(":?", "").split(":", 1)[0]
        for block in definition.build_layer_modules(tree, include_capture_only=True)
        for entry in block
    }
    if not any(EXPERT_INDEX_PLACEHOLDER in template for template in templates):
        return

    assert definition.dynamic_expert_index, (
        f"{model_type}: module tree declares routed-expert placeholders but the "
        "definition has no dynamic_expert_index to expand them"
    )

    # Expansion must leave no placeholder behind.  Every nesting precedence the
    # definitions use (top level, ``text_config``, ``thinker_config``,
    # ``ffn_config``) gets the same field so this stays registry-generic.
    field = definition.dynamic_expert_index
    inner = SimpleNamespace(**{field: 2})
    config = SimpleNamespace(
        **{
            field: 2,
            "text_config": SimpleNamespace(**{field: 2}),
            "thinker_config": SimpleNamespace(text_config=inner),
            "ffn_config": SimpleNamespace(**{field: 2}),
        }
    )
    expanded = definition.build_moe_modules_if_need(
        config, definition.build_layer_modules(tree, include_capture_only=True)
    )
    leftovers = [
        entry
        for block in expanded
        for entry in block
        if EXPERT_INDEX_PLACEHOLDER in entry
    ]
    assert not leftovers, (
        f"{model_type}: expert placeholder survived expansion even though "
        f"{field} is declared: {leftovers[:5]}"
    )


@pytest.mark.parametrize("model_type", ["ernie4_5_vl_moe", "ernie4_5_moe_vl"])
def test_non_scalar_expert_count_fails_closed_instead_of_raising(model_type):
    """A per-modality expert list must not crash placeholder expansion.

    The published ERNIE 4.5 VL checkpoints (``baidu/ERNIE-4.5-VL-28B-A3B``,
    ``...-28B-A3B-PT``, ``...-424B-A47B-PT``) ship a *flat* remote-code config
    with ``model_type = "ernie4_5_moe_vl"`` and a per-modality
    ``moe_num_experts = [64, 64]``. ``Ernie4_5_VLMoeQModel`` owns both routed
    families and its ``dynamic_expert_index`` is ``moe_num_experts``.

    That value cannot size ``range()``. Before this guard the expansion raised a
    bare ``TypeError: 'list' object cannot be interpreted as an integer`` from
    ``build_moe_modules_if_need``, so the published checkpoint failed to
    quantize at all. The contract here: a non-scalar count is reported as
    unresolved, the placeholder is preserved, and the consumer records the miss
    (which is what ``layer_modules_strict = False`` does) instead of raising.
    """

    definition = MODEL_MAP[model_type]
    field = definition.dynamic_expert_index
    assert field == "moe_num_experts"

    config = SimpleNamespace(model_type=model_type, **{field: [64, 64]})

    assert definition.get_num_experts(config) is None
    assert (
        definition.get_num_experts_for_module(
            config, f"mlp.text_moe.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj"
        )
        is None
    )

    blocks = definition.simple_layer_modules(
        model_config=config, quantize_config=SimpleNamespace(dynamic=None)
    )
    entries = [entry for block in blocks for entry in block]

    # The expansion must return, and every routed-expert spec must survive as a
    # literal placeholder so the miss is explicit rather than a wrong count.
    leftovers = [entry for entry in entries if EXPERT_INDEX_PLACEHOLDER in entry]
    assert leftovers, "a non-scalar count must not silently produce concrete names"
    assert all(".experts." in entry for entry in leftovers), leftovers

    # Scalar configs still expand: the guard narrows only the non-scalar case.
    scalar = SimpleNamespace(model_type=model_type, **{field: 8})
    scalar_entries = [
        entry
        for block in definition.simple_layer_modules(
            model_config=scalar, quantize_config=SimpleNamespace(dynamic=None)
        )
        for entry in block
    ]
    assert not [entry for entry in scalar_entries if EXPERT_INDEX_PLACEHOLDER in entry]
    assert len([entry for entry in scalar_entries if ".text_moe.experts." in entry]) == 24


def test_missing_expert_field_still_raises():
    """A definition/config mismatch keeps raising instead of degrading silently."""

    definition = MODEL_MAP["qwen3_moe"]
    with pytest.raises(AttributeError):
        definition.get_num_experts(SimpleNamespace())
