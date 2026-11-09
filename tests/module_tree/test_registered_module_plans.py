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
