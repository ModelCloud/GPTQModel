# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from types import SimpleNamespace

import pytest

from gptqmodel.models._const import EXPERT_INDEX_PLACEHOLDER
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.gpt_oss import GPTOSSGPTQ
from gptqmodel.models.definitions.granitemoehybrid import GraniteMoeHybridQModel


def _layer_mapping(tree):
    after_layer_placeholder = False
    for item in tree:
        if item == "#":
            after_layer_placeholder = True
            continue
        if after_layer_placeholder and isinstance(item, dict):
            return item
    return None


def _contains_expert_collection(model_cls, node) -> bool:
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                aliases = model_cls._parse_module_aliases(key)
                if any("expert" in alias for alias in aliases):
                    return True
            if _contains_expert_collection(model_cls, value):
                return True
        return False
    if isinstance(node, (list, tuple)):
        return any(_contains_expert_collection(model_cls, item) for item in node)
    return False


def _registered_moe_model_classes():
    model_types_by_class = defaultdict(list)
    for model_type, model_cls in MODEL_MAP.items():
        model_types_by_class[model_cls].append(model_type)

    cases = []
    for model_cls, model_types in model_types_by_class.items():
        if model_cls.module_tree is None:
            # Unsupported compatibility placeholders, such as the native DBRX
            # definition, do not describe a quantizable layer tree.
            continue

        variants = model_cls._iter_module_tree_variants()
        is_moe = any(
            (
                getattr(model_cls, "dynamic_expert_index", None) is not None,
                getattr(model_cls, "moe_lifecycle_hooks", None) is not None,
                any(_contains_expert_collection(model_cls, tree) for tree in variants),
                any("moe" in model_type.lower() for model_type in model_types),
            )
        )
        if is_moe:
            cases.append((model_cls, tuple(sorted(model_types))))

    return sorted(cases, key=lambda case: case[1])


@pytest.mark.parametrize(
    ("model_cls", "model_types"),
    _registered_moe_model_classes(),
    ids=lambda value: value.__name__ if isinstance(value, type) else ",".join(value),
)
def test_registered_moe_model_tree_marks_dynamic_expert_root(model_cls, model_types):
    for tree in model_cls._iter_module_tree_variants():
        mapping = _layer_mapping(tree)
        assert mapping is not None, f"{model_types}: missing layer mapping"

        marked_roots = [
            model_cls._parse_module_flags(key)[0]
            for key in mapping
            if model_cls.has_moe_flag(key)
        ]
        assert len(marked_roots) == 1, (
            f"{model_types}: expected one top-level :moe root per module-tree variant, "
            f"found {marked_roots}"
        )
        root = marked_roots[0]

        blocks = model_cls._build_layer_modules_for_tree(tree, include_capture_only=True)
        declared_modules = [
            model_cls._parse_module_flags(name)[0]
            for block in blocks
            for name in block
        ]
        root_modules = [
            name
            for name in declared_modules
            if name == root or name.startswith(f"{root}.")
        ]
        assert root_modules, f"{model_types}: marked root {root!r} has no declared modules"

        placeholder_modules = [
            name for name in root_modules if EXPERT_INDEX_PLACEHOLDER in name
        ]
        if getattr(model_cls, "dynamic_expert_index", None) is not None:
            assert placeholder_modules, (
                f"{model_types}: dynamic expert definition has no placeholder below {root!r}"
            )

        assert all(name.startswith(f"{root}.") for name in placeholder_modules)

        root_policy = model_cls.awq_input_feature_aggregation(root)
        child_modules = [name for name in root_modules if name != root]
        assert child_modules, f"{model_types}: marked root {root!r} has no pointwise children"
        child_policy = model_cls.awq_input_feature_aggregation(child_modules[-1])
        assert root_policy == {
            "mode": "token_rows",
            "capture_root": True,
        }
        assert child_policy == {
            "mode": "token_rows",
        }


@pytest.mark.parametrize(
    ("model_cls", "root", "expected_modules"),
    [
        (
            GPTOSSGPTQ,
            "mlp",
            {
                "mlp.experts.0.gate_proj",
                "mlp.experts.0.up_proj",
                "mlp.experts.0.down_proj",
                "mlp.experts.1.gate_proj",
                "mlp.experts.1.up_proj",
                "mlp.experts.1.down_proj",
            },
        ),
        (
            GraniteMoeHybridQModel,
            "block_sparse_moe",
            {
                "block_sparse_moe.input_linear.0.linear",
                "block_sparse_moe.output_linear.0.linear",
                "block_sparse_moe.input_linear.1.linear",
                "block_sparse_moe.output_linear.1.linear",
            },
        ),
    ],
)
def test_corrected_expert_roots_expand_runtime_pointwise_paths(model_cls, root, expected_modules):
    model_config = SimpleNamespace(num_local_experts=2)
    blocks = model_cls.full_layer_modules(
        model_config=model_config,
        is_awq_quantize=True,
        include_capture_only=True,
    )
    declared_modules = {
        model_cls._parse_module_flags(name)[0]
        for block in blocks
        for name in block
        if model_cls._parse_module_flags(name)[0].startswith(f"{root}.")
    }

    assert declared_modules == expected_modules
