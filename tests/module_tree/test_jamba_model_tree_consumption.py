# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-model module-tree contract for the hybrid sparse Jamba decoder.

Jamba interleaves Mamba, attention, dense-MLP and routed-MoE layers inside one
``layers`` list, so ``layer_modules_strict`` is False and a missing branch is
silently skipped. The definition must therefore declare both ``feed_forward``
shapes and expand the routed experts; otherwise a supported sparse Jamba
checkpoint is quantized with every expert left in its source dtype.
"""

from __future__ import annotations

import copy
import re
from types import SimpleNamespace

import pytest
import torch
import transformers
from module_tree.looper_doubles import no_auto_dtype_cast_forward

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.models._const import EXPERT_INDEX_PLACEHOLDER
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


QC = SimpleNamespace(dynamic=None)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ROLES = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


def _mixed_jamba_config():
    config = transformers.AutoConfig.for_model("jamba")
    # Shrink the default 32-layer stack and interleave every branch so one
    # config exercises Mamba, attention, dense MLP and routed-MoE layers.
    config.num_hidden_layers = 4
    config.attn_layer_period = 2
    config.attn_layer_offset = 1
    config.expert_layer_period = 2
    config.expert_layer_offset = 1
    config.num_experts = 2
    return config


def _build_case() -> SimpleNamespace:
    config = _mixed_jamba_config()
    definition = MODEL_MAP["jamba"]
    with torch.device("meta"):
        model = definition.loader.from_config(config)

    _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, layer_names = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert layers, "jamba: no decoder layers resolved"

    return SimpleNamespace(
        config=config,
        definition=definition,
        model=model,
        layers=layers,
        layer_names=layer_names,
        plan=ModulePlan.from_tree(definition.module_tree),
    )


@pytest.fixture(scope="module")
def jamba_case():
    return _build_case()


def _quantizable_specs(definition, config) -> tuple[str, ...]:
    blocks = definition.simple_layer_modules(model_config=config, quantize_config=QC)
    return tuple(sorted({name.split(":", 1)[0] for block in blocks for name in block}))


def _matches(spec: str, name: str) -> bool:
    pattern = "^" + re.escape(spec).replace(re.escape(EXPERT_INDEX_PLACEHOLDER), r"\d+") + "$"
    return re.match(pattern, name) is not None


def _resolved_modules(layers) -> set[str]:
    resolved = set()
    for layer in layers:
        resolved.update(find_modules(layer))
    return resolved


def _expert_owners(layer) -> list[str]:
    return sorted({key.rsplit(".", 1)[0] for key in layer.state_dict() if ".experts." in key})


class _StubProcessor:
    """Minimal processor contract used by ``create_named_modules``."""

    def preprocess(self, module, **kwargs):
        return None

    def is_skipped(self, module) -> bool:
        return False


class _CoverageRecorder:
    def record_hit(self, *args, **kwargs):
        return None

    def record_miss(self, *args, **kwargs):
        return None


def _looper(plan: ModulePlan) -> ModuleLooper:
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = SimpleNamespace(
        module_plan=lambda: plan,
        layer_modules_strict=False,
        lm_head="lm_head",
        quant_region_timer=None,
        _maybe_auto_dtype_cast_forward_module=no_auto_dtype_cast_forward,
    )
    looper.layer_module_coverage = _CoverageRecorder()
    looper.input_embeddings_name = None
    looper.output_embeddings_name = None
    return looper


def test_jamba_tree_declares_both_feed_forward_shapes(jamba_case):
    plan = jamba_case.plan

    assert tuple(spec.template for spec in plan.moe_roots()) == ("feed_forward",)
    assert plan.expert_container_leaf("feed_forward") == "experts"
    assert plan.shared_container_leaf("feed_forward") is None

    router = plan.find("feed_forward.router")
    assert router is not None
    assert not router.quantize

    for projection in PROJECTIONS:
        dense = plan.find(f"feed_forward.{projection}")
        assert dense is not None and dense.quantize
        expert = plan.find(f"feed_forward.experts.0.{projection}")
        assert expert is not None and expert.quantize
        assert expert.roles == (ROLES[projection],)

    assert plan.is_expert("feed_forward.experts.1.gate_proj")
    assert plan.expert_root("feed_forward.experts.1.down_proj") == "feed_forward.experts.1"
    assert plan.same_moe_family(
        "feed_forward.experts.1.gate_proj",
        "feed_forward.experts.1.down_proj",
    )
    assert not plan.same_moe_family(
        "feed_forward.experts.0.gate_proj",
        "feed_forward.experts.1.down_proj",
    )


def test_jamba_experts_are_selected_and_resolve_on_real_layers(jamba_case):
    specs = _quantizable_specs(jamba_case.definition, jamba_case.config)
    expert_specs = [spec for spec in specs if ".experts." in spec]

    assert expert_specs, "jamba: definition emits no routed-expert specs"
    assert not any(EXPERT_INDEX_PLACEHOLDER in spec for spec in specs)
    assert {f"feed_forward.{projection}" for projection in PROJECTIONS} <= set(specs)

    resolved = _resolved_modules(jamba_case.layers)
    missing = [
        spec for spec in expert_specs if not any(_matches(spec, name) for name in resolved)
    ]
    assert not missing, (
        "jamba: routed-expert specs matched no decoder layer and would stay "
        f"in the source dtype: {missing[:10]}"
    )

    for name in sorted(name for name in resolved if ".experts." in name):
        spec = jamba_case.plan.find(name)
        assert spec is not None, name
        assert spec.quantize, name


def test_jamba_expert_owners_are_checkpoint_weights(jamba_case):
    moe_layer = jamba_case.layers[1]
    dense_layer = jamba_case.layers[0]

    owners = _expert_owners(moe_layer)
    assert len(owners) == 3 * jamba_case.config.num_experts
    for owner in owners:
        spec = jamba_case.plan.find(owner)
        assert spec is not None, owner
        assert spec.quantize, owner

    # The dense layer keeps the projections directly on `feed_forward`.
    assert _expert_owners(dense_layer) == []
    assert "feed_forward.gate_proj.weight" in dense_layer.state_dict()

    # The router is checkpoint-owned but intentionally kept in the source dtype.
    assert "feed_forward.router.weight" in moe_layer.state_dict()
    router = jamba_case.plan.find("feed_forward.router")
    assert router is not None and not router.quantize


def test_jamba_looper_subsets_include_every_expert(jamba_case):
    looper = _looper(jamba_case.plan)
    blocks = jamba_case.definition.simple_layer_modules(
        model_config=jamba_case.config,
        quantize_config=QC,
    )
    ffn_blocks = [
        block for block in blocks if any(name.startswith("feed_forward.") for name in block)
    ]

    moe_layer = jamba_case.layers[1]
    moe_full = find_modules(moe_layer)
    moe_subsets = [
        looper.create_named_modules(
            moe_layer, moe_full, False, 1, jamba_case.layer_names[1], block,
            processor=_StubProcessor(), fallback=None,
        )
        for block in ffn_blocks
    ]
    selected = {name for subset in moe_subsets for name in subset}
    assert set(_expert_owners(moe_layer)) <= selected

    dense_layer = jamba_case.layers[0]
    dense_full = find_modules(dense_layer)
    dense_subsets = [
        looper.create_named_modules(
            dense_layer, dense_full, False, 0, jamba_case.layer_names[0], block,
            processor=_StubProcessor(), fallback=None,
        )
        for block in ffn_blocks
    ]
    dense_selected = {name for subset in dense_subsets for name in subset}
    assert {f"feed_forward.{projection}" for projection in PROJECTIONS} <= dense_selected


def test_jamba_is_moe_module_uses_exact_tree_paths(jamba_case):
    definition = jamba_case.definition
    moe_prefix = jamba_case.layer_names[1]
    dense_prefix = jamba_case.layer_names[0]

    assert definition.is_moe_module(f"{moe_prefix}.feed_forward")
    assert definition.is_moe_module(f"{moe_prefix}.feed_forward.experts.0.gate_proj")
    assert definition.is_moe_module("layers.1.feed_forward.experts.0.gate_proj")
    assert not definition.is_moe_module(f"{moe_prefix}.self_attn.q_proj")
    assert not definition.is_moe_module(f"{dense_prefix}.feed_forward.gate_proj.undeclared")


def test_jamba_moe_hooks_follow_the_declared_containers(jamba_case):
    hooks = copy.copy(jamba_case.definition.moe_lifecycle_hooks)
    assert hooks is not None
    hooks.module_plan = jamba_case.plan

    moe_block = hooks.get_moe_block(jamba_case.layers[1], jamba_case.definition)
    assert moe_block is not None
    assert hooks.get_experts_module_name(moe_block) == "experts"
    assert hooks.get_experts_module(moe_block, jamba_case.definition) is not None
    assert hooks.get_shared_experts_module(moe_block, jamba_case.definition) is None

    dense_block = hooks.get_moe_block(jamba_case.layers[0], jamba_case.definition)
    assert dense_block is not None
    assert hooks.get_experts_module(dense_block, jamba_case.definition) is None


def test_jamba_looper_groups_expert_paths_by_declared_family(jamba_case):
    looper = _looper(jamba_case.plan)

    assert looper._extract_moe_group_key("feed_forward.experts.0.gate_proj") == "feed_forward.experts.0"
    assert looper._extract_moe_group_key("feed_forward.experts.1.down_proj") == "feed_forward.experts.1"
    assert looper._extract_moe_group_key("feed_forward.gate_proj") is None
