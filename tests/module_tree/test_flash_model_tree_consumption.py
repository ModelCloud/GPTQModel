# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""End-to-end module-tree contract tests for current Flash model families.

These tests build the actual Transformers model shells on the ``meta`` device,
run the registered Defuser conversion, and then exercise the same tree consumers
used by quantization: ``simple_layer_modules``, ``ModulePlan``, the MoE lifecycle
hooks, and the module looper helpers.

The DeepSeek-V4.1-Flash checkpoint declares ``model_type=deepseek_v41``.  The
current Transformers release has no ``deepseek_v41`` config/model class, so a
true model test is impossible until upstream lands.  ``deepseek_v4`` is kept as
the supported V4-Flash control, and the V4.1 test activates automatically once
the upstream config exists.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.models._const import EXPERT_INDEX_PLACEHOLDER
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel
from gptqmodel.models.definitions.glm5_next import Glm5NextQModel
from gptqmodel.models.definitions.minimax_m3_vl import MiniMaxM3VLGPTQ
from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel
from gptqmodel.models.definitions.ricursive import RicursiveQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes
from tests.test_ricursive_support import _build_tiny_fixture, _reference_dir


QC = SimpleNamespace(dynamic=None)


@dataclass(frozen=True)
class FlashCase:
    model_type: str
    checkpoint: str
    definition: type[BaseQModel]
    config_class: str
    layers_node: str
    layer_count: int
    spec_count: int
    weight_owner_count: int
    moe: bool
    dense_layer: int | None
    moe_layer: int | None
    key_modules: tuple[str, ...]
    roles: tuple[tuple[str, tuple[str, ...]], ...]
    nonquantized: tuple[str, ...]
    # Trailing checkpoint-only blocks that the loader attaches beyond the
    # config's ``num_hidden_layers`` target stack (GLM-5.3 MTP layer 45).
    mtp_layers: int = 0


CASES = {
    "qwen3_5": FlashCase(
        model_type="qwen3_5",
        checkpoint="Qwen/Qwen3.8-27B",
        definition=Qwen3_5QModel,
        config_class="Qwen3_5Config",
        layers_node="model.language_model.layers",
        layer_count=32,
        spec_count=10,
        weight_owner_count=19,
        moe=False,
        dense_layer=0,
        moe_layer=None,
        key_modules=(
            "linear_attn.in_proj_qkv",
            "linear_attn.out_proj",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ),
        roles=(
            ("self_attn.q_proj", ("q",)),
            ("self_attn.k_proj", ("k",)),
            ("self_attn.v_proj", ("v",)),
            ("self_attn.o_proj", ("o",)),
            ("mlp.gate_proj", ("gate",)),
            ("mlp.up_proj", ("up",)),
            ("mlp.down_proj", ("down",)),
        ),
        nonquantized=(
            "input_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "linear_attn.norm",
            "linear_attn.conv1d",
            "linear_attn",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_a",
            "post_attention_layernorm",
        ),
    ),
    "glm5_next": FlashCase(
        model_type="glm5_next",
        checkpoint="zai-org/GLM-5.3-Flash",
        definition=Glm5NextQModel,
        config_class="Glm5NextConfig",
        layers_node="model.language_model.layers",
        # 45 target layers plus the checkpoint-only MTP block at index 45.
        layer_count=46,
        spec_count=873,
        # 37 target-layer owners plus the MTP envelope: enorm, hnorm, eh_proj
        # and shared_head.norm are checkpoint-owned and stay dense.
        weight_owner_count=41,
        moe=True,
        dense_layer=0,
        moe_layer=3,
        key_modules=(
            "self_attn.q_proj",
            "self_attn.q_a_proj",
            "self_attn.q_b_proj",
            "self_attn.kv_b_proj",
            "self_attn.o_proj",
            "mlp.experts.0.gate_proj",
            "mlp.experts.287.down_proj",
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
        ),
        roles=(
            ("self_attn.q_proj", ("q",)),
            ("self_attn.k_proj", ("k",)),
            ("self_attn.v_proj", ("v",)),
            ("self_attn.q_b_proj", ("q",)),
            ("self_attn.kv_b_proj", ("k", "v")),
            ("self_attn.o_proj", ("o",)),
            ("mlp.experts.0.gate_proj", ("gate",)),
            ("mlp.experts.0.up_proj", ("up",)),
            ("mlp.experts.0.down_proj", ("down",)),
            ("mlp.shared_experts.gate_proj", ("gate",)),
        ),
        nonquantized=(
            "input_layernorm",
            "self_attn.q_a_proj",
            "self_attn.kv_a_proj_with_mqa",
            "self_attn.indexer.wq_b",
            "self_attn.indexer.wk",
            "self_attn.indexer.weights_proj",
            "self_attn.indexer",
            "self_attn.indexer.k_norm",
            "self_attn.b_proj",
            "self_attn.conv1d",
            "self_attn.forget_gate",
            "self_attn.forget_gate.f_a_proj",
            "self_attn.forget_gate.f_b_proj",
            "self_attn.g_a_proj",
            "self_attn.g_b_proj",
            "self_attn.kv_a_layernorm",
            "self_attn.o_norm",
            "self_attn.q_a_layernorm",
            "post_attention_layernorm",
            "attn_hc",
            "attn_hc.input_norm",
            "ffn_hc",
            "ffn_hc.input_norm",
            "mlp.gate",
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
            # Checkpoint-only MTP layer 45 envelope, dense by design.
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head.norm",
        ),
        mtp_layers=1,
    ),
    "minimax_m3_vl": FlashCase(
        model_type="minimax_m3_vl",
        checkpoint="MiniMaxAI/MiniMax-M3.1-Flash",
        definition=MiniMaxM3VLGPTQ,
        config_class="MiniMaxM3VLConfig",
        layers_node="model.language_model.layers",
        layer_count=60,
        spec_count=394,
        weight_owner_count=20,
        moe=True,
        dense_layer=3,
        moe_layer=0,
        key_modules=(
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.indexer.q_proj",
            "self_attn.indexer.k_proj",
            "mlp.experts.0.gate_proj",
            "mlp.experts.127.down_proj",
            "mlp.shared_experts.gate_up_proj",
            "mlp.shared_experts.down_proj",
        ),
        roles=(
            ("self_attn.q_proj", ("q",)),
            ("self_attn.k_proj", ("k",)),
            ("self_attn.v_proj", ("v",)),
            ("self_attn.o_proj", ("o",)),
            ("self_attn.indexer.q_proj", ("q",)),
            ("self_attn.indexer.k_proj", ("k",)),
            ("mlp.experts.0.gate_proj", ("gate",)),
            ("mlp.experts.0.up_proj", ("up",)),
            ("mlp.experts.0.down_proj", ("down",)),
            ("mlp.shared_experts.gate_up_proj", ("gate", "up")),
        ),
        nonquantized=(
            "input_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "self_attn.indexer.q_norm",
            "self_attn.indexer.k_norm",
            "post_attention_layernorm",
            "mlp.gate",
        ),
    ),
    "deepseek_v4": FlashCase(
        model_type="deepseek_v4",
        checkpoint="deepseek-ai/DeepSeek-V4-Flash",
        definition=DeepSeekV4QModel,
        config_class="DeepseekV4Config",
        layers_node="model.layers",
        layer_count=43,
        spec_count=775,
        weight_owner_count=29,
        moe=True,
        dense_layer=None,
        moe_layer=0,
        key_modules=(
            "self_attn.q_a_proj",
            "self_attn.q_b_proj",
            "self_attn.kv_proj",
            "self_attn.o_b_proj",
            "mlp.experts.0.gate_proj",
            "mlp.experts.255.down_proj",
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
        ),
        roles=(
            ("self_attn.q_a_proj", ("q",)),
            ("self_attn.q_b_proj", ("q",)),
            ("self_attn.kv_proj", ("k", "v")),
            ("self_attn.o_b_proj", ("o",)),
            ("mlp.experts.0.gate_proj", ("gate",)),
            ("mlp.experts.0.up_proj", ("up",)),
            ("mlp.experts.0.down_proj", ("down",)),
            ("mlp.shared_experts.gate_proj", ("gate",)),
        ),
        nonquantized=(
            "input_layernorm",
            "self_attn.q_a_norm",
            "self_attn.q_b_norm",
            "self_attn.o_a_proj",
            "self_attn.kv_norm",
            "post_attention_layernorm",
            "attn_hc",
            "attn_hc.input_norm",
            "ffn_hc",
            "ffn_hc.input_norm",
            "self_attn",
            "self_attn.compressor",
            "self_attn.compressor.kv_norm",
            "self_attn.compressor.kv_proj",
            "self_attn.compressor.gate_proj",
            "self_attn.compressor.indexer",
            "self_attn.compressor.indexer.kv_norm",
            "self_attn.compressor.indexer.kv_proj",
            "self_attn.compressor.indexer.q_b_proj",
            "self_attn.compressor.indexer.gate_proj",
            "self_attn.compressor.indexer.scorer.weights_proj",
            "mlp.gate",
            "mlp.gate.e_score_correction_bias",
            "mlp.gate.tid2eid",
        ),
    ),
}


def _mix_minimax_layers(config) -> None:
    """Exercise both attention branches and both MLP branches in one config."""

    text_config = config.text_config
    text_config.layer_types = [
        "minimax_m3_sparse" if index % 2 == 0 else "full_attention"
        for index in range(text_config.num_hidden_layers)
    ]
    text_config.mlp_layer_types = [
        "dense" if index % 4 == 3 else "sparse"
        for index in range(text_config.num_hidden_layers)
    ]


CONFIG_OVERRIDES = {
    "minimax_m3_vl": _mix_minimax_layers,
}


def _quantizable_specs(definition: type[BaseQModel], config) -> tuple[str, ...]:
    blocks = definition.simple_layer_modules(
        model_config=config,
        quantize_config=QC,
    )
    return tuple(sorted({name.split(":", 1)[0] for block in blocks for name in block}))


def _missing_specs(specs: tuple[str, ...], resolved: set[str]) -> list[str]:
    missing = []
    for spec in specs:
        pattern = re.compile(
            "^" + re.escape(spec).replace(re.escape(EXPERT_INDEX_PLACEHOLDER), r"\d+") + "$"
        )
        if not any(pattern.match(name) for name in resolved):
            missing.append(spec)
    return missing


def _resolved_modules(layers) -> set[str]:
    resolved = set()
    for layer in layers:
        resolved.update(find_modules(layer))
    return resolved


def _weight_owners(layer) -> set[str]:
    """Return modules that own checkpoint state entries (weights/buffers).

    This intentionally reads ``state_dict()`` rather than ``named_parameters()``
    or ``named_buffers()``.  Runtime-generated non-persistent buffers are not
    checkpoint weights and must not become model-definition obligations.
    """

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


def _build_meta_case(case: FlashCase) -> SimpleNamespace:
    config = transformers.AutoConfig.for_model(case.model_type)
    override = CONFIG_OVERRIDES.get(case.model_type)
    if override is not None:
        override(config)

    definition = MODEL_MAP[case.model_type]
    with torch.device("meta"):
        model = definition.loader.from_config(config)

    _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, layer_names = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert layers, f"{case.model_type}: no decoder layers resolved"

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
def flash_case(request):
    return _build_meta_case(request.param)


def test_flash_definitions_build_real_meta_models(flash_case):
    case = next(item for item in CASES.values() if item.definition is flash_case.definition)

    assert MODEL_MAP[case.model_type] is case.definition
    assert type(flash_case.config).__name__ == case.config_class
    assert flash_case.definition.extract_layers_node() == [case.layers_node]

    text_config = getattr(flash_case.config, "text_config", flash_case.config)
    # The config only schedules target layers; a checkpoint-only auxiliary block
    # (GLM-5.3 MTP) is appended by the loader, so the resolved layer list is
    # longer by ``mtp_layers`` and each such layer reports the index past the
    # target stack.
    assert text_config.num_hidden_layers + case.mtp_layers == case.layer_count
    assert len(flash_case.layers) == case.layer_count
    for offset in range(case.mtp_layers):
        mtp_layer = flash_case.layers[case.layer_count - case.mtp_layers + offset]
        assert getattr(mtp_layer, "layer_idx", None) == text_config.num_hidden_layers + offset
    assert flash_case.layer_names[0] == f"{case.layers_node}.0"
    assert flash_case.layer_names[-1] == f"{case.layers_node}.{case.layer_count - 1}"
    assert len(flash_case.specs) == case.spec_count


def test_every_flash_quantizable_spec_resolves_on_the_real_model(flash_case):
    case = next(item for item in CASES.values() if item.definition is flash_case.definition)

    assert not _missing_specs(flash_case.specs, flash_case.resolved), (
        f"{case.model_type}: quantizable module-tree entries matched no decoder "
        f"layer: {_missing_specs(flash_case.specs, flash_case.resolved)[:12]}"
    )
    assert set(case.key_modules) <= flash_case.resolved


def test_flash_tree_marks_nonquantized_modules_as_dense(flash_case):
    case = next(item for item in CASES.values() if item.definition is flash_case.definition)
    simple = {
        name.split(":", 1)[0]
        for block in flash_case.definition.simple_layer_modules(flash_case.config, QC)
        for name in block
    }
    full = {
        name.split(":", 1)[0]
        for block in flash_case.definition.full_layer_modules(
            model_config=flash_case.config,
            include_capture_only=True,
        )
        for name in block
    }

    assert set(case.nonquantized) <= full
    assert set(case.nonquantized).isdisjoint(simple)
    # Every declared dense owner must resolve through the plan and carry an
    # explicit exclusion tag, including checkpoint-only MTP envelope tensors.
    for name in case.nonquantized:
        spec = flash_case.plan.find(name)
        assert spec is not None, f"{case.model_type}: dense owner {name!r} is undeclared"
        assert not spec.quantize, (
            f"{case.model_type}: dense owner {name!r} must carry an explicit `:!`/`:?` tag"
        )


def test_every_checkpoint_weight_owner_is_declared_with_the_right_tag(flash_case):
    case = next(item for item in CASES.values() if item.definition is flash_case.definition)
    quantized_patterns = [
        re.compile("^" + re.escape(spec).replace(re.escape(EXPERT_INDEX_PLACEHOLDER), r"\d+") + "$")
        for spec in flash_case.specs
    ]
    owners = set()
    for layer in flash_case.layers:
        owners.update(_weight_owners(layer))

    normalized_owners = {_normalize_owner(owner) for owner in owners}
    assert len(normalized_owners) == case.weight_owner_count

    for owner in owners:
        spec = flash_case.plan.find(owner)
        assert spec is not None, (
            f"{case.model_type}: checkpoint weights owned by undeclared module {owner!r}"
        )
        is_quantized = any(pattern.match(owner) for pattern in quantized_patterns)
        if not is_quantized:
            assert not spec.quantize, (
                f"{case.model_type}: non-quantized weight owner {owner!r} must carry "
                f"an explicit `:!`/`:?` tag"
            )


def test_flash_module_plan_resolves_roles_and_moe_families(flash_case):
    case = next(item for item in CASES.values() if item.definition is flash_case.definition)
    plan = flash_case.plan

    for path, expected_roles in case.roles:
        assert plan.roles(path) == expected_roles, path

    if case.moe:
        assert tuple(spec.template for spec in plan.moe_roots()) == ("mlp",)
        assert plan.expert_container_leaf("mlp") == "experts"
        assert plan.shared_container_leaf("mlp") == "shared_experts"
        assert plan.is_expert("mlp.experts.0.gate_proj")
        assert plan.expert_root("mlp.experts.17.down_proj") == "mlp.experts.17"
        assert plan.same_moe_family(
            "mlp.experts.17.gate_proj",
            "mlp.experts.17.down_proj",
        )
        assert not plan.same_moe_family(
            "mlp.experts.17.gate_proj",
            "mlp.experts.18.down_proj",
        )
        assert plan.is_moe("mlp.experts.17.gate_proj")
    else:
        assert plan.moe_roots() == ()
        assert plan.expert_containers() == ()
        assert plan.shared_expert_containers() == ()
        assert not plan.is_moe("mlp.down_proj")


def test_flash_is_moe_module_uses_exact_tree_paths(flash_case):
    case = next(item for item in CASES.values() if item.definition is flash_case.definition)
    definition = case.definition
    first_layer = flash_case.layer_names[0]

    if case.moe:
        assert definition.is_moe_module(f"{first_layer}.mlp")
        assert definition.is_moe_module(f"{first_layer}.mlp.experts.0.gate_proj")
        assert definition.is_moe_module("layers.0.mlp.experts.0.gate_proj")
        assert not definition.is_moe_module(f"{first_layer}.self_attn.q_proj")
    else:
        assert not definition.is_moe_module(f"{first_layer}.mlp")
        assert not definition.is_moe_module(f"{first_layer}.mlp.down_proj")


def test_flash_moe_hooks_follow_the_declared_containers(flash_case):
    case = next(item for item in CASES.values() if item.definition is flash_case.definition)
    hooks = getattr(case.definition, "moe_lifecycle_hooks", None)
    if hooks is None:
        assert not case.moe
        return

    hooks = copy.copy(hooks)
    hooks.module_plan = flash_case.plan

    moe_layer = flash_case.layers[case.moe_layer]
    moe_block = hooks.get_moe_block(moe_layer, case.definition)
    assert moe_block is not None
    assert hooks.get_experts_module_name(moe_block) == "experts"
    assert hooks.get_shared_experts_module_name(moe_block) == "shared_experts"
    assert hooks.get_experts_module(moe_block, case.definition) is not None
    assert hooks.get_shared_experts_module(moe_block, case.definition) is not None

    if case.dense_layer is not None:
        dense_layer = flash_case.layers[case.dense_layer]
        dense_block = hooks.get_moe_block(dense_layer, case.definition)
        assert dense_block is not None
        assert hooks.get_experts_module(dense_block, case.definition) is None
        assert hooks.get_shared_experts_module(dense_block, case.definition) is None


def test_flash_looper_consumers_use_the_module_plan(flash_case):
    case = next(item for item in CASES.values() if item.definition is flash_case.definition)
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = SimpleNamespace(module_plan=lambda: flash_case.plan)

    for path, roles in case.roles:
        assert looper._is_attention_module_name(path) == bool({"q", "k", "v"} & set(roles)), path

    if case.moe:
        assert looper._extract_moe_group_key("mlp.experts.17.gate_proj") == "mlp.experts.17"
        assert looper._extract_moe_group_key("mlp.shared_experts.gate_proj") is None
    else:
        assert looper._extract_moe_group_key("mlp.down_proj") is None


def test_ricursive_ripple_real_custom_model_matches_declared_tree(tmp_path):
    ref_dir = _reference_dir()
    if ref_dir is None:
        pytest.skip("Ripple custom code unavailable (set GPTQMODEL_RIPPLE_REF to run offline)")

    model_dir = tmp_path / "ripple"
    _build_tiny_fixture(ref_dir, model_dir)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()
    layers, layer_names = get_layers_with_prefixes(model, RicursiveQModel.extract_layers_node())
    assert len(layers) == 5
    assert layer_names[0] == "model.layers.0"

    specs = _quantizable_specs(RicursiveQModel, model.config)
    resolved = _resolved_modules(layers)
    assert not _missing_specs(specs, resolved)

    plan = ModulePlan.from_tree(RicursiveQModel.module_tree)
    assert plan.moe_roots() == ()
    assert plan.expert_containers() == ()
    assert plan.roles("self_attn.q_proj") == ("q",)
    assert plan.roles("self_attn.k_proj") == ("k",)
    assert plan.roles("self_attn.v_proj") == ("v",)
    assert plan.roles("self_attn.o_proj") == ("o",)
    assert plan.roles("self_attn.g_proj") == ()
    assert plan.roles("mlp.gate_proj") == ("gate",)
    assert plan.roles("mlp.up_proj") == ("up",)
    assert plan.roles("mlp.down_proj") == ("down",)

    owners = set()
    for layer in layers:
        owners.update(_weight_owners(layer))
    assert {_normalize_owner(owner) for owner in owners} == {
        "input_layernorm",
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "post_attention_layernorm",
        "post_feedforward_layernorm",
        "pre_feedforward_layernorm",
        "self_attn",
        "self_attn.g_proj",
        "self_attn.k_norm",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_norm",
        "self_attn.q_proj",
        "self_attn.v_proj",
    }
    assert all(plan.find(owner) is not None for owner in owners)
    assert not plan.find("self_attn").quantize

    assert not RicursiveQModel.is_moe_module("mlp.down_proj")
    assert not RicursiveQModel.is_moe_module("model.layers.0.mlp.down_proj")

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = SimpleNamespace(module_plan=lambda: plan)
    assert looper._is_attention_module_name("self_attn.q_proj")
    assert looper._is_attention_module_name("self_attn.k_proj")
    assert looper._is_attention_module_name("self_attn.v_proj")
    assert not looper._is_attention_module_name("self_attn.g_proj")
    assert not looper._is_attention_module_name("self_attn.o_proj")
    assert looper._extract_moe_group_key("mlp.down_proj") is None


def test_deepseek_v41_flash_model_tree_when_upstream_support_lands():
    """Build V4.1 against its real checkpoint only once Transformers supports it.

    The checkpoint ``deepseek-ai/DeepSeek-V4.1-Flash`` declares
    ``model_type=deepseek_v41``.  Transformers 5.22.2 (the version used by this
    suite) and its main branch contain only ``deepseek_v4``, so there is no
    native config or model class to build.  Keeping this test conditional means
    it becomes real V4.1 coverage as soon as the upstream class and the
    GPTQModel definition are added, instead of silently testing V4-Flash under a
    V4.1 label.
    """

    if "deepseek_v41" not in CONFIG_MAPPING:
        pytest.skip(
            "Transformers has no deepseek_v41 config/model implementation; "
            "DeepSeek-V4.1-Flash cannot be built as a true model yet"
        )
    if "deepseek_v41" not in MODEL_MAP:
        pytest.fail(
            "Transformers now exposes deepseek_v41 but GPTQModel has no "
            "DeepSeek-V4.1 definition; add the definition and tree tests"
        )

    config = transformers.AutoConfig.for_model("deepseek_v41")
    definition = MODEL_MAP["deepseek_v41"]
    with torch.device("meta"):
        model = definition.loader.from_config(config)
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, _ = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert layers

    specs = _quantizable_specs(definition, config)
    assert not _missing_specs(specs, _resolved_modules(layers))
