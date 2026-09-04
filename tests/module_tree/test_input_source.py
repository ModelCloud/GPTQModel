# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch import nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.stage_subset import CalibrationCoveragePolicy, SubsetPlan
from gptqmodel.models import MODEL_MAP, BaseQModel
from gptqmodel.models.definitions.deepseek_v3 import DeepSeekV3QModel
from gptqmodel.models.definitions.deepseek_v32 import DeepSeekV32QModel
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.definitions.qwen2_moe import Qwen2MoeQModel
from gptqmodel.models.definitions.qwen3_moe import Qwen3MoeQModel
from gptqmodel.models.input_source import (
    InputSourceId,
    NamedInput,
    UniqueInput,
    group_input_sources,
    parse_input_flag,
    resolve_input_source,
)


def test_input_flag_parser_preserves_non_input_flags():
    cases = [
        ("q_proj:0", "q_proj", ["0"], None),
        ("q_proj:0:input", "q_proj", ["0"], UniqueInput()),
        ("q_proj:0:input=foo", "q_proj", ["0"], NamedInput("foo")),
        ("q_proj:0:q", "q_proj", ["0", "q"], None),
        ("q_proj:0:q:input", "q_proj", ["0", "q"], UniqueInput()),
        ("q_proj:0:q:input=foo", "q_proj", ["0", "q"], NamedInput("foo")),
        ("x:!:input", "x", ["!"], UniqueInput()),
        ("x:?:1:input=a", "x", ["?", "1"], NamedInput("a")),
    ]

    for spec, expected_name, expected_flags, expected_input in cases:
        assert BaseQModel._parse_module_spec(spec) == (
            expected_name,
            expected_flags,
            expected_input,
        )
        assert BaseQModel._parse_module_flags(spec) == (expected_name, expected_flags)

    assert BaseQModel.has_moe_flag("mlp:moe:input")
    assert BaseQModel._parse_module_flags("mlp:moe:input") == ("mlp", ["moe"])
    assert BaseQModel._parse_module_spec("x:input=a:b") == ("x", ["b"], NamedInput("a"))


@pytest.mark.parametrize(
    "flags",
    [
        ["input="],
        ["input=a", "input=b"],
        ["input=a:b"],
        ["input=a b"],
    ],
)
def test_input_flag_parser_rejects_invalid_names(flags):
    with pytest.raises(ValueError):
        parse_input_flag(flags)


def test_build_layer_modules_strips_input_tokens_without_changing_blocks():
    expected_deepseek_v3 = [
        ["input_layernorm:!"],
        ["self_attn.q_proj", "self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"],
        ["self_attn.q_b_proj", "self_attn.kv_b_proj"],
        ["self_attn.o_proj"],
        ["post_attention_layernorm:!"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
        ["mlp.experts.{expert_index}.gate_proj", "mlp.experts.{expert_index}.up_proj"],
        ["mlp.experts.{expert_index}.down_proj"],
        ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"],
        ["mlp.shared_experts.down_proj"],
    ]
    expected_deepseek_v32_attention = [
        "self_attn.q_a_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.indexer.wk",
        "self_attn.indexer.weights_proj:!",
    ]

    assert (
        DeepSeekV3QModel.build_layer_modules(DeepSeekV3QModel.module_tree)
        == expected_deepseek_v3
    )
    deepseek_v32_blocks = DeepSeekV32QModel.build_layer_modules(
        DeepSeekV32QModel.module_tree
    )
    assert expected_deepseek_v32_attention in deepseek_v32_blocks
    assert [
        "self_attn.q_b_proj",
        "self_attn.kv_b_proj",
        "self_attn.indexer.wq_b",
    ] in deepseek_v32_blocks

    classes = {
        model_cls
        for model_cls in MODEL_MAP.values()
        if isinstance(model_cls, type)
        and issubclass(model_cls, BaseQModel)
        and model_cls.module_tree is not None
    }
    for model_cls in classes:
        blocks = model_cls.build_layer_modules(model_cls.module_tree)
        assert all(
            "input" not in flag
            for block in blocks
            for token in block
            for flag in token.split(":")[1:]
        )


def test_module_tree_entries_resolve_scopes_and_expert_templates():
    llama_entries = LlamaQModel.build_module_tree_entries()
    assert llama_entries["self_attn.q_proj"].scope == "self_attn"
    assert llama_entries["self_attn.q_proj"].subset_id == 0
    assert llama_entries["self_attn.q_proj"].input_spec is None
    assert llama_entries["mlp.down_proj"].subset_id == 1

    deepseek_entries = DeepSeekV3QModel.build_module_tree_entries()
    assert isinstance(deepseek_entries["self_attn.q_b_proj"].input_spec, UniqueInput)
    assert (
        deepseek_entries["mlp.experts.{expert_index}.gate_proj"].scope
        == "mlp.experts.{expert_index}"
    )
    assert deepseek_entries["mlp.shared_experts.up_proj"].scope == "mlp.shared_experts"

    assert (
        DeepSeekV32QModel.build_module_tree_entries()["self_attn.indexer.wq_b"].input_spec
        == NamedInput("q_latent")
    )
    expert_entry = DeepSeekV3QModel.resolve_module_tree_entry("mlp.experts.7.up_proj")
    assert expert_entry is not None
    assert expert_entry.scope == "mlp.experts.7"
    assert expert_entry.full_path == "mlp.experts.7.up_proj"
    assert DeepSeekV3QModel.resolve_module_tree_entry("self_attn.q_proj:?") is not None
    assert DeepSeekV3QModel.resolve_module_tree_entry("not_a_module") is None


def _named(name, full_name, scope=None, subset_id=None, input_spec=None):
    return NamedModule(
        nn.Linear(4, 4, bias=False),
        name=name,
        full_name=full_name,
        layer_index=0,
        tree_scope_id=scope,
        subset_id=subset_id,
        input_spec=input_spec,
    )


def test_input_source_grouping_defaults_to_scope_and_subset():
    modules = [
        _named(
            "q_proj", "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn", 0
        ),
        _named(
            "k_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.self_attn", 0
        ),
        _named(
            "v_proj", "model.layers.0.self_attn.v_proj", "model.layers.0.self_attn", 0
        ),
        _named(
            "o_proj", "model.layers.0.self_attn.o_proj", "model.layers.0.self_attn", 1
        ),
        _named(
            "gate_proj", "model.layers.0.mlp.gate_proj", "model.layers.0.mlp", 0
        ),
        _named("up_proj", "model.layers.0.mlp.up_proj", "model.layers.0.mlp", 0),
    ]
    grouped = group_input_sources(modules)
    subset_zero = next(
        group for key, group in grouped.items() if key.subset_id == 0
    )
    assert [module.name for module in subset_zero] == ["q_proj", "k_proj", "v_proj"]
    assert len(grouped) == 3
    assert all(key.kind == "subset" for key in grouped)


def test_input_source_grouping_handles_unique_named_and_unknown_modules():
    unique = _named(
        "q_b_proj",
        "model.layers.0.self_attn.q_b_proj",
        "scope",
        1,
        UniqueInput(),
    )
    named = _named(
        "wq_b",
        "model.layers.0.self_attn.indexer.wq_b",
        "scope",
        1,
        NamedInput("q_latent"),
    )
    named_again = _named(
        "other",
        "model.layers.0.self_attn.other",
        "scope",
        1,
        NamedInput("q_latent"),
    )
    unknown = _named("lm_head", "lm_head")
    grouped = group_input_sources([unique, named, named_again, unknown])

    assert resolve_input_source(unique).kind == "unique"
    assert resolve_input_source(named).kind == "named"
    assert grouped[resolve_input_source(named)] == [named, named_again]
    assert resolve_input_source(unknown).module == "lm_head"
    assert len(
        {
            InputSourceId(scope="scope", subset_id=0),
            InputSourceId(scope="scope", name="x"),
        }
    ) == 2


def test_deepseek_v3_input_source_grouping_separates_latents_and_experts():
    def module_for(path):
        entry = DeepSeekV3QModel.resolve_module_tree_entry(path)
        assert entry is not None
        return _named(
            path.rsplit(".", 1)[-1],
            f"model.layers.0.{path}",
            f"model.layers.0.{entry.scope}" if entry.scope else "model.layers.0",
            entry.subset_id,
            entry.input_spec,
        )

    modules = [
        module_for("self_attn.q_a_proj"),
        module_for("self_attn.kv_a_proj_with_mqa"),
        module_for("self_attn.q_b_proj"),
        module_for("self_attn.kv_b_proj"),
        module_for("mlp.experts.0.gate_proj"),
        module_for("mlp.experts.0.up_proj"),
        module_for("mlp.experts.1.gate_proj"),
        module_for("mlp.experts.1.up_proj"),
        module_for("mlp.shared_experts.gate_proj"),
        module_for("mlp.shared_experts.up_proj"),
    ]
    grouped = group_input_sources(modules)
    source_by_path = {
        module.full_name.rsplit("model.layers.0.", 1)[-1]: resolve_input_source(module)
        for module in modules
    }

    assert source_by_path["self_attn.q_a_proj"] == source_by_path[
        "self_attn.kv_a_proj_with_mqa"
    ]
    assert source_by_path["self_attn.q_b_proj"] != source_by_path["self_attn.kv_b_proj"]
    assert source_by_path["mlp.experts.0.gate_proj"] == source_by_path[
        "mlp.experts.0.up_proj"
    ]
    assert source_by_path["mlp.experts.1.gate_proj"] == source_by_path[
        "mlp.experts.1.up_proj"
    ]
    assert source_by_path["mlp.experts.0.gate_proj"] != source_by_path[
        "mlp.experts.1.gate_proj"
    ]
    assert source_by_path["mlp.shared_experts.gate_proj"] != source_by_path[
        "mlp.experts.0.gate_proj"
    ]
    assert len(grouped) == 6


def test_moe_expert_scopes_are_distinct_for_qwen_definitions():
    for model_cls in (Qwen2MoeQModel, Qwen3MoeQModel):
        first = model_cls.resolve_module_tree_entry("mlp.experts.0.gate_proj")
        second = model_cls.resolve_module_tree_entry("mlp.experts.1.gate_proj")
        assert first is not None and second is not None
        assert first.scope == "mlp.experts.0"
        assert second.scope == "mlp.experts.1"
        assert first.scope != second.scope


def test_subset_plan_groups_modules_and_recomputes_for_chunks():
    modules = {
        "q_proj": _named("q_proj", "model.layers.0.self_attn.q_proj", "self_attn", 0),
        "k_proj": _named("k_proj", "model.layers.0.self_attn.k_proj", "self_attn", 0),
        "o_proj": _named("o_proj", "model.layers.0.self_attn.o_proj", "self_attn", 1),
    }
    plan = SubsetPlan(
        modules=modules,
        subset_index=0,
        subset_total=1,
        execute_forward=True,
        replay_after_process=False,
        forward_mode="serial",
        batch_count=1,
        forward_row_counts=[1],
        forward_total_rows=1,
        moe_groups={},
        forward_device_map={},
        calibration_coverage_policy=CalibrationCoveragePolicy(
            validate_input_coverage=False,
            fallback_enabled=False,
            prune_uncovered_modules=False,
            record_dynamic_exclusions=False,
        ),
        module_chunks=[modules],
    )
    subset_zero = next(
        key for key in plan.input_sources if key.subset_id == 0
    )
    assert [module.name for module in plan.input_sources[subset_zero]] == [
        "q_proj",
        "k_proj",
    ]
    chunk = plan.for_modules({"o_proj": modules["o_proj"]})
    assert len(chunk.input_sources) == 1
    assert list(next(iter(chunk.input_sources.values()))) == [modules["o_proj"]]
