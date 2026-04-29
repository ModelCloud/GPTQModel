import sys
import types
from unittest.mock import MagicMock

import torch

from gptqmodel.looper.loop_processor import ExecutionConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.stage_subset import SubsetPlan, build_layer_subset_plans, build_subset_plan
from gptqmodel.quantization.config import VramStrategy


def _make_named_module(name: str, layer_index: int = 0) -> NamedModule:
    return NamedModule(
        torch.nn.Linear(4, 4, bias=False),
        name=name,
        full_name=f"model.layers.{layer_index}.{name}",
        layer_index=layer_index,
    )


def _planning_blocks(*blocks) -> list[list[str]]:
    planning_blocks = []
    for block in blocks:
        if isinstance(block, str):
            planning_blocks.append([block])
        else:
            planning_blocks.append(list(block))
    return planning_blocks


def _make_looper():
    looper = MagicMock()
    looper.gptq_model = types.SimpleNamespace(
        lm_head="lm_head",
        quantize_config=types.SimpleNamespace(
            auto_forward_data_parallel=True,
            moe=None,
        ),
    )
    looper._is_attention_module_name.return_value = False
    looper._extract_moe_group_key.return_value = None
    looper._moe_subset_threshold = 2
    looper._quant_devices = [torch.device("cpu")]
    looper._dense_quant_devices = [torch.device("cpu")]
    looper._moe_quant_devices = [torch.device("cpu")]
    looper._dense_vram_strategy = VramStrategy.EXCLUSIVE
    looper._moe_vram_strategy = VramStrategy.EXCLUSIVE
    looper._dense_vram_strategy_explicit = False
    looper._moe_vram_strategy_explicit = False
    looper._resolve_batch_total.return_value = 2
    looper._collect_row_counts.return_value = [3, 2]
    return looper


class _StubProcessor:
    def __init__(self, execution_config: ExecutionConfig):
        self.execution_config = execution_config


def test_build_subset_plan_skips_forward_for_no_forward_processor():
    looper = _make_looper()
    processor = _StubProcessor(ExecutionConfig(require_fwd=False))
    subset = {"mlp.down_proj": _make_named_module("mlp.down_proj")}

    plan = build_subset_plan(
        looper,
        processor=processor,
        subset=subset,
        subset_index=0,
        subset_total=1,
        full=subset,
        fallback=None,
        layer_inputs=[[torch.zeros(1, 4)]],
    )

    assert plan.execute_forward is False
    assert plan.replay_after_process is False
    assert plan.batch_count == 0
    assert plan.forward_row_counts == []
    assert plan.forward_total_rows == 1
    assert plan.forward_mode == "parallel"
    assert plan.module_chunks == [subset]
    assert plan.calibration_coverage_policy.validate_input_coverage is False


def test_subset_plan_for_modules_preserves_parent_ordered_module_names():
    modules = {
        "mlp.experts.0.gate_proj": _make_named_module("mlp.experts.0.gate_proj"),
        "mlp.shared_expert.gate_proj": _make_named_module("mlp.shared_expert.gate_proj"),
        "mlp.experts.1.gate_proj": _make_named_module("mlp.experts.1.gate_proj"),
    }
    plan = SubsetPlan(
        modules=modules,
        subset_index=0,
        subset_total=1,
        execute_forward=True,
        replay_after_process=False,
        forward_mode="parallel",
        batch_count=1,
        forward_row_counts=[1],
        forward_total_rows=1,
        moe_groups={},
        forward_device_map={},
        calibration_coverage_policy=types.SimpleNamespace(
            validate_input_coverage=False,
            fallback_enabled=False,
            prune_uncovered_modules=False,
            record_dynamic_exclusions=False,
        ),
        module_chunks=[modules],
        ordered_module_names=[
            "mlp.shared_expert.gate_proj",
            "mlp.experts.0.gate_proj",
            "mlp.experts.1.gate_proj",
        ],
    )

    chunk_modules = {
        "mlp.experts.0.gate_proj": modules["mlp.experts.0.gate_proj"],
        "mlp.shared_expert.gate_proj": modules["mlp.shared_expert.gate_proj"],
    }
    chunk_plan = plan.for_modules(chunk_modules)

    assert list(chunk_plan.modules.keys()) == [
        "mlp.experts.0.gate_proj",
        "mlp.shared_expert.gate_proj",
    ]
    assert chunk_plan.ordered_module_names == [
        "mlp.shared_expert.gate_proj",
        "mlp.experts.0.gate_proj",
    ]


def test_build_subset_plan_balanced_moe_uses_serial_forward_and_device_map():
    looper = _make_looper()
    looper._quant_devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    looper._moe_quant_devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    looper._moe_vram_strategy = VramStrategy.BALANCED
    looper._moe_vram_strategy_explicit = True

    def _group_key(name: str):
        parts = name.split(".")
        if "experts" not in parts:
            return None
        expert_index = parts.index("experts")
        return ".".join(parts[:expert_index + 2])

    looper._extract_moe_group_key.side_effect = _group_key

    processor = _StubProcessor(
        ExecutionConfig(
            require_fwd=True,
            fwd_replay_after_process=True,
        )
    )
    subset = {
        "mlp.experts.0.gate_proj": _make_named_module("mlp.experts.0.gate_proj"),
        "mlp.experts.0.up_proj": _make_named_module("mlp.experts.0.up_proj"),
        "mlp.experts.1.gate_proj": _make_named_module("mlp.experts.1.gate_proj"),
        "mlp.experts.1.up_proj": _make_named_module("mlp.experts.1.up_proj"),
    }

    plan = build_subset_plan(
        looper,
        processor=processor,
        subset=subset,
        subset_index=0,
        subset_total=1,
        full=subset,
        fallback=True,
        layer_inputs=[[torch.zeros(3, 4)], [torch.zeros(2, 4)]],
    )

    assert plan.execute_forward is True
    assert plan.replay_after_process is True
    assert plan.forward_mode == "serial"
    assert plan.subset_forward_serial is True
    assert plan.batch_count == 2
    assert plan.forward_row_counts == [3, 2]
    assert plan.forward_total_rows == 5
    assert plan.forward_device_map == {
        "mlp.experts.0.gate_proj": torch.device("cuda:0"),
        "mlp.experts.0.up_proj": torch.device("cuda:0"),
        "mlp.experts.1.gate_proj": torch.device("cuda:1"),
        "mlp.experts.1.up_proj": torch.device("cuda:1"),
    }


def test_build_subset_plan_balanced_moe_pins_untouched_modules_to_baseline_device():
    looper = _make_looper()
    looper._quant_devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    looper._moe_quant_devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    looper._moe_vram_strategy = VramStrategy.BALANCED
    looper._moe_vram_strategy_explicit = True

    def _group_key(name: str):
        parts = name.split(".")
        if "experts" not in parts:
            return None
        expert_index = parts.index("experts")
        return ".".join(parts[:expert_index + 2])

    looper._extract_moe_group_key.side_effect = _group_key

    processor = _StubProcessor(
        ExecutionConfig(
            require_fwd=True,
            fwd_replay_after_process=True,
        )
    )
    subset = {
        "mlp.experts.0.gate_proj": _make_named_module("mlp.experts.0.gate_proj"),
        "mlp.experts.0.up_proj": _make_named_module("mlp.experts.0.up_proj"),
        "mlp.experts.1.gate_proj": _make_named_module("mlp.experts.1.gate_proj"),
        "mlp.experts.1.up_proj": _make_named_module("mlp.experts.1.up_proj"),
    }
    full = {name: named.module for name, named in subset.items()}
    full["self_attn.o_proj"] = torch.nn.Linear(4, 4, bias=False)

    plan = build_subset_plan(
        looper,
        processor=processor,
        subset=subset,
        subset_index=0,
        subset_total=1,
        full=full,
        fallback=True,
        layer_inputs=[[torch.zeros(3, 4)], [torch.zeros(2, 4)]],
    )

    assert plan.forward_device_map["mlp.experts.0.gate_proj"] == torch.device("cuda:0")
    assert plan.forward_device_map["mlp.experts.1.gate_proj"] == torch.device("cuda:1")
    assert plan.forward_device_map["self_attn.o_proj"] == torch.device("cpu")
    assert plan.restore_forward_device_overrides is False
    assert subset["mlp.experts.0.gate_proj"].state["preferred_quant_device"] == torch.device("cuda:0")
    assert subset["mlp.experts.1.gate_proj"].state["preferred_quant_device"] == torch.device("cuda:1")


def test_build_subset_plan_split_pools_reserve_dense_and_moe_devices():
    looper = _make_looper()
    looper._quant_devices = [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        torch.device("cuda:2"),
    ]
    looper._dense_quant_devices = [torch.device("cuda:0")]
    looper._moe_quant_devices = [torch.device("cuda:1"), torch.device("cuda:2")]
    looper._dense_vram_strategy = VramStrategy.EXCLUSIVE
    looper._moe_vram_strategy = VramStrategy.BALANCED
    looper._dense_vram_strategy_explicit = True
    looper._moe_vram_strategy_explicit = True

    def _group_key(name: str):
        parts = name.split(".")
        if "experts" not in parts:
            return None
        expert_index = parts.index("experts")
        return ".".join(parts[:expert_index + 2])

    looper._extract_moe_group_key.side_effect = _group_key

    processor = _StubProcessor(
        ExecutionConfig(
            require_fwd=True,
            fwd_replay_after_process=True,
        )
    )
    subset = {
        "mlp.experts.0.gate_proj": _make_named_module("mlp.experts.0.gate_proj"),
        "mlp.experts.0.up_proj": _make_named_module("mlp.experts.0.up_proj"),
        "mlp.experts.1.gate_proj": _make_named_module("mlp.experts.1.gate_proj"),
        "mlp.experts.1.up_proj": _make_named_module("mlp.experts.1.up_proj"),
        "mlp.experts.2.gate_proj": _make_named_module("mlp.experts.2.gate_proj"),
        "mlp.experts.2.up_proj": _make_named_module("mlp.experts.2.up_proj"),
    }
    full = {name: named.module for name, named in subset.items()}
    full["self_attn.o_proj"] = torch.nn.Linear(4, 4, bias=False)
    full["mlp.shared_expert.down_proj"] = torch.nn.Linear(4, 4, bias=False)

    plan = build_subset_plan(
        looper,
        processor=processor,
        subset=subset,
        subset_index=0,
        subset_total=1,
        full=full,
        fallback=True,
        layer_inputs=[[torch.zeros(3, 4)]],
        planning_layer_modules=_planning_blocks(
            ("self_attn.q_norm:!", "self_attn.q_proj", "self_attn.k_norm:!", "self_attn.k_proj", "self_attn.v_proj"),
            ("self_attn.o_proj",),
            ("mlp.experts.0.gate_proj", "mlp.experts.0.up_proj", "mlp.experts.1.gate_proj", "mlp.experts.1.up_proj"),
        ),
    )

    assert plan.forward_mode == "serial"
    assert plan.restore_forward_device_overrides is False
    assert plan.forward_device_map["mlp.experts.0.gate_proj"] == torch.device("cuda:1")
    assert plan.forward_device_map["mlp.experts.1.gate_proj"] == torch.device("cuda:2")
    assert plan.forward_device_map["mlp.experts.2.gate_proj"] == torch.device("cuda:1")
    assert plan.forward_device_map["self_attn.o_proj"] == torch.device("cuda:0")
    assert plan.forward_device_map["mlp.shared_expert.down_proj"] == torch.device("cuda:0")
    assert subset["mlp.experts.0.gate_proj"].state["preferred_quant_device"] == torch.device("cuda:1")
    assert subset["mlp.experts.1.gate_proj"].state["preferred_quant_device"] == torch.device("cuda:2")
    assert subset["mlp.experts.2.gate_proj"].state["preferred_quant_device"] == torch.device("cuda:1")


def test_build_subset_plan_dense_balanced_keeps_qkv_group_together():
    looper = _make_looper()
    looper._quant_devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    looper._dense_quant_devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    looper._dense_vram_strategy = VramStrategy.BALANCED
    looper._dense_vram_strategy_explicit = True

    def _group_key(name: str):
        parts = name.split(".")
        if "experts" not in parts:
            return None
        expert_index = parts.index("experts")
        return ".".join(parts[:expert_index + 2])

    looper._extract_moe_group_key.side_effect = _group_key
    looper._is_attention_module_name.side_effect = lambda name: name.startswith("self_attn.")

    processor = _StubProcessor(
        ExecutionConfig(
            require_fwd=True,
            fwd_replay_after_process=True,
        )
    )
    subset = {
        "self_attn.q_proj": _make_named_module("self_attn.q_proj"),
        "self_attn.k_proj": _make_named_module("self_attn.k_proj"),
        "self_attn.v_proj": _make_named_module("self_attn.v_proj"),
    }
    full = {name: named.module for name, named in subset.items()}
    full["self_attn.o_proj"] = torch.nn.Linear(4, 4, bias=False)
    full["mlp.experts.0.gate_proj"] = torch.nn.Linear(4, 4, bias=False)
    full["mlp.experts.0.up_proj"] = torch.nn.Linear(4, 4, bias=False)
    full["mlp.experts.1.gate_proj"] = torch.nn.Linear(4, 4, bias=False)
    full["mlp.experts.1.up_proj"] = torch.nn.Linear(4, 4, bias=False)

    plan = build_subset_plan(
        looper,
        processor=processor,
        subset=subset,
        subset_index=0,
        subset_total=1,
        full=full,
        fallback=True,
        layer_inputs=[[torch.zeros(3, 4)]],
        planning_layer_modules=_planning_blocks(
            ("self_attn.q_norm:!", "self_attn.q_proj", "self_attn.k_norm:!", "self_attn.k_proj", "self_attn.v_proj"),
            ("self_attn.o_proj",),
            ("mlp.experts.0.gate_proj", "mlp.experts.0.up_proj", "mlp.experts.1.gate_proj", "mlp.experts.1.up_proj"),
        ),
    )

    assert plan.forward_mode == "serial"
    assert plan.restore_forward_device_overrides is False
    assert plan.forward_device_map["self_attn.q_proj"] == torch.device("cuda:0")
    assert plan.forward_device_map["self_attn.k_proj"] == torch.device("cuda:0")
    assert plan.forward_device_map["self_attn.v_proj"] == torch.device("cuda:0")
    assert plan.forward_device_map["self_attn.o_proj"] == torch.device("cuda:1")
    assert subset["self_attn.q_proj"].state["preferred_quant_device"] == torch.device("cuda:0")
    assert subset["self_attn.k_proj"].state["preferred_quant_device"] == torch.device("cuda:0")
    assert subset["self_attn.v_proj"].state["preferred_quant_device"] == torch.device("cuda:0")


def test_build_subset_plan_split_pools_pin_dense_subset_and_balance_experts():
    looper = _make_looper()
    looper._quant_devices = [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        torch.device("cuda:2"),
    ]
    looper._dense_quant_devices = [torch.device("cuda:0")]
    looper._moe_quant_devices = [torch.device("cuda:1"), torch.device("cuda:2")]
    looper._dense_vram_strategy = VramStrategy.EXCLUSIVE
    looper._moe_vram_strategy = VramStrategy.BALANCED
    looper._dense_vram_strategy_explicit = True
    looper._moe_vram_strategy_explicit = True

    def _group_key(name: str):
        parts = name.split(".")
        if "experts" not in parts:
            return None
        expert_index = parts.index("experts")
        return ".".join(parts[:expert_index + 2])

    looper._extract_moe_group_key.side_effect = _group_key
    looper._is_attention_module_name.side_effect = lambda name: name.startswith("self_attn.")

    processor = _StubProcessor(
        ExecutionConfig(
            require_fwd=True,
            fwd_replay_after_process=True,
        )
    )
    subset = {
        "self_attn.q_proj": _make_named_module("self_attn.q_proj"),
        "self_attn.k_proj": _make_named_module("self_attn.k_proj"),
        "self_attn.v_proj": _make_named_module("self_attn.v_proj"),
    }
    full = {name: named.module for name, named in subset.items()}
    full["self_attn.o_proj"] = torch.nn.Linear(4, 4, bias=False)
    full["mlp.experts.0.gate_proj"] = torch.nn.Linear(4, 4, bias=False)
    full["mlp.experts.0.up_proj"] = torch.nn.Linear(4, 4, bias=False)
    full["mlp.experts.1.gate_proj"] = torch.nn.Linear(4, 4, bias=False)
    full["mlp.experts.1.up_proj"] = torch.nn.Linear(4, 4, bias=False)

    plan = build_subset_plan(
        looper,
        processor=processor,
        subset=subset,
        subset_index=0,
        subset_total=1,
        full=full,
        fallback=True,
        layer_inputs=[[torch.zeros(3, 4)]],
    )

    assert plan.forward_mode == "serial"
    assert plan.restore_forward_device_overrides is False
    assert plan.forward_device_map["self_attn.q_proj"] == torch.device("cuda:0")
    assert plan.forward_device_map["self_attn.o_proj"] == torch.device("cuda:0")
    assert plan.forward_device_map["mlp.experts.0.gate_proj"] == torch.device("cuda:1")
    assert plan.forward_device_map["mlp.experts.1.gate_proj"] == torch.device("cuda:2")
    assert subset["self_attn.q_proj"].state["preferred_quant_device"] == torch.device("cuda:0")
    assert subset["self_attn.k_proj"].state["preferred_quant_device"] == torch.device("cuda:0")
    assert subset["self_attn.v_proj"].state["preferred_quant_device"] == torch.device("cuda:0")


def test_build_layer_subset_plans_merges_groups_for_single_pass_processors():
    looper = _make_looper()
    requested_name_groups = []

    def _create_named_modules(
        module,
        full,
        is_lm_head_module,
        layer_index,
        layers_prefix,
        names,
        processor,
        fallback,
        layer_module=None,
    ):
        requested_name_groups.append(list(names))
        return {name: _make_named_module(name, layer_index=layer_index) for name in names}

    looper.create_named_modules.side_effect = _create_named_modules

    processor = _StubProcessor(
        ExecutionConfig(
            require_fwd=True,
            fwd_replay_after_process=False,
            fwd_all_modules_in_single_pass=True,
        )
    )

    plans = build_layer_subset_plans(
        looper,
        processor=processor,
        module=torch.nn.Linear(4, 4),
        layer_modules=[["self_attn.q_proj"], ["mlp.down_proj"]],
        planning_layer_modules=_planning_blocks(
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            ("mlp.down_proj",),
        ),
        layer_inputs=[[torch.zeros(1, 4)]],
        full={},
        is_lm_head_module=False,
        layer_index=3,
        layers_prefix="model.layers",
        fallback=True,
    )

    assert requested_name_groups == [["self_attn.q_proj", "mlp.down_proj"]]
    assert len(plans) == 1
    assert plans[0].subset_index == 0
    assert plans[0].subset_total == 1
    assert list(plans[0].modules.keys()) == ["self_attn.q_proj", "mlp.down_proj"]


def test_emit_moe_parallel_quant_subset_telemetry_reports_gil_and_worker_fanout(monkeypatch):
    emitted = []
    stage_subset_module = sys.modules[build_subset_plan.__module__]

    monkeypatch.setattr(
        stage_subset_module,
        "emit_device_telemetry",
        lambda event, **fields: emitted.append((event, fields)),
    )
    monkeypatch.setattr(stage_subset_module, "has_gil_control", lambda: True)
    monkeypatch.setattr(stage_subset_module, "has_gil_disabled", lambda: True)
    monkeypatch.setattr(
        stage_subset_module.DEVICE_THREAD_POOL,
        "_collect_state_snapshot",
        lambda: {
            "workers": {"cuda:1": 4, "cuda:2": 4},
            "total_workers": 8,
            "total_inflight": 2,
        },
    )

    named = _make_named_module("mlp.experts.0.gate_proj")
    plan = stage_subset_module.SubsetPlan(
        modules={named.name: named},
        subset_index=0,
        subset_total=1,
        execute_forward=True,
        replay_after_process=True,
        forward_mode="serial",
        batch_count=1,
        forward_row_counts=[1],
        forward_total_rows=1,
        moe_groups={"mlp.experts.0": [named.name]},
        forward_device_map={},
        calibration_coverage_policy=stage_subset_module.CalibrationCoveragePolicy(
            validate_input_coverage=False,
            fallback_enabled=True,
            prune_uncovered_modules=False,
            record_dynamic_exclusions=False,
        ),
        module_chunks=[{named.name: named}],
    )

    stage_subset_module._emit_moe_parallel_quant_subset_telemetry(
        plan=plan,
        quant_target_devices={
            named.name: torch.device("cuda:1"),
            "mlp.experts.0.up_proj": torch.device("cuda:2"),
        },
        futures_count=2,
        layer_index=3,
    )

    assert len(emitted) == 1
    event, fields = emitted[0]
    assert event == "moe_parallel_quant_subset"
    assert fields["layer_index"] == 3
    assert fields["submitted_tasks"] == 2
    assert fields["quant_devices"] == ["cuda:1", "cuda:2"]
    assert fields["thread_pool_workers"] == {"cuda:1": 4, "cuda:2": 4}
    assert fields["python_gil_disabled"] is True
    assert fields["free_threaded_parallel_quant_active"] is True
