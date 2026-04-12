import types
from unittest.mock import MagicMock

import torch

from gptqmodel.looper.loop_processor import ExecutionConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.stage_subset import build_layer_subset_plans, build_subset_plan
from gptqmodel.quantization.config import VramStrategy


def _make_named_module(name: str, layer_index: int = 0) -> NamedModule:
    return NamedModule(
        torch.nn.Linear(4, 4, bias=False),
        name=name,
        full_name=f"model.layers.{layer_index}.{name}",
        layer_index=layer_index,
    )


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
    looper._vram_strategy = VramStrategy.EXCLUSIVE
    looper._quant_devices = [torch.device("cpu")]
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


def test_build_subset_plan_balanced_moe_uses_serial_forward_and_device_map():
    looper = _make_looper()
    looper._vram_strategy = VramStrategy.BALANCED
    looper._quant_devices = [torch.device("cuda:0"), torch.device("cuda:1")]

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
    looper._vram_strategy = VramStrategy.BALANCED
    looper._quant_devices = [torch.device("cuda:0"), torch.device("cuda:1")]

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
