# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module
from gptqmodel.looper.loop_processor import (
    CalibrationTypes,
    CheckpointSupport,
    ExecutionConfig,
    ForwardConfig,
    ProcessorLifecycleConfig,
)
from gptqmodel.looper.stage_subset import _resolve_subset_calibration_coverage_policy
from gptqmodel.models import ProcessorStep, QuantizationRunPlan
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig
from gptqmodel.utils.model import create_quant_module


def test_run_plan_is_the_single_processor_chain_contract():
    config = QuantizeConfig(
        bits=4,
        group_size=128,
        method=METHOD.GPTQ,
        dynamic={r"+:^model\.layers\.0\.self_attn\.q_proj$": {"bits": 3}},
    )

    plan = QuantizationRunPlan.build(
        config,
        format_code=FORMAT.GPTQ,
        pack_backend=None,
        loop_backend=None,
        batch_size=2,
        needs_lora=False,
    )

    assert plan.processors == (ProcessorStep.GPTQ,)
    assert plan.checkpoint_supported
    assert plan.describe()["format"] == FORMAT.GPTQ.value


def test_lifecycle_calibration_policy_is_declared_by_processor():
    processor = SimpleNamespace(
        lifecycle=ProcessorLifecycleConfig(
            calibration=CalibrationTypes.EACH_MODULE_NEEDS_INPUT,
            checkpointing=CheckpointSupport.PRIMARY,
        )
    )

    policy = _resolve_subset_calibration_coverage_policy(processor, fallback=None)

    assert policy.validate_input_coverage
    assert policy.prune_uncovered_modules


def test_processor_lifecycle_config_is_mutable():
    lifecycle = ProcessorLifecycleConfig()
    lifecycle.calibration = CalibrationTypes.EACH_MODULE_NEEDS_INPUT

    assert lifecycle.calibration is CalibrationTypes.EACH_MODULE_NEEDS_INPUT


def test_lifecycle_use_returns_configured_copy():
    lifecycle = ProcessorLifecycleConfig()
    configured = lifecycle.use(
        ExecutionConfig(
            forward=ForwardConfig(
                required=False,
                replay_after_process=False,
            )
        )
    )

    assert configured is not lifecycle
    assert configured.execution.forward.required is False
    assert configured.execution.forward.replay_after_process is False
    assert lifecycle.execution.forward.required is True
    assert lifecycle.execution.forward.replay_after_process is True


def test_effective_module_config_is_ordered_and_immutable():
    config = QuantizeConfig(
        bits=4,
        group_size=128,
        dynamic={
            r"-:^model\.layers\.0\.mlp\.down_proj$": {},
            r"+:^model\.layers\.0\.": {"bits": 3, "group_size": 64},
        },
    )

    selected = config.resolve_module_config("model.layers.0.self_attn.q_proj")
    excluded = config.resolve_module_config("model.layers.0.mlp.down_proj")
    assert selected.source_pattern.endswith(r"model\.layers\.0\.")
    assert selected.get("bits") == 3
    assert excluded.excluded

    config.dynamic[selected.source_pattern] = {"bits": 2, "group_size": 32}
    assert selected.get("bits") == 3
    assert config.resolve_module_config("model.layers.0.self_attn.q_proj").get("bits") == 2


def test_one_module_snapshot_drives_clone_and_pack_settings():
    config = QuantizeConfig(
        bits=4,
        group_size=128,
        dynamic={r"+:^proj$": {"bits": 3, "group_size": 64}},
    )
    effective = config.resolve_module_config("proj")
    config.dynamic[r"+:^proj$"] = {"bits": 2, "group_size": 32}

    clone = clone_gptq_config_for_module(config, "proj", effective=effective)

    assert clone.bits == 3
    assert clone.group_size == 64


def test_dynamic_pack_dtype_is_normalized_during_module_creation():
    config = QuantizeConfig(
        bits=4,
        group_size=32,
        dynamic={r"+:^proj$": {"bits": 8, "pack_dtype": "int16"}},
    )
    effective = config.resolve_module_config("proj")
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(128, 128, bias=False, dtype=torch.float16)

    create_quant_module(
        name="proj",
        linear_cls=TorchLinear,
        bits=4,
        desc_act=False,
        dynamic=config.dynamic,
        group_size=32,
        module=model,
        submodule=model.proj,
        sym=True,
        device=DEVICE.CPU,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        format=FORMAT.GPTQ,
        effective_module_config=effective,
    )

    assert model.proj.bits == 8
    assert model.proj.pack_dtype is torch.int16


def test_module_snapshot_cannot_leak_to_another_pack_name():
    config = QuantizeConfig(dynamic={r"+:^other$": {"bits": 3}})
    effective = config.resolve_module_config("other")
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(16, 16, bias=False, dtype=torch.float16)

    with pytest.raises(ValueError, match="does not match quantized module name"):
        create_quant_module(
            name="proj",
            linear_cls=TorchLinear,
            bits=4,
            desc_act=False,
            dynamic=config.dynamic,
            group_size=128,
            module=model,
            submodule=model.proj,
            sym=True,
            device=DEVICE.CPU,
            lm_head_name="lm_head",
            pack_dtype=torch.int32,
            format=FORMAT.GPTQ,
            effective_module_config=effective,
        )
