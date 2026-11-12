# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module
from gptqmodel.looper.loop_processor import (
    CalibrationTypes,
    CheckpointSupport,
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
        lifecycle_spec=ProcessorLifecycleConfig(
            calibration=CalibrationTypes.EACH_MODULE_NEEDS_INPUT,
            checkpointing=CheckpointSupport.PRIMARY,
        )
    )

    policy = _resolve_subset_calibration_coverage_policy(processor, fallback=None)

    assert policy.validate_input_coverage
    assert policy.prune_uncovered_modules


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
