# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.looper.output_replay import resolve_output_replay_execution
from gptqmodel.looper.stage_subset import CalibrationCoveragePolicy, SubsetPlan


def _plan(*, batches=2, moe=True, forward_mode="serial"):
    modules = {}
    return SubsetPlan(
        modules=modules,
        subset_index=0,
        subset_total=1,
        execute_forward=True,
        replay_after_process=True,
        forward_mode=forward_mode,
        batch_count=batches,
        forward_row_counts=[1] * batches,
        forward_total_rows=batches,
        moe_groups={"experts": ["mlp.experts.0.down_proj"]} if moe else {},
        forward_device_map={"mlp.experts.0.down_proj": torch.device("cuda:1")},
        calibration_coverage_policy=CalibrationCoveragePolicy(
            False, True, False, False
        ),
        module_chunks=[modules],
    )


def test_parallel_moe_output_replay_uses_replicas_without_subset_overrides():
    execution = resolve_output_replay_execution(_plan(), parallel_moe_replay=True)

    assert execution.forward_device_map == {}
    assert execution.force_serial is False
    assert execution.preserve_module_devices is False
    assert execution.install_device_overrides is False


def test_single_batch_moe_output_replay_keeps_subset_placement():
    plan = _plan(batches=1)
    execution = resolve_output_replay_execution(plan, parallel_moe_replay=True)

    assert execution.forward_device_map == plan.forward_device_map
    assert execution.force_serial is True
    assert execution.preserve_module_devices is True
    assert execution.install_device_overrides is True


def test_dense_or_disabled_output_replay_keeps_subset_placement():
    for plan, enabled in ((_plan(moe=False), True), (_plan(), False)):
        execution = resolve_output_replay_execution(plan, parallel_moe_replay=enabled)
        assert execution.forward_device_map == plan.forward_device_map
        assert execution.force_serial is True
        assert execution.preserve_module_devices is True


def test_unplanned_output_replay_retains_generic_parallel_policy():
    execution = resolve_output_replay_execution(None, parallel_moe_replay=True)

    assert execution.forward_device_map == {}
    assert execution.force_serial is False
    assert execution.preserve_module_devices is False
