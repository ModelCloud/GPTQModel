# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Execution policy attachment for accuracy-preserving layer-output replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .stage_subset import SubsetPlan


@dataclass(frozen=True)
class OutputReplayExecution:
    """Resolved replay placement without changing the subset lifecycle plan."""

    forward_device_map: Dict[str, torch.device]
    force_serial: bool
    preserve_module_devices: bool
    install_device_overrides: bool


def resolve_output_replay_execution(
    replay_plan: Optional[SubsetPlan],
    *,
    parallel_moe_replay: bool,
) -> OutputReplayExecution:
    """Allow independent MoE replay batches to use the existing replica executor.

    Subset quantization still follows its explicit per-module placement. Output
    replay has no hooks and only computes the next layer's ordered activations,
    so a multi-batch MoE layer can instead be replicated by ForwardExecutor and
    split across devices. Meta shell tensors remain meta during replica staging.
    """

    if replay_plan is None:
        return OutputReplayExecution(
            forward_device_map={},
            force_serial=False,
            preserve_module_devices=False,
            install_device_overrides=False,
        )

    use_parallel_replicas = (
        parallel_moe_replay
        and replay_plan.batch_count > 1
        and bool(replay_plan.moe_groups)
    )
    if use_parallel_replicas:
        return OutputReplayExecution(
            forward_device_map={},
            force_serial=False,
            preserve_module_devices=False,
            install_device_overrides=False,
        )

    return OutputReplayExecution(
        forward_device_map=replay_plan.forward_device_map,
        force_serial=replay_plan.subset_forward_serial,
        preserve_module_devices=replay_plan.preserve_module_devices,
        install_device_overrides=bool(replay_plan.forward_device_map),
    )
