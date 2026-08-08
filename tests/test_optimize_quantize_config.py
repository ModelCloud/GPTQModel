# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Configuration coverage for the production quantization CLI."""

from argparse import Namespace

import pytest

from gptqmodel.quantization.config import (
    ExpertsRoutingBypass,
    LengthAwareMode,
    VramStrategy,
)
from optimize._common import build_quantize_config


def _args(*, group_size: int, hessian_length_aware: bool) -> Namespace:
    return Namespace(
        bits=4,
        group_size=group_size,
        desc_act=False,
        act_group_aware=True,
        scale_search="activation",
        moe_routing_bypass=True,
        moe_batch_size=None,
        vram_strategy=None,
        dense_vram_strategy=VramStrategy.BALANCED.value,
        moe_vram_strategy=VramStrategy.BALANCED.value,
        calibration_data_device=None,
        adaptive_damping=False,
        adaptive_clipping=False,
        hessian_length_aware=hessian_length_aware,
    )


@pytest.mark.parametrize("group_size", [64, 128])
def test_build_quantize_config_matches_maca_moe_reproducer(group_size):
    qcfg = build_quantize_config(_args(group_size=group_size, hessian_length_aware=True))

    assert qcfg.bits == 4
    assert qcfg.group_size == group_size
    assert qcfg.desc_act is False
    assert qcfg.act_group_aware is True
    assert qcfg.scale_search == "activation"
    assert isinstance(qcfg.moe.routing, ExpertsRoutingBypass)
    assert qcfg.moe.execution.batch_size is None
    assert qcfg.moe.execution.parallel_input_capture is True
    assert qcfg.dense_vram_strategy is VramStrategy.BALANCED
    assert qcfg.moe_vram_strategy is VramStrategy.BALANCED
    assert qcfg.adaptive_damping.enabled is False
    assert qcfg.adaptive_clipping.enabled is False
    assert qcfg.hessian.length_aware.mode is LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT
    assert qcfg.hessian.length_aware.target_bucket_count == 6
    assert qcfg.hessian.length_aware.bucket_weight_exponent == 0.2


def test_hessian_length_aware_flag_changes_cli_configuration():
    disabled = build_quantize_config(_args(group_size=64, hessian_length_aware=False))
    enabled = build_quantize_config(_args(group_size=64, hessian_length_aware=True))

    assert disabled.hessian.length_aware.mode is LengthAwareMode.DISABLED
    assert enabled.hessian.length_aware.mode is LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT
    assert disabled.hessian.to_dict() != enabled.hessian.to_dict()
