# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Paro's stage LRs must follow the optimizer batch size (#5306).

#5279 raised the default ``opt_batch_size`` from 64 to 128, halving the number
of optimizer steps per epoch. At a fixed LR that also halves the per-epoch
update; on Llama-3.2-1B it cost 2.2% PPL. Scaling the stage LRs from the
64-row reference recovers most of that (+0.44% at the best coefficient), but
not all of it, so the default goes back to the accuracy-tuned 64 rows and the
scaling keeps larger opt-in batches close to it.
"""

import pytest

from gptqmodel.looper.paro_processor import ParoProcessor
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig


def _processor(**overrides):
    processor = ParoProcessor.__new__(ParoProcessor)
    processor.qcfg = QuantizeConfig(
        quant_method=METHOD.PARO,
        format=FORMAT.PARO,
        bits=4,
        group_size=128,
        **overrides,
    )
    return processor


def test_reference_batch_keeps_the_configured_lrs():
    """A 64-row step is the reference the defaults were tuned on."""

    assert _processor(opt_batch_size=64)._stage_learning_rates() == pytest.approx(
        (0.05, 1e-5, 1e-6)
    )


def test_default_batch_is_the_accuracy_tuned_reference():
    """The default must stay on the reference that was validated for accuracy.

    LR scaling closes most of the gap but not all of it (+0.44% PPL at the
    best coefficient), so the accuracy-first default is the 64-row step.
    """

    processor = _processor()
    assert processor.qcfg.opt_batch_size == 64
    assert processor._stage_learning_rates() == pytest.approx((0.05, 1e-5, 1e-6))


def test_opt_in_batch_scales_the_lrs_linearly():
    """Opting into 128 rows must still keep the per-epoch update (#5306)."""

    assert _processor(opt_batch_size=128)._stage_learning_rates() == pytest.approx(
        (0.1, 2e-5, 2e-6)
    )


def test_small_batch_keeps_the_configured_lrs():
    """Batches at or below the reference keep the LR they always had."""

    assert _processor(opt_batch_size=16)._stage_learning_rates() == pytest.approx(
        (0.05, 1e-5, 1e-6)
    )


def test_configured_lrs_are_scaled_not_replaced():
    """An explicit LR is still the reference at 64 rows and scales from there."""

    processor = _processor(
        opt_batch_size=128,
        opt_rotation_lr=0.02,
        opt_weight_lr=2e-5,
        opt_quantizer_lr=4e-6,
    )
    assert processor._stage_learning_rates() == pytest.approx((0.04, 4e-5, 8e-6))
