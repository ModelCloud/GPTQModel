# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTQProcessor must pass the calibration token count to every task (#5755).

GPTAQ and FOEM tasks were constructed without ``expected_nsamples``, so a
percentage fallback threshold such as ``"50%"`` was resolved against a base of
one.  A routed expert that saw 4 of 100 calibration tokens therefore ran the
Hessian solver instead of the RTN fallback, unlike the equivalent absolute
threshold.
"""

import pytest
import torch

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import (
    CalibrationConfig,
    Fallback,
    FOEMConfig,
    GPTAQConfig,
    QuantizeConfig,
)


CALIBRATION_TOKENS = 100
COLUMNS, ROWS = 16, 8


def _run(kind, *, threshold, observed):
    generator = torch.Generator().manual_seed(499)
    calibration = CalibrationConfig(
        dataset=[{"input_ids": torch.arange(CALIBRATION_TOKENS).reshape(1, CALIBRATION_TOKENS)}]
    ).bind(lambda calibration_dataset, **_: calibration_dataset)
    variant = {
        "gptq": {},
        "gptaq": {"gptaq": GPTAQConfig(alpha=0.25)},
        "foem": {"foem": FOEMConfig(alpha=0.25, beta=0.0)},
    }[kind]
    config = QuantizeConfig(
        bits=4,
        group_size=8,
        desc_act=False,
        act_group_aware=False,
        fallback=Fallback(threshold=threshold),
        **variant,
    )

    processor = GPTQProcessor(tokenizer=None, qcfg=config, calibration=calibration)
    assert processor.total_calibration_tokens == CALIBRATION_TOKENS

    layer = torch.nn.Linear(COLUMNS, ROWS, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(ROWS, COLUMNS, generator=generator) * 0.1)
    named = NamedModule(
        layer,
        name="down_proj",
        full_name="model.layers.0.down_proj",
        layer_index=0,
    )
    named.state["is_routed_expert"] = True
    named.state[NATIVE_INPUTS_STATE_KEY] = []

    processor.preprocess(named)
    task = processor.tasks[named.name]
    task.add_batch(torch.randn(observed, COLUMNS, generator=generator), None)
    weight, _, _, _, _, loss, _, count = task.quantize()

    assert count == observed
    return weight, str(loss).startswith("fallback(rtn)"), task.expected_nsamples


@pytest.mark.parametrize("kind", ["gptq", "gptaq", "foem"])
def test_processor_percentage_threshold_matches_absolute_below_cutoff(kind):
    absolute_weight, absolute_fell_back, absolute_expected = _run(
        kind, threshold=50, observed=4
    )
    percent_weight, percent_fell_back, percent_expected = _run(
        kind, threshold="50%", observed=4
    )

    assert absolute_expected == percent_expected == CALIBRATION_TOKENS
    assert absolute_fell_back and percent_fell_back
    assert torch.equal(percent_weight, absolute_weight)


@pytest.mark.parametrize("kind", ["gptq", "gptaq", "foem"])
def test_processor_percentage_threshold_runs_the_solver_above_cutoff(kind):
    solver_weight, fell_back, expected = _run(kind, threshold="50%", observed=60)
    reference_weight, reference_fell_back, _ = _run(kind, threshold=0, observed=60)

    assert expected == CALIBRATION_TOKENS
    assert not fell_back and not reference_fell_back
    assert torch.equal(solver_weight, reference_weight)
