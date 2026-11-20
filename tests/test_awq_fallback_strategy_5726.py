# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ must apply the configured fallback strategy, not always min/max RTN.

Issue #5726: ``fallback.strategy`` was accepted and serialized, but every AWQ
fallback module went through ``apply_quant(..., scales_list=[])`` and staged
plain min/max RTN. The strategy never reached the quantizer, so MIDPOINT,
MEAN, MEDIAN and STDCLIP produced the same checkpoint as RTN.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import bind_resolved_module_config
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import (
    AWQConfig,
    CalibrationConfig,
    Fallback,
    FallbackStrategy,
)


BITS = 4
GROUP_SIZE = 16


def _weight() -> torch.Tensor:
    """A skewed fixture whose robust centers and clip bounds differ from min/max."""

    generator = torch.Generator().manual_seed(5726)
    weight = torch.randn(8, 64, generator=generator) * 0.2
    weight[:, 0] = torch.linspace(2.0, 5.0, weight.shape[0])
    weight[:, 1] = -torch.linspace(0.5, 2.0, weight.shape[0])
    return weight


def _bounds(groups: torch.Tensor, strategy: FallbackStrategy):
    if strategy is FallbackStrategy.MIDPOINT:
        return groups.amin(dim=1, keepdim=True), groups.amax(dim=1, keepdim=True)
    if strategy is FallbackStrategy.MEAN:
        center = groups.mean(dim=1, keepdim=True)
        deviation = (groups - center).abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        return center - deviation, center + deviation
    if strategy is FallbackStrategy.MEDIAN:
        center = groups.median(dim=1, keepdim=True).values
        deviation = (groups - center).abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        return center - deviation, center + deviation
    if strategy is FallbackStrategy.STDCLIP:
        center = groups.mean(dim=1, keepdim=True)
        deviation = groups.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-8)
        return center - 3.0 * deviation, center + 3.0 * deviation
    raise AssertionError(f"oracle does not cover {strategy}")


def _oracle(weight: torch.Tensor, strategy: FallbackStrategy, sym: bool):
    """Independent AWQ-layout oracle for the strategy-selected range."""

    groups = weight.reshape(-1, GROUP_SIZE)
    minimum, maximum = _bounds(groups, strategy)
    rows = weight.shape[0]
    if sym:
        maximum = torch.maximum(minimum.abs(), maximum.abs()).clamp(min=1e-5)
        max_int = 2 ** (BITS - 1) - 1
        scale = maximum / max_int
        codes = torch.clamp(torch.round(groups / scale), -max_int - 1, max_int)
        zeros = torch.full_like(scale.reshape(rows, -1), 2 ** (BITS - 1))
        return (codes * scale).reshape_as(weight), scale.reshape(rows, -1), zeros

    zero = torch.zeros_like(minimum)
    minimum = torch.minimum(minimum, zero)
    maximum = torch.maximum(maximum, zero)
    max_code = 2**BITS - 1
    scale = (maximum - minimum).clamp(min=1e-5) / max_code
    zeros = torch.round(-minimum / scale).clamp(0, max_code)
    codes = torch.clamp(torch.round(groups / scale) + zeros, 0, max_code)
    return (
        ((codes - zeros) * scale).reshape_as(weight),
        scale.reshape(rows, -1),
        zeros.reshape(rows, -1),
    )


def _run(
    weight: torch.Tensor,
    strategy: FallbackStrategy,
    sym: bool,
    *,
    dynamic=None,
    scales_list=None,
):
    model = nn.Module()
    model.proj = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        model.proj.weight.copy_(weight)

    qcfg = AWQConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=sym,
        device="cpu",
        fallback=Fallback(strategy=strategy, threshold=False),
        dynamic=dynamic,
    )
    processor = AWQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=CalibrationConfig(),
        gptq_model=SimpleNamespace(
            model=model,
            qlinear_kernel=AwqTorchLinear,
            rotary_embedding=None,
            lm_head="lm_head",
        ),
        model=model,
        calculate_w_wq_diff=False,
    )
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None

    named = NamedModule(model.proj, "proj", "proj", 0)
    bind_resolved_module_config(named, qcfg.resolve_module_config("proj"))
    processor.apply_quant({"proj": named}, [] if scales_list is None else scales_list)
    return (
        named.state["wq"].clone(),
        named.state["q_scales"].clone(),
        named.state["q_zeros"].clone(),
    )


def _assert_matches(actual, expected):
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("sym", [False, True], ids=["affine", "symmetric"])
@pytest.mark.parametrize(
    "strategy",
    [FallbackStrategy.MEAN, FallbackStrategy.MEDIAN, FallbackStrategy.STDCLIP],
    ids=["mean", "median", "stdclip"],
)
def test_fallback_apply_quant_uses_the_configured_strategy(strategy, sym):
    weight = _weight()
    actual = _run(weight, strategy, sym)
    _assert_matches(actual, _oracle(weight, strategy, sym))

    rtn = _run(weight, FallbackStrategy.RTN, sym)
    assert not torch.equal(actual[0], rtn[0]) or not torch.equal(actual[1], rtn[1])


@pytest.mark.parametrize("sym", [False, True], ids=["affine", "symmetric"])
def test_midpoint_remains_the_rtn_control(sym):
    weight = _weight()
    actual = _run(weight, FallbackStrategy.MIDPOINT, sym)
    rtn = _run(weight, FallbackStrategy.RTN, sym)

    for got, want in zip(actual, rtn):
        assert torch.equal(got, want)


def test_dynamic_fallback_strategy_overrides_the_run_default():
    weight = _weight()
    dynamic = {"proj": {"fallback": {"strategy": "median", "threshold": False}}}

    actual = _run(weight, FallbackStrategy.RTN, False, dynamic=dynamic)
    _assert_matches(actual, _oracle(weight, FallbackStrategy.MEDIAN, False))

    default = _run(weight, FallbackStrategy.RTN, False)
    assert not torch.equal(actual[0], default[0])


def test_scaled_path_keeps_the_rtn_grid():
    """Only the empty-scales fallback path consumes ``fallback.strategy``."""

    weight = _weight()
    actual = _run(
        weight,
        FallbackStrategy.MEDIAN,
        False,
        scales_list=[("proj", ["proj"], torch.ones(weight.shape[1]), 0.0)],
    )
    rtn = _run(weight, FallbackStrategy.RTN, False)

    for got, want in zip(actual, rtn):
        assert torch.equal(got, want)
