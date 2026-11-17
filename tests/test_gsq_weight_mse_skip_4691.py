# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""RTN/AWQ GSQ cannot beat nearest rounding under weight MSE (#4691).

With ``learn_scales=False`` the weight-MSE objective is separable per element,
so the code the packer already rounded to is the exact minimizer. The relaxed
search used to run its full step budget on every linear and return the
baseline: 6.7x slower for 0 changed tensors. It must be skipped and reported.
"""

import pytest
import torch
from torch import nn

from gptqmodel.quantization import GSQConfig
from gptqmodel.quantization.config import RTNConfig
from gptqmodel.quantization.gsq_scalar import refine_affine_scalar
from gptqmodel.quantization.rtn import RTN


pytestmark = [pytest.mark.cpu]


def _tensors(rows=3, columns=8, bits=4, group_size=4):
    torch.manual_seed(7)
    target = torch.randn(rows, columns)
    groups = torch.arange(columns, dtype=torch.int32) // group_size
    scales = torch.full((rows, columns // group_size), 0.25)
    zeros = torch.full_like(scales, 8)
    codes = (target / scales[:, groups] + zeros[:, groups]).round().clamp(0, 2**bits - 1)
    weight = scales[:, groups] * (codes - zeros[:, groups])
    return weight, scales, zeros, groups, target


def test_weight_mse_with_fixed_scales_is_skipped():
    weight, scales, zeros, groups, target = _tensors()
    config = GSQConfig(enabled=True, steps=100, candidates=4)

    result = refine_affine_scalar(
        weight, scales, zeros, groups, target=target, bits=4, config=config
    )

    assert result.skipped is True
    assert result.history == [result.before]
    assert result.after == result.before
    assert torch.equal(result.weight, weight)
    assert torch.equal(result.scales, scales)


def test_learning_scales_still_runs_the_search():
    weight, scales, zeros, groups, target = _tensors()
    config = GSQConfig(enabled=True, steps=3, candidates=4, learn_scales=True)

    result = refine_affine_scalar(
        weight, scales, zeros, groups, target=target, bits=4, config=config
    )

    assert result.skipped is False
    assert len(result.history) == config.steps + 1


def test_calibration_objective_still_runs_the_search():
    weight, scales, zeros, groups, target = _tensors()
    config = GSQConfig(enabled=True, steps=3, candidates=4)

    result = refine_affine_scalar(
        weight,
        scales,
        zeros,
        groups,
        target=target,
        bits=4,
        config=config,
        hessian=torch.eye(target.shape[1]),
    )

    assert result.skipped is False
    assert len(result.history) == config.steps + 1


def test_skipped_chunked_rows_keep_one_history_entry():
    weight, scales, zeros, groups, target = _tensors(rows=5, columns=8)
    config = GSQConfig(
        enabled=True, steps=10, candidates=4, max_candidate_bytes=4 * 8 * 4
    )

    result = refine_affine_scalar(
        weight, scales, zeros, groups[:8], target=target, bits=4, config=config
    )

    assert result.skipped is True
    assert len(result.history) == 1
    assert torch.equal(result.weight, weight)


def _rtn_layer(columns=8, rows=3):
    torch.manual_seed(11)
    layer = nn.Linear(columns, rows, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(rows, columns))
    return layer


def test_rtn_with_gsq_reports_the_skip_and_keeps_the_plain_weight():
    plain = RTN(_rtn_layer(), RTNConfig(bits=4, group_size=4))
    plain.quantize()

    with_gsq = RTN(
        _rtn_layer(),
        RTNConfig(bits=4, group_size=4, gsq=GSQConfig(enabled=True, steps=100)),
    )
    with_gsq.quantize()

    assert with_gsq.gsq_diagnostics["skipped"] is True
    assert with_gsq.gsq_diagnostics["after"] == with_gsq.gsq_diagnostics["before"]
    assert torch.equal(with_gsq.module.weight, plain.module.weight)
