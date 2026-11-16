# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Asymmetric SmoothMSE must emit representable zero points (#3867).

An affine 4-bit grid only has zero points `0..maxq`. Fitting the raw group
minimum and maximum produced a negative zero point for a positive-only group
and one above `maxq` for a negative-only group, so packing moved the weights by
more than a full quantization step (about 1.57 for a `[0.5, 2.0]` group).
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import RTNConfig, SmoothMSE
from gptqmodel.quantization.rtn import RTN


def _quantize(sign: int):
    source = torch.nn.Linear(128, 32, bias=False)
    with torch.no_grad():
        source.weight.copy_(sign * torch.linspace(0.5, 2.0, 128).repeat(32, 1))

    config = RTNConfig(
        bits=4,
        group_size=128,
        sym=False,
        smooth=SmoothMSE(),
        offload_to_disk=False,
    )
    dequant, scales, zeros, groups, *_ = RTN(source, config).quantize()
    return source, dequant, scales, zeros, groups


@pytest.mark.parametrize("sign", [1, -1], ids=["positive_only", "negative_only"])
def test_asymmetric_smooth_mse_keeps_zero_point_representable(sign):
    maxq = 15
    _, _, scales, zeros, _ = _quantize(sign)

    assert zeros.min() >= 0, f"zero point {zeros.min().item()} is below 0"
    assert zeros.max() <= maxq, f"zero point {zeros.max().item()} is above {maxq}"
    # The stored scale must describe the same grid the optimizer selected.
    step = scales.float()
    assert step.min() > 0


@pytest.mark.parametrize("sign", [1, -1], ids=["positive_only", "negative_only"])
def test_asymmetric_smooth_mse_error_stays_within_one_step(sign):
    source, dequant, scales, _, _ = _quantize(sign)
    step = scales[0, 0].float()

    error = (dequant[0] - source.weight[0]).abs().max()
    assert error <= step / 2 + 1e-5, f"quantization error {error.item()} exceeds half a step {step.item() / 2}"


@pytest.mark.parametrize("sign", [1, -1], ids=["positive_only", "negative_only"])
def test_asymmetric_smooth_mse_packing_reproduces_the_scored_grid(sign):
    source, dequant, scales, zeros, groups = _quantize(sign)
    step = scales[0, 0].float()

    with torch.no_grad():
        source.weight.copy_(dequant)
    packed = TorchLinear(
        bits=4,
        group_size=128,
        sym=False,
        desc_act=False,
        in_features=128,
        out_features=32,
        bias=False,
    )
    packed.pack_block(source, scales, zeros, groups, workers=1)
    restored = packed.dequantize_weight().T

    # `scales` is re-encoded in the checkpoint dtype, so allow a small fraction
    # of a step. The previous out-of-range zero points shifted the grid by more
    # than ten steps.
    drift = (restored - dequant).abs().max()
    assert drift <= 0.01 * step, f"packed grid drifted {drift.item()} from the scored grid (step {step.item()})"
