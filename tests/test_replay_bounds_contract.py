# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""`ReplayBounds` is the only description of what a weight grid may reconstruct.

A weight grid is fitted on an FP32 working copy and narrowed back to the dtype
the module replays in. A symmetric grid's negative endpoint reconstructs
``2**bits * amax / (2**bits - 1)``, strictly wider than ``amax``, so an
unconstrained grid can narrow to ``inf`` and lose the codes the packer recovers
from the replay weight (#6001, #6012, #6020).

These tests pin the contract itself and the two solvers that consume it: the
bound is derived once from the replay dtype (and the smoothing factor), a fit
without it is refused, and the endpoint cases keep the exact values the landed
fixes established.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.config import FORMAT, QuantizeConfig, RTNConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.quantizer import SCALE_STORAGE_FLOOR, Quantizer, ReplayBounds
from gptqmodel.quantization.rtn import RTN


WIDTH = 32
GROUP_SIZE = 32
FP16_MAX = torch.finfo(torch.float16).max


def _symmetric_weight(magnitude: float, dtype: torch.dtype) -> torch.Tensor:
    """One group per row: ``[-a, +a, 0, ...]`` sets the fitted range."""

    weight = torch.zeros(WIDTH, WIDTH, dtype=dtype)
    weight[:, 0] = -magnitude
    weight[:, 1] = magnitude
    return weight


def _rtn(weight: torch.Tensor):
    layer = nn.Linear(WIDTH, WIDTH, bias=False, dtype=weight.dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
    qcfg = RTNConfig(
        bits=4, group_size=GROUP_SIZE, sym=True, format=FORMAT.GPTQ,
        desc_act=False, mse=None, gsq=None, offload_to_disk=False,
    )
    return layer, RTN(layer, qcfg).quantize()


def test_bounds_follow_the_replay_dtype():
    assert ReplayBounds(torch.float16).value_limit == FP16_MAX
    assert ReplayBounds(torch.bfloat16).value_limit == torch.finfo(torch.bfloat16).max
    # FP32 is the working dtype and integer weights express counts: no bound.
    assert ReplayBounds(torch.float32).value_limit is None
    assert ReplayBounds(torch.int8).value_limit is None
    assert ReplayBounds(torch.float16).scale_min == SCALE_STORAGE_FLOOR


def test_bounds_are_read_from_a_module_weight():
    layer = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)

    bounds = ReplayBounds.for_module(layer)

    assert bounds.replay_dtype == torch.bfloat16


def test_restore_factor_moves_both_bounds_into_the_working_domain():
    bounds = ReplayBounds(torch.float16, restore_factor=4.0)

    assert bounds.value_limit == pytest.approx(FP16_MAX / 4.0)
    assert bounds.scale_min == pytest.approx(SCALE_STORAGE_FLOOR / 4.0)


def test_a_tensor_restore_factor_scales_per_row():
    factor = torch.tensor([1.0, 2.0])

    bounds = ReplayBounds(torch.float16, restore_factor=factor)

    assert torch.allclose(bounds.value_limit, torch.tensor([FP16_MAX, FP16_MAX / 2.0]))
    assert torch.allclose(bounds.scale_min, SCALE_STORAGE_FLOOR / factor)


def test_bounds_reject_bad_input():
    with pytest.raises(TypeError, match="must be a torch.dtype"):
        ReplayBounds("float16")
    with pytest.raises(ValueError, match="must be positive"):
        ReplayBounds(torch.float16, scale_floor=0.0)
    with pytest.raises(ValueError, match="restore_factor"):
        ReplayBounds(torch.float16, restore_factor=0.0)
    with pytest.raises(TypeError, match="needs a module weight"):
        ReplayBounds.for_module(object())


def test_a_weight_fit_without_bounds_is_refused():
    quantizer = Quantizer(qcfg=QuantizeConfig(bits=4, group_size=GROUP_SIZE), name="probe")

    with pytest.raises(TypeError, match="requires ReplayBounds"):
        quantizer.fit_weight(torch.randn(8, 8), {"replay_dtype": torch.float16})


@pytest.mark.parametrize("magnitude,expected", [(61408.0, -FP16_MAX), (61440.0, -57344.0)])
def test_the_rtn_endpoint_comes_from_the_bounds(magnitude, expected):
    _layer, (quantized, scale, zero, g_idx, *_rest) = _rtn(_symmetric_weight(magnitude, torch.float16))

    assert torch.isfinite(quantized).all()
    assert quantized[0, 0].item() == expected

    # The packer recovers codes from the narrowed weight; the negative lane must
    # land on the code whose reconstruction is the expected finite value.
    codes = torch.clamp(
        torch.round(quantized.float() / scale.float()) + zero.float(), 0, 15
    ).to(torch.int64)
    assert codes[0, 0].item() >= 0
    rebuilt = ((codes.float() - zero.float()) * scale.float()).to(torch.float16)
    assert rebuilt[0, 0].item() == expected


def test_ordinary_range_packing_is_unchanged_by_the_contract():
    weight = _symmetric_weight(1000.0, torch.float16)
    layer, (quantized, scale, zero, *_rest) = _rtn(weight)

    # The same block fitted through find_params with explicit bounds.
    reference = Quantizer(qcfg=QuantizeConfig(bits=4, group_size=GROUP_SIZE), name="probe")
    reference.configure(perchannel=True)
    reference.find_params(
        layer.weight.data.clone(),
        weight=True,
        scale_min=SCALE_STORAGE_FLOOR,
        value_limit=FP16_MAX,
    )
    reference_dequant = reference.quantize(layer.weight.data.clone()).to(quantized.dtype)

    assert torch.equal(quantized, reference_dequant)
    assert torch.equal(scale, reference.scale)
    assert torch.equal(zero, reference.zero)


def _gptq(module, *, bits=4):
    task = GPTQ(module, QuantizeConfig(bits=bits, group_size=GROUP_SIZE, sym=True))
    # ``find_params`` reads the configured mode; GPTQ's own block pass does this
    # in ``quantize()``.
    task.quantizer.configure(perchannel=True)
    return task


def test_the_gptq_weight_fit_is_bounded_and_fp32_stays_free():
    fp16 = nn.Linear(WIDTH, WIDTH, bias=False, dtype=torch.float16)
    task = _gptq(fp16)
    task._fit_weight_params(_symmetric_weight(61440.0, torch.float16).to(torch.float32))

    assert task.quantizer.code_lo is not None
    assert int(task.quantizer.code_lo.min()) > 0, "the overflowing endpoint code is excluded"
    reconstruction = task.quantizer.quantize(
        _symmetric_weight(61440.0, torch.float16).to(torch.float32)
    )
    assert torch.isfinite(reconstruction).all()

    fp32 = nn.Linear(WIDTH, WIDTH, bias=False, dtype=torch.float32)
    free = _gptq(fp32)
    free._fit_weight_params(_symmetric_weight(61440.0, torch.float32))

    assert free.quantizer.code_lo is None
    assert free.quantizer.code_hi is None


def test_each_group_is_fitted_with_its_own_bound():
    # Row 0 stays in range (code 0 is legal); row 1 would overflow at code 0.
    small = torch.zeros(WIDTH, dtype=torch.float32)
    small[0], small[1] = -1000.0, 1000.0
    large = torch.zeros(WIDTH, dtype=torch.float32)
    large[0], large[1] = -61440.0, 61440.0

    quantizer = Quantizer(qcfg=QuantizeConfig(bits=4, group_size=GROUP_SIZE), name="probe")
    quantizer.configure(perchannel=True)
    quantizer.fit_weight(small.unsqueeze(0), ReplayBounds(torch.float16))
    small_lo = int(quantizer.code_lo.min())
    quantizer.fit_weight(large.unsqueeze(0), ReplayBounds(torch.float16))
    large_lo = int(quantizer.code_lo.min())

    assert small_lo == 0
    assert large_lo > 0
    assert small_lo < large_lo

def test_the_fit_returns_scale_zero_and_range_together():
    """A consumer must not recompute the range from a different scale."""

    quantizer = Quantizer(qcfg=QuantizeConfig(bits=4, group_size=GROUP_SIZE), name="probe")
    quantizer.configure(perchannel=True)
    weight = _symmetric_weight(61440.0, torch.float16).to(torch.float32)

    grid = quantizer.fit_weight(weight, ReplayBounds(torch.float16))

    assert grid.scale is quantizer.scale
    assert grid.zero is quantizer.zero
    assert grid.code_lo is quantizer.code_lo
    assert grid.bounded
    assert int(grid.code_lo.min()) > 0
    # The selected range is what the fused path reads through `_fused_code_bounds`.
    from gptqmodel.quantization.gptq import _fused_code_bounds

    lo, hi = _fused_code_bounds(quantizer, torch.device("cpu"))
    assert torch.equal(lo, grid.code_lo.flatten())
    assert torch.equal(hi, grid.code_hi.flatten())
