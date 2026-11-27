# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""`FitLimits` is the only description of what a weight grid may select.

A weight grid is fitted on an FP32 working copy and narrowed back to the dtype
the module replays in. A symmetric grid's negative endpoint reconstructs
``2**bits * amax / (2**bits - 1)``, strictly wider than ``amax``, so an
unconstrained grid can narrow to ``inf`` and lose the codes the packer recovers
from the replay weight (#6001, #6012, #6020).

These tests pin the limits themselves, the single entry that consumes them, the
solvers that call it, and the smoothing path where a per-row factor must not
broadcast the scale floor into ``[rows, rows]``.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.config import (
    FORMAT,
    Fallback,
    FallbackStrategy,
    QuantizeConfig,
    RTNConfig,
    SmoothRowCol,
)
from gptqmodel.quantization.gptq import GPTQ, _fused_code_bounds
from gptqmodel.quantization.quantizer import (
    SCALE_STORAGE_FLOOR,
    FitLimits,
    Quantizer,
)
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


def _gptq(module, *, bits=4):
    task = GPTQ(module, QuantizeConfig(bits=bits, group_size=GROUP_SIZE, sym=True))
    task.quantizer.configure(perchannel=True)
    return task


def test_limits_follow_the_replay_dtype():
    fp16 = FitLimits.for_dtype(torch.float16)
    assert fp16.max_abs_value == FP16_MAX
    assert fp16.scale_floor == SCALE_STORAGE_FLOOR
    assert FitLimits.for_dtype(torch.bfloat16).max_abs_value == torch.finfo(torch.bfloat16).max
    # FP32 is the working dtype and integer weights express counts: no cap.
    assert FitLimits.for_dtype(torch.float32).max_abs_value is None
    assert FitLimits.for_dtype(torch.int8).max_abs_value is None


def test_rtn_bounds_fp32_while_gptq_leaves_it_free():
    """The two solvers have different, documented FP32 rules (#6019 review)."""

    assert (
        FitLimits.for_dtype(torch.float32, bound_fp32=True).max_abs_value
        == torch.finfo(torch.float32).max
    )
    assert FitLimits.for_dtype(torch.float32).max_abs_value is None

    fp32 = nn.Linear(WIDTH, WIDTH, bias=False, dtype=torch.float32)
    free = _gptq(fp32)
    free._fit_weight_params(_symmetric_weight(61440.0, torch.float32))
    assert free.quantizer.code_lo is None


def test_limits_are_read_from_a_module_weight():
    layer = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)

    limits = FitLimits.for_module(layer)

    assert limits.dtype == torch.bfloat16


def test_restore_factor_moves_both_limits():
    limits = FitLimits.for_dtype(torch.float16, restore_factor=4.0)

    assert limits.max_abs_value == pytest.approx(FP16_MAX / 4.0)
    assert limits.scale_floor == pytest.approx(SCALE_STORAGE_FLOOR / 4.0)


def test_a_per_row_factor_is_flattened_for_the_fit():
    """A ``[rows, 1]`` smoothing factor must not become a ``[rows, rows]`` floor.

    This is the regression the review found: ``find_params`` fits ``[rows]`` and
    broadcasts the floor against it, so the old call sites flattened the factor
    explicitly and the first version of this contract did not.
    """

    rows = 4
    quantizer = Quantizer(qcfg=QuantizeConfig(bits=4, group_size=GROUP_SIZE), name="probe")
    quantizer.configure(perchannel=True)
    factor = torch.tensor([[1.0], [2.0], [4.0], [8.0]])

    grid = quantizer.fit_weight(
        torch.randn(rows, WIDTH, dtype=torch.float16),
        FitLimits.for_dtype(torch.float16, restore_factor=factor),
    )

    # One scale per fitted lane: `[rows]` or `[rows, 1]`, never the
    # `[rows, rows]` broadcast the unflattened floor produced.
    assert grid.scale.numel() == rows
    assert tuple(grid.scale.shape) in {(rows,), (rows, 1)}
    assert torch.isfinite(grid.scale).all()
    assert grid.scale.min() >= SCALE_STORAGE_FLOOR


def test_limits_reject_bad_input():
    with pytest.raises(TypeError, match="must be a torch.dtype"):
        FitLimits(dtype="float16", max_abs_value=None, scale_floor=SCALE_STORAGE_FLOOR)
    with pytest.raises(ValueError, match="scale_floor"):
        FitLimits(dtype=torch.float16, max_abs_value=1.0, scale_floor=0.0)
    with pytest.raises(ValueError, match="max_abs_value"):
        FitLimits(dtype=torch.float16, max_abs_value=-1.0, scale_floor=SCALE_STORAGE_FLOOR)
    with pytest.raises(ValueError, match="restore_factor"):
        FitLimits.for_dtype(torch.float16, restore_factor=0.0)
    with pytest.raises(TypeError, match="needs a module weight"):
        FitLimits.for_module(object())


def test_a_weight_fit_without_limits_is_refused():
    quantizer = Quantizer(qcfg=QuantizeConfig(bits=4, group_size=GROUP_SIZE), name="probe")

    with pytest.raises(TypeError, match="requires FitLimits"):
        quantizer.fit_weight(torch.randn(8, 8), {"dtype": torch.float16})


@pytest.mark.parametrize("magnitude,expected", [(61408.0, -FP16_MAX), (61440.0, -57344.0)])
def test_the_rtn_endpoint_comes_from_the_limits(magnitude, expected):
    _layer, (quantized, scale, zero, g_idx, *_rest) = _rtn(_symmetric_weight(magnitude, torch.float16))

    assert torch.isfinite(quantized).all()
    assert quantized[0, 0].item() == expected

    codes = torch.clamp(
        torch.round(quantized.float() / scale.float()) + zero.float(), 0, 15
    ).to(torch.int64)
    rebuilt = ((codes.float() - zero.float()) * scale.float()).to(torch.float16)
    assert rebuilt[0, 0].item() == expected


def test_ordinary_range_packing_is_unchanged_by_the_contract():
    weight = _symmetric_weight(1000.0, torch.float16)
    layer, (quantized, scale, zero, *_rest) = _rtn(weight)

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


def test_the_gptq_weight_fit_is_bounded():
    fp16 = nn.Linear(WIDTH, WIDTH, bias=False, dtype=torch.float16)
    task = _gptq(fp16)
    task._fit_weight_params(_symmetric_weight(61440.0, torch.float16).to(torch.float32))

    assert task.quantizer.code_lo is not None
    assert int(task.quantizer.code_lo.min()) > 0
    reconstruction = task.quantizer.quantize(
        _symmetric_weight(61440.0, torch.float16).to(torch.float32)
    )
    assert torch.isfinite(reconstruction).all()


def test_each_group_is_fitted_with_its_own_bound():
    small = torch.zeros(WIDTH, dtype=torch.float32)
    small[0], small[1] = -1000.0, 1000.0
    large = torch.zeros(WIDTH, dtype=torch.float32)
    large[0], large[1] = -61440.0, 61440.0

    quantizer = Quantizer(qcfg=QuantizeConfig(bits=4, group_size=GROUP_SIZE), name="probe")
    quantizer.configure(perchannel=True)
    quantizer.fit_weight(small.unsqueeze(0), FitLimits.for_dtype(torch.float16))
    small_lo = int(quantizer.code_lo.min())
    quantizer.fit_weight(large.unsqueeze(0), FitLimits.for_dtype(torch.float16))
    large_lo = int(quantizer.code_lo.min())

    assert small_lo == 0
    assert large_lo > small_lo


def test_the_fit_returns_scale_zero_and_range_together():
    """The fused consumer must read the range of the group it just fitted."""

    quantizer = Quantizer(qcfg=QuantizeConfig(bits=4, group_size=GROUP_SIZE), name="probe")
    quantizer.configure(perchannel=True)
    weight = _symmetric_weight(61440.0, torch.float16).to(torch.float32)

    grid = quantizer.fit_weight(weight, FitLimits.for_dtype(torch.float16))

    assert grid.scale is quantizer.scale
    assert grid.zero is quantizer.zero
    assert grid.bounded
    lo, hi = _fused_code_bounds(quantizer, torch.device("cpu"))
    assert torch.equal(lo, grid.code_lo.flatten())
    assert torch.equal(hi, grid.code_hi.flatten())


def test_rtn_row_smoothing_survives_the_real_path():
    """A real ``[rows, 1]`` factor through ``RTN.quantize`` (review regression)."""

    linear = nn.Linear(128, 8, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1e-8)
    qcfg = RTNConfig(
        bits=4, group_size=128, sym=True,
        smooth=SmoothRowCol(axis="row"), offload_to_disk=False,
    )

    quantized, scale, zero, _g_idx, *_ = RTN(linear, qcfg).quantize()

    assert torch.isfinite(quantized).all()
    assert scale.min().item() >= SCALE_STORAGE_FLOOR


def test_gptq_rtn_fallback_survives_row_smoothing():
    """The GPTQ fallback path with the same ``[rows, 1]`` factor."""

    module = nn.Linear(128, 32, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(-0.8, 0.9, 128).repeat(32, 1) * 2.0**-24)
    config = QuantizeConfig(
        bits=4, group_size=128, sym=True,
        fallback=Fallback(strategy=FallbackStrategy.RTN, smooth=SmoothRowCol(axis="row")),
        offload_to_disk=False,
    )
    task = GPTQ(module, config)
    task.quantizer.configure(perchannel=True)

    quantized, scale, zero, groups, *_ = task._fallback_quantize(
        FallbackStrategy.RTN, blocksize=128
    )

    assert torch.isfinite(quantized).all()
    assert scale.min().item() >= SCALE_STORAGE_FLOOR

def test_a_calibrated_gptq_fit_recovers_the_bounded_endpoint():
    """Calibrated GPTQ: fit through `FitLimits`, then the packer's code recovery."""

    layer = nn.Linear(WIDTH, WIDTH, bias=False, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.copy_(_symmetric_weight(61440.0, torch.float16))
    task = GPTQ(
        layer,
        QuantizeConfig(bits=4, group_size=GROUP_SIZE, sym=True, desc_act=False, offload_to_disk=False),
    )
    task.quantizer.configure(perchannel=True)
    task.add_batch(torch.eye(WIDTH, dtype=torch.float16).unsqueeze(0), None)
    quantized, scale, zero, g_idx, *_ = task.quantize()

    assert torch.isfinite(quantized).all()
    assert quantized[0, 0].item() == -57344.0

    codes = torch.clamp(
        torch.round(quantized.float().T / scale.float().T[g_idx.long()])
        + zero.float().T[g_idx.long()],
        0,
        15,
    ).T
    rebuilt = ((codes - zero.float()) * scale.float()).to(torch.float16)
    assert rebuilt[0, 0].item() == -57344.0
