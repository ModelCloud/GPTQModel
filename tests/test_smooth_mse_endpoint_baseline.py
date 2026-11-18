# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""SmoothMSE must score the unshrunk grid as a candidate.

``mse_optimal_quant`` started with ``best_p = 1`` but ``best_err = inf`` and
never evaluated ``p = 1``. The first interior ternary probe therefore replaced
the incumbent, and the later refinement only looked near that probe, so an
optimal unshrunk grid could never be selected. A constant row quantized to
``0.9979189634`` (asymmetric) and ``0.9313910604`` (symmetric) instead of the
exact/nearest grid.

The expected values below are computed from scratch in float64 rather than
from the search's own scoring, so they stay independent of the implementation.
"""

import pytest
import torch

from gptqmodel.quantization.config import (
    Fallback,
    FallbackStrategy,
    GPTQConfig,
    QQQConfig,
    QuantizeConfig,
    RTNConfig,
    SmoothMSE,
)
from gptqmodel.quantization.fallback_smooth import mse_optimal_quant
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ
from gptqmodel.quantization.rtn import RTN


MAXQ = 15
GROUP_SIZE = 128
DEVICES = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def _endpoint_baseline_mse(block: torch.Tensor, *, sym: bool) -> torch.Tensor:
    """Per-row MSE of the plain ``p = 1`` quantization, in float64."""
    ref = block.double()
    if sym:
        hi = ref.abs().amax(dim=1, keepdim=True)
        lo = -hi
        zero = torch.full_like(hi, (MAXQ + 1) / 2.0)
    else:
        # The affine grid only represents zero points in [0, maxq], so the
        # fitted range must contain zero for every group.
        zero = torch.zeros_like(ref[:, :1])
        lo = torch.minimum(ref.amin(dim=1, keepdim=True), zero)
        hi = torch.maximum(ref.amax(dim=1, keepdim=True), zero)
        zero = None
    scale = ((hi - lo) / MAXQ).clamp_min(1e-8)
    z = zero if sym else torch.round(-lo / scale)
    codes = (ref / scale + z).round().clamp(0, MAXQ)
    return ((codes - z) * scale - ref).pow(2).mean(dim=1)


def _mse(block: torch.Tensor, dequant: torch.Tensor) -> torch.Tensor:
    """Per-row MSE of ``dequant`` against ``block``, returned on the CPU."""
    return (dequant.double() - block.double()).pow(2).mean(dim=1).cpu()


def _search_mse(block: torch.Tensor, *, sym: bool, device: str) -> torch.Tensor:
    qcfg = QuantizeConfig(bits=4, group_size=GROUP_SIZE, sym=sym, smooth=SmoothMSE())
    dequant, _scale, _zero = mse_optimal_quant(
        block.to(device), qcfg, MAXQ, steps=32, maxshrink=0.8
    )
    return _mse(block.to(device), dequant)


CASES = {
    "normal": torch.ones(1, GROUP_SIZE),
    "negative_constant": -torch.ones(1, GROUP_SIZE),
    "all_zero": torch.zeros(1, GROUP_SIZE),
    "mixed_sign": torch.linspace(-2.0, 3.0, GROUP_SIZE).unsqueeze(0),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sym", [False, True], ids=["asymmetric", "symmetric"])
@pytest.mark.parametrize("case", sorted(CASES), ids=sorted(CASES))
def test_smooth_mse_is_never_worse_than_the_unshrunk_baseline(case, sym, device):
    block = CASES[case]
    baseline = _endpoint_baseline_mse(block, sym=sym)
    searched = _search_mse(block, sym=sym, device=device)

    assert bool((searched <= baseline + 1e-9).all()), (
        f"{case} sym={sym}: searched MSE {searched.tolist()} exceeds "
        f"the unshrunk baseline {baseline.tolist()}"
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sym", [False, True], ids=["asymmetric", "symmetric"])
def test_smooth_mse_still_picks_a_better_interior_shrink(sym, device):
    """Control: ending the search on an interior best is still allowed."""
    torch.manual_seed(0)
    block = torch.randn(4, GROUP_SIZE) * torch.tensor([1.0, 0.5, 0.2, 0.1]).reshape(4, 1)
    block[0, :4] *= 8

    baseline = _endpoint_baseline_mse(block, sym=sym)
    searched = _search_mse(block, sym=sym, device=device)

    assert bool((searched <= baseline + 1e-9).all())
    assert bool((searched < baseline).any()), (
        "the interior search must still beat the endpoint when it measures a "
        f"lower error: searched {searched.tolist()}, baseline {baseline.tolist()}"
    )


def _rtn_dequant(block, qcfg):
    source = torch.nn.Linear(GROUP_SIZE, block.shape[0], bias=False, device=block.device)
    with torch.no_grad():
        source.weight.copy_(block)
    return RTN(source, qcfg).quantize()[0]


def _gptq_dequant(block, qcfg):
    source = torch.nn.Linear(GROUP_SIZE, block.shape[0], bias=False, device=block.device)
    with torch.no_grad():
        source.weight.copy_(block)
    task = GPTQ(module=source, qcfg=qcfg)
    task.quantizer.configure(perchannel=True)
    dequant, *_ = task._fallback_quantize(FallbackStrategy.RTN, blocksize=GROUP_SIZE)
    return dequant


def _qqq_dequant(block, qcfg):
    source = torch.nn.Linear(GROUP_SIZE, block.shape[0], bias=False, device=block.device)
    with torch.no_grad():
        source.weight.copy_(block)
    task = QQQ(source, qcfg=qcfg)
    task.quantizer.configure(
        bits=qcfg.bits, perchannel=True, sym=qcfg.sym, groupsize=qcfg.group_size
    )
    dequant, *_ = task._fallback_quantize(FallbackStrategy.RTN)
    return dequant


CONSUMERS = {"rtn": (RTNConfig, _rtn_dequant), "gptq": (GPTQConfig, _gptq_dequant), "qqq": (QQQConfig, _qqq_dequant)}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sym", [False, True], ids=["asymmetric", "symmetric"])
@pytest.mark.parametrize("consumer", sorted(CONSUMERS))
def test_smooth_mse_consumers_return_a_grid_no_worse_than_the_baseline(
    consumer, sym, device
):
    """``rtn.py``, ``gptq.py`` and ``qqq.py`` share the same search contract."""
    config_cls, dequantize = CONSUMERS[consumer]
    block = torch.cat(
        [CASES["normal"], CASES["negative_constant"], CASES["all_zero"], CASES["mixed_sign"]],
        dim=0,
    ).to(device)
    qcfg = config_cls(
        bits=4,
        group_size=GROUP_SIZE,
        sym=sym,
        smooth=SmoothMSE(),
        fallback=Fallback(strategy=FallbackStrategy.RTN, smooth=SmoothMSE()),
        offload_to_disk=False,
    )
    searched = _mse(block, dequantize(block, qcfg))
    baseline = _endpoint_baseline_mse(block.cpu(), sym=sym)
    assert bool((searched <= baseline + 1e-9).all()), (
        f"{consumer} sym={sym}: MSE {searched.tolist()} exceeds baseline {baseline.tolist()}"
    )
