# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The low-sample RTN fallback must release the full Hessian before cloning (#5778).

``_fallback_quantize`` used ``self.H`` for the compute device and the SmoothMSE
channel diagonal, but it cleared the matrix only at return, after the weight
clone and the dequantized output buffer had both been allocated. Resolve the
device and the diagonal first, then drop the full matrix before ``clone_module``.
"""

import pytest
import torch

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import (
    Fallback,
    FallbackStrategy,
    FOEMConfig,
    GPTAQConfig,
    QuantizeConfig,
    SmoothMSE,
)
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ


ROWS, COLUMNS, GROUP_SIZE, BLOCKSIZE = 16, 256, 128, 128
SAMPLES = 4
ALPHA = 0.25
KINDS = ["gptq", "gptaq", "foem"]


def _weight() -> torch.Tensor:
    generator = torch.Generator().manual_seed(5778)
    return torch.randn(ROWS, COLUMNS, generator=generator) * 0.05


def _hessian() -> torch.Tensor:
    """A real PSD Hessian whose diagonal is deliberately non-uniform."""

    generator = torch.Generator().manual_seed(5779)
    skew = torch.randn(COLUMNS, COLUMNS, generator=generator)
    skew = (skew - skew.T) / 2
    return torch.diag(torch.linspace(1.0, 2.0, COLUMNS)) + 1e-3 * skew


def _task(kind: str, *, smooth=None, weight=None, hessian=None):
    layer = torch.nn.Linear(COLUMNS, ROWS, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(_weight() if weight is None else weight)

    named = NamedModule(layer, name="proj", full_name="model.layers.0.proj", layer_index=0)
    named.state[NATIVE_INPUTS_STATE_KEY] = []

    common = {
        "bits": 4,
        "group_size": GROUP_SIZE,
        "desc_act": False,
        "fallback": Fallback(strategy=FallbackStrategy.RTN, threshold=100, smooth=smooth),
    }
    if kind == "gptq":
        task = GPTQ(named, QuantizeConfig(**common))
    elif kind == "gptaq":
        task = GPTAQ(named, QuantizeConfig(gptaq=GPTAQConfig(alpha=ALPHA), **common))
    else:
        task = FOEM(named, QuantizeConfig(foem=FOEMConfig(alpha=ALPHA), **common))
    task.quantizer.configure(perchannel=True)
    task.H = _hessian() if hessian is None else hessian
    task.nsamples = SAMPLES
    task.expected_nsamples = None
    if kind == "gptaq":
        task._gptaq_batch_request_count = SAMPLES
    return task


def _spy_on_clone(task, monkeypatch):
    """Record the live Hessian at every weight-clone call."""

    seen = []
    original = task.clone_module

    def spy(*args, **kwargs):
        seen.append(task.H)
        return original(*args, **kwargs)

    monkeypatch.setattr(task, "clone_module", spy)
    return seen


@pytest.mark.parametrize("kind", KINDS)
def test_fallback_releases_the_hessian_before_cloning_the_weight(kind, monkeypatch):
    task = _task(kind)
    seen = _spy_on_clone(task, monkeypatch)

    task._fallback_quantize(FallbackStrategy.RTN, BLOCKSIZE)

    assert len(seen) == 1
    assert seen[0] is None, f"{kind}: the full Hessian was still live at the weight clone"
    assert task.H is None


@pytest.mark.parametrize("kind", KINDS)
def test_low_sample_quantize_releases_the_hessian_before_cloning(kind, monkeypatch):
    task = _task(kind)
    seen = _spy_on_clone(task, monkeypatch)

    task.quantize(blocksize=BLOCKSIZE)

    assert len(seen) == 1
    assert seen[0] is None, f"{kind}: the full Hessian survived into the fallback clone"
    assert task.H is None


def test_smooth_mse_diagonal_still_reaches_the_search():
    """Resolving the diagonal before the release must keep the exact weighting."""

    weight = torch.randn(ROWS, COLUMNS, generator=torch.Generator().manual_seed(5780)) * 0.2
    weight[:, 0] = 20.0
    weight[:, 1] = 1.3
    hessian = torch.zeros(COLUMNS, COLUMNS)
    hessian[1, 1] = 1.0

    weighted = _task("gptq", smooth=SmoothMSE(steps=8), weight=weight, hessian=hessian.clone())
    result = weighted._fallback_quantize(FallbackStrategy.RTN, BLOCKSIZE)

    control = _task("gptq", smooth=SmoothMSE(steps=8), weight=weight)
    control.H = None
    control_result = control._fallback_quantize(
        FallbackStrategy.RTN,
        BLOCKSIZE,
        hessian_diag=hessian.diagonal().clone(),
    )

    unweighted = _task("gptq", smooth=SmoothMSE(steps=8), weight=weight)
    unweighted.H = None
    unweighted_result = unweighted._fallback_quantize(FallbackStrategy.RTN, BLOCKSIZE)

    assert torch.equal(result[0], control_result[0])
    assert torch.equal(result[1], control_result[1])
    assert torch.equal(result[2], control_result[2])
    assert not torch.equal(result[0], unweighted_result[0])
