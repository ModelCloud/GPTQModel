# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTAQ and FOEM must honour the low-sample fallback threshold (#5377).

GPTQ decides ``Fallback.threshold`` before solving; GPTAQ and FOEM only checked
for a zero sample count or a missing Hessian, so a sparsely routed expert with a
few observations still ran the full Hessian solver.
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
)
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ


ROWS, COLUMNS, GROUP_SIZE = 8, 16, 8
SAMPLES = 4
BLOCKSIZE = 8
ALPHA = 0.25


def _weight() -> torch.Tensor:
    generator = torch.Generator().manual_seed(5377)
    return torch.randn(ROWS, COLUMNS, generator=generator) * 0.05


def _hessian() -> torch.Tensor:
    generator = torch.Generator().manual_seed(5378)
    skew = torch.randn(COLUMNS, COLUMNS, generator=generator)
    skew = (skew - skew.T) / 2
    diagonal = torch.linspace(1.0, 2.0, COLUMNS)[
        torch.randperm(COLUMNS, generator=generator)
    ]
    return torch.diag(diagonal) + 1e-3 * skew


def _dXXT() -> torch.Tensor:
    generator = torch.Generator().manual_seed(5379)
    return torch.randn(COLUMNS, COLUMNS, generator=generator) * 0.01


def _task(kind: str, *, threshold, samples: int = SAMPLES, expected=None):
    layer = torch.nn.Linear(COLUMNS, ROWS, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(_weight())

    named = NamedModule(layer, name="proj", full_name="model.layers.0.proj", layer_index=0)
    named.state[NATIVE_INPUTS_STATE_KEY] = []

    common = {
        "bits": 4,
        "group_size": GROUP_SIZE,
        "desc_act": False,
        "fallback": Fallback(strategy=FallbackStrategy.RTN, threshold=threshold),
    }
    if kind == "gptq":
        qcfg = QuantizeConfig(**common)
        cls = GPTQ
    elif kind == "gptaq":
        qcfg = QuantizeConfig(gptaq=GPTAQConfig(alpha=ALPHA), **common)
        cls = GPTAQ
    else:
        qcfg = QuantizeConfig(foem=FOEMConfig(alpha=ALPHA), **common)
        cls = FOEM

    task = cls(named, qcfg)
    task.quantizer.configure(perchannel=True)
    task.H = _hessian()
    task.dXXT = _dXXT()
    task.nsamples = samples
    task.expected_nsamples = expected
    if kind == "gptaq":
        task._gptaq_batch_request_count = samples
    return task


def _quantized(kind: str, **kwargs) -> torch.Tensor:
    return _task(kind, **kwargs).quantize(blocksize=BLOCKSIZE)[0].detach().clone()


@pytest.mark.parametrize("kind", ["gptq", "gptaq", "foem"])
def test_low_sample_threshold_changes_the_quantized_weight(kind):
    normal = _quantized(kind, threshold=0)
    fallback = _quantized(kind, threshold=100)

    assert not torch.equal(normal, fallback)


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_above_the_threshold_the_solver_still_runs(kind):
    no_threshold = _quantized(kind, threshold=0, samples=200)
    above_threshold = _quantized(kind, threshold=100, samples=200)

    assert torch.equal(no_threshold, above_threshold)


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_fractional_threshold_uses_the_expected_sample_count(kind):
    # 50% of 100 expected samples = 50: 40 observed samples falls back, 60 does not.
    below = _quantized(kind, threshold="50%", samples=40, expected=100)
    above = _quantized(kind, threshold="50%", samples=60, expected=100)

    assert not torch.equal(below, above)
