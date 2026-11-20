# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5734: GPTQ/GPTAQ/FOEM must not keep a full rows x columns loss buffer.

The block pass only reads the per-block loss as one scalar for ``avg_loss``.
It used to zero-fill a W-sized fp32 ``Losses`` tensor (plus a per-block
``Losses1`` in GPTAQ/FOEM). The tests below pin the replacement contract:

* the only W-sized fp32 ``zeros_like`` allocation is the quantized output
  ``Q``, and
* ``avg_loss`` still equals the independently summed per-column loss.

The loss oracle uses a positive diagonal Hessian: with no off-diagonal
correction every column is quantized from its pre-pass weight, so each loss
term can be recomputed column by column.
"""

import copy

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ


ROWS, COLUMNS, BLOCK, GROUP = 16, 64, 16, 16
SAMPLES = 32
BATCH_REQUESTS = 8
CLASSES = {"gptq": GPTQ, "gptaq": GPTAQ, "foem": FOEM}


def _qcfg(kind):
    common = {"bits": 4, "group_size": GROUP, "sym": True, "desc_act": False, "act_group_aware": False}
    if kind == "gptaq":
        return QuantizeConfig(**common, gptaq=GPTAQConfig(alpha=0.25))
    if kind == "foem":
        # ``beta=0`` makes FOEM's in-block correction a no-op, so the diagonal
        # Hessian keeps every column at its pre-pass weight and the oracle
        # below applies unchanged.
        return QuantizeConfig(**common, foem=FOEMConfig(alpha=0.25, beta=0.0))
    return QuantizeConfig(**common)


def _task(kind):
    torch.manual_seed(5734)
    linear = nn.Linear(COLUMNS, ROWS, bias=False)
    named = NamedModule(linear, name="proj", full_name="model.layers.0.proj", layer_index=0)
    # GPTAQ/FOEM consume the native capture at construction; the loss oracle
    # below re-derives the same grids, so an empty capture is enough here.
    named.state[NATIVE_INPUTS_STATE_KEY] = []
    task = CLASSES[kind](named, _qcfg(kind))
    task.quantizer.configure(perchannel=True)
    task.H = torch.eye(COLUMNS, dtype=torch.float32) * COLUMNS
    if kind in ("gptaq", "foem"):
        task.dXXT = torch.zeros_like(task.H)
    task.nsamples = SAMPLES
    task._gptaq_batch_request_count = BATCH_REQUESTS
    return task


def _spy_on_w_shaped_zeros_like(monkeypatch):
    real_zeros_like = torch.zeros_like
    calls = []

    def spy(tensor, *args, **kwargs):
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.dim() == 2
            and tuple(tensor.shape) == (ROWS, COLUMNS)
        ):
            calls.append(tensor.dtype)
        return real_zeros_like(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "zeros_like", spy)
    return calls


def _capture_inverse_factor(task):
    """Record the inverse factor the block pass actually used."""

    captured = {}
    real_hessian_inverse = task.hessian_inverse

    def recording_hessian_inverse(hessian):
        factor, damp = real_hessian_inverse(hessian)
        captured["factor"] = factor.detach().clone()
        return factor, damp

    task.hessian_inverse = recording_hessian_inverse
    return captured


def _expected_loss_total(task, factor):
    """Sum the per-column GPTQ loss terms that ``avg_loss`` reports."""

    weight = task.module.weight.detach().float()
    quantizer = copy.deepcopy(task.quantizer)
    total = torch.zeros((), dtype=torch.float32)
    for start in range(0, COLUMNS, GROUP):
        stop = min(start + GROUP, COLUMNS)
        quantizer.find_params(weight[:, start:stop], weight=True)
        for column in range(start, stop):
            q = quantizer.quantize(weight[:, column].unsqueeze(1)).flatten()
            d = factor[column, column]
            total += ((weight[:, column] - q) ** 2 / d**2).sum() / 2
    return total


def _expected_avg_loss(kind, total):
    if kind == "gptaq":
        return total.item() * BATCH_REQUESTS / SAMPLES**2
    return total.item() / SAMPLES


@pytest.mark.parametrize("kind", ["gptq", "gptaq", "foem"])
def test_block_pass_keeps_no_full_loss_buffer(kind, monkeypatch):
    task = _task(kind)
    calls = _spy_on_w_shaped_zeros_like(monkeypatch)

    task.quantize(blocksize=BLOCK)

    # ``Q`` is the result and must stay W-sized; the loss must not.
    assert calls == [torch.float32]


@pytest.mark.parametrize("kind", ["gptq", "gptaq", "foem"])
def test_block_pass_reports_the_same_avg_loss(kind):
    task = _task(kind)
    captured = _capture_inverse_factor(task)

    _Q, _scale, _zero, _g_idx, _duration, avg_loss, _damp, _samples = task.quantize(
        blocksize=BLOCK
    )

    expected = _expected_avg_loss(kind, _expected_loss_total(task, captured["factor"]))
    assert isinstance(avg_loss, float)
    assert avg_loss == pytest.approx(expected, rel=1e-5, abs=1e-6)
