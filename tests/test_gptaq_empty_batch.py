# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""An empty calibration batch must not touch GPTAQ's running Hessian.

A routed expert can receive only padding rows, so the calibration loop still
hands the quantizer a zero-token batch. FOEM already returns early for that case
(#5768); GPTAQ did not, and its request-count running average decayed the
Hessian by ``count / (count + 1)`` while adding nothing - the partial no longer
corresponded to any sample count, and the loss depended on how many empty
batches the router happened to produce.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ


def _solver(kind):
    module = nn.Linear(8, 4, bias=False)
    if kind == "gptaq":
        qcfg = QuantizeConfig(bits=4, group_size=8, gptaq=GPTAQConfig())
        cls = GPTAQ
    else:
        qcfg = QuantizeConfig(bits=4, group_size=8, foem=FOEMConfig(alpha=0.25))
        cls = FOEM

    named = NamedModule(module, name="proj", full_name="model.layers.0.proj", layer_index=0)
    # A routed expert replays no native input, which is also the branch where a
    # zero-token batch is possible.
    named.state["native_inp"] = []
    named.state["is_routed_expert"] = True
    solver = cls(named, qcfg)
    solver._hessian_capture_enabled = True
    return solver


def _run(kind, batches):
    torch.manual_seed(0)
    solver = _solver(kind)
    for batch in batches:
        solver.add_batch(batch, None)
    return solver


EMPTY = torch.zeros(1, 0, 8)


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_an_empty_batch_leaves_the_hessian_unchanged(kind):
    batch = torch.randn(1, 4, 8)

    with_empty = _run(kind, [batch, EMPTY])
    without = _run(kind, [batch])

    torch.testing.assert_close(with_empty.H, without.H, rtol=0, atol=0)
    assert with_empty.nsamples == without.nsamples == 4


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_an_empty_batch_between_two_real_batches_is_a_no_op(kind):
    first = torch.randn(1, 4, 8)
    second = torch.randn(1, 4, 8)

    with_empty = _run(kind, [first, EMPTY, second])
    without = _run(kind, [first, second])

    torch.testing.assert_close(with_empty.H, without.H, rtol=0, atol=0)


def test_gptaq_does_not_book_a_request_for_an_empty_batch():
    batch = torch.randn(1, 4, 8)

    with_empty = _run("gptaq", [batch, EMPTY])
    without = _run("gptaq", [batch])

    assert with_empty._gptaq_batch_request_count == 1
    assert with_empty._gptaq_batch_request_count == without._gptaq_batch_request_count
