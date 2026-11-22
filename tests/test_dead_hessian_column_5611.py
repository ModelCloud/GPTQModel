# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Dead Hessian columns must not set the per-channel grid (#5611).

A channel absent from every calibration sample carries no signal.  GPTQ and
QQQ zero its weight, but they used to fit the whole-row per-channel scale
first, so the discarded column's (arbitrarily large) value set the grid that
every live column was quantized with.  Changing only that column must leave
the scales and the live quantized weights unchanged.
"""

from __future__ import annotations

import copy

import pytest
import torch

from gptqmodel.quantization.config import QQQConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ


def _quantize(quantizer_cls, dead_weight: float, *, desc_act: bool):
    layer = torch.nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.weight[:, 0] = dead_weight
    config = (
        QQQConfig(bits=4, group_size=-1, desc_act=desc_act, sym=True, fallback=None)
        if quantizer_cls is QQQ
        else QuantizeConfig(bits=4, group_size=-1, desc_act=desc_act, sym=True, fallback=None)
    )
    quantizer = quantizer_cls(layer, qcfg=config)
    if quantizer_cls is QQQ:
        quantizer.quantizer.configure(4, perchannel=True, sym=True, groupsize=-1)
    else:
        quantizer.quantizer.configure(perchannel=True)
    # Channel 0 is never activated.
    x = torch.eye(32)[1:].unsqueeze(0)
    quantizer.add_batch(x, layer(x))
    return quantizer.quantize()


@pytest.mark.parametrize("quantizer_cls", [GPTQ, QQQ], ids=["gptq", "qqq"])
@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("dead_weight", [0.0, 128.0, -128.0])
def test_dead_column_does_not_change_live_quantization(quantizer_cls, desc_act, dead_weight):
    if quantizer_cls is QQQ and desc_act:
        pytest.skip("QQQ does not support desc_act")
    reference = _quantize(quantizer_cls, 0.0, desc_act=desc_act)
    changed = _quantize(quantizer_cls, dead_weight, desc_act=desc_act)

    quantized, scale, zero = reference[0], reference[1], reference[2]
    other_quantized, other_scale, other_zero = changed[0], changed[1], changed[2]

    assert torch.equal(scale.float(), other_scale.float())
    assert torch.equal(zero.float(), other_zero.float())
    assert torch.equal(quantized.float(), other_quantized.float())
    # The discarded column itself is neutralized in both runs.
    assert torch.count_nonzero(quantized[:, 0]) == 0


# ---------------------------------------------------------------------------
# #5714: the subclasses that re-implement the block pass (GPTAQ, FOEM) kept the
# pre-#5611 order.  Same invariant, driven through their own `quantize()`.
# ---------------------------------------------------------------------------

def _named_layer(layer, *, routed_expert: bool = False):
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY

    named = NamedModule(
        copy.deepcopy(layer), name="proj", full_name="model.layers.0.proj", layer_index=0
    )
    # GPTAQ pops the captured native inputs at construction; this fixture feeds
    # the Hessian directly, so there is no native delta to consume.
    named.state[NATIVE_INPUTS_STATE_KEY] = []
    if routed_expert:
        # Routed experts take the plain two-tensor neutralization branch.
        named.state["is_routed_expert"] = True
    return named


def _calibration(dead_weight: float):
    """A layer whose column 0 is never activated, plus that calibration batch."""

    layer = torch.nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.weight[:, 0] = dead_weight
    x = torch.eye(32)[1:]
    return layer, x


def _quantize_subclass(kind: str, dead_weight: float, *, desc_act: bool, routed_expert: bool = False):
    """Run GPTAQ/FOEM's block pass on a deterministic single-batch Hessian."""

    from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig
    from gptqmodel.quantization.foem import FOEM
    from gptqmodel.quantization.gptaq import GPTAQ

    layer, x = _calibration(dead_weight)
    if kind == "gptaq":
        qcfg = QuantizeConfig(
            bits=4, group_size=-1, desc_act=desc_act, sym=True, fallback=None,
            gptaq=GPTAQConfig(alpha=0.0, device="cpu"),
        )
        quantizer = GPTAQ(_named_layer(layer, routed_expert=routed_expert), qcfg=qcfg)
    else:
        qcfg = QuantizeConfig(
            bits=4, group_size=-1, desc_act=desc_act, sym=True, fallback=None,
            foem=FOEMConfig(alpha=0.0, beta=0.0, device="cpu"),
        )
        quantizer = FOEM(_named_layer(layer, routed_expert=routed_expert), qcfg=qcfg)
    quantizer.quantizer.configure(perchannel=True)
    # The same accumulation GPTQ.add_batch performs for one batch of `x`.
    quantizer.H = 2.0 / x.shape[0] * (x.T @ x)
    quantizer.nsamples = x.shape[0]
    if kind == "gptaq":
        quantizer.dXXT = torch.zeros_like(quantizer.H)
        quantizer._gptaq_batch_request_count = 1
    return quantizer.quantize()


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("dead_weight", [128.0, -128.0])
def test_subclass_grid_ignores_the_dead_column_weight(kind, desc_act, dead_weight):
    """Red on main: the fit ran before neutralization, so 2*1.0/15 set the grid."""

    reference = _quantize_subclass(kind, 0.0, desc_act=desc_act)
    changed = _quantize_subclass(kind, dead_weight, desc_act=desc_act)

    quantized, scale, zero = reference[0], reference[1], reference[2]
    other_quantized, other_scale, other_zero = changed[0], changed[1], changed[2]

    assert torch.equal(scale.float(), other_scale.float())
    assert torch.equal(zero.float(), other_zero.float())
    assert torch.equal(quantized.float(), other_quantized.float())
    # The fixture really has a dead column, and it is neutralized in both runs.
    assert torch.count_nonzero(quantized[:, 0]) == 0


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_subclass_reduced_config_matches_gptq_grid(kind):
    """With alpha=beta=0 the subclass must quantize on GPTQ's grid."""

    gptq = _quantize(GPTQ, 128.0, desc_act=False)
    subclass = _quantize_subclass(kind, 128.0, desc_act=False)

    torch.testing.assert_close(
        subclass[1].float(), gptq[1].float(), rtol=1e-6, atol=0.0
    )
    torch.testing.assert_close(
        subclass[2].float(), gptq[2].float(), rtol=1e-6, atol=0.0
    )


def test_routed_expert_grid_ignores_the_dead_column_weight():
    """The plain two-tensor neutralization branch has the same ordering."""

    reference = _quantize_subclass("gptaq", 0.0, desc_act=False, routed_expert=True)
    changed = _quantize_subclass("gptaq", 128.0, desc_act=False, routed_expert=True)

    assert torch.equal(reference[1].float(), changed[1].float())
    assert torch.equal(reference[0].float(), changed[0].float())
    assert torch.count_nonzero(changed[0][:, 0]) == 0
