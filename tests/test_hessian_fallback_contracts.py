# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Hessian-failure and fallback contracts shared by GPTQ, GPTAQ, FOEM and RTN.

These pin issue #4179: a non-invertible Hessian must degrade to the one shared
RTN fallback, a zero-sample module must never ship an all-zero weight, the
NaN-loss retry must keep activation ordering, GPTAQ/FOEM must honour
``act_group_aware`` and TP padding, and RTN scales must stay inside the FP16
storage range.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.qlinear import _scales_for_storage
from gptqmodel.quantization.config import (
    Fallback,
    FOEMConfig,
    GPTAQConfig,
    QuantizeConfig,
    RTNConfig,
    SmoothMSE,
    SmoothRowCol,
)
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.quantizer import SCALE_STORAGE_FLOOR
from gptqmodel.quantization.rtn import RTN


def _named(module, *, pad_cols=0, routed=False):
    named = NamedModule(module, name="proj", full_name="model.layers.0.proj", layer_index=0)
    if pad_cols:
        named.state["tp_pad_info"] = {"pad_cols": pad_cols}
    if routed:
        named.state["is_routed_expert"] = True
    # GPTAQ always pops this key, even when the routed-expert mark disables
    # the native delta.
    named.state["native_inp"] = []
    return named


def _ordering_config(**kwargs):
    kwargs.setdefault("fallback", None)
    return QuantizeConfig(bits=4, group_size=4, sym=True, **kwargs)


def _ordering_task(cls, linear, qcfg, *, diag, routed=True):
    named = _named(linear, routed=routed)
    task = cls(named, qcfg) if cls is not GPTQ else GPTQ(linear, qcfg)
    task.quantizer.configure(perchannel=True)
    task.H = torch.diag(diag)
    task.nsamples = int(diag.numel())
    return task


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_non_invertible_hessian_uses_the_shared_rtn_fallback(monkeypatch, kind):
    """``hessian_inverse -> (None, 1.0)`` must not abort GPTAQ or FOEM.

    Before the fix GPTAQ raised ``AttributeError: 'NoneType' object has no
    attribute 'T'`` and FOEM raised ``TypeError: 'NoneType' object is not
    subscriptable``. Both now return the same data-independent RTN grid GPTQ
    uses.
    """

    torch.manual_seed(0)
    linear = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.linspace(-1.0, 1.0, 32).reshape(4, 8))

    if kind == "gptaq":
        qcfg = _ordering_config(gptaq=GPTAQConfig())
        task = GPTAQ(_named(linear), qcfg)
        inverse_cls = GPTAQ
    else:
        qcfg = _ordering_config(foem=FOEMConfig(alpha=0, beta=0.2))
        task = FOEM(_named(linear), qcfg)
        inverse_cls = FOEM

    task.quantizer.configure(perchannel=True)
    task.H = torch.eye(8)
    task.nsamples = 8
    if kind == "gptaq":
        # Keep the native-delta branch active so the parent commit fails at the
        # `Hinv.T` dereference the issue reports.
        task.dXXT = torch.zeros(8, 8)
    monkeypatch.setattr(inverse_cls, "hessian_inverse", lambda self, hessian: (None, 1.0))

    quantized, _scale, _zero, _g_idx, _duration, avg_loss, _damp, nsamples = task.quantize(blocksize=4)

    assert nsamples == 8
    assert quantized.shape == linear.weight.shape
    assert torch.isfinite(quantized).all()
    assert quantized.abs().max().item() > 0
    assert str(avg_loss).startswith("fallback(rtn)")


@pytest.mark.parametrize(
    "fallback",
    [None, Fallback(threshold=False)],
    ids=["fallback_none", "threshold_disabled"],
)
def test_zero_sample_gptq_module_is_not_quantized_to_all_zeros(fallback):
    """An unrouted module with ``nsamples == 0`` must not ship a zero weight.

    With fallback disabled the zero Hessian made every column look dead, so
    ``_neutralize_dead_columns`` zeroed the whole weight and the only signal was
    ``loss 999999999``. The zero-sample guard forces the shared RTN fallback.
    """

    linear = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.arange(1, 33, dtype=torch.float32).reshape(4, 8))

    qcfg = QuantizeConfig(bits=4, group_size=4, fallback=fallback)
    task = GPTQ(linear, qcfg)
    task.quantizer.configure(perchannel=True)
    task.H = None
    task.nsamples = 0

    quantized, _scale, _zero, _g_idx, _duration, avg_loss, _damp, nsamples = task.quantize(blocksize=4)

    assert nsamples == 0
    assert quantized.abs().max().item() > 0
    assert str(avg_loss).startswith("fallback(rtn)")


def test_nan_loss_mock_retry_preserves_desc_act_order(monkeypatch):
    """The retry must reproduce the primary column order, not natural order.

    The retry state is now captured before H is permuted. The reference is a
    direct mock-quantization run over the same natural-order diagonal, which is
    exactly what the retry is supposed to reproduce.
    """

    torch.manual_seed(1)
    weight = torch.randn(4, 8)
    diag = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])

    def make_task(*, mock):
        linear = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            linear.weight.copy_(weight)
        qcfg = _ordering_config(
            desc_act=True,
            act_group_aware=False,
            mock_quantization=mock,
            fallback=None if mock else Fallback(threshold=True),
        )
        return _ordering_task(GPTQ, linear, qcfg, diag=diag), linear

    reference_task, _ = make_task(mock=True)
    reference_q, _s, _z, reference_g, *_ = reference_task.quantize(blocksize=4)

    retry_task, _ = make_task(mock=False)
    original = GPTQ.hessian_inverse
    calls = {"count": 0}

    def poisoned_inverse(self, hessian):
        factor, damp = original(self, hessian)
        calls["count"] += 1
        if calls["count"] == 1:
            factor = factor.clone()
            factor[0, 0] = float("nan")
        return factor, damp

    monkeypatch.setattr(GPTQ, "hessian_inverse", poisoned_inverse)

    retry_q, _s, _z, retry_g, *_ = retry_task.quantize(blocksize=4)

    assert calls["count"] >= 1
    assert retry_task.qcfg.mock_quantization is True
    torch.testing.assert_close(retry_q, reference_q, rtol=0, atol=0)
    assert torch.equal(retry_g, reference_g)


def test_nan_loss_mock_retry_reuses_the_primary_act_group_permutation(monkeypatch):
    """The act-group retry must order columns with the primary permutation.

    A mock retry uses an identity inverse, so the quantized values themselves
    are invariant to column order. The observable contract is therefore the
    permutation: the primary and retry must compute the same one from the same
    natural-order diagonal.
    """

    import gptqmodel.quantization.gptq as gptq_module

    torch.manual_seed(1)
    weight = torch.randn(4, 8)
    diag = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    recorded = []
    original_compose = gptq_module.compose_final_perm

    def recording_compose(local_perms, global_perm, groupsize):
        perm = original_compose(local_perms, global_perm, groupsize)
        recorded.append(perm.detach().clone())
        return perm

    monkeypatch.setattr(gptq_module, "compose_final_perm", recording_compose)

    linear = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        linear.weight.copy_(weight)
    qcfg = _ordering_config(
        desc_act=False,
        act_group_aware=True,
        fallback=Fallback(threshold=True),
    )
    task = _ordering_task(GPTQ, linear, qcfg, diag=diag)

    original_inverse = GPTQ.hessian_inverse
    calls = {"count": 0}

    def poisoned_inverse(self, hessian):
        factor, damp = original_inverse(self, hessian)
        calls["count"] += 1
        if calls["count"] == 1:
            factor = factor.clone()
            factor[0, 0] = float("nan")
        return factor, damp

    monkeypatch.setattr(GPTQ, "hessian_inverse", poisoned_inverse)

    task.quantize(blocksize=4)

    assert task.qcfg.mock_quantization is True
    assert len(recorded) == 2
    assert not torch.equal(recorded[0], torch.arange(8))
    assert torch.equal(recorded[0], recorded[1])


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_act_group_aware_matches_gptq_for_gptaq_and_foem(kind):
    """GPTAQ/FOEM must apply the same activation-group permutation as GPTQ."""

    torch.manual_seed(2)
    weight = torch.randn(4, 8)
    diag = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    # Off-diagonal coupling makes the within-group quantization order matter;
    # a diagonal Hessian alone is invariant to act_group_aware in the
    # data-independent (identity inverse) limit.
    hessian = torch.diag(diag) + 0.2 * torch.ones(8, 8)

    def make_task(cls, qcfg):
        linear = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            linear.weight.copy_(weight)
        task = _ordering_task(cls, linear, qcfg, diag=diag)
        task.H = hessian.clone()
        # GPTAQ/FOEM reduce to plain GPTQ with alpha/beta = 0.
        return task

    base = {"desc_act": False, "act_group_aware": True}
    reference_q, _s, _z, reference_g, *_ = make_task(
        GPTQ, _ordering_config(**base)
    ).quantize(blocksize=4)
    unordered_q, _s, _z, _g, *_ = make_task(
        GPTQ, _ordering_config(desc_act=False, act_group_aware=False)
    ).quantize(blocksize=4)
    assert not torch.allclose(reference_q, unordered_q, rtol=1e-4, atol=1e-5)

    if kind == "gptaq":
        qcfg = _ordering_config(gptaq=GPTAQConfig(), **base)
        task = make_task(GPTAQ, qcfg)
    else:
        qcfg = _ordering_config(foem=FOEMConfig(alpha=0, beta=0), **base)
        task = make_task(FOEM, qcfg)

    quantized, _s, _z, g_idx, *_ = task.quantize(blocksize=4)

    torch.testing.assert_close(quantized, reference_q, rtol=1e-4, atol=1e-5)
    assert torch.equal(g_idx, reference_g)


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_tp_padded_columns_are_truncated_for_gptaq_and_foem(kind):
    """GPTAQ/FOEM must drop tensor-parallel padding before reshaping to W."""

    torch.manual_seed(3)
    linear = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(4, 8))
    named = _named(linear, pad_cols=2, routed=True)
    diag = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0])

    if kind == "gptaq":
        qcfg = QuantizeConfig(
            bits=4, group_size=-1, sym=True, act_group_aware=False,
            fallback=None, gptaq=GPTAQConfig(),
        )
        task = GPTAQ(named, qcfg)
    else:
        qcfg = QuantizeConfig(
            bits=4, group_size=-1, sym=True, act_group_aware=False,
            fallback=None, foem=FOEMConfig(alpha=0, beta=0),
        )
        task = FOEM(named, qcfg)

    task.quantizer.configure(perchannel=True)
    task.H = torch.diag(diag)
    task.nsamples = int(diag.numel())

    quantized, scale, zero, g_idx, *_ = task.quantize(blocksize=4)

    assert quantized.shape == linear.weight.shape
    assert g_idx.numel() == linear.weight.shape[1]
    assert scale.shape[-1] == 1
    assert zero.shape[-1] == 1


def test_foem_nsamples_counts_tokens_not_sequences():
    """FOEM's loss denominator must match GPTQ's token count."""

    torch.manual_seed(4)
    linear = nn.Linear(8, 4, bias=False)
    qcfg = QuantizeConfig(bits=4, group_size=4, foem=FOEMConfig(alpha=0, beta=0.2))
    task = FOEM(_named(linear), qcfg)

    task.add_batch(torch.randn(2, 3, 8), None)

    assert task.nsamples == 6


def test_rtn_smooth_mse_uses_the_fp16_scale_storage_floor():
    """SmoothMSE must bound candidate scales before FP16 storage."""

    linear = nn.Linear(128, 8, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1e-7)

    qcfg = RTNConfig(
        bits=4, group_size=128, sym=True, smooth=SmoothMSE(), offload_to_disk=False,
    )
    _quantized, scale, _zero, _g_idx, *_ = RTN(linear, qcfg).quantize()

    assert scale.min().item() >= SCALE_STORAGE_FLOOR
    _scales_for_storage(scale, name="rtn.smooth_mse")


def test_rtn_smooth_rowcol_bounds_the_restored_scale():
    """SmoothRowCol must divide the storage floor by its restoring factor."""

    linear = nn.Linear(128, 8, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1e-8)

    qcfg = RTNConfig(
        bits=4,
        group_size=128,
        sym=True,
        smooth=SmoothRowCol(axis="row"),
        offload_to_disk=False,
    )
    _quantized, scale, _zero, _g_idx, *_ = RTN(linear, qcfg).quantize()

    assert scale.min().item() >= SCALE_STORAGE_FLOOR
    _scales_for_storage(scale, name="rtn.smooth_rowcol")
