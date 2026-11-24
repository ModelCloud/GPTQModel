# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""GPTAQ/FOEM cohorts share one inverse and one delta product.

Members of an explicit ``:in=<tag>`` group receive bitwise-identical inputs, so
they adopt the same ``H`` and ``dXXT``. Before this, only exact ``GPTQ`` tasks
entered ``SharedHessianFactorCache``: the subclasses that re-implement the block
pass recomputed the Cholesky and the ``[columns, columns]`` delta product once
per sibling. The cache now carries their tuple under its own scope, so a q/k/v
cohort pays one inverse instead of three.
"""

import copy
import threading

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import SharedHessianFactorCache


def _qcfg(kind):
    if kind == "gptaq":
        return QuantizeConfig(bits=4, group_size=32, gptaq=GPTAQConfig(alpha=0.25, device="cpu"))
    return QuantizeConfig(bits=4, group_size=32, foem=FOEMConfig(alpha=0.25, beta=0.0, device="cpu"))


def _cls(kind):
    return GPTAQ if kind == "gptaq" else FOEM


def _named(layer):
    named = NamedModule(layer, name="proj", full_name="model.layers.0.proj", layer_index=0)
    # One native entry per captured batch; ``_take_native_input`` pairs it with
    # the request. This keeps the delta product alive so the cached tuple is
    # the full (inverse, damp, delta) artifact.
    named.state[NATIVE_INPUTS_STATE_KEY] = []
    return named


def _cohort(kind, *, native_delta: bool):
    torch.manual_seed(0)
    layer = nn.Linear(32, 32, bias=False)
    named = _named(copy.deepcopy(layer))
    named_follower = _named(copy.deepcopy(layer))
    x = torch.randn(2, 8, 32)
    native = x + 0.1
    if native_delta:
        named.state[NATIVE_INPUTS_STATE_KEY] = [native]
        named_follower.state[NATIVE_INPUTS_STATE_KEY] = [native]
    else:
        named.state["is_routed_expert"] = True
        named_follower.state["is_routed_expert"] = True

    leader = _cls(kind)(named, _qcfg(kind))
    follower = _cls(kind)(named_follower, _qcfg(kind))
    for task in (leader, follower):
        task._hessian_capture_enabled = True
        task.quantizer.configure(perchannel=True)
    leader.add_batch(x, None)
    follower.adopt_hessian_from(leader)
    return leader, follower


def _count_cholesky(monkeypatch):
    original = torch.linalg.cholesky
    original_ex = torch.linalg.cholesky_ex
    calls = {"plain": 0, "ex": 0}

    def counted(*args, **kwargs):
        calls["plain"] += 1
        return original(*args, **kwargs)

    def counted_ex(*args, **kwargs):
        calls["ex"] += 1
        return original_ex(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", counted)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", counted_ex)
    return calls


def _assert_same_result(left, right):
    for a, b in zip(left[:4], right[:4]):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            assert a == b


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
@pytest.mark.parametrize("native_delta", [True, False])
def test_a_shared_cohort_pays_one_inverse(kind, native_delta, monkeypatch):
    leader, follower = _cohort(kind, native_delta=native_delta)
    cache = SharedHessianFactorCache(generation=7, cohort="qkv")
    leader.attach_shared_hessian_cache(cache)
    follower.attach_shared_hessian_cache(cache)

    calls = _count_cholesky(monkeypatch)
    leader_result = leader.quantize()
    follower_result = follower.quantize()

    assert calls["ex"] == 1, "the cohort must compute one Cholesky factor"
    _assert_same_result(leader_result, follower_result)


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
@pytest.mark.parametrize("native_delta", [True, False])
def test_without_a_cache_each_task_computes_its_own(kind, native_delta, monkeypatch):
    leader, follower = _cohort(kind, native_delta=native_delta)

    calls = _count_cholesky(monkeypatch)
    leader.quantize()
    follower.quantize()

    assert calls["ex"] == 2


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_cached_and_uncached_results_are_identical(kind, monkeypatch):
    cached_leader, cached_follower = _cohort(kind, native_delta=True)
    cache = SharedHessianFactorCache(generation=9, cohort="qkv")
    cached_leader.attach_shared_hessian_cache(cache)
    cached_follower.attach_shared_hessian_cache(cache)

    uncached_leader, uncached_follower = _cohort(kind, native_delta=True)

    cached = [cached_leader.quantize(), cached_follower.quantize()]
    uncached = [uncached_leader.quantize(), uncached_follower.quantize()]

    for cached_result, uncached_result in zip(cached, uncached):
        _assert_same_result(cached_result, uncached_result)


def _processor_with(leader, follower, leader_name="attn.q_proj", follower_name="attn.k_proj"):
    """A processor shaped just enough for ``end_shared_input_capture``."""

    from types import SimpleNamespace

    from gptqmodel.looper.gptq_processor import GPTQProcessor

    processor = object.__new__(GPTQProcessor)
    processor.lock = threading.RLock()
    processor.tasks = {leader_name: leader, follower_name: follower}
    processor._shared_input_leaders = {follower_name: leader_name}
    processor._shared_input_capture_generation = 1
    processor.shared_input_dedup_count = 0
    processor.shared_input_dedup_telemetry = {}
    processor.qcfg = SimpleNamespace(
        hessian=SimpleNamespace(dedup_shared_inputs=True),
    )
    return processor, leader_name, follower_name


def test_the_processor_publishes_one_cache_for_solver_siblings():
    """GPTQ used to be the only type that entered the cohort cache."""

    leader, follower = _cohort("gptaq", native_delta=True)
    processor, _, _ = _processor_with(leader, follower)

    processor.end_shared_input_capture(["proj"])

    assert leader._shared_hessian_cache is not None
    assert leader._shared_hessian_cache is follower._shared_hessian_cache


def test_mixing_solver_types_is_rejected_before_caching():
    """Adoption already refuses a cross-solver cohort; the cache never sees it."""

    leader, _ = _cohort("gptaq", native_delta=True)
    follower, _ = _cohort("foem", native_delta=True)
    processor, _, _ = _processor_with(leader, follower)

    with pytest.raises(TypeError, match="cannot share Hessian"):
        processor.end_shared_input_capture(["proj"])

    assert leader._shared_hessian_cache is None
    assert follower._shared_hessian_cache is None
