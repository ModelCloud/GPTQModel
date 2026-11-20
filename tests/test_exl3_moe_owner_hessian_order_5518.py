# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Owner-device Hessian accumulation must not follow thread arrival order (#5518).

Since #5093 every routed expert's Hessian has an owner device. ``GPTQ`` moves
each replica's batch to that owner, but the partial must stay keyed by the
device the activations were captured on: keying it under the owner made two
data-parallel replicas append ``X^T X`` into one tensor in whatever order their
worker threads finished, so the fp32 sum -- and EXL3's discrete trellis search
-- changed from run to run.

The tests drive the real ``process_batch``/``add_batch`` path on the host.
``get_device`` is redirected so a CPU capture can present two replica devices
while the owner stays the CPU: that is exactly the partial keying the two-GPU
data-parallel run performs.
"""

import torch
from torch import nn

from gptqmodel.quantization import gptq as gptq_mod
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


_COLUMNS = 4
# Four single-row batches with a ~1e12 spread in their outer products. The
# spread makes the fp32 sum of the per-batch ``X^T X`` depend on how the four
# terms are grouped, which is what thread arrival order changed.
_BATCHES = (
    (1540996.125, -0.293428897857666, -2.1787893729197094e-06, 568.4312744140625),
    (-1084522.375, -1.3985954523086548, 4.033468314901256e-07, 838.0263671875),
    (-719257.5625, -0.40334352850914, -5.966353455733042e-07, 182.03648376464844),
    (-856674.625, 1.1006041765213013, -1.0711873983382247e-06, 122.70124053955078),
)
# Batches 0 and 2 are captured on replica 0, batches 1 and 3 on replica 1.
_SOURCES = (
    torch.device("cuda:0"),
    torch.device("cuda:1"),
    torch.device("cuda:0"),
    torch.device("cuda:1"),
)


def _use_cpu_capture_sources(monkeypatch):
    """Let a CPU tensor carry the replica device its activations came from."""

    original = gptq_mod.get_device

    def _source_aware(obj):
        tagged = getattr(obj, "_capture_source", None)
        return tagged if tagged is not None else original(obj)

    monkeypatch.setattr(gptq_mod, "get_device", _source_aware)


def _task(*, owner: bool) -> GPTQ:
    task = GPTQ(
        nn.Linear(_COLUMNS, 1, bias=False),
        QuantizeConfig(bits=4, group_size=-1, fallback=None, act_group_aware=False),
    )
    task.quantizer.configure(perchannel=True)
    if owner:
        task.set_hessian_accumulator_device(torch.device("cpu"))
    return task


def _add(task: GPTQ, index: int) -> None:
    batch = torch.tensor([_BATCHES[index]], dtype=torch.float32)
    batch._capture_source = _SOURCES[index]
    task.add_batch(batch, None)


def _finalize(task: GPTQ) -> torch.Tensor:
    # Request the CPU explicitly: the fake replica devices only identify the
    # partials, they are not real accelerators to finalize on.
    return task.finalize_hessian(torch.device("cpu"))


def test_process_batch_keys_the_capture_source(monkeypatch):
    _use_cpu_capture_sources(monkeypatch)
    task = _task(owner=True)

    batch = torch.tensor([_BATCHES[0]], dtype=torch.float32)
    batch._capture_source = torch.device("cuda:1")
    token_count, xtx, key = task.process_batch(batch)

    assert token_count == 1
    # The owner computed the partial...
    assert xtx.device == torch.device("cpu")
    # ...but the key still names the replica the batch came from.
    assert key == torch.device("cuda:1")


def test_owner_partials_are_keyed_by_capture_source(monkeypatch):
    _use_cpu_capture_sources(monkeypatch)
    task = _task(owner=True)

    _add(task, 0)
    _add(task, 1)

    assert set(task._device_hessian_partials) == {
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    }
    for partial in task._device_hessian_partials.values():
        assert partial.device == torch.device("cpu")


def test_owner_accumulation_is_arrival_order_independent(monkeypatch):
    _use_cpu_capture_sources(monkeypatch)

    first = _task(owner=True)
    for index in (0, 1, 2, 3):
        _add(first, index)
    hessian_a = _finalize(first)

    second = _task(owner=True)
    for index in (0, 2, 1, 3):
        _add(second, index)
    hessian_b = _finalize(second)

    assert torch.equal(hessian_a, hessian_b), (
        "owner-device Hessian depends on worker-thread arrival order"
    )


def test_owner_accumulation_matches_the_plain_reduction(monkeypatch):
    _use_cpu_capture_sources(monkeypatch)

    owned = _task(owner=True)
    for index in (1, 0, 3, 2):
        _add(owned, index)

    plain = _task(owner=False)
    for index in (2, 3, 0, 1):
        _add(plain, index)

    assert torch.equal(_finalize(owned), _finalize(plain))


def test_fixture_exposes_the_fp32_grouping_difference():
    """Guard the data: without grouping sensitivity the regression is vacuous."""

    matrices = []
    for row in _BATCHES:
        batch = torch.tensor([row], dtype=torch.float32)
        matrices.append(batch.T @ batch)

    sequential = ((matrices[0] + matrices[1]) + matrices[2]) + matrices[3]
    interleaved = ((matrices[0] + matrices[2]) + matrices[1]) + matrices[3]

    assert not torch.equal(sequential, interleaved)
