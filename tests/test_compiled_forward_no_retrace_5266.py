# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``_CompiledModelForward`` must not put dynamo's frame counter in a guard.

#5240 taught ``_release_trace_garbage`` to skip ``gc.collect`` while an outer
``torch.compile`` traces the call, but it read ``counters["frames"]["total"]``
before that check. Dynamo traced the read and installed a guard on the global
counter; every later compile bumped the counter, so the guard failed on the
next call and every static-cache ``generate()`` after ``optimize()`` re-traced
forever (2.7 s vs 0.04 s per call, #5266).

These tests pin the check order directly, without a GPU or a real model:

* the counter is never read while a trace is active, and
* an unrelated compile (which bumps the counter) does not invalidate a
  previously compiled call that runs through the proxy.
"""

import pytest
import torch

from gptqmodel.models import base as base_module
from gptqmodel.models.base import _CompiledModelForward


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4))

    def forward(self, x):
        return x @ self.weight


def _proxy(module):
    def compiled(model, x):
        return model(x)

    return _CompiledModelForward(compiled, _Tiny.forward, module)


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _frames() -> int:
    return int(torch._dynamo.utils.counters["frames"].get("total", 0))


def test_counter_is_not_read_while_tracing(monkeypatch):
    """A traced call must not even look at dynamo's global frame counter."""

    module = _Tiny()
    proxy = _proxy(module)
    reads = []
    real = base_module.dynamo_traced_frames

    def spy():
        reads.append(True)
        return real()

    monkeypatch.setattr(base_module, "dynamo_traced_frames", spy)

    def body(x):
        out = proxy(x)
        return out.sum()

    compiled = torch.compile(body, fullgraph=True, backend="eager")
    compiled(torch.ones(4, 4))

    assert reads == [], (
        "the proxy read dynamo's frame counter while an outer trace was active; "
        "dynamo will guard that read and re-trace on every later compile"
    )


def test_counter_mutation_does_not_invalidate_the_proxy_call():
    """Mutating dynamo's frame counter must not re-trace a cached proxy call.

    The guard dynamo installs on a traced read is on the ``counters["frames"]``
    mapping itself, so any later compile that adds or removes an entry in that
    dict invalidates the cached graph. Reproduce that by touching the mapping
    directly and assert the compiled call is still served from its cache.
    """

    module = _Tiny()
    proxy = _proxy(module)

    def body(x):
        return proxy(x).sum()

    compiled = torch.compile(body, fullgraph=True, backend="eager")
    compiled(torch.ones(4, 4))
    compiled(torch.ones(4, 4))

    graphs_before = int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0))
    # A later compile in the process bumps the global frame counter. Mutating
    # the mapping reproduces the guard's trigger without a second model.
    frames = torch._dynamo.utils.counters["frames"]
    frames["total"] = int(frames.get("total", 0)) + 1

    compiled(torch.ones(4, 4))

    graphs_after = int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0))
    assert graphs_after == graphs_before, (
        "touching dynamo's frame counter invalidated the cached proxy call: the "
        "counter is in its guard set (#5266)"
    )
