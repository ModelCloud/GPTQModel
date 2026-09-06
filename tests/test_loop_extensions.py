# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, get_ident

import pytest

from gptqmodel.looper.extension import LoopExtensions, LoopStep

STEP = LoopStep("layer", 0, "model.layers.0")


def test_disabled_extensions_do_not_wait():
    pending = Future()
    LoopExtensions().publish(STEP, [pending])
    assert not pending.done()


def test_boundary_waits_for_earlier_steps_too():
    first, second = Future(), Future()
    observed = []

    class Extension:
        def on_boundary(self, boundary):
            if boundary.step.index == 1:
                observed.extend(boundary.quiesce())

    extensions = LoopExtensions([Extension()])
    extensions.publish(STEP, [first])
    first.set_result("first")
    second.set_result("second")
    extensions.publish(LoopStep("layer", 1, "model.layers.1"), [second])
    assert observed == ["first", "second"]


def test_failure_waits_for_all_workers_before_raising():
    started, release = Event(), Event()
    failed = Future()
    failed.set_exception(ValueError("packing failed"))

    def worker():
        started.set()
        assert release.wait(5)
        return "done"

    class Extension:
        def on_boundary(self, boundary):
            release.set()
            with pytest.raises(ValueError, match="packing failed"):
                boundary.quiesce()
            assert pending.done()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(worker)
        assert started.wait(5)
        LoopExtensions([Extension()]).publish(STEP, [failed, pending])


def test_boundary_rejects_retention_and_cross_thread_access():
    retained = []

    class Extension:
        def on_boundary(self, boundary):
            retained.append(boundary)
            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(RuntimeError, match="orchestration thread"):
                    pool.submit(boundary.quiesce).result(timeout=5)
            assert boundary.quiesce() == ()

    LoopExtensions([Extension()]).publish(STEP)
    with pytest.raises(RuntimeError, match="no longer active"):
        retained[0].quiesce()


def test_extension_failure_stops_dispatch():
    calls = []

    class Extension:
        def on_boundary(self, boundary):
            calls.append(boundary)
            raise OSError("snapshot write failed")

    with pytest.raises(OSError, match="snapshot write failed"):
        LoopExtensions([Extension(), Extension()]).publish(STEP)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="no longer active"):
        calls[0].quiesce()


def test_extension_on_real_moe_boundary(tmp_path, monkeypatch):
    from test_tiny_moe_quant_smoke import test_tiny_qwen3_moe_quantization_smoke

    from gptqmodel.looper.module_looper import ModuleLooper

    owner = get_ident()
    observed = []

    class Extension:
        def on_boundary(self, boundary):
            assert get_ident() == owner
            results = boundary.quiesce()
            observed.append((boundary.step, results))

    original = ModuleLooper.__init__

    def initialize(self, *args, **kwargs):
        kwargs["extensions"] = [Extension()]
        original(self, *args, **kwargs)

    monkeypatch.setattr(ModuleLooper, "__init__", initialize)
    test_tiny_qwen3_moe_quantization_smoke(tmp_path)
    assert len(observed) == 1
    step, results = observed[0]
    assert step.kind == "layer"
    assert step.index == 0
    assert len(results) == 16  # Four attention projections and twelve expert projections.
