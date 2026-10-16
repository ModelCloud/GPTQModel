import threading
import time
from types import SimpleNamespace

import torch

import gptqmodel as gptqmodel_module
from gptqmodel.utils import linalg_warmup
from gptqmodel.utils.threadx import WarmUpCtx


class _AttrCudaBackend:
    def __init__(self, preferred):
        self._preferred_linalg_library = preferred
        self.history = []

    @property
    def preferred_linalg_library(self):
        return self._preferred_linalg_library

    @preferred_linalg_library.setter
    def preferred_linalg_library(self, value):
        self.history.append(value)
        self._preferred_linalg_library = value


def test_run_torch_linalg_warmup_handles_attribute_style_cuda_backend(monkeypatch):
    calls = []
    backend = _AttrCudaBackend("cusolver")

    monkeypatch.setattr(linalg_warmup, "TORCH_GTE_210", False)
    monkeypatch.setattr(linalg_warmup, "_run_cholesky_and_eigh", lambda device, dtype: calls.append(("chol", dtype)))
    monkeypatch.setattr(linalg_warmup, "_run_svd", lambda device, dtype: calls.append(("svd", dtype)))
    monkeypatch.setattr(linalg_warmup, "_run_qr", lambda device, dtype: calls.append(("qr", dtype)))
    monkeypatch.setattr(linalg_warmup.torch, "backends", SimpleNamespace(cuda=backend))

    linalg_warmup.run_torch_linalg_warmup(torch.device("cuda"), WarmUpCtx.THREAD)

    assert backend.history == ["magma", "cusolver"]
    assert calls.count(("chol", torch.float32)) == 2
    assert calls.count(("chol", torch.float64)) == 1
    assert calls.count(("svd", torch.float32)) == 1
    assert calls.count(("svd", torch.float64)) == 1
    assert calls.count(("qr", torch.float32)) == 1
    assert calls.count(("qr", torch.float64)) == 1


def test_run_torch_linalg_warmup_always_runs_cholesky_and_eigh(monkeypatch):
    for torch_gte_210 in (False, True):
        calls = []

        monkeypatch.setattr(linalg_warmup, "TORCH_GTE_210", torch_gte_210)
        monkeypatch.setattr(linalg_warmup, "_run_cholesky_and_eigh", lambda device, dtype: calls.append(dtype))
        monkeypatch.setattr(linalg_warmup, "_run_svd", lambda device, dtype: None)
        monkeypatch.setattr(linalg_warmup, "_run_qr", lambda device, dtype: None)

        linalg_warmup.run_torch_linalg_warmup(torch.device("cpu"), WarmUpCtx.DEVICE)

        assert calls == [torch.float32, torch.float64]


def test_build_device_thread_pool_uses_thread_and_device_linalg_warmup(monkeypatch):
    captured = {}

    class DummyPool:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(gptqmodel_module, "DeviceThreadPool", DummyPool)

    pool = gptqmodel_module._build_device_thread_pool()
    warmups = captured["kwargs"]["warmups"]

    assert isinstance(pool, DummyPool)
    assert {key: task.scope for key, task in warmups.items()} == {
        "cuda": WarmUpCtx.THREAD_AND_DEVICE,
        "xpu": WarmUpCtx.THREAD_AND_DEVICE,
        "mps": WarmUpCtx.THREAD_AND_DEVICE,
        "cpu": WarmUpCtx.THREAD_AND_DEVICE,
    }


class _BlockingPrewarm:
    """Stand-in for `_run_linear_prewarm` that records calls and blocks until released."""

    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, device, dtype):
        self.calls.append((str(device), str(dtype)))
        self.started.set()
        self.release.wait(10)


def _isolate_linear_prewarm(monkeypatch):
    """Give each test its own prewarm registry so process-global dedupe cannot leak."""
    monkeypatch.setattr(linalg_warmup, "_LINEAR_PREWARM_RUNNING", set())
    monkeypatch.setattr(linalg_warmup, "_LINEAR_PREWARM_DONE", set())
    monkeypatch.setattr(linalg_warmup, "_LINEAR_PREWARM_THREADS", {})


def test_kick_torch_linear_prewarm_dedupes_by_device_type_and_dtype(monkeypatch):
    prewarm = _BlockingPrewarm()
    monkeypatch.setattr(linalg_warmup, "_run_linear_prewarm", prewarm)
    _isolate_linear_prewarm(monkeypatch)

    thread = linalg_warmup.kick_torch_linear_prewarm("cuda:1", torch.float16)
    assert thread is not None
    assert prewarm.started.wait(5)

    try:
        # The library init is process-global: another ordinal of the same device type
        # and dtype must not start a second warmup.
        assert linalg_warmup.kick_torch_linear_prewarm("cuda:0", torch.float16) is None
        assert linalg_warmup.kick_torch_linear_prewarm(torch.device("cuda"), torch.float16) is None
        # Unsupported dtype / device family are ignored instead of raising.
        assert linalg_warmup.kick_torch_linear_prewarm("cuda", torch.float64) is None
        assert linalg_warmup.kick_torch_linear_prewarm("cpu", torch.float16) is None

        other = linalg_warmup.kick_torch_linear_prewarm("cuda", torch.bfloat16)
        assert other is not None
    finally:
        prewarm.release.set()
        thread.join(5)
        if "other" in locals() and other is not None:
            other.join(5)

    assert prewarm.calls == [("cuda:1", "torch.float16"), ("cuda", "torch.bfloat16")]


def test_kick_torch_linear_prewarm_respects_kill_switch(monkeypatch):
    prewarm = _BlockingPrewarm()
    monkeypatch.setattr(linalg_warmup, "_run_linear_prewarm", prewarm)
    _isolate_linear_prewarm(monkeypatch)
    monkeypatch.setenv("GPTQMODEL_LINEAR_PREWARM", "0")

    assert linalg_warmup.kick_torch_linear_prewarm("cuda", torch.float16) is None
    linalg_warmup.await_torch_linear_prewarm("cuda", torch.float16, timeout=0.1)
    assert prewarm.calls == []


def test_await_torch_linear_prewarm_joins_only_the_matching_prewarm(monkeypatch):
    prewarm = _BlockingPrewarm()
    monkeypatch.setattr(linalg_warmup, "_run_linear_prewarm", prewarm)
    _isolate_linear_prewarm(monkeypatch)

    # Nothing in flight: the join must be a no-op.
    linalg_warmup.await_torch_linear_prewarm("cuda", torch.float16, timeout=0.1)

    thread = linalg_warmup.kick_torch_linear_prewarm("cuda", torch.float16)
    assert thread is not None
    assert prewarm.started.wait(5)

    try:
        # A dtype that is not being warmed has nothing to join.
        start = time.monotonic()
        linalg_warmup.await_torch_linear_prewarm("cuda", torch.bfloat16, timeout=1)
        assert time.monotonic() - start < 1

        joiner = threading.Thread(
            target=linalg_warmup.await_torch_linear_prewarm,
            args=("cuda", torch.float16),
            kwargs={"timeout": 5},
        )
        joiner.start()
        joiner.join(0.2)
        assert joiner.is_alive(), "await must block while the matching prewarm is in flight"

        prewarm.release.set()
        joiner.join(5)
        assert not joiner.is_alive()
    finally:
        prewarm.release.set()
        thread.join(5)


def test_await_torch_linear_prewarm_is_noop_after_completion(monkeypatch):
    prewarm = _BlockingPrewarm()
    monkeypatch.setattr(linalg_warmup, "_run_linear_prewarm", prewarm)
    _isolate_linear_prewarm(monkeypatch)

    thread = linalg_warmup.kick_torch_linear_prewarm("cuda", torch.float16)
    assert thread is not None
    prewarm.release.set()
    thread.join(5)

    start = time.monotonic()
    linalg_warmup.await_torch_linear_prewarm("cuda", torch.float16, timeout=1)
    assert time.monotonic() - start < 1

    # A completed warmup is shared, so it must not be restarted.
    assert linalg_warmup.kick_torch_linear_prewarm(torch.device("cuda:0"), torch.float16) is None
    assert prewarm.calls == [("cuda", "torch.float16")]


def test_kick_torch_linear_prewarm_registers_only_started_threads(monkeypatch):
    """`await` may run concurrently, and joining an unstarted thread raises."""
    prewarm = _BlockingPrewarm()
    monkeypatch.setattr(linalg_warmup, "_run_linear_prewarm", prewarm)
    _isolate_linear_prewarm(monkeypatch)

    thread = linalg_warmup.kick_torch_linear_prewarm("cuda", torch.float16)
    assert thread is not None
    assert thread.ident is not None

    try:
        linalg_warmup.await_torch_linear_prewarm(
            "cuda", torch.float16, timeout=0.1
        )  # must not raise RuntimeError
    finally:
        prewarm.release.set()
        thread.join(5)
