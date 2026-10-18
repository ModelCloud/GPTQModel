import threading
import time
from types import SimpleNamespace

import torch
import torch.utils._python_dispatch  # noqa: F401

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


def test_make_spd_is_symmetric_positive_definite_without_matmul():
    """The warmup input must not itself dispatch a GEMM (it loads hipBLASLt)."""
    dispatched = []

    class _Recorder(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            dispatched.append(str(func))
            return func(*args, **(kwargs or {}))

    for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        with _Recorder():
            spd = linalg_warmup._make_spd(4, torch.device("cpu"), dtype)

        assert torch.equal(spd, spd.transpose(-1, -2)), dtype
        # CPU cholesky has no half dtype, so check the spectrum in float64.
        assert float(torch.linalg.eigvalsh(spd.double()).min()) > 0, dtype

    banned = ("aten.mm.", "aten.matmul", "aten.addmm", "aten.bmm", "aten.linear")
    assert not [name for name in dispatched if name.startswith(banned)], dispatched


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


def test_kick_torch_quantize_prewarm_is_independent_of_the_linear_payload(monkeypatch):
    """Both payloads warm the same device/dtype, so each needs its own registry slot."""
    linear = _BlockingPrewarm()
    quantize = _BlockingPrewarm()
    monkeypatch.setattr(linalg_warmup, "_run_linear_prewarm", linear)
    monkeypatch.setattr(linalg_warmup, "_run_quantize_prewarm", quantize)
    _isolate_linear_prewarm(monkeypatch)

    linear_thread = linalg_warmup.kick_torch_linear_prewarm("cuda", torch.float16)
    quantize_thread = linalg_warmup.kick_torch_quantize_prewarm("cuda", torch.float16)
    assert linear_thread is not None
    assert quantize_thread is not None
    assert linear.started.wait(5)
    assert quantize.started.wait(5)

    try:
        # Dedupe is per payload: a second kick of either one is a no-op, and the
        # process-global device/dtype rules still apply.
        assert linalg_warmup.kick_torch_linear_prewarm("cuda:0", torch.float16) is None
        assert linalg_warmup.kick_torch_quantize_prewarm("cuda:0", torch.float16) is None
        assert linalg_warmup.kick_torch_quantize_prewarm("cpu", torch.float16) is None
        assert linalg_warmup.kick_torch_quantize_prewarm("cuda", torch.float64) is None
    finally:
        linear.release.set()
        quantize.release.set()
        linear_thread.join(5)
        quantize_thread.join(5)

    assert linear.calls == [("cuda", "torch.float16")]
    assert quantize.calls == [("cuda", "torch.float16")]


def test_kick_torch_quantize_prewarm_respects_kill_switch(monkeypatch):
    prewarm = _BlockingPrewarm()
    monkeypatch.setattr(linalg_warmup, "_run_quantize_prewarm", prewarm)
    _isolate_linear_prewarm(monkeypatch)
    monkeypatch.setenv("GPTQMODEL_QUANTIZE_PREWARM", "0")

    assert linalg_warmup.kick_torch_quantize_prewarm("cuda", torch.float16) is None
    linalg_warmup.await_torch_linear_prewarm("cuda", torch.float16, timeout=0.1)
    assert prewarm.calls == []


def test_await_torch_linear_prewarm_joins_every_payload_and_can_filter_one(monkeypatch):
    linear = _BlockingPrewarm()
    quantize = _BlockingPrewarm()
    monkeypatch.setattr(linalg_warmup, "_run_linear_prewarm", linear)
    monkeypatch.setattr(linalg_warmup, "_run_quantize_prewarm", quantize)
    _isolate_linear_prewarm(monkeypatch)

    thread = linalg_warmup.kick_torch_quantize_prewarm("cuda", torch.float16)
    assert thread is not None
    assert quantize.started.wait(5)

    try:
        # Filtering to the linear payload must not wait on the in-flight quantize one.
        start = time.monotonic()
        linalg_warmup.await_torch_linear_prewarm(
            "cuda", torch.float16, timeout=0.5, payload=linalg_warmup._PREWARM_LINEAR
        )
        assert time.monotonic() - start < 0.5

        joiner = threading.Thread(
            target=linalg_warmup.await_torch_linear_prewarm,
            args=("cuda", torch.float16),
            kwargs={"timeout": 5},
        )
        joiner.start()
        joiner.join(0.2)
        assert joiner.is_alive(), "the default join must cover the quantize payload"

        filtered = threading.Thread(
            target=linalg_warmup.await_torch_linear_prewarm,
            args=("cuda", torch.float16),
            kwargs={"timeout": 5, "payload": linalg_warmup._PREWARM_QUANTIZE},
        )
        filtered.start()
        filtered.join(0.2)
        assert filtered.is_alive(), "an explicit payload filter must join that payload"

        quantize.release.set()
        joiner.join(5)
        filtered.join(5)
        assert not joiner.is_alive()
        assert not filtered.is_alive()
    finally:
        quantize.release.set()
        thread.join(5)

    assert linear.calls == []


def test_run_quantize_prewarm_launches_kernels_without_touching_the_rng():
    """The payload may only warm kernels: perturbing the RNG would change results."""
    dispatched = []

    class _Recorder(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            dispatched.append(str(func))
            return func(*args, **(kwargs or {}))

    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()

    with _Recorder():
        linalg_warmup._run_quantize_prewarm(torch.device("cpu"), torch.float16)

    assert torch.equal(before, torch.get_rng_state())
    assert len(dispatched) >= 40, len(dispatched)
    assert any(name.startswith("aten.topk.") for name in dispatched)
