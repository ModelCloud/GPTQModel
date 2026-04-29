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
