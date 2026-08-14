from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from gptqmodel.utils import nvml


class _FakeNvml:
    def __init__(self):
        self.handle_calls = 0
        self.memory_calls = 0

    def nvmlDeviceGetHandleByPciBusId_v2(self, bus_id, handle_pointer):
        self.handle_calls += 1
        handle_pointer._obj.value = self.handle_calls
        return 0

    def nvmlDeviceGetMemoryInfo(self, handle, memory_pointer):
        self.memory_calls += 1
        memory_pointer._obj.used = handle.value * 1024
        return 0


def test_nvml_snapshot_is_process_wide_locked_and_cached(monkeypatch):
    fake = _FakeNvml()
    monkeypatch.setattr(nvml, "_load_nvml", lambda: fake)
    monkeypatch.setattr(nvml.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(nvml.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nvml.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        nvml.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(pci_domain_id=0, pci_bus_id=index + 1, pci_device_id=0),
    )
    nvml._reset_nvml_cache_for_tests()

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(lambda _: nvml.cuda_memory_used_snapshot(), range(64)))

    assert all(result == {"cuda:0": 1024, "cuda:1": 2048} for result in results)
    assert fake.handle_calls == 2
    assert fake.memory_calls == 2


def test_nvml_snapshot_refreshes_once_after_ttl(monkeypatch):
    fake = _FakeNvml()
    clock = iter((10.0, 10.5, 11.1))
    monkeypatch.setattr(nvml, "_load_nvml", lambda: fake)
    monkeypatch.setattr(nvml.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(nvml.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nvml.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        nvml.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(pci_domain_id=0, pci_bus_id=1, pci_device_id=0),
    )
    nvml._reset_nvml_cache_for_tests()

    nvml.cuda_memory_used_snapshot()
    nvml.cuda_memory_used_snapshot()
    nvml.cuda_memory_used_snapshot()
    assert fake.memory_calls == 2


def test_nvml_failure_is_cached_and_fails_closed(monkeypatch):
    monkeypatch.setattr(nvml, "_load_nvml", lambda: None)
    monkeypatch.setattr(nvml.time, "monotonic", lambda: 50.0)
    nvml._reset_nvml_cache_for_tests()
    assert nvml.cuda_memory_used_snapshot() is None
    assert nvml.cuda_memory_used_snapshot() is None
