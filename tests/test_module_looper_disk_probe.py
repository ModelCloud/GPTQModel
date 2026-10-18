import types

import torch

from gptqmodel.looper import module_looper


def _dummy_model():
    return types.SimpleNamespace(
        support_batch_quantize=False,
        write_shared_kv_cache=False,
        quantize_config=types.SimpleNamespace(
            device=torch.device("cpu"),
            dense_vram_strategy="exclusive",
            dense_vram_strategy_devices=None,
            moe_vram_strategy="exclusive",
            moe_vram_strategy_devices=None,
            compute_device_filter=None,
            calibration_data_device="balanced",
            auto_forward_data_parallel=True,
            moe_routing_bypass=lambda: False,
        ),
        layer_callback=None,
    )


def _isolate_probe(monkeypatch):
    monkeypatch.setattr(module_looper, "_IO_WRITE_SPEED_MB", None)


def test_io_write_performance_skips_the_probe_when_disabled(monkeypatch):
    """A run that never touches the disk must not write 100 MB to find that out."""
    calls = []
    monkeypatch.setattr(
        module_looper, "estimate_disk_io_speed", lambda: calls.append(1) or 1e9
    )
    _isolate_probe(monkeypatch)

    assert module_looper.io_write_performance(False) is None
    assert calls == []
    assert module_looper._IO_WRITE_SPEED_MB is None

    assert module_looper.io_write_performance(True) == 1e9 / (1024 * 1024)
    assert calls == [1]


def test_io_write_performance_probes_once_and_caches(monkeypatch):
    calls = []
    monkeypatch.setattr(
        module_looper, "estimate_disk_io_speed", lambda: calls.append(1) or 2 * 1024 * 1024
    )
    _isolate_probe(monkeypatch)

    assert module_looper.io_write_performance(True) == 2.0
    assert module_looper.io_write_performance(True) == 2.0
    assert calls == [1]


def test_module_looper_only_probes_the_disk_when_the_run_streams_through_it(monkeypatch):
    calls = []
    monkeypatch.setattr(
        module_looper, "estimate_disk_io_speed", lambda: calls.append(1) or 1e9
    )
    _isolate_probe(monkeypatch)
    monkeypatch.setattr(module_looper, "select_forward_devices", lambda _base: [torch.device("cpu")])

    module_looper.ModuleLooper(model=_dummy_model(), processors=[])
    assert calls == []

    module_looper.ModuleLooper(
        model=_dummy_model(), processors=[], disk_write_expected=True
    )
    assert calls == [1]
