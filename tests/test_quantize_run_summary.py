# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""The end-of-run summary has to be readable and honest.

A quantization run ends with a flushed region timer, which says how time was
spent but not what was produced or how much of the card it used. The summary
line is the one a user pastes into a report, so its formatting and its
"did this run touch the GPU" decision are both asserted here.
"""

import pytest

from gptqmodel.utils.logger import format_duration, format_quantize_summary


GIB = 1024 ** 3


def _fake_multi_device_cuda(monkeypatch, peaks):
    """Expose a multi-GPU CUDA view whose peak differs per device."""

    import gptqmodel.models.base as base

    calls = []

    def max_memory_allocated(device=None):
        calls.append(device)
        index = 0 if device is None else int(device.index)
        return peaks[index]

    monkeypatch.setattr(base.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(base.torch.cuda, "device_count", lambda: len(peaks))
    monkeypatch.setattr(base.torch.cuda, "max_memory_allocated", max_memory_allocated)
    return calls


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "0s"),
        (0.4, "0s"),
        (42.0, "42s"),
        (59.4, "59s"),
        (60.0, "1m00s"),
        (221.7, "3m42s"),
        (3599.0, "59m59s"),
        (3725.0, "1h02m"),
    ],
)
def test_duration_is_readable_at_every_scale(seconds, expected):
    assert format_duration(seconds) == expected


def test_summary_counts_layers_and_modules():
    result = {"model.layers.0": [{}, {}], "model.layers.1": [{}]}
    line = format_quantize_summary(result, elapsed_s=221.7, peak_vram_gib=4.25)
    assert line == (
        "Quantize summary: 2 layer(s), 3 module(s), 3m42s, peak VRAM 4.25 GiB (per-device max)"
    )


def test_summary_omits_vram_when_the_run_did_not_use_the_gpu():
    line = format_quantize_summary({"model.layers.0": [{}]}, elapsed_s=42.0)
    assert "VRAM" not in line
    assert line == "Quantize summary: 1 layer(s), 1 module(s), 42s"


def test_summary_survives_an_empty_result():
    assert format_quantize_summary(None, elapsed_s=0.4) == "Quantize summary: 0 layer(s), 0 module(s), 0s"
    assert format_quantize_summary({}, elapsed_s=0.4) == "Quantize summary: 0 layer(s), 0 module(s), 0s"


def test_summary_tolerates_missing_module_lists():
    line = format_quantize_summary({"layer": None}, elapsed_s=1.0)
    assert "1 layer(s), 0 module(s)" in line


def test_peak_vram_reads_every_quantization_device_not_the_current_one(monkeypatch):
    """The busiest card must be visible even when the thread sits on GPU 0."""
    from types import SimpleNamespace

    import gptqmodel.models.base as base

    calls = _fake_multi_device_cuda(monkeypatch, {0: 2 * GIB, 1: 9 * GIB})

    peak = base._peak_vram_gib(SimpleNamespace(device="cuda:1", compute_device_filter=None))

    assert peak == pytest.approx(9.0)
    assert [str(device) for device in calls] == ["cuda:0", "cuda:1"]


def test_peak_vram_honours_the_compute_device_filter(monkeypatch):
    """Only devices that survive `compute_device_filter` are read."""
    from types import SimpleNamespace

    import gptqmodel.models.base as base

    calls = _fake_multi_device_cuda(monkeypatch, {0: 2 * GIB, 1: 9 * GIB})
    config = SimpleNamespace(
        device="cuda",
        compute_device_filter=lambda devices: [device for device in devices if device.index == 1],
    )

    assert base._peak_vram_gib(config) == pytest.approx(9.0)
    assert [str(device) for device in calls] == ["cuda:1"]


def test_peak_reset_targets_only_the_run_devices(monkeypatch):
    from types import SimpleNamespace

    import gptqmodel.models.base as base

    _fake_multi_device_cuda(monkeypatch, {0: 2 * GIB, 1: 9 * GIB})
    reset_calls = []
    monkeypatch.setattr(
        base.torch.cuda,
        "reset_peak_memory_stats",
        lambda *args, **kwargs: reset_calls.append(args),
    )

    devices = base._quantize_vram_devices(SimpleNamespace(device="cuda", compute_device_filter=None))
    base._reset_peak_vram_stats(devices)

    assert [str(device) for device in devices] == ["cuda:0", "cuda:1"]
    assert [str(args[0]) for args in reset_calls] == ["cuda:0", "cuda:1"]


def test_auto_selected_device_still_reports_cuda_peaks(monkeypatch):
    """`device=None` means auto selection, which picks CUDA on a CUDA host."""
    from types import SimpleNamespace

    import gptqmodel.models.base as base

    calls = _fake_multi_device_cuda(monkeypatch, {0: 2 * GIB, 1: 9 * GIB})

    peak = base._peak_vram_gib(SimpleNamespace(device=None, compute_device_filter=None))

    assert peak == pytest.approx(9.0)
    assert [str(device) for device in calls] == ["cuda:0", "cuda:1"]


def test_cpu_run_never_touches_cuda_peak_counters(monkeypatch):
    """A CPU run on a machine with a card must not reset or read its peaks."""
    from types import SimpleNamespace

    import gptqmodel.models.base as base

    monkeypatch.setattr(base.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(base.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        base.torch.cuda,
        "reset_peak_memory_stats",
        lambda *args, **kwargs: pytest.fail("CPU run reset a CUDA peak counter"),
    )
    monkeypatch.setattr(
        base.torch.cuda,
        "max_memory_allocated",
        lambda *args, **kwargs: pytest.fail("CPU run read a CUDA peak counter"),
    )
    config = SimpleNamespace(device="cpu", compute_device_filter=None)

    devices = base._quantize_vram_devices(config)
    assert devices == []
    base._reset_peak_vram_stats(devices)
    assert base._peak_vram_gib(config) is None
