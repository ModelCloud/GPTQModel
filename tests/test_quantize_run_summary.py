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
    assert line == "Quantize summary: 2 layer(s), 3 module(s), 3m42s, peak VRAM 4.25 GiB"


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


def test_peak_vram_is_ignored_when_the_run_targeted_cpu(monkeypatch):
    """A CPU run on a machine with a card still allocates while loading."""
    from types import SimpleNamespace

    import gptqmodel.models.base as base

    monkeypatch.setattr(base.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(base.torch.cuda, "max_memory_allocated", lambda *a, **k: 4 * 1024**3)
    monkeypatch.setattr(base, "normalize_device", lambda value: SimpleNamespace(type="cpu"))

    assert base._peak_vram_gib(SimpleNamespace(device="cpu")) is None


def test_peak_vram_is_reported_for_a_cuda_run(monkeypatch):
    from types import SimpleNamespace

    import gptqmodel.models.base as base

    monkeypatch.setattr(base.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(base.torch.cuda, "max_memory_allocated", lambda *a, **k: 4 * 1024**3)
    monkeypatch.setattr(base, "normalize_device", lambda value: SimpleNamespace(type="cuda"))

    assert base._peak_vram_gib(SimpleNamespace(device="cuda")) == pytest.approx(4.0)
