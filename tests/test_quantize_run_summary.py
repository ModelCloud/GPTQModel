# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""The end-of-run summary has to be readable and honest.

A quantization run ends with a flushed region timer, which says how time was
spent but not what was produced or how much of the card it used. The summary
line is the one a user pastes into a report, so its formatting and its
"did this run touch the GPU" decision are both asserted here.
"""

from types import SimpleNamespace

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


def _advancing_fuse(clock, events, *, error=None):
    """Stand-in for `BaseQModel.fuse` that spends measurable time."""

    def fuse(*args, **kwargs):
        events.append("fuse")
        clock["now"] += 7.0
        if error is not None:
            raise error

    return fuse


def _stub_quantize_model(
    monkeypatch,
    *,
    clock,
    events,
    config,
    fuse,
    weight_only_result=None,
    rotation_error=None,
):
    """Build the smallest `BaseQModel` that runs the real `quantize()` orchestration.

    The quantization run itself is stubbed; the phases around it
    (`after_quantize`, rotation setup, fusion) are what these tests pin.
    """

    import gptqmodel.models.base as base

    lines = []
    real_format = base.format_quantize_summary

    def capture(result, **kwargs):
        events.append("summary")
        line = real_format(result, **kwargs)
        lines.append(line)
        return line

    def rotation(*args, **kwargs):
        events.append("rotation")
        if rotation_error is not None:
            raise rotation_error

    monkeypatch.setattr(base.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(base, "format_quantize_summary", capture)
    monkeypatch.setattr(base, "require_fusable_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(base, "_setup_rotation_online_had", rotation)

    qmodel = object.__new__(base.BaseQModel)
    qmodel.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    qmodel.quantize_config = config
    qmodel.quantized = False
    qmodel.support_batch_quantize = True
    qmodel._auto_fusion = True
    qmodel.quant_region_timer = None
    qmodel.after_quantize = lambda: events.append("after_quantize")
    qmodel.fuse = fuse
    qmodel.module_plan = lambda: None
    qmodel.extract_layers_node = lambda: ["model.layers"]
    qmodel._warn_fused_unit_dynamic_overrides = lambda: None
    qmodel._quantize_weight_only = lambda **kwargs: weight_only_result
    return qmodel, lines


def test_summary_is_logged_after_finalization(monkeypatch):
    """A success line must describe the finished run, not the pre-fusion state."""
    from gptqmodel.quantization.config import NF12Config

    clock = {"now": 100.0}
    events = []
    result = {"model.layers.0": [{}, {}], "model.layers.1": [{}]}
    qmodel, lines = _stub_quantize_model(
        monkeypatch,
        clock=clock,
        events=events,
        config=NF12Config(device="cpu"),
        fuse=_advancing_fuse(clock, events),
        weight_only_result=result,
    )

    assert qmodel.quantize() == result
    assert events == ["after_quantize", "rotation", "fuse", "summary"]
    # 7s is the time the finalization step spent; it must be included.
    assert lines == ["Quantize summary: 2 layer(s), 3 module(s), 7s"]


def test_summary_is_not_logged_when_fusion_raises(monkeypatch):
    from gptqmodel.quantization.config import NF12Config

    clock = {"now": 100.0}
    events = []
    qmodel, lines = _stub_quantize_model(
        monkeypatch,
        clock=clock,
        events=events,
        config=NF12Config(device="cpu"),
        fuse=_advancing_fuse(clock, events, error=RuntimeError("fusion failed")),
        weight_only_result={"model.layers.0": [{}]},
    )

    with pytest.raises(RuntimeError, match="fusion failed"):
        qmodel.quantize()

    assert lines == []
    assert events == ["after_quantize", "rotation", "fuse"]


def test_summary_is_not_logged_when_rotation_setup_raises(monkeypatch):
    from gptqmodel.quantization.config import NF12Config

    clock = {"now": 100.0}
    events = []
    qmodel, lines = _stub_quantize_model(
        monkeypatch,
        clock=clock,
        events=events,
        config=NF12Config(device="cpu"),
        fuse=_advancing_fuse(clock, events),
        weight_only_result={"model.layers.0": [{}]},
        rotation_error=RuntimeError("rotation setup failed"),
    )

    with pytest.raises(RuntimeError, match="rotation setup failed"):
        qmodel.quantize()

    assert lines == []


def _staged_gsq_config():
    from gptqmodel.quantization.config import GPTQConfig, GSQTrainingConfig

    return GPTQConfig(
        bits=4,
        group_size=128,
        device="cpu",
        gsq_training=GSQTrainingConfig(enabled=True),
    )


def _staged_gsq_blocks():
    return [
        {"layer_index": 0, "stages": {"q_proj": {}, "k_proj": {}}},
        {"layer_index": 1, "stages": {"q_proj": {}}},
    ]


def test_staged_gsq_summary_counts_blocks_as_layers(monkeypatch):
    """The staged-GSQ path must end with the same summary, counted by block."""
    import gptqmodel.looper.gsq_training_model as gsq_training_model

    clock = {"now": 100.0}
    events = []
    blocks = _staged_gsq_blocks()

    def fake_gsq_public(*args, **kwargs):
        events.append("gsq_run")
        return {"gsq_training": blocks}

    monkeypatch.setattr(gsq_training_model, "quantize_llama_gsq_public", fake_gsq_public)
    qmodel, lines = _stub_quantize_model(
        monkeypatch,
        clock=clock,
        events=events,
        config=_staged_gsq_config(),
        fuse=_advancing_fuse(clock, events),
    )

    assert qmodel.quantize() == {"gsq_training": blocks}
    assert events == ["gsq_run", "after_quantize", "fuse", "summary"]
    assert lines == ["Quantize summary: 2 layer(s), 3 module(s), 7s"]


def test_staged_gsq_summary_is_not_logged_when_finalization_raises(monkeypatch):
    import gptqmodel.looper.gsq_training_model as gsq_training_model

    clock = {"now": 100.0}
    events = []
    monkeypatch.setattr(
        gsq_training_model,
        "quantize_llama_gsq_public",
        lambda *args, **kwargs: {"gsq_training": _staged_gsq_blocks()},
    )
    qmodel, lines = _stub_quantize_model(
        monkeypatch,
        clock=clock,
        events=events,
        config=_staged_gsq_config(),
        fuse=_advancing_fuse(clock, events, error=RuntimeError("fusion failed")),
    )

    with pytest.raises(RuntimeError, match="fusion failed"):
        qmodel.quantize()

    assert lines == []


def test_gsq_summary_counts_map_each_block_to_a_layer():
    import gptqmodel.models.base as base

    result = {
        "gsq_training": [
            {"layer_index": 3, "stages": {"q_proj": {}, "o_proj": {}}},
            {"layer_index": 4, "stages": {}},
        ]
    }

    assert base._quantize_summary_counts(result) == {
        "layer.3": ("q_proj", "o_proj"),
        "layer.4": (),
    }
    # Ordinary results and unexpected shapes pass through untouched.
    assert base._quantize_summary_counts({"model.layers.0": [{}]}) == {"model.layers.0": [{}]}
    assert base._quantize_summary_counts(None) is None
