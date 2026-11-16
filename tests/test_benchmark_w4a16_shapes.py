# GPU=-1
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shape presets and bandwidth math for scripts/benchmark_w4a16_shapes.py."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_w4a16_shapes.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("benchmark_w4a16_shapes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves cls.__module__ through sys.modules, so register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_presets_are_well_formed(bench):
    assert bench.PRESETS, "at least one preset"
    for preset, shapes in bench.PRESETS.items():
        names = [shape.name for shape in shapes]
        assert len(names) == len(set(names)), f"duplicate shape name in {preset}"
        for shape in shapes:
            assert shape.k > 0 and shape.n > 0, f"{preset}/{shape.name}"
            assert shape.k % bench.GROUP_SIZE == 0, (
                f"{preset}/{shape.name}: k must be a multiple of group_size"
            )


def test_qwen3_8b_matches_the_released_config(bench):
    """hidden 4096, 32 query and 8 KV heads of width 128, FFN 12288."""
    shapes = {shape.name: shape for shape in bench.PRESETS["qwen3_8b"]}
    assert shapes["q_proj"] == bench.Shape("q_proj", 4096, 4096)
    assert shapes["kv_proj"] == bench.Shape("kv_proj", 4096, 1024)
    assert shapes["o_proj"] == bench.Shape("o_proj", 4096, 4096)
    assert shapes["gate_up"] == bench.Shape("gate_up", 4096, 12288)
    assert shapes["down"] == bench.Shape("down", 12288, 4096)


def test_fusion_bytes_match_the_sum_of_their_parts(bench):
    """A fusion must move exactly the bytes its parts moved, or the speedup lies."""
    for preset, groups in bench.FUSIONS.items():
        for group in groups:
            parts = sum(shape.k * shape.n for shape in group.parts)
            fused = group.fused.k * group.fused.n
            assert parts == fused, f"{preset}/{group.name}: {parts} != {fused}"


def test_fusion_names_are_unique_per_preset(bench):
    for preset, groups in bench.FUSIONS.items():
        names = [group.name for group in groups]
        assert len(names) == len(set(names)), f"duplicate fusion in {preset}"


def test_fused_width_is_repack_aligned(bench):
    """Marlin's repacked layout tiles N in blocks of 64; a fused width has to fit."""
    for preset, groups in bench.FUSIONS.items():
        for group in groups:
            assert group.fused.n % 64 == 0, f"{preset}/{group.name}"


def test_gbps_counts_weight_bytes_only(bench):
    """One millisecond means weight_bytes / 1e-3 seconds, reported in GB/s."""
    shape = bench.Shape("toy", 4096, 1024)
    weight_bytes = 4096 * 1024 * bench.BITS / 8
    assert bench.gbps(shape, ms=1.0) == pytest.approx(weight_bytes / 1e-3 / 1e9)


def test_gbps_scales_inversely_with_time(bench):
    shape = bench.Shape("toy", 4096, 1024)
    assert bench.gbps(shape, ms=0.5) == pytest.approx(2 * bench.gbps(shape, ms=1.0))


def test_passes_vary_measurement_order_without_losing_duplicate_names(bench, monkeypatch):
    calls = []

    def measure(shape, m, args, device, dtype):
        calls.append((shape.k, shape.n, m))
        return {
            "shape": shape.name, "k": shape.k, "n": shape.n, "m": m,
            "cold_min_ms": 1.0, "cold_median_ms": 1.0, "cold_gbps": 1.0,
            "hot_min_ms": 1.0, "cold_spread": 0.0,
        }

    monkeypatch.setattr(bench, "warm_up_gpu", lambda device: None)
    monkeypatch.setattr(bench, "_measure", measure)
    args = SimpleNamespace(
        device="cuda:0", dtype="float16", preset="qwen3_8b",
        shapes=["same:128:64", "same:256:64"], m=[1], passes=3,
    )
    rows = bench.run(args)

    assert calls[:2] != calls[2:4]
    assert {call: calls.count(call) for call in set(calls)} == {
        (128, 64, 1): 3, (256, 64, 1): 3,
    }
    assert [(row["k"], row["n"]) for row in rows] == [(128, 64), (256, 64)]


def test_fusion_passes_interleave_parts_and_fused_call(bench, monkeypatch):
    a = bench.Shape("a", 128, 64)
    b = bench.Shape("b", 128, 64)
    fused = bench.Shape("fused", 128, 128)
    monkeypatch.setitem(bench.FUSIONS, "test", (bench.Fusion("pair", (a, b), fused),))
    monkeypatch.setattr(bench, "warm_up_gpu", lambda device: None)
    calls = []

    def measure(shape, m, args, device, dtype):
        calls.append(shape.name)
        return {
            "shape": shape.name, "k": shape.k, "n": shape.n,
            "cold_min_ms": 1.0, "cold_gbps": 1.0,
        }

    monkeypatch.setattr(bench, "_measure", measure)
    args = SimpleNamespace(device="cuda:0", dtype="float16", preset="test", m=[1], passes=3)
    rows = bench.run_fusions(args)

    assert [set(calls[index:index + 3]) for index in (0, 3, 6)] == [
        {"a", "b", "fused"}, {"a", "b", "fused"}, {"a", "b", "fused"},
    ]
    assert len({tuple(calls[index:index + 3]) for index in (0, 3, 6)}) > 1
    assert rows[0]["parts_calls"] == 2
    assert rows[0]["speedup"] == pytest.approx(2.0)


PROFILE = {
    "available": True,
    "sms": 82,
    "max_shared_mem": 101376,
    "capability_major": 12,
    "capability_minor": 0,
    "l2_cache_size": 67108864,
    "max_threads_per_sm": 1536,
    "has_tensor_memory": 0,
}


def test_device_header_names_the_limits_selection_reads(bench):
    line = bench.format_device_header(PROFILE, "RTX 5090 Laptop")
    assert "RTX 5090 Laptop" in line
    assert "sms 82" in line
    assert "shared 99 KiB" in line
    assert "L2 64 MiB" in line
    assert "sm 12.0" in line
    assert "tensor memory no" in line


def test_device_header_reports_tensor_memory_when_present(bench):
    datacenter = {**PROFILE, "capability_major": 10, "has_tensor_memory": 1}
    assert "tensor memory yes" in bench.format_device_header(datacenter, "B200")


def test_device_header_says_why_the_profile_is_missing(bench):
    unavailable = {"available": False, "reason": "RuntimeError: no CUDA"}
    line = bench.format_device_header(unavailable, "unknown")
    assert "profile unavailable" in line
    assert "no CUDA" in line


def test_json_body_carries_device_and_rows(bench):
    rows = [{"shape": "q_proj", "k": 4096, "n": 4096, "m": 1, "cold_gbps": 600.0}]
    payload = bench.build_payload(rows, PROFILE, "RTX 5090 Laptop")

    assert payload["rows"] == rows
    assert payload["device"]["name"] == "RTX 5090 Laptop"
    assert payload["device"]["sms"] == 82
    # The rows keep their own shape so existing readers do not change.
    assert set(payload) == {"device", "rows"}


def test_json_body_records_a_profile_outage_instead_of_dropping_it(bench):
    payload = bench.build_payload([], {"available": False, "reason": "no CUDA"}, "unknown")
    assert payload["device"]["available"] is False
    assert payload["device"]["reason"] == "no CUDA"


def test_disabled_profile_is_recorded_rather_than_omitted(bench):
    """--no-device-profile must say so, so a reader can tell why it is missing."""
    payload = bench.build_payload([], {"available": False, "reason": "disabled by --no-device-profile"}, "card")
    assert payload["device"]["reason"] == "disabled by --no-device-profile"
    assert payload["device"]["name"] == "card"


def test_resolve_device_index_reads_the_explicit_ordinal(bench):
    """--device cuda:1 names GPU 1 even when the process inherited GPU 0."""
    assert bench.resolve_device_index("cuda:1") == 1
    assert bench.resolve_device_index("cuda:0") == 0


def test_resolve_device_index_falls_back_to_the_current_device(bench, monkeypatch):
    """A bare `cuda` keeps the inherited ordinal."""
    monkeypatch.setattr(bench, "torch", SimpleNamespace(
        device=bench.torch.device,
        cuda=SimpleNamespace(current_device=lambda: 3),
    ))
    assert bench.resolve_device_index("cuda") == 3


def test_resolve_device_index_rejects_a_non_cuda_device(bench):
    with pytest.raises(ValueError, match="requires a CUDA device"):
        bench.resolve_device_index("cpu")


def test_timing_brackets_the_requested_device(bench, monkeypatch):
    """Events and the synchronize must target --device, not the current GPU.

    ``torch.cuda.Event`` records on the current device's current stream. With
    the process on GPU 0 and ``--device cuda:1`` the un-fixed code brackets
    GPU 0's idle stream, so the reported latency is unrelated to the kernel.
    """

    seen = {"sync": [], "scopes": []}

    class FakeEvent:
        def __init__(self, enable_timing=False):
            self._ms = 0.0

        def record(self):
            self._ms = 1.0

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return self._ms

    class FakeDeviceScope:
        def __init__(self, device):
            seen["scopes"].append(device)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(bench, "torch", SimpleNamespace(
        device=bench.torch.device,
        cuda=SimpleNamespace(
            synchronize=lambda device=None: seen["sync"].append(device),
            Event=FakeEvent,
            device=FakeDeviceScope,
        ),
    ))

    samples = bench._samples(lambda: None, iters=3, warmup=0, flush=None, device="cuda:1")

    assert len(samples) == 3
    assert seen["sync"] == ["cuda:1"]
    assert seen["scopes"] == ["cuda:1"]


def test_main_profiles_and_names_the_requested_device(bench, monkeypatch, tmp_path):
    """The JSON device identity must describe --device, not the current GPU."""

    seen = {}

    monkeypatch.setattr(bench, "torch", SimpleNamespace(
        device=bench.torch.device,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            # The process inherited GPU 0. `--device cuda:1` must win.
            current_device=lambda: 0,
            get_device_name=lambda index: f"card{index}",
        ),
    ))

    def profile_for(index):
        seen["profile_index"] = index
        return {"available": True, **PROFILE}

    monkeypatch.setattr(bench, "_resolve_device_profile", profile_for)
    monkeypatch.setattr(bench, "run", lambda args: [])
    monkeypatch.setattr(bench, "format_device_header",
                        lambda profile, name: f"{name} header")

    out = tmp_path / "result.json"
    monkeypatch.setattr(
        sys, "argv",
        ["benchmark_w4a16_shapes.py", "--device", "cuda:1", "--json", str(out)],
    )

    assert bench.main() == 0
    assert seen["profile_index"] == 1
    payload = json.loads(out.read_text())
    assert payload["device"]["name"] == "card1"
    assert payload["device"]["sms"] == PROFILE["sms"]


def test_main_rejects_a_non_cuda_device(bench, monkeypatch):
    monkeypatch.setattr(bench, "torch", SimpleNamespace(
        device=bench.torch.device,
        cuda=SimpleNamespace(is_available=lambda: True),
    ))
    monkeypatch.setattr(sys, "argv", ["benchmark_w4a16_shapes.py", "--device", "cpu"])

    with pytest.raises(SystemExit):
        bench.main()
