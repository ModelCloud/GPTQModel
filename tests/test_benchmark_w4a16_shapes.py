# GPU=-1
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shape presets and bandwidth math for scripts/benchmark_w4a16_shapes.py."""

import importlib.util
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
