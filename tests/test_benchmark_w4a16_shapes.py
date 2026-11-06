# GPU=-1
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shape presets and bandwidth math for scripts/benchmark_w4a16_shapes.py."""

import importlib.util
import sys
from pathlib import Path

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


def test_gbps_counts_weight_bytes_only(bench):
    """One millisecond means weight_bytes / 1e-3 seconds, reported in GB/s."""
    shape = bench.Shape("toy", 4096, 1024)
    weight_bytes = 4096 * 1024 * bench.BITS / 8
    assert bench.gbps(shape, ms=1.0) == pytest.approx(weight_bytes / 1e-3 / 1e9)


def test_gbps_scales_inversely_with_time(bench):
    shape = bench.Shape("toy", 4096, 1024)
    assert bench.gbps(shape, ms=0.5) == pytest.approx(2 * bench.gbps(shape, ms=1.0))
