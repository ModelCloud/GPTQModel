# GPU=-1
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Timing-iteration contract for scripts/benchmark_ada_kernels.py.

``_timed_us`` partitions its samples into even (hot) and odd (L2-flushed)
iterations and reports a median of each. A run with fewer than two iterations
therefore has no flushed statistic to report, which used to surface as
``StatisticsError: no median for empty data`` after the benchmark had already
accepted the flag.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_ada_kernels.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("benchmark_ada_kernels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeL2:
    """Records how often the flush actually ran."""

    def __init__(self) -> None:
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1


def _fake_cuda(bench, monkeypatch, *, ms: float = 2.0) -> None:
    """Replace the CUDA timing surface with a deterministic clock."""

    class FakeEvent:
        def __init__(self, enable_timing: bool = False) -> None:
            self._ms = 0.0

        def record(self) -> None:
            self._ms = ms

        def elapsed_time(self, other) -> float:
            return self._ms

    monkeypatch.setattr(bench, "torch", SimpleNamespace(
        inference_mode=bench.torch.inference_mode,
        cuda=SimpleNamespace(
            synchronize=lambda device=None: None,
            Event=FakeEvent,
        ),
    ))


@pytest.mark.parametrize("iters", ["0", "1", "-3"])
def test_argparse_rejects_fewer_than_two_iterations(bench, monkeypatch, iters):
    """The flag must be rejected at the boundary, before any GPU work."""
    monkeypatch.setattr(sys, "argv", ["benchmark_ada_kernels.py", "--iters", iters])

    with pytest.raises(SystemExit) as excinfo:
        bench.parse_args()

    assert excinfo.value.code == 2


def test_argparse_accepts_two_iterations(bench, monkeypatch):
    """Two iterations is the smallest count that yields both medians."""
    monkeypatch.setattr(sys, "argv", ["benchmark_ada_kernels.py", "--iters", "2"])

    assert bench.parse_args().iters == 2


def test_argparse_rejects_a_negative_warmup(bench, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_ada_kernels.py", "--warmup", "-1"])

    with pytest.raises(SystemExit) as excinfo:
        bench.parse_args()

    assert excinfo.value.code == 2


def test_one_iteration_with_a_flush_names_the_missing_statistic(bench):
    """A direct caller must get a clear error, not ``StatisticsError``."""
    with pytest.raises(ValueError, match="at least two iterations"):
        bench._timed_us("cuda:0", lambda: None, warmup=0, iters=1, l2=_FakeL2())


def test_zero_iterations_is_rejected(bench):
    with pytest.raises(ValueError, match="at least one iteration"):
        bench._timed_us("cuda:0", lambda: None, warmup=0, iters=0, l2=None)


def test_two_iterations_report_hot_and_flushed_medians(bench, monkeypatch):
    """The flush runs on odd iterations only, and both medians are populated."""
    _fake_cuda(bench, monkeypatch, ms=2.0)
    l2 = _FakeL2()

    hot, flushed = bench._timed_us("cuda:0", lambda: None, warmup=0, iters=2, l2=l2)

    assert hot == pytest.approx(2000.0)      # 2.0 ms reported as microseconds
    assert flushed == pytest.approx(2000.0)
    assert l2.flushes == 1


def test_hot_only_timing_needs_a_single_iteration(bench, monkeypatch):
    """Without an L2 flush there is no odd/even split to satisfy."""
    _fake_cuda(bench, monkeypatch, ms=0.5)

    hot, flushed = bench._timed_us("cuda:0", lambda: None, warmup=0, iters=1, l2=None)

    assert hot == pytest.approx(500.0)
    assert flushed is None
