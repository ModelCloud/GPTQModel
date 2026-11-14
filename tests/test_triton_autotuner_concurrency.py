# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Autotuner scratch state must be per-thread, not shared on the kernel instance."""

import os
import sys
import threading

import pytest

from gptqmodel.nn_modules.triton_utils.custom_autotune import CustomizedTritonAutoTuner
from gptqmodel.utils import nogil_patcher


def _require_free_threaded():
    if not hasattr(sys, "_is_gil_enabled"):
        pytest.skip("Python runtime does not expose _is_gil_enabled; free-threading build required")
    if sys._is_gil_enabled():
        if os.environ.get("PYTHON_GIL") == "0":
            raise AssertionError("PYTHON_GIL=0 did not disable the GIL")
        pytest.skip("GIL is enabled - run with PYTHON_GIL=0 to exercise the autotuner race")


class _Arg:
    def __init__(self, label):
        self.label = label
        self.dtype = "float32"

    def __eq__(self, other):
        return isinstance(other, _Arg) and self.label == other.label

    def __hash__(self):
        return hash(self.label)


class _Config:
    pre_hook = None

    def __init__(self, name):
        self.name = name
        self.num_warps = 1
        self.num_stages = 1
        self.kwargs = {}

    def all_kwargs(self):
        return {}


class _FakeFn:
    def run(self, *args, **kwargs):
        return args


def _run_race(tuner, a_in, b_in, release_a, release_b, observed):
    errors = []

    def call(label):
        try:
            tuner.run(_Arg(label))
        except BaseException as exc:  # pragma: no cover - diagnostic path
            errors.append(exc)

    first = threading.Thread(target=call, args=("A",), daemon=True)
    first.start()
    assert a_in.wait(10), "thread A never entered its benchmark"

    second = threading.Thread(target=call, args=("B",), daemon=True)
    second.start()
    assert b_in.wait(10), "thread B never entered its benchmark"

    release_a.set()
    first.join(10)
    assert not first.is_alive(), "thread A did not finish"
    assert observed["A"] and set(observed["A"]) == {"A"}, (
        f"thread A benchmarked with {observed['A']!r}; another thread overwrote its nargs"
    )

    release_b.set()
    second.join(10)
    assert not second.is_alive(), "thread B did not finish"
    assert observed["B"] and set(observed["B"]) == {"B"}
    assert errors == []


def _blocking_bench(observed, a_in, b_in, release_a, release_b, configs):
    def bench(self, *args, config, **kwargs):
        label = self.nargs["value"].label
        observed[label].append(label)
        if label == "A":
            a_in.set()
            assert release_a.wait(10)
        else:
            b_in.set()
            assert release_b.wait(10)
        return 1.0 if config is configs[0] else 2.0

    return bench


def test_nogil_patch_keeps_autotuner_nargs_thread_local(monkeypatch):
    _require_free_threaded()
    triton_autotuner = pytest.importorskip("triton.runtime.autotuner")
    configs = [_Config("a"), _Config("b")]
    a_in, b_in = threading.Event(), threading.Event()
    release_a, release_b = threading.Event(), threading.Event()
    observed = {"A": [], "B": []}

    class _FakeAutotuner:
        def __init__(self):
            self.arg_names = ["value"]
            self.keys = ["value"]
            self.configs = configs
            self.cache = {}
            self.cache_results = False
            self.fn = _FakeFn()
            self.bench_time = None
            self.configs_timings = None

        def prune_configs(self, kwargs):
            return self.configs

        def pre_hook(self, nargs, reset_only):
            pass

    _FakeAutotuner._bench = _blocking_bench(
        observed, a_in, b_in, release_a, release_b, configs
    )

    monkeypatch.setattr(triton_autotuner, "Autotuner", _FakeAutotuner)
    monkeypatch.setattr(triton_autotuner, "CacheFuture", None, raising=False)
    monkeypatch.setattr(nogil_patcher, "version", lambda _package_name: "3.7.0")
    nogil_patcher.patch_triton_autotuner()

    _run_race(_FakeAutotuner(), a_in, b_in, release_a, release_b, observed)


def test_custom_autotuner_keeps_nargs_thread_local():
    _require_free_threaded()
    configs = [_Config("a"), _Config("b")]
    a_in, b_in = threading.Event(), threading.Event()
    release_a, release_b = threading.Event(), threading.Event()
    observed = {"A": [], "B": []}

    class _RaceTuner(CustomizedTritonAutoTuner):
        def prune_configs(self, kwargs, nargs=None):
            return self.configs

    _RaceTuner._bench = _blocking_bench(
        observed, a_in, b_in, release_a, release_b, configs
    )

    tuner = _RaceTuner(_FakeFn(), ["value"], configs, ["value"], None)
    _run_race(tuner, a_in, b_in, release_a, release_b, observed)
