# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
import time

import pytest

from gptqmodel.utils.cpp import (
    _COMPILE_PROGRESS_TOTAL_STEPS,
    _compile_progress_ratio,
    _compile_progress_step,
    _compile_progress_subtitle,
    _CompileProgressDisplay,
    default_jit_cuda_cflags,
)
from gptqmodel.utils.jit_compile_baselines import get_jit_compile_baseline_seconds


def test_known_jit_compile_baselines_are_recorded():
    assert get_jit_compile_baseline_seconds("gptqmodel_marlin_fp16_ops") == pytest.approx(116.863)
    assert get_jit_compile_baseline_seconds("gptqmodel_awq_ops") == pytest.approx(61.640)
    assert get_jit_compile_baseline_seconds("gptqmodel_paroquant_rotation") == pytest.approx(78.430)


def test_compile_progress_ratio_tracks_baseline_without_hitting_completion():
    assert _compile_progress_ratio(0.0, 120.0) == pytest.approx(0.0)
    assert _compile_progress_ratio(60.0, 120.0) == pytest.approx(0.475)
    assert _compile_progress_ratio(120.0, 120.0) == pytest.approx(0.95)
    assert 0.95 < _compile_progress_ratio(240.0, 120.0) < 0.99


def test_compile_progress_step_never_reaches_final_step_before_completion():
    assert _compile_progress_step(120.0, 120.0, total_steps=100) < 99
    assert _compile_progress_step(600.0, 120.0, total_steps=100) < 99


def test_compile_progress_subtitle_reports_overrun_against_baseline():
    subtitle = _compile_progress_subtitle(120.634, 116.863)
    assert "elapsed 121s" in subtitle
    assert "estimated ~117s" in subtitle
    assert "(+3.8s)" in subtitle


def test_default_jit_cuda_cflags_includes_nvcc_threads_by_default(monkeypatch):
    monkeypatch.delenv("NVCC_THREADS", raising=False)
    flags = default_jit_cuda_cflags()
    assert "--threads" in flags
    assert "8" in flags


def test_default_jit_cuda_cflags_honors_nvcc_threads_override(monkeypatch):
    monkeypatch.setenv("NVCC_THREADS", "16")
    flags = default_jit_cuda_cflags()
    assert "--threads" in flags
    assert "16" in flags


def test_default_jit_cuda_cflags_explicit_nvcc_threads_takes_precedence(monkeypatch):
    monkeypatch.setenv("NVCC_THREADS", "8")
    flags = default_jit_cuda_cflags(nvcc_threads=16)
    assert "--threads" in flags
    assert flags[flags.index("--threads") + 1] == "16"


class _FakeProgress:
    def __init__(self):
        self.current_iter_step = 0
        self._title = ""
        self._subtitle = ""
        self.closed = False

    def manual(self):
        return self

    def set(self, **_kwargs):
        return self

    def title(self, value):
        self._title = value
        return self

    def subtitle(self, value):
        self._subtitle = value
        return self

    def draw(self, force: bool = False):
        return self

    def close(self):
        self.closed = True
        return None


class _FakeSpinner:
    def close(self):
        return None


class _FakeLogger:
    def __init__(self):
        self.progress = _FakeProgress()
        self.spinner_handle = _FakeSpinner()

    def pb(self, _iterable):
        return self.progress

    def spinner(self, **_kwargs):
        return self.spinner_handle


def test_compile_progress_close_completes_immediately_when_build_finishes_early(monkeypatch):
    monkeypatch.setattr("gptqmodel.utils.cpp._COMPILE_PROGRESS_INTERVAL_SECONDS", 60.0)
    logger = _FakeLogger()
    display = _CompileProgressDisplay(
        logger=logger,
        title="Compiling extension: Marlin bf16...",
        baseline_seconds=120.0,
    )

    started = time.perf_counter()
    display.close(succeeded=True, elapsed_seconds=5.0)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5
    assert logger.progress.current_iter_step == _COMPILE_PROGRESS_TOTAL_STEPS
    assert logger.progress.closed is True
    assert logger.progress._subtitle == "elapsed 5.0s / estimated ~120s"


def test_compile_progress_close_does_not_wait_for_refresh_interval_on_failure(monkeypatch):
    monkeypatch.setattr("gptqmodel.utils.cpp._COMPILE_PROGRESS_INTERVAL_SECONDS", 60.0)
    logger = _FakeLogger()
    display = _CompileProgressDisplay(
        logger=logger,
        title="Compiling extension: AWQ...",
        baseline_seconds=60.0,
    )

    started = time.perf_counter()
    display.close(succeeded=False, elapsed_seconds=3.0)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5
    assert logger.progress.current_iter_step < _COMPILE_PROGRESS_TOTAL_STEPS
    assert logger.progress.closed is True
