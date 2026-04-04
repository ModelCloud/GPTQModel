# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.utils.cpp import (
    _compile_progress_ratio,
    _compile_progress_step,
    _compile_progress_subtitle,
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
    assert "baseline ~117s" in subtitle
    assert "(+3.8s)" in subtitle
