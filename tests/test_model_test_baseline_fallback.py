# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
from pathlib import Path

import pytest

from gptqmodel import BACKEND


sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from model_test import ModelTest  # noqa: E402


class _BaselineFallbackHarness(ModelTest):
    NATIVE_MODEL_ID = "/tmp/native-model"
    LOAD_BACKEND = BACKEND.TORCH
    DISABLE_NATIVE_BASELINE_FALLBACK = False
    EVAL_TASKS = {
        "arc_challenge": {
            "acc": {
                "value": 0.30,
                "floor_pct": 0.05,
                "ceil_pct": 0.10,
            },
        },
    }

    def __init__(self, native_results):
        super().__init__(methodName="runTest")
        self._stub_native_results = native_results
        self.evaluate_model_calls = 0

    def _model_test_mode(self) -> str:
        return self.MODEL_TEST_MODE_SLOW

    def evaluate_model(self, *args, **kwargs):  # pragma: no cover - exercised via check_results
        self.evaluate_model_calls += 1
        return self._stub_native_results


def test_check_results_accepts_current_native_baseline_when_static_value_is_stale():
    harness = _BaselineFallbackHarness(
        {"arc_challenge": {"acc,none": 0.26}},
    )

    harness.check_results({"arc_challenge": {"acc,none": 0.255}})

    assert harness.evaluate_model_calls == 1


def test_check_results_still_fails_when_quantized_result_misses_current_native_baseline():
    harness = _BaselineFallbackHarness(
        {"arc_challenge": {"acc,none": 0.22}},
    )

    with pytest.raises(AssertionError):
        harness.check_results({"arc_challenge": {"acc,none": 0.255}})

    assert harness.evaluate_model_calls == 1
