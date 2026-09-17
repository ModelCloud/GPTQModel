# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

import pytest


def _new_eval_case():
    # Keep the integration class out of this module's pytest collection. The
    # real test loads the official 9B checkpoint; these checks stay CPU-only.
    from test_zdtaichu5 import TestZDTaichu5

    return TestZDTaichu5(methodName="test_zdtaichu5")


def test_zdtaichu5_eval_uses_measured_native_arc_baseline():
    case = _new_eval_case()
    native_results = {
        "arc_challenge": {"acc,none": 0.51, "acc_norm,none": 0.57}
    }
    calls = []

    def native_baseline():
        assert case._in_model_compat_eval_flow()
        return native_results

    def quantize_and_evaluate():
        calls.append((case.EVAL_TASKS_SLOW, case.EVAL_TASKS_FAST))

    case._get_current_native_eval_results = native_baseline
    case.quantize_and_evaluate = quantize_and_evaluate
    case.test_zdtaichu5()

    assert len(calls) == 1
    slow_tasks, fast_tasks = calls[0]
    assert slow_tasks["arc_challenge"]["acc"]["value"] == 0.51
    assert slow_tasks["arc_challenge"]["acc_norm"]["value"] == 0.57
    assert slow_tasks["arc_challenge"]["acc"]["floor_pct"] == 0.04
    assert slow_tasks["arc_challenge"]["acc_norm"]["floor_pct"] == 0.04
    assert fast_tasks["arc_challenge"]["acc"]["value"] == 0.51
    assert fast_tasks["arc_challenge"]["acc_norm"]["value"] == 0.57


@pytest.mark.parametrize(
    "native_results",
    [
        {"arc_challenge": {"acc,none": 0.51}},
        {
            "arc_challenge": {
                "acc,none": math.nan,
                "acc_norm,none": 0.57,
            }
        },
        {
            "arc_challenge": {
                "acc,none": 0.51,
                "acc_norm,none": 0.0,
            }
        },
    ],
)
def test_zdtaichu5_eval_rejects_invalid_native_arc_baseline(native_results):
    case = _new_eval_case()
    quantized = []
    case._get_current_native_eval_results = lambda: native_results
    case.quantize_and_evaluate = lambda: quantized.append(True)

    with pytest.raises(AssertionError):
        case.test_zdtaichu5()
    assert quantized == []
