# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from scripts.validate_qvq_p4_live_prefix import _localized_summary


def test_localized_summary_preserves_original_yaqa_and_module_search_diagnostics():
    result = SimpleNamespace(
        yaqa_spectral_candidates={
            "r16_a1_t7_s2": {
                "rank": 16,
                "alpha": 1.0,
                "tile": 7,
                "segment": 2,
                "loss": 12.5,
                "relative_improvement": 0.0075,
                "search_loss": 6.25,
                "search_relative_improvement": 0.0125,
                "replay_score": 0.9988,
                "selected": True,
            }
        },
        yaqa_spectral_selector_churn=0.125,
        yaqa_spectral_family_changed=False,
    )

    summary = _localized_summary(result, {"baseline": {}, "proposal": {}, "accepted": True})

    assert summary["replayed_candidates"] == [
        {
            "name": "r16_a1_t7_s2",
            "generator": "spectral_push",
            "rank": 16,
            "alpha": 1.0,
            "tile": 7,
            "segment": 2,
            "original_yaqa_loss": 12.5,
            "original_yaqa_relative_improvement": 0.0075,
            "module_search_loss": 6.25,
            "module_search_relative_improvement": 0.0125,
            "propagated_first_order": None,
            "replay_score": 0.9988,
            "selected": True,
        }
    ]
