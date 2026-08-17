# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from scripts.validate_qvq_p4_live_prefix import (
    _fixed_v2b2_yaqa_family,
    _localized_summary,
)


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


def test_fixed_v2b2_family_overrides_the_outer_selection_contract():
    captured = {}

    def original(*args, **kwargs):
        captured.update(kwargs)
        return args

    result = _fixed_v2b2_yaqa_family(original, 3)(
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        (torch.empty(1),) * 4,
        family_mode="reselect",
        sample_strategy="64_16x16",
    )

    assert len(result) == 4
    assert captured["family_mode"] == "fixed_block_ldlq"
    assert captured["sample_strategy"] == "full"
    assert captured["block_family_id"] == 3


def test_fixed_v2b2_family_zero_emits_exact_canonical_shape(monkeypatch):
    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    states = torch.arange(5, dtype=torch.int64)

    def canonical(*args, **kwargs):
        del args, kwargs
        return weight, states

    monkeypatch.setattr("scripts.validate_qvq_p4_live_prefix.qvq_module.yaqa_inner", canonical)
    actual_weight, actual_states, selectors, inactive_family = _fixed_v2b2_yaqa_family(lambda: None, 0)(
        weight,
        torch.eye(4),
        torch.eye(4),
        (torch.empty(1),) * 4,
        bits=2,
    )

    assert actual_weight is weight
    assert actual_states is states
    assert selectors.shape == (40,)
    assert selectors.dtype == torch.uint8
    assert torch.count_nonzero(selectors) == 0
    assert inactive_family.item() == 1
