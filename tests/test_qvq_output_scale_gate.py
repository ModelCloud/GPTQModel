# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from scripts.analyze_qvq_output_scale_gate import (
    _build_parser,
    _cluster_bootstrap_delta,
    _device_unavailable_reason,
    _summary_row,
)


def test_output_scale_gate_bootstrap_resamples_prompts_and_preserves_paired_token_weighting():
    baseline = torch.tensor([1.0, 3.0, 10.0])
    candidate = torch.tensor([0.0, 2.0, 13.0])
    prompt_ids = torch.tensor([0, 0, 1])

    first = _cluster_bootstrap_delta(baseline, candidate, prompt_ids, seed=17, samples=2048)
    second = _cluster_bootstrap_delta(baseline, candidate, prompt_ids, seed=17, samples=2048)

    assert first == second
    assert first["delta"] == pytest.approx(1 / 3)
    assert first["prompt_wins"] == 1
    assert first["prompt_ties"] == 0
    assert first["prompt_count"] == 2
    assert first["bootstrap_samples"] == 2048
    assert first["ci95_low"] <= first["delta"] <= first["ci95_high"]


def test_output_scale_gate_summary_exports_local_live_layer_and_all_paired_intervals():
    scalar_metric = {
        "relative_l2": 0.25,
        "rmse": 0.5,
        "kl_forward": {"mean": 0.125, "p95": 0.25},
    }
    result = {
        "modules": {
            "proj": {
                "weight": {"relative_l2": 0.1},
                "local": scalar_metric,
                "live": scalar_metric,
                "scale_diagnostics": {"optimized_channels": 4},
            }
        },
        "layers": {"layer.0": scalar_metric},
        "logits": {
            "kl_forward": {"mean": 0.2, "p95": 0.4},
            "jensen_shannon": {"mean": 0.05},
            "relative_l2": 0.3,
            "sqnr_db": 10.0,
            "top1_agreement": 0.9,
            "top5_overlap": {"mean": 0.95},
        },
    }
    interval = {"delta": -0.01, "ci95_low": -0.02, "ci95_high": -0.001, "prompt_wins": 100}
    paired = {"alpha-half": dict.fromkeys(("kl_forward", "jensen_shannon", "top1", "top5_overlap"), interval)}

    row = _summary_row("w1", "alpha-half", result, paired)

    assert row["mean_local_relative_l2"] == 0.25
    assert row["mean_live_rmse"] == 0.5
    assert row["mean_layer_kl"] == 0.125
    for metric in ("kl_forward", "jensen_shannon", "top1", "top5_overlap"):
        assert row[f"{metric}_delta_vs_baseline"] == -0.01
        assert row[f"{metric}_ci95_low"] == -0.02
        assert row[f"{metric}_ci95_high"] == -0.001

    baseline = _summary_row("w1", "baseline", result, paired)
    assert baseline["jensen_shannon_delta_vs_baseline"] is None


def test_output_scale_gate_parser_exposes_cuda_and_runtime_check(monkeypatch):
    device_action = next(action for action in _build_parser()._actions if action.dest == "device")
    assert tuple(device_action.choices) == ("cuda", "mps", "cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _device_unavailable_reason("cuda") == "CUDA is unavailable"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _device_unavailable_reason("cuda") is None


@pytest.mark.parametrize(
    ("baseline", "candidate", "prompt_ids", "message"),
    (
        (torch.ones(2), torch.ones(1), torch.zeros(2, dtype=torch.long), "matching shapes"),
        (torch.ones(2), torch.ones(2), torch.zeros(1, dtype=torch.long), "matching shapes"),
    ),
)
def test_output_scale_gate_bootstrap_rejects_mismatched_geometry(baseline, candidate, prompt_ids, message):
    with pytest.raises(ValueError, match=message):
        _cluster_bootstrap_delta(baseline, candidate, prompt_ids, seed=0, samples=10)


@pytest.mark.parametrize(
    ("baseline", "candidate", "prompt_ids", "seed", "samples", "error", "message"),
    (
        (torch.ones(1, 1), torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.long), 0, 10, ValueError, "rank-1"),
        (torch.ones(1, dtype=torch.long), torch.ones(1), torch.zeros(1, dtype=torch.long), 0, 10, TypeError, "floating"),
        (torch.ones(1), torch.ones(1), torch.zeros(1), 0, 10, TypeError, "integer dtype"),
        (torch.tensor([float("nan")]), torch.ones(1), torch.zeros(1, dtype=torch.long), 0, 10, ValueError, "finite"),
        (torch.ones(1), torch.ones(1), torch.tensor([-1]), 0, 10, ValueError, "nonnegative"),
        (torch.ones(2), torch.ones(2), torch.tensor([0, 2]), 0, 10, ValueError, "contiguous"),
        (torch.ones(1), torch.ones(1), torch.zeros(1, dtype=torch.long), True, 10, TypeError, "seed"),
        (torch.ones(1), torch.ones(1), torch.zeros(1, dtype=torch.long), 0, 0, ValueError, "positive"),
    ),
)
def test_output_scale_gate_bootstrap_fails_closed_on_invalid_cluster_contract(
    baseline, candidate, prompt_ids, seed, samples, error, message
):
    with pytest.raises(error, match=message):
        _cluster_bootstrap_delta(baseline, candidate, prompt_ids, seed=seed, samples=samples)
