# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Metric math, metric configs and graded thresholds.

A metric that cannot return exactly zero when both inputs are the same tensor is
measuring something other than their difference. These tests pin that case first,
then the behaviours each metric exists for:

- a constant logit shift keeps every argmax and must still be visible (RMSE),
- more noise must move KLD, TOP_N and RMSE in one direction,
- grades are evaluated in order, so a value can satisfy only one.
"""

import re

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.measure import (
    GradedThresholdConfig,
    KLDConfig,
    MeasureConfig,
    MeasurementReport,
    Metric,
    MetricGrades,
    PPLConfig,
    RMSEConfig,
    TOP_NConfig,
    fuse,
    kl_divergence,
    logit_rmse,
    mean_negative_log_likelihood,
    topk_overlap,
)


def _logits(batch=2, sequence=5, vocab=16, seed=0):
    return torch.randn(batch, sequence, vocab, generator=torch.Generator().manual_seed(seed))


def _grades(pass_bound="<=0.02", warn_bound="<=0.05"):
    return GradedThresholdConfig(
        grades={MetricGrades.PASS: pass_bound, MetricGrades.WARN: warn_bound,
                MetricGrades.FAIL: "fallthrough"}
    )


def test_identity_is_exact():
    logits = _logits()
    assert kl_divergence(logits, logits.clone()) == 0.0
    assert logit_rmse(logits, logits.clone()) == 0.0
    for width in (1, 3, 16):
        assert topk_overlap(logits, logits.clone(), k=width) == 1.0


def test_negative_log_likelihood_matches_cross_entropy():
    logits = _logits()
    ids = torch.randint(0, logits.shape[-1], (2, 5), generator=torch.Generator().manual_seed(1))

    total, count = mean_negative_log_likelihood(logits, ids)

    shifted = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
    targets = ids[:, 1:].reshape(-1)
    expected = F.cross_entropy(shifted, targets, reduction="sum")
    assert count == 2 * 4
    assert total == pytest.approx(float(expected), rel=1e-6)


def test_ignore_index_drops_positions():
    logits = _logits()
    ids = torch.randint(0, logits.shape[-1], (2, 5), generator=torch.Generator().manual_seed(2))
    ids[0, 1:] = -100

    total, count = mean_negative_log_likelihood(logits, ids, ignore_index=-100)

    assert count == 4
    assert total > 0


def test_more_noise_moves_every_metric_the_same_way():
    reference = _logits(seed=3)
    small = reference + 0.05 * _logits(seed=4)
    large = reference + 0.5 * _logits(seed=5)

    assert kl_divergence(reference, small) < kl_divergence(reference, large)
    assert topk_overlap(reference, small, k=1) > topk_overlap(reference, large, k=1)
    assert logit_rmse(reference, small) < logit_rmse(reference, large)


def test_a_constant_shift_is_invisible_to_top_n_but_not_to_rmse():
    reference = _logits(seed=6)
    shifted = reference + 5.0

    assert topk_overlap(reference, shifted, k=1) == 1.0
    assert logit_rmse(reference, shifted) == pytest.approx(5.0, rel=1e-6)
    # A constant shift cancels inside softmax: that is why RMSE exists.
    assert abs(kl_divergence(reference, shifted)) < 1e-6


def test_topk_overlap_counts_partial_matches():
    reference = torch.tensor([[[5.0, 4.0, 3.0, 2.0, 1.0]]])
    candidate = torch.tensor([[[4.0, 5.0, 3.0, 1.0, 2.0]]])

    assert topk_overlap(reference, candidate, k=1) == 0.0
    assert topk_overlap(reference, candidate, k=3) == 1.0
    assert topk_overlap(reference, candidate, k=4) == pytest.approx(0.75)


def test_metric_guards():
    reference = _logits(vocab=8)
    with pytest.raises(ValueError, match="shapes differ"):
        kl_divergence(reference, reference[:, :3, :])
    with pytest.raises(ValueError, match="shapes differ"):
        logit_rmse(reference, reference[:, :3, :])
    with pytest.raises(ValueError, match="exceeds the vocabulary"):
        topk_overlap(reference, reference, k=9)
    with pytest.raises(ValueError, match="k must be positive"):
        topk_overlap(reference, reference, k=0)
    with pytest.raises(ValueError, match="temperature"):
        kl_divergence(reference, reference, temperature=0.0)
    with pytest.raises(ValueError, match="at least two positions"):
        mean_negative_log_likelihood(reference[:, :1, :], torch.zeros(2, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="must be \\[batch, sequence, vocab\\]"):
        kl_divergence(reference[0], reference[0])
    with pytest.raises(ValueError, match="reference logit RMS of zero"):
        logit_rmse(torch.zeros_like(reference), torch.zeros_like(reference), normalize=True)


def test_measure_config_takes_one_metric_or_many():
    single = MeasureConfig(metrics=KLDConfig(positions=8))
    assert [type(item).__name__ for item in single.metrics] == ["KLDConfig"]

    many = MeasureConfig(metrics=(Metric.PPL, TOP_NConfig(top_k=(1, 5)), "KLD"))
    assert [type(item).__name__ for item in many.metrics] == [
        "PPLConfig",
        "TOP_NConfig",
        "KLDConfig",
    ]
    assert many.config_for(Metric.TOP_N).top_k == (1, 5)
    assert many.config_for(Metric.RMSE) is None


def test_measure_config_rejects_ambiguous_runs():
    with pytest.raises(ValueError, match="names PPL twice"):
        MeasureConfig(metrics=(Metric.PPL, Metric.PPL))
    with pytest.raises(ValueError, match="unknown metric"):
        MeasureConfig(metrics=("PPL2",))
    with pytest.raises(ValueError, match="at least one metric"):
        MeasureConfig(metrics=())
    with pytest.raises(ValueError, match="max_batches"):
        MeasureConfig(max_batches=0)
    with pytest.raises(ValueError, match="batch_size"):
        MeasureConfig(batch_size=0)
    with pytest.raises(ValueError, match="window"):
        PPLConfig(window=1)
    with pytest.raises(ValueError, match="stride"):
        PPLConfig(stride=-1)
    with pytest.raises(ValueError, match="positions"):
        KLDConfig(positions=0)
    with pytest.raises(ValueError, match="positive"):
        TOP_NConfig(top_k=(1, 0))
    with pytest.raises(ValueError, match="at least one width"):
        TOP_NConfig(top_k=())


def test_grades_are_ordered_and_first_match_wins():
    threshold = _grades()
    assert threshold.grade_for(0.01) is MetricGrades.PASS
    assert threshold.grade_for(0.02) is MetricGrades.PASS      # inclusive
    assert threshold.grade_for(0.03) is MetricGrades.WARN
    assert threshold.grade_for(0.05) is MetricGrades.WARN
    assert threshold.grade_for(0.06) is MetricGrades.FAIL      # fallthrough


def test_grade_config_supports_floor_expressions():
    floor = GradedThresholdConfig(
        grades={MetricGrades.PASS: ">=0.99", MetricGrades.WARN: ">=0.95"}
    )
    assert floor.grade_for(0.995) is MetricGrades.PASS
    assert floor.grade_for(0.96) is MetricGrades.WARN
    assert floor.grade_for(0.5) is MetricGrades.FAIL


def test_grade_config_rejects_nonsense():
    with pytest.raises(ValueError, match="unknown grade"):
        GradedThresholdConfig(grades={"GREAT": "<=0.1"})
    with pytest.raises(ValueError, match="fallthrough"):
        GradedThresholdConfig(grades={MetricGrades.FAIL: "<=0.1"})
    with pytest.raises(ValueError, match="must start with one of"):
        GradedThresholdConfig(grades={MetricGrades.PASS: "0.02"})
    with pytest.raises(ValueError, match="is not a number"):
        GradedThresholdConfig(grades={MetricGrades.PASS: "<=abc"})
    with pytest.raises(ValueError, match="at least a PASS or WARN"):
        GradedThresholdConfig(grades={MetricGrades.FAIL: "fallthrough"})


def test_thresholds_live_on_the_metric_configs():
    config = MeasureConfig(
        metrics=(KLDConfig(threshold=_grades()), PPLConfig()),
    )
    thresholds = config.thresholds()
    assert list(thresholds) == ["KLD"]
    assert thresholds["KLD"].grade_for(0.01) is MetricGrades.PASS


def test_metrics_sharing_a_pass_are_fused():
    groups = fuse([PPLConfig(), KLDConfig(), TOP_NConfig(top_k=(1, 5)), RMSEConfig()])
    assert len(groups) == 1
    kinds, members = groups[0]
    assert kinds == ("forward",)
    assert len(members) == 4


def test_a_metric_with_its_own_capture_gets_its_own_pass():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _WeightMetric:
        capture_needs: frozenset = frozenset({"weights"})

    groups = fuse([PPLConfig(), _WeightMetric()])
    assert len(groups) == 2


def test_report_grades_and_verdicts():
    pass_report = MeasurementReport(
        values={"KLD": 0.01}, grades={"KLD": MetricGrades.PASS},
    )
    assert pass_report.overall is MetricGrades.PASS
    assert pass_report.passed is True
    assert "PASS" in pass_report.markdown()

    warn_report = MeasurementReport(values={"KLD": 0.03}, grades={"KLD": MetricGrades.WARN})
    assert warn_report.passed is False
    assert warn_report.warnings == {"KLD": 0.03}
    assert warn_report.violations == {}

    fail_report = MeasurementReport(values={"KLD": 0.2}, grades={"KLD": MetricGrades.FAIL})
    assert fail_report.passed is False
    assert fail_report.violations == {"KLD": 0.2}
    assert fail_report.to_json()["overall"] == "FAIL"


def test_report_without_grades_is_not_a_pass():
    report = MeasurementReport(values={"PPL": 4.1})
    assert report.overall is None
    assert report.passed is None
    assert "no thresholds configured" in report.markdown()
    assert "no thresholds configured" in str(report)
    assert "|" not in str(report)


def test_str_renders_aligned_logbar_columns():
    report = MeasurementReport(
        values={"PPL": 22.85, "KLD": 0.1183, "RMSE": 0.6487, "TOP_N": {1: 0.8242, 5: 0.8352}},
        grades={"KLD": MetricGrades.WARN, "TOP_N": MetricGrades.FAIL},
        reference={"PPL": 22.22},
        batches=4,
        positions=64,
    )

    rendered = str(report)
    assert rendered == report.render()
    assert "|" not in rendered
    assert "measured on 4 batch(es), 64 position(s)" in rendered
    assert rendered.splitlines()[-1] == "result: fail"

    table = rendered.split("\n\n")[0].splitlines()
    assert table[0].split() == ["metric", "value", "reference", "grade"]
    # Every cell starts at the same offset on every line, header through last row.
    starts = [[match.start() for match in re.finditer(r"\S+", line)] for line in table]
    assert all(row == starts[0] for row in starts)
    assert [re.split(r" {2,}", line.strip()) for line in table[2:]] == [
        ["PPL", "22.85", "22.22", "-"],
        ["KLD", "0.1183", "-", "WARN"],
        ["RMSE", "0.6487", "-", "-"],
        ["TOP_N@1", "0.8242", "-", "FAIL"],
        ["TOP_N@5", "0.8352", "-", "FAIL"],
    ]


def test_markdown_stays_a_github_table():
    report = MeasurementReport(
        values={"PPL": 22.85, "TOP_N": {1: 0.8242}},
        grades={"TOP_N": MetricGrades.FAIL},
        reference={"PPL": 22.22},
    )

    markdown = report.markdown().splitlines()
    assert markdown[0] == "| metric | value | reference | grade |"
    assert markdown[1] == "| --- | --- | --- | --- |"
    assert markdown[2] == "| PPL | 22.85 | 22.22 | - |"
    assert markdown[3] == "| TOP_N@1 | 0.8242 | - | FAIL |"
    assert markdown[-1] == "result: fail"
