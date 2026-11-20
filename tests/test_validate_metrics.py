# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""The divergence metrics must be exact for identical models.

A metric that cannot return exactly zero when both models are the same model is
not measuring the difference between them - it is measuring something else
(usually token streams or positions that drifted apart). These tests pin that
case first, then the behaviours each metric exists for:

- a constant logit shift keeps every argmax and must still be visible (RMSE),
- more noise must move KLD, TOP_N and RMSE in one direction,
- PPL is the sum/count form, so batches accumulate without weighting mistakes.
"""

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.validate import (
    KLDConfig,
    Metric,
    PPLConfig,
    TOP_NConfig,
    ValidateConfig,
    ValidationReport,
    kl_divergence,
    logit_rmse,
    mean_negative_log_likelihood,
    topk_overlap,
)


def _logits(batch=2, sequence=5, vocab=16, seed=0):
    return torch.randn(batch, sequence, vocab, generator=torch.Generator().manual_seed(seed))


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

    assert count == 4  # only the second row's predictions count
    assert total > 0


def test_more_noise_moves_every_metric_the_same_way():
    reference = _logits(seed=3)
    small = reference + 0.05 * _logits(seed=4)
    large = reference + 0.5 * _logits(seed=5)

    assert kl_divergence(reference, small) < kl_divergence(reference, large)
    assert topk_overlap(reference, small, k=1) > topk_overlap(reference, large, k=1)
    assert logit_rmse(reference, small) < logit_rmse(reference, large)


def test_a_constant_shift_is_invisible_to_top_n_but_not_to_rmse():
    """The case RMSE exists for: argmax unchanged, logits moved."""

    reference = _logits(seed=6)
    shifted = reference + 5.0

    assert topk_overlap(reference, shifted, k=1) == 1.0
    assert logit_rmse(reference, shifted) == pytest.approx(5.0, rel=1e-6)
    # A constant shift cancels inside softmax, so KLD is zero up to float
    # rounding: that is exactly why the RMSE metric exists.
    assert abs(kl_divergence(reference, shifted)) < 1e-6


def test_topk_overlap_counts_partial_matches():
    reference = torch.tensor([[[5.0, 4.0, 3.0, 2.0, 1.0]]])
    candidate = torch.tensor([[[4.0, 5.0, 3.0, 1.0, 2.0]]])  # same top-3 set, reordered

    assert topk_overlap(reference, candidate, k=1) == 0.0
    assert topk_overlap(reference, candidate, k=3) == 1.0
    assert topk_overlap(reference, candidate, k=4) == pytest.approx(0.75)


def test_metric_guards():
    reference = _logits(vocab=8)
    with pytest.raises(ValueError, match="shapes differ"):
        kl_divergence(reference, reference[:, :3, :])
    with pytest.raises(ValueError, match="shapes differ"):
        logit_rmse(reference, reference[:, :3, :])
    with pytest.raises(ValueError, match="shapes differ"):
        topk_overlap(reference, reference[:, :3, :], k=1)
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


def test_config_accepts_names_and_configs_together():
    config = ValidateConfig(metrics=(Metric.PPL, TOP_NConfig(top_k=(1, 5)), "KLD"))

    assert [type(item).__name__ for item in config.metrics] == [
        "PPLConfig",
        "TOP_NConfig",
        "KLDConfig",
    ]
    assert config.config_for(Metric.TOP_N).top_k == (1, 5)
    assert config.config_for(Metric.RMSE) is None


def test_config_rejects_ambiguous_runs():
    with pytest.raises(ValueError, match="names PPL twice"):
        ValidateConfig(metrics=(Metric.PPL, Metric.PPL))
    with pytest.raises(ValueError, match="unknown metric"):
        ValidateConfig(metrics=("PPL2",))
    with pytest.raises(ValueError, match="that metric was not requested"):
        ValidateConfig(metrics=(Metric.PPL,), thresholds={"KLD": 0.02})
    with pytest.raises(ValueError, match="at least one metric"):
        ValidateConfig(metrics=())
    with pytest.raises(ValueError, match="max_batches"):
        ValidateConfig(max_batches=0)
    with pytest.raises(ValueError, match="batch_size"):
        ValidateConfig(batch_size=0)
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
    with pytest.raises(TypeError, match="must be a number"):
        ValidateConfig(metrics=(Metric.KLD,), thresholds={"KLD": "0.02"})


def test_report_distinguishes_ceiling_and_floor_thresholds():
    ceiling = ValidationReport(values={"KLD": 0.01}, thresholds={"KLD": 0.02})
    assert ceiling.passed is True
    assert ceiling.violations == {}

    floor = ValidationReport(values={"TOP_N": {1: 0.9, 5: 0.99}}, thresholds={"TOP_N": 0.95})
    assert floor.passed is False
    assert floor.violations == {"TOP_N": 0.9}
    assert "**no**" in floor.markdown()
    assert "yes" in floor.markdown()


def test_report_without_thresholds_is_not_a_pass():
    report = ValidationReport(values={"PPL": 4.1})
    assert report.passed is None
    assert "no thresholds configured" in report.markdown()
    assert report.to_json()["passed"] is None
