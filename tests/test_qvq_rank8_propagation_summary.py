import math
from pathlib import Path

import pytest
import torch


def test_paired_document_summary_keeps_token_weighting(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_qvq_rank8_propagation import paired_summary

    rows = []
    for tokens in (1, 10, 100):
        rows.append({
            "teacher": {"tokens": tokens, "nll_sum": 0.5 * tokens},
            "fast": {"tokens": tokens, "nll_sum": tokens, "kl_sum": 0.1 * tokens, "top1_matches": tokens},
            "quality": {"tokens": tokens, "nll_sum": 0.9 * tokens, "kl_sum": 0.1 * tokens, "top1_matches": tokens},
        })
    summary = paired_summary(rows, samples=1000)
    assert summary["tokens"] == 111
    assert summary["modes"]["quality"]["perplexity"] == pytest.approx(math.exp(0.9))
    assert summary["quality_minus_fast"]["nll_sum"]["ci95"] == pytest.approx([-0.1, -0.1])
    assert summary["quality_minus_fast"]["kl_sum"]["ci95"] == [0, 0]
    rows[0]["quality"]["tokens"] = 2
    with pytest.raises(ValueError, match="identical prediction tokens"):
        paired_summary(rows)


def test_logit_metrics_exposes_low_temperature_margin_and_topk(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_qvq_rank8_propagation import logit_metrics, paired_summary

    torch.manual_seed(37)
    target = torch.tensor([[1, 2, 3, 4]])
    reference = torch.randn(1, 4, 40)
    candidate = reference + torch.randn_like(reference) * 0.01
    teacher = logit_metrics(reference, target)
    quality = logit_metrics(candidate, target, reference)
    assert teacher["tokens"] == quality["tokens"] == 3
    assert quality["kl_low_temp_sum"] >= 0
    assert quality["margin_abs_error_sum"] >= 0
    assert 0 <= quality["top32_overlap_sum"] <= 3 * 32
    rows = [{"teacher": teacher, "fast": quality, "quality": quality}]
    summary = paired_summary(rows, samples=1000)
    assert "low_temperature_kl_per_token" in summary["modes"]["quality"]
    assert "margin_absolute_error" in summary["modes"]["quality"]
    assert "top32_agreement" in summary["modes"]["quality"]
    assert "top32_overlap_sum" in summary["quality_minus_fast"]
