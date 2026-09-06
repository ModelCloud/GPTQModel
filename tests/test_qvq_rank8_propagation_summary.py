import math
from pathlib import Path

import pytest


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
