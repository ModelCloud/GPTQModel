"""CPU algebra checks; these do not establish model quality."""

import math

import torch

from scripts.p32_twenty.scorecard import effective_bpw, layer_metrics, logits_metrics


def test_scorecard_algebra():
    x = torch.arange(1, 21, dtype=torch.float64).reshape(2, 10)
    m = layer_metrics(x, x)
    assert m["mean_abs"] == m["max_abs"] == m["relative_l2"] == 0
    assert abs(m["cosine"] - 1) < 1e-14 and m["local_tolerance_pass"]
    # A tiny mean does not excuse a single over-limit element.
    a = torch.zeros(1000, dtype=torch.float64)
    a[0] = 0.05
    assert not layer_metrics(a, torch.zeros_like(a))["local_tolerance_pass"]
    assert layer_metrics(a, torch.zeros_like(a))["relative_l2"] is None
    # A constant logit offset must not change distributions or rankings.
    m = logits_metrics(x + 100, x)
    assert abs(m["kl_teacher_candidate"]) < 1e-12
    assert all(m[f"top{k}_agreement"] == 1 for k in (1, 5, 10))
    # Opposite rankings over a uniform-sized vocabulary give disjoint top five.
    m = logits_metrics(-x, x)
    assert m["kl_teacher_candidate"] > 0
    assert m["top1_agreement"] == m["top5_agreement"] == 0 and m["top10_agreement"] == 1
    assert (
        effective_bpw({"packed": 128, "metadata": 16, "padding": 16}, 256)[
            "effective_bpw"
        ]
        == 5
    )
    for f in (layer_metrics, logits_metrics):
        try:
            f(x * math.nan, x)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite values must fail")
    print("Scorecard CPU algebra checks passed; no model-quality evidence.")
