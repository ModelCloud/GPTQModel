# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from scripts.compare_qvq_activation_quality import (
    _A8_CONTRACT,
    _ArmAccumulator,
    _checkpoint_contract,
    _prediction_rows,
)


def test_activation_quality_uses_shifted_nonpadding_next_token_positions():
    logits = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    input_ids = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    rows, labels = _prediction_rows(logits, input_ids, attention_mask)

    torch.testing.assert_close(
        rows,
        torch.stack((logits[0, 0], logits[0, 1], logits[1, 0])),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(labels, torch.tensor([1, 2, 1]), rtol=0, atol=0)


def test_activation_quality_accumulator_reports_dense_identity_and_perplexity():
    logits = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ]
    )
    labels = torch.tensor([0, 4])
    dense_logp = torch.log_softmax(logits, dim=-1)
    dense_top = logits.topk(5, dim=-1).indices
    accumulator = _ArmAccumulator(dense_reference=True)

    accumulator.add(
        logits=logits,
        labels=labels,
        dense_logp=dense_logp,
        dense_top10=dense_top,
    )
    result = accumulator.result()

    expected_nll = torch.nn.functional.cross_entropy(logits, labels).item()
    assert result["mean_nll"] == pytest.approx(expected_nll)
    assert result["perplexity"] == pytest.approx(
        torch.exp(torch.tensor(expected_nll)).item()
    )
    assert result["kld_dense_to_arm_nats"]["mean"] == 0
    assert result["dense_top1_agreement"] == 1
    assert result["dense_top5_overlap"] == 1
    assert result["dense_top10_overlap"] == 1
    assert result["next_token_top1_accuracy"] == 1


def test_activation_quality_accumulator_reports_candidate_divergence_and_overlap():
    dense = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]])
    candidate = torch.tensor([[4.0, 9.0, 8.0, 7.0, 6.0, 5.0]])
    labels = torch.tensor([0])
    accumulator = _ArmAccumulator()

    accumulator.add(
        logits=candidate,
        labels=labels,
        dense_logp=torch.log_softmax(dense, dim=-1),
        dense_top10=dense.topk(6, dim=-1).indices,
    )
    result = accumulator.result()

    assert result["kld_dense_to_arm_nats"]["mean"] > 0
    assert result["dense_top1_agreement"] == 0
    assert result["dense_top5_overlap"] == pytest.approx(0.8)
    assert result["dense_top10_overlap"] == 1
    assert result["next_token_top1_accuracy"] == 0
    assert result["next_token_top5_accuracy"] == 0
    assert result["next_token_top10_accuracy"] == 1


def test_activation_quality_requires_explicit_experimental_replay_contract(tmp_path):
    checkpoint = tmp_path / "a8"
    checkpoint.mkdir()
    activation = {**_A8_CONTRACT, "replay_passes": 1}
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "method": "qvq",
                    "bits": 3.5,
                    "format": "qvq_v2b2_p32",
                    "activation": activation,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required A8 contract"):
        _checkpoint_contract(checkpoint, expect_a8=True)
    result = _checkpoint_contract(checkpoint, expect_a8=True, replay_passes=1)
    assert result["quantization_config"]["activation"] == activation
