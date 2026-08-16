# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from scripts.analyze_qvq_e2e_alignment import (
    _batches,
    _encoded_provenance,
    _evaluate,
    _floating_tensor_dtypes,
    _logit_digest,
    _parameter_state,
    _passes_accuracy_gate,
    _restore_floating_tensor_dtypes,
    _restore_parameter_state,
    _valid_next_token_logits,
)


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("stored_logits", logits)

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.stored_logits[: input_ids.shape[0], : input_ids.shape[1]])


def test_qvq_e2e_batching_next_token_mask_and_exact_digest_exclude_padding():
    encoded = {
        "input_ids": torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
    }
    batches = list(_batches(encoded, 1))
    assert len(batches) == 2
    assert batches[0]["input_ids"].shape == (1, 2)
    assert batches[1]["input_ids"].shape == (1, 3)

    left_padded = {
        "input_ids": torch.tensor([[0, 0, 1, 2], [0, 3, 4, 5]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
    }
    left_batch = next(_batches(left_padded, 2))
    torch.testing.assert_close(left_batch["input_ids"], torch.tensor([[0, 1, 2], [3, 4, 5]]))
    torch.testing.assert_close(left_batch["attention_mask"], torch.tensor([[0, 1, 1], [1, 1, 1]]))

    with pytest.raises(ValueError, match="no valid tokens"):
        next(_batches({"input_ids": torch.zeros(1, 3), "attention_mask": torch.zeros(1, 3)}, 1))

    logits = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    valid = _valid_next_token_logits(logits, encoded["attention_mask"])
    torch.testing.assert_close(valid, torch.stack((logits[0, 0], logits[1, 0], logits[1, 1])))
    trimmed = next(_batches(encoded, 2))
    trimmed_valid = _valid_next_token_logits(logits[:, : trimmed["input_ids"].shape[1]], trimmed["attention_mask"])
    torch.testing.assert_close(trimmed_valid, valid)

    changed_padding = logits.clone()
    changed_padding[0, 1:].add_(10_000)
    changed_padding[1, 2:].add_(10_000)
    digest = _logit_digest(_FixedLogitModel(logits), encoded, batch_size=2, device=torch.device("cpu"))
    changed_digest = _logit_digest(
        _FixedLogitModel(changed_padding),
        encoded,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert changed_digest == digest


def test_qvq_e2e_metrics_and_dtype_rollback_are_exact():
    encoded = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    model = _FixedLogitModel(torch.randn(1, 3, 8, dtype=torch.float16))
    metrics = _evaluate(model, model, encoded, batch_size=1, device=torch.device("cpu"))

    assert metrics == {
        "valid_tokens": 2.0,
        "forward_kld": 0.0,
        "jensen_shannon": 0.0,
        "top1_agreement": 1.0,
        "top5_overlap": 1.0,
        "top10_overlap": 1.0,
    }

    provenance = _encoded_provenance(encoded, offset=32)
    assert provenance["row_start"] == 32
    assert provenance["row_end_exclusive"] == 33
    assert provenance["valid_tokens"] == 3
    assert provenance["minimum_tokens_per_row"] == 3
    assert provenance["maximum_tokens_per_row"] == 3
    assert len(provenance["token_contract_sha256"]) == 64

    original = _floating_tensor_dtypes(model)
    model.float()
    assert model.stored_logits.dtype == torch.float32
    _restore_floating_tensor_dtypes(original)
    assert model.stored_logits.dtype == torch.float16


def test_qvq_e2e_parameter_state_is_exact_and_rejects_scope_drift():
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    parameters = [first, second]
    state = _parameter_state(parameters)
    first.data.zero_()
    second.data.fill_(9)

    _restore_parameter_state(parameters, state)
    torch.testing.assert_close(first, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(second, torch.tensor([3.0]))

    with pytest.raises(ValueError, match="count changed"):
        _restore_parameter_state(parameters[:1], state)


def test_qvq_e2e_accuracy_gate_uses_one_exact_finite_population():
    baseline = {
        "valid_tokens": 32.0,
        "forward_kld": 0.4,
        "jensen_shannon": 0.1,
        "top1_agreement": 0.7,
        "top5_overlap": 0.8,
        "top10_overlap": 0.85,
    }
    candidate = {
        "valid_tokens": 32.0,
        "forward_kld": 0.3,
        "jensen_shannon": 0.09,
        "top1_agreement": 0.7,
        "top5_overlap": 0.81,
        "top10_overlap": 0.86,
    }
    assert _passes_accuracy_gate(baseline, candidate)

    for metric in ("forward_kld", "jensen_shannon"):
        assert not _passes_accuracy_gate(baseline, candidate | {metric: baseline[metric]})
    for metric in ("top1_agreement", "top5_overlap", "top10_overlap"):
        assert not _passes_accuracy_gate(baseline, candidate | {metric: baseline[metric] - 0.01})

    with pytest.raises(ValueError, match="same valid-token population"):
        _passes_accuracy_gate(baseline, candidate | {"valid_tokens": 31.0})
    for baseline_metrics, candidate_metrics in (
        (baseline | {"valid_tokens": 0.0}, candidate),
        (baseline, candidate | {"valid_tokens": 0.0}),
    ):
        with pytest.raises(ValueError, match="positive valid-token population"):
            _passes_accuracy_gate(baseline_metrics, candidate_metrics)
    for baseline_metrics, candidate_metrics in (
        (baseline | {"forward_kld": float("nan")}, candidate),
        (baseline, candidate | {"forward_kld": float("nan")}),
    ):
        with pytest.raises(ValueError, match="non-finite"):
            _passes_accuracy_gate(baseline_metrics, candidate_metrics)
    for baseline_metrics, candidate_metrics in (
        ({key: value for key, value in baseline.items() if key != "top5_overlap"}, candidate),
        (baseline, {key: value for key, value in candidate.items() if key != "top5_overlap"}),
    ):
        with pytest.raises(ValueError, match="missing acceptance fields"):
            _passes_accuracy_gate(baseline_metrics, candidate_metrics)
