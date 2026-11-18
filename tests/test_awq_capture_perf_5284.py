# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AWQ capture-time costs on a MoE run (#5284).

Two per-capture costs dominated a 2-layer MoE AWQ run after #5236 removed the
buffer copy: the aggregation policy was rebuilt from the model's ``ModulePlan``
on every capture (51k calls, 3.4 s), and every routed-expert capture ran a
concat/topk/mask merge even when none of its rows could survive (8.6 s).

The policy is memoized per ``(class, module_name)`` and a full reservoir now
short-circuits a capture that cannot displace a kept row. Both must leave the
captured rows exactly as before.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor


def _processor() -> AWQProcessor:
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.lock = threading.Lock()
    processor._batch_tls = threading.local()
    processor._mask_tls = threading.local()
    processor.tasks = {}
    processor._reservoir_locks = {}
    processor._token_row_plans = {}
    return processor


class _CountingModel:
    """A stand-in model whose aggregation selector counts its calls."""

    def __init__(self, policies=None):
        self.calls = []
        self.policies = policies or {}

    def awq_input_feature_aggregation(self, module_name):
        self.calls.append(module_name)
        return self.policies.get(module_name, {"mode": "token_rows"})


# ---------------------------------------------------------------------------
# Aggregation policy memoization
# ---------------------------------------------------------------------------


def test_policy_is_resolved_once_per_module():
    processor = _processor()
    processor.gptq_model = _CountingModel()

    for _ in range(50):
        for name in ("mlp.experts.1.gate_proj", "self_attn.q_proj"):
            processor._feature_aggregation_policy(name)

    assert processor.gptq_model.calls == ["mlp.experts.1.gate_proj", "self_attn.q_proj"]


def test_memoized_policy_is_handed_out_as_a_copy():
    processor = _processor()
    processor.gptq_model = _CountingModel({"root": {"mode": "token_rows", "capture_root": True}})

    first = processor._feature_aggregation_policy("root")
    first["capture_root"] = False
    first["mode"] = "mutated"

    second = processor._feature_aggregation_policy("root")
    assert second == {"mode": "token_rows", "capture_root": True}
    assert second is not first


def test_memo_follows_a_replaced_model():
    processor = _processor()
    first_model = _CountingModel()
    processor.gptq_model = first_model
    assert processor._feature_aggregation_policy("mlp") == {"mode": "token_rows"}

    second_model = _CountingModel({"mlp": {"mode": "token_rows", "max_tokens": 128}})
    processor.gptq_model = second_model
    assert processor._feature_aggregation_policy("mlp") == {
        "mode": "token_rows",
        "max_tokens": 128,
    }
    assert second_model.calls == ["mlp"]


def test_replacing_the_selector_invalidates_the_memo():
    """A patched selector must not be answered from the previous one's cache."""
    processor = _processor()
    model = _CountingModel()
    processor.gptq_model = model
    assert processor._feature_aggregation_policy("mlp") == {"mode": "token_rows"}

    model.awq_input_feature_aggregation = lambda name: {"mode": "token_rows", "max_tokens": 64}

    assert processor._feature_aggregation_policy("mlp") == {
        "mode": "token_rows",
        "max_tokens": 64,
    }


def test_non_mapping_policy_still_raises():
    processor = _processor()
    processor.gptq_model = SimpleNamespace(awq_input_feature_aggregation=lambda name: "token_rows")

    with pytest.raises(TypeError, match="must return a mapping or None"):
        processor._feature_aggregation_policy("mlp")


def test_model_without_a_selector_returns_none():
    processor = _processor()
    processor.gptq_model = SimpleNamespace()

    assert processor._feature_aggregation_policy("mlp") is None


# ---------------------------------------------------------------------------
# Reservoir merge short-circuit
# ---------------------------------------------------------------------------

CAP = 8


def _reference_merge(entry, rows, priorities, *, source_index, budget_cap, row_indices=None):
    """The pre-#5284 merge body, kept as the equivalence oracle."""
    buffer = entry.get("reservoir_buffer")
    if buffer is None or buffer.shape[0] != budget_cap or buffer.shape[1] != rows.shape[1]:
        buffer = rows.new_empty((budget_cap, rows.shape[1]))
        entry["reservoir_buffer"] = buffer
        entry["reservoir_priorities"] = torch.empty((budget_cap,), dtype=torch.float64)
        entry["reservoir_sources"] = torch.empty((budget_cap,), dtype=torch.int64)
        entry["reservoir_rows"] = torch.empty((budget_cap,), dtype=torch.int64)
        entry["reservoir_used"] = 0
    priority_buffer = entry["reservoir_priorities"]
    source_buffer = entry["reservoir_sources"]
    row_buffer = entry["reservoir_rows"]
    used = int(entry.get("reservoir_used", 0))
    count = int(rows.shape[0])
    if row_indices is None:
        row_indices = torch.arange(count, dtype=torch.int64)
    if used + count <= budget_cap:
        buffer[used:used + count] = rows
        priority_buffer[used:used + count] = priorities
        source_buffer[used:used + count] = source_index
        row_buffer[used:used + count] = row_indices
        used += count
    else:
        combined = torch.cat([priority_buffer[:used], priorities])
        keep = torch.topk(combined, budget_cap, largest=False).indices
        kept_old = keep[keep < used]
        kept_new = keep[keep >= used] - used
        kept_old_mask = torch.zeros((used,), dtype=torch.bool)
        kept_old_mask[kept_old] = True
        slots = torch.cat(
            [
                (~kept_old_mask).nonzero(as_tuple=False).flatten(),
                torch.arange(used, budget_cap),
            ]
        )
        buffer[slots] = rows.index_select(0, kept_new)
        priority_buffer[slots] = priorities.index_select(0, kept_new)
        source_buffer[slots] = source_index
        row_buffer[slots] = row_indices.index_select(0, kept_new)
        used = budget_cap
    entry["reservoir_used"] = used
    entry["inputs"] = [buffer[:used].unsqueeze(0)]
    entry.pop("provenance", None)


def _fill(processor, entry, rows, priorities, *, budget_cap=CAP):
    processor._merge_reservoir_rows(
        entry, rows, priorities, source_index=0, budget_cap=budget_cap
    )


def _snapshot(entry):
    return (
        entry["reservoir_buffer"].clone(),
        entry["reservoir_priorities"].clone(),
        entry["reservoir_sources"].clone(),
        entry["reservoir_rows"].clone(),
        torch.tensor(int(entry["reservoir_used"])),
    )


def _full_entry(processor, *, rows=CAP, seed=0):
    generator = torch.Generator().manual_seed(seed)
    entry = {}
    _fill(
        processor,
        entry,
        torch.randn(rows, 4, generator=generator),
        torch.rand(rows, dtype=torch.float64, generator=generator),
    )
    return entry


def test_a_losing_capture_skips_the_merge(monkeypatch):
    processor = _processor()
    entry = _full_entry(processor)
    before = _snapshot(entry)

    calls = []
    real_topk = torch.topk
    monkeypatch.setattr(torch, "topk", lambda *a, **k: calls.append(a) or real_topk(*a, **k))

    worst_kept = entry["reservoir_priorities"][: entry["reservoir_used"]].max()
    losing = worst_kept + torch.arange(1, 4, dtype=torch.float64)
    processor._merge_reservoir_rows(
        entry, torch.randn(3, 4), losing, source_index=9, budget_cap=CAP
    )

    assert calls == []
    after = _snapshot(entry)
    for left, right in zip(before, after):
        assert torch.equal(left, right)


def test_a_winning_capture_still_merges(monkeypatch):
    processor = _processor()
    entry = _full_entry(processor)
    calls = []
    real_topk = torch.topk
    monkeypatch.setattr(torch, "topk", lambda *a, **k: calls.append(a) or real_topk(*a, **k))

    winner = torch.zeros(1, 4)
    processor._merge_reservoir_rows(
        entry, winner, torch.zeros(1, dtype=torch.float64), source_index=7, budget_cap=CAP
    )

    assert len(calls) == 1
    # The survivor lands in the slot of the row it displaced, not slot 0.
    survivors = (entry["reservoir_sources"] == 7).nonzero(as_tuple=False).flatten()
    assert survivors.numel() == 1
    assert torch.equal(entry["reservoir_buffer"][int(survivors[0])], winner[0])
    assert int(entry["reservoir_used"]) == CAP


def test_tied_priorities_take_the_merge_path(monkeypatch):
    processor = _processor()
    entry = _full_entry(processor)
    calls = []
    real_topk = torch.topk
    monkeypatch.setattr(torch, "topk", lambda *a, **k: calls.append(a) or real_topk(*a, **k))

    worst_kept = entry["reservoir_priorities"][: entry["reservoir_used"]].max()
    processor._merge_reservoir_rows(
        entry, torch.randn(1, 4), worst_kept.reshape(1), source_index=3, budget_cap=CAP
    )

    assert len(calls) == 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_merge_matches_the_previous_algorithm(seed):
    """Random captures, some losing: the kept rows must be identical."""
    generator = torch.Generator().manual_seed(seed)
    fast, reference = {}, {}
    processor = _processor()

    for capture in range(40):
        count = int(torch.randint(1, 5, (1,), generator=generator))
        rows = torch.randn(count, 3, generator=generator)
        priorities = torch.rand(count, dtype=torch.float64, generator=generator)
        processor._merge_reservoir_rows(
            fast, rows, priorities, source_index=capture, budget_cap=CAP
        )
        _reference_merge(
            reference, rows, priorities, source_index=capture, budget_cap=CAP
        )

    for fast_value, reference_value in zip(_snapshot(fast), _snapshot(reference)):
        assert torch.equal(fast_value, reference_value)


def test_merge_preserves_row_indices_and_provenance():
    processor = _processor()
    entry = {}
    processor._merge_reservoir_rows(
        entry,
        torch.randn(CAP, 2),
        torch.linspace(0.0, 1.0, CAP, dtype=torch.float64),
        source_index=0,
        budget_cap=CAP,
        row_indices=torch.arange(CAP, dtype=torch.int64) * 3,
    )
    before = _snapshot(entry)

    worst_kept = entry["reservoir_priorities"][: entry["reservoir_used"]].max()
    processor._merge_reservoir_rows(
        entry,
        torch.randn(2, 2),
        worst_kept + torch.tensor([1.0, 2.0], dtype=torch.float64),
        source_index=1,
        budget_cap=CAP,
        row_indices=torch.tensor([11, 12], dtype=torch.int64),
    )

    for left, right in zip(before, _snapshot(entry)):
        assert torch.equal(left, right)
