# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""The MoE AWQ capture path must not pay per-capture setup costs (#5284).

#5236 removed the copy of the whole kept buffer from the reservoir merge, and
#5295 memoized the aggregation policy and skipped captures that cannot enter.
Two costs remained on the routed-expert path:

* the merge still concatenated the kept priorities with every capture, ran a
  full ``topk`` and wrote every surviving captured row into a slot;
* every routed expert kept the layer-wide row budget, so the clip search (and
  the replay forward) ran on ~5x the rows the pre-#5175 per-expert budget used.

These tests pin both contracts without a GPU or a model checkpoint.
"""

import threading
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import CalibrationConfig


def _processor(*, max_tokens=None, hidden=4, layer_inputs=None):
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.lock = threading.Lock()
    processor._batch_tls = threading.local()
    processor._mask_tls = threading.local()
    processor.tasks = {}
    processor._reservoir_locks = {}
    processor._capture_host_cache = {}
    processor._capture_sampled_cache = {}
    processor._token_row_plans = {}
    processor._capture_host_batch = None
    processor._module_forward_kwargs = {}
    processor._feature_task_names = set()
    processor._feature_stats = {}
    processor._awq_feature_kwargs = {}
    processor._scale_feature_by_module = {}
    processor._nsamples_total = 1
    processor.inputs_cache = SimpleNamespace(
        layer_inputs=layer_inputs,
        attention_masks=[],
        position_ids=[],
        layer_input_kwargs=[],
    )
    processor.calibration = CalibrationConfig(exclude_attention_sinks=False)
    policy = {"mode": "token_rows"}
    if max_tokens is not None:
        policy["max_tokens"] = max_tokens
    processor.gptq_model = SimpleNamespace(
        awq_input_feature_aggregation=lambda name: dict(policy)
    )
    return processor


# ---------------------------------------------------------------------------
# 1. The incremental merge is exactly the reference bottom-k reservoir
# ---------------------------------------------------------------------------


def _reference_merge(kept_priorities, kept_rows, priorities, rows, cap):
    combined = torch.cat([kept_priorities, priorities])
    order = torch.argsort(combined, stable=True)[:cap]
    return combined[order], torch.cat([kept_rows, rows])[order]


@pytest.mark.parametrize("growing_cap", [False, True])
def test_incremental_merge_matches_the_reference_bottom_k(growing_cap):
    """The merge must keep the same rows the full sort would (#5284)."""

    processor = AWQProcessor.__new__(AWQProcessor)
    for trial in range(8):
        generator = torch.Generator().manual_seed(1000 + trial)
        hidden = 3
        cap = int(torch.randint(3, 12, (1,), generator=generator))
        entry = {}
        ref_priorities = torch.empty(0, dtype=torch.float64)
        ref_rows = torch.empty((0, hidden))
        for step in range(8):
            count = int(torch.randint(1, 20, (1,), generator=generator))
            priorities = torch.rand(count, generator=generator, dtype=torch.float64)
            # Encode the priority in the row so row/priority pairing is checkable.
            rows = priorities.unsqueeze(1).expand(count, hidden).contiguous()
            if growing_cap and step:
                cap = min(cap + int(torch.randint(0, 4, (1,), generator=generator)), 24)

            processor._merge_reservoir_rows(
                entry,
                rows.clone(),
                priorities.clone(),
                source_index=step,
                budget_cap=cap,
            )
            ref_priorities, ref_rows = _reference_merge(
                ref_priorities, ref_rows, priorities.clone(), rows.clone(), cap
            )

            used = int(entry["reservoir_used"])
            kept = entry["reservoir_priorities"][:used]
            assert used == ref_priorities.numel()
            assert torch.equal(torch.sort(kept).values, torch.sort(ref_priorities).values)
            assert torch.equal(entry["reservoir_buffer"][:used, 0], kept)


def test_full_reservoir_without_a_survivor_keeps_the_buffer_untouched():
    """A capture that cannot enter the sample must not rewrite it (#5284)."""

    processor = AWQProcessor.__new__(AWQProcessor)
    entry = {}
    generator = torch.Generator().manual_seed(7)
    hidden = 3
    cap = 8
    for step in range(4):
        rows = torch.randn(4, hidden, generator=generator)
        priorities = torch.rand(4, generator=generator, dtype=torch.float64)
        processor._merge_reservoir_rows(
            entry, rows, priorities, source_index=step, budget_cap=cap
        )

    buffer = entry["reservoir_buffer"]
    kept_rows = buffer[:cap].clone()
    kept_priorities = entry["reservoir_priorities"][:cap].clone()
    # Every new priority loses to the largest kept one, so nothing can enter.
    losing = torch.full((4,), float(kept_priorities.max()) + 1.0, dtype=torch.float64)
    processor._merge_reservoir_rows(
        entry,
        torch.randn(4, hidden, generator=generator),
        losing,
        source_index=99,
        budget_cap=cap,
    )

    assert entry["reservoir_buffer"] is buffer
    assert torch.equal(entry["reservoir_buffer"][:cap], kept_rows)
    assert torch.equal(entry["reservoir_priorities"][:cap], kept_priorities)


# ---------------------------------------------------------------------------
# 2. A routed expert sizes its reservoir from its own captures
# ---------------------------------------------------------------------------


def test_routed_expert_reservoir_uses_its_own_capture_budget():
    """Not the layer-wide plan budget, which was ~5x the pre-#5175 budget."""

    hidden = 4
    processor = _processor(hidden=hidden)
    name = "model.layers.0.mlp.experts.3.gate_proj"

    for batch in range(6):
        processor._set_current_batch_index(batch)
        processor._record_input_feature(name, torch.randn(1, 5, hidden))

    entry = processor.tasks[name]
    # Six captures of five rows: the pre-#5175 rule is max(largest, captures).
    assert int(entry["max_tokens"]) == 6
    assert int(entry["reservoir_used"]) == 6
    assert entry["inputs"][0].shape == (1, 6, hidden)


def test_configured_max_tokens_still_wins():
    """An explicit policy budget is not overridden by the capture counts."""

    hidden = 4
    processor = _processor(max_tokens=32, hidden=hidden)
    name = "model.layers.0.mlp.experts.0.up_proj"

    for batch in range(3):
        processor._set_current_batch_index(batch)
        processor._record_input_feature(name, torch.randn(1, 5, hidden))

    entry = processor.tasks[name]
    assert int(entry["max_tokens"]) == 32
    assert int(entry["reservoir_used"]) == 15


def test_growing_budget_carries_the_kept_rows_instead_of_dropping_them():
    """A larger capture grows the buffer; it must not reset the reservoir.

    The pre-#5284 resize path rebuilt the buffer and re-seeded it from
    ``inputs``, but the provenance list only exists at replay time, so every
    row captured before the growth was silently lost.
    """

    processor = AWQProcessor.__new__(AWQProcessor)
    hidden = 3
    entry = {}

    # One row with the best possible priority, so it must survive any growth.
    processor._merge_reservoir_rows(
        entry,
        torch.zeros(1, hidden),
        torch.zeros(1, dtype=torch.float64),
        source_index=0,
        budget_cap=1,
    )
    assert int(entry["reservoir_used"]) == 1

    # A much larger capture raises the budget to 40; the earlier row is still
    # the lowest priority and must be carried into the new buffer.
    priorities = torch.linspace(0.5, 1.0, 40, dtype=torch.float64)
    rows = priorities.unsqueeze(1).expand(40, hidden).contiguous()
    processor._merge_reservoir_rows(
        entry, rows, priorities, source_index=1, budget_cap=40
    )

    used = int(entry["reservoir_used"])
    assert used == 40
    kept = entry["reservoir_priorities"][:used]
    assert float(kept.min()) == 0.0
    assert entry["reservoir_sources"][:used].eq(0).sum().item() == 1
    assert torch.equal(entry["reservoir_buffer"][:used, 0], kept)
