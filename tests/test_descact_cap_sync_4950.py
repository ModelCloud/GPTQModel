# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #4950 (per-``add_batch`` sink-row-cap syncs).

#4909 added the ``desc_act`` sink-row cap (#4742) in
``GPTQ.resolve_sink_row_cap_scale``, which ``compute_hessian_xtx`` called on
*every* ``add_batch``. Resolving the cap forced four device->host reads per
call (``bool(isfinite(...).all())``, ``bool(isfinite(threshold))``,
``float(threshold)`` and ``bool((scale >= 1).all())``) and, whenever the cap
applied, moved the ``X^T X`` GEMM off the fp32 fast path onto the leased
chunk-workspace path. ``add_batch`` runs inside the layer forward hooks, so
every read drained the CUDA queue mid-calibration and ``desc_act=True``
quantization became ~2x slower (Qwen2.5-0.5B: 38 s -> 82 s).

These tests pin the fix without needing a GPU:

* the per-``add_batch`` path performs no host reads and no chunk lease;
* the cap is still decided from the batch being accumulated, so the Hessian is
  bit-identical to ``main``'s for one batch and for a batch sequence (a
  "resolve it once and cache it" shortcut is rejected: on Qwen2.5-0.5B it
  changed the quantized checkpoint, wikitext PPL 20.78 vs 20.32);
* the first batch is still capped exactly as #4742 requires.
"""

import math

import torch

from gptqmodel.quantization.gptq import GPTQ
from tests.test_descact_sink_4742 import (
    ROWS,
    SINK_CHANNELS,
    SINK_ROWS,
    _fixture,
    _task,
)


# Four ``add_batch`` calls on ``main`` cost 16 host reads (four per call). The
# fix does the whole cap on device, so the count is zero; the small allowance
# only keeps unrelated library changes from failing this regression.
MAX_HOST_READS_PER_MODULE = 2

_HOST_READ_METHODS = ("item", "__bool__", "__float__", "__int__")


def _count_host_reads(task: GPTQ, activations: torch.Tensor, batches: int):
    """Run ``batches`` ``add_batch`` calls and return the host-read method names."""

    reads = []
    originals = {name: getattr(torch.Tensor, name) for name in _HOST_READ_METHODS}

    def _recorder(name, original):
        def wrapper(self, *args, **kwargs):
            reads.append(name)
            return original(self, *args, **kwargs)

        return wrapper

    try:
        for name, original in originals.items():
            setattr(torch.Tensor, name, _recorder(name, original))
        for _ in range(batches):
            task.add_batch(activations, None)
    finally:
        for name, original in originals.items():
            setattr(torch.Tensor, name, original)
    return reads


def _median_row_norm(activations: torch.Tensor) -> float:
    norms = torch.linalg.vector_norm(activations.to(torch.float32), dim=1)
    return float(torch.kthvalue(norms, (activations.shape[0] + 1) // 2).values)


def test_descact_cap_makes_no_host_reads_per_add_batch():
    """``compute_hessian_xtx`` must stay on device once the cap is resolved."""

    activations, weight = _fixture()

    task = _task(weight, desc_act=True)
    reads = _count_host_reads(task, activations, batches=4)
    assert len(reads) <= MAX_HOST_READS_PER_MODULE, (
        "desc_act sink-row cap read the host on every add_batch: "
        f"{len(reads)} reads for 4 batches ({reads})"
    )

    # The count must not grow with the number of batches (i.e. it is per
    # module, not per add_batch).
    task_many = _task(weight, desc_act=True)
    reads_many = _count_host_reads(task_many, activations, batches=12)
    assert len(reads_many) <= MAX_HOST_READS_PER_MODULE, (
        "desc_act sink-row cap reads scale with add_batch calls: "
        f"{len(reads_many)} reads for 12 batches ({reads_many})"
    )
    assert len(reads_many) <= len(reads), (
        "host reads increased with more add_batch calls: "
        f"{len(reads)} for 4 vs {len(reads_many)} for 12"
    )


def test_descact_cap_decision_tracks_the_current_batch(monkeypatch):
    """The cap must be decided from the batch being accumulated.

    Caching the threshold on the first batch (the obvious way to skip the
    per-batch sort) changes the Hessian for later batches: on Qwen2.5-0.5B it
    moved wikitext PPL from 20.32 to 20.78. This pins the per-batch decision.
    """

    activations, weight = _fixture()
    generator = torch.Generator().manual_seed(11)
    batches = []
    for _ in range(3):
        batch = torch.randn(activations.shape, generator=generator) * 0.1
        for row, channels in zip(SINK_ROWS, SINK_CHANNELS):
            batch[row, list(channels)] *= 700.0
        batches.append(batch)

    task = _task(weight, desc_act=True)
    seen = []
    original = GPTQ.resolve_sink_row_cap_threshold

    def recording(self, matrix, chunk_size):
        threshold = original(self, matrix, chunk_size)
        seen.append(threshold)
        return threshold

    monkeypatch.setattr(GPTQ, "resolve_sink_row_cap_threshold", recording)

    for batch in batches:
        task.add_batch(batch, None)
    actual = task.finalize_hessian()

    # One decision per batch, and each is a fresh scalar for that batch.
    assert len(seen) == len(batches)
    thresholds = [float(threshold) for threshold in seen]
    assert len(set(thresholds)) > 1, f"the cap reused one stale threshold: {thresholds}"

    # ``main`` accumulates raw per-batch ``X^T X`` and normalizes once at
    # finalize (``2 / total_samples``), so the reference must do the same to
    # stay bitwise. The capture accumulates in the GEMM epilogue (#5731), so
    # the reference uses ``addmm_`` for the same fp32 rounding.
    expected = torch.zeros_like(actual)
    for batch, threshold in zip(batches, seen):
        norms = torch.linalg.vector_norm(batch.to(torch.float32), dim=1)
        scale = torch.clamp(
            threshold / norms.clamp_min(torch.finfo(torch.float32).tiny), max=1.0
        )
        clamped = batch.to(torch.float32) * scale.unsqueeze(1)
        expected.addmm_(clamped.T, clamped)
    expected.mul_(2.0 / float(len(batches) * ROWS))

    assert torch.equal(actual, expected), (
        "the capped Hessian does not match main's per-batch cap: "
        f"max abs diff {float((actual - expected).abs().max())}"
    )


def test_descact_capped_batch_stays_on_the_fp32_fast_path(monkeypatch):
    """A capped batch must not be routed through the chunk-workspace lease."""

    activations, weight = _fixture()
    task = _task(weight, desc_act=True)

    leases = []
    original = GPTQ.borrow_materialized_chunk_fp32

    def counting(self, chunk, rows):
        leases.append(int(rows))
        return original(self, chunk, rows)

    monkeypatch.setattr(GPTQ, "borrow_materialized_chunk_fp32", counting)

    task.add_batch(activations, None)

    assert leases == [], (
        "capped desc_act batch left the fp32 matmul fast path for the chunk "
        f"workspace: {leases}"
    )


def test_descact_cap_applies_on_the_first_batch():
    """The #4742 cap must still be in force from the very first batch."""

    activations, weight = _fixture()
    task = _task(weight, desc_act=True)
    task.add_batch(activations, None)
    capped = task.finalize_hessian()

    uncapped_task = _task(weight, desc_act=True, cap_multiple=0)
    uncapped_task.add_batch(activations, None)
    uncapped = uncapped_task.finalize_hessian()

    threshold = 4.0 * _median_row_norm(activations)
    sink_channels = [channel for channels in SINK_CHANNELS for channel in channels]
    bound = 2.05 * threshold ** 2 * 2.0 / ROWS

    assert float(capped.diagonal()[sink_channels].max()) <= bound
    assert float(uncapped.diagonal()[sink_channels].max()) > 50.0 * bound

    # The first batch must have cached a scale that actually caps the sinks.
    scale = task.resolve_sink_row_cap_scale(activations, None)
    assert scale is not None
    sink_rows = torch.tensor(SINK_ROWS)
    assert bool((scale[sink_rows] < 1.0).all())


def test_descact_capped_hessian_matches_the_main_formula_bitwise():
    """The cached fast path must reproduce main's capped Hessian exactly."""

    activations, weight = _fixture()
    task = _task(weight, desc_act=True)
    task.add_batch(activations, None)
    actual = task.finalize_hessian()

    # This is exactly what ``main`` computes per batch: fp32 row norms,
    # ``min(p99.9, 4 x median)``, then ``X^T X`` over the clamped rows.
    norms = torch.linalg.vector_norm(activations.to(torch.float32), dim=1)
    median = torch.kthvalue(norms, (ROWS + 1) // 2).values
    rank = max(1, min(ROWS, math.ceil(99.9 / 100.0 * ROWS)))
    percentile = torch.kthvalue(norms, rank).values
    threshold = torch.minimum(percentile, median * 4.0)
    scale = torch.clamp(
        threshold / norms.clamp_min(torch.finfo(torch.float32).tiny), max=1.0
    )
    clamped = activations.to(torch.float32) * scale.unsqueeze(1)
    expected = (clamped.T @ clamped) * (2.0 / ROWS)

    assert torch.equal(actual, expected), (
        "capped Hessian diverged from the pre-#4950 formula: "
        f"max abs diff {float((actual - expected).abs().max())}"
    )
