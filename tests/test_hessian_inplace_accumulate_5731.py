# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Hessian capture must accumulate in place, not materialize a second X^T X (#5731).

``compute_hessian_xtx`` returned a fresh ``columns x columns`` fp32 product for
every calibration batch, and ``add_batch`` added it into the per-device partial
afterwards. Peak capture memory was therefore two Hessians, and every batch
paid one extra memory-bound read-modify-write pass over ``columns ** 2``. The
chunked path did the same once per chunk.

The capture now accumulates in the BLAS epilogue (``addmm_``): the chunked path
writes into its local accumulator, and the unchunked path writes straight into
the existing per-device partial when there is one. Results still match an
independent FP64 ``X^T X`` within fp32 rounding.
"""

from __future__ import annotations

import threading

import torch

from gptqmodel.quantization.config import GPTQConfig, HessianConfig
from gptqmodel.quantization.gptq import GPTQ, release_workspace_cache


COLUMNS = 128
CHUNK = 256
ROWS = 1024


def _task(*, chunk_size=CHUNK, staging_dtype=torch.float32) -> GPTQ:
    layer = torch.nn.Linear(COLUMNS, 64, bias=False)
    cfg = GPTQConfig(
        bits=4,
        group_size=32,
        device="cpu",
        hessian=HessianConfig(chunk_size=chunk_size, staging_dtype=staging_dtype),
    )
    return GPTQ(layer, qcfg=cfg)


def _batch(rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, COLUMNS, generator=generator)


def _allocated_bytes(fn) -> int:
    """Bytes the profiled call allocated, counted once per allocation.

    ``self_cpu_memory_usage`` excludes the nested op that actually performed
    the allocation (``aten::zeros`` -> ``aten::empty``, ``aten::matmul`` ->
    ``aten::mm``), so summing it counts every new tensor exactly once.
    """

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as profile:
        fn()
    return sum(
        event.self_cpu_memory_usage
        for event in profile.events()
        if event.self_cpu_memory_usage > 0
    )


def _partial(task: GPTQ) -> torch.Tensor:
    (partial,) = task._device_hessian_partials.values()
    return partial


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = torch.linalg.vector_norm(actual.double() - expected)
    return float(error / torch.linalg.vector_norm(expected))


# ---------------------------------------------------------------------------
# The transient is gone.
# ---------------------------------------------------------------------------

def test_chunked_capture_allocates_only_the_accumulator():
    """Four chunks must not allocate four extra columns^2 products."""

    release_workspace_cache()
    task = _task()
    matrix = _batch(ROWS, 1)
    task.compute_hessian_xtx(matrix)  # warm the leased workspaces

    allocated = _allocated_bytes(lambda: task.compute_hessian_xtx(matrix))

    # One accumulator is the Hessian itself. A per-chunk product would add
    # ``rows / chunk`` more of them: four chunks used to allocate five.
    assert allocated <= 1.5 * COLUMNS * COLUMNS * 4, allocated


def test_unchunked_capture_does_not_allocate_a_product_per_batch():
    """A batch that lands in an existing partial allocates no new Hessian."""

    task = _task(chunk_size=None)
    task.add_batch(_batch(64, 2), None)
    before = _partial(task).untyped_storage().data_ptr()
    # Materialize the batch first: the profiled call must only be the capture.
    batch = _batch(64, 3)

    allocated = _allocated_bytes(lambda: task.add_batch(batch, None))

    assert _partial(task).untyped_storage().data_ptr() == before
    assert allocated < COLUMNS * COLUMNS * 4, allocated


def test_the_partial_is_updated_in_place():
    task = _task(chunk_size=None)
    task.add_batch(_batch(32, 4), None)
    partial = _partial(task)
    storage = partial.untyped_storage().data_ptr()
    snapshot = partial.clone()

    task.add_batch(_batch(32, 5), None)

    assert _partial(task).untyped_storage().data_ptr() == storage
    assert not torch.equal(_partial(task), snapshot)


# ---------------------------------------------------------------------------
# compute_hessian_xtx's ``out`` contract.
# ---------------------------------------------------------------------------

def test_compute_hessian_xtx_accumulates_into_a_matching_out():
    task = _task(chunk_size=None)
    matrix = _batch(96, 6)
    out = torch.ones((COLUMNS, COLUMNS), dtype=torch.float32) * 3.0
    storage = out.untyped_storage().data_ptr()

    result = task.compute_hessian_xtx(matrix, out=out)

    assert result.untyped_storage().data_ptr() == storage
    assert result is out
    fp32 = matrix.to(torch.float32)
    expected = 3.0 + fp32.T @ fp32
    assert _relative_error(result, expected) < 1e-6


def test_compute_hessian_xtx_chunked_accumulates_into_a_matching_out():
    task = _task()
    matrix = _batch(ROWS, 7)
    out = torch.zeros((COLUMNS, COLUMNS), dtype=torch.float32)
    storage = out.untyped_storage().data_ptr()

    result = task.compute_hessian_xtx(matrix, out=out)

    assert result.untyped_storage().data_ptr() == storage
    fp32 = matrix.to(torch.float32)
    assert _relative_error(result, fp32.T @ fp32) < 1e-6


def test_a_mismatched_out_is_ignored():
    task = _task(chunk_size=None)
    matrix = _batch(48, 8)
    fp32 = matrix.to(torch.float32)
    expected = fp32.T @ fp32

    for wrong in (
        torch.zeros((COLUMNS + 1, COLUMNS + 1), dtype=torch.float32),
        torch.zeros((COLUMNS, COLUMNS), dtype=torch.float16),
    ):
        before = wrong.clone()
        result = task.compute_hessian_xtx(matrix, out=wrong)
        assert result.untyped_storage().data_ptr() != wrong.untyped_storage().data_ptr()
        assert torch.equal(wrong, before), "the rejected accumulator was written"
        assert _relative_error(result, expected) < 1e-6


def test_an_empty_batch_leaves_a_supplied_out_untouched():
    task = _task(chunk_size=None)
    out = torch.full((COLUMNS, COLUMNS), 2.0)

    result = task.compute_hessian_xtx(torch.empty((0, COLUMNS)), out=out)

    assert result is out
    assert torch.equal(out, torch.full((COLUMNS, COLUMNS), 2.0))


# ---------------------------------------------------------------------------
# Accuracy and bookkeeping are unchanged.
# ---------------------------------------------------------------------------

def test_multi_batch_capture_matches_an_fp64_oracle():
    task = _task(chunk_size=None)
    batches = [_batch(rows, seed) for rows, seed in ((64, 9), (96, 10), (17, 11))]

    for batch in batches:
        task.add_batch(batch, None)

    stacked = torch.cat([batch.to(torch.float32) for batch in batches])
    oracle = stacked.double().T @ stacked.double()
    # ``finalize_hessian`` normalizes the raw sum by ``2 / samples``.
    expected = oracle * (2.0 / stacked.shape[0])

    assert _relative_error(task.finalize_hessian(), expected) < 1e-6


def test_sample_and_forward_counters_still_track_every_batch():
    task = _task(chunk_size=None)

    task.add_batch(_batch(11, 12), None)
    task.add_batch(_batch(13, 13), None)

    assert task.nsamples == 24
    assert task.fwd_counter == 2
    assert task._device_sample_counts == {torch.device("cpu"): 24}


def test_the_first_batch_still_owns_a_fresh_product():
    task = _task(chunk_size=None)

    task.add_batch(_batch(16, 14), None)

    partial = _partial(task)
    fp32 = _batch(16, 14).to(torch.float32)
    assert _relative_error(partial, fp32.T @ fp32) < 1e-6


# ---------------------------------------------------------------------------
# Concurrency: one accumulator, one writer at a time.
# ---------------------------------------------------------------------------

def test_concurrent_batches_do_not_lose_an_accumulation():
    task = _task(chunk_size=None)
    batches = [_batch(32, 100 + index) for index in range(8)]
    # Seed the partial so every worker takes the in-place path.
    task.add_batch(_batch(1, 99), None)

    threads = [
        threading.Thread(target=task.add_batch, args=(batch, None)) for batch in batches
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    stacked = torch.cat([batch.to(torch.float32) for batch in batches])
    oracle = stacked.double().T @ stacked.double()
    assert task.nsamples == 1 + 32 * len(batches)
    # Compare the accumulated sum, not ``finalize_hessian``'s normalization.
    partial = _partial(task).double() - _batch(1, 99).to(torch.float64).T @ _batch(1, 99).to(torch.float64)
    assert _relative_error(partial, oracle) < 1e-6
