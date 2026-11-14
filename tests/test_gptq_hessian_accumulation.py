# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def test_gptq_hessian_finalization_accumulates_later_batches():
    task = GPTQ(
        torch.nn.Linear(2, 1, bias=False),
        QuantizeConfig(bits=4, group_size=-1, fallback=None),
    )
    task.quantizer.configure(perchannel=True)
    first = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    second = torch.tensor([[3.0, 0.0], [0.0, 4.0]])

    task.add_batch(first, None)
    task.finalize_hessian()
    task.add_batch(second, None)
    actual = task.finalize_hessian()
    all_rows = torch.cat((first, second)).double()
    expected = 2 * all_rows.T @ all_rows / len(all_rows)

    torch.testing.assert_close(actual.double(), expected)
    assert task.nsamples == len(all_rows)


def test_gptq_adopted_hessian_accumulates_later_batches():
    config = QuantizeConfig(bits=4, group_size=-1, fallback=None)
    leader = GPTQ(torch.nn.Linear(2, 1, bias=False), config)
    follower = GPTQ(torch.nn.Linear(2, 1, bias=False), config)
    first = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    second = torch.tensor([[3.0, 0.0], [0.0, 4.0]])

    leader.add_batch(first, None)
    leader.finalize_hessian()
    follower.adopt_hessian_from(leader)
    follower.add_batch(second, None)
    actual = follower.finalize_hessian()
    all_rows = torch.cat((first, second)).double()
    expected = 2 * all_rows.T @ all_rows / len(all_rows)

    torch.testing.assert_close(actual.double(), expected)
    assert follower.nsamples == len(all_rows)
    assert follower._hessian_total_samples == len(all_rows)


def test_gptq_snapshot_hessian_merges_finalized_and_pending_batches():
    task = GPTQ(
        torch.nn.Linear(2, 1, bias=False),
        QuantizeConfig(bits=4, group_size=-1, fallback=None),
    )
    first = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    second = torch.tensor([[3.0, 0.0], [0.0, 4.0]])

    task.add_batch(first, None)
    finalized = task.finalize_hessian()
    task.add_batch(second, None)
    finalized_before = finalized.clone()
    partial = task._device_hessian_partials[torch.device("cpu")]
    partial_before = partial.clone()

    snapshot = task.snapshot_hessian(torch.device("cpu"))

    all_rows = torch.cat((first, second)).double()
    oracle = 2 * all_rows.T @ all_rows / len(all_rows)
    normalized_error = torch.linalg.vector_norm(
        snapshot.double() - oracle
    ) / torch.linalg.vector_norm(oracle)
    assert float(normalized_error) <= 1e-6

    # The snapshot must not consume or renormalize the live capture.
    assert torch.equal(finalized, finalized_before)
    assert torch.equal(partial, partial_before)
    assert task._hessian_dirty
    assert task.nsamples == len(all_rows)
    assert task._hessian_total_samples == len(first)
    torch.testing.assert_close(task.finalize_hessian().double(), oracle)


def _embedding_task() -> GPTQ:
    task = GPTQ(
        nn.Embedding(5, 3, dtype=torch.float32),
        QuantizeConfig(bits=4, group_size=-1, fallback=None),
    )
    task.quantizer.configure(perchannel=True)
    return task


def test_gptq_snapshot_hessian_embedding_matches_finalize():
    task = _embedding_task()
    task.add_batch(torch.tensor([[0, 1, 1, 3]]), None)
    first_counts = torch.tensor([1, 2, 0, 1, 0], dtype=torch.float32)
    expected = 2.0 * first_counts / 4.0

    snapshot = task.snapshot_hessian(torch.device("cpu"))
    torch.testing.assert_close(snapshot, expected)
    assert task.H is None
    assert task._hessian_dirty
    assert task.nsamples == 4
    torch.testing.assert_close(task.finalize_hessian(), expected)

    # Continued capture after finalization merges the finalized diagonal with
    # the new token counts instead of returning zeros or drifting.
    task.add_batch(torch.tensor([[4, 4]]), None)
    merged_counts = first_counts + torch.tensor([0, 0, 0, 0, 2], dtype=torch.float32)
    expected = 2.0 * merged_counts / 6.0

    snapshot = task.snapshot_hessian(torch.device("cpu"))
    torch.testing.assert_close(snapshot, expected)
    assert task._hessian_dirty
    assert task.nsamples == 6
    torch.testing.assert_close(task.finalize_hessian(), expected)


def test_gptq_snapshot_hessian_survives_later_finalization():
    """A same-device FP32 snapshot must not alias the live capture (#3862).

    ``Tensor.to(device=..., dtype=float32)`` is a no-op that returns the same
    storage when the device and dtype already match, and
    ``materialize_global_hessian`` mutates ``self.H`` in place on the next
    finalization. The durable capture-frontier contract needs a stable value.
    """

    task = GPTQ(
        torch.nn.Linear(2, 2, bias=False),
        QuantizeConfig(bits=4, group_size=-1, fallback=None),
    )
    task.add_batch(torch.tensor([[1.0, 0.0]]), None)
    task.finalize_hessian(torch.device("cpu"))

    snapshot = task.snapshot_hessian(torch.device("cpu"))
    saved = snapshot.clone()

    assert snapshot.data_ptr() != task.H.data_ptr()

    task.add_batch(torch.tensor([[0.0, 2.0]]), None)
    task.finalize_hessian(torch.device("cpu"))

    torch.testing.assert_close(snapshot, saved, rtol=0.0, atol=0.0)
    # The live capture did move on; only the snapshot is frozen.
    assert not torch.equal(task.H, saved)


def test_gptq_snapshot_hessian_embedding_survives_later_finalization():
    """The embedding diagonal fast path aliased ``_H_diag`` in the same way."""

    task = _embedding_task()
    task.add_batch(torch.tensor([[0, 1, 1]]), None)
    task.finalize_hessian(torch.device("cpu"))

    snapshot = task.snapshot_hessian(torch.device("cpu"))
    saved = snapshot.clone()

    assert snapshot.data_ptr() != task._H_diag.data_ptr()

    task.add_batch(torch.tensor([[2, 3]]), None)
    task.finalize_hessian(torch.device("cpu"))

    torch.testing.assert_close(snapshot, saved, rtol=0.0, atol=0.0)
    assert not torch.equal(task._H_diag, saved)


def test_gptq_snapshot_hessian_is_writable_without_touching_the_capture():
    """A caller may keep or edit a snapshot; the capture must not see it."""

    task = GPTQ(
        torch.nn.Linear(2, 2, bias=False),
        QuantizeConfig(bits=4, group_size=-1, fallback=None),
    )
    task.add_batch(torch.tensor([[1.0, 0.0]]), None)
    task.finalize_hessian(torch.device("cpu"))
    live = task.H.clone()

    snapshot = task.snapshot_hessian(torch.device("cpu"))
    snapshot.fill_(-1.0)

    torch.testing.assert_close(task.H, live, rtol=0.0, atol=0.0)

    embedding = _embedding_task()
    embedding.add_batch(torch.tensor([[0, 1, 1]]), None)
    embedding.finalize_hessian(torch.device("cpu"))
    live_diag = embedding._H_diag.clone()

    diag_snapshot = embedding.snapshot_hessian(torch.device("cpu"))
    diag_snapshot.fill_(-1.0)

    torch.testing.assert_close(embedding._H_diag, live_diag, rtol=0.0, atol=0.0)
