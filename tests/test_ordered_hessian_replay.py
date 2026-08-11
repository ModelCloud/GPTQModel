import threading

import pytest
import torch

from gptqmodel.looper.ordered_hessian_replay import (
    GPTQOrderedHessianReplayAttachment,
    OrderedHessianReplayBlock,
    move_replay_tensor,
)


def test_ordered_replay_requires_consecutive_increasing_wave():
    replay = OrderedHessianReplayBlock()

    with pytest.raises(ValueError, match="increasing"):
        replay.begin([1, 0], torch.device("cpu"))
    with pytest.raises(ValueError, match="consecutive"):
        replay.begin([0, 2], torch.device("cpu"))


def test_ordered_replay_flushes_batches_in_order_after_parallel_capture():
    replay = OrderedHessianReplayBlock()
    replay.begin([4, 5], torch.device("cpu"))
    observed = []
    barrier = threading.Barrier(2)

    def capture(batch_index, labels):
        barrier.wait()
        for label in labels:
            replay.append(batch_index, lambda _device, _cache, value=label: observed.append(value))

    workers = [
        threading.Thread(target=capture, args=(4, ["4a", "4b"])),
        threading.Thread(target=capture, args=(5, ["5a", "5b"])),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    replay.flush()

    assert all(not worker.is_alive() for worker in workers)
    assert observed == ["4a", "4b", "5a", "5b"]
    assert replay.active is False


def test_ordered_replay_deduplicates_retained_storage_and_transfer():
    replay = OrderedHessianReplayBlock(maximum_retained_bytes_per_device=64)
    replay.begin([0], torch.device("cpu"))
    source = torch.arange(8, dtype=torch.float32)
    first = replay.retain(source[:4])
    second = replay.retain(source[:4])
    transferred = []

    def replay_entry(device, cache):
        transferred.append(move_replay_tensor(first, device, cache))
        transferred.append(move_replay_tensor(second, device, cache))

    replay.append(0, replay_entry)
    replay.flush()

    assert transferred[0] is transferred[1]
    torch.testing.assert_close(transferred[0], source[:4], atol=0, rtol=0)


def test_ordered_replay_fails_closed_on_retained_storage_budget():
    replay = OrderedHessianReplayBlock(maximum_retained_bytes_per_device=16)
    replay.begin([0], torch.device("cpu"))

    with pytest.raises(RuntimeError, match="retained-activation budget"):
        replay.retain(torch.zeros(8, dtype=torch.float32))

    replay.abort()
    assert replay.active is False


def test_ordered_replay_abort_releases_active_wave():
    replay = OrderedHessianReplayBlock()
    replay.begin([0], torch.device("cpu"))
    replay.retain(torch.ones(2))
    replay.append(0, lambda _device, _cache: None)

    replay.abort()

    assert replay.active is False
    replay.begin([1], torch.device("cpu"))
    replay.abort()


def test_ordered_replay_rejects_empty_nested_and_out_of_wave_operations():
    replay = OrderedHessianReplayBlock()
    with pytest.raises(ValueError, match="at least one"):
        replay.begin([], torch.device("cpu"))
    with pytest.raises(RuntimeError, match="outside an active"):
        replay.retain(torch.ones(1))
    with pytest.raises(RuntimeError, match="not part"):
        replay.append(0, lambda *_args: None)

    replay.begin([0], torch.device("cpu"))
    with pytest.raises(RuntimeError, match="already active"):
        replay.begin([1], torch.device("cpu"))
    with pytest.raises(RuntimeError, match="not part"):
        replay.append(1, lambda *_args: None)
    replay.abort()

    # Inactive/foreign seals and flushes are intentionally harmless cleanup.
    replay.seal_batch(0, torch.device("cpu"))
    replay.flush()


def test_ordered_replay_aborts_when_replay_callback_raises():
    replay = OrderedHessianReplayBlock()
    replay.begin([0], torch.device("cpu"))
    replay.append(0, lambda *_args: (_ for _ in ()).throw(ValueError("replay failed")))

    with pytest.raises(ValueError, match="replay failed"):
        replay.flush()
    assert replay.active is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ordered_replay_cuda_event_seals_forward_stream_before_replay():
    replay = OrderedHessianReplayBlock()
    device = torch.device("cuda:0")
    replay.begin([0], device)
    source = torch.arange(32, device=device, dtype=torch.float32)
    expected = source.square().sum().item()
    observed = []
    replay.append(0, lambda _device, _cache: observed.append(source.square().sum().item()))
    replay.seal_batch(0, device)
    replay.flush()

    assert observed == [expected]
    assert replay.active is False


class _ReplayTask:
    @staticmethod
    def _sequence_count_for_input(inp):
        return inp.shape[0]

    @staticmethod
    def _reshape_input(inp):
        reshaped = inp.reshape(-1, inp.shape[-1])
        return reshaped.shape[0], reshaped, inp.shape[0]


class _ReplayProcessor:
    def __init__(self):
        self.batch_index = 0
        self.hessian = torch.zeros(3, 3, dtype=torch.float32)
        self.calls = []

    def current_batch_index(self):
        return self.batch_index

    def _add_batch_with_shared_hessian_immediate(
        self, task, inp, out, *, batch_index, cache_source, cache_extra
    ):
        del task, out
        materialized = inp.reshape(-1, inp.shape[-1]).float()
        self.hessian.addmm_(materialized.T, materialized)
        self.calls.append(("batch", batch_index, cache_source.clone(), cache_extra))

    def _record_moe_shared_input_followers_immediate(self, **kwargs):
        self.calls.append(("followers", kwargs))


def test_gptq_ordered_attachment_matches_serial_hessian_and_follower_order():
    task = _ReplayTask()
    processor = _ReplayProcessor()
    attachment = GPTQOrderedHessianReplayAttachment(processor)
    inputs = [
        torch.tensor([[[1.0, 2.0, 3.0]]]),
        torch.tensor([[[4.0, 5.0, 6.0]]]),
    ]
    expected = torch.zeros(3, 3)
    for value in inputs:
        flat = value.reshape(-1, 3)
        expected.addmm_(flat.T, flat)

    attachment.begin_parallel_forward_wave([0, 1], torch.device("cpu"))
    assert attachment.active
    processor.batch_index = 1
    assert attachment.defer_batch(
        task,
        inputs[1],
        batch_index=1,
        cache_source=inputs[1],
        cache_extra=("same",),
    ) == (1, 1)
    processor.batch_index = 0
    separate_cache_source = inputs[0].clone()
    assert attachment.defer_batch(
        task,
        inputs[0],
        batch_index=0,
        cache_source=separate_cache_source,
        cache_extra=("separate",),
    ) == (1, 1)
    attachment.defer_followers(
        source_name="source",
        follower_names=["follower"],
        batch_token_size=1,
        observation_count=1,
        sequence_count=1,
    )
    attachment.seal_parallel_forward_batch(0, torch.device("cpu"))
    attachment.flush_parallel_forward_wave()

    torch.testing.assert_close(processor.hessian, expected, atol=0, rtol=0)
    assert [call[0] for call in processor.calls] == ["batch", "followers", "batch"]
    assert processor.calls[0][2] is not separate_cache_source
    assert processor.calls[0][3] == ("separate",)
    assert attachment.active is False
    attachment.abort_parallel_forward_wave()


def test_gptq_ordered_attachment_requires_explicit_batch_indices():
    processor = _ReplayProcessor()
    attachment = GPTQOrderedHessianReplayAttachment(processor)
    attachment.begin_parallel_forward_wave([0], torch.device("cpu"))
    with pytest.raises(RuntimeError, match="explicit calibration batch"):
        attachment.defer_batch(
            _ReplayTask(),
            torch.ones(1, 1, 3),
            batch_index=None,
            cache_source=torch.ones(1, 1, 3),
            cache_extra=None,
        )
    processor.batch_index = None
    with pytest.raises(RuntimeError, match="explicit calibration batch"):
        attachment.defer_followers(
            source_name="source",
            follower_names=["follower"],
            batch_token_size=1,
            observation_count=1,
            sequence_count=1,
        )
    attachment.abort_parallel_forward_wave()
