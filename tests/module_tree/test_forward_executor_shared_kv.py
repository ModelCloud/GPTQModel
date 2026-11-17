# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Cross-layer KV reuse must be per batch and refreshed by the replay pass.

Hymba's ``reuse_kv`` layers consume the previous layer's attention K/V through
``kv_last_layer``. The looper caches that K/V in ``shared_kv_cache_dict`` while
it walks the calibration batches. Two invariants keep the cached tensors paired
with the activations they belong to:

1. the cache is keyed by ``(layer_index, batch_idx)``, so batch ``b`` of the
   consuming layer only sees the K/V batch ``b`` produced; and
2. the post-quant replay overwrites the pre-quant (FP) capture for that slot,
   because the replay is what produces the quantized hidden states that feed
   the next layer.

Before the fix every consumer batch received batch 0's FP K/V
(``kv_last_layer values: [0.0, 0.0, 0.0, 0.0]``), silently corrupting the
Hessian for batches 1..N-1.
"""

from types import SimpleNamespace

import torch

from gptqmodel.looper.forward_executor import ForwardExecutor


BATCHES = 4


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _ImmediatePool:
    """Run submissions inline so the test needs no worker threads."""

    def submit(self, device, fn, *args, **kwargs):
        del device
        return _ImmediateFuture(fn(*args, **kwargs))

    submit_serial = submit


class _RecordingModule(torch.nn.Module):
    """Return a batch-specific K/V and record the K/V it was handed."""

    def __init__(self, tag: float = 0.0) -> None:
        super().__init__()
        self.tag = tag
        self.seen_kv = []

    def forward(self, hidden, **kwargs):
        kv = kwargs.get("kv_last_layer")
        self.seen_kv.append(None if kv is None else kv[0].clone())
        produced = (hidden * 10.0 + self.tag, hidden * 10.0 + self.tag + 1.0)
        return hidden + 1.0, produced


class _StubLooper:
    def __init__(self, *, write_shared_kv_cache: bool) -> None:
        self.gptq_model = SimpleNamespace(
            quantize_config=SimpleNamespace(calibration_data_device=None),
            write_shared_kv_cache=write_shared_kv_cache,
            prepare_layer_replay_kwargs=lambda **kwargs: kwargs["additional_inputs"],
            prepare_layer_replay_keep_mask=lambda **kwargs: kwargs["keep_mask"],
        )
        self.moe_routing_override = None
        self.moe_routing_bypass = False
        self.support_batch_quantize = False

    def _resolve_batch_total(self, raw_count, fallback_sequence):
        del raw_count
        return len(fallback_sequence)

    def _collect_row_counts(self, layer_inputs):
        return [int(batch[0].shape[0]) for batch in layer_inputs]

    def _batch_row_count(self, batch_inputs):
        return int(batch_inputs[0].shape[0]) if batch_inputs else 0

    def _set_processor_mask(self, processor, mask, padding_mask=None):
        del processor, mask, padding_mask


class _Processor:
    num_batches = BATCHES

    def _set_current_batch_index(self, index):
        del index


def _inputs():
    return [[torch.full((1, 2), float(index))] for index in range(BATCHES)]


def _run_single(module, *, layer_index, reuse_kv, cache, write_shared_kv_cache):
    executor = ForwardExecutor(
        _StubLooper(write_shared_kv_cache=write_shared_kv_cache)
    )
    return executor.run_single(
        module=module,
        processor=_Processor(),
        layer_inputs=_inputs(),
        layer_input_kwargs=[{} for _ in range(BATCHES)],
        position_ids=[None for _ in range(BATCHES)],
        attention_masks=[None for _ in range(BATCHES)],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict=cache,
        layer_index=layer_index,
        need_outputs=False,
        reuse_kv=reuse_kv,
    )


def test_consumer_receives_each_batchs_own_kv():
    cache = {}
    producer = _RecordingModule(tag=0.0)
    _run_single(
        producer,
        layer_index=0,
        reuse_kv=False,
        cache=cache,
        write_shared_kv_cache=True,
    )

    consumer = _RecordingModule(tag=0.0)
    _run_single(
        consumer,
        layer_index=1,
        reuse_kv=True,
        cache=cache,
        write_shared_kv_cache=True,
    )

    assert sorted(key for key in cache if key[0] == 0) == [(0, index) for index in range(BATCHES)]
    observed = [kv.flatten().tolist() for kv in consumer.seen_kv]
    # Batch i produced ``hidden * 10`` with ``hidden == i``.
    assert observed == [[float(index * 10)] * 2 for index in range(BATCHES)]
    assert len({tuple(values) for values in observed}) == BATCHES


def test_replay_overwrites_the_pre_quant_kv_capture():
    cache = {}
    # Pre-quant (FP) forward: tag 0.
    _run_single(
        _RecordingModule(tag=0.0),
        layer_index=0,
        reuse_kv=False,
        cache=cache,
        write_shared_kv_cache=True,
    )
    # Post-quant replay for the same layer: tag 100. Its KV must replace the FP
    # capture rather than lose to the old ``is None`` guard.
    _run_single(
        _RecordingModule(tag=100.0),
        layer_index=0,
        reuse_kv=False,
        cache=cache,
        write_shared_kv_cache=True,
    )

    consumer = _RecordingModule(tag=0.0)
    _run_single(
        consumer,
        layer_index=1,
        reuse_kv=True,
        cache=cache,
        write_shared_kv_cache=True,
    )

    observed = [kv.flatten().tolist() for kv in consumer.seen_kv]
    assert observed == [[float(index * 10 + 100)] * 2 for index in range(BATCHES)]
    # The FP capture (tag 0) must not survive the replay.
    assert all(values[0] >= 100.0 for values in observed)


def _run_parallel(worker, *, layer_index, reuse_kv, cache, write_shared_kv_cache):
    executor = ForwardExecutor(
        _StubLooper(write_shared_kv_cache=write_shared_kv_cache)
    )
    return executor.run_parallel(
        module=torch.nn.Identity(),
        processor=_Processor(),
        layer_inputs=_inputs(),
        layer_input_kwargs=[{} for _ in range(BATCHES)],
        position_ids=[None for _ in range(BATCHES)],
        attention_masks=[None for _ in range(BATCHES)],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict=cache,
        layer_index=layer_index,
        need_outputs=False,
        reuse_kv=reuse_kv,
        devices=[torch.device("cpu"), torch.device("meta")],
        clone_module_for_devices_fn=lambda module, devices, **kwargs: dict.fromkeys(
            devices, module
        ),
        forward_batch_worker_fn=worker,
        device_thread_pool=_ImmediatePool(),
    )


def test_parallel_worker_gets_its_own_batch_kv_and_refreshes_on_replay():
    cache = {}

    def producer(module, processor, batch_index, layer_input, *args, **kwargs):
        del module, processor, args, kwargs
        hidden = layer_input[0]
        return batch_index, hidden, (hidden * 10.0, hidden * 10.0 + 1.0)

    _run_parallel(
        producer,
        layer_index=0,
        reuse_kv=False,
        cache=cache,
        write_shared_kv_cache=True,
    )
    # Replay republishes the same slots with a distinguishable KV.
    def replay(module, processor, batch_index, layer_input, *args, **kwargs):
        del module, processor, args, kwargs
        hidden = layer_input[0]
        return batch_index, hidden, (hidden * 10.0 + 100.0, hidden * 10.0 + 101.0)

    _run_parallel(
        replay,
        layer_index=0,
        reuse_kv=False,
        cache=cache,
        write_shared_kv_cache=True,
    )

    observed = {}

    def consumer(module, processor, batch_index, layer_input, *args, **kwargs):
        del module, processor, layer_input, args
        observed[batch_index] = kwargs["prev_kv"][0].flatten().tolist()
        hidden = torch.zeros(1, 2)
        return batch_index, hidden, None

    _run_parallel(
        consumer,
        layer_index=1,
        reuse_kv=True,
        cache=cache,
        # Consumers may not write; the producer already populated the cache.
        write_shared_kv_cache=False,
    )

    assert observed == {
        index: [float(index * 10 + 100)] * 2 for index in range(BATCHES)
    }
