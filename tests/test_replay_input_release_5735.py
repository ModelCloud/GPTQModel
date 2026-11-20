# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Post-quant replay releases each consumed calibration batch at its last use.

``_replay_layer_outputs`` feeds ``processor.receive_layer_inputs`` immediately
after the replay, so the replay is the last reader of the cached activations.
The executor previously kept every consumed batch alive until the whole replay
finished, so every layer boundary held two full activation sets.
"""

import gc
import weakref
from collections.abc import Sequence
from types import SimpleNamespace

import torch

from gptqmodel.looper.forward_executor import ForwardExecutor
from gptqmodel.looper.stage_layer import _replay_layer_outputs


NUM_BATCHES = 4


class _AddOne(torch.nn.Module):
    def forward(self, hidden, **kwargs):
        del kwargs
        return hidden + 1.0


class _Progress:
    def __init__(self):
        self.current_iter_step = 0

    def manual(self):
        return self

    def set(self, **kwargs):
        del kwargs
        return self

    def title(self, *args, **kwargs):
        del args, kwargs
        return self

    def subtitle(self, *args, **kwargs):
        del args, kwargs
        return self

    def draw(self):
        return self

    def close(self):
        return None


class _Log:
    def pb(self, *args, **kwargs):
        del args, kwargs
        return _Progress()


class _StubLooper:
    def __init__(self, num_batches: int = NUM_BATCHES) -> None:
        self.gptq_model = SimpleNamespace(
            quantize_config=SimpleNamespace(
                compute_device_filter=None,
                calibration_data_device=None,
                auto_forward_data_parallel=False,
            ),
            write_shared_kv_cache=False,
        )
        self.moe_routing_override = None
        self.moe_routing_bypass = False
        self.support_batch_quantize = False
        self._num_batches = num_batches
        self._current_subset = None
        self.gptq_model.prepare_layer_replay_kwargs = (
            lambda **kwargs: kwargs["additional_inputs"]
        )
        self.gptq_model.prepare_layer_replay_keep_mask = (
            lambda **kwargs: kwargs["keep_mask"]
        )

    def _resolve_batch_total(self, raw_count, fallback_sequence):
        del raw_count, fallback_sequence
        return self._num_batches

    def _collect_row_counts(self, layer_inputs):
        return [int(batch[0].shape[0]) for batch in layer_inputs]

    def _batch_row_count(self, batch_inputs):
        return int(batch_inputs[0].shape[0])

    def _set_processor_mask(self, processor, mask, padding_mask=None):
        del processor, mask, padding_mask


class _Processor:
    def __init__(
        self,
        num_batches: int = NUM_BATCHES,
        replay_after_process: bool = True,
    ) -> None:
        self.num_batches = num_batches
        self.forward = SimpleNamespace(replay_after_process=replay_after_process)
        self.layer_inputs = None
        self.snapshots = []

    def _set_current_batch_index(self, index):
        del index

    def forward_batch_completed(self, *, layer_index, batch_index):
        del layer_index, batch_index
        if self.layer_inputs is not None:
            self.snapshots.append([len(slot) for slot in self.layer_inputs])


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


class _ImmutableInputs(Sequence):
    """Mirror ``DiskBackedLayerOutputSequence``: readable, not assignable."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, index):
        return self._batches[index]


class _ReplayLooper(_StubLooper):
    """Run the real executor behind the replay helper's ``_run_forward_batches``."""

    def __init__(self, num_batches: int = NUM_BATCHES) -> None:
        super().__init__(num_batches)
        self._forward_executor = ForwardExecutor(self)
        self.calls = []

    def _run_forward_batches(self, **kwargs):
        self.calls.append(dict(kwargs))
        kwargs.pop("force_serial", None)
        kwargs.pop("preserve_module_devices", None)
        return self._forward_executor.run(**kwargs)


def _batch_inputs(num_batches: int = NUM_BATCHES):
    return [[torch.full((1, 4), float(index))] for index in range(num_batches)]


def _run_single(layer_inputs, processor, **kwargs):
    executor = ForwardExecutor(_StubLooper(len(layer_inputs)))
    return executor.run_single(
        module=_AddOne(),
        processor=processor,
        layer_inputs=layer_inputs,
        layer_input_kwargs=[{} for _ in layer_inputs],
        position_ids=[None for _ in layer_inputs],
        attention_masks=[None for _ in layer_inputs],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=1,
        need_outputs=True,
        reuse_kv=False,
        **kwargs,
    )


def _replay(looper, processor, layer_inputs, **kwargs):
    return _replay_layer_outputs(
        looper,
        module=_AddOne(),
        processor=processor,
        layer_inputs=layer_inputs,
        layer_input_kwargs=[{} for _ in layer_inputs],
        position_ids=[None for _ in layer_inputs],
        attention_masks=[None for _ in layer_inputs],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=1,
        layer_descriptor="model.layers.1",
        full={},
        log=_Log(),
        region_timer=None,
        **kwargs,
    )


def _assert_outputs(outputs, num_batches: int = NUM_BATCHES) -> None:
    assert len(outputs) == num_batches
    for index, output in enumerate(outputs):
        torch.testing.assert_close(
            output[0], torch.full((1, 4), float(index) + 1.0)
        )


def test_serial_replay_releases_each_batch_at_its_last_use():
    layer_inputs = _batch_inputs()
    processor = _Processor()
    processor.layer_inputs = layer_inputs
    refs = [weakref.ref(batch[0]) for batch in layer_inputs]

    outputs = _run_single(layer_inputs, processor, consume_inputs=True)

    assert processor.snapshots == [
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ]
    assert layer_inputs == [[], [], [], []]
    _assert_outputs(outputs)
    gc.collect()
    assert [ref() for ref in refs] == [None] * NUM_BATCHES


def test_serial_replay_without_consume_keeps_the_whole_set():
    layer_inputs = _batch_inputs()
    processor = _Processor()
    processor.layer_inputs = layer_inputs
    refs = [weakref.ref(batch[0]) for batch in layer_inputs]

    outputs = _run_single(layer_inputs, processor)

    assert processor.snapshots == [[1, 1, 1, 1]] * NUM_BATCHES
    _assert_outputs(outputs)
    gc.collect()
    assert all(ref() is not None for ref in refs)


def test_parallel_replay_releases_each_resolved_batch():
    layer_inputs = _batch_inputs()
    processor = _Processor()
    processor.layer_inputs = layer_inputs

    def worker(module, processor, batch_index, layer_input, *args, **kwargs):
        del processor, args, kwargs
        return batch_index, module(*layer_input), None

    executor = ForwardExecutor(_StubLooper())
    outputs = executor.run_parallel(
        module=_AddOne(),
        processor=processor,
        layer_inputs=layer_inputs,
        layer_input_kwargs=[{} for _ in layer_inputs],
        position_ids=[None for _ in layer_inputs],
        attention_masks=[None for _ in layer_inputs],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=1,
        need_outputs=True,
        reuse_kv=False,
        devices=[torch.device("cpu")],
        devices_filtered=True,
        consume_inputs=True,
        clone_module_for_devices_fn=(
            lambda module, devices, **kwargs: dict.fromkeys(devices, module)
        ),
        forward_batch_worker_fn=worker,
        device_thread_pool=_ImmediatePool(),
    )

    assert processor.snapshots == [
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ]
    assert layer_inputs == [[], [], [], []]
    _assert_outputs(outputs)


def test_serial_replay_skips_immutable_disk_backed_inputs():
    layer_inputs = _ImmutableInputs(_batch_inputs())
    processor = _Processor()

    outputs = _run_single(layer_inputs, processor, consume_inputs=True)

    _assert_outputs(outputs)
    assert [len(batch) for batch in layer_inputs] == [1, 1, 1, 1]


def test_replay_consumes_inputs_only_when_it_feeds_the_next_layer():
    layer_inputs = _batch_inputs()
    processor = _Processor()
    processor.layer_inputs = layer_inputs
    looper = _ReplayLooper()

    outputs = _replay(looper, processor, layer_inputs)

    assert processor.snapshots == [
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ]
    assert looper.calls[0]["consume_inputs"] is True
    _assert_outputs(outputs)


def test_replay_keeps_inputs_for_a_separate_pristine_cache():
    layer_inputs = _batch_inputs()
    processor = _Processor()
    processor.layer_inputs = layer_inputs
    looper = _ReplayLooper()

    _replay(looper, processor, layer_inputs, consume_inputs=False)

    assert processor.snapshots == [[1, 1, 1, 1]] * NUM_BATCHES
    assert looper.calls[0]["consume_inputs"] is False
    assert [len(batch) for batch in layer_inputs] == [1, 1, 1, 1]


def test_replay_keeps_inputs_when_another_processor_still_reads_them():
    layer_inputs = _batch_inputs()
    processor = _Processor(replay_after_process=False)
    processor.layer_inputs = layer_inputs
    looper = _ReplayLooper()

    _replay(looper, processor, layer_inputs)

    assert looper.calls[0].get("consume_inputs", False) is False
    assert processor.snapshots == [[1, 1, 1, 1]] * NUM_BATCHES
