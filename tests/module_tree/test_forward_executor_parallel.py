# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the forward result contract.

`StopForward` (MoE early-stop) legitimately yields no layer output. Dropping
that batch instead of keeping its slot shifts every later batch onto the
previous batch's input kwargs, mask, and position ids. Both the serial and the
data-parallel executors must keep positional slots.
"""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.forward_executor import ForwardExecutor
from gptqmodel.nn_modules.hooked_linear import StopForward


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


class _StubLooper:
    def __init__(self, num_batches: int) -> None:
        self.gptq_model = SimpleNamespace(
            quantize_config=SimpleNamespace(
                compute_device_filter=None,
                calibration_data_device=None,
            ),
            write_shared_kv_cache=False,
        )
        self.moe_routing_override = None
        self.moe_routing_bypass = False
        self.support_batch_quantize = False
        self._num_batches = num_batches
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


def _run_parallel(worker, *, num_batches: int = 2, need_outputs: bool = True):
    devices = [torch.device("cpu"), torch.device("meta")]
    executor = ForwardExecutor(_StubLooper(num_batches))
    return executor.run_parallel(
        module=torch.nn.Identity(),
        processor=SimpleNamespace(num_batches=num_batches),
        layer_inputs=[[torch.zeros(1, 4)] for _ in range(num_batches)],
        layer_input_kwargs=[{} for _ in range(num_batches)],
        position_ids=[None for _ in range(num_batches)],
        attention_masks=[None for _ in range(num_batches)],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=need_outputs,
        reuse_kv=False,
        devices=devices,
        clone_module_for_devices_fn=lambda module, devs, **kwargs: dict.fromkeys(devs, module),
        forward_batch_worker_fn=worker,
        device_thread_pool=_ImmediatePool(),
    )


class _BatchModule(torch.nn.Module):
    """Return a batch-specific output and raise ``StopForward`` on request."""

    def __init__(self, stop_batches=()):
        super().__init__()
        self.stop_batches = set(stop_batches)
        self.calls = []

    def forward(self, hidden, **kwargs):
        del kwargs
        index = int(hidden[0, 0].item())
        self.calls.append(index)
        if index in self.stop_batches:
            raise StopForward("early stop")
        return hidden + 1.0


class _Processor:
    num_batches = 4

    def _set_current_batch_index(self, index):
        del index


def _run_single(module, *, num_batches: int = 4, need_outputs: bool = True):
    executor = ForwardExecutor(_StubLooper(num_batches))
    return executor.run_single(
        module=module,
        processor=_Processor(),
        layer_inputs=[
            [torch.full((1, 4), float(index))] for index in range(num_batches)
        ],
        layer_input_kwargs=[{} for _ in range(num_batches)],
        position_ids=[None for _ in range(num_batches)],
        attention_masks=[None for _ in range(num_batches)],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=need_outputs,
        reuse_kv=False,
    )


def test_parallel_forward_keeps_slots_for_early_stop_without_outputs():
    def worker(module, processor, batch_index, *args, **kwargs):
        del module, processor, args, kwargs
        return batch_index, None, None

    assert _run_parallel(worker) == [[], []]


def test_parallel_forward_keeps_each_batch_on_its_own_slot():
    def worker(module, processor, batch_index, layer_input, *args, **kwargs):
        del module, processor, args, kwargs
        if batch_index == 1:
            return batch_index, None, None
        return batch_index, layer_input[0] + batch_index, None

    outputs = _run_parallel(worker, num_batches=4)

    assert len(outputs) == 4
    assert outputs[1] == []
    for batch_index in (0, 2, 3):
        torch.testing.assert_close(
            outputs[batch_index][0],
            torch.full((1, 4), float(batch_index)),
        )


def test_serial_forward_keeps_slots_for_early_stopped_batches():
    module = _BatchModule(stop_batches={1})
    outputs = _run_single(module, num_batches=4)

    assert module.calls == [0, 1, 2, 3]
    assert len(outputs) == 4
    assert outputs[1] == []
    for index in (0, 2, 3):
        torch.testing.assert_close(
            outputs[index][0], torch.full((1, 4), float(index) + 1.0)
        )


def test_serial_forward_pads_trailing_early_stop_slots():
    module = _BatchModule(stop_batches={3})
    outputs = _run_single(module, num_batches=4)

    assert len(outputs) == 4
    assert outputs[3] == []


def test_parallel_forward_keeps_batch_order_for_real_outputs():
    def worker(module, processor, batch_index, layer_input, *args, **kwargs):
        del module, processor, args, kwargs
        return batch_index, layer_input[0] + batch_index, None

    outputs = _run_parallel(worker)

    assert len(outputs) == 2
    for batch_index, batch_output in enumerate(outputs):
        assert len(batch_output) == 1
        torch.testing.assert_close(batch_output[0], torch.full((1, 4), float(batch_index)))


class _DeferredFuture:
    """Run the submitted worker only when the executor resolves it."""

    def __init__(self, fn, args, kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def result(self):
        return self._fn(*self._args, **self._kwargs)


class _DeferredPool:
    def __init__(self):
        self.futures = []

    def submit(self, device, fn, *args, **kwargs):
        del device
        future = _DeferredFuture(fn, args, kwargs)
        self.futures.append(future)
        return future

    submit_serial = submit


def test_parallel_forward_drains_every_submitted_worker_before_raising():
    drained = []

    def worker(module, processor, batch_index, *args, **kwargs):
        del module, processor, args, kwargs
        if batch_index == 0:
            raise RuntimeError("worker failed")
        drained.append(batch_index)
        return batch_index, None, None

    pool = _DeferredPool()
    executor = ForwardExecutor(_StubLooper(4))

    with pytest.raises(RuntimeError, match="worker failed"):
        executor.run_parallel(
            module=torch.nn.Identity(),
            processor=_Processor(),
            layer_inputs=[[torch.zeros(1, 4)] for _ in range(4)],
            layer_input_kwargs=[{} for _ in range(4)],
            position_ids=[None for _ in range(4)],
            attention_masks=[None for _ in range(4)],
            cur_layer_device=torch.device("cpu"),
            is_lm_head_module=False,
            shared_kv_cache_dict={},
            layer_index=0,
            need_outputs=False,
            reuse_kv=False,
            devices=[torch.device("cpu"), torch.device("meta")],
            clone_module_for_devices_fn=lambda module, devices, **kwargs: dict.fromkeys(
                devices, module
            ),
            forward_batch_worker_fn=worker,
            device_thread_pool=pool,
        )

    # The worker submitted alongside the failing one must still be resolved.
    assert 2 in drained
    assert len(pool.futures) == 2
