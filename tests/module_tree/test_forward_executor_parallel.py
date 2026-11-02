# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the data-parallel forward result contract.

`StopForward` (MoE early-stop) legitimately yields no layer output. The serial
executor skips such batches; the data-parallel executor must agree instead of
failing every multi-device quantization run that reaches an early-stopping
subset.
"""

from types import SimpleNamespace

import torch

from gptqmodel.looper.forward_executor import ForwardExecutor


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

    def _resolve_batch_total(self, raw_count, fallback_sequence):
        del raw_count, fallback_sequence
        return self._num_batches

    def _collect_row_counts(self, layer_inputs):
        return [int(batch[0].shape[0]) for batch in layer_inputs]

    def _batch_row_count(self, batch_inputs):
        return int(batch_inputs[0].shape[0])


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


def test_parallel_forward_tolerates_early_stop_without_outputs():
    def worker(module, processor, batch_index, *args, **kwargs):
        del module, processor, args, kwargs
        return batch_index, None, None

    assert _run_parallel(worker) == []


def test_parallel_forward_keeps_batch_order_for_real_outputs():
    def worker(module, processor, batch_index, layer_input, *args, **kwargs):
        del module, processor, args, kwargs
        return batch_index, layer_input[0] + batch_index, None

    outputs = _run_parallel(worker)

    assert len(outputs) == 2
    for batch_index, batch_output in enumerate(outputs):
        assert len(batch_output) == 1
        torch.testing.assert_close(batch_output[0], torch.full((1, 4), float(batch_index)))
