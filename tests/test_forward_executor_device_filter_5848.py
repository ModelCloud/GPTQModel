# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #5848: apply ``compute_device_filter`` once per dispatch.

``ForwardExecutor.run`` resolves the filter before choosing serial or parallel
dispatch. ``run_parallel`` must stage replicas and assign batches from that same
filtered pool; a second filter call can collapse a positional selector such as
``devices[1:]`` onto the last device and leave a staged replica idle.
"""

from collections import Counter
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gptqmodel.looper.forward_executor import ForwardExecutor
from gptqmodel.utils.looper_helpers import clone_module_for_devices


class _Linear(torch.nn.Linear):
    def forward(self, x, **kwargs):
        return super().forward(x)


class _InlinePool:
    """Run submissions inline and record each batch's assigned ordinal."""

    def __init__(self):
        self.submissions = Counter()
        self.assignments = {}

    def submit(self, device, fn, *args, **kwargs):
        self.submissions[device.index] += 1
        self.assignments[args[2]] = device.index
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future

    submit_serial = submit


def _looper(compute_device_filter):
    return SimpleNamespace(
        support_batch_quantize=False,
        moe_routing_override=None,
        moe_routing_bypass=False,
        _resolve_batch_total=lambda num_batches, _fallback: num_batches,
        _collect_row_counts=lambda batches: [1] * len(batches),
        gptq_model=SimpleNamespace(
            quantize_config=SimpleNamespace(
                compute_device_filter=compute_device_filter,
                calibration_data_device=None,
            ),
            prepare_layer_replay_kwargs=lambda **kwargs: kwargs["additional_inputs"],
            prepare_layer_replay_keep_mask=lambda **kwargs: kwargs["keep_mask"],
        ),
    )


def _run_parallel_dispatch(compute_device_filter, num_batches=4):
    devices = [torch.device(f"cpu:{index}") for index in range(3)]
    filter_inputs = []
    replicas = {}
    pool = _InlinePool()

    def recording_filter(candidates):
        filter_inputs.append([device.index for device in candidates])
        return compute_device_filter(candidates)

    def recording_clone(module, targets, **kwargs):
        staged = clone_module_for_devices(module, targets, **kwargs)
        replicas.update(
            {device.index: replica.weight.untyped_storage().nbytes() for device, replica in staged.items()}
        )
        return staged

    executor = ForwardExecutor(_looper(recording_filter))
    run_parallel = executor.run_parallel
    layer = _Linear(1024, 1024, bias=False)
    inputs = [
        [torch.randn((1, 1, 1024), generator=torch.Generator().manual_seed(index))] for index in range(num_batches)
    ]

    with patch.object(
        executor,
        "run_parallel",
        lambda **kwargs: run_parallel(
            **kwargs,
            clone_module_for_devices_fn=recording_clone,
            device_thread_pool=pool,
        ),
    ):
        outputs = executor.run(
            module=layer,
            processor=SimpleNamespace(
                num_batches=num_batches,
                _set_current_batch_index=lambda _index: None,
            ),
            layer_inputs=inputs,
            layer_input_kwargs=[{} for _ in inputs],
            position_ids=[],
            attention_masks=[None] * num_batches,
            cur_layer_device=torch.device("cpu"),
            is_lm_head_module=False,
            shared_kv_cache_dict={},
            layer_index=0,
            need_outputs=True,
            reuse_kv=False,
            select_forward_devices_fn=lambda _device: devices,
        )

    return outputs, inputs, layer, filter_inputs, replicas, pool


@pytest.mark.parametrize(
    "compute_device_filter",
    [
        pytest.param(lambda candidates: candidates[1:], id="positional-suffix"),
        pytest.param(
            lambda candidates: [d for d in candidates if d.index != 0],
            id="index-membership",
        ),
    ],
)
def test_parallel_dispatch_filters_pool_once_and_uses_every_survivor(
    compute_device_filter,
):
    outputs, inputs, layer, filter_inputs, replicas, pool = _run_parallel_dispatch(compute_device_filter)

    # `run` resolves the pool; `run_parallel` must reuse it instead of filtering
    # the already-filtered [1, 2] pool again.
    assert filter_inputs == [[0, 1, 2]]
    assert set(replicas) == {1, 2}
    assert set(pool.submissions) == {1, 2}
    assert pool.submissions[1] > 0
    assert pool.submissions[2] > 0
    assert sum(pool.submissions.values()) == len(inputs)
    assert set(pool.assignments.values()) == {1, 2}

    # Parallel dispatch must not change results or their original batch order.
    expected = [layer(batch[0]) for batch in inputs]
    assert len(outputs) == len(expected)
    assert all(torch.equal(output[0], reference) for output, reference in zip(outputs, expected))


def test_single_surviving_device_keeps_serial_fallback():
    devices = [torch.device(f"cpu:{index}") for index in range(3)]
    filter_inputs = []

    def keep_second_device(candidates):
        filter_inputs.append([device.index for device in candidates])
        return candidates[1:2]

    executor = ForwardExecutor(_looper(keep_second_device))
    captured = {}

    def fake_run_single(**kwargs):
        captured.update(kwargs)
        return []

    def fail_run_parallel(**kwargs):
        raise AssertionError(f"parallel dispatch should not run: {kwargs}")

    executor.run_single = fake_run_single
    executor.run_parallel = fail_run_parallel

    executor.run(
        module=torch.nn.Linear(1, 1),
        processor=SimpleNamespace(num_batches=1),
        layer_inputs=[[torch.zeros(1, 1)]],
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[None],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=False,
        reuse_kv=False,
        select_forward_devices_fn=lambda _device: devices,
    )

    assert filter_inputs == [[0, 1, 2]]
    assert captured["cur_layer_device"] == devices[1]
