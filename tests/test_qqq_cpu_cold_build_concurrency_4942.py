# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Concurrent cold QQQ CPU forwards must not read a dropped checkpoint.

#4856 (fix of #4805) drops the checkpoint ``B``/``s_group``/``s_channel``
buffers once the packed cache proves it can rebuild them. The cold path of
``_packed_weight_for_cpu`` still read those attributes directly and without the
transform lock, so a second thread that passed the cache check while the first
one was still building could read ``None`` after the drop (#4942):

    AttributeError: 'NoneType' object has no attribute 'device'

The build now runs under ``_cpu_transform_lock``, like the other CPU int4
kernels' ``_ensure_cpu_layout``. These tests drive the packed path with a
deterministic stand-in kernel (the AVX512 ``qqq_int4`` kernel is x86-only) and
widen the window between the cache check and the buffer reads.
"""

from __future__ import annotations

import threading
import time

import torch

from gptqmodel.nn_modules.qlinear import qqq as qqq_module
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from tests.test_qqq_kernel_memory_4387 import _build_torch_module


class _FakePackedKernel:
    """Deterministic ``qqq_int4`` stand-in that reads every operand."""

    def __call__(self, activation, packed, scales, group_size):
        weight = packed.to(torch.float32).sum(dim=1)
        scale = scales.to(torch.float32).sum() if scales.numel() else 1.0
        return activation.to(torch.float32).sum(dim=1, keepdim=True) + weight.unsqueeze(0) * scale


def _run_cold_forward_burst(module, x, *, threads: int = 4, stagger: float = 0.01):
    barrier = threading.Barrier(threads)
    outputs: list[torch.Tensor | None] = [None] * threads
    errors: list[str] = []

    def work(index: int) -> None:
        barrier.wait()
        time.sleep(stagger * index)
        try:
            with torch.inference_mode():
                outputs[index] = module(x).clone()
        except Exception as exc:  # surfaced by the caller
            errors.append(f"{type(exc).__name__}: {exc}")

    workers = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return outputs, errors


def test_concurrent_cold_forwards_do_not_read_a_dropped_checkpoint(monkeypatch):
    module = _build_torch_module(in_features=256, out_features=128, group_size=128)
    monkeypatch.setattr(qqq_module, "_qqq_cpu_kernel", _FakePackedKernel)

    # Widen the gap between "no cache yet" and the checkpoint reads, so a
    # thread that loses the race deterministically observes the dropped buffers.
    original_key = QQQTorchLinear._torch_weight_state_key

    def slow_key(self):
        time.sleep(0.05)
        return original_key(self)

    monkeypatch.setattr(QQQTorchLinear, "_torch_weight_state_key", slow_key)

    x = torch.randn(4, module.in_features, dtype=torch.float16)
    with torch.inference_mode():
        reference = module(x).clone()
    module.clear_weight_cache()
    assert module._torch_dropped_checkpoint is None

    outputs, errors = _run_cold_forward_burst(module, x, stagger=0.01)

    assert not errors, errors
    for index, output in enumerate(outputs):
        assert torch.equal(output, reference), f"thread {index} diverged"
    # The cache replaced the checkpoint layout, and the later forward still works.
    assert module._torch_dropped_checkpoint is not None
    with torch.inference_mode():
        assert torch.equal(module(x), reference)


def test_repeated_cold_bursts_keep_the_cache_consistent(monkeypatch):
    module = _build_torch_module(in_features=256, out_features=128, group_size=128)
    monkeypatch.setattr(qqq_module, "_qqq_cpu_kernel", _FakePackedKernel)

    x = torch.randn(2, module.in_features, dtype=torch.float16)
    with torch.inference_mode():
        reference = module(x).clone()

    for _ in range(3):
        module.clear_weight_cache()
        outputs, errors = _run_cold_forward_burst(module, x, threads=3, stagger=0.0)
        assert not errors, errors
        for output in outputs:
            assert torch.equal(output, reference)
