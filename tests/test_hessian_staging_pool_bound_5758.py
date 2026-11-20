# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Idle Hessian staging storage must not grow with every projection width (#5758).

The staging pool is keyed by device and dtype, and a slot is only reusable at
its exact column width. A worker that visits different widths in sequence
therefore kept one slot per width ever seen: eight widths with a 1 MiB
``chunk_bytes`` budget retained 16 idle buffers (24.00 MiB) while later layers
and the inverse Hessian needed that memory.

Releases now bound idle storage per device: the shape just released stays
resident (the next layer and concurrent workers reuse it) and other shapes are
evicted least-recently-released first until the pool is at most two working
sets. ``_WORKSPACE_CACHE`` always aliases a retained slot, so an eviction frees
its allocation instead of leaving the pool's own compatibility view holding it.
"""

from __future__ import annotations

import threading

import torch

from gptqmodel.quantization import gptq
from gptqmodel.quantization.config import HessianConfig, QuantizeConfig


BUDGET = 1 << 20


def _config() -> QuantizeConfig:
    return QuantizeConfig(
        hessian=HessianConfig(chunk_bytes=BUDGET, staging_dtype=torch.bfloat16)
    )


def _cache_key() -> tuple:
    return gptq._workspace_cache_key(torch.device("cpu"))


def _slots() -> list:
    key = _cache_key()
    return [
        slot
        for slot_key, slots in gptq._WORKSPACE_SLOTS.items()
        if slot_key[0] == key
        for slot in slots
    ]


def _idle_bytes() -> int:
    return sum(slot.tensor.numel() * slot.tensor.element_size() for slot in _slots())


def _shape_bytes(shape: tuple, dtype: torch.dtype) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size * torch.empty((0,), dtype=dtype).element_size()


def _working_set_bytes(shape: tuple) -> int:
    """Staging plus FP32 accumulator bytes for one width at this chunk budget."""

    return _shape_bytes(shape, torch.bfloat16) + _shape_bytes(shape, torch.float32)


def _run_width(width: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = BUDGET // (2 * width)
    task = gptq.GPTQ(torch.nn.Linear(width, 64, bias=False), _config())
    samples = torch.randn(rows, width, dtype=torch.bfloat16)
    return task.compute_hessian_xtx(samples), samples


def test_idle_storage_is_bounded_by_the_retained_working_set():
    torch.manual_seed(71)
    gptq.release_workspace_cache()
    try:
        for width in (128, 160, 192, 224, 256, 288, 320, 352):
            result, samples = _run_width(width)
            # The pool must not change the arithmetic it stages for.
            oracle = samples.double().T @ samples.double()
            error = torch.linalg.vector_norm(result.double() - oracle)
            error /= torch.linalg.vector_norm(oracle)
            assert error < 1e-5, error

            cached = gptq._WORKSPACE_CACHE[_cache_key()]
            assert cached.shape[1] == width
            # The newest width keeps its working set; anything else fits in
            # one more.
            assert _idle_bytes() <= 2 * _working_set_bytes(tuple(cached.shape))
    finally:
        gptq.release_workspace_cache()


def test_the_compatibility_alias_never_holds_an_evicted_allocation():
    gptq.release_workspace_cache()
    try:
        for width in (128, 160, 192):
            _run_width(width)

        cached = gptq._WORKSPACE_CACHE[_cache_key()]
        retained = {
            slot.tensor.untyped_storage().data_ptr() for slot in _slots()
        }
        assert cached.untyped_storage().data_ptr() in retained
        assert _idle_bytes() <= 2 * _working_set_bytes(tuple(cached.shape))
    finally:
        gptq.release_workspace_cache()


def test_a_repeated_width_reuses_its_slot_without_allocating():
    torch.manual_seed(3)
    gptq.release_workspace_cache()
    try:
        _run_width(256)
        slots_after_first = len(_slots())
        first_ids = {id(slot) for slot in _slots()}

        _run_width(256)

        assert len(_slots()) == slots_after_first
        assert {id(slot) for slot in _slots()} == first_ids
    finally:
        gptq.release_workspace_cache()


def test_concurrent_workers_of_one_width_keep_their_slots():
    """The bound must not evict a peer's slot while the pool is in use."""

    gptq.release_workspace_cache()
    device = torch.device("cpu")
    cols, rows, workers = 256, 64, 8
    barrier = threading.Barrier(workers)

    def hold() -> None:
        with gptq._lease_workspace(device, torch.bfloat16, cols, rows):
            barrier.wait()
            torch.zeros((rows, cols), dtype=torch.bfloat16)

    threads = [threading.Thread(target=hold) for _ in range(workers)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        slots = _slots()
        assert len(slots) == workers
        assert {tuple(slot.tensor.shape) for slot in slots} == {(rows, cols)}
    finally:
        gptq.release_workspace_cache()


def test_release_workspace_cache_still_drops_everything():
    gptq.release_workspace_cache()
    for width in (128, 160, 192):
        _run_width(width)
    assert _slots()

    gptq.release_workspace_cache()

    assert _slots() == []
    assert gptq._WORKSPACE_CACHE == {}
