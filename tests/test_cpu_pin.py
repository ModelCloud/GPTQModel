# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import os
import time
import types
from typing import Callable, List, Tuple
from unittest import mock

import pytest
import torch

from gptqmodel.utils.threadx import DeviceThreadPool


pytestmark = pytest.mark.ci


def _affinity_supported() -> bool:
    return hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")


def _available_cpu_affinity() -> set[int]:
    if not hasattr(os, "sched_getaffinity"):
        return set()
    return set(os.sched_getaffinity(0))


def _run_in_pool(
    *,
    with_pinning: bool,
    worker_count: int = 1,
    heavyweight_fn: Callable[..., Tuple[float, set[int]]] | None = None,
    loops: int = 5,
    pin_cpu_workers: bool | None = None,
) -> Tuple[float, set[int], int | None]:
    """Execute a callable on the CPU worker and collect timing and affinity."""
    context: contextlib.AbstractContextManager
    if with_pinning:
        context = contextlib.nullcontext()
    else:
        context = mock.patch("os.sched_setaffinity", new=lambda pid, mask: None)

    if pin_cpu_workers is None:
        pin_cpu_workers = with_pinning

    heavy = heavyweight_fn or _mem_bandwidth_task
    cpu = torch.device("cpu")

    with context:
        pool = DeviceThreadPool(
            devices=[cpu],
            workers={"cpu": worker_count},
            pin_cpu_workers=pin_cpu_workers,
        )
        a = b = out = None
        try:
            worker = pool._worker_groups["cpu"][0]
            target_core = worker._target_cpu_core
            if target_core is None and with_pinning:
                pytest.skip("Thread pinning unavailable on this platform or affinity mask is empty")

            bytes_per_elem = max(1, torch.finfo(torch.float16).bits // 8)
            elems = (200 * 1024 * 1024) // bytes_per_elem
            a = torch.ones(elems, dtype=torch.float16, device=cpu)
            b = torch.full((elems,), 0.5, dtype=torch.float16, device=cpu)
            out = torch.empty_like(a)

            # Warm-up to reduce cold-start variance.
            pool.do(cpu, heavy, a, b, out)

            loops = max(1, int(loops))
            durations: List[float] = []
            affinities: List[set[int]] = []
            for _ in range(loops):
                duration, affinity = pool.do(cpu, heavy, a, b, out)
                durations.append(duration)
                affinities.append(affinity)

            best_idx = min(range(len(durations)), key=durations.__getitem__)
            duration = durations[best_idx]
            affinity = affinities[best_idx]
        finally:
            pool.shutdown(wait=True)
            for tensor in (a, b, out):
                if tensor is not None:
                    del tensor

    return duration, affinity, target_core


def _mem_bandwidth_task(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
) -> Tuple[float, set[int]]:
    start = time.perf_counter()
    torch.add(lhs, rhs, out=out)
    out.add_(rhs)
    out.add_(1.0)
    elapsed = time.perf_counter() - start
    affinity = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()
    return elapsed, affinity


@pytest.mark.skipif(not _affinity_supported(), reason="POSIX CPU affinity APIs are unavailable")
def test_device_thread_pool_cpu_worker_is_pinned():
    duration, affinity, target_core = _run_in_pool(with_pinning=True, pin_cpu_workers=True)
    assert duration >= 0.0
    assert target_core is not None
    assert affinity == {target_core}


@pytest.mark.skipif(not _affinity_supported(), reason="POSIX CPU affinity APIs are unavailable")
@pytest.mark.skipif(len(_available_cpu_affinity()) < 2, reason="requires >=2 CPU cores to compare affinity masks")
def test_cpu_thread_pinning_benchmark_vs_unpinned():
    baseline_affinity = _available_cpu_affinity()

    pinned_duration, pinned_affinity, target_core = _run_in_pool(with_pinning=True, pin_cpu_workers=True)
    assert target_core is not None
    assert pinned_affinity == {target_core}

    unpinned_duration, unpinned_affinity, _ = _run_in_pool(with_pinning=False)
    assert len(unpinned_affinity) >= 1
    assert pinned_duration > 0.0 and unpinned_duration > 0.0

    # Affinity mask should widen when pinning is disabled; expect default mask restoration.
    assert unpinned_affinity == baseline_affinity

    print(
        f"Pinned duration: {pinned_duration:.4f}s (affinity {sorted(pinned_affinity)}); "
        f"Unpinned duration: {unpinned_duration:.4f}s (affinity {sorted(unpinned_affinity)})"
    )


def test_plan_worker_affinity_skips_accelerators_when_not_pinned():
    pool = DeviceThreadPool.__new__(DeviceThreadPool)
    pool._resolve_workers_for_device = types.MethodType(lambda self, dev, table: 1, pool)

    def _fake_key(self, dev):
        suffix = f":{dev.index}" if dev.index is not None else ""
        return f"{dev.type}{suffix}"

    pool._key = types.MethodType(_fake_key, pool)

    cuda_device = torch.device("cuda", 0)

    plan_disabled = DeviceThreadPool._plan_worker_affinity(
        pool,
        devices=[cuda_device],
        worker_table={},
        pin_cpu_workers=False,
        pin_accelerator_workers=False,
    )
    assert plan_disabled == {}

    plan_enabled = DeviceThreadPool._plan_worker_affinity(
        pool,
        devices=[cuda_device],
        worker_table={},
        pin_cpu_workers=False,
        pin_accelerator_workers=True,
    )

    key = pool._key(cuda_device)
    assert (key, 0) in plan_enabled
    assigned_core = plan_enabled[(key, 0)]
    assert assigned_core is None or isinstance(assigned_core, int)
