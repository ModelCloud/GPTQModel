# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device is required for D2H benchmarks"
)


class PinnedBufferPool:
    """
    Simple pinned host memory pool keyed by (shape, dtype, layout).

    This lets us model the impact of reusing contiguous host buffers instead of
    re-allocating every time we enqueue a device-to-host transfer.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[Any, ...], list[torch.Tensor]] = {}
        self.hits = 0
        self.misses = 0

    def acquire(
        self, shape: torch.Size, dtype: torch.dtype, layout: torch.layout
    ) -> torch.Tensor:
        key = (tuple(shape), dtype, layout)
        bucket = self._store.get(key)
        if bucket:
            self.hits += 1
            return bucket.pop()
        self.misses += 1
        return torch.empty(
            shape,
            dtype=dtype,
            layout=layout,
            device="cpu",
            pin_memory=True,
        )

    def release(self, tensor: torch.Tensor) -> None:
        key = (tuple(tensor.shape), tensor.dtype, tensor.layout)
        self._store.setdefault(key, []).append(tensor)


def _aggregate_records(records: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {"samples": float(len(records))}
    if not records:
        return summary

    keys = {
        "enqueue_ms",
        "wait_ms",
        "total_ms",
        "copy_ms",
        "alloc_ms",
        "acquire_ms",
    }
    for key in keys:
        values = [entry[key] for entry in records if key in entry]
        if values:
            summary[key] = statistics.mean(values)
            summary[f"{key}_p50"] = statistics.median(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)
            if len(values) >= 2:
                summary[f"{key}_p95"] = statistics.quantiles(
                    values, n=100, method="inclusive"
                )[94]
    return summary


def _run_variant(
    tensors: list[torch.Tensor],
    *,
    warmup: int,
    runner: Callable[[torch.Tensor], dict[str, float]],
) -> dict[str, Any]:
    records: list[dict[str, float]] = []
    for idx, tensor in enumerate(tensors):
        record = runner(tensor)
        if idx >= warmup:
            records.append(record)
    summary = _aggregate_records(records)
    summary["raw"] = records
    return summary


def _run_sync_to_cpu(src: torch.Tensor, *, device: torch.device) -> dict[str, float]:
    """
    Baseline: blocking `.cpu()` call. Producer thread stalls until data lands on host.
    """
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    host = src.detach().cpu()
    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1e3
    # Access the tensor once so the copy is not elided by Python's lifetime analysis.
    _ = float(host.view(-1)[0].item())
    return {
        "enqueue_ms": total_ms,
        "wait_ms": 0.0,
        "total_ms": total_ms,
        "copy_ms": total_ms,
        "alloc_ms": 0.0,
    }


def _run_async_with_fresh_pinned(
    src: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    """
    Async copy each time with a new pinned buffer. Allocations dominate for small copies.
    """
    torch.cuda.synchronize(device)
    alloc_start = time.perf_counter()
    host = torch.empty_like(src, device="cpu", pin_memory=True)
    alloc_ms = (time.perf_counter() - alloc_start) * 1e3

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    host.copy_(src, non_blocking=True)
    end_evt.record()
    enqueue_end = time.perf_counter()
    enqueue_ms = (enqueue_end - alloc_start) * 1e3

    wait_start = time.perf_counter()
    end_evt.synchronize()
    wait_ms = (time.perf_counter() - wait_start) * 1e3
    total_ms = (time.perf_counter() - alloc_start) * 1e3
    copy_ms = start_evt.elapsed_time(end_evt)
    _ = float(host.view(-1)[0].item())

    return {
        "enqueue_ms": enqueue_ms,
        "wait_ms": wait_ms,
        "total_ms": total_ms,
        "copy_ms": copy_ms,
        "alloc_ms": alloc_ms,
    }


def _run_async_with_pool(
    src: torch.Tensor,
    *,
    device: torch.device,
    pool: PinnedBufferPool,
) -> dict[str, float]:
    """
    Async copy backed by reusable pinned buffers.
    """
    torch.cuda.synchronize(device)
    acquire_start = time.perf_counter()
    host = pool.acquire(src.shape, src.dtype, src.layout)
    acquire_ms = (time.perf_counter() - acquire_start) * 1e3

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    host.copy_(src, non_blocking=True)
    end_evt.record()
    enqueue_end = time.perf_counter()
    enqueue_ms = (enqueue_end - acquire_start) * 1e3

    wait_start = time.perf_counter()
    end_evt.synchronize()
    wait_ms = (time.perf_counter() - wait_start) * 1e3
    total_ms = (time.perf_counter() - acquire_start) * 1e3
    copy_ms = start_evt.elapsed_time(end_evt)
    _ = float(host.view(-1)[0].item())

    pool.release(host)
    return {
        "enqueue_ms": enqueue_ms,
        "wait_ms": wait_ms,
        "total_ms": total_ms,
        "copy_ms": copy_ms,
        "acquire_ms": acquire_ms,
    }


def _run_async_with_pool_stream(
    src: torch.Tensor,
    *,
    device: torch.device,
    pool: PinnedBufferPool,
    stream: torch.cuda.Stream,
) -> dict[str, float]:
    """
    Async copy on a dedicated stream with CUDA events, modelling an event-driven consumer.
    """
    torch.cuda.synchronize(device)
    acquire_start = time.perf_counter()
    host = pool.acquire(src.shape, src.dtype, src.layout)
    acquire_ms = (time.perf_counter() - acquire_start) * 1e3

    start_evt = torch.cuda.Event(enable_timing=True)
    done_evt = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        start_evt.record(stream)
        host.copy_(src, non_blocking=True)
        done_evt.record(stream)

    enqueue_end = time.perf_counter()
    enqueue_ms = (enqueue_end - acquire_start) * 1e3

    wait_start = time.perf_counter()
    done_evt.synchronize()
    wait_ms = (time.perf_counter() - wait_start) * 1e3
    total_ms = (time.perf_counter() - acquire_start) * 1e3
    copy_ms = start_evt.elapsed_time(done_evt)
    _ = float(host.view(-1)[0].item())

    pool.release(host)
    return {
        "enqueue_ms": enqueue_ms,
        "wait_ms": wait_ms,
        "total_ms": total_ms,
        "copy_ms": copy_ms,
        "acquire_ms": acquire_ms,
    }


def _bench_variants(
    tensors: list[torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    results["sync"] = _run_variant(
        tensors,
        warmup=warmup,
        runner=lambda src: _run_sync_to_cpu(src, device=device),
    )

    results["async_fresh"] = _run_variant(
        tensors,
        warmup=warmup,
        runner=lambda src: _run_async_with_fresh_pinned(src, device=device),
    )

    pool_reuse = PinnedBufferPool()
    hits_before = pool_reuse.hits
    misses_before = pool_reuse.misses
    results["async_pool"] = _run_variant(
        tensors,
        warmup=warmup,
        runner=lambda src: _run_async_with_pool(src, device=device, pool=pool_reuse),
    )
    results["async_pool"]["pool_hits"] = float(pool_reuse.hits - hits_before)
    results["async_pool"]["pool_misses"] = float(pool_reuse.misses - misses_before)

    pool_stream = PinnedBufferPool()
    stream = torch.cuda.Stream(device=device)
    hits_before_stream = pool_stream.hits
    misses_before_stream = pool_stream.misses
    results["async_pool_stream"] = _run_variant(
        tensors,
        warmup=warmup,
        runner=lambda src: _run_async_with_pool_stream(
            src, device=device, pool=pool_stream, stream=stream
        ),
    )
    results["async_pool_stream"]["pool_hits"] = float(
        pool_stream.hits - hits_before_stream
    )
    results["async_pool_stream"]["pool_misses"] = float(
        pool_stream.misses - misses_before_stream
    )

    return results


def _log_summary(header: str, metrics: dict[str, dict[str, Any]]) -> None:
    print(f"\n[CUDA->CPU D2H] {header}")
    for name, stats in metrics.items():
        enqueue = stats.get("enqueue_ms", float("nan"))
        enqueue_p95 = stats.get("enqueue_ms_p95", float("nan"))
        wait = stats.get("wait_ms", float("nan"))
        total = stats.get("total_ms", float("nan"))
        total_p95 = stats.get("total_ms_p95", float("nan"))
        total_min = stats.get("total_ms_min", float("nan"))
        total_max = stats.get("total_ms_max", float("nan"))
        copy_ms = stats.get("copy_ms", float("nan"))
        extra = ""
        if "alloc_ms" in stats:
            extra += f" | alloc {stats['alloc_ms']:.3f} ms"
        if "acquire_ms" in stats:
            extra += f" | acquire {stats['acquire_ms']:.3f} ms"
        if "pool_hits" in stats:
            extra += (
                f" | pool hits/misses {int(stats['pool_hits'])}/"
                f"{int(stats['pool_misses'])}"
            )
        print(
            f"  {name:>18}: enqueue {enqueue:7.3f} ms (p95 {enqueue_p95:7.3f}) | "
            f"wait {wait:7.3f} ms | total {total:7.3f} ms (p95 {total_p95:7.3f}, "
            f"range {total_min:7.3f}-{total_max:7.3f}) | copy {copy_ms:7.3f} ms{extra}"
        )


def _make_constant_tensors(
    *,
    numel: int,
    total: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    base = torch.empty(numel, dtype=dtype, device=device)
    base.uniform_()
    return [base] * total


def _make_shape_cycle(
    *,
    numel: int,
    total: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    divisors = [1, 2, 4, 8, 16]
    tensors: list[torch.Tensor] = []
    idx = 0
    while len(tensors) < total:
        factor = divisors[idx % len(divisors)]
        if factor == 1 or numel % factor == 0:
            main = max(1, numel // factor)
            shape = (main, factor) if factor > 1 else (numel,)
        else:
            shape = (numel,)
        tensor = torch.empty(shape, dtype=dtype, device=device)
        tensor.uniform_()
        tensors.append(tensor)
        idx += 1
    return tensors


def _bytes_to_mib(size_bytes: int) -> float:
    return size_bytes / (1024**2)


def test_even_d2h_latency_profile():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    dtype = torch.float16
    element_size = torch.empty((), dtype=dtype).element_size()

    warmup = 2
    steps = 8
    total_iters = warmup + steps

    sizes_mib = [0.125, 0.5, 1, 4, 8, 16, 32]
    summaries: dict[float, dict[str, dict[str, Any]]] = {}

    for size_mib in sizes_mib:
        size_bytes = int(size_mib * 1024**2)
        numel = max(1, size_bytes // element_size)
        tensors = _make_constant_tensors(
            numel=numel, total=total_iters, device=device, dtype=dtype
        )
        metrics = _bench_variants(tensors, device=device, warmup=warmup)
        summaries[size_mib] = metrics
        _log_summary(f"{size_mib:.3f} MiB", metrics)

    # Expect pooling to drastically shrink producer stall time once buffers are warm.
    for size_mib, metrics in summaries.items():
        async_pool = metrics["async_pool"]
        async_fresh = metrics["async_fresh"]
        assert async_pool["pool_hits"] >= steps, "Pool should be re-used after warmup"
        pooled = async_pool["enqueue_ms"]
        fresh = async_fresh["enqueue_ms"]
        assert (
            pooled <= fresh * 0.95
            or math.isclose(pooled, fresh, rel_tol=0.05, abs_tol=0.01)
        ), (
            f"Pooled enqueue should be meaningfully faster or comparable for {size_mib:.3f} MiB "
            f"(pooled {pooled:.3f} ms vs fresh {fresh:.3f} ms)"
        )

    # Async streaming should reduce producer blocking for at least the larger tensors.
    streaming_benefit_sizes = [
        size
        for size, metrics in summaries.items()
        if metrics["async_pool_stream"]["enqueue_ms"]
        < metrics["sync"]["enqueue_ms"] * 0.5
    ]
    assert streaming_benefit_sizes, "Expected event-driven streaming to beat sync copies"

    # Even with async dispatch, total copy time remains comparable to the sync baseline.
    for size_mib, metrics in summaries.items():
        sync_total = metrics["sync"]["total_ms"]
        async_total = metrics["async_pool_stream"]["total_ms"]
        upper_factor = 1.6 if sync_total < 0.2 else 1.25
        assert sync_total * 0.05 <= async_total <= sync_total * upper_factor, (
            f"Data readiness latency should stay within reasonable bounds (size={size_mib:.3f} MiB, "
            f"sync={sync_total:.3f} ms, async={async_total:.3f} ms)"
        )


def test_even_d2h_pool_shape_sensitivity():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    dtype = torch.float16
    element_size = torch.empty((), dtype=dtype).element_size()

    warmup = 2
    steps = 8
    total_iters = warmup + steps

    size_mib = 8
    size_bytes = int(size_mib * 1024**2)
    numel = max(1, size_bytes // element_size)

    tensors = _make_shape_cycle(
        numel=numel, total=total_iters, device=device, dtype=dtype
    )

    async_fresh = _run_variant(
        tensors,
        warmup=warmup,
        runner=lambda src: _run_async_with_fresh_pinned(src, device=device),
    )

    pool = PinnedBufferPool()
    hits_before, misses_before = pool.hits, pool.misses
    async_pool = _run_variant(
        tensors,
        warmup=warmup,
        runner=lambda src: _run_async_with_pool(src, device=device, pool=pool),
    )
    async_pool["pool_hits"] = float(pool.hits - hits_before)
    async_pool["pool_misses"] = float(pool.misses - misses_before)

    _log_summary(
        "8.000 MiB (shape cycle)",
        {"async_fresh": async_fresh, "async_pool": async_pool},
    )

    assert async_pool["pool_hits"] <= async_pool["pool_misses"], (
        "Shape cycling should not produce more hits than misses; otherwise reuse dominates"
    )

    # Pooling should fall back to fresh allocations without exploding unboundedly.
    ratio = async_pool["enqueue_ms"] / async_fresh["enqueue_ms"]
    assert 1.0 <= ratio <= 120.0, (
        "Pooling under shape churn should be slower but stay within a reasonable bound "
        f"(ratio={ratio:.2f})"
    )
    assert async_pool.get("acquire_ms", 0.0) >= async_fresh.get("alloc_ms", 0.0) * 2, (
        "Pooling slowdown should be attributable to expensive host buffer acquisition"
    )


def test_even_d2h_request_wall():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    dtype = torch.float16
    element_size = torch.empty((), dtype=dtype).element_size()

    size_mib = 16
    size_bytes = int(size_mib * 1024**2)
    numel = max(1, size_bytes // element_size)

    max_transfers = 8
    device_buffers = [
        torch.empty(numel, dtype=dtype, device=device).uniform_()
        for _ in range(max_transfers)
    ]
    host_buffers = [
        torch.empty_like(device_buffers[0], device="cpu", pin_memory=True)
        for _ in range(max_transfers)
    ]

    def measure_serial(n: int) -> float:
        stream = torch.cuda.Stream(device=device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            for idx in range(n):
                host_buffers[idx].copy_(device_buffers[idx], non_blocking=True)
        torch.cuda.synchronize(device)
        return time.perf_counter() - t0

    def measure_parallel(n: int) -> float:
        streams = [torch.cuda.Stream(device=device) for _ in range(n)]
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for idx in range(n):
            with torch.cuda.stream(streams[idx]):
                host_buffers[idx].copy_(device_buffers[idx], non_blocking=True)
        torch.cuda.synchronize(device)
        return time.perf_counter() - t0

    serial_times: dict[int, float] = {}
    parallel_times: dict[int, float] = {}
    for n in (1, 2, 4, 8):
        serial_times[n] = measure_serial(n)
        parallel_times[n] = measure_parallel(n)
        total_gib = (n * size_bytes) / (1024**3)
        print(
            f"[D2H wall] {n} transfers of {size_mib:.1f} MiB -> "
            f"serial {serial_times[n]:.4f}s ({total_gib/serial_times[n]:.2f} GiB/s), "
            f"parallel {parallel_times[n]:.4f}s ({total_gib/parallel_times[n]:.2f} GiB/s)"
        )

    baseline = parallel_times[1]
    stall_observed = any(parallel_times[n] >= baseline * n * 0.8 for n in (2, 4, 8))
    assert stall_observed, "Expected concurrent D2H copies to serialize onto one engine"

    # Serial vs parallel should stay within reasonable bounds for all batch sizes.
    for n in (1, 2, 4, 8):
        ratio = parallel_times[n] / serial_times[n]
        assert 0.8 <= ratio <= 1.3, (
            f"Parallel vs serial time deviated unexpectedly for {n} transfers: ratio={ratio:.2f}"
        )
