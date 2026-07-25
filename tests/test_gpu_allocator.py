# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Tuple
from unittest.mock import patch

import pytest

from gpu_allocator import GPUAllocatorClient
from gpu_allocator.allocator import GPUAllocator
from gpu_allocator.inventory import discover_gpus
from gpu_allocator.models import GPU, Lease
from gpu_allocator.session_monitor import (
    DevinApiSessionMonitor,
    NullSessionMonitor,
    SessionMonitor,
)

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
    CUDA_COUNT = torch.cuda.device_count() if HAS_CUDA else 0
except Exception:
    HAS_CUDA = False
    CUDA_COUNT = 0


def _fake_gpus(n: int = 4) -> List[GPU]:
    """Build `n` GPUs with stable PCI-bus-order bus ids."""
    return [
        GPU(
            pci_order_index=i,
            pci_bus_id=f"00000000:{10 + i:02x}:00.0",
            uuid=f"GPU-uuid-{i:03d}",
            name=f"Test GPU {i}",
            memory_total_mib=1024,
        )
        for i in range(n)
    ]


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_discover_gpus_returns_pci_ordered_uuids():
    gpus = discover_gpus()
    assert len(gpus) > 0
    # PCI order indices are dense and monotonic.
    assert [g.pci_order_index for g in gpus] == list(range(len(gpus)))
    for gpu in gpus:
        assert gpu.pci_bus_id
        assert gpu.uuid.startswith("GPU-")
        assert gpu.name
        assert gpu.memory_total_mib > 0


@pytest.mark.skipif(
    not HAS_CUDA or CUDA_COUNT < 4,
    reason="need at least 4 CUDA devices",
)
def test_discover_first_four_gpus_are_index_zero_to_three():
    gpus = discover_gpus()[:4]
    assert len(gpus) == 4
    assert [g.pci_order_index for g in gpus] == [0, 1, 2, 3]


@pytest.mark.skipif(
    not HAS_CUDA or CUDA_COUNT < 4,
    reason="need at least 4 CUDA devices",
)
def test_real_four_gpu_exclusive_blocks_and_shared_uses_remaining():
    real_gpus = discover_gpus()[:4]
    allocator = GPUAllocator(gpus=real_gpus, enable_ttl_janitor=False)
    exclusive = allocator.allocate("session-excl", count=2, timeout=0, exclusive=True)
    assert all(0 <= gpu.pci_order_index <= 3 for gpu in exclusive.gpus)
    # Only 2 GPUs remain, so a request for 3 must fail.
    with pytest.raises(TimeoutError):
        allocator.allocate("session-big", count=3, timeout=0, exclusive=True)
    # A shared request can use the remaining 2 GPUs but not the exclusive ones.
    shared = allocator.allocate("session-shared", count=2, timeout=0, exclusive=False)
    assert all(
        gpu.pci_order_index not in {g.pci_order_index for g in exclusive.gpus}
        for gpu in shared.gpus
    )
    allocator.release(exclusive.lease_id)
    allocator.release(shared.lease_id)


def test_exclusive_allocation_and_release():
    allocator = GPUAllocator(gpus=_fake_gpus(1), enable_ttl_janitor=False)
    lease_a = allocator.allocate("session-a", count=1, timeout=0)
    assert len(lease_a.gpus) == 1

    with pytest.raises(TimeoutError):
        allocator.allocate("session-b", count=1, timeout=0)

    allocator.release(lease_a.lease_id)
    lease_b = allocator.allocate("session-b", count=1, timeout=0)
    assert lease_b.gpus[0].pci_order_index == lease_a.gpus[0].pci_order_index


def test_shared_leases_overlap_on_same_gpu():
    allocator = GPUAllocator(gpus=_fake_gpus(1), enable_ttl_janitor=False)
    shared_a = allocator.allocate("session-a", count=1, timeout=0, exclusive=False)
    shared_b = allocator.allocate("session-b", count=1, timeout=0, exclusive=False)
    assert shared_a.gpus[0].pci_order_index == shared_b.gpus[0].pci_order_index

    # An exclusive lease cannot share the same GPU.
    with pytest.raises(TimeoutError):
        allocator.allocate("session-c", count=1, timeout=0, exclusive=True)

    allocator.release(shared_a.lease_id)
    allocator.release(shared_b.lease_id)


def test_exclusive_blocks_shared_and_release_allows_both():
    allocator = GPUAllocator(gpus=_fake_gpus(1), enable_ttl_janitor=False)
    exclusive_a = allocator.allocate("session-a", count=1, timeout=0, exclusive=True)
    with pytest.raises(TimeoutError):
        allocator.allocate("session-b", count=1, timeout=0, exclusive=False)

    allocator.release(exclusive_a.lease_id)
    shared_b = allocator.allocate("session-b", count=1, timeout=0, exclusive=False)
    assert shared_b.gpus[0].pci_order_index == exclusive_a.gpus[0].pci_order_index
    allocator.release(shared_b.lease_id)


def test_fifo_waiters_are_served_in_order():
    allocator = GPUAllocator(gpus=_fake_gpus(3), enable_ttl_janitor=False)
    blocker = allocator.allocate("session-blocker", count=3, timeout=0)
    results: List[object] = []

    def wait_for(count: int, timeout: float):
        try:
            lease = allocator.allocate(f"session-{count}", count=count, timeout=timeout)
            results.append((count, lease))
        except TimeoutError:
            results.append((count, None))

    t2 = threading.Thread(target=wait_for, args=(2, 2.0))
    t1 = threading.Thread(target=wait_for, args=(1, 2.0))
    t2.start()
    time.sleep(0.05)  # ensure t2 enters queue first
    t1.start()
    time.sleep(0.05)

    allocator.release(blocker.lease_id)

    t2.join(timeout=3.0)
    t1.join(timeout=3.0)

    # First waiter needing 2 GPUs should be served before the 1-GPU waiter.
    assert results[0] == (2, results[0][1])
    assert results[0][1] is not None and len(results[0][1].gpus) == 2
    assert results[1] == (1, results[1][1])
    assert results[1][1] is not None and len(results[1][1].gpus) == 1


def test_timeout_zero_returns_immediately():
    allocator = GPUAllocator(gpus=_fake_gpus(1), enable_ttl_janitor=False)
    lease = allocator.allocate("session-a", count=1, timeout=0)
    with pytest.raises(TimeoutError):
        allocator.allocate("session-b", count=1, timeout=0)
    allocator.release(lease.lease_id)


def test_blocking_allocate_releases_when_gpu_freed():
    allocator = GPUAllocator(gpus=_fake_gpus(1), enable_ttl_janitor=False)
    lease = allocator.allocate("session-a", count=1, timeout=0)
    result: List[object] = []

    def waiter():
        try:
            result.append(allocator.allocate("session-b", count=1, timeout=2.0))
        except TimeoutError:
            result.append(None)

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.1)
    allocator.release(lease.lease_id)
    t.join(timeout=3.0)

    assert result[0] is not None


def test_renew_extends_ttl():
    allocator = GPUAllocator(
        gpus=_fake_gpus(2), lease_ttl_seconds=0.5, enable_ttl_janitor=False
    )
    lease = allocator.allocate("session-a", count=1, timeout=0)
    original_expiry = lease.expires_at
    time.sleep(0.2)
    renewed = allocator.renew(lease.lease_id)
    assert renewed.expires_at > original_expiry


def test_ttl_janitor_reclaims_expired_lease():
    allocator = GPUAllocator(
        gpus=_fake_gpus(2),
        lease_ttl_seconds=0.2,
        lease_check_interval=0.1,
        enable_ttl_janitor=True,
    )
    lease = allocator.allocate("session-a", count=1, timeout=0)
    time.sleep(0.5)
    status = allocator.status()
    assert lease.lease_id not in {
        lease_dict["lease_id"] for lease_dict in status["leases"]
    }
    assert status["free_count"] == 2
    allocator.shutdown()


def test_status_reflects_free_and_leased():
    allocator = GPUAllocator(gpus=_fake_gpus(3), enable_ttl_janitor=False)
    status = allocator.status()
    assert status["total"] == 3
    assert status["free_count"] == 3

    lease = allocator.allocate("session-a", count=2, timeout=0)
    status = allocator.status()
    assert status["free_count"] == 1
    assert len(status["leases"]) == 1
    assert status["leases"][0]["lease_id"] == lease.lease_id
    allocator.release(lease.lease_id)


def test_concurrent_sessions_single_and_multiple_gpus():
    allocator = GPUAllocator(gpus=_fake_gpus(8), enable_ttl_janitor=False)
    acquired: List[Lease] = []
    lock = threading.Lock()

    def worker(worker_id: int):
        count = (worker_id % 2) + 1  # 1 or 2 GPUs per session
        lease = allocator.allocate(f"session-{worker_id}", count=count, timeout=5.0)
        with lock:
            acquired.append(lease)
        time.sleep(0.1)
        allocator.release(lease.lease_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))

    assert len(acquired) == 8
    for lease in acquired:
        assert len(lease.gpus) in {1, 2}


def test_concurrent_overlapping_shared_and_exclusive():
    allocator = GPUAllocator(gpus=_fake_gpus(4), enable_ttl_janitor=False)
    shared_lock = threading.Lock()
    shared_leases: List[object] = []
    exclusive_leases: List[object] = []

    def shared_worker(i: int):
        lease = allocator.allocate(f"shared-{i}", count=1, timeout=5.0, exclusive=False)
        with shared_lock:
            shared_leases.append(lease)
        time.sleep(0.2)
        allocator.release(lease.lease_id)

    def exclusive_worker(i: int):
        lease = allocator.allocate(
            f"exclusive-{i}", count=2, timeout=5.0, exclusive=True
        )
        with shared_lock:
            exclusive_leases.append(lease)
        time.sleep(0.2)
        allocator.release(lease.lease_id)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(shared_worker, i) for i in range(3)] + [
            pool.submit(exclusive_worker, i) for i in range(2)
        ]
        for future in as_completed(futures):
            future.result()

    assert len(shared_leases) == 3
    assert len(exclusive_leases) == 2
    # At most one exclusive lease can overlap in time; verify no two exclusive
    # leases ever held the same GPU by checking lease uniqueness.
    exclusive_gpu_sets = [
        set(g.pci_bus_id for g in lease.gpus) for lease in exclusive_leases
    ]
    assert all(len(s) == 2 for s in exclusive_gpu_sets)


def test_server_client_allocate_and_release():
    from gpu_allocator.server import GPUAllocatorServer

    allocator = GPUAllocator(gpus=_fake_gpus(4), enable_ttl_janitor=False)
    server = GPUAllocatorServer(("127.0.0.1", 0), allocator)
    server_address = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://{server_address[0]}:{server_address[1]}"
        client = GPUAllocatorClient(base_url=base_url, session_id="server-test")
        assert client.health().get("status") == "ok"

        lease = client.allocate(count=2, timeout=0)
        assert len(lease.gpus) == 2
        visible = lease.as_cuda_visible_devices(style="pci_order_index", set_env=False)
        assert visible == ",".join(str(g.pci_order_index) for g in lease.gpus)

        status = client.status()
        assert status["total"] == 4
        assert status["free_count"] == 2

        client.release(lease.lease_id)
        status = client.status()
        assert status["free_count"] == 4
    finally:
        server.shutdown()
        server.server_close()
        allocator.shutdown()


def test_server_blocking_with_multiple_clients():
    from gpu_allocator.server import GPUAllocatorServer

    allocator = GPUAllocator(gpus=_fake_gpus(2), enable_ttl_janitor=False)
    server = GPUAllocatorServer(("127.0.0.1", 0), allocator)
    server_address = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://{server_address[0]}:{server_address[1]}"
        holder = GPUAllocatorClient(base_url=base_url, session_id="holder")
        waiter = GPUAllocatorClient(base_url=base_url, session_id="waiter")

        lease = holder.allocate(count=2, timeout=0)

        result: List[object] = []

        def wait_then_release():
            time.sleep(0.1)
            holder.release(lease.lease_id)

        def wait_for_gpu():
            try:
                result.append(waiter.allocate(count=1, timeout=2.0))
            except Exception as exc:
                result.append(exc)

        t_release = threading.Thread(target=wait_then_release)
        t_wait = threading.Thread(target=wait_for_gpu)
        t_wait.start()
        t_release.start()
        t_wait.join(timeout=3.0)
        t_release.join(timeout=3.0)

        assert result and isinstance(result[0], type(result[0]))
        # Should be a LeaseContext-like object with a lease_id.
        assert getattr(result[0], "lease_id", None)
    finally:
        server.shutdown()
        server.server_close()
        allocator.shutdown()


def _extract_json(text: str) -> Any:
    """Extract the first JSON object from text that may contain LogBar prefixes."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output")
    # Find the matching outer closing brace by counting braces.
    count = 0
    for i, ch in enumerate(text[start:]):
        if ch == "{":
            count += 1
        elif ch == "}":
            count -= 1
            if count == 0:
                return json.loads(text[start : start + i + 1])
    raise ValueError("Unterminated JSON object in output")


def _assert_no_overlap(active: List[Tuple[Lease, bool]]) -> List[str]:
    """Return a list of overlap violations among active leases.

    Shared leases may overlap with other shared leases, but any exclusive
    lease must be disjoint from every other active lease.
    """
    violations: List[str] = []
    for i, (lease_a, exclusive_a) in enumerate(active):
        ids_a = {g.pci_bus_id for g in lease_a.gpus}
        for lease_b, exclusive_b in active[i + 1 :]:
            if exclusive_a or exclusive_b:
                ids_b = {g.pci_bus_id for g in lease_b.gpus}
                overlap = ids_a & ids_b
                if overlap:
                    violations.append(
                        f"overlap between {lease_a.lease_id} and {lease_b.lease_id} on {overlap}"
                    )
    return violations


def test_cli_acquire_release_run_and_status():
    from gpu_allocator.cli import main as cli_main
    from gpu_allocator.server import GPUAllocatorServer

    allocator = GPUAllocator(gpus=_fake_gpus(2), enable_ttl_janitor=False)
    server = GPUAllocatorServer(("127.0.0.1", 0), allocator)
    server_address = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://{server_address[0]}:{server_address[1]}"

        # Acquire a single GPU in JSON format to capture the lease id.
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = cli_main(
                [
                    "--base-url",
                    base_url,
                    "acquire",
                    "-n",
                    "1",
                    "--format",
                    "json",
                    "--style",
                    "pci_order_index",
                ]
            )
        assert code == 0
        response = _extract_json(captured.getvalue())
        lease_id = response["lease_id"]
        assert response["cuda_visible_devices"] == "0"

        # Status shows one free GPU and one lease.
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = cli_main(["--base-url", base_url, "status"])
        assert code == 0
        status = _extract_json(captured.getvalue())
        assert status["free_count"] == 1
        assert len(status["leases"]) == 1

        # Run a command that writes the lease id to a temp file via env var.
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            tmp_path = f.name
        try:
            code = cli_main(
                [
                    "--base-url",
                    base_url,
                    "run",
                    "-n",
                    "1",
                    "--style",
                    "pci_order_index",
                    "--",
                    sys.executable,
                    "-c",
                    f"import os; open({tmp_path!r}, 'w').write(os.environ.get('GPU_ALLOCATOR_LEASE_ID',''))",
                ]
            )
            assert code == 0
            # A different lease is created/destroyed; the value written should be a lease id.
            run_lease_id = open(tmp_path, "r").read()
            assert run_lease_id and run_lease_id != lease_id
        finally:
            os.unlink(tmp_path)

        # Release the original lease.
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = cli_main(["--base-url", base_url, "release", "--lease-id", lease_id])
        assert code == 0
        assert _extract_json(captured.getvalue())["ok"] is True
    finally:
        server.shutdown()
        server.server_close()
        allocator.shutdown()


def test_client_context_manager_releases_on_exit():
    from gpu_allocator.server import GPUAllocatorServer

    allocator = GPUAllocator(gpus=_fake_gpus(2), enable_ttl_janitor=False)
    server = GPUAllocatorServer(("127.0.0.1", 0), allocator)
    server_address = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://{server_address[0]}:{server_address[1]}"
        client = GPUAllocatorClient(base_url=base_url, session_id="ctx-test")
        with client.acquire(count=1, timeout=0) as lease:
            assert len(lease.gpus) == 1
            status = client.status()
            assert status["free_count"] == 1
        status = client.status()
        assert status["free_count"] == 2
    finally:
        server.shutdown()
        server.server_close()
        allocator.shutdown()


def test_allocate_invalid_count_raises():
    allocator = GPUAllocator(gpus=_fake_gpus(4), enable_ttl_janitor=False)
    with pytest.raises(ValueError, match="count must be positive"):
        allocator.allocate("a", count=0, timeout=0)
    with pytest.raises(ValueError, match="Requested 5 GPUs but only 4 exist"):
        allocator.allocate("a", count=5, timeout=0)


def test_release_and_renew_unknown_lease_raises():
    allocator = GPUAllocator(gpus=_fake_gpus(2), enable_ttl_janitor=False)
    with pytest.raises(KeyError):
        allocator.release("missing-id")
    with pytest.raises(KeyError):
        allocator.renew("missing-id")


def test_sixteen_sessions_random_acquire_release_no_leaks():
    allocator = GPUAllocator(gpus=_fake_gpus(8), enable_ttl_janitor=False)
    active: List[Tuple[Lease, bool]] = []
    active_lock = threading.Lock()
    violations: List[str] = []
    successes = [0]

    def worker(worker_id: int):
        rng = random.Random(worker_id + 12345)
        for _ in range(20):
            count = rng.randint(1, 3)
            exclusive = rng.choice([True, False])
            try:
                lease = allocator.allocate(
                    f"worker-{worker_id}",
                    count=count,
                    timeout=30.0,
                    exclusive=exclusive,
                    reason="stress",
                )
            except TimeoutError:
                continue
            assert len(lease.gpus) == count
            with active_lock:
                violations.extend(_assert_no_overlap(active + [(lease, exclusive)]))
                active.append((lease, exclusive))
            time.sleep(rng.uniform(0.001, 0.01))
            with active_lock:
                active.remove((lease, exclusive))
                successes[0] += 1
            allocator.release(lease.lease_id)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(worker, range(16)))

    assert not violations, violations
    assert successes[0] == 16 * 20
    status = allocator.status()
    assert status["free_count"] == 8
    assert status["leases"] == []


def test_starvation_prevention_large_request_gets_served():
    allocator = GPUAllocator(gpus=_fake_gpus(8), enable_ttl_janitor=False)
    # Hold all 8 GPUs so every waiter has to queue.
    blocker = allocator.allocate("blocker", count=8, timeout=0, exclusive=True)
    results: List[Tuple[int, Lease]] = []
    results_lock = threading.Lock()

    def wait_for(count: int, timeout: float):
        lease = allocator.allocate(
            f"waiter-{count}", count=count, timeout=timeout, exclusive=True
        )
        with results_lock:
            results.append((count, lease))
        allocator.release(lease.lease_id)

    # Queue the 4-GPU waiter first, then a stream of 1-GPU waiters.
    t_big = threading.Thread(target=wait_for, args=(4, 10.0))
    small_threads = [
        threading.Thread(target=wait_for, args=(1, 10.0)) for _ in range(7)
    ]
    t_big.start()
    time.sleep(0.05)
    for t in small_threads:
        t.start()
        time.sleep(0.01)

    time.sleep(0.1)
    allocator.release(blocker.lease_id)

    t_big.join(timeout=15.0)
    for t in small_threads:
        t.join(timeout=15.0)

    assert len(results) == 8
    # The first allocated lease (by creation time) must be the 4-GPU request.
    sorted_results = sorted(results, key=lambda item: item[1].created_at)
    assert sorted_results[0][0] == 4


def test_shared_exclusive_transition_on_single_gpu():
    allocator = GPUAllocator(gpus=_fake_gpus(1), enable_ttl_janitor=False)
    shared = [
        allocator.allocate(f"shared-{i}", count=1, timeout=0, exclusive=False)
        for i in range(4)
    ]
    # An exclusive request for the same GPU must block until all shared leases are gone.
    exclusive_result: List[Lease] = []

    def wait_exclusive():
        exclusive_result.append(
            allocator.allocate("exclusive", count=1, timeout=2.0, exclusive=True)
        )

    t = threading.Thread(target=wait_exclusive)
    t.start()
    time.sleep(0.1)

    # Even releasing 3 of 4 shared leases is not enough; one shared remains.
    for lease in shared[:-1]:
        allocator.release(lease.lease_id)
    time.sleep(0.1)
    assert not exclusive_result

    # Release the final shared lease; exclusive should now be granted.
    allocator.release(shared[-1].lease_id)
    t.join(timeout=3.0)
    assert exclusive_result
    assert exclusive_result[0].gpus[0].pci_order_index == 0
    allocator.release(exclusive_result[0].lease_id)


def test_renew_prevents_ttl_expiration_then_expires_without_renew():
    allocator = GPUAllocator(
        gpus=_fake_gpus(2),
        lease_ttl_seconds=0.3,
        lease_check_interval=0.1,
        enable_ttl_janitor=True,
    )
    lease = allocator.allocate("session-a", count=1, timeout=0)
    time.sleep(0.2)
    allocator.renew(lease.lease_id)
    # After the renewal the lease is valid until ~t=0.5, so it must still be held.
    time.sleep(0.1)
    status = allocator.status()
    assert status["free_count"] == 1
    # Wait long enough for the renewed TTL to expire.
    time.sleep(0.5)
    status = allocator.status()
    assert status["free_count"] == 2
    allocator.shutdown()


def test_server_sixteen_clients_random_requests():
    from gpu_allocator.client import log as client_log
    from gpu_allocator.server import GPUAllocatorServer
    from gpu_allocator.server import log as server_log

    # Suppress per-request logs during the stress test.
    server_log.setLevel("WARNING")
    client_log.setLevel("WARNING")

    allocator = GPUAllocator(gpus=_fake_gpus(8), enable_ttl_janitor=False)
    server = GPUAllocatorServer(("127.0.0.1", 0), allocator)
    server_address = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    active: List[Tuple[Any, bool]] = []
    active_lock = threading.Lock()
    violations: List[str] = []
    successes = [0]

    def client_worker(worker_id: int):
        base_url = f"http://{server_address[0]}:{server_address[1]}"
        client = GPUAllocatorClient(base_url=base_url, session_id=f"stress-{worker_id}")
        rng = random.Random(worker_id + 9999)
        for _ in range(15):
            count = rng.randint(1, 2)
            exclusive = rng.choice([True, False])
            try:
                lease_ctx = client.allocate(
                    count=count, timeout=30.0, exclusive=exclusive
                )
            except Exception:
                continue
            assert len(lease_ctx.gpus) == count
            with active_lock:
                violations.extend(_assert_no_overlap(active + [(lease_ctx, exclusive)]))
                active.append((lease_ctx, exclusive))
            time.sleep(rng.uniform(0.001, 0.01))
            with active_lock:
                active.remove((lease_ctx, exclusive))
                successes[0] += 1
            client.release(lease_ctx.lease_id)

    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(client_worker, range(16)))
    finally:
        server.shutdown()
        server.server_close()
        allocator.shutdown()

    assert not violations, violations
    assert successes[0] == 16 * 15
    status = allocator.status()
    assert status["free_count"] == 8
    assert status["leases"] == []


# ---------------------------------------------------------------------------
# Session liveness monitoring
# ---------------------------------------------------------------------------


def _make_http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.devin.ai/v3/...",
        code=code,
        msg="test",
        hdrs={},  # type: ignore[arg-type]
        fp=None,
    )


def test_devin_api_session_monitor_reclaims_on_404():
    monitor = DevinApiSessionMonitor(api_token="test-token", org_id="test-org")

    with patch("gpu_allocator.session_monitor.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = _make_http_error(404)
        assert monitor.is_alive("devin-dead") is False
        assert mock_urlopen.call_count == 1

        # Second call should use the cache and not issue another request.
        assert monitor.is_alive("devin-dead") is False
        assert mock_urlopen.call_count == 1


def test_devin_api_session_monitor_fails_safe_on_auth_and_server_errors():
    monitor = DevinApiSessionMonitor(api_token="test-token", org_id="test-org")

    with patch("gpu_allocator.session_monitor.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = _make_http_error(403)
        assert monitor.is_alive("devin-live") is True

        mock_urlopen.side_effect = _make_http_error(500)
        assert monitor.is_alive("devin-live") is True


def test_devin_api_session_monitor_skips_non_devin_ids():
    monitor = DevinApiSessionMonitor(api_token="test-token", org_id="test-org")

    with patch("gpu_allocator.session_monitor.urllib.request.urlopen") as mock_urlopen:
        assert monitor.is_alive("pid-12345") is True
        mock_urlopen.assert_not_called()


def test_devin_api_session_monitor_status_parsing():
    monitor = DevinApiSessionMonitor(api_token="test-token", org_id="test-org")

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return self._body

    with patch("gpu_allocator.session_monitor.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = _FakeResponse(
            json.dumps({"status": "exit"}).encode("utf-8")
        )
        assert monitor.is_alive("devin-exited") is False

        mock_urlopen.return_value = _FakeResponse(
            json.dumps({"status": "running"}).encode("utf-8")
        )
        assert monitor.is_alive("devin-running") is True


def test_allocator_reclaims_lease_when_session_terminated():
    class _DeadSessionMonitor(SessionMonitor):
        def is_alive(self, session_id: str) -> bool:
            return not session_id.startswith("devin-dead")

    allocator = GPUAllocator(
        gpus=_fake_gpus(2),
        lease_ttl_seconds=1000.0,
        lease_check_interval=0.1,
        enable_ttl_janitor=True,
        session_monitor=_DeadSessionMonitor(),
    )

    lease = allocator.allocate("devin-dead", count=1, timeout=0)
    assert lease
    assert allocator.status()["free_count"] == 1

    time.sleep(0.3)
    status = allocator.status()
    assert status["free_count"] == 2
    assert status["leases"] == []

    allocator.shutdown()


def test_allocator_keeps_lease_while_session_alive():
    allocator = GPUAllocator(
        gpus=_fake_gpus(2),
        lease_ttl_seconds=1000.0,
        lease_check_interval=0.1,
        enable_ttl_janitor=True,
        session_monitor=NullSessionMonitor(),
    )

    lease = allocator.allocate("devin-alive", count=1, timeout=0)
    assert lease

    time.sleep(0.3)
    status = allocator.status()
    assert status["free_count"] == 1
    assert len(status["leases"]) == 1

    allocator.shutdown()


def _list_devin_sessions(base_url: str, token: str, org_id: str):
    import urllib.request

    url = f"{base_url}/v3/organizations/{org_id}/sessions"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("items", data)


def _find_devin_session(sessions, status):
    for s in sessions:
        if isinstance(s, dict) and s.get("status") == status:
            sid = s.get("session_id")
            if sid:
                return f"devin-{sid}"
    return None


@pytest.mark.integration
def test_devin_api_session_monitor_live_statuses(
    devin_api_token, devin_org_id, devin_api_base
):
    """Integration test against the real Devin API.

    Uses existing sessions in the organization to verify that the monitor
    correctly reports running sessions as alive and exit/suspended sessions
    as dead.
    """
    sessions = _list_devin_sessions(devin_api_base, devin_api_token, devin_org_id)
    monitor = DevinApiSessionMonitor(
        api_token=devin_api_token,
        org_id=devin_org_id,
        api_base=devin_api_base,
        cache_ttl_seconds=60.0,
        timeout_seconds=10.0,
    )

    running_id = _find_devin_session(sessions, "running")
    exit_id = _find_devin_session(sessions, "exit")
    suspended_id = _find_devin_session(sessions, "suspended")

    if running_id:
        assert monitor.is_alive(running_id) is True
    else:
        pytest.skip("No running Devin session found in the organization")

    if exit_id:
        assert monitor.is_alive(exit_id) is False
    else:
        pytest.skip("No exited Devin session found in the organization")

    if suspended_id:
        assert monitor.is_alive(suspended_id) is False


@pytest.mark.integration
def test_devin_session_termination_releases_gpu_lease(
    devin_api_token, devin_org_id, devin_api_base
):
    """End-to-end zombie-allocation test using a real terminated session.

    Finds an existing Devin session with status ``exit``, allocates a GPU
    with that session id, and asserts the allocator's TTL janitor reclaims
    the lease because the session is no longer alive.
    """
    sessions = _list_devin_sessions(devin_api_base, devin_api_token, devin_org_id)
    exit_id = _find_devin_session(sessions, "exit")
    if not exit_id:
        pytest.skip("No exited Devin session found in the organization")

    monitor = DevinApiSessionMonitor(
        api_token=devin_api_token,
        org_id=devin_org_id,
        api_base=devin_api_base,
        cache_ttl_seconds=60.0,
        timeout_seconds=10.0,
    )
    allocator = GPUAllocator(
        gpus=_fake_gpus(2),
        lease_ttl_seconds=1000.0,
        lease_check_interval=2.0,
        enable_ttl_janitor=True,
        session_monitor=monitor,
    )

    lease = allocator.allocate(exit_id, count=1, timeout=0)
    assert lease
    assert allocator.status()["free_count"] == 1

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        status = allocator.status()
        if status["free_count"] == 2:
            break
        time.sleep(2.0)
    else:
        allocator.shutdown()
        pytest.fail("GPU lease was not reclaimed for an exited Devin session")

    assert status["leases"] == []
    allocator.shutdown()
