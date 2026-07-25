# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Deque, Dict, List, Optional, Set

from .inventory import discover_gpus
from .models import GPU, Lease
from .session_monitor import NullSessionMonitor, SessionMonitor


@dataclass
class WaitRequest:
    """A session waiting for a GPU allocation."""

    session_id: str
    count: int
    reason: Optional[str]
    exclusive: bool
    deadline: Optional[float]
    condition: threading.Condition
    assigned: Optional[Lease] = None
    cancelled: bool = False


class GPUAllocator:
    """Thread-safe, free-threading-compatible GPU allocator.

    Supports both exclusive and shared leases. Exclusive leases block all other
    access to the assigned GPUs. Shared leases may overlap with other shared
    leases on the same GPU, but may not overlap with an exclusive lease.
    """

    def __init__(
        self,
        gpus: Optional[List[GPU]] = None,
        lease_ttl_seconds: float = 600.0,
        lease_check_interval: float = 5.0,
        enable_ttl_janitor: bool = True,
        session_monitor: Optional[SessionMonitor] = None,
    ):
        if gpus is None:
            gpus = discover_gpus()
        if not gpus:
            raise RuntimeError("No GPUs discovered")

        sorted_gpus = sorted(gpus, key=lambda g: g.pci_order_index)
        self._gpus: List[GPU] = sorted_gpus
        self._gpu_by_bus_id: Dict[str, GPU] = {
            gpu.pci_bus_id: gpu for gpu in sorted_gpus
        }
        self._all_bus_ids: Set[str] = set(self._gpu_by_bus_id.keys())

        self._lock = threading.Lock()
        self._leases: Dict[str, Lease] = {}
        self._gpu_leases: Dict[str, Set[str]] = {}
        self._waiters: Deque[WaitRequest] = deque()

        self._lease_ttl_seconds = float(lease_ttl_seconds)
        self._lease_check_interval = float(lease_check_interval)
        self._stop_event = threading.Event()
        self._janitor: Optional[threading.Thread] = None

        self._session_monitor = (
            session_monitor if session_monitor is not None else NullSessionMonitor()
        )

        if enable_ttl_janitor:
            self._janitor = threading.Thread(
                target=self._ttl_janitor, name="GPUAllocator-TTL", daemon=True
            )
            self._janitor.start()

    # ------------------------------------------------------------------ #
    # Internal helpers (lock must be held by caller where noted)
    # ------------------------------------------------------------------ #

    def _free_gpu_bus_ids(self) -> List[str]:
        """Return bus ids with no active lease."""
        return sorted(
            (
                bus_id
                for bus_id in self._all_bus_ids
                if not self._gpu_leases.get(bus_id)
            ),
            key=lambda b: self._gpu_by_bus_id[b].pci_order_index,
        )

    def _is_exclusively_leased(self, bus_id: str) -> bool:
        lease_ids = self._gpu_leases.get(bus_id)
        if not lease_ids:
            return False
        if len(lease_ids) > 1:
            return False
        lease_id = next(iter(lease_ids))
        return self._leases[lease_id].exclusive

    def _shared_candidate_bus_ids(self) -> List[str]:
        """Return bus ids not exclusively leased, sorted to prefer already-shared GPUs."""
        return sorted(
            (
                bus_id
                for bus_id in self._all_bus_ids
                if not self._is_exclusively_leased(bus_id)
            ),
            key=lambda b: (
                -len(self._gpu_leases.get(b, set())),
                self._gpu_by_bus_id[b].pci_order_index,
            ),
        )

    def _can_satisfy(self, waiter: WaitRequest) -> bool:
        if waiter.exclusive:
            return len(self._free_gpu_bus_ids()) >= waiter.count
        return len(self._shared_candidate_bus_ids()) >= waiter.count

    def _allocate_locked(
        self,
        session_id: str,
        count: int,
        reason: Optional[str],
        exclusive: bool,
    ) -> Lease:
        """Create a lease for `count` GPUs. Caller must hold `_lock`."""
        if exclusive:
            candidates = self._free_gpu_bus_ids()
        else:
            candidates = self._shared_candidate_bus_ids()

        if len(candidates) < count:
            raise RuntimeError("Not enough available GPUs")

        chosen_bus_ids = candidates[:count]
        lease_id = uuid.uuid4().hex
        for bus_id in chosen_bus_ids:
            self._gpu_leases.setdefault(bus_id, set()).add(lease_id)

        chosen_gpus = [self._gpu_by_bus_id[bus_id] for bus_id in chosen_bus_ids]
        now = time.monotonic()
        lease = Lease(
            lease_id=lease_id,
            session_id=session_id,
            gpus=chosen_gpus,
            reason=reason,
            exclusive=exclusive,
            created_at=now,
            expires_at=now + self._lease_ttl_seconds,
        )
        self._leases[lease_id] = lease
        return lease

    def _remove_waiter_locked(self, waiter: WaitRequest) -> None:
        try:
            self._waiters.remove(waiter)
        except ValueError:
            pass
        waiter.cancelled = True

    def _assign_waiters_locked(self) -> None:
        """Walk the FIFO wait queue and satisfy any requests that now fit."""
        while self._waiters:
            waiter = self._waiters[0]
            if waiter.cancelled:
                self._waiters.popleft()
                continue
            if not self._can_satisfy(waiter):
                break
            lease = self._allocate_locked(
                waiter.session_id,
                waiter.count,
                waiter.reason,
                waiter.exclusive,
            )
            waiter.assigned = lease
            self._waiters.popleft()
            waiter.condition.notify()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def total_gpus(self) -> int:
        with self._lock:
            return len(self._gpus)

    def allocate(
        self,
        session_id: str,
        count: int,
        timeout: Optional[float] = None,
        reason: Optional[str] = None,
        exclusive: bool = True,
    ) -> Lease:
        """Acquire a lease for `count` GPUs.

        - ``timeout=None`` blocks until satisfied.
        - ``timeout=0`` returns immediately if GPUs are not available.
        - ``timeout>0`` blocks up to that many seconds.

        A shared lease may be placed on a GPU that already has shared leases.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        if count > len(self._gpus):
            raise ValueError(f"Requested {count} GPUs but only {len(self._gpus)} exist")

        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._lock:
            if not self._waiters and self._can_satisfy(
                WaitRequest(
                    session_id=session_id,
                    count=count,
                    reason=reason,
                    exclusive=exclusive,
                    deadline=None,
                    condition=threading.Condition(self._lock),
                )
            ):
                return self._allocate_locked(session_id, count, reason, exclusive)

            if timeout is not None and timeout <= 0:
                raise TimeoutError(f"No {count} GPU(s) available immediately")

            waiter = WaitRequest(
                session_id=session_id,
                count=count,
                reason=reason,
                exclusive=exclusive,
                deadline=deadline,
                condition=threading.Condition(self._lock),
            )
            self._waiters.append(waiter)

            while waiter.assigned is None and not waiter.cancelled:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._remove_waiter_locked(waiter)
                        raise TimeoutError(f"Timed out waiting for {count} GPU(s)")

                if not waiter.condition.wait(timeout=remaining):
                    self._remove_waiter_locked(waiter)
                    if waiter.assigned is not None:
                        return waiter.assigned
                    raise TimeoutError(f"Timed out waiting for {count} GPU(s)")

            if waiter.assigned is not None:
                return waiter.assigned
            raise RuntimeError("Waiter was cancelled")

    def release(self, lease_id: str) -> List[GPU]:
        """Release a lease and return the GPUs that were freed."""
        with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                raise KeyError(f"Unknown lease id: {lease_id}")
            for gpu in lease.gpus:
                lease_ids = self._gpu_leases.get(gpu.pci_bus_id)
                if lease_ids:
                    lease_ids.discard(lease_id)
                    if not lease_ids:
                        del self._gpu_leases[gpu.pci_bus_id]
            self._assign_waiters_locked()
            return lease.gpus

    def renew(self, lease_id: str, duration: Optional[float] = None) -> Lease:
        """Extend a lease's TTL and return an immutable snapshot."""
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise KeyError(f"Unknown lease id: {lease_id}")
            add = self._lease_ttl_seconds if duration is None else float(duration)
            new_lease = replace(lease, expires_at=time.monotonic() + add)
            self._leases[lease_id] = new_lease
            return new_lease

    def status(self) -> Dict[str, object]:
        """Return a snapshot of free GPUs and active leases."""
        with self._lock:
            free = self._free_gpu_bus_ids()
            return {
                "total": len(self._gpus),
                "free_count": len(free),
                "free_gpus": [self._gpu_by_bus_id[bus_id].to_dict() for bus_id in free],
                "leases": [lease.to_dict() for lease in self._leases.values()],
            }

    def gpu_status_rows(self) -> List[Dict[str, Any]]:
        """Return a list of per-GPU status rows for table logging."""
        with self._lock:
            now = time.monotonic()
            rows: List[Dict[str, Any]] = []
            for gpu in self._gpus:
                bus_id = gpu.pci_bus_id
                lease_ids = self._gpu_leases.get(bus_id, set())
                if not lease_ids:
                    rows.append(
                        {
                            "index": gpu.pci_order_index,
                            "pci_bus_id": gpu.pci_bus_id,
                            "uuid": gpu.uuid,
                            "status": "free",
                            "lease_id": "",
                            "session_id": "",
                            "expires_in": "",
                        }
                    )
                    continue
                for lease_id in lease_ids:
                    lease = self._leases[lease_id]
                    rows.append(
                        {
                            "index": gpu.pci_order_index,
                            "pci_bus_id": gpu.pci_bus_id,
                            "uuid": gpu.uuid,
                            "status": "exclusive" if lease.exclusive else "shared",
                            "lease_id": lease.lease_id,
                            "session_id": lease.session_id,
                            "expires_in": f"{max(0.0, lease.expires_at - now):.0f}s",
                        }
                    )
            return rows

    # ------------------------------------------------------------------ #
    # Background maintenance
    # ------------------------------------------------------------------ #

    def _ttl_janitor(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._lease_check_interval)

            with self._lock:
                now = time.monotonic()
                expired = [
                    lease_id
                    for lease_id, lease in self._leases.items()
                    if lease.expires_at < now
                ]
                leases_snapshot = list(self._leases.values())

            terminated: List[str] = []
            for lease in leases_snapshot:
                try:
                    if not self._session_monitor.is_alive(lease.session_id):
                        terminated.append(lease.lease_id)
                except Exception:
                    # Fail safe: never kill a lease because the monitor broke.
                    pass

            with self._lock:
                for lease_id in expired + terminated:
                    lease = self._leases.pop(lease_id, None)
                    if lease is None:
                        continue
                    for gpu in lease.gpus:
                        lease_ids = self._gpu_leases.get(gpu.pci_bus_id)
                        if lease_ids:
                            lease_ids.discard(lease_id)
                            if not lease_ids:
                                del self._gpu_leases[gpu.pci_bus_id]
                if expired or terminated:
                    self._assign_waiters_locked()

    def shutdown(self) -> None:
        """Stop the TTL janitor and free all leases."""
        self._stop_event.set()
        if self._janitor is not None:
            self._janitor.join(timeout=1.0)
        with self._lock:
            for lease in list(self._leases.values()):
                for gpu in lease.gpus:
                    lease_ids = self._gpu_leases.get(gpu.pci_bus_id)
                    if lease_ids:
                        lease_ids.discard(lease.lease_id)
                        if not lease_ids:
                            del self._gpu_leases[gpu.pci_bus_id]
            self._leases.clear()
            self._assign_waiters_locked()
