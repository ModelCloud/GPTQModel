# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from logbar import LogBar

from .env import force_pci_bus_order
from .models import GPU


log = LogBar("gpu_allocator.client")
# Register the LogBar instance with the stdlib manager so setLevel()
# invalidates its level cache correctly.
logging.Logger.manager.loggerDict[log.name] = log

force_pci_bus_order()


def _default_session_id() -> str:
    return (
        os.environ.get("DEVIN_OUTPOST_SESSION_ID")
        or os.environ.get("DEVIN_SESSION_ID")
        or f"pid-{os.getpid()}"
    )


def _request_json(
    url: str,
    *,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Any:
    data = None
    headers: Dict[str, str] = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return None
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8", errors="replace"))
        except Exception:
            detail = {"error": exc.reason}
        raise GPUAllocatorError(exc.code, detail.get("error", exc.reason)) from exc


class GPUAllocatorError(Exception):
    """Raised when the allocator server returns an error response."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"GPU allocator error {status}: {message}")


@dataclass
class LeaseContext:
    """A held GPU lease with helper methods for CUDA environment setup."""

    lease_id: str
    gpus: List[GPU]

    def as_cuda_visible_devices(
        self,
        style: str = "uuid",
        set_env: bool = False,
    ) -> str:
        """Render a `CUDA_VISIBLE_DEVICES` value from the leased GPUs.

        `style` may be ``"uuid"`` (default, unambiguous), ``"pci_bus_id"``,
        or ``"pci_order_index"`` (requires `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
        """
        if style == "uuid":
            values = [gpu.uuid for gpu in self.gpus]
        elif style == "pci_bus_id":
            values = [gpu.pci_bus_id for gpu in self.gpus]
        elif style == "pci_order_index":
            values = [str(gpu.pci_order_index) for gpu in self.gpus]
        else:
            raise ValueError(f"Unknown style: {style!r}")

        rendered = ",".join(values)
        if set_env:
            if style == "pci_order_index":
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = rendered
        return rendered


class GPUAllocatorClient:
    """HTTP client for the GPU allocator daemon."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.base_url = (
            base_url or os.environ.get("GPU_ALLOCATOR_URL", "http://127.0.0.1:17351")
        ).rstrip("/")
        self.session_id = session_id or _default_session_id()
        log.info(
            "GPUAllocatorClient initialized: base_url=%s session_id=%s",
            self.base_url,
            self.session_id,
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> Dict[str, Any]:
        log.debug("GET /health")
        return _request_json(self._url("/health"), timeout=10.0) or {}

    def status(self) -> Dict[str, Any]:
        log.debug("GET /status")
        return _request_json(self._url("/status"), timeout=10.0) or {}

    def allocate(
        self,
        count: int,
        *,
        timeout: Optional[float] = None,
        reason: Optional[str] = None,
        exclusive: bool = True,
    ) -> LeaseContext:
        """Request a GPU lease.

        - ``timeout=0`` returns immediately if GPUs are not free.
        - ``timeout=None`` blocks until GPUs are available.
        - ``timeout>0`` blocks up to that many seconds.
        """
        body: Dict[str, Any] = {
            "session_id": self.session_id,
            "count": count,
            "timeout": timeout,
            "exclusive": exclusive,
        }
        if reason is not None:
            body["reason"] = reason

        # The urllib timeout should be at least the server-side timeout plus
        # a small margin so the server's response (success or timeout) is not
        # cut off by the client socket.
        client_timeout: Optional[float]
        if timeout is None:
            client_timeout = None
        else:
            client_timeout = timeout + 10.0 if timeout > 0 else 10.0

        log.info(
            "POST /allocate: count=%d timeout=%s exclusive=%s reason=%s",
            count,
            timeout,
            exclusive,
            reason,
        )
        response = _request_json(
            self._url("/allocate"),
            method="POST",
            body=body,
            timeout=client_timeout,
        )
        if not isinstance(response, dict):
            log.error("Invalid allocate response from allocator")
            raise GPUAllocatorError(500, "Invalid response from allocator")
        if not response.get("ok"):
            log.error(
                "Allocate failed: status=%s error=%s",
                response.get("status"),
                response.get("error"),
            )
            raise GPUAllocatorError(
                response.get("status", 500), response.get("error", "Allocation failed")
            )

        gpus = [GPU(**gpu_dict) for gpu_dict in response["gpus"]]
        log.info(
            "Allocated lease %s with %d GPU(s) for session %s",
            response["lease_id"],
            len(gpus),
            self.session_id,
        )
        return LeaseContext(lease_id=response["lease_id"], gpus=gpus)

    def release(self, lease_id: str) -> None:
        log.info("POST /release: lease_id=%s", lease_id)
        response = _request_json(
            self._url("/release"),
            method="POST",
            body={"lease_id": lease_id},
            timeout=10.0,
        )
        if isinstance(response, dict) and not response.get("ok"):
            log.error(
                "Release failed: status=%s error=%s",
                response.get("status"),
                response.get("error"),
            )
            raise GPUAllocatorError(
                response.get("status", 500), response.get("error", "Release failed")
            )

    @contextmanager
    def acquire(
        self,
        count: int,
        *,
        timeout: Optional[float] = None,
        reason: Optional[str] = None,
        exclusive: bool = True,
    ) -> Generator[LeaseContext, None, None]:
        """Context manager that acquires and automatically releases a GPU lease."""
        lease = self.allocate(
            count=count, timeout=timeout, reason=reason, exclusive=exclusive
        )
        try:
            yield lease
        finally:
            self.release(lease.lease_id)


def acquire(
    count: int,
    *,
    timeout: Optional[float] = None,
    reason: Optional[str] = None,
    exclusive: bool = True,
    base_url: Optional[str] = None,
    session_id: Optional[str] = None,
) -> LeaseContext:
    """Create a one-shot client and acquire a lease as a context manager.

    Example::

        from gpu_allocator import acquire

        with acquire(2, timeout=300, reason="benchmark") as lease:
            os.environ["CUDA_VISIBLE_DEVICES"] = lease.as_cuda_visible_devices("uuid")
            # run GPU work
    """
    return GPUAllocatorClient(base_url=base_url, session_id=session_id).acquire(
        count=count,
        timeout=timeout,
        reason=reason,
        exclusive=exclusive,
    )
