# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from logbar import LogBar

from .allocator import GPUAllocator
from .env import force_pci_bus_order
from .inventory import discover_gpus, get_all_gpu_status
from .models import GPU
from .session_monitor import DevinApiSessionMonitor, NullSessionMonitor, SessionMonitor


log = LogBar("gpu_allocator")
# Register the LogBar instance with the stdlib manager so setLevel()
# invalidates its level cache correctly.
logging.Logger.manager.loggerDict[log.name] = log


class _GPUAllocatorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the GPU allocator."""

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info(fmt % args)

    def _send_json(self, status: int, body: Dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        length_str = self.headers.get("Content-Length")
        if not length_str:
            return None
        try:
            length = int(length_str)
            body = self.rfile.read(length).decode("utf-8")
            return json.loads(body)
        except Exception:
            return None

    def _get_allocator(self) -> GPUAllocator:
        return self.server.allocator  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/status":
            allocator = self._get_allocator()
            self._send_json(200, allocator.status())
            return
        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        allocator = self._get_allocator()
        payload = self._read_json()
        if payload is None:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body"})
            return

        if self.path == "/allocate":
            session_id = payload.get("session_id") or payload.get("session")
            count = payload.get("count")
            reason = payload.get("reason")
            exclusive = payload.get("exclusive", True)

            if not session_id or not isinstance(session_id, str):
                self._send_json(400, {"ok": False, "error": "session_id is required"})
                return
            if not isinstance(count, int) or count <= 0:
                self._send_json(
                    400, {"ok": False, "error": "count must be a positive integer"}
                )
                return

            _SENTINEL = object()
            raw_timeout = payload.get("timeout", _SENTINEL)
            timeout_val: Optional[float]
            if raw_timeout is _SENTINEL:
                timeout_val = 0.0  # default: immediate return
            elif raw_timeout is None:
                timeout_val = None  # block indefinitely
            else:
                try:
                    timeout_val = float(raw_timeout)
                    if timeout_val < 0:
                        timeout_val = None
                except (TypeError, ValueError):
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": "timeout must be a number, null, or omitted",
                        },
                    )
                    return

            try:
                lease = allocator.allocate(
                    session_id=str(session_id),
                    count=int(count),
                    timeout=timeout_val,
                    reason=str(reason) if reason is not None else None,
                    exclusive=bool(exclusive),
                )
            except TimeoutError as exc:
                self._send_json(408, {"ok": False, "error": str(exc)})
                return
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return

            self._send_json(
                200,
                {
                    "ok": True,
                    "lease_id": lease.lease_id,
                    "gpus": [gpu.to_dict() for gpu in lease.gpus],
                    "expires_at": lease.expires_at,
                },
            )
            return

        if self.path == "/release":
            lease_id = payload.get("lease_id")
            if not lease_id or not isinstance(lease_id, str):
                self._send_json(400, {"ok": False, "error": "lease_id is required"})
                return
            try:
                allocator.release(str(lease_id))
            except KeyError as exc:
                self._send_json(404, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return
            self._send_json(200, {"ok": True})
            return

        if self.path == "/renew":
            lease_id = payload.get("lease_id")
            duration = payload.get("duration")
            if not lease_id or not isinstance(lease_id, str):
                self._send_json(400, {"ok": False, "error": "lease_id is required"})
                return
            duration_val = None
            if duration is not None:
                try:
                    duration_val = float(duration)
                except (TypeError, ValueError):
                    self._send_json(
                        400, {"ok": False, "error": "duration must be a number or null"}
                    )
                    return
            try:
                lease = allocator.renew(str(lease_id), duration=duration_val)
            except KeyError as exc:
                self._send_json(404, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return
            self._send_json(200, {"ok": True, "lease": lease.to_dict()})
            return

        self._send_json(404, {"ok": False, "error": "Not found"})


class GPUAllocatorServer(ThreadingHTTPServer):
    """Multi-threaded HTTP server wrapping a GPUAllocator."""

    request_queue_size = 128

    def __init__(
        self,
        server_address: tuple[str, int],
        allocator: GPUAllocator,
    ) -> None:
        super().__init__(server_address, _GPUAllocatorHandler)
        self.allocator = allocator


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU allocator daemon for shared multi-session GPU access"
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=17351, help="bind port")
    parser.add_argument(
        "--lease-ttl",
        type=float,
        default=600.0,
        help="Default lease TTL in seconds before automatic reclamation",
    )
    parser.add_argument(
        "--disable-ttl-janitor",
        action="store_true",
        help="Disable background TTL reclamation (not recommended)",
    )
    parser.add_argument(
        "--disable-session-monitor",
        action="store_true",
        help="Disable periodic Devin session liveness checks",
    )
    parser.add_argument(
        "--devin-api-token",
        default=os.environ.get("DEVIN_API_TOKEN") or os.environ.get("DEVIN_API_KEY"),
        help="Devin API token (default: $DEVIN_API_TOKEN or $DEVIN_API_KEY)",
    )
    parser.add_argument(
        "--devin-org-id",
        default=os.environ.get("DEVIN_ORG_ID"),
        help="Devin organization id (default: $DEVIN_ORG_ID)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging level",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=30.0,
        help="Seconds between periodic GPU status table logs (0 disables)",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Subset of GPUs to manage by pci_order_index, e.g. '0-3' or '0,2,4' (default: all)",
    )
    parser.add_argument(
        "--idle-memory-mib",
        type=int,
        default=100,
        help="Maximum used memory (MiB) for a GPU to be considered idle (default: 100, -1 disables)",
    )
    parser.add_argument(
        "--idle-gpu-percent",
        type=int,
        default=0,
        help="Maximum GPU utilization percent for an exclusive lease (default: 0, -1 disables)",
    )
    parser.add_argument(
        "--gpu-status-interval",
        type=float,
        default=5.0,
        help="Seconds between GPU status (memory/utilization) refreshes (default: 5.0)",
    )
    parser.add_argument(
        "--max-shared",
        type=int,
        default=4,
        help="Maximum number of shared leases per GPU (default: 4)",
    )
    return parser.parse_args(argv)


def _log_status_table(allocator: GPUAllocator) -> None:
    rows = allocator.gpu_status_rows()
    if not rows:
        log.info("GPU status snapshot: no GPUs discovered")
        return

    cols = log.columns(
        cols=[
            {"label": "index", "width": "fit"},
            {"label": "pci_bus_id", "width": "fit"},
            {"label": "uuid", "width": 36},
            {"label": "status", "width": "fit"},
            {"label": "lease_id", "width": 34},
            {"label": "session_id", "width": "fit"},
            {"label": "expires_in", "width": "fit"},
            {"label": "mem_total", "width": "fit"},
            {"label": "mem_used", "width": "fit"},
            {"label": "mem_free", "width": "fit"},
            {"label": "util%", "width": "fit"},
        ],
        padding=1,
    )
    log.info("GPU status snapshot")
    cols.info.header()
    for row in rows:
        cols.info(
            row["index"],
            row["pci_bus_id"],
            row["uuid"],
            row["status"],
            row["lease_id"],
            row["session_id"],
            row["expires_in"],
            row["memory_total_mib"],
            row["memory_used_mib"],
            row["memory_free_mib"],
            row["utilization_gpu"],
        )


def _status_reporter(
    allocator: GPUAllocator,
    stop_event: threading.Event,
    interval_seconds: float,
) -> None:
    while not stop_event.wait(timeout=interval_seconds):
        if stop_event.is_set():
            break
        try:
            _log_status_table(allocator)
        except Exception as exc:
            log.error("Status reporter failed: %s", exc)


def _parse_gpu_subset(spec: Optional[str]) -> Optional[set[int]]:
    """Parse a GPU subset spec like '0-3' or '0,2,4' into a set of indices."""
    if not spec:
        return None
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if start > end:
                start, end = end, start
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return indices


def _filter_gpus(
    gpus: List[GPU],
    allowed: Optional[set[int]],
) -> List[GPU]:
    """Return GPUs whose pci_order_index is in the allowed set."""
    if allowed is None:
        return gpus
    available = {gpu.pci_order_index for gpu in gpus}
    missing = sorted(allowed - available)
    if missing:
        raise ValueError(
            f"Requested GPU indices not discovered: {missing}. "
            f"Available pci_order_index values: {sorted(available)}"
        )
    return [gpu for gpu in gpus if gpu.pci_order_index in allowed]


def _session_monitor_from_args(args: argparse.Namespace) -> SessionMonitor:
    if args.disable_session_monitor:
        return NullSessionMonitor()
    if args.devin_api_token and args.devin_org_id:
        return DevinApiSessionMonitor(
            api_token=args.devin_api_token,
            org_id=args.devin_org_id,
        )
    monitor = DevinApiSessionMonitor.from_env()
    if monitor is not None:
        return monitor
    return NullSessionMonitor()


def main(argv: Optional[list[str]] = None) -> int:
    force_pci_bus_order()
    args = _parse_args(argv)
    log.setLevel(args.log_level)

    all_gpus = discover_gpus()
    allowed = _parse_gpu_subset(args.gpus)
    gpus = _filter_gpus(all_gpus, allowed)
    log.info("Discovered %d GPUs; managing %d", len(all_gpus), len(gpus))
    for gpu in gpus:
        log.info(
            "  pci_order_index=%d pci_bus_id=%s uuid=%s name=%s",
            gpu.pci_order_index,
            gpu.pci_bus_id,
            gpu.uuid,
            gpu.name,
        )

    session_monitor = _session_monitor_from_args(args)
    allocator = GPUAllocator(
        gpus=gpus,
        lease_ttl_seconds=args.lease_ttl,
        enable_ttl_janitor=not args.disable_ttl_janitor,
        session_monitor=session_monitor,
        idle_memory_mib=args.idle_memory_mib,
        idle_gpu_percent=args.idle_gpu_percent,
        gpu_status_interval=args.gpu_status_interval,
        gpu_status_checker=get_all_gpu_status,
        max_shared_per_gpu=args.max_shared,
    )
    server = GPUAllocatorServer((args.host, args.port), allocator)
    log.info("GPU allocator listening on http://%s:%d", args.host, args.port)

    stop_event = threading.Event()
    reporter: Optional[threading.Thread] = None
    if args.status_interval > 0:
        reporter = threading.Thread(
            target=_status_reporter,
            args=(allocator, stop_event, args.status_interval),
            name="GPUAllocator-StatusReporter",
            daemon=True,
        )
        reporter.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        stop_event.set()
        server.shutdown()
        allocator.shutdown()
        server.server_close()
        if reporter is not None:
            reporter.join(timeout=1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
