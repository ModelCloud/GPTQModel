# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Per-layer and per-quantization-run disk I/O telemetry for large models with disk offload."""

import threading
from typing import Any, Dict, Optional


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def _fmt_rate(bytes_: int, seconds: float) -> str:
    if seconds <= 0.0:
        return "N/A"
    return f"{_fmt_bytes(bytes_ / seconds)}/s"


def _empty_bucket() -> Dict[str, Any]:
    return {"bytes": 0, "seconds": 0.0, "calls": 0}


def _add_to_bucket(bucket: Dict[str, Any], bytes_: int, seconds: float, calls: int = 1) -> None:
    bucket["bytes"] += bytes_
    bucket["seconds"] += seconds
    bucket["calls"] += calls


class _DiskTelemetry:
    """Thread-safe accumulator for disk read/write bytes and wall-clock seconds.

    Maintains both a per-layer window (reset after each summary) and running
    all-time totals so callers can emit per-layer handoff reports and a final
    quantization-wide summary without double-counting.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._read = _empty_bucket()
        self._write = _empty_bucket()
        self._lazy_read = _empty_bucket()
        self._accel_read = _empty_bucket()

        self._all_read = _empty_bucket()
        self._all_write = _empty_bucket()
        self._all_lazy_read = _empty_bucket()
        self._all_accel_read = _empty_bucket()

    def record_read(self, bytes_: int, seconds: float, source: str = "lazy_turtle") -> None:
        with self._lock:
            _add_to_bucket(self._read, bytes_, seconds)
            _add_to_bucket(self._all_read, bytes_, seconds)
            if source == "lazy_turtle":
                _add_to_bucket(self._lazy_read, bytes_, seconds)
                _add_to_bucket(self._all_lazy_read, bytes_, seconds)
            elif source == "accelerate":
                _add_to_bucket(self._accel_read, bytes_, seconds)
                _add_to_bucket(self._all_accel_read, bytes_, seconds)

    def record_write(self, bytes_: int, seconds: float) -> None:
        with self._lock:
            _add_to_bucket(self._write, bytes_, seconds)
            _add_to_bucket(self._all_write, bytes_, seconds)

    def summary(self, reset: bool = True, all_time: bool = False) -> Dict[str, Any]:
        with self._lock:
            if all_time:
                return {
                    "read": dict(self._all_read),
                    "write": dict(self._all_write),
                    "lazy_read": dict(self._all_lazy_read),
                    "accel_read": dict(self._all_accel_read),
                }

            summary = {
                "read": dict(self._read),
                "write": dict(self._write),
                "lazy_read": dict(self._lazy_read),
                "accel_read": dict(self._accel_read),
            }
            if reset:
                for d in (self._read, self._write, self._lazy_read, self._accel_read):
                    d["bytes"] = 0
                    d["seconds"] = 0.0
                    d["calls"] = 0
            return summary

    def log_summary(
        self,
        log,
        label: str = "layer",
        total_model_bytes: Optional[int] = None,
        all_time: bool = False,
    ) -> None:
        summary = self.summary(reset=not all_time, all_time=all_time)
        lines = []
        for bucket_name, bucket in (
            ("lazy_read", summary["lazy_read"]),
            ("accel_read", summary["accel_read"]),
            ("write", summary["write"]),
        ):
            bytes_ = bucket["bytes"]
            seconds = bucket["seconds"]
            calls = bucket["calls"]
            if calls == 0 and bytes_ == 0:
                continue
            direction = "read" if bucket_name.endswith("_read") else "write"
            source = bucket_name.replace("_read", "").replace("_write", "")
            pct = ""
            if total_model_bytes and total_model_bytes > 0:
                pct = f" ({100.0 * bytes_ / total_model_bytes:.2f}% of total)"
            lines.append(
                f"  {direction:5} {source:12} | calls={calls:4} | "
                f"bytes={_fmt_bytes(bytes_):>10} | time={seconds:7.3f}s | "
                f"throughput={_fmt_rate(bytes_, seconds)}{pct}"
            )

        if lines:
            log.info("Disk telemetry for %s:\n%s", label, "\n".join(lines))
        else:
            log.info("Disk telemetry for %s: no disk I/O recorded", label)


disk_telemetry = _DiskTelemetry()
