# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Structured device telemetry scoped to one quantization invocation."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch

from .logger import setup_logger

log = setup_logger()

_enabled = ContextVar("gptqmodel_device_telemetry", default=False)
_records_lock = threading.Lock()
_records: list[dict[str, Any]] = []


def device_telemetry_enabled() -> bool:
    """Return ``True`` when device telemetry should be emitted."""

    return _enabled.get()


@contextmanager
def device_telemetry_scope(enabled: bool):
    if type(enabled) is not bool:
        raise ValueError("device_telemetry must be a boolean")
    token = _enabled.set(enabled)
    try:
        yield
    finally:
        _enabled.reset(token)


def with_quantization_device_telemetry(fn):
    @wraps(fn)
    def run(model, *args, **kwargs):
        telemetry = getattr(getattr(model, "quantize_config", None), "telemetry", None)
        enabled = getattr(telemetry, "device", False)
        with device_telemetry_scope(enabled):
            return fn(model, *args, **kwargs)

    return run


def capture_device_telemetry(fn):
    """Propagate only this setting to queued work; reset reused workers afterward."""
    enabled = device_telemetry_enabled()

    @wraps(fn)
    def run(*args, **kwargs):
        with device_telemetry_scope(enabled):
            return fn(*args, **kwargs)

    return run


def _normalize_field(value: Any) -> Any:
    """Convert telemetry values into log-friendly primitives."""

    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return str(value.device)
    if isinstance(value, (list, tuple)):
        return [_normalize_field(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_field(v) for k, v in value.items()}
    return value


def emit_device_telemetry(event: str, **fields: Any) -> None:
    """Record and log one structured telemetry event when enabled."""

    if not device_telemetry_enabled():
        return

    record = {
        "event": event,
        "ts": round(time.time(), 6),
    }
    for key, value in fields.items():
        record[key] = _normalize_field(value)

    with _records_lock:
        _records.append(record)

    log.info(f"DeviceTelemetry: {record}")


def clear_device_telemetry_records() -> None:
    """Discard previously captured telemetry records."""

    with _records_lock:
        _records.clear()


def get_device_telemetry_records() -> list[dict[str, Any]]:
    """Return a copy of the captured telemetry records."""

    with _records_lock:
        return [dict(record) for record in _records]


__all__ = [
    "capture_device_telemetry",
    "clear_device_telemetry_records",
    "device_telemetry_enabled",
    "device_telemetry_scope",
    "emit_device_telemetry",
    "get_device_telemetry_records",
    "with_quantization_device_telemetry",
]
