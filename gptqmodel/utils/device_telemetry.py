# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Structured, env-gated device telemetry for placement debugging."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

import torch

from .env import env_flag
from .logger import setup_logger


log = setup_logger()

_DEVICE_TELEMETRY_ENV = "GPTQMODEL_DEVICE_TELEMETRY"
_records_lock = threading.Lock()
_records: List[Dict[str, Any]] = []


def device_telemetry_enabled() -> bool:
    """Return ``True`` when device telemetry should be emitted."""

    return env_flag(_DEVICE_TELEMETRY_ENV, default="0")


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


def get_device_telemetry_records() -> List[Dict[str, Any]]:
    """Return a copy of the captured telemetry records."""

    with _records_lock:
        return [dict(record) for record in _records]


__all__ = [
    "clear_device_telemetry_records",
    "device_telemetry_enabled",
    "emit_device_telemetry",
    "get_device_telemetry_records",
]
