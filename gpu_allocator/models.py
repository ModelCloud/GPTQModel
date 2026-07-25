# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GPU:
    """A discovered accelerator device identified by stable PCI and UUID data."""

    pci_order_index: int
    pci_bus_id: str
    uuid: str
    name: str
    memory_total_mib: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pci_order_index": self.pci_order_index,
            "pci_bus_id": self.pci_bus_id,
            "uuid": self.uuid,
            "name": self.name,
            "memory_total_mib": self.memory_total_mib,
        }


@dataclass(frozen=True)
class Lease:
    """An active allocation of one or more GPUs to a session."""

    lease_id: str
    session_id: str
    gpus: List[GPU]
    reason: Optional[str]
    exclusive: bool
    created_at: float
    expires_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "session_id": self.session_id,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "reason": self.reason,
            "exclusive": self.exclusive,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }
