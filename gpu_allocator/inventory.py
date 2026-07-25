# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import re
import shutil
import subprocess
from typing import List, Tuple

from .models import GPU


def _parse_pci_bus_id(bus_id: str) -> Tuple[int, int, int, int]:
    """Parse '00000000:25:00.0' into (domain, bus, device, function)."""
    bus_id = bus_id.strip()
    match = re.fullmatch(
        r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-9a-fA-F])", bus_id
    )
    if match is None:
        raise ValueError(f"Unexpected PCI bus id format: {bus_id!r}")
    domain = int(match.group(1), 16)
    bus = int(match.group(2), 16)
    device = int(match.group(3), 16)
    function = int(match.group(4), 16)
    return domain, bus, device, function


def _pci_bus_id_from_properties(
    pci_domain_id: int,
    pci_bus_id: int,
    pci_device_id: int,
    function: int = 0,
) -> str:
    """Reconstruct a full PCI bus id from torch cuda property integers."""
    return f"{pci_domain_id:08x}:{pci_bus_id:02x}:{pci_device_id:02x}.{function:x}"


def _from_nvidia_smi() -> List[GPU]:
    """Enumerate GPUs using nvidia-smi and return them sorted by PCI bus id."""
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found")

    command = [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    rows: List[Tuple[Tuple[int, int, int, int], GPU]] = []
    reader = csv.reader(result.stdout.splitlines())
    for line in reader:
        if not line:
            continue
        fields = [f.strip() for f in line]
        if len(fields) < 5:
            continue
        _, bus_id, uuid, name, memory_total_str = fields[:5]
        memory_total_mib = int(memory_total_str)
        sort_key = _parse_pci_bus_id(bus_id)
        gpu = GPU(
            pci_order_index=-1,
            pci_bus_id=bus_id.lower(),
            uuid=uuid.strip(),
            name=name,
            memory_total_mib=memory_total_mib,
        )
        rows.append((sort_key, gpu))

    rows.sort(key=lambda item: item[0])
    return [gpu for _, gpu in rows]


def _from_torch() -> List[GPU]:
    """Enumerate GPUs using torch.cuda properties as a fallback."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - fallback only
        raise RuntimeError("torch is not available for GPU discovery") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("torch reports no CUDA devices")

    rows: List[Tuple[Tuple[int, int, int, int], GPU]] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        domain = getattr(props, "pci_domain_id", 0)
        bus = getattr(props, "pci_bus_id", 0)
        device = getattr(props, "pci_device_id", 0)
        function = 0
        bus_id = _pci_bus_id_from_properties(domain, bus, device, function)
        uuid = str(props.uuid)
        # torch returns a bare UUID; CUDA_VISIBLE_DEVICES expects the GPU- prefix.
        if not uuid.startswith("GPU-"):
            uuid = f"GPU-{uuid}"
        sort_key = (domain, bus, device, function)
        gpu = GPU(
            pci_order_index=-1,
            pci_bus_id=bus_id,
            uuid=uuid,
            name=props.name,
            memory_total_mib=int(props.total_memory / (1024 * 1024)),
        )
        rows.append((sort_key, gpu))

    rows.sort(key=lambda item: item[0])
    return [gpu for _, gpu in rows]


def discover_gpus(prefer_nvidia_smi: bool = True) -> List[GPU]:
    """Discover all GPUs and assign stable PCI-bus-order indices."""
    errors: List[Exception] = []
    if prefer_nvidia_smi:
        try:
            gpus = _from_nvidia_smi()
        except Exception as exc:
            errors.append(exc)
            gpus = _from_torch()
    else:
        try:
            gpus = _from_torch()
        except Exception as exc:
            errors.append(exc)
            gpus = _from_nvidia_smi()

    ordered: List[GPU] = []
    for idx, gpu in enumerate(gpus):
        ordered.append(
            GPU(
                pci_order_index=idx,
                pci_bus_id=gpu.pci_bus_id,
                uuid=gpu.uuid,
                name=gpu.name,
                memory_total_mib=gpu.memory_total_mib,
            )
        )
    return ordered
