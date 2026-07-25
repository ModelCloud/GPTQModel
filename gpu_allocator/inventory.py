# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import re
import shutil
import subprocess
from typing import Dict, List, Tuple

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
        "--query-gpu=index,pci.bus_id,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    rows: List[Tuple[Tuple[int, int, int, int], GPU]] = []
    reader = csv.reader(result.stdout.splitlines())
    for line in reader:
        if not line:
            continue
        fields = [f.strip() for f in line]
        if len(fields) < 7:
            continue
        _, bus_id, uuid, name, memory_total_str, memory_used_str, util_str = fields[:7]
        memory_total_mib = int(memory_total_str)
        memory_used_mib = int(memory_used_str)
        memory_free_mib = memory_total_mib - memory_used_mib
        # utilization.gpu is returned as "0 %"; strip the percent sign.
        utilization_gpu = int(util_str.replace("%", "").strip())
        sort_key = _parse_pci_bus_id(bus_id)
        gpu = GPU(
            pci_order_index=-1,
            pci_bus_id=bus_id.lower(),
            uuid=uuid.strip(),
            name=name,
            memory_total_mib=memory_total_mib,
            memory_used_mib=memory_used_mib,
            memory_free_mib=memory_free_mib,
            utilization_gpu=utilization_gpu,
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
        total_mib = int(props.total_memory / (1024 * 1024))
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            free_mib = int(free_bytes / (1024 * 1024))
            total_mib = int(total_bytes / (1024 * 1024))
            used_mib = total_mib - free_mib
        except Exception:
            used_mib = 0
            free_mib = total_mib
        gpu = GPU(
            pci_order_index=-1,
            pci_bus_id=bus_id,
            uuid=uuid,
            name=props.name,
            memory_total_mib=total_mib,
            memory_used_mib=used_mib,
            memory_free_mib=free_mib,
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
                memory_used_mib=gpu.memory_used_mib,
                memory_free_mib=gpu.memory_free_mib,
            )
        )
    return ordered


def _query_nvidia_smi_status() -> Dict[str, Tuple[int, int, int]]:
    """Return a mapping from lower-case PCI bus id to (used_mib, total_mib, util_percent)."""
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found")

    command = [
        "nvidia-smi",
        "--query-gpu=pci.bus_id,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    status: Dict[str, Tuple[int, int, int]] = {}
    reader = csv.reader(result.stdout.splitlines())
    for line in reader:
        if not line:
            continue
        fields = [f.strip() for f in line]
        if len(fields) < 4:
            continue
        bus_id, used_str, total_str, util_str = fields[:4]
        used_mib = int(used_str)
        total_mib = int(total_str)
        util_percent = int(util_str.replace("%", "").strip())
        status[bus_id.lower()] = (used_mib, total_mib, util_percent)
    return status


def _query_torch_status() -> Dict[str, Tuple[int, int, int]]:
    """Return a mapping from lower-case PCI bus id to (used_mib, total_mib, util_percent) via torch."""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is not available for GPU status query") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("torch reports no CUDA devices")

    status: Dict[str, Tuple[int, int, int]] = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        domain = getattr(props, "pci_domain_id", 0)
        bus = getattr(props, "pci_bus_id", 0)
        device = getattr(props, "pci_device_id", 0)
        bus_id = _pci_bus_id_from_properties(domain, bus, device, 0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        total_mib = int(total_bytes / (1024 * 1024))
        free_mib = int(free_bytes / (1024 * 1024))
        used_mib = total_mib - free_mib
        # torch does not expose per-device utilization easily.
        status[bus_id.lower()] = (used_mib, total_mib, 0)
    return status


def get_all_gpu_status(
    prefer_nvidia_smi: bool = True,
) -> Dict[str, Tuple[int, int, int]]:
    """Return a mapping from lower-case PCI bus id to (used_mib, total_mib, util_percent)."""
    errors: List[Exception] = []
    funcs = [_query_nvidia_smi_status, _query_torch_status]
    if not prefer_nvidia_smi:
        funcs = [_query_torch_status, _query_nvidia_smi_status]
    for func in funcs:
        try:
            return func()
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise errors[0]
    raise RuntimeError("No GPU status source available")


def get_gpu_status(
    pci_bus_id: str, prefer_nvidia_smi: bool = True
) -> Tuple[int, int, int]:
    """Return (used_mib, total_mib, util_percent) for a single GPU identified by PCI bus id."""
    status = get_all_gpu_status(prefer_nvidia_smi=prefer_nvidia_smi)
    normalized_bus_id = pci_bus_id.lower()
    if normalized_bus_id in status:
        return status[normalized_bus_id]
    for key, value in status.items():
        if key.endswith(normalized_bus_id) or normalized_bus_id.endswith(key):
            return value
    raise RuntimeError(f"GPU {pci_bus_id} not found for status query")
