# SPDX-License-Identifier: Apache-2.0
"""Strict checkpoint device identity; no implicit topology remapping."""

import csv
import io
import subprocess

import torch

from .checkpoint_store import CheckpointError


def _gpu_serials_by_uuid() -> dict[str, str]:
    """Resolve physical serials by UUID, never by nvidia-smi's device order."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,serial", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    serials = {}
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) != 2:
            continue
        uuid, serial = (field.strip() for field in row)
        if serial.lower() in {
            "",
            "n/a",
            "[n/a]",
            "not supported",
            "[not supported]",
            "unknown",
        }:
            continue
        serials[uuid.removeprefix("GPU-").lower()] = serial
    return serials


def checkpoint_identity_without_physical_gpu_ids(identity: dict) -> dict:
    """Explicit override only; retain topology, GPU models and capabilities."""
    topology = identity.get("device_topology")
    if not isinstance(topology, dict) or "visible_cuda" not in topology:
        return identity
    return {
        **identity,
        "device_topology": {
            **topology,
            "visible_cuda": [
                {
                    key: value
                    for key, value in device.items()
                    if key not in {"uuid", "serial"}
                }
                for device in topology["visible_cuda"]
            ],
        },
    }


def checkpoint_device_topology(pools: dict[str, list[str]]) -> dict:
    normalized = {}
    used_cuda = set()
    for role, entries in pools.items():
        normalized[role] = []
        for entry in entries:
            device = torch.device(entry)
            if device.type not in {"cpu", "cuda"}:
                raise NotImplementedError(
                    "checkpoint device restoration currently supports CPU and CUDA"
                )
            if device.type == "cuda":
                device = torch.device(
                    "cuda",
                    torch.cuda.current_device()
                    if device.index is None
                    else device.index,
                )
                used_cuda.add(device.index)
            normalized[role].append(str(device))
    visible = []
    if used_cuda:
        serials = _gpu_serials_by_uuid()
        count = torch.cuda.device_count()
        if any(index >= count for index in used_cuda):
            raise CheckpointError(
                "checkpoint execution references an unavailable CUDA device"
            )
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            uuid = getattr(properties, "uuid", None)
            if not uuid:
                raise CheckpointError(
                    "GPU UUID is required for strict checkpoint device identity"
                )
            visible.append(
                {
                    "index": index,
                    "uuid": str(uuid),
                    "serial": serials.get(str(uuid).removeprefix("GPU-").lower()),
                    "name": properties.name,
                    "capability": [properties.major, properties.minor],
                }
            )
    return {
        "version": 2,
        "gpu_count": len(used_cuda),
        "pools": normalized,
        "visible_cuda": visible,
    }
