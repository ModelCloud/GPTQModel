# SPDX-License-Identifier: Apache-2.0
"""Strict checkpoint device identity; no implicit topology remapping."""

import torch

from .checkpoint_store import CheckpointError


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
                    "name": properties.name,
                    "capability": [properties.major, properties.minor],
                }
            )
    return {
        "version": 1,
        "gpu_count": len(used_cuda),
        "pools": normalized,
        "visible_cuda": visible,
    }
