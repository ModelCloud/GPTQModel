# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic weights for the MiniCPM-o clean-room fixtures.

Some fixtures (the full DVAE and the ConditionalChatTTS module that owns it)
are too large to ship as tensors, because the DVAE geometry is fixed by the
checkpoint format (~13M parameters). Their weights are instead regenerated
from a shape manifest with this recipe. Only torch is required.

Recipe: walk the manifest keys in lexicographic order. For the key at
position ``i`` draw ``z = torch.randn(shape, generator=torch.Generator().manual_seed(seed + i))``
in float32 and map it with :func:`_map`.
"""

import hashlib
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def _map(key: str, z: torch.Tensor) -> torch.Tensor:
    leaf = key.rsplit(".", 1)[-1]
    if key in ("coef", "dvae.coef"):
        return 1.0 + 0.1 * z  # mel channel scale (used as divisor and multiplier)
    if leaf == "coef":
        return 0.5 + 0.1 * z  # ConvNeXt layer scale
    if "norm" in key and leaf == "weight":
        return 1.0 + 0.1 * z
    if leaf == "bias" or z.dim() == 1:
        return 0.02 * z
    fan_in = 1
    for size in z.shape[1:]:
        fan_in *= size
    return z / fan_in**0.5


def build_state_dict(manifest: dict, seed: int) -> dict:
    out = {}
    for index, key in enumerate(sorted(manifest)):
        generator = torch.Generator().manual_seed(seed + index)
        z = torch.randn(manifest[key], generator=generator, dtype=torch.float32)
        out[key] = _map(key, z).contiguous()
    return out


def digest(state_dict: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode())
        h.update(state_dict[key].detach().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def load(name: str) -> dict:
    """Return the regenerated state_dict for ``<name>.manifest.json`` and verify its digest."""
    spec = json.loads((HERE / f"{name}.manifest.json").read_text())
    state_dict = build_state_dict({k: tuple(v) for k, v in spec["shapes"].items()}, spec["seed"])
    actual = digest(state_dict)
    if actual != spec["sha256"]:
        raise AssertionError(f"{name}: weight recipe digest mismatch {actual} != {spec['sha256']}")
    return state_dict
