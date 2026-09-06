# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Declare optional rank8 buffer shapes before Accelerate places a P32 checkpoint."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .qvq_rank8 import RANK8_BUFFERS


def prepare_rank8_checkpoint_buffers(model, checkpoint, keys):
    from ..nn_modules.qlinear.qvq import QVQLinear

    names = {key for key in (keys or ()) if key.rsplit(".", 1)[-1] in RANK8_BUFFERS}
    if not names:
        return
    path = Path(checkpoint)
    if path.is_dir():
        single = path / "model.safetensors"
        indexes = list(path.glob("*.index.json"))
        if single.is_file():
            path = single
        elif len(indexes) == 1:
            path = indexes[0]
        else:
            raise ValueError("rank8 checkpoint requires one safetensors file or shard index")
    if path.suffix == ".json":
        index = json.loads(path.read_text())
        mapping = index.get("weight_map", index)
        files = {key: path.parent / mapping[key] for key in names}
    elif path.suffix == ".safetensors":
        files = dict.fromkeys(names, path)
    else:
        raise ValueError("rank8 checkpoint requires safetensors")
    modules = dict(model.named_modules())
    pending = []
    for prefix in sorted({name.rsplit(".", 1)[0] for name in names}):
        child = modules.get(prefix)
        if not isinstance(child, QVQLinear) or not child.v2b2_p32:
            raise ValueError(f"rank8 payload has no P32 module: {prefix}")
        expected = {f"{prefix}.{name}" for name in RANK8_BUFFERS}
        if not expected <= names:
            raise ValueError(f"incomplete rank8 payload: {prefix}")
        for name in RANK8_BUFFERS:
            key = f"{prefix}.{name}"
            with safe_open(str(files[key]), framework="pt", device="cpu") as handle:
                tensor = handle.get_slice(key)
                shape, dtype = tuple(tensor.get_shape()), tensor.get_dtype()
            if name == "rank8_metadata":
                valid = dtype == "U8" and len(shape) == 1 and shape[0] > 0
                torch_dtype = torch.uint8
            else:
                expected_shape = (child.in_features, 8) if name == "rank8_A" else (8, child.out_features)
                valid = dtype == "F16" and shape == expected_shape
                torch_dtype = torch.float16
            if not valid:
                raise ValueError(f"invalid rank8 checkpoint shape/dtype: {key}")
            pending.append((child, name, shape, torch_dtype))
    # Validate every header before mutating the model shell. No factor bytes
    # are read here; normal sharded checkpoint loading owns their placement.
    for child, name, shape, dtype in pending:
        setattr(child, name, torch.empty(shape, dtype=dtype, device=child.trellis.device))
