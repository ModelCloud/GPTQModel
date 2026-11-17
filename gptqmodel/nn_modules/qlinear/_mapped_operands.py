# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Detach CPU int4 operands from the checkpoint's file mapping.

``safetensors`` loads a checkpoint through an mmap and hands the model
zero-copy views of it. The CPU int4 transforms replace the packed weight with
an anonymous runtime tensor, but a few small operands (``g_idx``, the
checkpoint-dtype ``scales``, ``bias``) used to stay views of the file. Each
surviving view pins the *whole* shard mapping, so every page the loader
touched stays resident and the model's quantized weights are counted twice in
RSS (issue #4785).

The fix copies only those small operands out of the mapping. The packed weight
is never copied a second time: the transform rebuilds it in place.
"""

from __future__ import annotations

import torch
import torch.nn as nn


#: Checkpoint file suffixes. Restricting the sweep to these leaves unrelated
#: file mappings (shared memory, databases) untouched.
_CHECKPOINT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".dat",
)


def _checkpoint_mapping_ranges() -> tuple[tuple[int, int], ...]:
    """Virtual-address ranges backed by a checkpoint file in this process."""

    try:
        with open("/proc/self/maps", "r", encoding="ascii", errors="replace") as maps:
            lines = maps.readlines()
    except OSError:
        return ()

    ranges = []
    for line in lines:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        path = parts[5].strip().removesuffix(" (deleted)")
        if not path.endswith(_CHECKPOINT_SUFFIXES):
            continue
        try:
            start, end = (int(value, 16) for value in parts[0].split("-"))
        except ValueError:
            continue
        ranges.append((start, end))
    return tuple(ranges)


def _storage_is_mapped(tensor: torch.Tensor, ranges) -> bool:
    try:
        storage = tensor.untyped_storage()
        start = storage.data_ptr()
        end = start + storage.nbytes()
    except (NotImplementedError, RuntimeError):
        # Meta and other virtual storages have no address to match.
        return False
    return any(lo <= start < hi or lo < end <= hi for lo, hi in ranges)


def _replace(owner: nn.Module, name: str, tensor: torch.Tensor) -> None:
    current = owner._parameters.get(name)
    if current is not None:
        owner._parameters[name] = nn.Parameter(
            tensor, requires_grad=current.requires_grad
        )
        return
    if name in owner._buffers:
        owner._buffers[name] = tensor
        return
    setattr(owner, name, tensor)


def detach_checkpoint_mapping(module: nn.Module) -> list[str]:
    """Copy every module tensor still backed by a checkpoint file mapping.

    Called at the end of a CPU int4 repack. The returned names are the
    operands that were copied; an empty list means the module already owned
    anonymous storage. The copy runs outside ``inference_mode`` so the
    detached operand is a normal tensor that a later ``load_state_dict`` may
    write.
    """

    ranges = _checkpoint_mapping_ranges()
    if not ranges:
        return []

    detached = []
    with torch.inference_mode(False):
        for owner in module.modules():
            tensors = list(owner._parameters.items()) + list(owner._buffers.items())
            for name, tensor in tensors:
                if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                    continue
                if not _storage_is_mapped(tensor, ranges):
                    continue
                _replace(owner, name, tensor.detach().clone())
                detached.append(name)
    return detached


__all__ = ["detach_checkpoint_mapping"]
