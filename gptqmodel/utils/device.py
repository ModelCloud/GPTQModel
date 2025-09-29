# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from __future__ import annotations

from typing import Optional, Union

import torch
from device_smi import Device
from torch import nn as nn

from ..models._const import CPU, CUDA_0


# unit: GiB
def get_gpu_usage_memory():
    smi = Device(CUDA_0)
    return smi.memory_used() / 1024 / 1024 / 1024 #GB

# unit: GiB
def get_cpu_usage_memory():
    smi = Device(CPU)
    return smi.memory_used() / 1024 / 1024 / 1024 #GB

def get_device(obj: torch.Tensor | nn.Module) -> torch.device:
    if isinstance(obj, torch.Tensor):
        return obj.device

    params = list(obj.parameters())
    buffers = list(obj.buffers())
    if len(params) > 0:
        return params[0].device
    elif len(buffers) > 0:
        return buffers[0].device
    else:
        return CPU

def get_device_new(
    obj: torch.Tensor | nn.Module,
    recursive: bool = False,
    assert_mode: bool = False,
    expected: Optional[Union[str, torch.device]] = None,
    check_index: bool = False,
) -> torch.device:
    """
    Return a representative device for a Tensor/Module and optionally assert uniformity.

    Args:
        obj: Tensor or nn.Module.
        recursive: If obj is an nn.Module, traverse submodules (parameters/buffers)
                   recursively (like module.parameters(recurse=True)).
        assert_mode: If True, perform assertions about device placement:
            - If `expected` is provided: assert that ALL params/buffers live on a device
              whose .type matches `expected`'s .type (and, if check_index, the same index).
            - If `expected` is None: assert that ALL params/buffers share a single uniform
              device type (and, if check_index, the same index).
        expected: A target device or device string (e.g., "cpu", "cuda", "cuda:1").
        check_index: If True, also require the same device index (e.g., all on cuda:0).

    Returns:
        torch.device: A representative device. Priority order:
            - Tensor: its own device
            - Module: the first parameter device, else first buffer device, else CPU
    """
    # --- Helper to normalize an "expected" device to (type, index) ---
    def _normalize_expected(exp: Optional[Union[str, torch.device]]):
        if exp is None:
            return None, None
        dev = torch.device(exp) if isinstance(exp, str) else exp
        return dev.type, dev.index

    # --- Collect devices present on the object ---
    if isinstance(obj, torch.Tensor):
        devices = [obj.device]
    elif isinstance(obj, nn.Module):
        # Pull parameters/buffers; recurse if requested
        params = list(obj.parameters(recurse=recursive))
        buffs = list(obj.buffers(recurse=recursive))
        devices = []
        if params:
            devices.extend(p.device for p in params)
        if buffs:
            devices.extend(b.device for b in buffs)
        if not devices:
            devices = [CPU]
    else:
        raise TypeError(f"get_device() expects Tensor or nn.Module, got {type(obj)}")

    # Representative device (keep legacy behavior)
    rep_device = devices[0]

    # --- Assertions (if requested) ---
    if assert_mode:
        exp_type, exp_index = _normalize_expected(expected)

        def _key(d: torch.device):
            return (d.type, d.index if check_index else None)

        if exp_type is not None:
            # Check against expected device TYPE (and optionally INDEX)
            mismatches = [
                d for d in devices
                if d.type != exp_type or (check_index and d.index != exp_index)
            ]
            if mismatches:
                # Build a concise error message with a few examples
                sample = ", ".join({f"{d.type}:{d.index}" for d in mismatches[:5]})
                target = f"{exp_type}" + (f":{exp_index}" if check_index else "")
                raise AssertionError(
                    f"Device assertion failed: expected all tensors on {target}, "
                    f"but found mismatches (e.g., {sample}). Total tensors checked: {len(devices)}."
                )
        else:
            # Ensure uniformity across all devices (by type, and optionally index)
            unique = { _key(d) for d in devices }
            if len(unique) > 1:
                # Summarize what we actually found
                summary = ", ".join(sorted(f"{t}:{i}" for (t, i) in unique))
                detail = ", ".join({f"{d.type}:{d.index}" for d in devices[:8]})
                msg = (
                    "Device assertion failed: tensors are on multiple devices. "
                    f"Found {{{summary}}}. Examples: {detail}."
                )
                if not check_index:
                    msg += " (Tip: set check_index=True to also require same device index.)"
                raise AssertionError(msg)

    return rep_device
