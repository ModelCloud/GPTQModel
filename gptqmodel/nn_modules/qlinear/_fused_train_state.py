# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Restore packed checkpoint state when a fused CPU layer resumes training."""

from __future__ import annotations

import torch
import torch.nn as nn


_STATE_NAMES = ("qweight", "qzeros", "scales")
_STATE_BUFFER_PREFIX = "_fused_train_state_"


def capture_fused_train_state(module: nn.Module) -> None:
    """Retain packed checkpoint tensors before an inference-only transform."""

    if getattr(module, "_fused_train_state", None) is not None:
        return
    snapshot = {}
    for name in _STATE_NAMES:
        tensor = getattr(module, name, None)
        if isinstance(tensor, torch.Tensor):
            buffer_name = f"{_STATE_BUFFER_PREFIX}{name}"
            module.register_buffer(
                buffer_name,
                tensor.detach().clone(),
                persistent=False,
            )
            snapshot[name] = buffer_name
    module._fused_train_state = snapshot


def _replace_tensor(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    current = getattr(module, name, None)
    if isinstance(current, nn.Parameter):
        module._parameters[name] = nn.Parameter(
            tensor,
            requires_grad=current.requires_grad,
        )
    elif name in module._buffers:
        module._buffers[name] = tensor
    else:
        setattr(module, name, tensor)


def restore_fused_train_state(module: nn.Module) -> bool:
    """Restore checkpoint-packed tensors captured before a fused transform."""

    if getattr(module, "_suspend_fused_train_restore", False):
        return False
    snapshot = getattr(module, "_fused_train_state", None)
    if not snapshot:
        return False
    for name, buffer_name in snapshot.items():
        tensor = module._buffers.pop(buffer_name, None)
        if tensor is not None:
            _replace_tensor(module, name, tensor)
    discard_fused_train_state(module)
    return True


def discard_fused_train_state(module: nn.Module) -> None:
    snapshot = getattr(module, "_fused_train_state", None)
    if snapshot:
        for buffer_name in snapshot.values():
            module._buffers.pop(buffer_name, None)
    module._fused_train_state = None


__all__ = [
    "capture_fused_train_state",
    "discard_fused_train_state",
    "restore_fused_train_state",
]
