# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Restore packed checkpoint state when a fused CPU layer resumes training."""

from __future__ import annotations

import torch
import torch.nn as nn


_STATE_NAMES = ("qweight", "qzeros", "scales")
_STATE_BUFFER_PREFIX = "_fused_train_state_"


def capture_fused_train_state(module: nn.Module, *, names=None) -> None:
    """Retain packed checkpoint tensors before an inference-only transform.

    The clone deliberately escapes the caller's ``inference_mode``. Caches are
    captured from inside an inference forward, but the snapshot is later
    restored as the checkpoint tensor: ``load_state_dict`` copies into it in
    place outside ``inference_mode``, which Torch rejects for inference tensors.
    """

    if getattr(module, "_fused_train_state", None) is not None:
        return
    if names is None:
        names = getattr(module, "_FUSED_TRAIN_STATE_NAMES", _STATE_NAMES)
    snapshot = {}
    with torch.inference_mode(False):
        for name in names:
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


def state_dict_matches_checkpoint_layout(module: nn.Module, state_dict, prefix: str = "") -> bool:
    """Report whether an incoming state dict describes the checkpoint layout.

    A CPU inference forward replaces the checkpoint operands with int4pack
    shapes in place. Two very different state dicts can then arrive:

    * a real checkpoint (``save()`` output or the file being resumed), whose
      shapes match the ``_fused_train_state`` snapshot, and
    * the model's own runtime ``state_dict()``, whose shapes match the live
      transformed operands.

    Only the first must be restored into the snapshot layout; restoring for the
    second would reject a self round-trip that must keep working.
    """

    snapshot = getattr(module, "_fused_train_state", None)
    if not snapshot:
        return False
    for name, buffer_name in snapshot.items():
        incoming = state_dict.get(f"{prefix}{name}")
        expected = module._buffers.get(buffer_name)
        if not isinstance(incoming, torch.Tensor) or expected is None:
            continue
        if tuple(incoming.shape) == tuple(expected.shape):
            return True
    return False


__all__ = [
    "capture_fused_train_state",
    "discard_fused_train_state",
    "restore_fused_train_state",
    "state_dict_matches_checkpoint_layout",
]
