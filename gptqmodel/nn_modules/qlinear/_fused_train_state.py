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


def materialize_fused_runtime_state(module: nn.Module) -> bool:
    """Replace inference-tensor operands with normal tensors in place.

    The CPU inference transform runs inside ``inference_mode``, so the live
    ``qweight``/``qzeros``/``scales`` (and a widened ``bias``) are inference
    tensors. ``load_state_dict(assign=False)`` copies into them outside
    inference mode, which Torch rejects. Cloning them into normal tensors keeps
    the runtime layout while allowing that copy; ``assign=True`` does not need
    it because the load replaces the tensors outright.
    """

    snapshot = getattr(module, "_fused_train_state", None)
    if not snapshot:
        return False
    materialized = False
    with torch.inference_mode(False):
        for name in snapshot:
            live = getattr(module, name, None)
            if isinstance(live, torch.Tensor) and torch.is_inference(live):
                _replace_tensor(module, name, live.detach().clone())
                materialized = True
    return materialized


def _discriminating_operands(module: nn.Module):
    """Operands whose snapshot shape differs from the live transformed shape.

    These, and only these, can tell a checkpoint-shaped state dict apart from
    the model's own runtime ``state_dict()``. The CPU int4pack transform keeps
    ``scales`` at its checkpoint shape and the AVX512 transform keeps ``bias``
    at its checkpoint shape (only the dtype widens), so neither is decisive.
    """

    snapshot = getattr(module, "_fused_train_state", None) or {}
    for name, buffer_name in snapshot.items():
        expected = module._buffers.get(buffer_name)
        live = getattr(module, name, None)
        if not isinstance(expected, torch.Tensor) or not isinstance(live, torch.Tensor):
            continue
        if tuple(expected.shape) != tuple(live.shape):
            yield name, buffer_name


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

    The decision uses every operand whose shape actually changed. A single
    coincidentally equal operand - notably ``scales``, which the int4pack
    transform leaves at its checkpoint shape - must never carry the verdict on
    its own, and requiring all discriminating operands to agree is what keeps
    the model's own runtime dict from being mistaken for a checkpoint.

    When no operand changed shape (nothing to discriminate on), or any
    discriminating operand does not match the snapshot, this returns ``False``
    so the load proceeds against the live layout.
    """

    matched = 0
    for name, buffer_name in _discriminating_operands(module):
        incoming = state_dict.get(f"{prefix}{name}")
        expected = module._buffers.get(buffer_name)
        if not isinstance(incoming, torch.Tensor):
            return False
        if tuple(incoming.shape) != tuple(expected.shape):
            return False
        matched += 1
    return matched > 0


def state_dict_matches_runtime_layout(module: nn.Module, state_dict, prefix: str = "") -> bool:
    """Report whether an incoming state dict matches the live transformed layout.

    This identifies the model's own ``state_dict()``. Such a load must copy
    into the live int4pack tensors and then rebuild derived caches *without*
    restoring the checkpoint snapshot, which would rewrite the shapes the
    caller just supplied.
    """

    matched = 0
    for name, _buffer_name in _discriminating_operands(module):
        incoming = state_dict.get(f"{prefix}{name}")
        live = getattr(module, name, None)
        if not isinstance(incoming, torch.Tensor) or not isinstance(live, torch.Tensor):
            return False
        if tuple(incoming.shape) != tuple(live.shape):
            return False
        matched += 1
    return matched > 0


__all__ = [
    "capture_fused_train_state",
    "discard_fused_train_state",
    "materialize_fused_runtime_state",
    "restore_fused_train_state",
    "state_dict_matches_checkpoint_layout",
    "state_dict_matches_runtime_layout",
]
