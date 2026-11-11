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
        module._fused_train_state_exportable = True


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
    if not getattr(module, "_fused_train_state_exportable", True):
        raise RuntimeError(
            "Cannot restore or save a modified CPU int4pack runtime state as a checkpoint. "
            "Load checkpoint-layout weights to restore the checkpoint format."
        )
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
    module.__dict__.pop("_fused_train_state_exportable", None)


def runtime_state_matches_live(module: nn.Module, state_dict, prefix: str = "") -> bool:
    """Whether a runtime-layout load leaves every local checkpoint operand unchanged."""

    for name, live in (*module._parameters.items(), *module._buffers.items()):
        if name in module._non_persistent_buffers_set:
            continue
        incoming = state_dict.get(f"{prefix}{name}")
        if incoming is None:
            continue
        if not isinstance(live, torch.Tensor) or not isinstance(incoming, torch.Tensor):
            return False
        if live.dtype != incoming.dtype or live.shape != incoming.shape or not torch.equal(live, incoming):
            return False
    return True


def materialize_fused_runtime_state(module: nn.Module, state_dict, prefix: str = "") -> bool:
    """Replace inference-tensor operands with normal tensors in place.

    The CPU inference transform runs inside ``inference_mode``, so the live
    checkpoint operands, including ``g_idx``, can be inference tensors.
    ``load_state_dict(assign=False)`` copies into them outside
    inference mode, which Torch rejects. Cloning them into normal tensors keeps
    the runtime layout while allowing that copy; ``assign=True`` does not need
    it because the load replaces the tensors outright.
    """

    if not getattr(module, "_fused_train_state", None):
        return False
    materialized = False
    with torch.inference_mode(False):
        for name in (*module._parameters, *module._buffers):
            if f"{prefix}{name}" not in state_dict:
                continue
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


def classify_fused_state_layout(module: nn.Module, state_dict, prefix: str = "") -> str:
    """Classify the supplied local operands before changing the live tensors.

    A partial load can supply only one shape-changing operand. Every supplied
    discriminator must agree on its layout; missing operands are not evidence
    against that layout. Shared-shape operands alone use the live layout so a
    partial load cannot silently discard their new values. A load with no local
    operands leaves the transformed module untouched.
    """

    layout = None
    for name, buffer_name in _discriminating_operands(module):
        key = f"{prefix}{name}"
        if key not in state_dict:
            continue
        incoming = state_dict[key]
        expected = module._buffers[buffer_name]
        live = getattr(module, name)
        if not isinstance(incoming, torch.Tensor):
            return "invalid"
        if tuple(incoming.shape) == tuple(expected.shape):
            candidate = "checkpoint"
        elif tuple(incoming.shape) == tuple(live.shape):
            candidate = "runtime"
        else:
            return "invalid"
        if layout is not None and layout != candidate:
            return "invalid"
        layout = candidate
    if layout is not None:
        return layout

    for name in (*module._parameters, *module._buffers):
        if name not in module._non_persistent_buffers_set and f"{prefix}{name}" in state_dict:
            return "runtime"
    return "empty"


def state_dict_matches_checkpoint_layout(module: nn.Module, state_dict, prefix: str = "") -> bool:
    """Whether supplied shape-changing operands match the checkpoint snapshot."""

    return classify_fused_state_layout(module, state_dict, prefix) == "checkpoint"


def state_dict_matches_runtime_layout(module: nn.Module, state_dict, prefix: str = "") -> bool:
    """Whether supplied operands must load into the live transformed layout."""

    return classify_fused_state_layout(module, state_dict, prefix) == "runtime"


__all__ = [
    "capture_fused_train_state",
    "classify_fused_state_layout",
    "discard_fused_train_state",
    "materialize_fused_runtime_state",
    "runtime_state_matches_live",
    "restore_fused_train_state",
    "state_dict_matches_checkpoint_layout",
    "state_dict_matches_runtime_layout",
]
