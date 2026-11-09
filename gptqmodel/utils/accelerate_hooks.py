# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Accelerate hooks that route cross-device activation moves safely.

``accelerate.hooks.AlignDevicesHook.pre_forward`` moves activations with the
module-level ``accelerate.hooks.send_to_device``, which calls ``Tensor.to``
directly. On an asymmetric ROCm peer topology that call segfaults inside
``hipMemcpyPeerAsync`` instead of raising. Overriding the bound method on an
instance cannot fix this, because ``pre_forward`` resolves the module-level
function, so this module subclasses the hook and routes every tensor through
:func:`gptqmodel.utils.p2p.safe_move_to`.
"""

from __future__ import annotations

from typing import Mapping

import torch
from accelerate.hooks import AlignDevicesHook, SequentialHook, find_device
from accelerate.utils import honor_type

from .p2p import safe_move_to
from .torch import as_torch_device


def _resolve_device(device):
    """Resolve accelerate's ``str``/``int``/``torch.device`` device forms."""

    if isinstance(device, int):
        from accelerate.utils import torch_device

        return torch.device(torch_device(), device)
    return as_torch_device(device)


def safe_send_to_device(tensor, device, non_blocking: bool = False, skip_keys=None):
    """Recursively move tensors to ``device`` through :func:`safe_move_to`.

    This mirrors ``accelerate.hooks.send_to_device``: nested lists, tuples and
    mappings are rebuilt with their original container type, and non-tensor
    objects that expose ``to`` keep accelerate's behavior.
    """

    if isinstance(tensor, torch.Tensor):
        return safe_move_to(tensor, device, non_blocking=non_blocking)
    if hasattr(tensor, "to"):
        try:
            return tensor.to(device, non_blocking=non_blocking)
        except TypeError:
            return tensor.to(device)
    if isinstance(tensor, (tuple, list)):
        return honor_type(
            tensor,
            (
                safe_send_to_device(item, device, non_blocking=non_blocking, skip_keys=skip_keys)
                for item in tensor
            ),
        )
    if isinstance(tensor, Mapping):
        if isinstance(skip_keys, str):
            skip_keys = [skip_keys]
        elif skip_keys is None:
            skip_keys = []
        return type(tensor)(
            {
                key: value
                if key in skip_keys
                else safe_send_to_device(value, device, non_blocking=non_blocking, skip_keys=skip_keys)
                for key, value in tensor.items()
            }
        )
    return tensor


class GPTQModelAlignDevicesHook(AlignDevicesHook):
    """``AlignDevicesHook`` with direction-aware activation moves.

    ``pre_forward`` moves the inputs before delegating to the upstream hook, so
    the upstream ``send_to_device`` only sees tensors already on the execution
    device and becomes a no-op. ``post_forward`` does the same for the output
    returned to the caller's input device. Offloaded weights still go through
    the upstream implementation.
    """

    def pre_forward(self, module, *args, **kwargs):
        if self.execution_device is None:
            return super().pre_forward(module, *args, **kwargs)

        # Capture the caller's device before moving the inputs: upstream's
        # io_same_device handling returns the output to this device.
        input_device = find_device([args, kwargs]) if self.io_same_device else None
        device = _resolve_device(self.execution_device)
        args = safe_send_to_device(args, device)
        kwargs = safe_send_to_device(kwargs, device, skip_keys=self.skip_keys)
        result = super().pre_forward(module, *args, **kwargs)
        if self.io_same_device:
            self.input_device = input_device
        return result

    def post_forward(self, module, output):
        if self.io_same_device and self.input_device is not None:
            output = safe_send_to_device(output, self.input_device, skip_keys=self.skip_keys)
        return super().post_forward(module, output)


def _upgrade_hook(hook) -> int:
    if isinstance(hook, GPTQModelAlignDevicesHook):
        return 0
    if isinstance(hook, AlignDevicesHook):
        # In-place upgrade keeps weights maps, tied-parameter state and any
        # other attributes the hook accumulated while attached.
        hook.__class__ = GPTQModelAlignDevicesHook
        return 1
    if isinstance(hook, SequentialHook):
        return sum(_upgrade_hook(child) for child in hook.hooks)
    return 0


def install_safe_align_devices_hooks(model: torch.nn.Module) -> int:
    """Replace already-attached ``AlignDevicesHook`` instances with the safe one.

    ``simple_dispatch_model`` creates the safe hook directly. This helper also
    covers a model that was dispatched elsewhere (for example by
    ``accelerate.dispatch_model``), including hooks nested in a
    ``SequentialHook``. Returns the number of upgraded hooks.
    """

    upgraded = 0
    for module in model.modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is not None:
            upgraded += _upgrade_hook(hook)
    return upgraded
