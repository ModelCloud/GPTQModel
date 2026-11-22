# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPT-QModel-owned Accelerate device-alignment hooks.

Accelerate's :class:`~accelerate.hooks.AlignDevicesHook` moves tensors through
``set_module_tensor_to_device`` with its default ``clear_cache=True``, so every
offloaded forward (and every hook init / detach) on an accelerator empties the
device cache. GPT-QModel drives offloaded modules from several worker threads
at once, where a cache flush per forward is unsafe and slow.

GPT-QModel used to silence that by replacing
``accelerate.utils.modeling.clear_device_cache`` process-wide, which changed
Accelerate for every other user in the process. Instead, the hooks GPT-QModel
attaches (or adopts on its own model instance) are
:class:`GPTQModelAlignDevicesHook`, which performs the same moves with
``clear_cache=False``. Accelerate itself is left exactly as it ships.
"""

from __future__ import annotations

import functools
import os
from typing import Mapping, Optional

import torch
from accelerate.hooks import AlignDevicesHook, SequentialHook, add_hook_to_module
from accelerate.utils import (
    OffloadedWeightsLoader,
    PrefixedDataset,
    find_device,
    named_module_tensors,
    offload_state_dict,
    send_to_device,
    set_module_tensor_to_device,
)
from accelerate.utils.modeling import get_non_persistent_buffers
from torch import nn


__all__ = [
    "GPTQModelAlignDevicesHook",
    "adopt_align_device_hooks",
    "attach_align_device_hook",
    "disk_offload",
]


def _compiler_disable(fn):
    """Exclude weight onloading from ``torch.compile`` tracing, lazily."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        compiled = wrapper.__dict__.get("_disabled_fn")
        if compiled is None:
            compiled = torch.compiler.disable(fn)
            wrapper._disabled_fn = compiled
        return compiled(*args, **kwargs)

    return wrapper


def _normalize_device(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return device
    return torch.device(device) if device is not None else None


class GPTQModelAlignDevicesHook(AlignDevicesHook):
    """``AlignDevicesHook`` that never empties the device cache.

    Identical placement semantics to Accelerate's hook; every tensor move passes
    ``clear_cache=False``. ``post_forward`` (moves to ``meta`` only) is inherited.
    """

    def _fp16_statistics(self, name, value):
        if value is None or value.dtype != torch.int8 or "weight" not in name:
            return None
        statistics_name = name.replace("weight", "SCB")
        if self.weights_map is not None and statistics_name in self.weights_map:
            return self.weights_map[statistics_name]
        return None

    def init_hook(self, module):
        if self.execution_device == "meta" or self.execution_device == torch.device("meta"):
            self.tied_params_map = None

        if not self.offload and self.execution_device is not None:
            for name, _ in named_module_tensors(module, recurse=self.place_submodules):
                set_module_tensor_to_device(
                    module, name, self.execution_device,
                    tied_params_map=self.tied_params_map, clear_cache=False,
                )
        elif self.offload:
            self.original_devices = {
                name: param.device
                for name, param in named_module_tensors(module, recurse=self.place_submodules)
            }
            if self.weights_map is None:
                self.weights_map = {
                    name: param.to("cpu")
                    for name, param in named_module_tensors(
                        module, include_buffers=self.offload_buffers, recurse=self.place_submodules
                    )
                }
            for name, _ in named_module_tensors(
                module, include_buffers=self.offload_buffers, recurse=self.place_submodules,
                remove_non_persistent=True,
            ):
                if (
                    self.tied_params_map is not None
                    and _recursive_getattr(module, name).data_ptr() in self.tied_params_map
                ):
                    self.tied_params_names.add(name)
                set_module_tensor_to_device(module, name, "meta", clear_cache=False)

            if self.execution_device is not None:
                names = (
                    get_non_persistent_buffers(module, recurse=self.place_submodules)
                    if self.offload_buffers
                    else [name for name, _ in module.named_buffers(recurse=self.place_submodules)]
                )
                for name in names:
                    set_module_tensor_to_device(
                        module, name, self.execution_device,
                        tied_params_map=self.tied_params_map, clear_cache=False,
                    )
        return module

    @_compiler_disable
    def _onload_weights(self, module):
        self.tied_pointers_to_remove = set()
        execution_device = _normalize_device(self.execution_device)
        for name, _ in named_module_tensors(
            module, include_buffers=self.offload_buffers, recurse=self.place_submodules,
            remove_non_persistent=True,
        ):
            value = self.weights_map[name]
            if name in self.tied_params_names and value.data_ptr() not in self.tied_params_map:
                self.tied_params_map[value.data_ptr()] = {}
            if (
                value is not None
                and self.tied_params_map is not None
                and value.data_ptr() in self.tied_params_map
                and execution_device not in self.tied_params_map[value.data_ptr()]
            ):
                self.tied_pointers_to_remove.add((value.data_ptr(), execution_device))
            set_module_tensor_to_device(
                module, name, execution_device, value=value,
                fp16_statistics=self._fp16_statistics(name, value),
                tied_params_map=self.tied_params_map, clear_cache=False,
            )

    def pre_forward(self, module, *args, **kwargs):
        if self.io_same_device:
            self.input_device = find_device([args, kwargs])
        if self.offload:
            self._onload_weights(module)
        return send_to_device(args, self.execution_device), send_to_device(
            kwargs, self.execution_device, skip_keys=self.skip_keys
        )

    def detach_hook(self, module):
        if self.offload:
            for name, device in self.original_devices.items():
                if device != torch.device("meta"):
                    value = self.weights_map.get(name, None)
                    set_module_tensor_to_device(
                        module, name, device, value=value,
                        fp16_statistics=self._fp16_statistics(name, value), clear_cache=False,
                    )
        return module

    @classmethod
    def adopt(cls, hook: AlignDevicesHook) -> "GPTQModelAlignDevicesHook":
        """Return a GPT-QModel hook carrying ``hook``'s already initialized state."""

        if isinstance(hook, cls):
            return hook
        adopted = cls.__new__(cls)
        adopted.__dict__.update(hook.__dict__)
        return adopted


def _recursive_getattr(obj, attr: str):
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def attach_align_device_hook(
    module: nn.Module,
    execution_device: Optional[torch.device] = None,
    offload: bool = False,
    weights_map: Optional[Mapping] = None,
    offload_buffers: bool = False,
    module_name: str = "",
    skip_keys=None,
    preload_module_classes=None,
    tied_params_map=None,
) -> None:
    """Recursive ``attach_align_device_hook`` that installs GPT-QModel hooks."""

    directs = list(named_module_tensors(module))
    full_offload = (
        offload and preload_module_classes is not None
        and module.__class__.__name__ in preload_module_classes
    )
    if directs or full_offload:
        prefixed = None
        if weights_map is not None:
            prefix = f"{module_name}." if module_name else ""
            prefixed = PrefixedDataset(weights_map, prefix)
        add_hook_to_module(
            module,
            GPTQModelAlignDevicesHook(
                execution_device=execution_device,
                offload=offload,
                weights_map=prefixed,
                offload_buffers=offload_buffers,
                place_submodules=full_offload,
                skip_keys=skip_keys,
                tied_params_map=tied_params_map,
            ),
            append=True,
        )
    if full_offload:
        return
    for child_name, child in module.named_children():
        attach_align_device_hook(
            child,
            execution_device=execution_device,
            offload=offload,
            weights_map=weights_map,
            offload_buffers=offload_buffers,
            module_name=f"{module_name}.{child_name}" if module_name else child_name,
            skip_keys=skip_keys,
            preload_module_classes=preload_module_classes,
            tied_params_map=tied_params_map,
        )


def disk_offload(
    model: nn.Module,
    offload_dir,
    execution_device: Optional[torch.device] = None,
    offload_buffers: bool = False,
    preload_module_classes=None,
) -> nn.Module:
    """``accelerate.disk_offload`` built from :class:`GPTQModelAlignDevicesHook`."""

    if not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json")):
        offload_state_dict(offload_dir, model.state_dict())
    if execution_device is None:
        execution_device = next(iter(model.parameters())).device
    weights_map = OffloadedWeightsLoader(save_folder=offload_dir)
    add_hook_to_module(model, GPTQModelAlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(
        model,
        execution_device=execution_device,
        offload=True,
        offload_buffers=offload_buffers,
        weights_map=weights_map,
        preload_module_classes=preload_module_classes,
    )
    return model


def adopt_align_device_hooks(model: nn.Module) -> int:
    """Swap stock Accelerate ``AlignDevicesHook`` objects on ``model`` for GPT-QModel hooks.

    Transformers' ``from_pretrained(device_map=...)`` dispatches through
    Accelerate and attaches stock hooks. This rebinds only the hook objects on
    this model instance (state is carried over; ``init_hook`` is not rerun), so
    later forwards on GPT-QModel's model do not flush the device cache.
    Returns the number of hooks adopted.
    """

    adopted = 0
    if not isinstance(model, nn.Module):
        return adopted

    def _convert(hook):
        nonlocal adopted
        if isinstance(hook, SequentialHook):
            hook.hooks = type(hook.hooks)(_convert(inner) for inner in hook.hooks)
            return hook
        if type(hook) is AlignDevicesHook:
            adopted += 1
            return GPTQModelAlignDevicesHook.adopt(hook)
        return hook

    for submodule in model.modules():
        hook = submodule.__dict__.get("_hf_hook")
        if hook is None:
            continue
        converted = _convert(hook)
        if converted is not hook:
            submodule._hf_hook = converted
    return adopted
