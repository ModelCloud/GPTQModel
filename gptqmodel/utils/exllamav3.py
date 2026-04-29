# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Portions of this file are adapted from turboderp-org/exllamav3.
# Credits: TurboDerp / ExLlamaV3 contributors.

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Type

import torch
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd

from ..looper.named_module import NamedModule
from ..nn_modules.exllamav3 import ExllamaV3Linear
from .model import recurse_setattr


def _resolve_linear_shape(submodule: nn.Module) -> tuple[int, int]:
    named = submodule if isinstance(submodule, NamedModule) else None
    target = named.module if named is not None else submodule

    if named is not None:
        in_features = named.state.get("in_features")
        out_features = named.state.get("out_features")
        if in_features is not None and out_features is not None:
            return int(in_features), int(out_features)

    if isinstance(target, nn.Linear):
        return target.in_features, target.out_features
    if isinstance(target, _ConvNd):
        return target.in_channels, target.out_channels
    if isinstance(target, transformers.Conv1D):
        return target.weight.shape[0], target.weight.shape[1]

    in_features = getattr(target, "in_features", None)
    out_features = getattr(target, "out_features", None)
    if in_features is not None and out_features is not None:
        return int(in_features), int(out_features)

    raise NotImplementedError(f"Unsupported EXL3 module type: {target.__class__.__name__}")


def create_exllamav3_module(
    *,
    module_root: nn.Module,
    name: str,
    submodule: nn.Module,
    tensors: Dict[str, torch.Tensor],
    module_cls: Type[nn.Module] = ExllamaV3Linear,
) -> nn.Module:
    in_features, out_features = _resolve_linear_shape(submodule)
    new_module = module_cls.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name=name,
        tensors=tensors,
    )
    recurse_setattr(module_root, name, new_module)
    return new_module


def build_exllamav3_tensor_storage(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    storage: Dict[str, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        if getattr(module, "QUANT_TYPE", None) == "exl3" and hasattr(module, "tensor_storage_entry"):
            storage[name] = module.tensor_storage_entry()
    return storage


def replace_exllamav3_placeholders(
    *,
    model: nn.Module,
    module_names: Iterable[str],
    tensor_storage: Optional[Dict[str, Dict[str, Any]]] = None,
    module_cls: Type[nn.Module] = ExllamaV3Linear,
) -> None:
    module_lookup = dict(model.named_modules())
    storage_map = tensor_storage or {}

    for module_name in module_names:
        submodule = module_lookup.get(module_name)
        if submodule is None:
            continue

        if not isinstance(submodule, (nn.Linear, transformers.Conv1D, _ConvNd)):
            continue

        in_features, out_features = _resolve_linear_shape(submodule)
        new_module = module_cls(
            in_features=in_features,
            out_features=out_features,
            name=module_name,
            tensor_storage=storage_map.get(module_name),
        )
        recurse_setattr(model, module_name, new_module)
