# SPDX-License-Identifier: Apache-2.0
"""Explicit constructors for packed checkpoint modules; no dynamic imports from data."""

from enum import Enum

import torch

from ..nn_modules.exllamav3 import ExllamaV3Linear
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.paroquant import ParoLinear
from ..quantization.config import resolve_quant_format, serialize_quant_bits
from ..quantization.module_config import ModuleConstructorConfig, module_constructor_config
from ..utils.importer import get_auto_backend_kernel_mapping
from .checkpoint_store import CheckpointError


def is_packed_module(module):
    return isinstance(module, (BaseQuantLinear, ExllamaV3Linear))


def _serialize_value(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    return str(value)


def _deserialize_pack_dtype(value, default):
    if value is None:
        return default
    if not isinstance(value, str):
        raise CheckpointError("invalid checkpoint pack dtype")
    dtype = getattr(torch, value, None)
    if dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise CheckpointError(f"unsupported checkpoint pack dtype: {value}")
    return dtype


def _kernel_path(kernel):
    return f"{kernel.__module__}.{kernel.__qualname__}"


def _resolve_checkpoint_kernel(checkpoint_kernel, default_kernel, config, format_value):
    if checkpoint_kernel is None:
        return default_kernel
    candidates = {default_kernel}
    registry = get_auto_backend_kernel_mapping()
    candidates.update(
        registry.get(config.method, {}).get(format_value, {}).values()
    )
    for kernel in candidates:
        if _kernel_path(kernel) == checkpoint_kernel:
            return kernel
    raise CheckpointError(
        f"checkpoint module kernel is unavailable: {checkpoint_kernel}"
    )


def packed_module_config(module, config, *, module_name=None):
    dimensions = {
        "in_features": module.in_features,
        "out_features": module.out_features,
    }
    if type(module) is ExllamaV3Linear:
        return {
            **dimensions,
            "kind": "exl3",
            "out_dtype": str(module.out_dtype),
            "buffers": {
                key: {
                    "shape": list(value.shape),
                    "torch_dtype": str(value.dtype).removeprefix("torch."),
                }
                for key, value in module.state_dict().items()
            },
        }
    constructor = module_constructor_config(module, config, module_name=module_name)
    module_config = {
        **dimensions,
        "bits": serialize_quant_bits(constructor.bits),
        "group_size": constructor.group_size,
        "desc_act": constructor.desc_act,
        "sym": constructor.sym,
        "bias": getattr(module, "bias", None) is not None,
        "pack_dtype": _serialize_value(constructor.pack_dtype),
        "format": _serialize_value(constructor.format),
    }
    if constructor.kind is not None:
        module_config["kind"] = constructor.kind
    if constructor.options:
        module_config["options"] = _serialize_value(constructor.options_dict())
    if is_packed_module(module):
        module_config["kernel"] = _kernel_path(type(module))
    if getattr(module, "QUANT_TYPE", None) == "bitsandbytes":
        module_config["compute_dtype"] = str(module.compute_dtype) if module.compute_dtype is not None else None
        if module.is_4bit:
            # Packed QuantState JSON length depends on values, not dimensions.
            module_config["quant_state_shape"] = list(module.weight_quant_state.shape)
    if getattr(module, "QUANT_TYPE", None) in {"gptq_bitblas", "awq_bitblas"}:
        module_config["compute_dtype"] = str(module.quant_config.torch_dtype)
    if constructor.kind == "qqq":
        module_config["group_scales"] = "s_group" in module.state_dict()
    return module_config


def restore_packed_module(module_config, *, name, config, kernel, lm_head_name):
    with torch.device("meta"):
        if module_config.get("kind") == "exl3":
            dtype = {
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
                "torch.float32": torch.float32,
            }.get(module_config["out_dtype"])
            if dtype is None:
                raise CheckpointError("invalid EXL3 output dtype")
            return ExllamaV3Linear(
                in_features=module_config["in_features"],
                out_features=module_config["out_features"],
                name=name,
                out_dtype=dtype,
                tensor_storage={
                    "stored_tensors": {
                        f"{name}.{key}": value for key, value in module_config["buffers"].items()
                    }
                },
            )
        module_config = dict(module_config)
        kind = module_config.pop("kind", None)
        group_scales = module_config.pop("group_scales", True)
        checkpoint_kernel = module_config.pop("kernel", None)
        checkpoint_pack_dtype = module_config.pop("pack_dtype", None)
        checkpoint_format = module_config.pop("format", None)
        has_compute_dtype = "compute_dtype" in module_config
        dtype_name = module_config.pop("compute_dtype", None)
        quant_state_shape = module_config.pop("quant_state_shape", None)
        if kind == "paro":
            kernel = ParoLinear
        module_format = resolve_quant_format(
            checkpoint_format if checkpoint_format is not None else config.format,
            config.method,
        )
        kernel = _resolve_checkpoint_kernel(
            checkpoint_kernel,
            kernel,
            config,
            module_format,
        )
        # The checkpoint's own constructor config is authoritative. Global method
        # defaults only fill fields the config does not carry.
        constructor = ModuleConstructorConfig(
            bits=module_config.pop("bits"),
            group_size=module_config.pop("group_size"),
            desc_act=module_config.pop("desc_act"),
            sym=module_config.pop("sym"),
            pack_dtype=_deserialize_pack_dtype(
                checkpoint_pack_dtype, config.pack_dtype
            ),
            format=module_format,
            options=tuple(dict(module_config.pop("options", {})).items()),
            kind=kind,
        )
        kwargs = dict(config.quant_linear_init_kwargs())
        kwargs.update(constructor.constructor_kwargs())
        kwargs.update({
            "name": name,
            "lm_head_name": lm_head_name,
            "register_buffers": True,
        })
        if has_compute_dtype and dtype_name is None:
            kwargs["dtype"] = None
        elif dtype_name is not None:
            dtype = {
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
                "torch.float32": torch.float32,
            }.get(dtype_name)
            if dtype is None:
                raise CheckpointError("invalid packed compute dtype")
            kwargs["dtype"] = dtype
        module = kernel(**module_config, **kwargs)
        if quant_state_shape is not None:
            if (
                len(quant_state_shape) != 1
                or type(quant_state_shape[0]) is not int
                or not 0 < quant_state_shape[0] <= 65536
            ):
                raise CheckpointError("invalid packed BNB metadata shape")
            module.weight_quant_state = torch.empty(
                quant_state_shape, dtype=torch.uint8, device="meta"
            )
        if kind == "qqq" and not group_scales:
            # QQQ's pack path omits this optional buffer for per-channel scales.
            del module.s_group
        return module
