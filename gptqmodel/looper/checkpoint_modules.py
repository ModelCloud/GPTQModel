# SPDX-License-Identifier: Apache-2.0
"""Explicit constructors for packed checkpoint modules; no dynamic imports from data."""

from enum import Enum

import torch

from ..nn_modules.exllamav3 import ExllamaV3Linear
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.paroquant import ParoLinear
from ..nn_modules.qlinear.qqq import QQQLinear
from ..quantization.config import resolve_quant_format, serialize_quant_bits
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


def _module_checkpoint_format(module, config, module_name):
    format_value = getattr(module, "format", None)
    if format_value is None and module_name is not None:
        format_value = config.resolve_module_config(module_name).get(
            "format", config.format
        )
    if format_value is None or format_value is False:
        format_value = config.format
    return resolve_quant_format(format_value, config.method)


def packed_module_spec(module, config, *, module_name=None):
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
    spec = {
        **dimensions,
        "bits": serialize_quant_bits(module.bits),
        "group_size": getattr(module, "group_size", config.group_size),
        "desc_act": getattr(module, "desc_act", config.desc_act),
        "sym": getattr(module, "sym", config.sym),
        "bias": getattr(module, "bias", None) is not None,
        "pack_dtype": _serialize_value(
            getattr(module, "pack_dtype", config.pack_dtype)
        ),
        "format": _serialize_value(
            _module_checkpoint_format(module, config, module_name)
        ),
    }
    if is_packed_module(module):
        spec["kernel"] = _kernel_path(type(module))
    if type(module) is ParoLinear:
        spec["kind"] = "paro"
        spec["options"] = {
            key: getattr(module, key)
            for key in (
                "krot",
                "fp32_accum",
                "cache_runtime_dtype",
                "auto_cache_bf16_runtime_dtype",
                "cache_rotation_dtype",
                "auto_cache_bf16_rotation_dtype",
            )
        }
    if getattr(module, "QUANT_TYPE", None) == "fp8":
        spec["options"] = {
            "format": module.fp8_format,
            "weight_scale_method": module.weight_scale_method,
            "weight_block_size": (
                list(module.weight_block_size)
                if module.weight_block_size is not None
                else None
            ),
            "weight_scale_semantics": module.weight_scale_semantics,
        }
    if getattr(module, "QUANT_TYPE", None) == "bitsandbytes":
        spec["compute_dtype"] = str(module.compute_dtype) if module.compute_dtype is not None else None
        spec["options"] = {
            "format": module.bnb_format,
            "block_size": module.bnb_block_size,
            "compress_statistics": module.bnb_compress_statistics,
        }
        if module.is_4bit:
            # Packed QuantState JSON length depends on values, not dimensions.
            spec["quant_state_shape"] = list(module.weight_quant_state.shape)
    if getattr(module, "QUANT_TYPE", None) in {"gptq_bitblas", "awq_bitblas"}:
        spec["compute_dtype"] = str(module.quant_config.torch_dtype)
    if type(module) is QQQLinear:
        spec["kind"] = "qqq"
        spec["group_scales"] = "s_group" in module.state_dict()
    return spec


def restore_packed_module(spec, *, name, config, kernel, lm_head_name):
    with torch.device("meta"):
        if spec.get("kind") == "exl3":
            dtype = {
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
                "torch.float32": torch.float32,
            }.get(spec["out_dtype"])
            if dtype is None:
                raise CheckpointError("invalid EXL3 output dtype")
            return ExllamaV3Linear(
                in_features=spec["in_features"],
                out_features=spec["out_features"],
                name=name,
                out_dtype=dtype,
                tensor_storage={
                    "stored_tensors": {
                        f"{name}.{key}": value for key, value in spec["buffers"].items()
                    }
                },
            )
        spec = dict(spec)
        kind = spec.pop("kind", None)
        group_scales = spec.pop("group_scales", True)
        options = spec.pop("options", {})
        checkpoint_kernel = spec.pop("kernel", None)
        checkpoint_pack_dtype = spec.pop("pack_dtype", None)
        checkpoint_format = spec.pop("format", None)
        has_compute_dtype = "compute_dtype" in spec
        dtype_name = spec.pop("compute_dtype", None)
        quant_state_shape = spec.pop("quant_state_shape", None)
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
        # Global method defaults come first so the checkpoint's resolved
        # per-module settings always win during reconstruction.
        kwargs = dict(config.quant_linear_init_kwargs())
        kwargs.update({
            "pack_dtype": _deserialize_pack_dtype(
                checkpoint_pack_dtype, config.pack_dtype
            ),
            "name": name,
            "lm_head_name": lm_head_name,
            "format": module_format,
            "register_buffers": True,
        })
        kwargs.update(options)
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
        module = kernel(**spec, **kwargs)
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
