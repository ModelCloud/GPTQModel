# SPDX-License-Identifier: Apache-2.0
"""Explicit constructors for packed checkpoint modules; no dynamic imports from data."""

import torch

from ..nn_modules.exllamav3 import ExllamaV3Linear
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.paroquant import ParoLinear
from ..nn_modules.qlinear.qqq import QQQLinear
from ..quantization.config import resolve_quant_format, serialize_quant_bits
from .checkpoint_store import CheckpointError


def is_packed_module(module):
    return isinstance(module, (BaseQuantLinear, ExllamaV3Linear))


def packed_module_spec(module, config):
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
    }
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
    if getattr(module, "QUANT_TYPE", None) == "bitsandbytes":
        spec["compute_dtype"] = str(module.compute_dtype) if module.compute_dtype is not None else None
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
        has_compute_dtype = "compute_dtype" in spec
        dtype_name = spec.pop("compute_dtype", None)
        quant_state_shape = spec.pop("quant_state_shape", None)
        if kind == "paro":
            kernel = ParoLinear
        kwargs = {
            "pack_dtype": config.pack_dtype,
            "name": name,
            "lm_head_name": lm_head_name,
            "format": resolve_quant_format(config.format, config.method),
            "register_buffers": True,
        }
        kwargs.update(config.quant_linear_init_kwargs())
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
