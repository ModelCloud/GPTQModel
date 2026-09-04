# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed FP8 KV-cache storage for QVQ A8 checkpoints."""

from __future__ import annotations

import inspect
import types
from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicLayer

from ..quantization.config import QVQActivationConfig
from ..quantization.qvq_activation import (
    QVQ_FP8_ACTIVATION_FORMAT,
    dequantize_qvq_fp8_activation,
    quantize_qvq_fp8_activation,
)


def _activation_config(
    value: QVQActivationConfig | dict[str, Any],
) -> QVQActivationConfig:
    if isinstance(value, QVQActivationConfig):
        value.__post_init__()
        return value
    if isinstance(value, dict):
        return QVQActivationConfig(**value)
    raise TypeError("QVQ FP8 KV cache requires a QVQActivationConfig or dictionary.")


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


class QVQFP8CacheLayer(DynamicLayer):
    """Dynamic cache layer with no full-precision persistent K/V residual."""

    def __init__(self, activation_quantization: QVQActivationConfig | dict[str, Any]):
        super().__init__()
        self.activation_quantization = _activation_config(activation_quantization)
        self.key_scales: torch.Tensor | None = None
        self.value_scales: torch.Tensor | None = None
        self.source_dtype: torch.dtype | None = None
        self.update_calls = 0
        self.quantized_elements = 0
        self.dequantized_elements = 0

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError(
                "QVQ FP8 KV cache expects K/V tensors shaped [batch, heads, tokens, width]."
            )
        if key_states.shape[:-1] != value_states.shape[:-1]:
            raise ValueError(
                "QVQ FP8 KV cache requires matching K/V batch, head, and token dimensions."
            )
        if (
            key_states.device != value_states.device
            or key_states.dtype != value_states.dtype
        ):
            raise ValueError(
                "QVQ FP8 KV cache requires K/V on the same device and with the same dtype."
            )
        if not key_states.is_floating_point() or key_states.dtype == getattr(
            torch, QVQ_FP8_ACTIVATION_FORMAT
        ):
            raise TypeError(
                "QVQ FP8 KV cache inputs must be full-precision floating-point tensors."
            )

        self.dtype = key_states.dtype
        self.source_dtype = key_states.dtype
        self.device = key_states.device
        fp8_dtype = getattr(torch, self.activation_quantization.format)
        self.keys = torch.empty(
            (*key_states.shape[:-2], 0, key_states.shape[-1]),
            dtype=fp8_dtype,
            device=self.device,
        )
        self.values = torch.empty(
            (*value_states.shape[:-2], 0, value_states.shape[-1]),
            dtype=fp8_dtype,
            device=self.device,
        )
        scale_shape = (*key_states.shape[:-2], 0, 1)
        self.key_scales = torch.empty(
            scale_shape, dtype=torch.float32, device=self.device
        )
        self.value_scales = torch.empty(
            scale_shape, dtype=torch.float32, device=self.device
        )
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if (
            key_states.dtype != self.source_dtype
            or value_states.dtype != self.source_dtype
        ):
            raise TypeError(
                "QVQ FP8 KV cache source dtype changed after initialization: "
                f"expected {self.source_dtype}, got K={key_states.dtype}, V={value_states.dtype}."
            )

        validate = key_states.device.type == "cpu"
        quantized_keys, key_scales = quantize_qvq_fp8_activation(
            key_states,
            format=self.activation_quantization.format,
            scale_method=self.activation_quantization.scale_method,
            validate=validate,
        )
        quantized_values, value_scales = quantize_qvq_fp8_activation(
            value_states,
            format=self.activation_quantization.format,
            scale_method=self.activation_quantization.scale_method,
            validate=validate,
        )
        self.keys = torch.cat((self.keys, quantized_keys), dim=-2)
        self.values = torch.cat((self.values, quantized_values), dim=-2)
        self.key_scales = torch.cat((self.key_scales, key_scales), dim=-2)
        self.value_scales = torch.cat((self.value_scales, value_scales), dim=-2)
        self.update_calls += 1
        self.quantized_elements += key_states.numel() + value_states.numel()
        self.assert_fp8_storage()

        keys = dequantize_qvq_fp8_activation(
            self.keys, self.key_scales, dtype=self.source_dtype
        )
        values = dequantize_qvq_fp8_activation(
            self.values, self.value_scales, dtype=self.source_dtype
        )
        self.dequantized_elements += keys.numel() + values.numel()
        return keys, values

    def assert_fp8_storage(self) -> None:
        if not self.is_initialized:
            return
        fp8_dtype = getattr(torch, self.activation_quantization.format)
        if self.keys.dtype != fp8_dtype or self.values.dtype != fp8_dtype:
            raise RuntimeError(
                f"QVQ A8 requires FP8 KV payloads, got K={self.keys.dtype}, V={self.values.dtype}."
            )
        if (
            self.key_scales.dtype != torch.float32
            or self.value_scales.dtype != torch.float32
        ):
            raise RuntimeError("QVQ A8 requires FP32 per-token KV scales.")
        if (
            self.keys.shape[:-1] != self.key_scales.shape[:-1]
            or self.key_scales.shape[-1] != 1
        ):
            raise RuntimeError("QVQ A8 key payload and scale geometry differ.")
        if (
            self.values.shape[:-1] != self.value_scales.shape[:-1]
            or self.value_scales.shape[-1] != 1
        ):
            raise RuntimeError("QVQ A8 value payload and scale geometry differ.")

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]
        self.key_scales = self.key_scales[..., :max_length, :]
        self.value_scales = self.value_scales[..., :max_length, :]

    def reset(self) -> None:
        if not self.is_initialized:
            return
        self.keys = self.keys[..., :0, :]
        self.values = self.values[..., :0, :]
        self.key_scales = self.key_scales[..., :0, :]
        self.value_scales = self.value_scales[..., :0, :]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() < 1:
            return
        index = beam_idx.to(self.device)
        self.keys = self.keys.index_select(0, index)
        self.values = self.values.index_select(0, index)
        self.key_scales = self.key_scales.index_select(0, index)
        self.value_scales = self.value_scales.index_select(0, index)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() < 1:
            return
        self.keys = self.keys.repeat_interleave(repeats, dim=0)
        self.values = self.values.repeat_interleave(repeats, dim=0)
        self.key_scales = self.key_scales.repeat_interleave(repeats, dim=0)
        self.value_scales = self.value_scales.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() < 1:
            return
        self.keys = self.keys[indices, ...]
        self.values = self.values[indices, ...]
        self.key_scales = self.key_scales[indices, ...]
        self.value_scales = self.value_scales[indices, ...]

    def offload(self) -> None:
        if not self.is_initialized:
            return
        super().offload()
        self.key_scales = self.key_scales.to("cpu", non_blocking=True)
        self.value_scales = self.value_scales.to("cpu", non_blocking=True)

    def prefetch(self) -> None:
        if not self.is_initialized or self.keys.device == self.device:
            return
        super().prefetch()
        self.key_scales = self.key_scales.to(self.device, non_blocking=True)
        self.value_scales = self.value_scales.to(self.device, non_blocking=True)

    def telemetry(self) -> dict[str, Any]:
        if not self.is_initialized:
            return {
                "initialized": False,
                "sequence_length": 0,
                "storage_bytes": 0,
            }
        self.assert_fp8_storage()
        payload_bytes = _tensor_bytes(self.keys) + _tensor_bytes(self.values)
        scale_bytes = _tensor_bytes(self.key_scales) + _tensor_bytes(self.value_scales)
        dense_bytes = (self.keys.numel() + self.values.numel()) * torch.empty(
            (), dtype=self.source_dtype
        ).element_size()
        return {
            "initialized": True,
            "sequence_length": self.get_seq_length(),
            "key_shape": list(self.keys.shape),
            "value_shape": list(self.values.shape),
            "payload_dtype": str(self.keys.dtype),
            "scale_dtype": str(self.key_scales.dtype),
            "source_dtype": str(self.source_dtype),
            "payload_bytes": payload_bytes,
            "scale_bytes": scale_bytes,
            "storage_bytes": payload_bytes + scale_bytes,
            "dense_equivalent_bytes": dense_bytes,
            "update_calls": self.update_calls,
            "quantized_elements": self.quantized_elements,
            "dequantized_elements": self.dequantized_elements,
        }


class QVQFP8DynamicCache(Cache):
    """Full-layer FP8 cache automatically required by QVQ A8 models."""

    def __init__(
        self, config, activation_quantization: QVQActivationConfig | dict[str, Any]
    ):
        activation_quantization = _activation_config(activation_quantization)
        text_config = config.get_text_config(decoder=True)
        if getattr(text_config, "is_encoder_decoder", False):
            raise ValueError(
                "QVQ A8 FP8 KV cache currently supports decoder-only models."
            )
        layers = [
            QVQFP8CacheLayer(activation_quantization)
            for _ in range(text_config.num_hidden_layers)
        ]
        super().__init__(layers=layers)
        self.activation_quantization = activation_quantization

    def assert_fp8_storage(self) -> None:
        for layer in self.layers:
            layer.assert_fp8_storage()

    def telemetry(self) -> dict[str, Any]:
        layers = [layer.telemetry() for layer in self.layers]
        initialized = [layer for layer in layers if layer["initialized"]]
        storage_bytes = sum(layer["storage_bytes"] for layer in initialized)
        dense_bytes = sum(layer["dense_equivalent_bytes"] for layer in initialized)
        payload_dtypes = sorted({layer["payload_dtype"] for layer in initialized})
        scale_dtypes = sorted({layer["scale_dtype"] for layer in initialized})
        sequence_lengths = sorted({layer["sequence_length"] for layer in initialized})
        return {
            "schema": "qvq.fp8-kv-cache.v1",
            "format": self.activation_quantization.format,
            "scale_method": self.activation_quantization.scale_method,
            "layers": layers,
            "layer_count": len(layers),
            "initialized_layer_count": len(initialized),
            "sequence_lengths": sequence_lengths,
            "payload_dtypes": payload_dtypes,
            "scale_dtypes": scale_dtypes,
            "storage_bytes": storage_bytes,
            "dense_equivalent_bytes": dense_bytes,
            "storage_ratio_vs_dense": storage_bytes / dense_bytes
            if dense_bytes
            else None,
            "all_payloads_fp8": payload_dtypes
            in ([], [f"torch.{QVQ_FP8_ACTIVATION_FORMAT}"]),
            "no_full_precision_residual": True,
        }


def _argument_value(args, kwargs, names: list[str], name: str, default):
    if name in kwargs:
        return kwargs[name]
    try:
        index = names.index(name)
    except ValueError:
        return default
    return args[index] if index < len(args) else default


def _set_argument(args, kwargs, names: list[str], name: str, value):
    if name in kwargs:
        kwargs[name] = value
        return args, kwargs
    try:
        index = names.index(name)
    except ValueError as error:
        raise RuntimeError(
            f"Model forward has no `{name}` argument required by QVQ A8."
        ) from error
    if index < len(args):
        mutable = list(args)
        mutable[index] = value
        return tuple(mutable), kwargs
    kwargs[name] = value
    return args, kwargs


def install_qvq_fp8_kv_cache(model: torch.nn.Module, activation_quantization) -> None:
    """Require FP8 K/V storage for every cache-enabled forward of an A8 model."""

    activation_quantization = _activation_config(activation_quantization)
    installed = getattr(model, "_qvq_fp8_kv_cache_config", None)
    if installed == activation_quantization:
        return
    if installed is not None:
        raise RuntimeError(
            "QVQ FP8 KV cache was already installed with a different configuration."
        )
    if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
        raise ValueError("QVQ A8 FP8 KV cache currently supports decoder-only models.")

    forward_names = list(inspect.signature(model.forward).parameters)
    if "past_key_values" not in forward_names:
        raise ValueError(
            "QVQ A8 requires a model forward with `past_key_values` support."
        )

    def forward_pre_hook(module, args, kwargs):
        use_cache = _argument_value(
            args,
            kwargs,
            forward_names,
            "use_cache",
            getattr(module.config, "use_cache", True),
        )
        if use_cache is False:
            return args, kwargs
        cache = _argument_value(args, kwargs, forward_names, "past_key_values", None)
        if cache is None:
            cache = QVQFP8DynamicCache(module.config, activation_quantization)
            return _set_argument(args, kwargs, forward_names, "past_key_values", cache)
        if not isinstance(cache, QVQFP8DynamicCache):
            raise TypeError(
                "QVQ A8 requires QVQFP8DynamicCache whenever `use_cache=True`; "
                f"received {type(cache).__name__}."
            )
        cache.assert_fp8_storage()
        return args, kwargs

    model.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)

    if hasattr(model, "_prepare_cache_for_generation"):

        def prepare_cache_for_generation(
            this,
            generation_config,
            model_kwargs,
            generation_mode,
            batch_size,
            max_cache_length,
        ):
            del generation_mode, batch_size, max_cache_length
            existing = model_kwargs.get("past_key_values")
            requested = generation_config.cache_implementation
            if existing is not None:
                if requested is not None:
                    raise ValueError(
                        "Do not combine an explicit QVQ FP8 cache with `cache_implementation`."
                    )
                if not isinstance(existing, QVQFP8DynamicCache):
                    raise ValueError(
                        "QVQ A8 generation requires QVQFP8DynamicCache; "
                        f"received {type(existing).__name__}."
                    )
                return
            if generation_config.use_cache is False:
                return
            if requested is not None:
                raise ValueError(
                    "QVQ A8 fixes `cache_implementation` to its FP8 dynamic cache; "
                    f"received `{requested}`."
                )
            if generation_config.cache_config is not None:
                raise ValueError(
                    "QVQ A8 does not accept a separate Transformers `cache_config`."
                )
            model_kwargs["past_key_values"] = QVQFP8DynamicCache(
                this.config,
                activation_quantization,
            )
            return

        model._prepare_cache_for_generation = types.MethodType(
            prepare_cache_for_generation, model
        )

    model._qvq_fp8_kv_cache_config = activation_quantization


__all__ = [
    "QVQFP8CacheLayer",
    "QVQFP8DynamicCache",
    "install_qvq_fp8_kv_cache",
]
