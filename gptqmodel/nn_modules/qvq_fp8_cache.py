# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed FP8 KV-cache storage for QVQ A8 checkpoints."""

from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

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


@dataclass(frozen=True)
class QVQFP8KVView:
    """Opaque FP8 cache view consumed only by the QVQ attention interface."""

    payload: torch.Tensor
    scales: torch.Tensor
    source_dtype: torch.dtype
    layer: "QVQFP8CacheLayer"
    kind: str


def _pad_fp8_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    if tensor.shape[0] == rows:
        return tensor
    return F.pad(tensor, (0, 0, 0, rows - tensor.shape[0]))


def _attention_mask_slice(
    attention_mask: torch.Tensor | None,
    *,
    batch: int,
    head: int,
    query_tokens: int,
    key_tokens: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 4:
        mask_head = 0 if attention_mask.shape[1] == 1 else head
        return attention_mask[batch, mask_head, :query_tokens, :key_tokens]
    if attention_mask.ndim == 3:
        return attention_mask[batch, :query_tokens, :key_tokens]
    if attention_mask.ndim == 2:
        return attention_mask[batch, :key_tokens].unsqueeze(0).expand(query_tokens, -1)
    raise ValueError(
        f"QVQ FP8 attention requires a 2D--4D attention mask, got {attention_mask.ndim}D."
    )


def _causal_mask(query_tokens: int, key_tokens: int, device: torch.device) -> torch.Tensor:
    if query_tokens > key_tokens:
        raise ValueError("QVQ causal attention cannot have more query than key tokens.")
    query_positions = torch.arange(query_tokens, device=device).unsqueeze(1)
    key_positions = torch.arange(key_tokens, device=device).unsqueeze(0)
    cache_offset = key_tokens - query_tokens
    return key_positions > query_positions + cache_offset


def qvq_fp8_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: QVQFP8KVView,
    value: QVQFP8KVView,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Consume per-token-scaled E4M3 K/V without a dense prefix materialization."""

    output_attentions = bool(kwargs.get("output_attentions", False))
    if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
        # `use_cache=False` has no retained prefix to consume. Keep that valid
        # without weakening the cache-enabled A8 contract below.
        groups = module.num_key_value_groups
        dense_key = key.repeat_interleave(groups, dim=1)
        dense_value = value.repeat_interleave(groups, dim=1)
        dense_weights = torch.matmul(query, dense_key.transpose(2, 3)) * scaling
        dense_weights.masked_fill_(
            _causal_mask(query.shape[-2], key.shape[-2], query.device),
            float("-inf"),
        )
        if attention_mask is not None:
            dense_weights = dense_weights + attention_mask
        dense_weights = torch.softmax(dense_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        dense_weights = F.dropout(
            dense_weights, p=dropout, training=module.training
        )
        dense_output = torch.matmul(dense_weights, dense_value)
        return dense_output.transpose(1, 2).contiguous(), dense_weights
    if not isinstance(key, QVQFP8KVView) or not isinstance(value, QVQFP8KVView):
        raise TypeError("QVQ FP8 attention requires opaque FP8 K/V cache views.")
    if key.layer is not value.layer or key.kind != "key" or value.kind != "value":
        raise ValueError("QVQ FP8 attention received mismatched K/V cache views.")
    if query.device.type != "cuda" or torch.cuda.get_device_capability(query.device)[0] < 9:
        raise RuntimeError("QVQ native FP8 attention requires an NVIDIA SM90+ GPU.")
    fp8_dtype = getattr(torch, QVQ_FP8_ACTIVATION_FORMAT)
    if key.payload.dtype != fp8_dtype or value.payload.dtype != fp8_dtype:
        raise TypeError("QVQ native FP8 attention requires E4M3 K/V payloads.")
    if query.ndim != 4 or key.payload.ndim != 4 or value.payload.ndim != 4:
        raise ValueError("QVQ FP8 attention requires [batch, heads, tokens, width] tensors.")
    if key.payload.shape != value.payload.shape:
        raise ValueError("QVQ FP8 attention requires matching K/V payload geometry.")
    if query.shape[0] != key.payload.shape[0] or query.shape[-1] != key.payload.shape[-1]:
        raise ValueError("QVQ FP8 attention query and cache geometry differ.")

    batch_size, query_heads, query_tokens, _ = query.shape
    _, kv_heads, key_tokens, _ = key.payload.shape
    if query_heads % kv_heads:
        raise ValueError("QVQ FP8 attention requires query heads divisible by KV heads.")
    groups = query_heads // kv_heads
    padded_keys = ((key_tokens + 15) // 16) * 16
    causal_mask = _causal_mask(query_tokens, key_tokens, query.device)
    one = torch.ones((), dtype=torch.float32, device=query.device)
    outputs: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []

    for batch in range(batch_size):
        batch_outputs: list[torch.Tensor] = []
        batch_weights: list[torch.Tensor] = []
        for kv_head in range(kv_heads):
            first_head = kv_head * groups
            query_group = query[batch, first_head : first_head + groups]
            query_matrix = query_group.reshape(groups * query_tokens, -1)
            query_fp8, query_scale = quantize_qvq_fp8_activation(
                query_matrix,
                format=QVQ_FP8_ACTIVATION_FORMAT,
                scale_method="dynamic_per_token",
                validate=False,
            )
            key_fp8 = key.payload[batch, kv_head]
            key_scale = key.scales[batch, kv_head]
            if padded_keys != key_tokens:
                key_fp8 = _pad_fp8_rows(key_fp8, padded_keys)
            raw_logits = torch._scaled_mm(
                query_fp8,
                key_fp8.T,
                one,
                one,
                out_dtype=torch.float32,
                use_fast_accum=False,
            )[:, :key_tokens]
            logits = raw_logits * query_scale.float() * key_scale.T.float()
            logits.mul_(scaling)
            logits.reshape(groups, query_tokens, key_tokens).masked_fill_(
                causal_mask, float("-inf")
            )
            if attention_mask is not None:
                group_masks = [
                    _attention_mask_slice(
                        attention_mask,
                        batch=batch,
                        head=head,
                        query_tokens=query_tokens,
                        key_tokens=key_tokens,
                    )
                    for head in range(first_head, first_head + groups)
                ]
                logits.add_(torch.stack(group_masks).reshape_as(logits).float())
            probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
            probabilities = F.dropout(
                probabilities, p=dropout, training=module.training
            )
            if output_attentions:
                batch_weights.append(
                    probabilities.reshape(groups, query_tokens, key_tokens).to(
                        query.dtype
                    )
                )

            # Absorb each V row's dynamic scale into the probability column.
            # The second native E4M3 GEMM can then consume the cached V payload
            # directly without ever constructing a BF16/FP16 V prefix.
            scaled_probabilities = probabilities * value.scales[
                batch, kv_head, :, 0
            ].float().unsqueeze(0)
            probability_fp8, probability_scale = quantize_qvq_fp8_activation(
                scaled_probabilities,
                format=QVQ_FP8_ACTIVATION_FORMAT,
                scale_method="dynamic_per_token",
                validate=False,
            )
            value_fp8 = value.payload[batch, kv_head]
            if padded_keys != key_tokens:
                probability_fp8 = F.pad(
                    probability_fp8, (0, padded_keys - key_tokens)
                )
                value_fp8 = _pad_fp8_rows(value_fp8, padded_keys)
            # cuBLASLt FP8 accepts row-major A and column-major B. This
            # transpose-copy-transpose remains FP8 and is not a dense K/V copy.
            value_column_major = value_fp8.T.contiguous().T
            raw_output = torch._scaled_mm(
                probability_fp8,
                value_column_major,
                one,
                one,
                out_dtype=torch.float32,
                use_fast_accum=False,
            )
            batch_outputs.append(
                (raw_output * probability_scale.float())
                .reshape(groups, query_tokens, -1)
                .to(query.dtype)
            )
        outputs.append(torch.cat(batch_outputs, dim=0))
        if output_attentions:
            weights.append(torch.cat(batch_weights, dim=0))

    key.layer.native_attention_calls += 1
    key.layer.native_qk_fp8_mm_calls += batch_size * kv_heads
    key.layer.native_pv_fp8_mm_calls += batch_size * kv_heads
    key.layer.native_attention_query_tokens += batch_size * query_heads * query_tokens
    output = torch.stack(outputs, dim=0).transpose(1, 2).contiguous()
    return output, torch.stack(weights, dim=0) if output_attentions else None


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
        self.native_attention_calls = 0
        self.native_qk_fp8_mm_calls = 0
        self.native_pv_fp8_mm_calls = 0
        self.native_attention_query_tokens = 0

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

        if key_states.device.type == "cuda":
            return (
                QVQFP8KVView(self.keys, self.key_scales, self.source_dtype, self, "key"),
                QVQFP8KVView(
                    self.values, self.value_scales, self.source_dtype, self, "value"
                ),
            )

        # CPU remains a portable reference path for cache unit tests. CUDA A8
        # is fail-closed on the registered native FP8 attention consumer.
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
            "native_attention_calls": self.native_attention_calls,
            "native_qk_fp8_mm_calls": self.native_qk_fp8_mm_calls,
            "native_pv_fp8_mm_calls": self.native_pv_fp8_mm_calls,
            "native_attention_query_tokens": self.native_attention_query_tokens,
            "native_attention_backend": "torch._scaled_mm_cublaslt_fp8",
            "dense_kv_prefix_materializations": 0
            if self.keys.device.type == "cuda"
            else self.update_calls,
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
        native_attention_calls = sum(
            layer["native_attention_calls"] for layer in initialized
        )
        native_qk_fp8_mm_calls = sum(
            layer["native_qk_fp8_mm_calls"] for layer in initialized
        )
        native_pv_fp8_mm_calls = sum(
            layer["native_pv_fp8_mm_calls"] for layer in initialized
        )
        dequantized_elements = sum(
            layer["dequantized_elements"] for layer in initialized
        )
        dense_kv_prefix_materializations = sum(
            layer["dense_kv_prefix_materializations"] for layer in initialized
        )
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
            "native_attention_backend": "torch._scaled_mm_cublaslt_fp8",
            "native_attention_calls": native_attention_calls,
            "native_qk_fp8_mm_calls": native_qk_fp8_mm_calls,
            "native_pv_fp8_mm_calls": native_pv_fp8_mm_calls,
            "dequantized_elements": dequantized_elements,
            "dense_kv_prefix_materializations": dense_kv_prefix_materializations,
            "native_fp8_attention": bool(initialized)
            and native_attention_calls >= len(initialized)
            and native_qk_fp8_mm_calls > 0
            and native_pv_fp8_mm_calls > 0
            and dequantized_elements == 0
            and dense_kv_prefix_materializations == 0,
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

    ALL_ATTENTION_FUNCTIONS.register("qvq_fp8", qvq_fp8_attention_forward)
    text_config = model.config.get_text_config(decoder=True)
    text_config._attn_implementation = "qvq_fp8"

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
    "QVQFP8KVView",
    "install_qvq_fp8_kv_cache",
    "qvq_fp8_attention_forward",
]
