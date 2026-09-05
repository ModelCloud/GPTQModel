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
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..quantization.config import QVQActivationConfig
from ..quantization.qvq_activation import (
    QVQ_FP8_ACTIVATION_FORMAT,
    dequantize_qvq_fp8_activation,
    quantize_qvq_fp8_activation,
)

_QVQ_FP8_PREFILL_QUERY_CHUNK = 2048


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
    layer: QVQFP8CacheLayer
    kind: str
    sequence_length: int


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
        keep = attention_mask[batch, :key_tokens].to(torch.bool)
        return torch.where(
            keep,
            torch.zeros((), device=keep.device, dtype=torch.float32),
            torch.full((), float("-inf"), device=keep.device, dtype=torch.float32),
        ).unsqueeze(0).expand(query_tokens, -1)
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


def _normalized_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    query_heads: int,
    kv_heads: int,
    groups: int,
    query_tokens: int,
    key_tokens: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.shape[0] != batch_size:
        raise ValueError("QVQ FP8 attention mask batch dimension differs from query.")
    if attention_mask.ndim == 2:
        keep = attention_mask[:, :key_tokens].to(torch.bool)
        additive = torch.where(
            keep,
            torch.zeros((), device=keep.device, dtype=torch.float32),
            torch.full((), float("-inf"), device=keep.device, dtype=torch.float32),
        )
        return additive[:, None, None, None, :]
    if attention_mask.ndim == 3:
        return attention_mask[:, None, None, :query_tokens, :key_tokens]
    if attention_mask.ndim != 4:
        raise ValueError(
            f"QVQ FP8 attention requires a 2D--4D attention mask, got {attention_mask.ndim}D."
        )
    sliced = attention_mask[..., :query_tokens, :key_tokens]
    if sliced.shape[1] == 1:
        return sliced[:, :, None]
    if sliced.shape[1] != query_heads:
        raise ValueError(
            "QVQ FP8 attention mask head dimension must be one or match query heads."
        )
    return sliced.reshape(batch_size, kv_heads, groups, query_tokens, key_tokens)


def _qvq_fp8_grouped_attention(
    query: torch.Tensor,
    key: QVQFP8KVView,
    value: QVQFP8KVView,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    dropout: float,
    training: bool,
    output_attentions: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use one grouped cuBLASLt launch each for all decode QK and PV GEMMs."""

    batch_size, query_heads, query_tokens, width = query.shape
    _, kv_heads, _, _ = key.payload.shape
    groups = query_heads // kv_heads
    gemm_groups = batch_size * kv_heads
    matrix_rows = groups * query_tokens
    key_tokens = key.sequence_length
    padded_keys = ((key_tokens + 15) // 16) * 16

    query_matrix = query.reshape(
        batch_size, kv_heads, groups, query_tokens, width
    ).reshape(gemm_groups, matrix_rows, width)
    query_fp8, query_scale = quantize_qvq_fp8_activation(
        query_matrix,
        format=QVQ_FP8_ACTIVATION_FORMAT,
        scale_method="dynamic_per_token",
        validate=False,
    )
    key_matrix = key.payload[..., :padded_keys, :].transpose(-1, -2).reshape(
        gemm_groups, width, padded_keys
    )
    key_scale = key.scales[..., :key_tokens, 0].reshape(gemm_groups, key_tokens)
    if padded_keys != key_tokens:
        key_scale = F.pad(key_scale, (0, padded_keys - key_tokens))
    raw_logits = torch._scaled_grouped_mm(
        query_fp8,
        key_matrix,
        query_scale.squeeze(-1),
        key_scale,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )[..., :key_tokens]
    logits = raw_logits.float()
    logits.mul_(scaling)
    logits = logits.reshape(
        batch_size, kv_heads, groups, query_tokens, key_tokens
    )
    logits.masked_fill_(
        _causal_mask(query_tokens, key_tokens, query.device)[None, None, None],
        float("-inf"),
    )
    normalized_mask = _normalized_attention_mask(
        attention_mask,
        batch_size=batch_size,
        query_heads=query_heads,
        kv_heads=kv_heads,
        groups=groups,
        query_tokens=query_tokens,
        key_tokens=key_tokens,
    )
    if normalized_mask is not None:
        logits.add_(normalized_mask.float())
    # Keep the attention-score workspace single-buffered.  ``logits`` is
    # already FP32 and is dead after softmax, so an out= softmax avoids a
    # second [B, H, M, K] allocation during prefill.
    probabilities = torch.softmax(
        logits, dim=-1, dtype=torch.float32, out=logits
    )
    probabilities = F.dropout(probabilities, p=dropout, training=training)
    weights = (
        probabilities.reshape(batch_size, query_heads, query_tokens, key_tokens).to(
            query.dtype
        )
        if output_attentions
        else None
    )

    probability_matrix = probabilities.reshape(gemm_groups, matrix_rows, key_tokens)
    value_scale = value.scales[..., :key_tokens, 0].reshape(
        gemm_groups, 1, key_tokens
    )
    scaled_probabilities = probability_matrix
    scaled_probabilities.mul_(value_scale.float())
    probability_fp8, probability_scale = quantize_qvq_fp8_activation(
        scaled_probabilities,
        format=QVQ_FP8_ACTIVATION_FORMAT,
        scale_method="dynamic_per_token",
        validate=False,
    )
    if padded_keys != key_tokens:
        probability_fp8 = F.pad(probability_fp8, (0, padded_keys - key_tokens))
    value_matrix = value.payload[..., :padded_keys, :].reshape(
        gemm_groups, padded_keys, width
    )
    one_value = torch.ones(
        (gemm_groups, width), dtype=torch.float32, device=query.device
    )
    raw_output = torch._scaled_grouped_mm(
        probability_fp8,
        value_matrix,
        probability_scale.squeeze(-1),
        one_value,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )
    output = raw_output.float().reshape(
        batch_size, query_heads, query_tokens, width
    )

    key.layer.native_attention_calls += 1
    key.layer.native_grouped_attention_calls += 1
    key.layer.native_qk_fp8_mm_calls += gemm_groups
    key.layer.native_pv_fp8_mm_calls += gemm_groups
    key.layer.native_qk_fp8_launches += 1
    key.layer.native_pv_fp8_launches += 1
    key.layer.native_attention_query_tokens += batch_size * query_heads * query_tokens
    return output.transpose(1, 2).to(query.dtype).contiguous(), weights


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
            normalized_mask = _normalized_attention_mask(
                attention_mask,
                batch_size=query.shape[0],
                query_heads=query.shape[1],
                kv_heads=key.shape[1],
                groups=groups,
                query_tokens=query.shape[-2],
                key_tokens=key.shape[-2],
            )
            dense_weights = dense_weights + normalized_mask.reshape(
                query.shape[0], query.shape[1], query.shape[-2], key.shape[-2]
            )
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
    _, kv_heads, storage_capacity, _ = key.payload.shape
    key_tokens = key.sequence_length
    if value.sequence_length != key_tokens or key_tokens > storage_capacity:
        raise ValueError("QVQ FP8 attention received invalid K/V logical lengths.")
    if query_heads % kv_heads:
        raise ValueError("QVQ FP8 attention requires query heads divisible by KV heads.")
    groups = query_heads // kv_heads
    if query_tokens <= 16:
        return _qvq_fp8_grouped_attention(
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            training=module.training,
            output_attentions=output_attentions,
        )
    padded_keys = ((key_tokens + 15) // 16) * 16
    causal_mask = _causal_mask(query_tokens, key_tokens, query.device)
    one = torch.ones((), dtype=torch.float32, device=query.device)
    query_chunks = (
        query_tokens + _QVQ_FP8_PREFILL_QUERY_CHUNK - 1
    ) // _QVQ_FP8_PREFILL_QUERY_CHUNK
    outputs: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []

    for batch in range(batch_size):
        batch_outputs: list[torch.Tensor] = []
        batch_weights: list[torch.Tensor] = []
        for kv_head in range(kv_heads):
            first_head = kv_head * groups
            query_group = query[batch, first_head : first_head + groups]
            key_fp8 = key.payload[batch, kv_head, :padded_keys]
            key_scale = key.scales[batch, kv_head, :key_tokens]
            value_fp8 = value.payload[batch, kv_head, :padded_keys]
            chunk_outputs: list[torch.Tensor] = []
            chunk_weights: list[torch.Tensor] = []
            for query_start in range(0, query_tokens, _QVQ_FP8_PREFILL_QUERY_CHUNK):
                query_end = min(
                    query_start + _QVQ_FP8_PREFILL_QUERY_CHUNK, query_tokens
                )
                chunk_tokens = query_end - query_start
                query_matrix = query_group[:, query_start:query_end].reshape(
                    groups * chunk_tokens, -1
                )
                query_fp8, query_scale = quantize_qvq_fp8_activation(
                    query_matrix,
                    format=QVQ_FP8_ACTIVATION_FORMAT,
                    scale_method="dynamic_per_token",
                    validate=False,
                )
                raw_logits = torch._scaled_mm(
                    query_fp8,
                    key_fp8.T,
                    one,
                    one,
                    out_dtype=torch.float32,
                    use_fast_accum=False,
                )[:, :key_tokens]
                logits = raw_logits
                logits.mul_(query_scale.float())
                logits.mul_(key_scale.T.float())
                logits.mul_(scaling)
                logits = logits.reshape(groups, chunk_tokens, key_tokens)
                logits.masked_fill_(
                    causal_mask[query_start:query_end], float("-inf")
                )
                if attention_mask is not None:
                    group_masks = [
                        _attention_mask_slice(
                            attention_mask,
                            batch=batch,
                            head=head,
                            query_tokens=query_tokens,
                            key_tokens=key_tokens,
                        )[query_start:query_end]
                        for head in range(first_head, first_head + groups)
                    ]
                    logits.add_(torch.stack(group_masks).float())
                # Reuse the FP32 score allocation for softmax.  This removes
                # one full query-chunk score buffer from peak prefill memory.
                probabilities = torch.softmax(
                    logits, dim=-1, dtype=torch.float32, out=logits
                )
                probabilities = F.dropout(
                    probabilities, p=dropout, training=module.training
                )
                if output_attentions:
                    chunk_weights.append(probabilities.to(query.dtype))

                # Absorb each V row's dynamic scale into the probability
                # column. Q chunking bounds all FP32 score temporaries while
                # preserving native E4M3 QK and PV GEMMs and opaque FP8 K/V.
                probability_matrix = probabilities.reshape(
                    groups * chunk_tokens, key_tokens
                )
                scaled_probabilities = probability_matrix
                scaled_probabilities.mul_(
                    value.scales[batch, kv_head, :key_tokens, 0]
                    .float()
                    .unsqueeze(0)
                )
                probability_fp8, probability_scale = quantize_qvq_fp8_activation(
                    scaled_probabilities,
                    format=QVQ_FP8_ACTIVATION_FORMAT,
                    scale_method="dynamic_per_token",
                    validate=False,
                )
                if padded_keys != key_tokens:
                    probability_fp8 = F.pad(
                        probability_fp8, (0, padded_keys - key_tokens)
                    )
                raw_output = torch._scaled_mm(
                    probability_fp8,
                    value_fp8,
                    one,
                    one,
                    out_dtype=torch.float32,
                    use_fast_accum=False,
                )
                chunk_outputs.append(
                    (raw_output * probability_scale.float())
                    .reshape(groups, chunk_tokens, -1)
                    .to(query.dtype)
                )
            batch_outputs.append(
                torch.cat(chunk_outputs, dim=1)
            )
            if output_attentions:
                batch_weights.append(torch.cat(chunk_weights, dim=1))
        outputs.append(torch.cat(batch_outputs, dim=0))
        if output_attentions:
            weights.append(torch.cat(batch_weights, dim=0))

    key.layer.native_attention_calls += 1
    native_prefill_gemms = batch_size * kv_heads * query_chunks
    key.layer.native_qk_fp8_mm_calls += native_prefill_gemms
    key.layer.native_pv_fp8_mm_calls += native_prefill_gemms
    key.layer.native_qk_fp8_launches += native_prefill_gemms
    key.layer.native_pv_fp8_launches += native_prefill_gemms
    key.layer.native_attention_query_tokens += batch_size * query_heads * query_tokens
    output = torch.stack(outputs, dim=0).transpose(1, 2).contiguous()
    return output, torch.stack(weights, dim=0) if output_attentions else None


class QVQFP8CacheLayer(DynamicLayer):
    """Page-allocated FP8 cache with row-major K and column-major V storage."""

    def __init__(
        self,
        activation: QVQActivationConfig | dict[str, Any],
        *,
        max_cache_length: int | None = None,
        page_size: int = 256,
    ):
        super().__init__()
        self.activation = _activation_config(activation)
        if max_cache_length is not None:
            max_cache_length = int(max_cache_length)
        if max_cache_length is not None and max_cache_length < 1:
            raise ValueError("QVQ FP8 max_cache_length must be positive when set.")
        if page_size < 16 or page_size % 16:
            raise ValueError("QVQ FP8 cache page_size must be a positive multiple of 16.")
        self.max_cache_length = max_cache_length
        self.page_size = page_size
        self.capacity = 0
        self.sequence_length = 0
        self.allocations = 0
        self.reallocations = 0
        self.reallocated_elements_copied = 0
        self.key_scales: torch.Tensor | None = None
        self.value_scales: torch.Tensor | None = None
        self.source_dtype: torch.dtype | None = None
        self.update_calls = 0
        self.quantized_elements = 0
        self.dequantized_elements = 0
        self.native_attention_calls = 0
        self.native_grouped_attention_calls = 0
        self.native_qk_fp8_mm_calls = 0
        self.native_pv_fp8_mm_calls = 0
        self.native_qk_fp8_launches = 0
        self.native_pv_fp8_launches = 0
        self.native_attention_query_tokens = 0

    def _rounded_capacity(self, required: int) -> int:
        # A dynamic cache's first prefill already reveals its useful length;
        # round only to the GEMM alignment there. Subsequent growth uses full
        # pages. Static generation still reserves its known maximum once.
        page_size = (
            1
            if self.device.type == "cpu"
            else 16
            if self.capacity == 0 and self.max_cache_length is None
            else self.page_size
        )
        capacity = ((required + page_size - 1) // page_size) * page_size
        if self.max_cache_length is not None:
            if required > self.max_cache_length:
                raise ValueError(
                    f"QVQ FP8 cache length {required} exceeds configured maximum "
                    f"{self.max_cache_length}."
                )
            configured = ((self.max_cache_length + page_size - 1) // page_size) * page_size
            capacity = max(capacity, configured)
        return capacity

    def _allocate(self, capacity: int, key_states: torch.Tensor) -> None:
        fp8_dtype = getattr(torch, self.activation.format)
        prefix = key_states.shape[:-2]
        width = key_states.shape[-1]
        new_keys = torch.zeros(
            (*prefix, capacity, width), dtype=fp8_dtype, device=self.device
        )
        # Storing V through a transposed contiguous backing makes every
        # [tokens, width] head view column-major for cuBLASLt FP8 PV GEMMs.
        new_values = torch.zeros(
            (*prefix, width, capacity), dtype=fp8_dtype, device=self.device
        ).transpose(-1, -2)
        scale_shape = (*prefix, capacity, 1)
        new_key_scales = torch.zeros(
            scale_shape, dtype=torch.float32, device=self.device
        )
        new_value_scales = torch.zeros(
            scale_shape, dtype=torch.float32, device=self.device
        )
        if self.sequence_length:
            length = self.sequence_length
            new_keys[..., :length, :].copy_(self.keys[..., :length, :])
            new_values[..., :length, :].copy_(self.values[..., :length, :])
            new_key_scales[..., :length, :].copy_(
                self.key_scales[..., :length, :]
            )
            new_value_scales[..., :length, :].copy_(
                self.value_scales[..., :length, :]
            )
            self.reallocated_elements_copied += (
                2 * self.keys[..., :length, :].numel()
                + self.key_scales[..., :length, :].numel()
                + self.value_scales[..., :length, :].numel()
            )
        was_allocated = self.capacity > 0
        self.keys = new_keys
        self.values = new_values
        self.key_scales = new_key_scales
        self.value_scales = new_value_scales
        self.capacity = capacity
        self.allocations += 1
        self.reallocations += int(was_allocated)

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
        self._allocate(self._rounded_capacity(key_states.shape[-2]), key_states)
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
            format=self.activation.format,
            scale_method=self.activation.scale_method,
            validate=validate,
        )
        quantized_values, value_scales = quantize_qvq_fp8_activation(
            value_states,
            format=self.activation.format,
            scale_method=self.activation.scale_method,
            validate=validate,
        )
        old_length = self.sequence_length
        new_length = old_length + key_states.shape[-2]
        if new_length > self.capacity:
            self._allocate(self._rounded_capacity(new_length), key_states)
        target = slice(old_length, new_length)
        self.keys[..., target, :].copy_(quantized_keys)
        self.values[..., target, :].copy_(quantized_values)
        self.key_scales[..., target, :].copy_(key_scales)
        self.value_scales[..., target, :].copy_(value_scales)
        self.sequence_length = new_length
        self.update_calls += 1
        self.quantized_elements += key_states.numel() + value_states.numel()
        self.assert_fp8_storage()

        if key_states.device.type == "cuda":
            return (
                QVQFP8KVView(
                    self.keys,
                    self.key_scales,
                    self.source_dtype,
                    self,
                    "key",
                    self.sequence_length,
                ),
                QVQFP8KVView(
                    self.values,
                    self.value_scales,
                    self.source_dtype,
                    self,
                    "value",
                    self.sequence_length,
                ),
            )

        # CPU remains a portable reference path for cache unit tests. CUDA A8
        # is fail-closed on the registered native FP8 attention consumer.
        keys = dequantize_qvq_fp8_activation(
            self.keys[..., :new_length, :].contiguous(),
            self.key_scales[..., :new_length, :].contiguous(),
            dtype=self.source_dtype,
        )
        values = dequantize_qvq_fp8_activation(
            self.values[..., :new_length, :].contiguous(),
            self.value_scales[..., :new_length, :].contiguous(),
            dtype=self.source_dtype,
        )
        self.dequantized_elements += keys.numel() + values.numel()
        return keys, values

    def assert_fp8_storage(self) -> None:
        if not self.is_initialized:
            return
        fp8_dtype = getattr(torch, self.activation.format)
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
        if self.keys.stride(-1) != 1:
            raise RuntimeError("QVQ A8 K cache must remain row-major.")
        if self.values.stride(-2) != 1:
            raise RuntimeError("QVQ A8 V cache must remain column-major in token/width.")

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        self.sequence_length = max(0, max_length)

    def reset(self) -> None:
        if not self.is_initialized:
            return
        length = self.sequence_length
        self.keys[..., :length, :].zero_()
        self.values[..., :length, :].zero_()
        self.key_scales[..., :length, :].zero_()
        self.value_scales[..., :length, :].zero_()
        self.sequence_length = 0

    def get_seq_length(self) -> int:
        return self.sequence_length if self.is_initialized else 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_length if self.max_cache_length is not None else -1

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() < 1:
            return
        index = beam_idx.to(self.device)
        self.keys = self.keys.index_select(0, index)
        self.values = self.values.index_select(0, index)
        self.values = self.values.transpose(-1, -2).contiguous().transpose(-1, -2)
        self.key_scales = self.key_scales.index_select(0, index)
        self.value_scales = self.value_scales.index_select(0, index)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() < 1:
            return
        self.keys = self.keys.repeat_interleave(repeats, dim=0)
        self.values = self.values.repeat_interleave(repeats, dim=0)
        self.values = self.values.transpose(-1, -2).contiguous().transpose(-1, -2)
        self.key_scales = self.key_scales.repeat_interleave(repeats, dim=0)
        self.value_scales = self.value_scales.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() < 1:
            return
        self.keys = self.keys[indices, ...]
        self.values = self.values[indices, ...]
        self.values = self.values.transpose(-1, -2).contiguous().transpose(-1, -2)
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
        logical_keys = self.keys[..., : self.sequence_length, :]
        logical_values = self.values[..., : self.sequence_length, :]
        payload_bytes = _tensor_bytes(self.keys) + _tensor_bytes(self.values)
        scale_bytes = _tensor_bytes(self.key_scales) + _tensor_bytes(self.value_scales)
        dense_bytes = (logical_keys.numel() + logical_values.numel()) * torch.empty(
            (), dtype=self.source_dtype
        ).element_size()
        return {
            "initialized": True,
            "sequence_length": self.get_seq_length(),
            "key_shape": list(logical_keys.shape),
            "value_shape": list(logical_values.shape),
            "allocated_key_shape": list(self.keys.shape),
            "allocated_value_shape": list(self.values.shape),
            "capacity": self.capacity,
            "page_size": self.page_size if self.device.type == "cuda" else 1,
            "allocation_strategy": "static"
            if self.max_cache_length is not None
            else "paged_dynamic",
            "allocations": self.allocations,
            "reallocations": self.reallocations,
            "reallocated_elements_copied": self.reallocated_elements_copied,
            "key_layout": "row_major",
            "value_layout": "column_major_tokens_width",
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
            "native_grouped_attention_calls": self.native_grouped_attention_calls,
            "native_qk_fp8_mm_calls": self.native_qk_fp8_mm_calls,
            "native_pv_fp8_mm_calls": self.native_pv_fp8_mm_calls,
            "native_qk_fp8_launches": self.native_qk_fp8_launches,
            "native_pv_fp8_launches": self.native_pv_fp8_launches,
            "native_attention_query_tokens": self.native_attention_query_tokens,
            "native_attention_backend": "torch._scaled_grouped_mm_and_scaled_mm_cublaslt_fp8",
            "dense_kv_prefix_materializations": 0
            if self.keys.device.type == "cuda"
            else self.update_calls,
        }


class QVQFP8DynamicCache(Cache):
    """Full-layer page-allocated FP8 cache automatically required by A8 models."""

    def __init__(
        self,
        config,
        activation: QVQActivationConfig | dict[str, Any],
        *,
        max_cache_length: int | None = None,
        page_size: int = 256,
    ):
        activation = _activation_config(activation)
        text_config = config.get_text_config(decoder=True)
        if getattr(text_config, "is_encoder_decoder", False):
            raise ValueError(
                "QVQ A8 FP8 KV cache currently supports decoder-only models."
            )
        layers = [
            QVQFP8CacheLayer(
                activation,
                max_cache_length=max_cache_length,
                page_size=page_size,
            )
            for _ in range(text_config.num_hidden_layers)
        ]
        super().__init__(layers=layers)
        self.activation = activation
        self.max_cache_length = max_cache_length
        self.page_size = page_size

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
        native_grouped_attention_calls = sum(
            layer["native_grouped_attention_calls"] for layer in initialized
        )
        native_qk_fp8_launches = sum(
            layer["native_qk_fp8_launches"] for layer in initialized
        )
        native_pv_fp8_launches = sum(
            layer["native_pv_fp8_launches"] for layer in initialized
        )
        dequantized_elements = sum(
            layer["dequantized_elements"] for layer in initialized
        )
        dense_kv_prefix_materializations = sum(
            layer["dense_kv_prefix_materializations"] for layer in initialized
        )
        capacities = sorted({layer["capacity"] for layer in initialized})
        allocations = sum(layer["allocations"] for layer in initialized)
        reallocations = sum(layer["reallocations"] for layer in initialized)
        reallocated_elements_copied = sum(
            layer["reallocated_elements_copied"] for layer in initialized
        )
        return {
            "schema": "qvq.fp8-kv-cache.v1",
            "format": self.activation.format,
            "scale_method": self.activation.scale_method,
            "layers": layers,
            "layer_count": len(layers),
            "initialized_layer_count": len(initialized),
            "sequence_lengths": sequence_lengths,
            "capacities": capacities,
            "page_size": self.page_size,
            "allocation_strategy": "static"
            if self.max_cache_length is not None
            else "paged_dynamic",
            "allocations": allocations,
            "reallocations": reallocations,
            "reallocated_elements_copied": reallocated_elements_copied,
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
            "native_attention_backend": "torch._scaled_grouped_mm_and_scaled_mm_cublaslt_fp8",
            "native_attention_calls": native_attention_calls,
            "native_grouped_attention_calls": native_grouped_attention_calls,
            "native_qk_fp8_mm_calls": native_qk_fp8_mm_calls,
            "native_pv_fp8_mm_calls": native_pv_fp8_mm_calls,
            "native_qk_fp8_launches": native_qk_fp8_launches,
            "native_pv_fp8_launches": native_pv_fp8_launches,
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


def install_qvq_fp8_kv_cache(model: torch.nn.Module, activation) -> None:
    """Require FP8 K/V storage for every cache-enabled forward of an A8 model."""

    activation = _activation_config(activation)
    installed = getattr(model, "_qvq_fp8_kv_cache_config", None)
    if installed == activation:
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
    # Without a registered builder Transformers discards the model's 2D mask
    # before calling a custom attention backend. Eager construction preserves
    # padding, packed-sequence, causal, and sliding-window semantics in one
    # additive 4D mask.
    ALL_MASK_ATTENTION_FUNCTIONS.register("qvq_fp8", eager_mask)
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
            cache = QVQFP8DynamicCache(module.config, activation)
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
            del generation_mode, batch_size
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
                activation,
                max_cache_length=max_cache_length,
            )
            return

        model._prepare_cache_for_generation = types.MethodType(
            prepare_cache_for_generation, model
        )

    model._qvq_fp8_kv_cache_config = activation


__all__ = [
    "QVQFP8CacheLayer",
    "QVQFP8DynamicCache",
    "QVQFP8KVView",
    "install_qvq_fp8_kv_cache",
    "qvq_fp8_attention_forward",
]
