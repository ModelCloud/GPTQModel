# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Estimate the KV-cache footprint of a model config.

Weight quantization decides how big the checkpoint is; the KV cache decides how
much context fits next to it. The two are independent - 4-bit weights on a 32k
context can still need more VRAM for the cache than for the model - and the
question users ask after quantizing is exactly "what will 8k cost me".

This is arithmetic on the config, not a runtime measurement: it reports what the
declared shapes imply, so a config that hides its shapes returns ``None`` rather
than a number that looks authoritative and is wrong.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch


_DTYPE_ITEMSIZE = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}

# Configs whose attention state is not one K/V pair per KV head per layer.
# MLA keeps a single compressed latent, so the usual per-head formula does not
# describe its cache and returning a number would be misleading.
_UNSUPPORTED_SHAPE_FIELDS = ("kv_lora_rank", "q_lora_rank")


@dataclass(frozen=True)
class KVCacheEstimate:
    """Per-token KV-cache cost implied by a config, for one batch element."""

    layers: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype
    sliding_window: Optional[int] = None

    @property
    def itemsize(self) -> int:
        return _DTYPE_ITEMSIZE[self.dtype]

    @property
    def bytes_per_token(self) -> int:
        """K and V, every layer, one batch element."""

        return self.layers * 2 * self.kv_heads * self.head_dim * self.itemsize

    def bytes_for(self, tokens: int, *, batch_size: int = 1) -> int:
        return self.bytes_per_token * max(0, int(tokens)) * max(1, int(batch_size))

    def tokens_for(self, budget_bytes: int, *, batch_size: int = 1) -> int:
        per_token = self.bytes_per_token * max(1, int(batch_size))
        if per_token <= 0:
            return 0
        return max(0, int(budget_bytes)) // per_token

    def format(self, contexts: Sequence[int] = (8192, 32768)) -> str:
        """One line: per-token cost and what a few context lengths need."""

        parts = [
            f"KV cache: {_human(self.bytes_per_token)}/token"
            f" ({self.layers}L x {self.kv_heads}KV x {self.head_dim}d, {_dtype_name(self.dtype)})"
        ]
        for context in contexts:
            parts.append(f"{_tokens_name(context)} ctx {_human(self.bytes_for(context))}")
        if self.sliding_window:
            parts.append(f"sliding window {_tokens_name(self.sliding_window)} caps growth")
        return " | ".join(parts)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _tokens_name(tokens: int) -> str:
    if tokens % 1024 == 0 and tokens >= 1024:
        return f"{tokens // 1024}k"
    return str(tokens)


def _human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}".replace(".0 ", " ")
        value /= 1024
    return f"{value:.1f} TiB"


def _text_config(config: Any) -> Any:
    """Unwrap the language-model half of a multimodal config."""

    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def estimate_kv_cache(
        config: Any,
        *,
        dtype: Optional[torch.dtype] = None,
) -> Optional[KVCacheEstimate]:
    """Return the KV-cache cost this config implies, or None if it does not say."""

    if config is None:
        return None
    source = _text_config(config)

    for field in _UNSUPPORTED_SHAPE_FIELDS:
        if getattr(source, field, None):
            return None

    layers = getattr(source, "num_hidden_layers", None)
    heads = getattr(source, "num_attention_heads", None)
    hidden = getattr(source, "hidden_size", None)
    if not layers or not heads or not hidden:
        return None

    kv_heads = getattr(source, "num_key_value_heads", None) or heads
    head_dim = getattr(source, "head_dim", None) or (hidden // heads)
    if not kv_heads or not head_dim:
        return None

    resolved_dtype = dtype or getattr(source, "torch_dtype", None) or torch.float16
    if not isinstance(resolved_dtype, torch.dtype) or resolved_dtype not in _DTYPE_ITEMSIZE:
        return None

    window = getattr(source, "sliding_window", None)
    window = int(window) if isinstance(window, int) and window > 0 else None

    return KVCacheEstimate(
        layers=int(layers),
        kv_heads=int(kv_heads),
        head_dim=int(head_dim),
        dtype=resolved_dtype,
        sliding_window=window,
    )
