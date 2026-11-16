# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Estimate the KV-cache footprint of a model config.

Weight quantization decides how big the checkpoint is; the KV cache decides how
much context fits next to it, and the two are independent - 4-bit weights on a
32k context can still need more VRAM for the cache than for the model.

The arithmetic depends on how a model stores attention state, and recent model
families no longer agree on one layout:

``gqa``/``mqa``  every layer keeps one K and one V per KV head, so the cache is
                 ``2 * kv_heads * head_dim`` per token per layer.
``mla``          DeepSeek-style attention keeps a compressed latent
                 (``kv_lora_rank``) plus one decoupled RoPE key
                 (``qk_rope_head_dim``) per token, shared across heads. There is
                 no per-head K/V pair and no factor of two; summing per-head
                 shapes here would overstate the cache by an order of magnitude.
``linear``       recurrent layers (``linear_attention`` and friends) keep a
                 constant-size state. They do not grow with context, so they are
                 reported separately and never counted in the per-token slope.
``sliding``      layers with a window cap growth at ``min(context, window)``.

Mixed stacks (GLM5, DeepSeek V4.1 Flash and similar) declare their layout in
``layer_types``; the estimator resolves per layer and reports what it could not
classify instead of folding it into a total that looks authoritative.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import torch


_DTYPE_ITEMSIZE = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}

# Layer types whose cache grows with the number of tokens.
_FULL_ATTENTION_TYPES = frozenset({
    "full_attention", "attention", "self_attention", "indexed_attention",
    "deepseek_sparse_attention", "sparse_attention", "mla", "multi_latent_attention",
})
_SLIDING_ATTENTION_TYPES = frozenset({
    "sliding_attention", "sliding_window_attention", "window_attention",
})
# Layer types that keep a constant-size recurrent state.
_LINEAR_ATTENTION_TYPES = frozenset({
    "linear_attention", "gated_deltanet", "deltanet", "gla", "recurrent",
    "mamba", "mamba2", "ssm", "lightning_attention",
})

_LAYER_TYPE_FIELDS = ("layer_types", "attention_layer_types", "attn_layer_types")


@dataclass(frozen=True)
class KVCacheEstimate:
    """Per-token KV-cache cost implied by a config, for one batch element."""

    layer_type: str                      # "gqa" | "mla"
    growing_layers: int                  # attention layers whose cache grows
    dims_per_layer_token: int            # elements per token per growing layer
    dtype: torch.dtype
    sliding_layers: int = 0
    sliding_window: Optional[int] = None
    linear_layers: int = 0               # constant-state layers, not counted
    mtp_layers: int = 0                  # next-token-prediction layers, counted
    unclassified: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def itemsize(self) -> int:
        return _DTYPE_ITEMSIZE[self.dtype]

    @property
    def exact(self) -> bool:
        """False when some layer type could not be classified."""

        return not self.unclassified

    @property
    def full_layers(self) -> int:
        return self.growing_layers - self.sliding_layers

    def _bytes_per_layer_token(self) -> int:
        return self.dims_per_layer_token * self.itemsize

    @property
    def bytes_per_token(self) -> int:
        """Asymptotic growth per token: sliding layers do not grow past the window."""

        return self.full_layers * self._bytes_per_layer_token()

    def bytes_for(self, tokens: int, *, batch_size: int = 1) -> int:
        tokens = max(0, int(tokens))
        batch = max(1, int(batch_size))
        window = self.sliding_window or tokens
        sliding_tokens = min(tokens, window)
        growing = self.full_layers * tokens + self.sliding_layers * sliding_tokens
        return growing * self._bytes_per_layer_token() * batch

    def tokens_for(self, budget_bytes: int, *, batch_size: int = 1) -> int:
        """Largest context whose cache fits in ``budget_bytes``."""

        budget = max(0, int(budget_bytes)) // max(1, int(batch_size))
        per_layer = self._bytes_per_layer_token()
        if per_layer <= 0 or budget <= 0:
            return 0

        window = self.sliding_window or 0
        # Up to the window every layer pays; past it only the full layers do.
        first_stage = (self.full_layers + self.sliding_layers) * per_layer
        if window and first_stage * window >= budget:
            return budget // first_stage
        if not window:
            return budget // first_stage if first_stage else 0

        remaining = budget - first_stage * window
        slope = self.full_layers * per_layer
        if slope <= 0:
            return window
        return window + remaining // slope

    def format(self, contexts: Sequence[int] = (8192, 32768)) -> str:
        """One line: per-token cost, a few context sizes, and the assumptions."""

        shape = (
            f"{self.growing_layers}L x {self.dims_per_layer_token} elem/token, {self.layer_type}"
        )
        parts = [f"KV cache: {_human(self.bytes_per_token)}/token ({shape}, {_dtype_name(self.dtype)})"]
        for context in contexts:
            parts.append(f"{_tokens_name(context)} ctx {_human(self.bytes_for(context))}")
        if self.sliding_layers and self.sliding_window:
            parts.append(
                f"{self.sliding_layers} sliding layer(s) capped at {_tokens_name(self.sliding_window)}"
            )
        if self.linear_layers:
            parts.append(f"{self.linear_layers} linear-attention layer(s) hold constant state")
        if self.mtp_layers:
            parts.append(f"incl. {self.mtp_layers} MTP layer(s)")
        if self.unclassified:
            parts.append(f"unknown layer type(s): {', '.join(self.unclassified)}")
        return " | ".join(parts)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _tokens_name(tokens: int) -> str:
    if tokens >= 1024 and tokens % 1024 == 0:
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


def _layer_types(source: Any, layers: int) -> Tuple[Optional[list], Tuple[str, ...]]:
    """Return the declared per-layer types, or a single-type list of ``None``."""

    for field_name in _LAYER_TYPE_FIELDS:
        declared = getattr(source, field_name, None)
        if isinstance(declared, (list, tuple)) and declared:
            declared = [str(entry).lower() for entry in declared]
            if len(declared) > layers:
                return None, ()
            # MTP layers reuse the last declared attention layer's layout.
            mtp = int(getattr(source, "num_nextn_predict_layers", 0) or 0)
            return declared + [declared[-1]] * mtp, ()
    return [None] * layers, ()


def _is_mla(source: Any) -> Tuple[bool, Optional[int]]:
    """MLA caches a compressed latent plus a decoupled RoPE key."""

    latent = getattr(source, "kv_lora_rank", None)
    if not latent:
        return False, None
    rope = getattr(source, "qk_rope_head_dim", 0) or 0
    try:
        return True, int(latent) + int(rope)
    except (TypeError, ValueError):
        return False, None


def estimate_kv_cache(
        config: Any,
        *,
        dtype: Optional[torch.dtype] = None,
) -> Optional[KVCacheEstimate]:
    """Return the KV-cache cost this config implies, or None if it does not say."""

    if config is None:
        return None
    source = _text_config(config)

    layers = getattr(source, "num_hidden_layers", None)
    if not layers:
        return None

    mtp = int(getattr(source, "num_nextn_predict_layers", 0) or 0)
    declared_types, unclassified_declared = _layer_types(source, int(layers))

    is_mla, mla_dims = _is_mla(source)
    if is_mla:
        if not mla_dims:
            return None
        layer_type, dims = "mla", mla_dims
    else:
        heads = getattr(source, "num_attention_heads", None)
        hidden = getattr(source, "hidden_size", None)
        if not heads or not hidden:
            return None
        kv_heads = getattr(source, "num_key_value_heads", None) or heads
        head_dim = getattr(source, "head_dim", None) or (hidden // heads)
        if not kv_heads or not head_dim:
            return None
        layer_type, dims = "gqa", 2 * int(kv_heads) * int(head_dim)

    resolved_dtype = dtype or getattr(source, "torch_dtype", None) or torch.float16
    if not isinstance(resolved_dtype, torch.dtype) or resolved_dtype not in _DTYPE_ITEMSIZE:
        return None

    window = getattr(source, "sliding_window", None)
    window = int(window) if isinstance(window, int) and window > 0 else None

    growing = sliding = linear = 0
    unclassified = list(unclassified_declared)
    for entry in declared_types:
        if entry is None:
            growing += 1
        elif entry in _FULL_ATTENTION_TYPES:
            growing += 1
        elif entry in _SLIDING_ATTENTION_TYPES:
            growing += 1
            sliding += 1
        elif entry in _LINEAR_ATTENTION_TYPES:
            linear += 1
        else:
            if entry not in unclassified:
                unclassified.append(entry)

    if growing == 0 and not unclassified:
        # Every layer is recurrent: there is no per-token cache to report.
        return None

    return KVCacheEstimate(
        layer_type=layer_type,
        growing_layers=growing,
        dims_per_layer_token=dims,
        dtype=resolved_dtype,
        sliding_layers=sliding,
        sliding_window=window if sliding else None,
        linear_layers=linear,
        mtp_layers=mtp,
        unclassified=tuple(unclassified),
    )
