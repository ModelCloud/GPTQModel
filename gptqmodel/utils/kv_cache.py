# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Estimate the KV-cache footprint of a model config.

Weight quantization decides how big the checkpoint is; the KV cache decides how
much context fits next to it, and the two are independent - 4-bit weights on a
32k context can still need more VRAM for the cache than for the model.

One formula does not cover the current model landscape, so the estimate is built
per layer:

``gqa``/``mqa``  one K and one V per KV head: ``2 * kv_heads * head_dim``.
``mla``          a compressed latent (``kv_lora_rank``) plus one decoupled RoPE
                 key (``qk_rope_head_dim``), shared across heads, with no factor
                 of two. Reading per-head shapes here overstates by ~60x.
``linear``       recurrent layers keep a constant-size state. They do not grow
                 with context and are reported separately, never in the slope.
``sliding``      growth capped at ``min(context, window)``; chunked attention
                 (Llama 4) uses ``attention_chunk_size`` as its window.
``shared``       Gemma 3n/4 tail layers reuse an earlier layer's K/V and add
                 nothing; they are reported but never counted.

Stacks that mix those (GLM5, DeepSeek V4.1 Flash, Gemma4 and similar) declare the
mix either through ``layer_types`` or through a heterogeneous
``per_layer_config`` where per-layer attributes refuse global reads. Both are
resolved layer by layer; anything unclassifiable is named instead of being folded
into a total that looks authoritative.
"""

from collections.abc import Sequence as _SequenceABC
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import torch


_DTYPE_ITEMSIZE = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}

# Serving engines store the KV cache in fp8 to halve it. One byte per element is
# the whole cache except for the per-head scale, which is negligible here.
for _fp8_name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    _fp8_dtype = getattr(torch, _fp8_name, None)
    if isinstance(_fp8_dtype, torch.dtype):
        _DTYPE_ITEMSIZE[_fp8_dtype] = 1

_FULL_ATTENTION_TYPES = frozenset({
    "full_attention", "attention", "self_attention", "indexed_attention",
    "deepseek_sparse_attention", "sparse_attention", "mla", "multi_latent_attention",
    # A layer that mixes a recurrent branch with an attention branch still keeps
    # standard K/V for the attention half; the recurrent half is constant state.
    "hybrid",
})
_SLIDING_ATTENTION_TYPES = frozenset({
    "sliding_attention", "sliding_window_attention", "window_attention",
    "hybrid_sliding",
})
# Llama4's "chunked attention" is not full attention: transformers builds a
# sliding-window cache for it with ``sliding_window = attention_chunk_size``.
_CHUNKED_ATTENTION_TYPES = frozenset({"chunked_attention"})
_COMPRESSED_ATTENTION_TYPES = frozenset({
    "heavily_compressed_attention", "compressed_sparse_attention", "compressed_attention",
})
# Block types that carry no attention at all: Nemotron-H dispatches these to
# ``NemotronHMoE`` / ``NemotronHMLP`` mixers instead of an attention module, so
# they contribute nothing to the cache.
_NO_ATTENTION_TYPES = frozenset({"moe", "mlp"})
_LINEAR_ATTENTION_TYPES = frozenset({
    "linear_attention", "gated_deltanet", "deltanet", "gla", "recurrent",
    "mamba", "mamba2", "ssm", "lightning_attention",
})

_LAYER_TYPE_FIELDS = ("layer_types", "attention_layer_types", "attn_layer_types")


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a config attribute without letting heterogeneity guards escape.

    Transformers raises ``AmbiguousGlobalPerLayerAttributeError`` when a field
    varies per layer; that is a signal to read the layer configs, not an error
    the caller should see.
    """

    try:
        return getattr(obj, name, default)
    except Exception:
        return default


@dataclass(frozen=True)
class LayerCache:
    """What one decoder layer contributes to the cache.

    ``dims`` is the sliding/full branch: ``window`` tokens at most. Compressed
    attention (DeepSeek V4 and friends) additionally emits one pooled entry per
    ``compress_rate`` source tokens, which is the part that grows unbounded.
    """

    kind: str                       # "gqa" | "mla" | "linear" | "shared" | "block" | "unknown"
    dims: int = 0                   # elements cached per token (0 when not growing)
    window: Optional[int] = None    # cap for sliding layers
    compressed_dims: int = 0        # elements per compressed entry
    compress_rate: int = 0          # source tokens per compressed entry


@dataclass(frozen=True)
class KVCacheEstimate:
    layers: Tuple[LayerCache, ...]
    dtype: torch.dtype
    mtp_layers: int = 0
    context_limit: Optional[int] = None
    unclassified: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def itemsize(self) -> int:
        return _DTYPE_ITEMSIZE[self.dtype]

    @property
    def exact(self) -> bool:
        return not self.unclassified and all(layer.kind != "unknown" for layer in self.layers)

    @property
    def growing_layers(self) -> int:
        return sum(1 for layer in self.layers if layer.dims > 0)

    @property
    def sliding_layers(self) -> int:
        return sum(1 for layer in self.layers if layer.dims > 0 and layer.window)

    @property
    def window(self) -> Optional[int]:
        windows = {layer.window for layer in self.layers if layer.window}
        return max(windows) if windows else None

    @property
    def linear_layers(self) -> int:
        return sum(1 for layer in self.layers if layer.kind == "linear")

    @property
    def block_layers(self) -> int:
        return sum(1 for layer in self.layers if layer.kind == "block")

    @property
    def shared_layers(self) -> int:
        """Decoder layers that reuse an earlier layer's K/V and cache nothing."""

        return sum(1 for layer in self.layers if layer.kind == "shared")

    @property
    def layer_type(self) -> str:
        kinds = {layer.kind for layer in self.layers if layer.kind in ("gqa", "mla", "compressed")}
        if not kinds:
            return "unknown"
        return kinds.pop() if len(kinds) == 1 else "mixed"

    @property
    def bytes_per_token(self) -> int:
        """Asymptotic growth: the sliding branch stops, the compressed one does not."""

        elements = 0.0
        for layer in self.layers:
            if not layer.window:
                elements += layer.dims
            if layer.compress_rate and layer.compressed_dims:
                elements += layer.compressed_dims / layer.compress_rate
        return int(round(elements)) * self.itemsize

    def bytes_for(self, tokens: int, *, batch_size: int = 1) -> int:
        tokens = max(0, int(tokens))
        batch = max(1, int(batch_size))
        elements = 0
        for layer in self.layers:
            if layer.dims > 0:
                span = min(tokens, layer.window) if layer.window else tokens
                elements += layer.dims * span
            if layer.compress_rate and layer.compressed_dims:
                entries = (tokens + layer.compress_rate - 1) // layer.compress_rate
                elements += layer.compressed_dims * entries
        return elements * self.itemsize * batch

    def tokens_for(self, budget_bytes: int, *, batch_size: int = 1) -> int:
        """Largest usable context for ``budget_bytes`` at ``batch_size``.

        Capped by the model's own context limit: a cache that would fit 160k
        tokens is still only usable up to ``max_position_embeddings``, and
        answering past it would be a planning error, not a rounding one.
        """

        budget = max(0, int(budget_bytes)) // max(1, int(batch_size))
        if budget <= 0 or not self.growing_layers:
            return 0
        low, high = 0, self.context_limit or (1 << 20)
        while low < high:
            mid = (low + high + 1) // 2
            if self.bytes_for(mid) <= budget:
                low = mid
            else:
                high = mid - 1
        return low

    def max_batch(self, budget_bytes: int, tokens: int) -> int:
        """How many sequences of ``tokens`` fit in ``budget_bytes``."""

        per_sequence = self.bytes_for(tokens, batch_size=1)
        if per_sequence <= 0:
            return 0
        return max(0, int(budget_bytes)) // per_sequence

    def plan(self, budget_bytes: int, batches: Sequence[int] = (1, 2, 4, 8, 16, 32)) -> Tuple[Tuple[int, int], ...]:
        """``(batch, usable context)`` for each batch size, dropping empty rows."""

        rows = []
        for batch in batches:
            context = self.tokens_for(budget_bytes, batch_size=batch)
            if context > 0:
                rows.append((int(batch), int(context)))
        return tuple(rows)

    def format_plan(self, budget_bytes: int, batches: Sequence[int] = (1, 2, 4, 8, 16, 32)) -> str:
        """One line: what this budget buys, collapsing equal context widths."""

        rows = self.plan(budget_bytes, batches)
        if not rows:
            return f"KV plan: {_human(budget_bytes)} fits no usable context"

        # Rows are ordered by increasing batch, so contexts never grow; equal
        # neighbours collapse into "up to N seq".
        groups: list[list[int]] = []
        for batch, context in rows:
            if groups and groups[-1][0] == context:
                groups[-1][1] = batch
            else:
                groups.append([context, batch])

        parts = [
            f"{_tokens_name(context)} ctx up to {batch} seq" if batch > 1
            else f"{_tokens_name(context)} ctx at 1 seq"
            for context, batch in groups
        ]
        capped = " (model limit)" if self.context_limit and rows[0][1] >= self.context_limit else ""
        return f"KV plan: {_human(budget_bytes)} -> " + ", ".join(parts) + capped

    def format(self, contexts: Sequence[int] = (8192, 32768)) -> str:
        parts_of_shape = [f"{self.growing_layers} growing layer(s)"]
        if self.sliding_layers:
            parts_of_shape.append(f"{self.sliding_layers} sliding")
        shape = ", ".join(parts_of_shape) + f", {self.layer_type}"
        parts = [f"KV cache: {_human(self.bytes_per_token)}/token ({shape}, {_dtype_name(self.dtype)})"]
        for context in contexts:
            parts.append(f"{_tokens_name(context)} ctx {_human(self.bytes_for(context))}")
        if self.sliding_layers and self.window:
            parts.append(f"{self.sliding_layers} sliding layer(s) capped at {_tokens_name(self.window)}")
        if self.linear_layers:
            parts.append(f"{self.linear_layers} linear-attention layer(s) hold constant state")
        if self.block_layers:
            parts.append(f"{self.block_layers} non-attention block(s)")
        if self.shared_layers:
            parts.append(f"{self.shared_layers} KV-shared layer(s)")
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


_STACK_FIELDS = ("text_config", "thinker_config", "talker_config", "llm_config")


def _language_stacks(config: Any) -> list:
    """Every nested config that owns a decoder stack.

    Multimodal wrappers put the language model behind one or more of these
    fields; omni models run a thinker and a talker, and both hold caches. A
    wrapper without any of them is itself the language stack.
    """

    stacks = []
    for field_name in _STACK_FIELDS:
        nested = _safe_get(config, field_name)
        if nested is None:
            continue
        if _safe_get(nested, "num_hidden_layers"):
            stacks.append(nested)
        else:
            stacks.extend(_language_stacks(nested))
    if not stacks and _safe_get(config, "num_hidden_layers"):
        stacks.append(config)
    return stacks


def _declared_layer_types(source: Any, layers: int) -> Optional[list]:
    for field_name in _LAYER_TYPE_FIELDS:
        declared = _safe_get(source, field_name)
        if isinstance(declared, (list, tuple)) and declared:
            return [str(entry).lower() for entry in declared]
    return None


def _per_layer_configs(source: Any) -> Optional[list]:
    """Per-layer configs for heterogeneous stacks, or None.

    Transformers exposes this as ``_PerLayerConfigView``, a
    ``collections.abc.Sequence`` rather than a ``list``; accepting only
    ``list``/``tuple`` silently discarded Gemma 4's per-layer head shapes.
    """

    per_layer = _safe_get(source, "per_layer_config")
    if per_layer is None or isinstance(per_layer, (str, bytes)):
        return None
    if isinstance(per_layer, (list, tuple)):
        return list(per_layer) if per_layer else None
    if isinstance(per_layer, _SequenceABC):
        try:
            return [per_layer[index] for index in range(len(per_layer))]
        except Exception:
            return None
    return None


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _read_attr(obj: Any, name: str) -> Tuple[Any, bool]:
    """Read an attribute and report whether heterogeneity made it ambiguous."""

    try:
        return getattr(obj, name, None), False
    except Exception as exc:
        return None, type(exc).__name__ == "AmbiguousGlobalPerLayerAttributeError"


def _layer_attr(source: Any, layer_cfg: Any, name: str) -> Tuple[Any, bool]:
    value, ambiguous = _read_attr(layer_cfg, name)
    if value is None and layer_cfg is not source:
        value, ambiguous = _read_attr(source, name)
    return value, ambiguous


def _inferred_layer_type(source: Any, layer_cfg: Any) -> str:
    """Resolve a layer kind the way ``transformers.cache_utils`` does.

    When ``layer_types`` is absent, a per-layer ``sliding_window`` means the
    layer is sliding, otherwise a per-layer ``attention_chunk_size`` means it is
    chunked, otherwise it is full attention. Mistral, Phi-3 and Starcoder2
    checkpoints rely on this: they set ``sliding_window`` without declaring
    ``layer_types``.
    """

    window = _positive_int(_safe_get(layer_cfg, "sliding_window"))
    if window is None and layer_cfg is not source:
        window = _positive_int(_safe_get(source, "sliding_window"))
    if window is not None:
        return "sliding_attention"
    chunk = _positive_int(_safe_get(layer_cfg, "attention_chunk_size"))
    if chunk is None and layer_cfg is not source:
        chunk = _positive_int(_safe_get(source, "attention_chunk_size"))
    if chunk is not None:
        return "chunked_attention"
    return "full_attention"


def _mla_dims(source: Any, layer_cfg: Any) -> Optional[int]:
    latent = _safe_get(layer_cfg, "kv_lora_rank", _safe_get(source, "kv_lora_rank"))
    if not latent:
        return None
    rope = _safe_get(layer_cfg, "qk_rope_head_dim", _safe_get(source, "qk_rope_head_dim", 0)) or 0
    try:
        return int(latent) + int(rope)
    except (TypeError, ValueError):
        return None


def _gqa_dims(source: Any, layer_cfg: Any) -> Optional[int]:
    heads, heads_ambiguous = _layer_attr(source, layer_cfg, "num_attention_heads")
    hidden, hidden_ambiguous = _layer_attr(source, layer_cfg, "hidden_size")
    kv_heads, kv_ambiguous = _layer_attr(source, layer_cfg, "num_key_value_heads")
    head_dim, head_dim_ambiguous = _layer_attr(source, layer_cfg, "head_dim")
    if heads_ambiguous or hidden_ambiguous or kv_ambiguous or head_dim_ambiguous:
        # The value varies per layer and the caller did not give us the layer
        # view. Guessing ``hidden_size // heads`` here is what made Gemma 4
        # report 210 KiB/token, so report unknown instead.
        return None
    kv_heads = kv_heads or heads
    if not head_dim and heads and hidden:
        head_dim = hidden // heads
    if not (heads and kv_heads and head_dim):
        return None
    try:
        return 2 * int(kv_heads) * int(head_dim)
    except (TypeError, ValueError):
        return None


def _resolve_dtype(source: Any, dtype: Optional[torch.dtype]) -> Optional[torch.dtype]:
    resolved = dtype or _safe_get(source, "torch_dtype") or _safe_get(source, "dtype")
    if isinstance(resolved, str):
        resolved = getattr(torch, resolved.replace("torch.", ""), None)
    if not isinstance(resolved, torch.dtype) or resolved not in _DTYPE_ITEMSIZE:
        return torch.float16 if dtype is None and resolved is None else None
    return resolved


def estimate_kv_cache(
        config: Any,
        *,
        dtype: Optional[torch.dtype] = None,
) -> Optional[KVCacheEstimate]:
    """Return the KV-cache cost this config implies, or None if it does not say."""

    if config is None:
        return None

    stacks = _language_stacks(config)
    if not stacks:
        return None

    all_layers: list[LayerCache] = []
    unclassified_all: list[str] = []
    mtp_total = 0
    resolved_dtype = None
    limits = []
    for source in stacks:
        partial = _estimate_for_stack(source, dtype=dtype)
        if partial is None:
            continue
        all_layers.extend(partial.layers)
        unclassified_all.extend(partial.unclassified)
        mtp_total += partial.mtp_layers
        resolved_dtype = resolved_dtype or partial.dtype
        if partial.context_limit:
            limits.append(partial.context_limit)

    if not all_layers:
        return None
    return KVCacheEstimate(
        layers=tuple(all_layers),
        dtype=resolved_dtype or torch.float16,
        mtp_layers=mtp_total,
        context_limit=min(limits) if limits else None,
        unclassified=tuple(dict.fromkeys(unclassified_all)),
    )


def _estimate_for_stack(
        source: Any,
        *,
        dtype: Optional[torch.dtype] = None,
) -> Optional[KVCacheEstimate]:
    """Per-layer estimate for one decoder stack."""

    layers = _safe_get(source, "num_hidden_layers")
    per_layer_configs = _per_layer_configs(source)
    layer_count = int(layers) if layers else (len(per_layer_configs) if per_layer_configs else 0)
    if layer_count <= 0:
        return None

    resolved_dtype = _resolve_dtype(source, dtype)
    if resolved_dtype is None:
        return None

    global_window = _positive_int(_safe_get(source, "sliding_window"))

    declared_types = _declared_layer_types(source, layer_count)
    mtp = int(_safe_get(source, "num_nextn_predict_layers", 0) or 0)
    shared = _positive_int(_safe_get(source, "num_kv_shared_layers")) or 0
    shared_from = max(0, layer_count - shared)

    resolved: list[LayerCache] = []
    unclassified: list[str] = []
    for index in range(layer_count):
        # Gemma 3n/4 reuse an earlier layer's K/V for the tail; transformers does
        # not call ``past_key_values.update`` for them, so they add no bytes.
        if index >= shared_from:
            resolved.append(LayerCache(kind="shared"))
            continue

        layer_cfg = per_layer_configs[index] if per_layer_configs and index < len(per_layer_configs) else source
        layer_type = _safe_get(layer_cfg, "layer_type")
        if layer_type is not None:
            layer_type = str(layer_type).lower() or None
        if layer_type is None and declared_types and index < len(declared_types):
            layer_type = declared_types[index]
        if layer_type is None:
            layer_type = _inferred_layer_type(source, layer_cfg)
        window = _positive_int(_safe_get(layer_cfg, "sliding_window", global_window))

        if layer_type in _LINEAR_ATTENTION_TYPES:
            resolved.append(LayerCache(kind="linear"))
            continue
        if layer_type in _NO_ATTENTION_TYPES:
            resolved.append(LayerCache(kind="block"))
            continue

        dims = _mla_dims(source, layer_cfg)
        kind = "mla"
        if dims is None:
            dims = _gqa_dims(source, layer_cfg)
            kind = "gqa"
        if dims is None:
            if layer_type is None:
                resolved.append(LayerCache(kind="unknown"))
            else:
                unclassified.append(layer_type)
                resolved.append(LayerCache(kind="unknown"))
            continue

        if layer_type in _COMPRESSED_ATTENTION_TYPES:
            # DeepSeek V4-style: a bounded sliding branch plus pooled entries
            # every `compress_rate` source tokens. Shared-KV MQA stores one
            # tensor for keys and values, so there is no factor of two.
            rates = _safe_get(source, "compress_rates") or {}
            rate = rates.get(layer_type) if isinstance(rates, dict) else None
            head_dim = _safe_get(layer_cfg, "head_dim", _safe_get(source, "head_dim"))
            window = window or _safe_get(source, "sliding_window")
            if rate and head_dim:
                resolved.append(LayerCache(
                    kind="compressed",
                    dims=int(head_dim),
                    window=int(window) if window else None,
                    compressed_dims=int(head_dim),
                    compress_rate=int(rate),
                ))
            else:
                unclassified.append(layer_type)
                resolved.append(LayerCache(kind="unknown"))
            continue

        if layer_type in _CHUNKED_ATTENTION_TYPES:
            # Llama4 caps each chunked layer at ``attention_chunk_size`` tokens.
            chunk = _positive_int(_safe_get(layer_cfg, "attention_chunk_size"))
            if chunk is None and layer_cfg is not source:
                chunk = _positive_int(_safe_get(source, "attention_chunk_size"))
            if chunk is None:
                unclassified.append(layer_type)
                resolved.append(LayerCache(kind="unknown"))
            else:
                resolved.append(LayerCache(kind=kind, dims=dims, window=chunk))
            continue

        known_attention = layer_type in _FULL_ATTENTION_TYPES
        if layer_type in _SLIDING_ATTENTION_TYPES:
            resolved.append(LayerCache(kind=kind, dims=dims, window=window or 1024))
        elif known_attention:
            resolved.append(LayerCache(kind=kind, dims=dims))
        else:
            # An unrecognized layer type may or may not grow; counting it as a
            # full attention layer would produce a confident wrong number.
            unclassified.append(layer_type)
            resolved.append(LayerCache(kind="unknown"))

    resolved.extend(resolved[-1] for _ in range(mtp) if resolved)

    if not any(layer.dims > 0 for layer in resolved) and not unclassified:
        # Every layer is recurrent: there is no per-token cache to report.
        return None
    if all(layer.kind == "unknown" for layer in resolved):
        return None

    limit = _safe_get(source, "max_position_embeddings")
    limit = int(limit) if isinstance(limit, int) and limit > 0 else None

    return KVCacheEstimate(
        layers=tuple(resolved),
        dtype=resolved_dtype,
        mtp_layers=mtp,
        context_limit=limit,
        unclassified=tuple(dict.fromkeys(unclassified)),
    )


def kv_cache_unavailable_reason(config: Any) -> str:
    """Why ``estimate_kv_cache`` returned None, in words.

    Callers that report the estimate also need to explain its absence: a config
    that hides its shapes is a different situation from one whose layers are all
    recurrent, and both differ from a config with no decoder stack at all.
    """

    if config is None:
        return "no config"
    stacks = _language_stacks(config)
    if not stacks:
        return "no decoder stack"

    reasons = []
    for source in stacks:
        layers = _safe_get(source, "num_hidden_layers")
        if not layers:
            reasons.append("layer count not declared")
            continue
        declared = _declared_layer_types(source, int(layers)) or []
        if declared and all(entry in _LINEAR_ATTENTION_TYPES for entry in declared):
            reasons.append("every layer is recurrent")
            continue
        if _mla_dims(source, source) is None and _gqa_dims(source, source) is None:
            reasons.append("attention head shapes not declared")
            continue
        reasons.append("unresolved layer layout")
    return ", ".join(dict.fromkeys(reasons)) or "unknown"
