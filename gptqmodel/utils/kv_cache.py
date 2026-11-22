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
``indexed``      an MLA/GQA layer with a persistent sparse indexer (DeepSeek
                 V32, GLM MoE DSA, AXK2): the branch above plus ``index_head_dim``
                 indexer keys per token, retained at the indexer cache dtype.
``linear``       recurrent layers keep a constant-size state. They do not grow
                 with context; each layer's declared state is added once per
                 sequence, never in the slope. A stack whose state shapes the
                 config does not declare is reported partial, not exact.
``sliding``      growth capped at ``min(context, window)``; chunked attention
                 (Llama 4) uses ``attention_chunk_size`` as its window.
``shared``       Gemma 3n/4 tail layers reuse an earlier layer's K/V and add
                 nothing; they are reported but never counted.
``compressed``   DeepSeek V4 CSA/HCA: a bounded sliding branch plus pooled
                 entries every ``compress_rate`` source tokens. CSA keeps a
                 compressor and a Lightning Indexer stream, a window buffer and
                 the two-series overlap state; HCA keeps only the compressor.

Stacks that mix those (GLM5, DeepSeek V4.1 Flash, Gemma4 and similar) declare the
mix either through ``layer_types`` or through a heterogeneous
``per_layer_config`` where per-layer attributes refuse global reads. Both are
resolved layer by layer; anything unclassifiable is named instead of being folded
into a total that looks authoritative.
"""

from collections.abc import Mapping as _MappingABC
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
# transformers maps these to ``DynamicIndexedLayer``, which retains a second
# ``[batch, seq, index_head_dim]`` key cache next to the latent K/V.
_INDEXED_ATTENTION_TYPES = frozenset({
    "indexed_attention", "deepseek_sparse_attention", "qwen_sparse_attention",
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

_LAYER_TYPE_FIELDS = (
    "layer_types", "attention_layer_types", "attn_layer_types", "layers_block_type",
)
# Nemotron-H/Jamba-style remote configs publish only ``hybrid_override_pattern``:
# ``M`` is a Mamba2 mixer, ``*`` a full-attention layer, ``-`` an MLP and ``E`` a MoE
# block. The remote NemotronH config also exposes ``layers_block_type`` built from it;
# the pattern is the fallback for configs that do not.
_HYBRID_OVERRIDE_TYPES = {
    "m": "mamba", "*": "attention", "-": "mlp", "e": "moe",
}


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
    attention (DeepSeek V4 and friends) additionally emits pooled entries every
    ``compress_rate`` source tokens, which is the part that grows unbounded.
    ``compressed_dims`` counts every pooled stream (CSA keeps a compressor and a
    Lightning Indexer), ``buffer_dims`` the per-source-token scratch held between
    windows, and ``overlap_dims`` the fixed state the two-series CSA window
    carries across a forward boundary. An ``indexed`` layer additionally keeps
    ``indexer_dims`` per token; ``indexer_dtype`` is the dtype that stream was
    allocated with (``None`` means the estimate's cache dtype). ``state_bytes``
    is the fixed per-sequence state of a recurrent branch (Mamba conv/SSM,
    gated-delta conv/recurrent); it never grows with the context and is
    already in the dtypes transformers allocates it with. ``state_unsized``
    marks a recurrent branch whose state shape the config does not declare:
    the byte count is unknown, not zero.
    """

    kind: str                       # "gqa" | "mla" | "indexed" | "linear" | "shared" | "block" | "unknown"
    dims: int = 0                   # elements cached per token (0 when not growing)
    window: Optional[int] = None    # cap for sliding layers
    compressed_dims: int = 0        # elements per compressed entry, all streams
    compress_rate: int = 0          # source tokens per compressed entry
    compressed_streams: int = 1     # pooled streams folded into compressed_dims
    buffer_dims: int = 0            # elements per buffered source token, all streams
    overlap_dims: int = 0           # fixed elements retained, all streams
    indexer_dims: int = 0           # indexer elements cached per token
    indexer_dtype: Optional[torch.dtype] = None  # dtype of the indexer stream
    state_bytes: int = 0            # fixed bytes per sequence (recurrent state)
    state_unsized: bool = False     # state exists but its shape is undeclared


@dataclass(frozen=True)
class KVCacheEstimate:
    """Per-context KV-cache cost implied by a config.

    Every number here is an **estimate** derived from the declared shapes and
    the weight dtype - nothing is measured from a running engine. Names carry
    their units (``kv_bytes_per_token``, ``max_context_tokens``) so a caller
    does not have to read this class to know what a value counts.
    """

    layers: Tuple[LayerCache, ...]
    dtype: torch.dtype
    mtp_layers: int = 0
    context_limit: Optional[int] = None
    unclassified: Tuple[str, ...] = field(default_factory=tuple)
    partial: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def kv_element_bytes(self) -> int:
        return _DTYPE_ITEMSIZE[self.dtype]

    def _indexer_element_bytes(self, layer: LayerCache) -> int:
        dtype = layer.indexer_dtype or self.dtype
        return _DTYPE_ITEMSIZE.get(dtype, self.kv_element_bytes)

    @property
    def is_exact(self) -> bool:
        return (
            not self.unclassified
            and not self.partial
            and all(layer.kind != "unknown" for layer in self.layers)
        )

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
    def state_layers(self) -> int:
        """Layers that keep a sized constant state (a recurrent branch)."""

        return sum(1 for layer in self.layers if layer.state_bytes > 0)

    @property
    def state_bytes_per_sequence(self) -> int:
        """Constant recurrent state one sequence keeps, across every layer."""

        return sum(layer.state_bytes for layer in self.layers)

    @property
    def unsized_state_layers(self) -> int:
        """Layers whose recurrent state exists but the config does not size."""

        return sum(1 for layer in self.layers if layer.state_unsized)

    @property
    def block_layers(self) -> int:
        return sum(1 for layer in self.layers if layer.kind == "block")

    @property
    def compressed_streams(self) -> int:
        """Pooled streams across compressed layers (CSA keeps compressor + indexer)."""

        return sum(
            layer.compressed_streams
            for layer in self.layers
            if layer.compress_rate and layer.compressed_dims
        )

    @property
    def indexer_layers(self) -> int:
        """Layers that also retain a persistent sparse-indexer key cache."""

        return sum(1 for layer in self.layers if layer.indexer_dims > 0)

    @property
    def shared_layers(self) -> int:
        """Decoder layers that reuse an earlier layer's K/V and cache nothing."""

        return sum(1 for layer in self.layers if layer.kind == "shared")

    @property
    def layer_type(self) -> str:
        kinds = {
            layer.kind
            for layer in self.layers
            if layer.kind in ("gqa", "mla", "indexed", "compressed")
        }
        if not kinds:
            return "unknown"
        return kinds.pop() if len(kinds) == 1 else "mixed"

    @property
    def kv_bytes_per_token(self) -> int:
        """Asymptotic growth: the sliding branch stops, the compressed one does not."""

        elements = 0.0
        indexer_bytes = 0.0
        for layer in self.layers:
            if not layer.window:
                elements += layer.dims
                if layer.indexer_dims:
                    indexer_bytes += layer.indexer_dims * self._indexer_element_bytes(layer)
            if layer.compress_rate and layer.compressed_dims:
                elements += layer.compressed_dims / layer.compress_rate
        return int(round(elements)) * self.kv_element_bytes + int(round(indexer_bytes))

    def kv_bytes_for(self, context_tokens: int, *, batch_size: int = 1) -> int:
        context_tokens = max(0, int(context_tokens))
        batch = max(1, int(batch_size))
        total = 0
        for layer in self.layers:
            elements = 0
            if layer.dims > 0:
                span = min(context_tokens, layer.window) if layer.window else context_tokens
                elements += layer.dims * span
                if layer.indexer_dims:
                    total += layer.indexer_dims * span * self._indexer_element_bytes(layer)
            if layer.compress_rate and layer.compressed_dims:
                entries = context_tokens // layer.compress_rate
                elements += layer.compressed_dims * entries
                if entries:
                    elements += layer.overlap_dims
                if context_tokens and layer.compress_rate > 1:
                    # The window buffer holds up to ``compress_rate - 1`` source
                    # tokens. Plan for a full buffer: that is the peak a run of
                    # this length reaches, and a constant (rather than
                    # ``context_tokens % compress_rate``) keeps this function
                    # monotonic for ``max_context_tokens``'s binary search.
                    elements += layer.buffer_dims * min(
                        context_tokens, layer.compress_rate - 1)
            total += elements * self.kv_element_bytes
            if context_tokens > 0:
                # Recurrent (Mamba/linear-attention) state is constant per
                # sequence, not per token: a live sequence keeps one copy.
                total += layer.state_bytes
        return total * batch

    def max_context_tokens(self, kv_budget_bytes: int, *, batch_size: int = 1) -> Optional[int]:
        """Largest usable context for ``budget_bytes`` at ``batch_size``.

        Capped by the model's own context limit: a cache that would fit 160k
        tokens is still only usable up to ``max_position_embeddings``, and
        answering past it would be a planning error, not a rounding one.

        ``None`` means the answer is unknown. A cache with no per-token growth
        is not capped by the budget at all: when the model also declares no
        context limit, the answer is unknown rather than the ``1 << 20`` bound
        used to search a growing cache.
        """

        budget = max(0, int(kv_budget_bytes)) // max(1, int(batch_size))
        if budget <= 0:
            return 0
        # Recurrent state is constant per live sequence, so every sequence
        # spends it before its first token. A device that owns only recurrent
        # layers never grows its cache but still binds the batch size (#5658).
        if self.state_bytes_per_sequence > budget:
            return 0
        if not self.growing_layers:
            # Nothing per-token to count: only the model's own limit caps it,
            # and an undeclared limit is unknown, not the search sentinel.
            return self.context_limit
        low, high = 0, self.context_limit or (1 << 20)
        while low < high:
            mid = (low + high + 1) // 2
            if self.kv_bytes_for(mid) <= budget:
                low = mid
            else:
                high = mid - 1
        return low

    def max_sequences(self, kv_budget_bytes: int, context_tokens: int) -> int:
        """How many sequences of ``context_tokens`` fit in ``kv_budget_bytes``."""

        per_sequence = self.kv_bytes_for(context_tokens, batch_size=1)
        if per_sequence <= 0:
            return 0
        return max(0, int(kv_budget_bytes)) // per_sequence

    def context_batch_plan(self, kv_budget_bytes: int, batches: Sequence[int] = (1, 2, 4, 8, 16, 32)) -> Optional[Tuple[Tuple[int, int], ...]]:
        """``(batch_size, max_context_tokens)`` per batch size, dropping empty rows.

        ``None`` when any row's context is unknown: a partial plan would imply
        the missing batch sizes fit nothing when they are merely unmeasured.
        """

        rows = []
        for batch in batches:
            context = self.max_context_tokens(kv_budget_bytes, batch_size=batch)
            if context is None:
                return None
            if context > 0:
                rows.append((int(batch), int(context)))
        return tuple(rows)

    def format_plan_rows(self, rows: Sequence[Tuple[int, int]], *,
                         kv_budget_bytes: Optional[int] = None,
                         suffix: str = "") -> str:
        """One line from precomputed ``(batch_size, max_context_tokens)`` rows.

        ``kv_budget_bytes`` labels the line with the budget the rows came from.
        It is optional because a sharded caller plans each device separately and
        has no single whole-model number that those rows correspond to.
        """

        if not rows:
            head = f"KV plan: {human_bytes(kv_budget_bytes)} " if kv_budget_bytes is not None else "KV plan: "
            return head + "fits no usable context" + suffix

        # Rows are ordered by increasing batch, so contexts never grow; equal
        # neighbours collapse into "up to N seq".
        groups: list[list[int]] = []
        for batch, context in rows:
            if groups and groups[-1][0] == context:
                groups[-1][1] = batch
            else:
                groups.append([context, batch])

        parts = [
            f"{human_tokens(context)} ctx up to {batch} seq" if batch > 1
            else f"{human_tokens(context)} ctx at 1 seq"
            for context, batch in groups
        ]
        capped = " (model limit)" if self.context_limit and rows[0][1] >= self.context_limit else ""
        prefix = f"KV plan: {human_bytes(kv_budget_bytes)} -> " if kv_budget_bytes is not None else "KV plan: "
        return prefix + ", ".join(parts) + capped + suffix

    def format_context_batch_plan(self, kv_budget_bytes: int, batches: Sequence[int] = (1, 2, 4, 8, 16, 32)) -> Optional[str]:
        """One line: what this budget buys, or None when the context is unknown."""

        rows = self.context_batch_plan(kv_budget_bytes, batches)
        if rows is None:
            return None
        return self.format_plan_rows(rows, kv_budget_bytes=int(kv_budget_bytes))

    def describe(self, contexts: Sequence[int] = (8192, 32768)) -> str:
        parts_of_shape = [f"{self.growing_layers} growing layer(s)"]
        if self.sliding_layers:
            parts_of_shape.append(f"{self.sliding_layers} sliding")
        if self.indexer_layers:
            parts_of_shape.append(f"{self.indexer_layers} indexed")
        if self.compressed_streams > 1:
            parts_of_shape.append(f"{self.compressed_streams} compressed streams")
        shape = ", ".join(parts_of_shape) + f", {self.layer_type}"
        parts = [f"KV cache: {human_bytes(self.kv_bytes_per_token)}/token ({shape}, {_dtype_name(self.dtype)})"]
        for context in contexts:
            parts.append(f"{human_tokens(context)} ctx {human_bytes(self.kv_bytes_for(context))}")
        if self.sliding_layers and self.window:
            parts.append(f"{self.sliding_layers} sliding layer(s) capped at {human_tokens(self.window)}")
        if self.linear_layers:
            if self.state_layers == self.linear_layers:
                parts.append(
                    f"{self.linear_layers} linear-attention layer(s) hold "
                    f"{human_bytes(self.state_bytes_per_sequence)} constant state")
            else:
                parts.append(f"{self.linear_layers} linear-attention layer(s) hold constant state")
        elif self.state_layers:
            parts.append(
                f"{self.state_layers} hybrid layer(s) hold "
                f"{human_bytes(self.state_bytes_per_sequence)} constant state")
        if self.block_layers:
            parts.append(f"{self.block_layers} non-attention block(s)")
        if self.shared_layers:
            parts.append(f"{self.shared_layers} KV-shared layer(s)")
        if self.mtp_layers:
            parts.append(f"incl. {self.mtp_layers} MTP layer(s)")
        if self.unclassified:
            parts.append(f"unknown layer type(s): {', '.join(self.unclassified)}")
        if self.partial:
            parts.append(f"partial: {', '.join(self.partial)}")
        return " | ".join(parts)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def human_tokens(tokens: int) -> str:
    if tokens >= 1024 and tokens % 1024 == 0:
        return f"{tokens // 1024}k"
    return str(tokens)


def human_bytes(num_bytes: int) -> str:
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
    pattern = _safe_get(source, "hybrid_override_pattern")
    if isinstance(pattern, str) and pattern:
        if layers and len(pattern) != int(layers):
            return None
        return [_HYBRID_OVERRIDE_TYPES.get(char.lower(), char.lower()) for char in pattern]
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


def _field(obj: Any, name: str) -> Any:
    """Read a field from a config object or a raw config ``dict``.

    Remote configs arrive either as the class their ``auto_map`` names (which
    may build ``block_configs`` into dataclasses) or as the plain dictionary
    from ``config.json`` when the remote module is unavailable.
    """

    if isinstance(obj, _MappingABC):
        return obj.get(name)
    return _safe_get(obj, name)


@dataclass(frozen=True)
class _LayerLayout:
    """Per-layer KV layout declared outside ``num_key_value_heads``.

    ``kv_heads`` overrides the global head count, ``skip`` names why the layer
    keeps no K/V at all, and ``partial`` records a declaration the estimator
    cannot model (the answer is then reported as inexact).
    """

    kv_heads: Optional[int] = None
    skip: Optional[str] = None
    partial: Optional[str] = None


def _per_layer_layout(source: Any, index: Optional[int]) -> _LayerLayout:
    """Resolve DeciLM/Nemotron-NAS per-layer KV declarations for one layer.

    DeciLM publishes ``num_key_value_heads_per_layer``; Nemotron-NAS publishes
    ``block_configs[i].attention`` where ``n_heads_in_group`` is the query/KV
    grouping and ``no_op`` / ``replace_with_linear`` mean the layer has no
    attention module and therefore no cache. Reading only the global
    ``num_key_value_heads`` (absent on both) assumed MHA and overstated the
    cache by 15-40x while still reporting ``is_exact``.
    """

    if index is None:
        return _LayerLayout()

    per_layer_heads = _safe_get(source, "num_key_value_heads_per_layer")
    if isinstance(per_layer_heads, (list, tuple)) and per_layer_heads:
        if index >= len(per_layer_heads):
            return _LayerLayout(
                partial="num_key_value_heads_per_layer is shorter than the stack")
        kv_heads = _positive_int(per_layer_heads[index])
        if kv_heads is None:
            return _LayerLayout(
                partial=f"num_key_value_heads_per_layer[{index}] is not a positive int")
        return _LayerLayout(kv_heads=kv_heads)

    block_configs = _safe_get(source, "block_configs")
    if isinstance(block_configs, (list, tuple)) and block_configs:
        if index >= len(block_configs):
            return _LayerLayout(partial="block_configs is shorter than the stack")
        attention = _field(block_configs[index], "attention")
        if attention is None:
            return _LayerLayout(
                partial=f"block_configs[{index}] declares no attention block")
        if _field(attention, "no_op"):
            return _LayerLayout(skip="no-op attention")
        if _field(attention, "replace_with_linear"):
            return _LayerLayout(skip="linear attention replacement")
        partial = None
        if (_field(attention, "window_length")
                or _field(attention, "num_sink_tokens")
                or _field(attention, "sparsify")):
            partial = "block_configs attention options are not modelled"
        heads = _positive_int(_safe_get(source, "num_attention_heads"))
        in_group = _positive_int(_field(attention, "n_heads_in_group"))
        if in_group is None:
            return _LayerLayout(
                partial=partial or "block_configs attention does not declare n_heads_in_group")
        if heads is None or heads % in_group:
            return _LayerLayout(
                partial="n_heads_in_group does not divide num_attention_heads")
        return _LayerLayout(kv_heads=heads // in_group, partial=partial)
    return _LayerLayout()


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


def _gqa_dims(
        source: Any,
        layer_cfg: Any,
        *,
        kv_heads: Optional[int] = None,
) -> Optional[int]:
    heads, heads_ambiguous = _layer_attr(source, layer_cfg, "num_attention_heads")
    hidden, hidden_ambiguous = _layer_attr(source, layer_cfg, "hidden_size")
    if kv_heads is None:
        kv_heads, kv_ambiguous = _layer_attr(source, layer_cfg, "num_key_value_heads")
    else:
        kv_ambiguous = False
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


_HYBRID_TYPES = frozenset({"hybrid"})

_STATE_CACHE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _first_positive_int(source: Any, layer_cfg: Any, *names: str) -> Optional[int]:
    """First declared positive int among ``names``, layer view before the stack."""

    for name in names:
        value = _positive_int(_safe_get(layer_cfg, name))
        if value is None and layer_cfg is not source:
            value = _positive_int(_safe_get(source, name))
        if value is not None:
            return value
    return None


def _recurrent_state_dtype(source: Any, layer_cfg: Any) -> torch.dtype:
    """Dtype transformers allocates a stack's SSM/recurrent state with.

    Mamba and gated-delta mixers build the recurrent state in FP32 whatever the
    weight dtype; only the conv state follows the activations. Nemotron-H is
    the exception that declares ``mamba_ssm_cache_dtype``.
    """

    declared = _safe_get(layer_cfg, "mamba_ssm_cache_dtype")
    if declared is None and layer_cfg is not source:
        declared = _safe_get(source, "mamba_ssm_cache_dtype")
    if isinstance(declared, torch.dtype):
        return declared
    if isinstance(declared, str):
        return _STATE_CACHE_DTYPES.get(declared.replace("torch.", ""), torch.float32)
    return torch.float32


def _state_bytes(source: Any, layer_cfg: Any, dtype: torch.dtype) -> Optional[int]:
    """Constant per-sequence bytes a recurrent branch keeps, or None if unknown.

    Each formula mirrors the tensors the HF cache layer allocates: the conv
    state follows the activation dtype, the SSM/recurrent state its own dtype.
    ``None`` means the config does not declare enough to size the state, so the
    caller must report a partial estimate instead of counting zero bytes.
    """

    conv_bytes = _DTYPE_ITEMSIZE.get(dtype)
    state_element_bytes = _DTYPE_ITEMSIZE.get(_recurrent_state_dtype(source, layer_cfg))
    if conv_bytes is None or state_element_bytes is None:
        return None

    # Gated-delta (Qwen3-Next/3.5): a conv state over the key/value channels
    # plus one value-head x key-head x value-dim recurrent state.
    value_heads = _first_positive_int(source, layer_cfg, "linear_num_value_heads")
    if value_heads:
        key_heads = _first_positive_int(source, layer_cfg, "linear_num_key_heads")
        key_dim = _first_positive_int(source, layer_cfg, "linear_key_head_dim")
        value_dim = _first_positive_int(source, layer_cfg, "linear_value_head_dim")
        conv_kernel = _first_positive_int(source, layer_cfg, "linear_conv_kernel_dim")
        if not (key_heads and key_dim and value_dim and conv_kernel):
            return None
        recurrent = value_heads * key_dim * value_dim
        conv = (2 * key_heads * key_dim + value_heads * value_dim) * conv_kernel
        return recurrent * state_element_bytes + conv * conv_bytes

    # Mamba-2 (Bamba, Falcon-H1, Nemotron-H, Zamba2): the recurrent state is
    # heads x head_dim x d_state; the conv sees d_ssm + 2 * groups * d_state.
    heads = _first_positive_int(
        source, layer_cfg, "mamba_n_heads", "mamba_num_heads", "n_mamba_heads", "num_heads")
    head_dim = _first_positive_int(
        source, layer_cfg, "mamba_d_head", "mamba_head_dim", "mamba_headdim", "head_dim")
    groups = _first_positive_int(
        source, layer_cfg, "mamba_n_groups", "mamba_ngroups", "n_groups")
    d_state = _first_positive_int(
        source, layer_cfg, "mamba_d_state", "state_size", "ssm_state_size")
    d_conv = _first_positive_int(source, layer_cfg, "mamba_d_conv", "conv_kernel")
    if heads and groups and d_state and d_conv:
        d_ssm = _first_positive_int(source, layer_cfg, "mamba_d_ssm")
        if head_dim is None and d_ssm and d_ssm % heads == 0:
            head_dim = d_ssm // heads
        if head_dim is None:
            return None
        if d_ssm is None:
            expand = _first_positive_int(source, layer_cfg, "mamba_expand", "expand")
            hidden = _first_positive_int(source, layer_cfg, "hidden_size")
            if not (expand and hidden):
                return None
            d_ssm = expand * hidden
        recurrent = heads * head_dim * d_state
        conv = (d_ssm + 2 * groups * d_state) * d_conv
        return recurrent * state_element_bytes + conv * conv_bytes

    # Mamba-1 (Jamba, Zamba, Falcon-Mamba): d_inner = expand * hidden, and both
    # the conv and the recurrent state keep ``d_inner`` rows.
    d_state = _first_positive_int(
        source, layer_cfg, "mamba_d_state", "state_size", "ssm_state_size")
    d_conv = _first_positive_int(source, layer_cfg, "mamba_d_conv", "conv_kernel")
    expand = _first_positive_int(source, layer_cfg, "mamba_expand", "expand")
    hidden = _first_positive_int(source, layer_cfg, "hidden_size")
    if not (d_state and d_conv and expand and hidden):
        return None
    d_inner = expand * hidden
    return d_inner * d_state * state_element_bytes + d_inner * d_conv * conv_bytes


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
    partial_all: list[str] = []
    mtp_total = 0
    resolved_dtype = None
    limits = []
    for source in stacks:
        stack = _estimate_for_stack(source, dtype=dtype)
        if stack is None:
            continue
        all_layers.extend(stack.layers)
        unclassified_all.extend(stack.unclassified)
        partial_all.extend(stack.partial)
        mtp_total += stack.mtp_layers
        resolved_dtype = resolved_dtype or stack.dtype
        if stack.context_limit:
            limits.append(stack.context_limit)

    if not all_layers:
        return None
    return KVCacheEstimate(
        layers=tuple(all_layers),
        dtype=resolved_dtype or torch.float16,
        mtp_layers=mtp_total,
        context_limit=min(limits) if limits else None,
        unclassified=tuple(dict.fromkeys(unclassified_all)),
        partial=tuple(dict.fromkeys(partial_all)),
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

    if _safe_get(source, "model_type") == "deepseek_v41_text":
        return _estimate_deepseek_v41(source, layer_count, resolved_dtype)

    global_window = _positive_int(_safe_get(source, "sliding_window"))

    declared_types = _declared_layer_types(source, layer_count)
    if declared_types is None and _safe_get(source, "hybrid_override_pattern"):
        # A pattern that does not match the stack cannot be mapped; falling
        # back to "every layer is attention" would be confidently wrong.
        partial_note = "hybrid_override_pattern does not match the layer count"
    else:
        partial_note = None
    if declared_types is not None and len(declared_types) != layer_count:
        partial_note = (
            f"declared layer layout covers {len(declared_types)} of {layer_count} layers")
    mtp = int(_safe_get(source, "num_nextn_predict_layers", 0) or 0)
    shared = _positive_int(_safe_get(source, "num_kv_shared_layers")) or 0
    shared_from = max(0, layer_count - shared)

    resolved: list[LayerCache] = []
    unclassified: list[str] = []
    partial: list[str] = [partial_note] if partial_note else []
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
            state_bytes = _state_bytes(source, layer_cfg, resolved_dtype)
            state_unsized = state_bytes is None
            if state_unsized:
                partial.append(f"{layer_type} state layout not declared")
                state_bytes = 0
            resolved.append(LayerCache(
                kind="linear", state_bytes=state_bytes, state_unsized=state_unsized))
            continue
        if layer_type in _NO_ATTENTION_TYPES:
            resolved.append(LayerCache(kind="block"))
            continue

        layout = _per_layer_layout(source, index)
        if layout.partial:
            partial.append(layout.partial)
        if layout.skip:
            resolved.append(LayerCache(kind="block"))
            continue

        dims = _mla_dims(source, layer_cfg)
        kind = "mla"
        if dims is None:
            dims = _gqa_dims(source, layer_cfg, kv_heads=layout.kv_heads)
            kind = "gqa"
        if dims is None:
            if layer_type is None:
                resolved.append(LayerCache(kind="unknown"))
            else:
                unclassified.append(layer_type)
                resolved.append(LayerCache(kind="unknown"))
            continue

        # Indexed attention keeps a second key cache: one ``index_head_dim``
        # vector per token, in the indexer's own dtype. Without the declared
        # width the latent branch is still countable, but the total is not
        # exact and must say so rather than look authoritative.
        indexer_dims = 0
        indexer_dtype = None
        if layer_type in _INDEXED_ATTENTION_TYPES:
            indexer_dims = _positive_int(
                _safe_get(layer_cfg, "index_head_dim",
                          _safe_get(source, "index_head_dim"))) or 0
            if indexer_dims:
                kind = "indexed"
                indexer_dtype = resolved_dtype
            else:
                partial.append(f"{layer_type} indexer layout not declared")

        hybrid_state = 0
        hybrid_unsized = False
        if layer_type in _HYBRID_TYPES:
            # A parallel hybrid keeps standard K/V for its attention half and a
            # constant recurrent state for its Mamba half (Falcon-H1, Zamba).
            sized = _state_bytes(source, layer_cfg, resolved_dtype)
            if sized is None:
                partial.append(f"{layer_type} state layout not declared")
                hybrid_unsized = True
            else:
                hybrid_state = sized

        if layer_type in _COMPRESSED_ATTENTION_TYPES:
            # DeepSeek V4-style: a bounded sliding branch plus pooled entries
            # every `compress_rate` source tokens. Shared-KV MQA stores one
            # tensor for keys and values, so there is no factor of two.
            rates = _safe_get(source, "compress_rates") or {}
            rate = rates.get(layer_type) if isinstance(rates, dict) else None
            head_dim = _safe_get(layer_cfg, "head_dim", _safe_get(source, "head_dim"))
            window = window or _safe_get(source, "sliding_window")
            if rate and head_dim:
                csa = layer_type == "compressed_sparse_attention"
                # CSA runs a Lightning Indexer next to the compressor. The
                # indexer compresses the same windows at `index_head_dim` and
                # keeps its own entries, buffer and overlap state, so both
                # streams are retained. HCA has only the compressor.
                indexer_dims = 0
                if csa:
                    indexer_dims = _positive_int(
                        _safe_get(layer_cfg, "index_head_dim",
                                  _safe_get(source, "index_head_dim"))) or int(head_dim)
                pooled = int(head_dim) + indexer_dims
                resolved.append(LayerCache(
                    kind="compressed",
                    dims=int(head_dim),
                    window=int(window) if window else None,
                    compressed_dims=pooled,
                    compress_rate=int(rate),
                    compressed_streams=2 if csa else 1,
                    # The CSA two-series projector writes ``2 * head_dim`` per
                    # source token for each stream and keeps a K and a gate
                    # buffer; HCA writes ``head_dim`` and has no overlap state.
                    buffer_dims=(4 if csa else 2) * pooled,
                    overlap_dims=2 * int(rate) * pooled if csa else 0,
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
                resolved.append(LayerCache(
                    kind=kind, dims=dims, window=chunk,
                    indexer_dims=indexer_dims, indexer_dtype=indexer_dtype,
                    state_bytes=hybrid_state, state_unsized=hybrid_unsized,
                ))
            continue

        known_attention = layer_type in _FULL_ATTENTION_TYPES
        if layer_type in _SLIDING_ATTENTION_TYPES:
            resolved.append(LayerCache(
                kind=kind, dims=dims, window=window or 1024,
                indexer_dims=indexer_dims, indexer_dtype=indexer_dtype,
                state_bytes=hybrid_state, state_unsized=hybrid_unsized,
            ))
        elif known_attention:
            resolved.append(LayerCache(
                kind=kind, dims=dims,
                indexer_dims=indexer_dims, indexer_dtype=indexer_dtype,
                state_bytes=hybrid_state, state_unsized=hybrid_unsized,
            ))
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
        partial=tuple(dict.fromkeys(partial)),
    )


def _estimate_deepseek_v41(source, layer_count, dtype):
    """Count the window for each layer and the compressed bank once per source."""
    head_dim = _positive_int(_safe_get(source, "head_dim"))
    index_dim = _positive_int(_safe_get(source, "index_head_dim"))
    window = _positive_int(_safe_get(source, "sliding_window"))
    ratios = _safe_get(source, "compress_ratios") or []
    sources = _safe_get(source, "kv_source_layer_ids") or []
    if not (head_dim and index_dim and window) or len(ratios) < layer_count:
        return None
    if any(not 0 <= index < layer_count or not ratios[index] for index in sources):
        return None

    layers = [LayerCache(kind="mla", dims=head_dim if window > 1 else 0, window=window - 1)
              for _ in range(layer_count)]
    for index in sources:
        layers[index] = LayerCache(
            kind="compressed", dims=layers[index].dims, window=window - 1,
            compressed_dims=head_dim + index_dim, compress_rate=ratios[index], compressed_streams=2,
            # The incomplete pooling group's values and gates remain FP32.
            buffer_dims=2 * head_dim * _DTYPE_ITEMSIZE[torch.float32] // _DTYPE_ITEMSIZE[dtype],
        )
    return KVCacheEstimate(
        layers=tuple(layers), dtype=dtype, context_limit=_positive_int(_safe_get(source, "max_position_embeddings")),
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
