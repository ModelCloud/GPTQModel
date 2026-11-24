# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Architecture-specific GGUF metadata and row-safe weight transforms.

The packed payload of a GPT-QModel GGUF checkpoint is a matrix of whole ggml
blocks: one block spans a contiguous run of ``in_features`` values inside a
single output row. A transform is therefore only byte-safe when it moves or
splits **output rows**. Anything that reorders input columns, or that changes a
block's scale/min/code, crosses a block boundary and needs a
dequantize/requantize pass that packaging does not perform; those layouts are
refused instead of written incorrectly.

An architecture with no registered spec has no known transform set. Packaging
refuses it by default and only falls back to the generic name map and metadata
when the caller explicitly accepts that no weight transform is applied.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .errors import CheckpointFormatError, UnsafeTransformError, UnsupportedArchitectureError


def head_dim(config: Mapping[str, Any]) -> int:
    explicit = config.get("head_dim")
    if explicit:
        return int(explicit)
    return int(config["hidden_size"]) // int(config["num_attention_heads"])


def query_dim(config: Mapping[str, Any]) -> int:
    return int(config["num_attention_heads"]) * head_dim(config)


def key_value_dim(config: Mapping[str, Any]) -> int:
    heads = config.get("num_key_value_heads")
    return int(heads if heads is not None else config["num_attention_heads"]) * head_dim(config)


def dense_decoder_shape(gguf_name: str, config: Mapping[str, Any]) -> tuple[int, ...] | None:
    """Return the logical HF ``[out, in]`` shape a dense decoder tensor must have."""

    hidden = int(config["hidden_size"])
    vocab = int(config["vocab_size"])
    query = query_dim(config)
    kv = key_value_dim(config)
    dim = head_dim(config)

    direct: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
        "output.weight": (vocab, hidden),
        "rope_freqs.weight": (dim // 2,),
    }
    if gguf_name in direct:
        return direct[gguf_name]

    match = re.fullmatch(r"blk\.\d+\.(.+)", gguf_name)
    if match is None:
        return None

    block: dict[str, tuple[int, ...]] = {
        "attn_norm.weight": (hidden,),
        "attn_q.weight": (query, hidden),
        "attn_k.weight": (kv, hidden),
        "attn_v.weight": (kv, hidden),
        "attn_output.weight": (hidden, query),
        "attn_q.bias": (query,),
        "attn_k.bias": (kv,),
        "attn_v.bias": (kv,),
        "attn_q_norm.weight": (dim,),
        "attn_k_norm.weight": (dim,),
        "ffn_norm.weight": (hidden,),
    }
    intermediate = config.get("intermediate_size")
    if intermediate:
        size = int(intermediate)
        block["ffn_gate.weight"] = (size, hidden)
        block["ffn_up.weight"] = (size, hidden)
        block["ffn_down.weight"] = (hidden, size)
    return block.get(match.group(1))


def llama_row_permutation(rows: int, n_head: int) -> np.ndarray:
    """Row order llama.cpp applies to ``q_proj``/``k_proj`` weights and biases.

    Mirrors ``convert_hf_to_gguf.LlamaModel.permute``: the per-head rotary
    halves are interleaved. The permutation moves whole rows, so a packed
    ``qweight`` keeps every block intact.
    """

    if n_head < 1 or rows % (2 * n_head) != 0:
        raise UnsafeTransformError(
            f"llama.cpp's q/k permutation needs `rows` divisible by 2*heads, got rows={rows}, heads={n_head}."
        )
    dim = rows // n_head // 2
    return np.arange(rows).reshape(n_head, 2, dim).swapaxes(1, 2).reshape(-1)


def _rope_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    parameters = config.get("rope_parameters")
    if parameters is None:
        parameters = config.get("rope_scaling")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        raise CheckpointFormatError("package(): RoPE parameters must be a JSON object.")
    if "full_attention" in parameters or "sliding_attention" in parameters:
        raise UnsafeTransformError("package(): layer-specific RoPE parameters are unsupported.")

    rope = dict(parameters)
    rope["rope_type"] = str(rope.get("rope_type") or rope.get("type") or "default").lower()
    rope.setdefault("rope_theta", config.get("rope_theta", 10000.0))
    return rope


def _positive_rope_value(rope: Mapping[str, Any], key: str, default: float | None = None) -> float:
    value = rope.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise CheckpointFormatError(f"package(): RoPE {key} must be a positive finite number, got {value!r}.")
    return float(value)


def llama3_rope_freqs(config: Mapping[str, Any]) -> np.ndarray:
    """Build the ``rope_freqs.weight`` table llama.cpp's Llama-3 scaling expects."""

    rope = _rope_parameters(config)
    base = _positive_rope_value(rope, "rope_theta")
    dim = head_dim(config)
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

    factor = _positive_rope_value(rope, "factor", 8.0)
    low_factor = _positive_rope_value(rope, "low_freq_factor", 1.0)
    high_factor = _positive_rope_value(rope, "high_freq_factor", 4.0)
    old_context = _positive_rope_value(rope, "original_max_position_embeddings", 8192)
    if high_factor <= low_factor:
        raise CheckpointFormatError("package(): Llama-3 RoPE requires high_freq_factor > low_freq_factor.")
    low_wavelen = old_context / low_factor
    high_wavelen = old_context / high_factor

    values = []
    for freq in freqs:
        wavelen = 2 * math.pi / float(freq)
        if wavelen < high_wavelen:
            values.append(1.0)
        elif wavelen > low_wavelen:
            values.append(factor)
        else:
            smooth = (old_context / wavelen - low_factor) / (high_factor - low_factor)
            values.append(1.0 / ((1.0 - smooth) / factor + smooth))
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class ArchSpec:
    """Declared layout knowledge for one GGUF architecture string."""

    arch: str = ""
    tokenizer_pre_by_model: Mapping[str, str] = field(default_factory=dict)
    verified: bool = False
    required_block_leaves: tuple[str, ...] = ()
    rope_types: tuple[str, ...] = ("default", "linear")

    def tokenizer_pre(self, model_kind: str) -> str | None:
        return self.tokenizer_pre_by_model.get(model_kind)

    def row_permutation(self, gguf_name: str, config: Mapping[str, Any]) -> np.ndarray | None:
        """Row order for a packed or dense tensor, or ``None`` to keep it as stored."""

        return None

    def extra_tensors(self, config: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
        return {}

    def extra_metadata(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        rope = _rope_parameters(config)
        rope_type = rope["rope_type"]
        if self.verified and rope_type not in self.rope_types:
            raise UnsafeTransformError(f"package(): the {self.arch} layout cannot export RoPE type {rope_type!r}.")
        rotary_factor = rope.get("partial_rotary_factor", config.get("partial_rotary_factor", 1.0))
        if self.verified and rotary_factor not in (None, 1.0):
            raise UnsafeTransformError(f"package(): the {self.arch} layout requires full rotary dimensions.")

        metadata: dict[str, Any] = {f"{self.arch}.rope.freq_base": _positive_rope_value(rope, "rope_theta")}
        if rope_type == "linear":
            metadata[f"{self.arch}.rope.scaling.type"] = "linear"
            metadata[f"{self.arch}.rope.scaling.factor"] = _positive_rope_value(rope, "factor")
        return metadata

    def validate_attention(self, config: Mapping[str, Any], n_blocks: int) -> None:
        if self.arch not in ("qwen2", "qwen3"):
            return
        layer_types = config.get("layer_types")
        if layer_types is not None:
            sliding = "sliding_attention" in layer_types
        else:
            sliding = (
                config.get("use_sliding_window")
                and config.get("sliding_window", 4096)
                and n_blocks > config.get("max_window_layers", 28)
            )
        if sliding:
            raise UnsafeTransformError(f"package(): the {self.arch} layout cannot export sliding-window attention.")

    def expected_shape(self, gguf_name: str, config: Mapping[str, Any]) -> tuple[int, ...] | None:
        return dense_decoder_shape(gguf_name, config)


@dataclass(frozen=True)
class LlamaSpec(ArchSpec):
    """Llama layout: rotary-interleaved q/k rows and optional Llama-3 rope table."""

    arch: str = "llama"
    tokenizer_pre_by_model: Mapping[str, str] = field(default_factory=lambda: {"gpt2": "llama-bpe"})
    verified: bool = True
    rope_types: tuple[str, ...] = ("default", "linear", "llama3")

    def row_permutation(self, gguf_name: str, config: Mapping[str, Any]) -> np.ndarray | None:
        heads = int(config["num_attention_heads"])
        if gguf_name.endswith(".attn_q.weight") or gguf_name.endswith(".attn_q.bias"):
            return llama_row_permutation(heads * head_dim(config), heads)
        if gguf_name.endswith(".attn_k.weight") or gguf_name.endswith(".attn_k.bias"):
            kv_heads = int(config.get("num_key_value_heads") or heads)
            return llama_row_permutation(kv_heads * head_dim(config), kv_heads)
        return None

    def extra_tensors(self, config: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
        if _rope_parameters(config)["rope_type"] != "llama3":
            return {}
        return {"rope_freqs.weight": llama3_rope_freqs(config)}


#: Architectures whose packed layout and metadata this writer has verified.
ARCH_SPECS: Mapping[str, ArchSpec] = {
    "qwen2": ArchSpec(
        arch="qwen2",
        tokenizer_pre_by_model={"gpt2": "qwen2"},
        verified=True,
        required_block_leaves=("attn_q.bias", "attn_k.bias", "attn_v.bias"),
    ),
    "qwen3": ArchSpec(
        arch="qwen3",
        tokenizer_pre_by_model={"gpt2": "qwen2"},
        verified=True,
        required_block_leaves=("attn_q_norm.weight", "attn_k_norm.weight"),
    ),
    "llama": LlamaSpec(),
}


def resolve_spec(arch: str, allow_unverified: bool) -> ArchSpec:
    spec = ARCH_SPECS.get(arch)
    if spec is not None:
        return spec
    if not allow_unverified:
        raise UnsupportedArchitectureError(
            f"package(): no verified GGUF layout is registered for architecture `{arch}`. "
            f"Registered: {sorted(ARCH_SPECS)}. Pass `allow_unverified_arch=True` to use the generic "
            "name map and metadata; that applies no weight transform, so an architecture that needs a "
            "column-wise or fused-layout transform will be written incorrectly."
        )
    return ArchSpec(arch=arch, verified=False)


__all__ = [
    "ARCH_SPECS",
    "ArchSpec",
    "LlamaSpec",
    "dense_decoder_shape",
    "head_dim",
    "key_value_dim",
    "llama3_rope_freqs",
    "llama_row_permutation",
    "query_dim",
    "resolve_spec",
]
