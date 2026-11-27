# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer metadata for the llama.cpp vocabulary keys.

llama.cpp stores the vocabulary as flat arrays: one entry per embedding row,
plus merges, special-token ids and the pre-tokenizer name. A GPT-QModel
checkpoint keeps a Hugging Face tokenizer instead, whose ``get_vocab()`` may be
narrower than the embedding (Qwen3's 26 added tokens live outside
``model.vocab``) and whose rows past the vocabulary are unused. The builder
below fills every row, so the file never has fewer tokens than the embedding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import PackageError


# On-disk ``gguf.TokenType`` values; they are part of the GGUF contract.
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER_DEFINED = 4
TOKEN_TYPE_UNUSED = 5
TOKEN_TYPE_BYTE = 6

_SPECIAL_ID_FIELDS = (
    ("bos", "bos_token_id"),
    ("eos", "eos_token_id"),
    ("unk", "unk_token_id"),
    ("pad", "pad_token_id"),
)
_SPECIAL_PROBE_TEXT = "a"


@dataclass(frozen=True)
class TokenizerMetadata:
    """Everything the writer needs for the ``tokenizer.*`` GGUF keys."""

    model_kind: str
    pre: str
    tokens: tuple[str, ...]
    token_types: tuple[int, ...]
    merges: tuple[str, ...]
    special_ids: Mapping[str, int]
    add_special: Mapping[str, bool]
    chat_template: str | None


def tokenizer_kind(tokenizer: Any) -> str:
    """Return the GGUF ``tokenizer.ggml.model`` value for a fast tokenizer."""

    backend = getattr(tokenizer, "backend_tokenizer", None)
    model = getattr(backend, "model", None)
    kind = type(model).__name__ if model is not None else ""
    if kind == "BPE" and not model.byte_fallback:
        return "gpt2"
    raise PackageError(
        "package(): exporting a GGUF tokenizer requires a fast BPE tokenizer without SentencePiece byte "
        f"fallback, got {kind or type(tokenizer).__name__}."
    )


def _escape_space(part: str) -> str:
    """Encode a literal space the way llama.cpp's merge table does."""

    return "".join(chr(ord(char) + 256) if char == " " else char for char in part)


def _merges(tokenizer: Any) -> tuple[str, ...]:
    """Read the merge table from the backend tokenizer.

    Recent ``tokenizers`` builds do not expose ``model.merges`` as an
    attribute, so the serialized form is the portable source. Both the legacy
    ``"left right"`` strings and the list-of-pairs form are accepted.
    """

    backend = getattr(tokenizer, "backend_tokenizer", None)
    raw = None
    if backend is not None:
        try:
            raw = json.loads(backend.to_str()).get("model", {}).get("merges")
        except Exception:
            raw = None
    if raw is None:
        raw = getattr(getattr(backend, "model", None), "merges", None)
    if not isinstance(raw, (list, tuple)) or not raw:
        return ()
    merges = []
    for item in raw:
        if isinstance(item, str):
            # The legacy form is already "left right" with spaces encoded.
            merges.append(item)
            continue
        parts = list(item)
        if len(parts) != 2:
            raise PackageError(f"package(): unsupported merge entry {item!r} in the tokenizer.")
        merges.append(" ".join(_escape_space(str(part)) for part in parts))
    return tuple(merges)


def _normalize_added_token(tokenizer: Any, token: str) -> str:
    try:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
    except Exception:
        return token
    return decoded if isinstance(decoded, str) and decoded else token


def _probe_token_ids(tokenizer: Any, *, add_special_tokens: bool) -> list[int] | None:
    try:
        encoded = tokenizer(_SPECIAL_PROBE_TEXT, add_special_tokens=add_special_tokens)
        ids = encoded["input_ids"]
    except Exception:
        return None
    if not isinstance(ids, list) or not all(isinstance(index, int) for index in ids):
        return None
    return ids


def _observed_add_special(
    tokenizer: Any,
    special_ids: Mapping[str, int],
) -> dict[str, bool]:
    """Read the flags from what the tokenizer actually does to a plain string.

    Transformers 5 reports ``add_bos_token=False`` for Llama-3 even though the
    backend ``TemplateProcessing`` post-processor prepends
    ``<|begin_of_text|>``. Comparing a probe with and without special tokens
    follows the tokenizer instead of that attribute.
    """

    with_special = _probe_token_ids(tokenizer, add_special_tokens=True)
    without_special = _probe_token_ids(tokenizer, add_special_tokens=False)
    if with_special is None or without_special is None or not with_special:
        return {}

    flags: dict[str, bool] = {}
    bos_id = special_ids.get("bos")
    if bos_id is not None:
        flags["bos"] = (
            with_special[0] == bos_id
            and with_special.count(bos_id) > without_special.count(bos_id)
        )
    eos_id = special_ids.get("eos")
    if eos_id is not None:
        flags["eos"] = (
            with_special[-1] == eos_id
            and with_special.count(eos_id) > without_special.count(eos_id)
        )
    return flags


def _iter_post_processors(node: Any):
    if not isinstance(node, dict):
        return
    yield node
    children = node.get("processors")
    if isinstance(children, list):
        for child in children:
            yield from _iter_post_processors(child)


def _template_special_token_id(entry: Any, processor: Mapping[str, Any]) -> int | None:
    if not isinstance(entry, dict):
        return None
    special = entry.get("SpecialToken")
    if not isinstance(special, dict):
        return None
    content = special.get("id")
    if not isinstance(content, str):
        return None
    special_tokens = processor.get("special_tokens")
    if not isinstance(special_tokens, dict):
        return None
    token_info = special_tokens.get(content)
    if not isinstance(token_info, dict):
        return None
    ids = token_info.get("ids")
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int):
        return ids[0]
    return None


def _post_processor_add_special(
    tokenizer: Any,
    special_ids: Mapping[str, int],
) -> dict[str, bool]:
    """Fallback for a tokenizer whose ``__call__`` cannot be probed."""

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return {}
    try:
        post_processor = json.loads(backend.to_str()).get("post_processor")
    except Exception:
        return {}

    flags: dict[str, bool] = {}
    for processor in _iter_post_processors(post_processor):
        kind = processor.get("type")
        if kind == "RobertaProcessing":
            if "bos" in special_ids:
                flags["bos"] = True
            if "eos" in special_ids:
                flags["eos"] = True
            continue
        if kind != "TemplateProcessing":
            continue
        single = processor.get("single")
        if not isinstance(single, list) or not single:
            continue
        bos_id = special_ids.get("bos")
        if bos_id is not None:
            flags["bos"] = _template_special_token_id(single[0], processor) == bos_id
        eos_id = special_ids.get("eos")
        if eos_id is not None:
            flags["eos"] = _template_special_token_id(single[-1], processor) == eos_id
    return flags


def build_tokenizer_metadata(
    tokenizer: Any,
    *,
    vocab_size: int,
    tokenizer_pre: str | None,
    chat_template: str | None = None,
) -> TokenizerMetadata:
    """Map a Hugging Face tokenizer onto llama.cpp's flat vocabulary arrays."""

    if not isinstance(vocab_size, int) or vocab_size < 1:
        raise PackageError(f"package(): `vocab_size` must be a positive integer, got {vocab_size!r}.")

    kind = tokenizer_kind(tokenizer)
    vocab = tokenizer.get_vocab()
    reverse = {int(index): str(token) for token, index in vocab.items()}
    added = {str(token): int(index) for token, index in (tokenizer.get_added_vocab() or {}).items()}
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    special_tokens = set(getattr(tokenizer, "all_special_tokens", None) or ())

    overflow = sorted(index for index in reverse if index >= vocab_size)
    if overflow:
        raise PackageError(
            f"package(): the tokenizer has {len(overflow)} token id(s) at or above the model's "
            f"vocab_size={vocab_size} (first: {overflow[0]}). The embedding cannot represent them."
        )

    tokens: list[str] = []
    token_types: list[int] = []
    for index in range(vocab_size):
        token = reverse.get(index)
        if token is None:
            tokens.append(f"[PAD{index}]")
            token_types.append(TOKEN_TYPE_UNUSED)
            continue
        if token in added:
            entry = decoder.get(index) or decoder.get(str(index))
            if entry is not None and not getattr(entry, "normalized", True):
                token = _normalize_added_token(tokenizer, token)
            if (entry is not None and getattr(entry, "special", False)) or token in special_tokens:
                token_types.append(TOKEN_TYPE_CONTROL)
            else:
                # llama.cpp expects user-defined tokens pre-normalized.
                tokens.append(token.replace("\u2581", " "))
                token_types.append(TOKEN_TYPE_USER_DEFINED)
                continue
        else:
            token_types.append(TOKEN_TYPE_NORMAL)
        tokens.append(token)

    special_ids: dict[str, int] = {}
    for name, attr in _SPECIAL_ID_FIELDS:
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and 0 <= value < vocab_size:
            special_ids[name] = value

    add_special = _observed_add_special(tokenizer, special_ids)
    for name, value in _post_processor_add_special(tokenizer, special_ids).items():
        add_special.setdefault(name, value)

    template = chat_template if chat_template else getattr(tokenizer, "chat_template", None)
    if template is not None and not isinstance(template, str):
        raise PackageError("package(): `chat_template` must be a string.")

    return TokenizerMetadata(
        model_kind=kind,
        pre=tokenizer_pre or "default",
        tokens=tuple(tokens),
        token_types=tuple(token_types),
        merges=_merges(tokenizer),
        special_ids=special_ids,
        add_special=add_special,
        chat_template=template,
    )


__all__ = [
    "TOKEN_TYPE_BYTE",
    "TOKEN_TYPE_CONTROL",
    "TOKEN_TYPE_NORMAL",
    "TOKEN_TYPE_UNKNOWN",
    "TOKEN_TYPE_UNUSED",
    "TOKEN_TYPE_USER_DEFINED",
    "TokenizerMetadata",
    "build_tokenizer_metadata",
    "tokenizer_kind",
]
