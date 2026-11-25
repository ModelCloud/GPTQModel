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
_ADD_SPECIAL_FIELDS = (("bos", "add_bos_token"), ("eos", "add_eos_token"))


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

    add_special: dict[str, bool] = {}
    for name, attr in _ADD_SPECIAL_FIELDS:
        value = getattr(tokenizer, attr, None)
        if isinstance(value, bool):
            add_special[name] = value

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
