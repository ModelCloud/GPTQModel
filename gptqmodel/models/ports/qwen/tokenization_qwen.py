# Copyright (c) Alibaba Cloud.
# Modified by ModelCloud.ai for Transformers 5; see NOTICE and LICENSE.
"""Source original Qwen TikToken vocabulary and NFC/prompt semantics."""

import base64
import unicodedata
from pathlib import Path

import tiktoken
from transformers.tokenization_python import PythonBackend


PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = {
    token: 151643 + idx
    for idx, token in enumerate(
        ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] + [f"<|extra_{i}|>" for i in range(205)]
    )
}


def _read_bpe(path, require_bytes=True):
    try:
        if path is None:
            raise ValueError("required qwen.tiktoken is missing")
        pairs = [
            (base64.b64decode(token, validate=True), int(rank))
            for line in Path(path).read_bytes().splitlines()
            if line
            for token, rank in [line.split()]
        ]
        ranks = dict(pairs)
        if len(ranks) != len(pairs):
            raise ValueError("duplicate BPE token")
        if len(set(ranks.values())) != len(ranks) or any(i < 0 for i in ranks.values()):
            raise ValueError("duplicate or negative BPE rank")
        if require_bytes and not all(bytes([i]) in ranks for i in range(256)):
            raise ValueError("missing byte token")
        if require_bytes and set(ranks.values()) & set(SPECIAL_TOKENS.values()):
            raise ValueError("BPE rank overlaps Qwen special token")
        return ranks
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid original Qwen vocabulary {path}: {exc}") from exc


class QWenTokenizer(PythonBackend):
    vocab_files_names = {"vocab_file": "qwen.tiktoken"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, errors="replace", extra_vocab_file=None, **kwargs):
        self.errors = errors
        self.mergeable_ranks = _read_bpe(vocab_file)
        self.special_tokens = dict(SPECIAL_TOKENS)
        if extra_vocab_file is not None:
            used = set(self.mergeable_ranks.values()) | set(self.special_tokens.values())
            for token, rank in _read_bpe(extra_vocab_file, require_bytes=False).items():
                if token not in self.mergeable_ranks and rank not in used:
                    self.mergeable_ranks[token] = rank
                    used.add(rank)
        self.tokenizer = tiktoken.Encoding(
            "Qwen", pat_str=PAT_STR, mergeable_ranks=self.mergeable_ranks, special_tokens=self.special_tokens
        )
        self.decoder = {rank: token for token, rank in self.mergeable_ranks.items()}
        self.decoder.update({rank: token for token, rank in self.special_tokens.items()})
        self.eod_id, self.im_start_id, self.im_end_id = 151643, 151644, 151645
        # Serialized token metadata is native PythonBackend's responsibility;
        # corrective padding/BOS/chat normalization remains in Tokenicer.
        super().__init__(errors=errors, **kwargs)

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def get_vocab(self):
        return {**self.mergeable_ranks, **self.special_tokens}

    def _add_tokens(self, new_tokens, special_tokens=False):
        for token in new_tokens:
            if str(token) not in self.special_tokens:
                raise ValueError("Original Qwen does not support adding unknown tokens")
        return 0

    def tokenize(self, text, allowed_special="all", disallowed_special=(), **kwargs):
        return [
            self.decoder[i]
            for i in self.tokenizer.encode(
                unicodedata.normalize("NFC", text),
                allowed_special=allowed_special,
                disallowed_special=disallowed_special,
            )
        ]

    def _tokenize(self, text, **kwargs):
        return self.tokenize(text, **kwargs)

    def _convert_token_to_id(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError(f"Unknown Qwen token {token!r}")

    def _convert_id_to_token(self, index):
        if index not in self.decoder:
            raise ValueError(f"Unknown Qwen token ID {index}")
        return self.decoder[index]

    def convert_tokens_to_string(self, tokens):
        return b"".join(t if isinstance(t, bytes) else t.encode() for t in tokens).decode("utf-8", errors=self.errors)

    def _decode(self, token_ids, skip_special_tokens=False, errors=None, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)

    def save_vocabulary(self, save_directory, filename_prefix=None, **kwargs):
        name = (filename_prefix + "-" if filename_prefix else "") + "qwen.tiktoken"
        path = Path(save_directory) / name
        path.write_bytes(
            b"".join(
                base64.b64encode(token) + b" " + str(rank).encode() + b"\n"
                for token, rank in self.mergeable_ranks.items()
            )
        )
        return (str(path),)


def make_context(tokenizer, query, history=None, system="", max_window_size=6144, chat_format="chatml"):
    """Source ChatML truncation uses complete history turns and a strict window."""
    if chat_format == "raw":
        return query, tokenizer.encode(query)
    if chat_format != "chatml":
        raise ValueError(f"Unknown original Qwen chat format {chat_format!r}")
    nl = tokenizer.encode("\n")

    def segment(role, text):
        return (
            [tokenizer.im_start_id]
            + tokenizer.encode(role, allowed_special=set())
            + nl
            + tokenizer.encode(text, allowed_special=set())
            + [tokenizer.im_end_id]
        )

    prefix = segment("system", system)
    context, raw = [], ""
    for question, answer in reversed(history or []):
        turn = nl + segment("user", question) + nl + segment("assistant", answer)
        if len(prefix) + len(turn) + len(context) >= max_window_size:
            break
        context = turn + context
        raw = f"\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>" + raw
    tokens = (
        prefix
        + context
        + nl
        + segment("user", query)
        + nl
        + [tokenizer.im_start_id]
        + tokenizer.encode("assistant")
        + nl
    )
    raw = (
        f"<|im_start|>system\n{system}<|im_end|>"
        + raw
        + f"\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    )
    return raw, tokens
