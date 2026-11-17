# SPDX-FileCopyrightText: 2025 Moonshot AI
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: LicenseRef-Kimi-Modified-MIT
"""Standalone Kimi-K2 tokenizer; pinned source and license in PROVENANCE.md / LICENSE-KIMI."""

import os
from hashlib import sha256
from logging import getLogger
from pathlib import Path
from shutil import copyfile
from typing import Any

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from tokenizers import AddedToken
from transformers.tokenization_python import PythonBackend

from .tokenization_utils import bytes_to_unicode


logger = getLogger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tiktoken.model"}


class TikTokenTokenizer(PythonBackend):
    """Keep K2's positional APIs; Transformers options are accepted by keyword."""
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    num_reserved_special_tokens = 256
    pat_str = "|".join([
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ])

    def __init__(
        self, vocab_file, bos_token: str | AddedToken = "[BOS]", eos_token: str | AddedToken = "[EOS]",
        unk_token=None, pad_token=None, additional_special_tokens=None, added_tokens_decoder=None, **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Kimi-K2 tokenizer vocabulary file is unavailable: {vocab_file}")
        if added_tokens_decoder is None:
            raise ValueError("Kimi-K2 requires its checkpoint added_tokens_decoder mapping")
        if additional_special_tokens is None:
            additional_special_tokens = [
                "<|im_end|>", "<|im_user|>", "<|im_assistant|>", "<|start_header_id|>",
                "<|end_header_id|>", "[EOT]", "<|im_system|>", "<|im_middle|>",
            ]
        special_tokens_mapping = {i: token.content for i, token in added_tokens_decoder.items()}
        self.vocab_file = vocab_file
        mergeable_ranks = load_tiktoken_bpe(vocab_file, expected_hash=sha256(Path(vocab_file).read_bytes()).hexdigest())
        num_base_tokens = len(mergeable_ranks)
        self.special_tokens = {
            special_tokens_mapping.get(i, f"<|reserved_token_{i}|>"): i
            for i in range(num_base_tokens, num_base_tokens + self.num_reserved_special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(vocab_file).name, pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens,
        )
        self.n_words = self.model.n_vocab
        self.bos_id = self.special_tokens[str(bos_token)]
        self.eos_id = self.special_tokens[str(eos_token)]
        self.pad_id = self.special_tokens[str(pad_token)]
        self.unk_id = self.special_tokens[str(unk_token)]
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        self.decoder = {
            i: "".join(self.byte_encoder[byte] for byte in self.model.decode_single_token_bytes(i))
            for i in range(self.n_words)
        }
        self.encoder = {value: key for key, value in self.decoder.items()}
        super().__init__(
            bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, pad_token=pad_token,
            additional_special_tokens=additional_special_tokens, added_tokens_decoder=added_tokens_decoder, **kwargs,
        )
        self.all_special_ids_set = set(self.all_special_ids)

    def encode(self, text: str, allow_special_tokens: bool = True, **kwargs) -> list[int]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if kwargs:
            return super().encode(text, **kwargs)
        assert type(text) is str
        tokens = []
        for segment in self.pre_tokenizer_process(text):
            for start in range(0, len(segment), 400_000):
                for chunk in self._split_whitespaces_or_nonwhitespaces(segment[start:start + 400_000], 25_000):
                    if allow_special_tokens:
                        tokens.extend(self.model.encode(chunk, allowed_special="all"))
                    else:
                        tokens.extend(self.model.encode(chunk, disallowed_special=()))
        return tokens

    def decode(self, token_ids, **kwargs) -> str | list[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if kwargs:
            return super().decode(token_ids, **kwargs)
        if type(token_ids) is int:
            token_ids = [token_ids]
        return self.model.decode(token_ids)

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(text: str, max_consecutive_slice_len: int):
        length = 0
        is_space = text[0].isspace() if text else False
        start = 0
        for i, char in enumerate(text):
            now_space = char.isspace()
            if is_space != now_space:
                length = 1
                is_space = now_space
            else:
                length += 1
                if length > max_consecutive_slice_len:
                    yield text[start:i]
                    start = i
                    length = 1
        yield text[start:]

    def pre_tokenizer_process(self, text: str) -> list[str]:
        return [text]

    @property
    def vocab_size(self) -> int:
        return self.n_words

    def get_vocab(self) -> dict[str, int]:
        return self.encoder

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return [self.decoder[token] for token in self.encode(text)]

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.unk_id)

    def _convert_id_to_token(self, index: int) -> str | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.decoder.get(index)

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return out_string

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return bytearray(self.byte_decoder[char] for char in "".join(tokens)).decode("utf-8", "replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        if not os.path.isdir(save_directory):
            raise ValueError(f"vocabulary path ({save_directory}) should be a directory")
        target = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "tiktoken.model")
        if os.path.abspath(self.vocab_file) != os.path.abspath(target) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, target)
        return (target,)

    def apply_chat_template(self, conversation, tools=None, tokenize=False, add_generation_prompt=True, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().apply_chat_template(
            conversation, tools=deep_sort_dict(tools), tokenize=tokenize,
            add_generation_prompt=add_generation_prompt, **kwargs,
        )


def deep_sort_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: deep_sort_dict(value) for key, value in sorted(obj.items())}
    if isinstance(obj, list):
        return [deep_sort_dict(item) for item in obj]
    return obj
