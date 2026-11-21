# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Select a packaged tokenizer from released GLM tokenizer metadata only."""

import json

from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
from transformers.utils.hub import cached_file, resolve_revision


_HUB_KWARGS = ("cache_dir", "force_download", "proxies", "token", "local_files_only", "subfolder")


def get_tokenizer_config(path, revision=None, **kwargs):
    """Read ``tokenizer_config.json`` ({} when absent), like Transformers' AutoTokenizer helper."""

    selection = {key: kwargs[key] for key in _HUB_KWARGS if key in kwargs}
    revision = resolve_revision(
        path,
        revision,
        token=selection.get("token"),
        local_files_only=selection.get("local_files_only", False),
        cache_dir=selection.get("cache_dir"),
    )
    resolved = cached_file(
        path,
        TOKENIZER_CONFIG_FILE,
        revision=revision,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        **selection,
    )
    if resolved is None:
        return {}
    with open(resolved, encoding="utf-8") as reader:
        return json.load(reader)


class ChatGLMTokenizer:
    """Definition-owned tokenizer dispatcher, with no checkpoint Python lookup.

    GLM2 and GLM3 published the same tokenizer_class name but different role
    vocabularies. Their released metadata distinguishes those contracts; saved
    packaged tokenizers also carry their concrete class name.
    """

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        metadata = get_tokenizer_config(path, **kwargs)
        name = metadata.get("tokenizer_class")
        if name == "ChatGLMTokenizer":
            role_tokens = {"<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"}
            has_roles = any(token.get("content") in role_tokens for token in
                            metadata.get("added_tokens_decoder", {}).values() if isinstance(token, dict))
            name = "ChatGLM3Tokenizer" if has_roles or metadata.get("chat_template") else "ChatGLM2Tokenizer"
        if name == "ChatGLM2Tokenizer":
            from .tokenization_glm2 import ChatGLM2Tokenizer
            tokenizer_class = ChatGLM2Tokenizer
        elif name == "ChatGLM3Tokenizer":
            from .tokenization_glm3 import ChatGLM3Tokenizer
            tokenizer_class = ChatGLM3Tokenizer
        elif name == "ChatGLM4Tokenizer":
            from .tokenization_glm4 import ChatGLM4Tokenizer
            tokenizer_class = ChatGLM4Tokenizer
        else:
            raise ValueError(f"Unsupported ChatGLM tokenizer metadata: {name!r}; checkpoint Python is disabled")
        return tokenizer_class.from_pretrained(path, *args, **kwargs)
