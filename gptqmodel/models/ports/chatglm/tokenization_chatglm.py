# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Select a packaged tokenizer from released GLM tokenizer metadata only."""

from transformers.models.auto.tokenization_auto import get_tokenizer_config


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
