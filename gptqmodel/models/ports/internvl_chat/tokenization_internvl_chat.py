# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-data-only tokenizer selection for the legacy InternVL wrapper."""

import json

from transformers.utils.hub import cached_file

from .configuration_internvl_chat import InternVLChatConfig


_HUB_KEYS = {"revision", "cache_dir", "subfolder", "token", "local_files_only", "force_download", "proxies", "_commit_hash"}
_TOKENIZER_NAMES = {
    "InternLM2ForCausalLM": {"InternLM2Tokenizer"},
    "LlamaForCausalLM": {"LlamaTokenizer", "LlamaTokenizerFast"},
    "Phi3ForCausalLM": {"LlamaTokenizer", "LlamaTokenizerFast"},
    "Qwen2ForCausalLM": {"Qwen2Tokenizer", "Qwen2TokenizerFast"},
    "Qwen3ForCausalLM": {"Qwen2Tokenizer", "Qwen2TokenizerFast"},
    "Qwen3MoeForCausalLM": {"Qwen2Tokenizer", "Qwen2TokenizerFast"},
}


def _asset(path, name, hub_kwargs, *, required=False):
    resolved = cached_file(path, name, _raise_exceptions_for_missing_entries=False, **hub_kwargs)
    if required and resolved is None:
        raise FileNotFoundError(f"InternVL tokenizer requires {name} in {path}")
    return resolved


class InternVLTokenizerFactory:
    """Reuse InternLM2's Python tokenizer or an exact native sibling class.

    No checkpoint class/import path is used. Generic normalization is owned by
    Tokenicer at LocalModelLoader's boundary and after this factory returns.
    """

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        config = kwargs.pop("config", None)
        model_config = kwargs.pop("model_config", None)
        config = config if config is not None else model_config
        hub_kwargs = {key: value for key, value in kwargs.items() if key in _HUB_KEYS}
        if config is None:
            config = InternVLChatConfig.from_pretrained(path, **hub_kwargs)
        if not isinstance(config, InternVLChatConfig):
            raise ValueError("InternVL tokenizer requires a package-owned InternVLChatConfig")
        config = InternVLChatConfig(**config.to_dict())
        architecture = config.llm_config.architectures[0]
        allowed = _TOKENIZER_NAMES[architecture]
        metadata_path = _asset(path, "tokenizer_config.json", hub_kwargs)
        if metadata_path is not None:
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            if not isinstance(metadata, dict):
                raise ValueError("InternVL tokenizer_config.json must contain an object")
            declared = metadata.get("tokenizer_class")
        else:
            declared = None
        for name in (declared, getattr(config, "tokenizer_class", None),
                     getattr(config.llm_config, "tokenizer_class", None)):
            if name is not None and (not isinstance(name, str) or name not in allowed):
                raise ValueError(f"InternVL {architecture} does not support tokenizer_class={name!r}")

        if architecture == "InternLM2ForCausalLM":
            _asset(path, "tokenizer.model", hub_kwargs, required=True)
            from ..internlm2.tokenization_internlm2 import InternLM2Tokenizer

            tokenizer_class = InternLM2Tokenizer
        elif architecture in {"LlamaForCausalLM", "Phi3ForCausalLM"}:
            if _asset(path, "tokenizer.json", hub_kwargs) is None:
                _asset(path, "tokenizer.model", hub_kwargs, required=True)
            # load_hf_tokenizer registers the vendored class with Tokenicer as
            # Transformers' LlamaTokenizer, so normalization is unchanged.
            from ..native.llama.tokenization_llama import LlamaTokenizer

            tokenizer_class = LlamaTokenizer
        else:
            if _asset(path, "tokenizer.json", hub_kwargs) is None:
                _asset(path, "vocab.json", hub_kwargs, required=True)
                _asset(path, "merges.txt", hub_kwargs, required=True)
            from ..native.qwen2.tokenization_qwen2 import Qwen2Tokenizer

            tokenizer_class = Qwen2Tokenizer
        return tokenizer_class.from_pretrained(path, trust_remote_code=False, **kwargs)
