# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Select the standalone Kimi-K2 tokenizer without changing native DeepSeek."""

import json
from importlib import import_module
from pathlib import Path

from transformers import PreTrainedConfig, TokenizersBackend
from transformers.utils.hub import cached_file

from ..native.auto import _registry
from .configuration_deepseek_v3 import DeepseekV3Config


# Declared names that mean "the framework's serialized tokenizers backend".
_FRAMEWORK_BACKENDS = {None, "PreTrainedTokenizerFast", "TokenizersBackend"}


_DOWNLOAD_KEYS = {"cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder", "_commit_hash"}


def _download_kwargs(kwargs):
    return {key: value for key, value in kwargs.items() if key in _DOWNLOAD_KEYS}


def _tokenizer_metadata(model_id_or_path, kwargs):
    path = cached_file(
        model_id_or_path, "tokenizer_config.json", _raise_exceptions_for_missing_entries=False,
        **_download_kwargs(kwargs),
    )
    return {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))


def _vendored_tokenizer_class(declared_class):
    """Resolve a declared tokenizer class against GPT-QModel's vendored registry only."""

    names = {declared_class, declared_class.removesuffix("Fast")}
    for module, name in _registry.TOKENIZER.values():
        if name in names:
            return getattr(import_module(module), name)
    raise ValueError(
        f"Native DeepSeek tokenizer class {declared_class!r} is not available natively; "
        "the owned Kimi tokenizer requires model_type/source_model_type='kimi_k2'."
    )


class DeepseekV3Tokenizer:
    @classmethod
    def from_pretrained(cls, model_id_or_path, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        kwargs.pop("config", None)
        payload, _ = PreTrainedConfig.get_config_dict(model_id_or_path, **kwargs)
        if "kimi_k2" in (payload.get("model_type"), payload.get("source_model_type")):
            from .tokenization_kimi import TikTokenTokenizer

            return TikTokenTokenizer.from_pretrained(model_id_or_path, **kwargs)
        config = DeepseekV3Config.from_dict(payload)
        metadata = _tokenizer_metadata(model_id_or_path, kwargs)
        declared_class = metadata.get("tokenizer_class") or getattr(config, "tokenizer_class", None)
        if declared_class == "TikTokenTokenizer" and cached_file(
            model_id_or_path, "tiktoken.model", _raise_exceptions_for_missing_entries=False,
            **_download_kwargs(kwargs),
        ) is not None:
            # Moonlight and Kimi-K2 ship the same byte-identical tiktoken.model
            # and pretokenizer regex; the package-owned class reads each
            # checkpoint's own added_tokens_decoder mapping.
            from .tokenization_kimi import TikTokenTokenizer

            return TikTokenTokenizer.from_pretrained(model_id_or_path, **kwargs)
        tokenizer_class = TokenizersBackend
        if declared_class not in _FRAMEWORK_BACKENDS:
            tokenizer_class = _vendored_tokenizer_class(declared_class)
        if declared_class in {None, "LlamaTokenizer", "LlamaTokenizerFast", "PreTrainedTokenizerFast", "TokenizersBackend"}:
            serialized = kwargs.get("tokenizer_file") or cached_file(
                model_id_or_path, "tokenizer.json", _raise_exceptions_for_missing_entries=False,
                **_download_kwargs(kwargs),
            )
            if serialized is not None:
                specification = json.loads(Path(serialized).read_text(encoding="utf-8"))
                pending = [specification.get("pre_tokenizer") or {}]
                while pending:
                    component = pending.pop()
                    if component.get("type") == "ByteLevel" and specification.get("model", {}).get("type") == "BPE":
                        return TokenizersBackend.from_pretrained(model_id_or_path, **kwargs)
                    if component.get("type") == "Sequence":
                        pending.extend(component.get("pretokenizers", []))
        if metadata.get("auto_map"):
            raise ValueError(
                "Unqualified native DeepSeek tokenizer metadata: checkpoint AutoTokenizer code is disabled; "
                "only the native serialized ByteLevel BPE layout is qualified with tokenizer auto_map."
            )
        return tokenizer_class.from_pretrained(model_id_or_path, **kwargs)
