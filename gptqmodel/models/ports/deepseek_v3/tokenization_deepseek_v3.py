# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Select the standalone Kimi-K2 tokenizer without changing native DeepSeek."""

import json
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedConfig, TokenizersBackend
from transformers.models.auto.tokenization_auto import get_tokenizer_config, tokenizer_class_from_name
from transformers.utils.hub import cached_file

from .configuration_deepseek_v3 import DeepseekV3Config


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
        metadata = get_tokenizer_config(model_id_or_path, **kwargs)
        declared_class = metadata.get("tokenizer_class") or getattr(config, "tokenizer_class", None)
        native_class = tokenizer_class_from_name(declared_class) if declared_class else None
        if declared_class and (native_class is None or not native_class.__module__.startswith("transformers.")):
            raise ValueError(
                f"Native DeepSeek tokenizer class {declared_class!r} is not available natively; "
                "the owned Kimi tokenizer requires model_type/source_model_type='kimi_k2'."
            )
        if declared_class in {None, "LlamaTokenizer", "LlamaTokenizerFast", "PreTrainedTokenizerFast", "TokenizersBackend"}:
            download_kwargs = {key: value for key, value in kwargs.items() if key in {
                "cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder", "_commit_hash",
            }}
            serialized = kwargs.get("tokenizer_file") or cached_file(
                model_id_or_path, "tokenizer.json", _raise_exceptions_for_missing_entries=False, **download_kwargs,
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
        return AutoTokenizer.from_pretrained(
            model_id_or_path, config=config, trust_remote_code=False, **kwargs,
        )
