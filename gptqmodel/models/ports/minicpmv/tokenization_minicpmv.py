# SPDX-FileCopyrightText: 2024 OpenBMB
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Definition-owned binding for the inspected MiniCPM-V tokenizer wrappers.

The upstream wrappers add marker aliases only. Their native serialized/Sentence-
Piece algorithms and checkpoint assets are preserved; Tokenicer owns all outer
normalization. This is class ownership, not a custom-class recovery heuristic.
"""

import json
from pathlib import Path

from transformers import TokenizersBackend
from transformers.utils import cached_file

from ..native.llama.tokenization_llama import LlamaTokenizer
from .configuration_minicpm import MiniCPMVConfig


class MiniCPMVTokenizer:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        config = MiniCPMVConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        path = cached_file(pretrained_model_name_or_path, "tokenizer_config.json", **kwargs)
        declared = json.loads(Path(path).read_text()).get("tokenizer_class")
        allowed = {"MiniCPMVTokenizerFast", "LlamaTokenizerWrapper", "PreTrainedTokenizerFast",
                   "TokenizersBackend", "TokenizersBackendFast", "LlamaTokenizer", "LlamaTokenizerFast", "Qwen2Tokenizer",
                   "Qwen2TokenizerFast", "MiniCPMVTokenizer"}
        if declared not in allowed:
            raise ValueError(f"Unqualified MiniCPM-V tokenizer class {declared!r}")
        asset = cached_file(pretrained_model_name_or_path, "tokenizer.json",
                            _raise_exceptions_for_missing_entries=False, **kwargs)
        if config.version < 2.5 and asset is None:
            # The original LlamaTokenizerWrapper adds only marker properties.
            # Native LlamaTokenizer owns the exact SentencePiece implementation.
            return LlamaTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # All four modern wrappers have an authoritative serialized backend;
        # bind it directly so no checkpoint class or native-family fallback can
        # replace its pre-tokenizer/model/post-processor.
        if not asset:
            raise OSError("MiniCPM-V requires tokenizer.json")
        return TokenizersBackend.from_pretrained(pretrained_model_name_or_path, **kwargs)
