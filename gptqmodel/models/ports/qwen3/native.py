# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native class aliases, imported only when the definition's loader is used."""

from transformers import PreTrainedConfig, Qwen2Tokenizer, Qwen3Config, Qwen3ForCausalLM, TokenizersBackend
from transformers.utils.hub import cached_file


class NativeQwen3Config:
    """Read data only, then construct an exact native config without AutoConfig.

    The supported Marin-32B release declares ``model_type=qwen3``. There is no
    inspected source for a distinct ``marin`` schema; a registry alias alone
    cannot authorize interpreting an unknown checkpoint as Qwen3.
    """

    @classmethod
    def from_pretrained(cls, model_id_or_path, **kwargs):
        payload, kwargs = PreTrainedConfig.get_config_dict(model_id_or_path, **kwargs)
        if payload.get("model_type") != "qwen3":
            raise ValueError(
                "The supported Marin-32B checkpoint declares model_type=qwen3; "
                "raw marin metadata has no verified native layout. Use the original "
                "release config rather than relabeling an unknown architecture."
            )
        return Qwen3Config.from_dict(payload, **kwargs)


class Qwen3Tokenizer:
    """Keep the serialized pipeline or use native Qwen BPE for legacy assets."""

    @classmethod
    def from_pretrained(cls, model_id_or_path, **kwargs):
        from ....utils.hf import get_config_selection_kwargs

        tokenizer_file = kwargs.get("tokenizer_file")
        if tokenizer_file is None:
            tokenizer_file = cached_file(
                model_id_or_path,
                "tokenizer.json",
                _raise_exceptions_for_missing_entries=False,
                **get_config_selection_kwargs(kwargs),
            )
        tokenizer_class = TokenizersBackend if tokenizer_file is not None else Qwen2Tokenizer
        return tokenizer_class.from_pretrained(model_id_or_path, **kwargs)


__all__ = ["NativeQwen3Config", "Qwen3ForCausalLM", "Qwen3Tokenizer"]
