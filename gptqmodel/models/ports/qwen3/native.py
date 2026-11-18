# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native class aliases, imported only when the definition's loader is used."""

from transformers import PreTrainedConfig, Qwen2Tokenizer, Qwen3Config, Qwen3ForCausalLM, TokenizersBackend
from transformers.models.auto.tokenization_auto import get_tokenizer_config


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
    """Select the tokenizer class the way TF5 ``AutoTokenizer`` does for ``qwen3``.

    Qwen3 releases declare ``Qwen2Tokenizer`` (also TF5's registered class for
    ``qwen3``), which rebuilds the BPE pipeline from vocab/merges. Releases that
    declare a generic fast tokenizer (e.g. Marin's ``PreTrainedTokenizerFast``)
    keep the serialized ``tokenizer.json`` via ``TokenizersBackend``.
    """

    @classmethod
    def from_pretrained(cls, model_id_or_path, **kwargs):
        from ....utils.hf import get_config_selection_kwargs

        declared = get_tokenizer_config(model_id_or_path, **get_config_selection_kwargs(kwargs)).get("tokenizer_class")
        if declared is None or declared.removesuffix("Fast") == "Qwen2Tokenizer":
            tokenizer_class = Qwen2Tokenizer
        else:
            tokenizer_class = TokenizersBackend
        return tokenizer_class.from_pretrained(model_id_or_path, **kwargs)


__all__ = ["NativeQwen3Config", "Qwen3ForCausalLM", "Qwen3Tokenizer"]
