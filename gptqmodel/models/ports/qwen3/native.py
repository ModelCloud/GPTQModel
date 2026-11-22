# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Vendored Transformers 5.23 Qwen3 class aliases, imported only when the definition's loader is used."""

import json

from transformers.configuration_utils import PreTrainedConfig
from transformers.tokenization_utils_tokenizers import TokenizersBackend
from transformers.utils import cached_file

from ..native.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from ..native.qwen3.configuration_qwen3 import Qwen3Config
from ..native.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def _tokenizer_config(model_id_or_path, **kwargs):
    """Read ``tokenizer_config.json`` (or ``{}``) the way Transformers' ``get_tokenizer_config`` does."""

    resolved = cached_file(
        model_id_or_path,
        "tokenizer_config.json",
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        **kwargs,
    )
    if resolved is None:
        return {}
    with open(resolved, encoding="utf-8") as reader:
        return json.load(reader)


class NativeQwen3Config:
    """Read data only, then construct the vendored native config without AutoConfig.

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

        declared = _tokenizer_config(model_id_or_path, **get_config_selection_kwargs(kwargs)).get("tokenizer_class")
        if declared is None or declared.removesuffix("Fast") == "Qwen2Tokenizer":
            tokenizer_class = Qwen2Tokenizer
        else:
            tokenizer_class = TokenizersBackend
        return tokenizer_class.from_pretrained(model_id_or_path, **kwargs)


__all__ = ["NativeQwen3Config", "Qwen3ForCausalLM", "Qwen3Tokenizer"]
