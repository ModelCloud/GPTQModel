# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native class aliases, imported only when the definition's loader is used."""

from transformers import PreTrainedConfig, Qwen3Config, Qwen3ForCausalLM
from transformers import TokenizersBackend as Qwen3Tokenizer


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


__all__ = ["NativeQwen3Config", "Qwen3ForCausalLM", "Qwen3Tokenizer"]
