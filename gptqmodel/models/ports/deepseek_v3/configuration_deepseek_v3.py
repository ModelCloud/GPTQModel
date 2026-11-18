# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Canonicalize the standalone Kimi-K2 alias before native weight conversion."""

from ..native.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config as NativeDeepseekV3Config


class DeepseekV3Config(NativeDeepseekV3Config):
    def __init__(self, **kwargs):
        source_model_type = kwargs.pop("model_type", self.model_type)
        if source_model_type == "kimi_k2":
            # Transformers' checkpoint conversion and Defuser both key on model_type.
            kwargs["source_model_type"] = source_model_type
        super().__init__(**kwargs)
        validate_kimi_k2_config(self)


def validate_kimi_k2_config(config):
    source_model_type = getattr(config, "source_model_type", config.model_type)
    if source_model_type == "kimi_k2" and config.n_group != config.topk_group:
        raise ValueError(
            "Kimi-K2 partial-group routing is not qualified for native DeepSeek-V3: "
            "n_group must equal topk_group. The published Kimi-K2 checkpoint uses both equal to 1; "
            "its masked-score convention differs from native DeepSeek-V3 when groups are excluded."
        )
