# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native computation with a construction-time standalone Kimi-K2 contract."""

from transformers import DeepseekV3ForCausalLM as NativeDeepseekV3ForCausalLM

from .configuration_deepseek_v3 import DeepseekV3Config, validate_kimi_k2_config


class DeepseekV3ForCausalLM(NativeDeepseekV3ForCausalLM):
    config_class = DeepseekV3Config

    def __init__(self, config):
        validate_kimi_k2_config(config)
        super().__init__(config)
