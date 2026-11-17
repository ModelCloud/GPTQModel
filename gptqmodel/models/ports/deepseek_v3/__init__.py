# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Definition-owned native DeepSeek-V3 and standalone Kimi-K2 loading."""

from .configuration_deepseek_v3 import DeepseekV3Config
from .modeling_deepseek_v3 import DeepseekV3ForCausalLM


__all__ = ["DeepseekV3Config", "DeepseekV3ForCausalLM"]
