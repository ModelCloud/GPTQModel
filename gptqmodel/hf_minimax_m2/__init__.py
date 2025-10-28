# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# """MiniMax M2 Hugging Face remote code support."""

from .configuration_minimax_m2 import MiniMaxM2Config
from .modeling_minimax_m2 import (
    MiniMaxForCausalLM,
    MiniMaxM2ForCausalLM,
    MiniMaxM2Model,
    MiniMaxM2PreTrainedModel,
    MiniMaxModel,
    MiniMaxPreTrainedModel,
)

__all__ = [
    "MiniMaxM2Config",
    "MiniMaxM2PreTrainedModel",
    "MiniMaxM2Model",
    "MiniMaxM2ForCausalLM",
    "MiniMaxPreTrainedModel",
    "MiniMaxModel",
    "MiniMaxForCausalLM",
]
