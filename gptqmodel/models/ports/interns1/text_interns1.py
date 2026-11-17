# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3 MoE with the original Intern-S1 per-expert checkpoint layout."""

import torch
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP, Qwen3MoeModel


class InternS1Experts(nn.ModuleList):
    def __init__(self, config):
        super().__init__(Qwen3MoeMLP(config, config.moe_intermediate_size) for _ in range(config.num_experts))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        for index, expert in enumerate(self):
            tokens, slots = torch.where(top_k_index == index)
            values = expert(hidden_states[tokens]) * top_k_weights[tokens, slots, None]
            output.index_add_(0, tokens, values.to(hidden_states.dtype))
        return output


class InternS1MoeModel(Qwen3MoeModel):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            if hasattr(layer.mlp, "experts"):
                layer.mlp.experts = InternS1Experts(config)
        self.post_init()
