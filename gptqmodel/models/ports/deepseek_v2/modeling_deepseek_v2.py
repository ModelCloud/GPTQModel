# SPDX-FileCopyrightText: 2023 DeepSeek-AI and The HuggingFace Inc. team
# SPDX-FileCopyrightText: 2025 HuggingFace Inc. team
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native Transformers 5 MLA/cache/generation with legacy V2 layout semantics.

Adapted from the pinned Apache-2.0 sources documented in README.md. No
checkpoint Python or global Transformers registration is used.
"""

from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2Experts,
    DeepseekV2MLP,
    DeepseekV2PreTrainedModel,
    DeepseekV2RMSNorm,
    DeepseekV2RotaryEmbedding,
)
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2DecoderLayer as NativeDeepseekV2DecoderLayer,
)
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2ForCausalLM as NativeDeepseekV2ForCausalLM,
)
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Model as NativeDeepseekV2Model,
)
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Moe as NativeDeepseekV2Moe,
)
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2TopkRouter as NativeDeepseekV2TopkRouter,
)

from .configuration_deepseek_v2 import DeepseekV2Config


class DeepseekV2TopkRouter(NativeDeepseekV2TopkRouter):
    def __init__(self, config):
        super().__init__(config)
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states):
        logits, weights, indices = super().forward(hidden_states)
        if self.top_k > 1 and self.norm_topk_prob:
            # The legacy normalized branch does not apply routed_scaling_factor.
            # Undo native scaling before its epsilon-stabilized normalization.
            weights = weights / self.routed_scaling_factor
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights, indices


class DeepseekV2Moe(NativeDeepseekV2Moe):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.experts = DeepseekV2Experts(config)
        self.gate = DeepseekV2TopkRouter(config)
        self.shared_experts = (
            DeepseekV2MLP(config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)
            if config.n_shared_experts is not None else None
        )

    def forward(self, hidden_states):
        if self.shared_experts is not None:
            return super().forward(hidden_states)
        shape = hidden_states.shape
        _, weights, indices = self.gate(hidden_states)
        return self.experts(hidden_states.reshape(-1, shape[-1]), indices, weights).view(shape)


class DeepseekV2DecoderLayer(NativeDeepseekV2DecoderLayer):
    def __init__(self, config, layer_idx):
        # Retain the native decoder forward, including residual and cache order.
        GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = DeepseekV2Attention(config, layer_idx)
        is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        self.mlp = DeepseekV2Moe(config) if is_moe else DeepseekV2MLP(config)
        self.input_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class _NativeCheckpointLayout:
    @classmethod
    def is_custom_code(cls):
        # Transformers otherwise skips its native checkpoint conversion map for
        # every class outside transformers.*. These subclasses intentionally
        # retain the DeepSeek V2 tensor ABI, including fused expert conversion.
        # This only declares serialization layout; it does not register classes
        # or authorize checkpoint code. LocalModelLoader owns class selection.
        return False


class DeepseekV2Model(_NativeCheckpointLayout, NativeDeepseekV2Model):
    config_class = DeepseekV2Config

    def __init__(self, config):
        DeepseekV2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(DeepseekV2DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class DeepseekV2ForCausalLM(_NativeCheckpointLayout, NativeDeepseekV2ForCausalLM):
    config_class = DeepseekV2Config

    def __init__(self, config):
        DeepseekV2PreTrainedModel.__init__(self, config)
        self.model = DeepseekV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
