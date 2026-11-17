# SPDX-FileCopyrightText: 2023 DeepSeek-AI and The HuggingFace Inc. team
# SPDX-FileCopyrightText: 2025 HuggingFace Inc. team
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native Transformers 5 MLA/cache/generation with legacy V2 layout semantics.

Adapted from the pinned Apache-2.0 sources documented in README.md. No
checkpoint Python or global Transformers registration is used.
"""

from contextlib import contextmanager
from functools import wraps

import torch
from torch import nn
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2Experts,
    DeepseekV2MLP,
    DeepseekV2PreTrainedModel,
    DeepseekV2RMSNorm,
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
    DeepseekV2RotaryEmbedding as NativeDeepseekV2RotaryEmbedding,
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
        if not (self.top_k > 1 and self.norm_topk_prob):
            return super().forward(hidden_states)

        # Pinned legacy MoEGate.forward normalizes BEFORE (and instead of)
        # scaling. Native scale-then-divide loses finite FP32 values at extreme
        # Python-finite factors. Keep native topk/group-mask/tie semantics.
        logits = nn.functional.linear(hidden_states.reshape(-1, self.hidden_dim).float(), self.weight.float())
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        if self.topk_method == "group_limited_greedy":
            group_scores = scores.view(-1, self.num_group, self.num_experts // self.num_group).max(dim=-1).values
            group_indices = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_indices, 1)
            mask = group_mask.unsqueeze(-1).expand(-1, self.num_group, self.num_experts // self.num_group)
            scores = scores.masked_fill(~mask.reshape(-1, self.num_experts).bool(), 0.0)
        weights, indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        return logits, weights / (weights.sum(dim=-1, keepdim=True) + 1e-20), indices


class DeepseekV2RotaryEmbedding(NativeDeepseekV2RotaryEmbedding):
    def _refresh_frequencies(self):
        # These are derived, nonpersistent FP32 tables, not learned tensors.
        # Recompute from the native recipe rather than widening rounded values.
        device = self.inv_freq.device
        init_fn = self.compute_default_rope_parameters if self.rope_type == "default" else ROPE_INIT_FUNCTIONS[self.rope_type]
        with torch.device(device):
            original, scaling = init_fn(self.config, device=device)
            current = original
            if "dynamic" in self.rope_type:
                seq_len = self.max_seq_len_cached
                if isinstance(seq_len, torch.Tensor):
                    seq_len = seq_len.to(device)
                current, scaling = init_fn(self.config, device=device, seq_len=seq_len)
        self.register_buffer("inv_freq", current.to(dtype=torch.float32), persistent=False)
        self.register_buffer("original_inv_freq", original.to(dtype=torch.float32), persistent=False)
        self.attention_scaling = scaling
        # Native longrope lazily rebuilds this plain attribute on the next call.
        if hasattr(self, "long_inv_freq"):
            self.long_inv_freq = None

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._refresh_frequencies()
        return result


@contextmanager
def _cache_transaction(cache, num_layers):
    """Keep native in-place success and restore the caller's state on failure.

    Qualified DynamicLayer.update appends with torch.cat, replacing rather than
    modifying prefix tensors. Retain their references and the complete layer
    metadata, including lazy-initialization state. Rollback also removes layers
    appended to an empty lazy cache, retaining the original list/layer objects.
    The CausalLM boundary covers head, loss and native tuple conversion failures;
    the decoder boundary also protects direct DeepseekV2Model callers.
    """
    if cache is None:
        yield
        return
    if (
        type(cache) is not DynamicCache or cache.offloading
        or cache.layer_class_to_replicate not in (None, DynamicLayer)
        or any(type(layer) is not DynamicLayer for layer in cache.layers)
    ):
        raise TypeError("DeepSeek V2 cache must be a non-offloaded native DynamicCache with DynamicLayer layers")
    lengths = [layer.get_seq_length() for layer in cache.layers]
    if len(cache.layers) not in (0, num_layers) or len(set(lengths)) > 1:
        raise ValueError("DeepSeek V2 cache must have a complete, uniform decoder prefix")
    layers = cache.layers
    states = [(layer, vars(layer).copy()) for layer in layers]
    try:
        yield
    except BaseException:
        for layer, state in states:
            vars(layer).clear()
            vars(layer).update(state)
        layers[:] = [layer for layer, _ in states]
        cache.layers = layers
        raise


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
    # Cache rollback uses eager Python metadata snapshots; do not inherit the
    # native fullgraph/static-cache generation capability advertisement.
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, DeepseekV2RotaryEmbedding):
            # HF's meta/low-dtype materialization may allocate nonpersistent
            # buffers in the parameter dtype before initializing them.
            module._refresh_frequencies()

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

    @wraps(NativeDeepseekV2Model.forward)
    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs,
    ):
        with _cache_transaction(past_key_values, self.config.num_hidden_layers):
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
            )


class DeepseekV2ForCausalLM(_NativeCheckpointLayout, NativeDeepseekV2ForCausalLM):
    config_class = DeepseekV2Config

    def __init__(self, config):
        DeepseekV2PreTrainedModel.__init__(self, config)
        self.model = DeepseekV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @wraps(NativeDeepseekV2ForCausalLM.forward)
    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
        logits_to_keep=0, **kwargs,
    ):
        with _cache_transaction(past_key_values, self.config.num_hidden_layers):
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                labels=labels, use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs,
            )
