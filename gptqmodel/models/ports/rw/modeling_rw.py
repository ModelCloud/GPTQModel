# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Use native Falcon, correcting only its eager ALiBi double-bias boundary."""

from transformers.models.falcon.modeling_falcon import FalconAttention, FalconForCausalLM


class RWAlibiAttention(FalconAttention):
    def forward(
        self, hidden_states, alibi, attention_mask, position_ids=None,
        layer_past=None, use_cache=False, output_attentions=False,
        position_embeddings=None, **kwargs,
    ):
        if alibi is not None and (
            self.config._attn_implementation != "sdpa" or output_attentions
        ):
            # FalconModel has already inserted this bias for SDPA; native eager
            # attention inserts it itself. Undo only that first insertion.
            bias = alibi.reshape(hidden_states.shape[0], self.num_heads, 1, -1) * self.inv_norm_factor
            attention_mask = attention_mask - bias
        return super().forward(
            hidden_states, alibi, attention_mask, position_ids=position_ids,
            layer_past=layer_past, use_cache=use_cache,
            output_attentions=output_attentions, position_embeddings=position_embeddings,
            **kwargs,
        )


class RWAlibiForCausalLM(FalconForCausalLM):
    """Native decoder/cache/generation and parameters, with one attention fix."""

    def __init__(self, config):
        super().__init__(config)
        for index, layer in enumerate(self.transformer.h):
            original = layer.self_attention
            if type(original) is not FalconAttention:
                # Preserve native FlashAttention selection and its own support
                # checks; the measured defect is in FalconAttention only.
                continue
            corrected = RWAlibiAttention(config, layer_idx=index)
            # Retain the actual initialized/meta native parameters and their
            # loader initialization markers; no state-dict conversion exists.
            corrected.query_key_value = original.query_key_value
            corrected.dense = original.dense
            layer.self_attention = corrected


class RWModelLoader:
    """Definition-local selection; RoPE models retain the exact native class."""

    @staticmethod
    def _from_config(config, **kwargs):
        model_class = RWAlibiForCausalLM if config.alibi else FalconForCausalLM
        return model_class._from_config(config, **kwargs)

    @staticmethod
    def from_pretrained(model_id_or_path, *, config, **kwargs):
        model_class = RWAlibiForCausalLM if config.alibi else FalconForCausalLM
        return model_class.from_pretrained(model_id_or_path, config=config, **kwargs)
