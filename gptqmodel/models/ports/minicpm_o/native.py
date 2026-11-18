# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native decoders with instance-owned caches for the legacy Omni streaming API."""

from transformers import LlamaModel as NativeLlamaModel
from transformers import Qwen2ForCausalLM as NativeQwen2ForCausalLM
from transformers import Qwen3ForCausalLM as NativeQwen3ForCausalLM

from .cache import DynamicCache


class _LocalCache:
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        if use_cache if use_cache is not None else self.config.use_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )


class Qwen3ForCausalLM(_LocalCache, NativeQwen3ForCausalLM):
    pass


class Qwen2ForCausalLM(_LocalCache, NativeQwen2ForCausalLM):
    pass


class LlamaModel(_LocalCache, NativeLlamaModel):
    pass
