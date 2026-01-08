# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class Ernie4_5_MoeQModel(BaseQModel):
    require_trust_remote_code = True
    support_batch_quantize = False
    require_monkeypatch = True
    layer_modules_strict = False

    dynamic_expert_index = "moe_num_experts"

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "gate_proj": ("gate_proj",),
                "up_proj": ("up_proj",),
                "down_proj": ("down_proj",),
                "gate": ("gate:!",),
                "shared_experts": {
                    "gate_proj": ("gate_proj:0",),
                    "up_proj": ("up_proj:0",),
                    "down_proj": ("down_proj:1",),
                },
                "experts": {
                    "#": ("gate_proj:0", "upe_proj:0", "down_proj:1"),
                },
            },
        }
    ]

    def monkey_patch(self):
        from dataclasses import dataclass
        from functools import partial
        from typing import Optional

        import torch
        from transformers.cache_utils import Cache, DynamicCache
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
        from transformers.modeling_outputs import ModelOutput
        from transformers.processing_utils import Unpack

        @dataclass
        class Erine4_5_MoeModelOutputWithPast(ModelOutput):
            last_hidden_state: Optional[torch.FloatTensor] = None
            past_key_values: Optional[Cache] = None
            hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
            attentions: Optional[tuple[torch.FloatTensor, ...]] = None
            router_loss: Optional[torch.FloatTensor] = None
            gate_logits: Optional[tuple[torch.FloatTensor, ...]] = None

        def ernie4_5_moe_model_forward(
                self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
        ):
            """Forward pass through the ERNIE model."""
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    use_cache = False

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            inputs_embeds = inputs_embeds.to(self.embed_tokens.weight.dtype)

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )

            hidden_states = inputs_embeds

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            all_router_loss = torch.tensor(0.0, device=inputs_embeds.device) if self.config.use_moe else None
            all_gate_logits = ()

            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        partial(decoder_layer.__call__, **flash_attn_kwargs),
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                if self.config.use_moe:
                    layer_outputs, gate_logits = layer_outputs[:-1], layer_outputs[-1]
                    all_gate_logits = all_gate_logits + (gate_logits,)

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # assert all_router_loss is None, f'moe not support `return-dict`'
            return Erine4_5_MoeModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                router_loss=all_router_loss,
                gate_logits=all_gate_logits,
            )

        if not self.load_quantized_model:
            ernie4_5_model = type(self.model.model)
            ernie4_5_model.forward = ernie4_5_moe_model_forward
