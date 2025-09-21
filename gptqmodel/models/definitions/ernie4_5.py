# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from . import LlamaQModel


class Ernie4_5QModel(LlamaQModel):
    require_trust_remote_code = True
    support_batch_quantize = False
    require_monkeypatch = True

    def monkey_patch(self):
        from typing import Optional, Tuple

        import torch
        from transformers.modeling_outputs import BaseModelOutputWithPast

        def ernie4_5_decode_layer_forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                attn_mask_start_row_indices: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = False,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: Optional[bool] = False,
        ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
            residual = hidden_states

            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            (hidden_states, self_attn_weights, present_key_value) = self.self_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                attn_mask_start_row_indices=attn_mask_start_row_indices,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                token_type_ids=token_type_ids,
            )
            hidden_states = self.residual_add1(hidden_states, residual)

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)

            hidden_states = self.residual_add2(hidden_states, residual)
            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)

            if use_cache:
                outputs += (present_key_value,)

            return outputs

        def ernie4_5_model_forward(
                self,
                input_ids=None,
                position_ids=None,
                token_type_ids=None,
                attention_mask=None,
                attn_mask_start_row_indices=None,
                inputs_embeds=None,
                use_cache=None,
                past_key_values=None,
                output_attentions=False,
                output_hidden_states=None,
                return_dict=False,
        ):
            use_cache = use_cache if use_cache is not None else self.config.use_cache

            # retrieve input_ids and inputs_embeds
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
                )
            elif input_ids is not None:
                _, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                _, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError(
                    "You have to specify either decoder_input_ids or decoder_inputs_embeds"
                )

            if past_key_values is None:
                past_key_values = tuple([None] * len(self.layers))

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.to(self.embed_tokens.weight.dtype)

            hidden_states = inputs_embeds

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = () if use_cache else None

            for idx, (decoder_layer) in enumerate(self.layers):

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                past_key_value = (
                    past_key_values[idx] if past_key_values is not None else None
                )

                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    attn_mask_start_row_indices=attn_mask_start_row_indices,
                    position_ids=position_ids,
                    token_type_ids=token_type_ids,
                    output_attentions=output_attentions,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )

                if isinstance(layer_outputs, (tuple, list)):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                # apply kv cache
                if past_key_value is not None:
                    hidden_states = hidden_states[:, -1:, :]

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = next_decoder_cache if use_cache else None

            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        next_cache,
                        all_hidden_states,
                        all_self_attns,
                    ]
                    if v is not None
                )

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

        if not self.load_quantized_model:
            ernie4_5_model = type(self.model.model)
            ernie4_5_model.forward = ernie4_5_model_forward

            ernie4_5_layer = type(self.model.model.layers[0])
            ernie4_5_layer.forward = ernie4_5_decode_layer_forward
