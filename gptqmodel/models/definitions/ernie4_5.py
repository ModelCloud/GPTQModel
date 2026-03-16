# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ...utils.logger import setup_logger
from . import LlamaQModel


log = setup_logger()


class Ernie4_5QModel(LlamaQModel):
    require_trust_remote_code = True
    support_batch_quantize = False
    require_monkeypatch = True
    _NATIVE_MODULE_PREFIX = "transformers.models.ernie4_5."

    @classmethod
    def _is_native_ernie4_5_module(cls, obj) -> bool:
        module_name = getattr(type(obj), "__module__", "")
        return module_name.startswith(cls._NATIVE_MODULE_PREFIX)

    def _should_apply_native_monkeypatch(self) -> bool:
        if self.load_quantized_model:
            return False

        model = getattr(getattr(self, "model", None), "model", None)
        if model is None:
            return False

        layers = getattr(model, "layers", None)
        first_layer = layers[0] if layers else None
        if first_layer is None:
            return False

        if self._is_native_ernie4_5_module(model) and self._is_native_ernie4_5_module(first_layer):
            return True

        log.info(
            "ERNIE 4.5: skipping native monkeypatch for non-native model classes `%s` / `%s`.",
            getattr(type(model), "__module__", "unknown"),
            getattr(type(first_layer), "__module__", "unknown"),
        )
        return False

    def monkey_patch(self):
        from typing import Optional

        import torch
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.models.ernie4_5.modeling_ernie4_5 import create_causal_mask

        def ernie4_5_decode_layer_forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                position_embeddings=None,
                output_attentions: Optional[bool] = False,
                past_key_values=None,
                use_cache: Optional[bool] = False,
                **kwargs,
        ):
            residual_add1 = getattr(self, "residual_add1", None)
            residual_add2 = getattr(self, "residual_add2", None)
            residual = hidden_states

            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                **kwargs,
            )
            hidden_states = residual_add1(hidden_states, residual) if residual_add1 is not None else residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)

            hidden_states = residual_add2(hidden_states, residual) if residual_add2 is not None else residual + hidden_states
            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)

            if use_cache:
                outputs += (past_key_values,)

            return outputs

        def ernie4_5_model_forward(
                self,
                input_ids=None,
                position_ids=None,
                attention_mask=None,
                inputs_embeds=None,
                use_cache=None,
                past_key_values=None,
                output_attentions=False,
                output_hidden_states=None,
                return_dict=True,
                **kwargs,
        ):
            def _coerce_legacy_past_key_values(cache_like):
                if cache_like is None:
                    return None

                if isinstance(cache_like, tuple):
                    return cache_like

                if hasattr(cache_like, "__iter__"):
                    legacy = []
                    for layer_cache in cache_like:
                        if layer_cache is None:
                            legacy.append(None)
                            continue
                        if isinstance(layer_cache, (tuple, list)) and len(layer_cache) >= 2:
                            if layer_cache[0] is None or layer_cache[1] is None:
                                legacy.append(None)
                            else:
                                legacy.append((layer_cache[0], layer_cache[1]))
                            continue
                        legacy.append(None)

                    if len(legacy) < len(self.layers):
                        legacy.extend([None] * (len(self.layers) - len(legacy)))

                    return tuple(legacy)

                return cache_like

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

            past_key_values = _coerce_legacy_past_key_values(past_key_values)
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.to(self.embed_tokens.weight.dtype)

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            if position_ids is None:
                if past_key_values is None:
                    past_seen_tokens = 0
                elif hasattr(past_key_values, "get_seq_length"):
                    past_seen_tokens = past_key_values.get_seq_length()
                else:
                    past_seen_tokens = 0
                    for layer_cache in past_key_values:
                        if isinstance(layer_cache, (tuple, list)) and layer_cache:
                            key_cache = layer_cache[0]
                            if key_cache is not None and hasattr(key_cache, "shape"):
                                past_seen_tokens = key_cache.shape[-2]
                                break
                position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_seen_tokens
                position_ids = position_ids.unsqueeze(0)

            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = () if use_cache else None

            for idx, (decoder_layer) in enumerate(self.layers):

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    output_attentions=output_attentions,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs,
                )

                if isinstance(layer_outputs, (tuple, list)):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = next_decoder_cache if use_cache else None

            if return_dict is False:
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

        if not self._should_apply_native_monkeypatch():
            return

        ernie4_5_model = type(self.model.model)
        ernie4_5_model.forward = ernie4_5_model_forward
        ernie4_5_layer = type(self.model.model.layers[0])
        ernie4_5_layer.forward = ernie4_5_decode_layer_forward
