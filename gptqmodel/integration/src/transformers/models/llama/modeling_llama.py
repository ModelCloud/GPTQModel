import time
from typing import List, Optional, Tuple, Union, Unpack

import torch
import torch.utils.checkpoint
from transformers import add_start_docstrings, LlamaPreTrainedModel, Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LLAMA_START_DOCSTRING, LLAMA_INPUTS_DOCSTRING
from transformers.utils import add_start_docstrings_to_model_forward

from gptqmodel.utils.model import nested_move_to


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

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
        next_decoder_cache = None
        decoder_layers = self.layers[: self.config.num_hidden_layers]
        layers_len = len(decoder_layers)

        for index, decoder_layer in enumerate(decoder_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
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
                print(f"modeling_llama formawrd index: {index}")
                # now = time.time()
                torch.cuda.synchronize()
                if index < layers_len - 1: # put next layer on cuda
                    decoder_layers[index + 1].to("cuda", non_blocking=True)
                    # print(f"=== move next layer to cuda, time: {time.time() - now}")
                else: # for last layer, put layer 0 on cpu for next loop
                    decoder_layers[0].to("cuda", non_blocking=True)
                    # print(f"=== move layer 0 to cuda, time: {time.time() - now}")
                nested_move_to(decoder_layer, "cuda")
                nested_move_to(hidden_states, "cuda")
                nested_move_to(position_ids, "cuda")
                nested_move_to(cache_position, "cuda")
                nested_move_to(position_embeddings, "cuda")
                # decoder_layer.to("cuda")
                # hidden_states.to("cuda")
                # position_ids.to("cuda")
                # cache_position.to("cuda")
                # position_embeddings[0].to("cuda")
                # position_embeddings[1].to("cuda")
                # decoder_layer.input_layernorm.to("cuda")
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
                # now = time.time()
                decoder_layer.to("cpu", non_blocking=True)
                # print(f"=== move layer back to cpu, time: {time.time() - now}")
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
