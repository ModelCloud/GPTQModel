# Copyright (C) 2025 AIDC-AI
# SPDX-License-Identifier: Apache-2.0
# GPT-QModel adaptation; source revisions and changes: NOTICE.
"""Ovis2.5 local wrapper: native Qwen3 plus the source's custom NaViT/soft vocabulary."""

from typing import Optional, Union

import torch
from transformers import GenerationMixin, PreTrainedModel
from transformers.generation.utils import GenerateOutput

from ..native.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from .configuration_ovis2_5 import Ovis2_5_Config
from .vision_ovis2_5 import (
    INDICATOR_IDS,
    VISUAL_ATOM_ID,
    Siglip2NavitModel,
    VisualEmbedding,
    VisualTokenizer,
)


class Ovis2_5(PreTrainedModel, GenerationMixin):
    config_class = Ovis2_5_Config
    base_model_prefix = "ovis2_5"
    _supports_sdpa = True
    _no_split_modules = ["Qwen3DecoderLayer", "Siglip2EncoderLayer"]
    _tied_weights_keys = {"llm.lm_head.weight": "llm.model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.llm = Qwen3ForCausalLM(config.llm_config)
        self.visual_tokenizer = VisualTokenizer(Siglip2NavitModel(config.vit_config), config.visual_vocab_size)
        self.vte = VisualEmbedding(config.visual_vocab_size, config.hidden_size)
        self.tokenizer = None
        self.processor = None
        self.post_init()

    @property
    def text_tokenizer(self):
        return self.tokenizer

    def get_wte(self):
        return self.llm.get_input_embeddings()

    def get_input_embeddings(self):
        return self.get_wte()

    def set_input_embeddings(self, embeddings):
        self.llm.set_input_embeddings(embeddings)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, embeddings):
        self.llm.set_output_embeddings(embeddings)

    def preprocess_inputs(self, messages, **kwargs):
        if self.processor is None:
            if self.tokenizer is None:
                raise ValueError("Attach Ovis2_5Processor or load through GPTQModel before preprocessing")
            from .processing_ovis2_5 import Ovis2_5Processor
            processor = Ovis2_5Processor(self.visual_tokenizer.image_processor, self.tokenizer, self.config)
        else:
            processor = self.processor
        return processor.preprocess_inputs(messages, **kwargs)

    def merge_multimodal(self, input_ids, pixel_values=None, grid_thws=None):
        invalid = (input_ids < 0) & ~torch.isin(input_ids, input_ids.new_tensor([VISUAL_ATOM_ID, *INDICATOR_IDS]))
        if invalid.any():
            raise ValueError("Ovis2.5 received an unexpanded or unknown negative visual placeholder")
        mask = input_ids < 0
        embeds = self.get_wte()(input_ids.masked_fill(mask, 0))
        if pixel_values is None:
            if mask.any():
                raise ValueError("Ovis2.5 visual placeholders require pixel_values and grid_thws")
            return embeds
        if grid_thws is None:
            raise ValueError("Ovis2.5 pixel_values require grid_thws")
        if int((input_ids == VISUAL_ATOM_ID).sum()) != int(grid_thws.prod(-1).sum()) // self.config.vit_config.hidden_stride ** 2:
            raise ValueError("Ovis2.5 visual atom count does not match the patch grid")
        # These are source-defined final four rows. A slice avoids a root-level
        # nonpersistent buffer that a checkpoint-backed meta shell cannot load.
        indicators = self.vte.weight[-4:].to(embeds)
        tokens = self.visual_tokenizer(pixel_values, grid_thws)
        visual = self.vte(tokens).to(embeds)
        for i, indicator in enumerate(INDICATOR_IDS):
            embeds[input_ids == indicator] = indicators[i]
        embeds[input_ids == VISUAL_ATOM_ID] = visual
        return embeds

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, grid_thws=None,
                labels=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("Specify input_ids or inputs_embeds, not both")
        if inputs_embeds is None:
            inputs_embeds = self.merge_multimodal(input_ids, pixel_values, grid_thws)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        inputs = kwargs.pop("input_ids", None) if inputs is None else inputs
        if inputs is None:
            raise ValueError("Ovis2.5 generation requires input_ids")
        # HF loads the release policy onto this wrapper, rather than its nested decoder.
        generation_config = kwargs.pop("generation_config", None) or self.generation_config
        kwargs["generation_config"] = generation_config
        pad_id = self.text_tokenizer.pad_token_id if self.text_tokenizer is not None else self.config.llm_config.pad_token_id
        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is None:
            attention_mask = inputs.ne(pad_id) if pad_id is not None else torch.ones_like(inputs)
        inputs_embeds = self.merge_multimodal(
            input_ids=inputs,
            pixel_values=kwargs.pop('pixel_values', None),
            grid_thws=kwargs.pop('grid_thws', None)
        )
        enable_thinking = kwargs.pop('enable_thinking', False)
        enable_thinking_budget = kwargs.pop('enable_thinking_budget', False)
        thinking_budget = kwargs.pop('thinking_budget', 1024)

        if enable_thinking and enable_thinking_budget:
            if (inputs.shape[0] != 1
                    or kwargs.get('num_return_sequences', generation_config.num_return_sequences) not in (None, 1)
                    or kwargs.get('return_dict_in_generate', generation_config.return_dict_in_generate)):
                raise ValueError('Ovis2.5 budgeted thinking requires one prompt and tensor output')
            if self.text_tokenizer is None:
                raise ValueError('Ovis2.5 budgeted thinking requires the text tokenizer')
            actual_max_new_tokens = kwargs.get('max_new_tokens', generation_config.max_new_tokens)
            if actual_max_new_tokens is None:
                raise ValueError('Ovis2.5 budgeted thinking requires max_new_tokens')
            kwargs['max_new_tokens'] = thinking_budget
            generated_ids = self.llm.generate(inputs=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
            output_ids = generated_ids
            output_ids_list = generated_ids[0]

            # check if the generation has already finished (151645 is <|im_end|>)
            if 151645 not in output_ids_list:
                # check if the thinking process has finished (151668 is </think>)
                # and prepare the second model input
                if 151668 not in output_ids_list:
                    early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                    early_stopping_ids = self.text_tokenizer(early_stopping_text, return_tensors="pt", return_attention_mask=False).input_ids.to(inputs.device)
                    input_ids_appendent = torch.cat([output_ids, early_stopping_ids], dim=-1)
                    kwargs['streamer'].put(early_stopping_ids) if 'streamer' in kwargs else None
                else:
                    input_ids_appendent = output_ids


                # second generation
                new_inputs = torch.cat([inputs, input_ids_appendent], dim=-1)
                attention_mask = torch.ne(new_inputs, self.text_tokenizer.pad_token_id).to(device=inputs.device)
                inputs_embeds_appendent = self.merge_multimodal(
                    input_ids=input_ids_appendent,
                    pixel_values=None,
                    grid_thws=None
                )
                new_inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_appendent], dim=-2)

                kwargs['max_new_tokens'] = inputs_embeds.size(-2) + actual_max_new_tokens - new_inputs_embeds.size(-2)
                generated_ids2 = self.llm.generate(inputs=None, inputs_embeds=new_inputs_embeds, attention_mask=attention_mask, **kwargs)
                kwargs['streamer'].manual_end() if 'streamer' in kwargs else None
                return torch.cat([input_ids_appendent, generated_ids2], dim=-1)

            else:
                kwargs['streamer'].manual_end() if 'streamer' in kwargs else None
                return generated_ids

        else:
            generated_ids = self.llm.generate(inputs=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
            kwargs['streamer'].manual_end() if 'streamer' in kwargs else None
            return generated_ids
