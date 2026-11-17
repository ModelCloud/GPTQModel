# SPDX-FileCopyrightText: 2024 AIDC-AI
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Ovis2.6 wrapper with exact native MoE/Next decoders, without checkpoint code."""

import torch
from transformers import GenerationMixin, PreTrainedModel, Qwen3MoeForCausalLM, Qwen3NextForCausalLM

from .configuration_ovis2_6 import Ovis2_6_Moe_Config, Ovis2_6_Next_Config
from .vision import INDICATOR_IDS, VISUAL_ATOM_ID, Siglip2NavitModel, VisualEmbedding, VisualTokenizer


class Ovis2_6_MoeForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = Ovis2_6_Moe_Config
    base_model_prefix = "ovis2_6_moe"
    decoder_class = Qwen3MoeForCausalLM
    _supports_sdpa = True
    _supports_flash_attn = False
    _keys_to_ignore_on_load_missing = [r"visual_tokenizer.vit.vision_model.post_layernorm\.(weight|bias)"]
    _no_split_modules = ["Siglip2EncoderLayer", "Qwen3MoeDecoderLayer", "Qwen3NextDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        for subconfig in (config.llm_config, config.vit_config):
            if subconfig._attn_implementation is None:
                subconfig._attn_implementation = config._attn_implementation
        self.llm = self.decoder_class(config.llm_config)
        self.visual_tokenizer = VisualTokenizer(Siglip2NavitModel(config.vit_config), config.visual_vocab_size)
        self.vte = VisualEmbedding(config.visual_vocab_size, config.hidden_size)
        self.post_init()

    @property
    def indicator_token_indices(self):
        return torch.arange(
            self.config.visual_vocab_size - 4, self.config.visual_vocab_size, device=self.vte.weight.device
        )

    @property
    def text_tokenizer(self):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Bind a Tokenicer-normalized tokenizer before Ovis2.6 prompt preprocessing")
        return tokenizer

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def get_wte(self):
        return self.get_input_embeddings()

    def tie_weights(self, *args, **kwargs):
        self.llm.tie_weights(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        grid_thws=None,
        inputs_embeds=None,
        labels=None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            if input_ids is not None or pixel_values is not None:
                raise ValueError("Pass input_ids/media or inputs_embeds, not both")
        else:
            inputs_embeds = self.merge_multimodal(input_ids, pixel_values, grid_thws)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    def merge_multimodal(self, input_ids, pixel_values=None, grid_thws=None):
        if input_ids is None:
            raise ValueError("input_ids are required")
        negative = input_ids < 0
        valid_visual = input_ids == VISUAL_ATOM_ID
        for indicator in INDICATOR_IDS:
            valid_visual |= input_ids == indicator
        if (negative & ~valid_visual).any():
            raise ValueError("Unexpanded or unknown Ovis2.6 visual placeholder")
        if pixel_values is None and negative.any():
            raise ValueError("Visual placeholders require pixel_values and grid_thws")
        embeds = self.get_wte()(input_ids.masked_fill(negative, 0))
        if pixel_values is not None:
            if grid_thws is None:
                raise ValueError("grid_thws are required with pixel_values")
            visual_tokens = self.visual_tokenizer(pixel_values, grid_thws)
            visual_embeds = self.vte(visual_tokens).to(embeds)
            if visual_embeds.shape[0] != (input_ids == VISUAL_ATOM_ID).sum().item():
                raise ValueError("Visual atom count does not match encoded image/video patches")
            indicators = self.vte(self.indicator_token_indices.to(self.vte.weight.device)).to(embeds)
            for index, token in enumerate(INDICATOR_IDS):
                embeds[input_ids == token] = indicators[index]
            embeds[input_ids == VISUAL_ATOM_ID] = visual_embeds
        return embeds

    def preprocess_inputs(self, *args, **kwargs):
        from .processing_ovis2_6 import Ovis26Processor

        return Ovis26Processor(
            self.config, self.text_tokenizer, self.visual_tokenizer.image_processor
        ).preprocess_inputs(*args, **kwargs)

    def generate(
        self,
        inputs=None,
        **kwargs,
    ):
        inputs = kwargs.pop("input_ids", inputs)
        if inputs is None:
            raise ValueError("Ovis2.6 generate requires inputs or input_ids")
        pad = self.config.pad_token_id
        attention_mask = kwargs.pop("attention_mask", inputs.ne(pad) if pad is not None else torch.ones_like(inputs))
        inputs_embeds = self.merge_multimodal(
            input_ids=inputs, pixel_values=kwargs.pop("pixel_values", None), grid_thws=kwargs.pop("grid_thws", None)
        )
        enable_thinking = kwargs.pop("enable_thinking", False)
        enable_thinking_budget = kwargs.pop("enable_thinking_budget", False)
        thinking_budget = kwargs.pop("thinking_budget", 1024)

        if enable_thinking and enable_thinking_budget:
            if (
                inputs.shape[0] != 1
                or kwargs.get("num_beams", 1) != 1
                or kwargs.get("num_return_sequences", 1) != 1
                or kwargs.get("return_dict_in_generate", False)
            ):
                raise ValueError("Source thinking-budget continuation requires one non-beam tensor result")
            if thinking_budget < 1 or kwargs.get("max_new_tokens", 0) <= thinking_budget + 25:
                raise ValueError("max_new_tokens must exceed thinking_budget + 25")
            actual_max_new_tokens = kwargs["max_new_tokens"]
            kwargs["max_new_tokens"] = thinking_budget
            generated_ids = self.llm.generate(
                inputs=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
            )
            output_ids = generated_ids
            output_ids_list = generated_ids[0]

            # check if the generation has already finished (151645 is <|im_end|>)
            if 151645 not in output_ids_list:
                # check if the thinking process has finished (151668 is </think>)
                # and prepare the second model input
                if 151668 not in output_ids_list:
                    early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                    early_stopping_ids = self.text_tokenizer(
                        early_stopping_text, return_tensors="pt", return_attention_mask=False
                    ).input_ids.to(inputs.device)
                    input_ids_appendent = torch.cat([output_ids, early_stopping_ids], dim=-1)
                    kwargs["streamer"].put(early_stopping_ids) if "streamer" in kwargs else None
                else:
                    input_ids_appendent = output_ids

                # second generation
                new_inputs = torch.cat([inputs, input_ids_appendent], dim=-1)
                attention_mask = torch.ne(new_inputs, self.text_tokenizer.pad_token_id).to(device=inputs.device)
                inputs_embeds_appendent = self.merge_multimodal(
                    input_ids=input_ids_appendent, pixel_values=None, grid_thws=None
                )
                new_inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_appendent], dim=-2)

                kwargs["max_new_tokens"] = inputs_embeds.size(-2) + actual_max_new_tokens - new_inputs_embeds.size(-2)
                generated_ids2 = self.llm.generate(
                    inputs=None, inputs_embeds=new_inputs_embeds, attention_mask=attention_mask, **kwargs
                )
                kwargs["streamer"].manual_end() if hasattr(kwargs.get("streamer"), "manual_end") else None
                return torch.cat([input_ids_appendent, generated_ids2], dim=-1)

            else:
                kwargs["streamer"].manual_end() if hasattr(kwargs.get("streamer"), "manual_end") else None
                return generated_ids

        else:
            generated_ids = self.llm.generate(
                inputs=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
            )
            kwargs["streamer"].manual_end() if hasattr(kwargs.get("streamer"), "manual_end") else None
            return generated_ids


class Ovis2_6_NextForCausalLM(Ovis2_6_MoeForCausalLM):
    config_class = Ovis2_6_Next_Config
    base_model_prefix = "ovis2_6_next"
    decoder_class = Qwen3NextForCausalLM
