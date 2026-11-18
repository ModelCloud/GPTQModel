# Copyright (c) 2024 OpenGVLab
# SPDX-License-Identifier: MIT
"""Legacy InternVL parameter layout, running package-owned Transformers 5 code."""

import warnings

import torch
from torch import nn
from transformers import (
    GenerationMixin,
    LlamaForCausalLM,
    Phi3ForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)
from transformers.modeling_utils import PreTrainedModel

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel


def decoder_model_class(architecture):
    if architecture == "InternLM2ForCausalLM":
        from .modeling_internlm2 import InternVLInternLM2ForCausalLM

        return InternVLInternLM2ForCausalLM
    return {
        "LlamaForCausalLM": LlamaForCausalLM,
        "Qwen2ForCausalLM": Qwen2ForCausalLM,
        "Qwen3ForCausalLM": Qwen3ForCausalLM,
        "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
        "Phi3ForCausalLM": Phi3ForCausalLM,
    }[architecture]


class InternVLChatModel(PreTrainedModel, GenerationMixin):
    config_class = InternVLChatConfig
    main_input_name = "input_ids"
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternVisionEncoderLayer"]
    _tied_weights_keys = {"language_model.lm_head.weight": "language_model.model.embed_tokens.weight"}

    def __init__(self, config, vision_model=None, language_model=None):
        # Revalidate mutable config before any model tensors are allocated.
        config = InternVLChatConfig(**config.to_dict())
        architecture = config.llm_config.architectures[0]
        if config.use_backbone_lora or config.use_llm_lora:
            raise ValueError("Merge InternVL LoRA adapters into the base checkpoint before loading")
        # Validate the conversation name before allocating a decoder.
        template = get_conv_template(config.template)
        super().__init__(config)
        image_size = config.force_image_size or config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.num_image_token = int((image_size // self.patch_size) ** 2 * self.downsample_ratio ** 2)
        # Vision uses the inspected eager implementation, independent of optional
        # flash-attn/apex/timm imports or checkpoint use_fa3 flags.
        self.vision_model = vision_model if vision_model is not None else InternVisionModel(config.vision_config)
        self.language_model = language_model if language_model is not None else decoder_model_class(architecture)(config.llm_config)
        width = config.vision_config.hidden_size * int(1 / self.downsample_ratio) ** 2
        self.mlp1 = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, config.llm_config.hidden_size),
                                  nn.GELU(), nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size))
        self.img_context_token_id = getattr(config, "img_context_token_id", None)
        self.conv_template = template
        self.system_message = getattr(config, "system_message", template.system_message)
        if architecture == "InternLM2ForCausalLM":
            self._tied_weights_keys = {"language_model.output.weight": "language_model.model.tok_embeddings.weight"}
        self.post_init()

    def _image_embeddings(self, input_ids, pixel_values=None, image_flags=None, visual_features=None):
        embeddings = self.language_model.get_input_embeddings()(input_ids)
        if pixel_values is None and visual_features is None:
            return embeddings
        if self.img_context_token_id is None:
            raise ValueError("Set img_context_token_id before supplying images")
        if visual_features is not None:
            features = visual_features
        else:
            features = self.extract_feature(pixel_values)
        if image_flags is not None:
            flags = image_flags.reshape(-1)
            if flags.numel() != features.shape[0]:
                raise ValueError("image_flags must contain one entry per image tile")
            features = features[flags == 1]
        selected = input_ids == self.img_context_token_id
        features = features.reshape(-1, embeddings.shape[-1])
        if int(selected.sum()) != features.shape[0]:
            raise ValueError("Image context token count does not match vision features")
        embeddings = embeddings.clone()
        embeddings[selected] = features.to(device=embeddings.device, dtype=embeddings.dtype)
        return embeddings

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, position_ids=None,
                image_flags=None, past_key_values=None, labels=None, use_cache=None,
                inputs_embeds=None, **kwargs):
        if inputs_embeds is not None and (input_ids is not None or pixel_values is not None):
            raise ValueError("Supply inputs_embeds or input_ids/images, not both")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self._image_embeddings(input_ids, pixel_values, image_flags)
        return self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                   position_ids=position_ids, past_key_values=past_key_values, labels=labels,
                                   use_cache=use_cache, **kwargs)

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / scale_factor ** 2))
        if self.ps_version == "v1":
            warnings.warn("ps_version v1 preserves the legacy transposed image grid", stacklevel=2)
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        output = self.vision_model(pixel_values=pixel_values, output_hidden_states=self.select_layer != -1,
                                   return_dict=True)
        features = output.last_hidden_state if self.select_layer == -1 else output.hidden_states[self.select_layer]
        features = features[:, 1:]
        side = int(features.shape[1] ** 0.5)
        features = features.reshape(features.shape[0], side, side, -1)
        features = self.pixel_shuffle(features, self.downsample_ratio)
        return self.mlp1(features.reshape(features.shape[0], -1, features.shape[-1]))

    @torch.no_grad()
    def generate(self, pixel_values=None, input_ids=None, attention_mask=None, visual_features=None,
                 generation_config=None, output_hidden_states=None, image_flags=None, **kwargs):
        embeddings = self._image_embeddings(input_ids, pixel_values, image_flags, visual_features)
        # Legacy chat returns generated tokens only; delegate cache ownership to
        # the decoder and do not replay images during cached continuation.
        kwargs.setdefault("use_cache", True)
        return self.language_model.generate(inputs_embeds=embeddings, attention_mask=attention_mask,
                                            generation_config=generation_config,
                                            output_hidden_states=output_hidden_states, **kwargs)

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config = dict(generation_config)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config = dict(generation_config)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
