# Copyright 2026 the HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers.processing_utils import ProcessorMixin

from ..native.auto import NativeTokenizer


class Inkling2Processor(ProcessorMixin):
    """Combine the Inkling image patches and their placeholder tokens."""

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, image_token="<|unused_200054|>"):
        self.image_token = getattr(tokenizer, "image_token", image_token)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        return self.image_token * int(image_inputs["image_grid_thw"][image_idx].prod())

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        from .image_processing import Inkling2ImageProcessor

        kwargs.pop("config", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
            tokenizer = NativeTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        image_processor = Inkling2ImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return [image_processor, tokenizer]
