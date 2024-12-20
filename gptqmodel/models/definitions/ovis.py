import copy
import logging
from typing import Dict

from ...utils.calibration import batched
from ...utils.image import fetch_image
from ...utils.model import MODALITY
import torch

from ..base import BaseGPTQModel


class OvisGPTQ(BaseGPTQModel):
    base_modules = ["llm.model.embed_tokens", "llm.model.norm", "visual_tokenizer", "vte"]

    layers_node = "llm.model.layers"
    layer_type = ["LlamaDecoderLayer", "Gemma2DecoderLayer"]
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    modality = [MODALITY.IMAGE_TO_TEXT]

    IGNORE_ID = -100

    def preprocess_inputs(self, sample: Dict) -> Dict:
        text_max_length = 832
        conversations = copy.deepcopy(sample["conversations"])
        images = [fetch_image(sample)]
        max_partition = 9

        prompt, input_ids, pixel_values, labels = self.model.preprocess_inputs(
            conversations,
            images,
            max_partition=max_partition,
            generation_preface=None,
            return_labels=True,
            propagate_exception=False
        )

        if pixel_values is None:
            pixel_values, _ = self.visual_tokenizer.mock_input()

        input_ids = input_ids[:text_max_length]
        labels = labels[:text_max_length]

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )

    def prepare_dataset(
            self,
            calibration_dataset,
            batch_size: int = 1,
            tokenizer=None, ):
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, self.preprocess_inputs):
            pixel_values, input_ids, labels = tuple([instance[key] for instance in batch]
                                                    for key in ("pixel_values", "input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.text_tokenizer.pad_token_id)
            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=self.IGNORE_ID)

            num_valid_label = torch.not_equal(labels, self.IGNORE_ID).sum().item()
            if num_valid_label == 0:
                logging.warning(
                    f'[DataCollatorForMultimodalDatasetGPTQ] All labels are ignored, may causing training instability\n{input_ids=}\n{attention_mask=}\n{labels=}')
            calib_data.append(dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values
            ))

        return calib_data

    def generate(self, inputs, **kwargs):
        """shortcut for model.generate"""
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.generate(inputs, **kwargs)
