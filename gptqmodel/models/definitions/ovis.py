# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
from typing import Dict

import torch

from ...utils.calibration import batched
from ...utils.image import fetch_image
from ...utils.model import MODALITY, move_to
from ..base import BaseGPTQModel
from .._const import CPU


class OvisGPTQ(BaseGPTQModel):
    base_modules = ["llm.model.embed_tokens", "llm.model.norm", "visual_tokenizer", "vte"]
    pre_lm_head_norm_module = "llm.model.norm"

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

    def pre_quantize_generate_hook_start(self):
        self.model.visual_tokenizer = move_to(self.model.visual_tokenizer, self.quantize_config.device)
        self.model.vte = move_to(self.model.vte, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        self.model.visual_tokenizer = move_to(self.model.visual_tokenizer, CPU)
        self.model.vte = move_to(self.model.vte, CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
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

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
        }

    def prepare_dataset(
            self,
            calibration_dataset,
            batch_size: int = 1,
            tokenizer=None, ):
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, self.preprocess_dataset):
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
            calib_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values,
            })

        return calib_data

    def generate(self, inputs, **kwargs):
        """shortcut for model.generate"""
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.generate(inputs, **kwargs)