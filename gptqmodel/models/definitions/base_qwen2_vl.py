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

from typing import Dict, Optional

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY, move_to
from .._const import CPU
from ..base import BaseGPTQModel


class BaseQwen2VLGPTQ(BaseGPTQModel):
    loader = AutoModelForVision2Seq

    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = "model.layers"

    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    def pre_quantize_generate_hook_start(self):
        self.model.visual = move_to(self.model.visual, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        self.model.visual = move_to(self.model.visual, device=CPU)

    @staticmethod
    def process_vision_info(
            conversations: list[dict] | list[list[dict]],
    ) -> Optional[list[Image.Image]]:
        vision_infos = extract_vision_info(conversations)
        # Read images
        image_inputs = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            else:
                raise ValueError("image, image_url should in content.")
        if len(image_inputs) == 0:
            image_inputs = None
        return image_inputs

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path)

    def prepare_dataset(self, calibration_dataset, calibration_dataset_concat_size=None, batch_size: int = 1):
        processor = self.load_processor()
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            text = processor.apply_chat_template(
                batch, tokenize=False, add_generation_prompt=True
            )
            image_inputs = self.process_vision_info(batch)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            calib_data.append(inputs)
        del processor
        return calib_data
