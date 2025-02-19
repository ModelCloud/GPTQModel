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

import os.path
import shutil
from typing import Dict, Optional

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from ...utils.calibration import batched
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY, move_to
from .._const import CPU
from ..base import BaseGPTQModel


class Qwen2VLGPTQ(BaseGPTQModel):
    loader = AutoModelForVision2Seq

    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = "model.layers"
    layer_type = "Qwen2VLDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    quant_override_files = {
        "preprocessor_config.json": {
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [
                0.48145466,
                0.4578275,
                0.40821073
            ],
            "image_processor_type": "Qwen2VLImageProcessor",
            "image_std": [
                0.26862954,
                0.26130258,
                0.27577711
            ],
            "max_pixels": 1003520,
            "merge_size": 2,
            "min_pixels": 3136,
            "patch_size": 14,
            "processor_class": "Qwen2VLProcessor",
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "size": {
                "max_pixels": 1003520,
                "min_pixels": 3136
            },
            "temporal_patch_size": 2,
            "vision_token_id": 151654
        }
    }

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

    def prepare_dataset(
            self,
            calibration_dataset,
            calibration_dataset_concat_size,
            batch_size: int = 1,
            tokenizer=None, ):
        import json
        import tempfile

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_local_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            chat_template_file = os.path.join(self.model_local_path, "chat_template.json")
            if os.path.exists(chat_template_file):
                shutil.copyfile(chat_template_file, os.path.join(tmp_dir, "chat_template.json"))
            tokenizer.save_pretrained(tmp_dir)
            with open(os.path.join(tmp_dir, "preprocessor_config.json"), "w") as f:
                f.write(json.dumps(self.quant_override_files["preprocessor_config.json"]))
            processor = AutoProcessor.from_pretrained(tmp_dir)
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
