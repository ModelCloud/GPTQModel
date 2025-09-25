# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Dict, Optional

from PIL import Image
from transformers import AutoModelForTextToWaveform, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY
from .._const import CPU
from ..base import BaseQModel


class BaseQwen2_5_OmniGPTQ(BaseQModel):
    loader = AutoModelForTextToWaveform

    pre_lm_head_norm_module = "thinker.model.norm"

    module_tree = [
        "thinker",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    def pre_quantize_generate_hook_start(self):

        self.model.thinker.model.embed_tokens = self.model.thinker.model.embed_tokens.to(self.quantize_config.device)
        self.model.thinker.visual = self.model.thinker.visual.to(self.quantize_config.device)
        self.model.thinker.audio_tower = self.model.thinker.audio_tower.to(self.quantize_config.device)

        self.model.thinker.visual.rotary_pos_emb = self.model.thinker.visual.rotary_pos_emb.to(self.quantize_config.device)
        self.model.thinker.model.rotary_emb = self.model.thinker.model.rotary_emb.to(self.quantize_config.device)

        for layer in self.model.thinker.model.layers:
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        self.model.thinker.model.embed_tokens = self.model.thinker.model.embed_tokens.to(CPU)
        self.model.thinker.visual = self.model.thinker.visual.to(CPU)
        self.model.thinker.audio_tower = self.model.thinker.audio_tower.to(CPU)

        self.model.thinker.visual.rotary_pos_emb = self.model.thinker.visual.rotary_pos_emb.to(CPU)
        self.model.thinker.model.rotary_emb = self.model.thinker.model.rotary_emb.to(CPU)

        for layer in self.model.thinker.model.layers:
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(CPU)
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

    def prepare_dataset(self, calibration, calibration_concat_size, calibration_sort, batch_size, calibration_min_length):
        processor = self.load_processor()
        calib_data = []
        for batch in batched(calibration, batch_size, process_func=self.preprocess_dataset):
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
