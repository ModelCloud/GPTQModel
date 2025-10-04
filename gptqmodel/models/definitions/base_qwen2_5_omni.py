# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
from typing import Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForTextToWaveform, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class BaseQwen2_5_OmniGPTQ(BaseQModel):
    ATTENTION_MASKS_REQUIRED_FOR_INPUT = True
    ATTENTION_MASKS_DTYPE = torch.long

    INPUT_EMBEDDING_EXTRA_ARGS = {
        "return_audio": False,
    }

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
        # load speaker
        spk_path = os.path.join(self.model_local_path, "spk_dict.pt")
        self.model.load_speakers(spk_path)

        self.shell_module_materialize(self.model.thinker.model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(self.model.thinker.visual, self.quantize_config.device)
        self.shell_module_materialize(self.model.thinker.audio_tower, self.quantize_config.device)
        self.shell_module_materialize(self.model.thinker.visual.rotary_pos_emb, self.quantize_config.device)
        self.shell_module_materialize(self.model.thinker.model.rotary_emb, self.quantize_config.device)
        if hasattr(self.model, "talker"):
            self.shell_module_materialize(self.model.talker, self.quantize_config.device)
        if hasattr(self.model, "token2wav"):
            self.shell_module_materialize(self.model.token2wav, self.quantize_config.device)
        for layer in self.model.thinker.model.layers:
            self.shell_module_materialize(layer.self_attn.rotary_emb, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            offload_to_disk(model=self.model.thinker.model,
                            module=self.model.thinker.model.embed_tokens,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            offload_to_disk(model=self.model.thinker,
                            module=self.model.thinker.visual,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            offload_to_disk(model=self.model.thinker,
                            module=self.model.thinker.audio_tower,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            offload_to_disk(model=self.model.thinker.visual,
                            module=self.model.thinker.visual.rotary_pos_emb,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            offload_to_disk(model=self.model.thinker.model,
                            module=self.model.thinker.model.rotary_emb,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            if hasattr(self.model, "talker"):
                offload_to_disk(model=self.model,
                                module=self.model.talker,
                                disk_path=self.quantize_config.offload_to_disk_path,
                                )
            if hasattr(self.model, "token2wav"):
                offload_to_disk(model=self.model,
                                module=self.model.token2wav,
                                disk_path=self.quantize_config.offload_to_disk_path,
                                )

            for layer in self.model.thinker.model.layers:
                layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(CPU)

            return

        self.model.thinker.model.embed_tokens = self.model.thinker.model.embed_tokens.to(CPU)
        self.model.thinker.visual = self.model.thinker.visual.to(CPU)
        self.model.thinker.audio_tower = self.model.thinker.audio_tower.to(CPU)
        if hasattr(self.model, "talker"):
            self.model.talker = self.model.talker.to(CPU)
        if hasattr(self.model, "token2wav"):
            self.model.token2wav = self.model.token2wav.to(CPU)

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

    def forward(self, *args, **kwargs):
        """Delegate textual forward passes to the thinker submodule.

        The top-level Hugging Face wrapper leaves ``forward`` unimplemented when
        ``trust_remote_code`` is disabled, so we expose the thinker equivalent to
        keep tooling such as lm-eval operational in quantized environments.
        """

        return self.model.thinker(*args, **kwargs)

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path)

    def prepare_dataset(self, calibration_dataset, calibration_dataset_concat_size=None, batch_size: int = 1, **kwargs):
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
