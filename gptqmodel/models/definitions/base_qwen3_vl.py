# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from qwen_vl_utils.vision_process import fetch_video
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY, move_to
from .._const import CPU
from ..base import BaseQModel


class BaseQwen3VLGPTQ(BaseQModel):
    loader = AutoModelForImageTextToText

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1", "q_norm:!", "k_norm:!"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    def pre_quantize_generate_hook_start(self):
        self.model.visual = move_to(self.model.visual, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        self.model.visual = move_to(self.model.visual, device=CPU)

    @staticmethod
    def process_vision_info(
        conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        return_video_kwargs: bool = False,
        return_video_metadata: bool = False,
        image_patch_size: int = 14,
    ) -> Tuple[
        Optional[List[Image.Image]],
        Optional[List[Union[torch.Tensor, List[Image.Image]]]],
        Optional[Dict[str, Any]],
    ]:

        vision_infos = extract_vision_info(conversations)
        ## Read images or videos
        image_inputs = []
        video_inputs = []
        video_sample_fps_list = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(
                    fetch_image(vision_info, image_patch_size=image_patch_size)
                )
            elif "video" in vision_info:
                video_input, video_sample_fps = fetch_video(
                    vision_info,
                    return_video_sample_fps=True,
                    image_patch_size=image_patch_size,
                    return_video_metadata=return_video_metadata,
                )
                video_sample_fps_list.append(video_sample_fps)
                video_inputs.append(video_input)
            else:
                raise ValueError("image, image_url or video should in content.")
        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None

        video_kwargs = {"do_sample_frames": False}
        if not return_video_metadata:  # BC for qwen2.5vl
            video_kwargs.update({"fps": video_sample_fps_list})

        if return_video_kwargs:
            return image_inputs, video_inputs, video_kwargs
        return image_inputs, video_inputs

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path)

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        processor = self.load_processor()
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            text = processor.apply_chat_template(
                batch, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = self.process_vision_info(batch)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            calib_data.append(inputs)
        del processor
        return calib_data
