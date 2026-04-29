# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY, get_module, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


QWEN_VL_UTILS_INSTALL_HINT = (
    "qwen_vl_utils is required for Qwen3-VL video inputs. "
    "Install it with `pip install qwen-vl-utils`."
)


def _load_fetch_video():
    try:
        from qwen_vl_utils.vision_process import fetch_video
    except ImportError as exc:
        raise ImportError(QWEN_VL_UTILS_INSTALL_HINT) from exc

    return fetch_video


class BaseQwen3VLGPTQ(BaseQModel):
    loader = AutoModelForImageTextToText

    pre_lm_head_norm_module = ["model.norm", "norm"]

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

    @classmethod
    def extract_layers_node(cls):
        return ["model.language_model.layers", "language_model.layers"]

    @classmethod
    def get_base_modules(cls, model):
        prefix, core_model = cls._resolve_multimodal_layout(model)
        base_modules = []
        for name, _ in core_model.named_children():
            if name != "language_model":
                base_modules.append(f"{prefix}.{name}" if prefix else name)
        return base_modules

    @classmethod
    def _resolve_multimodal_layout(cls, model):
        for prefix in ("model", ""):
            core_model = get_module(model, prefix) if prefix else model
            if core_model is None:
                continue
            if hasattr(core_model, "language_model") and hasattr(core_model, "visual"):
                return prefix, core_model

        raise AttributeError("Unable to resolve a Qwen VL core model with `language_model` and `visual` modules.")

    def _core_multimodal_model(self):
        _, core_model = self._resolve_multimodal_layout(self.model)
        return core_model

    def _materialize_core_module(self, parent, attr_name: str):
        module = getattr(parent, attr_name)
        if "_turtle_lock" not in self.__dict__ and "shell_module_materialize" not in self.__dict__:
            setattr(parent, attr_name, move_to(module, device=self.quantize_config.device))
            return
        setattr(
            parent,
            attr_name,
            self.shell_module_materialize(module, self.quantize_config.device),
        )

    def pre_quantize_generate_hook_start(self):
        core_model = self._core_multimodal_model()
        self._materialize_core_module(core_model.language_model, "embed_tokens")
        self._materialize_core_module(core_model.language_model, "rotary_emb")
        self._materialize_core_module(core_model, "visual")

    def pre_quantize_generate_hook_end(self):
        core_model = self._core_multimodal_model()
        if self.quantize_config.offload_to_disk:
            offload_to_disk(model=core_model.language_model,
                            module=core_model.language_model.embed_tokens,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )
            offload_to_disk(model=core_model.language_model,
                            module=core_model.language_model.rotary_emb,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )
            offload_to_disk(model=core_model,
                            module=core_model.visual,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )
            return

        core_model.language_model.embed_tokens = move_to(core_model.language_model.embed_tokens, device=CPU)
        core_model.language_model.rotary_emb = move_to(core_model.language_model.rotary_emb, device=CPU)
        core_model.visual = move_to(core_model.visual, device=CPU)

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
        fetch_video = None
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(
                    fetch_image(vision_info, image_patch_size=image_patch_size)
                )
            elif "video" in vision_info:
                if fetch_video is None:
                    fetch_video = _load_fetch_video()
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
        if video_inputs is None and not return_video_metadata:
            # Keep the image-only call contract aligned with the earlier VL
            # adapters so processor(images=...) can use the return value directly.
            return image_inputs
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
            vision_inputs = self.process_vision_info(batch)
            if isinstance(vision_inputs, tuple):
                image_inputs, video_inputs = vision_inputs
            else:
                image_inputs, video_inputs = vision_inputs, None
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
