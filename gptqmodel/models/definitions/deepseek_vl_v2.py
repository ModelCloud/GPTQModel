# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Dict, Optional

import torch
from PIL import Image
from torch import nn
from transformers import AutoModelForCausalLM, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.image import fetch_image
from ...utils.model import MODALITY, get_module, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class DeepSeekVLV2QModel(BaseQModel):
    loader = AutoModelForCausalLM
    require_trust_remote_code = True
    require_load_processor = True
    support_batch_quantize = False

    require_pkgs = ["xformers>=0.0.21", "flash-attn"]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    dynamic_expert_index = "n_routed_experts"
    pre_lm_head_norm_module = "language.model.norm"
    layer_modules_strict = False
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "language",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_a_proj:0", "q_b_proj:0", "q_proj:0", "kv_a_proj_with_mqa:0", "kv_b_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    _direct_parameter_names = ("image_newline", "view_seperator", "view_separator", "tile_indicators")

    @classmethod
    def extract_layers_node(cls):
        return ["language.model.layers"]

    @classmethod
    def get_num_experts(cls, model_config):
        language_config = getattr(model_config, "language_config", model_config)
        return getattr(language_config, cls.dynamic_expert_index)

    @classmethod
    def get_base_modules(cls, model):
        base_modules = []
        for module_name in ("vision", "projector", "language.model.embed_tokens", "language.model.norm"):
            if get_module(model, module_name) is not None:
                base_modules.append(module_name)
        return base_modules

    def _move_direct_parameters(self, device: torch.device) -> None:
        for name in self._direct_parameter_names:
            parameter = getattr(self.model, name, None)
            if not isinstance(parameter, nn.Parameter) or parameter.device == device:
                continue
            setattr(
                self.model,
                name,
                nn.Parameter(parameter.to(device=device), requires_grad=parameter.requires_grad),
            )

    def pre_quantize_generate_hook_start(self):
        if self.turtle_model is not None:
            self.shell_direct_meta_materialize(target_submodule=self.model, device=self.quantize_config.device)
        language_model = self.model.language.model
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.norm, self.quantize_config.device)
        self.shell_module_materialize(self.model.vision, self.quantize_config.device)
        self.shell_module_materialize(self.model.projector, self.quantize_config.device)
        self._move_direct_parameters(torch.device(self.quantize_config.device))

    def pre_quantize_generate_hook_end(self):
        language_model = self.model.language.model
        self._move_direct_parameters(CPU)
        if self.quantize_config.offload_to_disk:
            offload_to_disk(
                model=language_model,
                module=language_model.embed_tokens,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=language_model,
                module=language_model.norm,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=self.model,
                module=self.model.vision,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=self.model,
                module=self.model.projector,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        language_model.embed_tokens = move_to(language_model.embed_tokens, device=CPU)
        language_model.norm = move_to(language_model.norm, device=CPU)
        self.model.vision = move_to(self.model.vision, device=CPU)
        self.model.projector = move_to(self.model.projector, device=CPU)

    @staticmethod
    def process_vision_info(conversation: list[dict]) -> Optional[list[Image.Image]]:
        image_inputs = []
        for message in conversation:
            for image in message.get("images", []) or []:
                if isinstance(image, Image.Image):
                    image_inputs.append(image.convert("RGB"))
                else:
                    image_inputs.append(fetch_image({"image": image}).convert("RGB"))
        return image_inputs or None

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=True)

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del batch_size, kwargs
        processor = self.load_processor()
        calib_data = []
        for batch in batched(calibration_dataset, 1, process_func=self.preprocess_dataset):
            conversation = batch[0]
            inputs = processor(
                conversations=conversation,
                images=self.process_vision_info(conversation),
                force_batchify=True,
                system_prompt="",
            )
            calib_data.append({
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": inputs.labels,
                "images": inputs.images,
                "images_seq_mask": inputs.images_seq_mask,
                "images_spatial_crop": inputs.images_spatial_crop,
            })
        del processor
        return calib_data

    def move_input_capture_example(self, example: Dict[str, Any], data_device: torch.device) -> Dict[str, Any]:
        example = super().move_input_capture_example(example, data_device)
        images = example.get("images")
        if torch.is_tensor(images):
            first_parameter = next(self.model.vision.parameters(), None)
            vision_device = getattr(first_parameter, "device", images.device)
            vision_dtype = getattr(self.model.vision, "dtype", None)
            if not isinstance(vision_dtype, torch.dtype):
                vision_dtype = getattr(first_parameter, "dtype", None)
            if isinstance(vision_dtype, torch.dtype):
                example["images"] = images.to(device=vision_device, dtype=vision_dtype)
        return example


__all__ = ["DeepSeekVLV2QModel"]
