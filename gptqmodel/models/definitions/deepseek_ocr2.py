# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Dict

import torch
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.model import MODALITY, get_module
from ...utils.model import move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class DeepSeekOCR2QModel(BaseQModel):
    loader = AutoModelForImageTextToText
    require_load_processor = True
    support_batch_quantize = False

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    layer_modules_strict = False
    dynamic_expert_index = "n_routed_experts"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"
    _direct_parameter_names = ("view_separator",)

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate": ("gate:!",),
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    @classmethod
    def get_base_modules(cls, model):
        base_modules = []
        for module_name in (
            "model.vision_tower",
            "model.multi_modal_projector",
            "model.language_model.embed_tokens",
            "model.language_model.norm",
            "model.language_model.rotary_emb",
        ):
            if get_module(model, module_name) is not None:
                base_modules.append(module_name)
        return base_modules

    def _move_direct_parameters(self, device: torch.device) -> None:
        core_model = getattr(self.model, "model", self.model)
        for name in self._direct_parameter_names:
            parameter = getattr(core_model, name, None)
            if not isinstance(parameter, nn.Parameter) or parameter.device == device:
                continue
            setattr(
                core_model,
                name,
                nn.Parameter(parameter.to(device=device), requires_grad=parameter.requires_grad),
            )

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.norm, self.quantize_config.device)
        self.shell_module_materialize(language_model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(core_model.vision_tower, self.quantize_config.device)
        self.shell_module_materialize(core_model.multi_modal_projector, self.quantize_config.device)
        self._move_direct_parameters(torch.device(self.quantize_config.device))

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        language_model = core_model.language_model
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
                model=language_model,
                module=language_model.rotary_emb,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=core_model,
                module=core_model.vision_tower,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=core_model,
                module=core_model.multi_modal_projector,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        language_model.embed_tokens = move_to(language_model.embed_tokens, device=CPU)
        language_model.norm = move_to(language_model.norm, device=CPU)
        language_model.rotary_emb = move_to(language_model.rotary_emb, device=CPU)
        core_model.vision_tower = move_to(core_model.vision_tower, device=CPU)
        core_model.multi_modal_projector = move_to(core_model.multi_modal_projector, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=False)

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del kwargs
        processor = self.load_processor()
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            inputs = processor(
                images=[sample["image"] for sample in batch],
                text=[sample["text"] for sample in batch],
                padding=True,
                return_tensors="pt",
            )
            calib_data.append(inputs)
        del processor
        return calib_data

    def move_input_capture_example(self, example: Dict[str, Any], data_device: torch.device) -> Dict[str, Any]:
        example = super().move_input_capture_example(example, data_device)
        pixel_values = example.get("pixel_values")
        if torch.is_tensor(pixel_values):
            vision_tower = self.model.model.vision_tower
            first_parameter = next(vision_tower.parameters(), None)
            vision_device = getattr(first_parameter, "device", pixel_values.device)
            vision_dtype = getattr(vision_tower, "dtype", None)
            if not isinstance(vision_dtype, torch.dtype):
                vision_dtype = getattr(first_parameter, "dtype", None)
            if isinstance(vision_dtype, torch.dtype):
                example["pixel_values"] = pixel_values.to(device=vision_device, dtype=vision_dtype)
        return example


__all__ = ["DeepSeekOCR2QModel"]
