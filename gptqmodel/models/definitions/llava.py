# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, Dict

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class LlavaQModel(BaseQModel):
    """Hugging Face Transformers LLaVA (e.g. LLaVA-1.5) model definition."""

    loader = AutoModelForImageTextToText
    require_load_processor = True
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    # LLaVA's Transformers runtime nests the vision, projector, and language
    # modules under `model`, while the HF checkpoint stores them at the root.
    # Its language checkpoint also has an extra `model` segment.
    HF_CONVERSION_MAP_REVERSED = (
        SimpleNamespace(
            source_patterns=[r"^lm_head"],
            target_patterns=[r"^language_model.lm_head"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"^model\.language_model\.(.+)$"],
            target_patterns=[r"^language_model.model.\1"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"^model\.vision_tower\.(.+)$"],
            target_patterns=[r"^vision_tower.vision_model.\1"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"^model\.multi_modal_projector\.(.+)$"],
            target_patterns=[r"^multi_modal_projector.\1"],
            operations=[],
        ),
    )

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=False)

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        modules = (
            (language_model.embed_tokens, "model.language_model.embed_tokens"),
            (language_model.norm, "model.language_model.norm"),
            (language_model.rotary_emb, "model.language_model.rotary_emb"),
            (core_model.vision_tower, "model.vision_tower"),
            (core_model.multi_modal_projector, "model.multi_modal_projector"),
        )
        for module, module_path in modules:
            self.shell_module_materialize(
                target_submodule=module,
                device=self.quantize_config.device,
                module_path=module_path,
            )

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        language_model = core_model.language_model
        modules = (
            (language_model, language_model.embed_tokens),
            (language_model, language_model.norm),
            (language_model, language_model.rotary_emb),
            (core_model, core_model.vision_tower),
            (core_model, core_model.multi_modal_projector),
        )
        if self.quantize_config.offload_to_disk:
            for parent, module in modules:
                offload_to_disk(
                    model=parent,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
        else:
            for parent, module in modules:
                name = next(name for name, child in parent.named_children() if child is module)
                setattr(parent, name, move_to(module, device=CPU))

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del kwargs
        processor = self.processor or self.load_processor()
        calibration = []
        for batch in batched_conversations(calibration_dataset, batch_size):
            calibration.append(
                processor.apply_chat_template(
                    batch,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            )
        return calibration

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
            else:
                example["pixel_values"] = pixel_values.to(device=vision_device)
        return example
