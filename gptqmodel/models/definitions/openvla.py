# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""GPT-QModel support for the OpenVLA vision-language-action model."""

from __future__ import annotations

from typing import Any, Dict

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.image import fetch_image
from ...utils.model import MODALITY, get_module, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class OpenVLAQModel(BaseQModel):
    """Quantization definition for OpenVLA checkpoints such as `openvla/openvla-7b`.

    OpenVLA/Prismatic mounts a Llama-2 causal LM below `language_model` while a
    fused DINOv2+SigLIP vision backbone and its projector live at the wrapper
    root. Only the language decoder is quantized; the vision stack keeps its
    native dtype, matching the reference OpenVLA inference path.
    """

    loader = AutoModelForImageTextToText

    require_trust_remote_code = True
    require_load_processor = True
    support_batch_quantize = False

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    lm_head = "language_model.lm_head"
    pre_lm_head_norm_module = "language_model.model.norm"
    rotary_embedding = "language_model.model.rotary_emb"

    module_tree = [
        "language_model",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    @classmethod
    def extract_layers_node(cls):
        return ["language_model.model.layers"]

    @classmethod
    def get_base_modules(cls, model):
        base_modules = super().get_base_modules(model)
        seen = set(base_modules)
        for module_name in ("vision_backbone", "projector"):
            if get_module(model, module_name) is not None and module_name not in seen:
                base_modules.append(module_name)
                seen.add(module_name)
        return base_modules

    def _materialize_module(self, parent, name: str, module_path: str) -> None:
        module = getattr(parent, name)
        setattr(
            parent,
            name,
            self.shell_module_materialize(
                module,
                self.quantize_config.device,
                module_path=module_path,
            ),
        )

    def pre_quantize_generate_hook_start(self):
        language_model = self.model.language_model.model
        self._materialize_module(
            language_model, "embed_tokens", "language_model.model.embed_tokens"
        )
        self._materialize_module(
            language_model, "norm", "language_model.model.norm"
        )
        self._materialize_module(
            language_model, "rotary_emb", "language_model.model.rotary_emb"
        )
        self._materialize_module(self.model, "vision_backbone", "vision_backbone")
        self._materialize_module(self.model, "projector", "projector")

    def pre_quantize_generate_hook_end(self):
        language_model = self.model.language_model.model
        modules = (
            (language_model, "embed_tokens"),
            (language_model, "norm"),
            (language_model, "rotary_emb"),
            (self.model, "vision_backbone"),
            (self.model, "projector"),
        )

        if self.quantize_config.offload_to_disk:
            for parent, name in modules:
                offload_to_disk(
                    model=parent,
                    module=getattr(parent, name),
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        for parent, name in modules:
            setattr(parent, name, move_to(getattr(parent, name), device=CPU))

    def _vision_backbone_dtype(self) -> torch.dtype | None:
        vision_backbone = get_module(self.model, "vision_backbone")
        if vision_backbone is None:
            return None
        for parameter in vision_backbone.parameters():
            if parameter.dtype.is_floating_point:
                return parameter.dtype
        return None

    def move_input_capture_example(
        self,
        example: Dict[str, Any],
        data_device: torch.device,
    ) -> Dict[str, Any]:
        example = super().move_input_capture_example(example, data_device)
        pixel_values = example.get("pixel_values")
        vision_dtype = self._vision_backbone_dtype()
        if torch.is_tensor(pixel_values) and vision_dtype is not None and pixel_values.dtype != vision_dtype:
            example["pixel_values"] = pixel_values.to(dtype=vision_dtype)
        return example

    @staticmethod
    def _extract_prompt_and_image(conversation) -> tuple[str, Any]:
        instruction = None
        image = None
        for message in conversation:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                if instruction is None:
                    instruction = content
                continue
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if image is None and ("image" in item or "image_url" in item):
                    image = fetch_image(item)
                elif instruction is None and item.get("type") == "text":
                    instruction = item.get("text")

        instruction = (instruction or "").strip()
        if instruction.startswith("In:"):
            prompt = instruction
        else:
            # OpenVLA's reference prompt template, including the trailing space
            # that `predict_action()` expects before the action tokens.
            prompt = f"In: What action should the robot take to {instruction}?\nOut: "
        if not prompt.endswith(" "):
            prompt += " "
        return prompt, image

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(
            self.model_local_path,
            trust_remote_code=True,
        )

    def prepare_dataset(
        self,
        calibration_dataset,
        batch_size: int = 1,
        **kwargs,
    ) -> list[Dict[str, Any]]:
        del kwargs
        processor = self.load_processor()
        calibration_data = []
        for batch in batched_conversations(
            calibration_dataset,
            batch_size,
            process_func=self.preprocess_dataset,
        ):
            for conversation in batch:
                prompt, image = self._extract_prompt_and_image(conversation)
                if image is None:
                    # Text-only calibration stays supported; the vision stack is
                    # never quantized, so image samples only refine the language
                    # activations.
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
                else:
                    inputs = processor(
                        text=[prompt],
                        images=[image],
                        padding=True,
                        return_tensors="pt",
                    )
                calibration_data.append(dict(inputs))
        del processor
        return calibration_data


__all__ = ["OpenVLAQModel"]
