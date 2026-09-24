# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace
from typing import Any, Dict

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class InternVLQModel(BaseQModel):
    """Quantization definition for the Transformers-native InternVL model."""

    # InternVL checkpoints keep the text model under ``language_model.model``
    # while the Transformers runtime exposes it as ``model.language_model``.
    # The generic conversion registry only removes the outer prefix, leaving
    # the extra ``model`` segment unresolved for LazyTurtle materialization.
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
            source_patterns=[r"^model\.vision_tower"],
            target_patterns=[r"^vision_tower"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"^model\.multi_modal_projector"],
            target_patterns=[r"^multi_modal_projector"],
            operations=[],
        ),
    )

    loader = AutoModelForImageTextToText
    require_load_processor = True
    support_batch_quantize = False

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    lm_head = "lm_head"
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

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.norm, self.quantize_config.device)
        self.shell_module_materialize(language_model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(core_model.vision_tower, self.quantize_config.device)
        self.shell_module_materialize(core_model.multi_modal_projector, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        language_model = core_model.language_model
        if self.quantize_config.offload_to_disk:
            for module in (
                language_model.embed_tokens,
                language_model.norm,
                language_model.rotary_emb,
            ):
                offload_to_disk(
                    model=language_model,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            for module in (core_model.vision_tower, core_model.multi_modal_projector):
                offload_to_disk(
                    model=core_model,
                    module=module,
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

    @classmethod
    def prepare_inputs_for_conversations(
        cls,
        processor: ProcessorMixin,
        conversations: list[dict] | list[list[dict]],
    ):
        return processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del kwargs
        processor = self.load_processor()
        calib_data = []
        for batch in batched_conversations(
            calibration_dataset,
            batch_size,
            process_func=self.preprocess_dataset,
        ):
            calib_data.append(self.prepare_inputs_for_conversations(processor, batch))
        del processor
        return calib_data

    def move_input_capture_example(
        self,
        example: Dict[str, Any],
        data_device: torch.device,
    ) -> Dict[str, Any]:
        example = super().move_input_capture_example(example, data_device)
        pixel_values = example.get("pixel_values")
        if torch.is_tensor(pixel_values):
            vision_tower = self.model.model.vision_tower
            first_parameter = next(vision_tower.parameters(), None)
            vision_dtype = getattr(vision_tower, "dtype", None)
            if not isinstance(vision_dtype, torch.dtype):
                vision_dtype = getattr(first_parameter, "dtype", None)
            if isinstance(vision_dtype, torch.dtype):
                example["pixel_values"] = pixel_values.to(dtype=vision_dtype)
        return example


__all__ = ["InternVLQModel"]
