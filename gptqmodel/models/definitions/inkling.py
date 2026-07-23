# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Dict

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, ProcessorMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from ...utils.calibration import batched
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class InklingMMQModel(BaseQModel):
    loader = AutoModelForMultimodalLM

    require_load_processor = True
    require_trust_remote_code = False
    layer_modules_strict = False

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    dynamic_expert_index = "n_routed_experts"
    defuser_auto_detect_moe = True
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    pre_lm_head_norm_module = "model.language_model.norm"

    # GQA output shapes differ across the attention projections. AWQ should
    # optimize the output projection against its actual attention input.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "r_proj:0",
                "q_norm:!",
                "k_norm:!",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                # Inkling's first decoder blocks use a dense MLP.
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                # Shared experts are native 3D parameters rather than Linear
                # modules. Keep these two accuracy-sensitive experts dense;
                # routed experts are expanded and quantized individually.
            },
        },
    ]

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.embed_norm, self.quantize_config.device)
        self.shell_module_materialize(core_model.vision_tower, self.quantize_config.device)
        self.shell_module_materialize(core_model.audio_tower, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        language_model = core_model.language_model
        if self.quantize_config.offload_to_disk:
            offload_to_disk(
                model=language_model,
                module=language_model.embed_tokens,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=language_model,
                module=language_model.embed_norm,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=core_model,
                module=core_model.vision_tower,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=core_model,
                module=core_model.audio_tower,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        language_model.embed_tokens = move_to(language_model.embed_tokens, device=CPU)
        language_model.embed_norm = move_to(language_model.embed_norm, device=CPU)
        core_model.vision_tower = move_to(core_model.vision_tower, device=CPU)
        core_model.audio_tower = move_to(core_model.audio_tower, device=CPU)

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
            tokenize=True,
            add_generation_prompt=True,
            reasoning_effort="medium",
            return_dict=True,
            return_tensors="pt",
        )

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del kwargs
        processor = self.load_processor()
        calibration_data = []
        for batch in batched(calibration_dataset, batch_size):
            calibration_data.append(self.prepare_inputs_for_conversations(processor, batch))
        del processor
        return calibration_data

    def move_input_capture_example(
        self,
        example: Dict[str, Any],
        data_device: torch.device,
    ) -> Dict[str, Any]:
        example = super().move_input_capture_example(example, data_device)
        pixel_values = example.get("pixel_values")
        if not torch.is_tensor(pixel_values) or not pixel_values.is_floating_point():
            return example

        first_parameter = next(self.model.model.vision_tower.parameters(), None)
        if first_parameter is not None:
            example["pixel_values"] = pixel_values.to(
                device=first_parameter.device,
                dtype=first_parameter.dtype,
            )
        return example

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        additional_inputs = super().prepare_layer_replay_kwargs(
            layer,
            layer_input,
            additional_inputs,
            target_device,
        )
        if not layer_input or not torch.is_tensor(layer_input[0]):
            return additional_inputs

        self_attn = getattr(layer, "self_attn", None)
        layer_config = getattr(self_attn, "config", None)
        if layer_config is None:
            return additional_inputs

        # The first-layer hook captures the sliding layer's prepared 4D mask.
        # Rebuild it for every replayed layer so full-attention blocks do not
        # accidentally inherit the sliding window. ``conv_mask`` retains the
        # original 2D padding signal when the calibration batch is padded.
        padding_mask = additional_inputs.get("conv_mask")
        if not torch.is_tensor(padding_mask) or padding_mask.ndim != 2:
            captured_mask = additional_inputs.get("attention_mask")
            padding_mask = captured_mask if torch.is_tensor(captured_mask) and captured_mask.ndim == 2 else None

        mask_factory = (
            create_sliding_window_causal_mask
            if getattr(layer, "layer_type", None) == "hybrid_sliding"
            else create_causal_mask
        )
        additional_inputs["attention_mask"] = mask_factory(
            config=layer_config,
            inputs_embeds=layer_input[0],
            attention_mask=padding_mask,
            past_key_values=additional_inputs.get("past_key_values"),
            position_ids=additional_inputs.get("position_ids"),
            layer_idx=getattr(self_attn, "layer_idx", None),
        )
        return additional_inputs


__all__ = ["InklingMMQModel"]
