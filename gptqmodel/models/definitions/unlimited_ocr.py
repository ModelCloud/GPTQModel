# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Dict

import torch

from ...utils.calibration import batched
from ...utils.image import fetch_image
from ...utils.model import MODALITY, get_module, move_to
from ...utils.offload import offload_to_disk
from ...utils.torch import as_torch_device
from .._const import CPU
from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class UnlimitedOCRQModel(BaseQModel):
    """Native Transformers 5 Unlimited-OCR conditional generation and conversion."""

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.unlimited_ocr.UnlimitedOcrConfigLoader",
        model_class="gptqmodel.models.ports.unlimited_ocr.UnlimitedOcrForConditionalGeneration",
        processor_class="gptqmodel.models.ports.unlimited_ocr.UnlimitedOcrProcessor",
        tokenizer_class="gptqmodel.models.ports.unlimited_ocr.UnlimitedOcrTokenizer",
    )
    modules_with_direct_meta_tensors = ["model"]
    out_of_model_tensors = {"tensors": ["model.vision_model.embeddings.patch_embedding.weight"]}
    require_trust_remote_code = False
    require_load_processor = True
    support_batch_quantize = False
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    layer_modules_strict = False
    dynamic_expert_index = "n_routed_experts"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()
    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"
    _direct_parameter_names = ("image_newline", "view_separator")
    _base_module_names = (
        "model.vision_tower", "model.multi_modal_projector",
        "model.language_model.embed_tokens", "model.language_model.norm",
        "model.language_model.rotary_emb",
    )

    module_tree = [
        "model", "language_model", "layers", "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate": ("gate:!",),
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {"#": ("gate_proj:0", "up_proj:0", "down_proj:1")},
            },
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        return []

    @classmethod
    def resolve_hf_conversion_map_reversed(cls, target_model=None):
        from ..ports.unlimited_ocr.checkpoint import checkpoint_mapping

        return checkpoint_mapping()

    @staticmethod
    def checkpoint_key_for_load(name):
        # GPT-QModel writes native keys; native HF's unanchored `model.layers`
        # pattern also matches the suffix of `model.language_model.layers`.
        return name

    @staticmethod
    def checkpoint_key_for_save(name):
        return name

    @classmethod
    def get_base_modules(cls, model):
        return [name for name in cls._base_module_names if get_module(model, name) is not None]

    def pre_quantize_generate_hook_start(self):
        for name in self.get_base_modules(self.model):
            self.shell_module_materialize(
                get_module(self.model, name), self.quantize_config.device, module_path=name,
            )
        self._move_direct_parameters(self.quantize_config.device)
        self._restore_vision_position_ids(self.model.model)

    @staticmethod
    def _restore_vision_position_ids(core_model):
        embeddings = core_model.vision_tower.vision_encoder.embeddings
        embeddings.position_ids = torch.arange(
            embeddings.num_positions, device=embeddings.position_embedding.weight.device,
        ).expand((1, -1))

    def _move_direct_parameters(self, device) -> None:
        device = as_torch_device(device)
        core_model = self.model.model
        for name in self._direct_parameter_names:
            parameter = getattr(core_model, name)
            if parameter.device != device:
                setattr(core_model, name, torch.nn.Parameter(
                    parameter.to(device=device), requires_grad=parameter.requires_grad,
                ))

    def pre_quantize_generate_hook_end(self):
        self._move_direct_parameters(CPU)
        for name in self.get_base_modules(self.model):
            parent_name, _, child_name = name.rpartition(".")
            parent = get_module(self.model, parent_name)
            if self.quantize_config.offload_to_disk:
                offload_to_disk(
                    model=parent, module=getattr(parent, child_name),
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            else:
                setattr(parent, child_name, move_to(getattr(parent, child_name), device=CPU))

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def _prepare_image_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = fetch_image({"image": sample["image"]}).convert("RGB")
        prompt = sample.get("text") or "<image>\nFree OCR."
        if prompt.count("<image>") != 1:
            raise ValueError("Unlimited-OCR calibration samples must contain exactly one <image> token.")
        if self.processor is None:
            raise ValueError("Unlimited-OCR calibration requires its native processor.")
        return dict(self.processor(images=image, text=prompt, return_tensors="pt"))

    def _prepare_calibration_batches(self, conversations):
        return [self._prepare_image_sample(batch[0]) for batch in batched(
            conversations, 1, process_func=self.preprocess_dataset,
        )]

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        if not isinstance(calibration_dataset, list):
            calibration_dataset = list(calibration_dataset)
        if not calibration_dataset or not isinstance(calibration_dataset[0], dict):
            return super().prepare_dataset(calibration_dataset, batch_size=batch_size, **kwargs)
        if "input_ids" in calibration_dataset[0]:
            return calibration_dataset
        return self._prepare_calibration_batches(calibration_dataset)

    def move_input_capture_example(self, example, data_device):
        example = super().move_input_capture_example(example, data_device)
        parameter = next(self.model.model.vision_tower.parameters())
        for name in ("pixel_values", "pixel_values_local"):
            if name in example:
                example[name] = example[name].to(device=parameter.device, dtype=parameter.dtype)
        return example


__all__ = ["UnlimitedOCRQModel"]
