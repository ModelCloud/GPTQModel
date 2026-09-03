# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForMultimodalLM, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.looper_helpers import normalize_device_like
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class ApertusQModel(BaseQModel):
    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "attention_layernorm": ("attention_layernorm:!"),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "feedforward_layernorm": ("feedforward_layernorm:!"),
            "mlp": ("up_proj:0", "down_proj:1"),
        }
    ]


class Apertus1p5TextQModel(ApertusQModel):
    """Text-only Apertus 1.5 checkpoints use the original Apertus decoder layout."""


class Apertus1p5QModel(BaseQModel):
    """Apertus 1.5 multimodal wrapper with an Apertus language backbone."""

    loader = AutoModelForMultimodalLM
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
            "attention_layernorm": ("attention_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "q_norm:!",
                "k_norm:!",
                "o_proj:1",
            ),
            "feedforward_layernorm": ("feedforward_layernorm:!",),
            "mlp": ("up_proj:0", "down_proj:1"),
        },
    ]

    def _materialize_module(self, parent, name: str, module_path: str):
        target_device = normalize_device_like(self.quantize_config.device) or CPU
        setattr(
            parent,
            name,
            self.shell_module_materialize(
                getattr(parent, name),
                target_device,
                module_path=module_path,
            ),
        )

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        modules = (
            (language_model, "embed_tokens", "model.language_model.embed_tokens"),
            (language_model, "norm", "model.language_model.norm"),
            (language_model, "rotary_emb", "model.language_model.rotary_emb"),
            (core_model, "vision_tokenizer", "model.vision_tokenizer"),
            (core_model, "audio_tokenizer", "model.audio_tokenizer"),
        )
        for parent, name, module_path in modules:
            self._materialize_module(parent, name, module_path)

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        language_model = core_model.language_model
        modules = (
            (language_model, "embed_tokens"),
            (language_model, "norm"),
            (language_model, "rotary_emb"),
            (core_model, "vision_tokenizer"),
            (core_model, "audio_tokenizer"),
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

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=False)

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del kwargs
        processor = self.processor or self.load_processor()
        calibration_data = []
        for batch in batched(calibration_dataset, batch_size):
            calibration_data.append(
                processor.apply_chat_template(
                    batch,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={"padding": True},
                )
            )
        return calibration_data


__all__ = ["Apertus1p5QModel", "Apertus1p5TextQModel", "ApertusQModel"]
