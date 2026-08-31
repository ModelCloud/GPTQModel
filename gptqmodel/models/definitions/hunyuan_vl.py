# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Dict

from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.looper_helpers import normalize_device_like
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class HunYuanVLQModel(BaseQModel):
    """Quantization definition for native Transformers Hunyuan-VL checkpoints."""

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
            "self_attn": (
                "query_layernorm:!",
                "q_proj:0",
                "key_layernorm:!",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    def _materialize_module(self, parent, name: str, module_path: str):
        module = getattr(parent, name)
        target_device = normalize_device_like(self.quantize_config.device) or CPU
        setattr(
            parent,
            name,
            self.shell_module_materialize(
                module,
                target_device,
                module_path=module_path,
            ),
        )

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        self._materialize_module(
            language_model, "embed_tokens", "model.language_model.embed_tokens"
        )
        self._materialize_module(language_model, "norm", "model.language_model.norm")
        self._materialize_module(
            language_model, "rotary_emb", "model.language_model.rotary_emb"
        )
        self._materialize_module(core_model, "vision_tower", "model.vision_tower")

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        language_model = core_model.language_model
        modules = (
            (language_model, "embed_tokens"),
            (language_model, "norm"),
            (language_model, "rotary_emb"),
            (core_model, "vision_tower"),
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

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(
            self.model_local_path,
            trust_remote_code=False,
            backend="pil",
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
        for batch in batched(
            calibration_dataset,
            batch_size,
            process_func=self.preprocess_dataset,
        ):
            inputs = processor.apply_chat_template(
                batch,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )
            calibration_data.append(inputs)
        del processor
        return calibration_data


__all__ = ["HunYuanVLQModel"]
