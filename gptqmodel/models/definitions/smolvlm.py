# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Dict

from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class SmolVLMQModel(BaseQModel):
    loader = AutoModelForImageTextToText

    pre_lm_head_norm_module = "model.text_model.norm"
    rotary_embedding = "model.text_model.rotary_emb"

    module_tree = [
        "model",
        "text_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    require_load_processor = True
    require_trust_remote_code = False
    # `SmolVLMProcessor.__init__` in transformers raises without `num2words`, so
    # surface the dependency here instead of failing deep inside the processor.
    require_pkgs = ["num2words"]

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        text_model = core_model.text_model
        self.shell_module_materialize(text_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(text_model.norm, self.quantize_config.device)
        self.shell_module_materialize(text_model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(core_model.vision_model, self.quantize_config.device)
        self.shell_module_materialize(core_model.connector, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        text_model = core_model.text_model
        if self.quantize_config.offload_to_disk:
            for module in (text_model.embed_tokens, text_model.norm, text_model.rotary_emb):
                offload_to_disk(
                    model=text_model,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            for module in (core_model.vision_model, core_model.connector):
                offload_to_disk(
                    model=core_model,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        text_model.embed_tokens = move_to(text_model.embed_tokens, device=CPU)
        text_model.norm = move_to(text_model.norm, device=CPU)
        text_model.rotary_emb = move_to(text_model.rotary_emb, device=CPU)
        core_model.vision_model = move_to(core_model.vision_model, device=CPU)
        core_model.connector = move_to(core_model.connector, device=CPU)

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
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        processor = self.load_processor()
        calib_data = []
        for batch in batched_conversations(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            calib_data.append(self.prepare_inputs_for_conversations(processor, batch))
        del processor
        return calib_data


__all__ = ["SmolVLMQModel"]
