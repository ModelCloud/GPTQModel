# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Dict

from transformers import AutoModelForPreTraining, AutoProcessor, ProcessorMixin

from ...utils.audio import process_audio_info
from ...utils.calibration import batched_conversations
from ...utils.model import move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class Qwen2AudioGPTQ(BaseQModel):
    loader = AutoModelForPreTraining

    # The audio tower and projector stay dense; only the Qwen2 text decoder is quantized.
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

    require_load_processor = True

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        self.shell_module_materialize(core_model.audio_tower, self.quantize_config.device)
        self.shell_module_materialize(core_model.multi_modal_projector, self.quantize_config.device)
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.rotary_emb, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        language_model = core_model.language_model
        if self.quantize_config.offload_to_disk:
            for module in (core_model.audio_tower, core_model.multi_modal_projector):
                offload_to_disk(
                    model=core_model,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            for module in (language_model.embed_tokens, language_model.rotary_emb):
                offload_to_disk(
                    model=language_model,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        core_model.audio_tower = move_to(core_model.audio_tower, device=CPU)
        core_model.multi_modal_projector = move_to(core_model.multi_modal_projector, device=CPU)
        language_model.embed_tokens = move_to(language_model.embed_tokens, device=CPU)
        language_model.rotary_emb = move_to(language_model.rotary_emb, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path)

    def prepare_dataset(
        self, calibration_dataset, calibration_dataset_concat_size=None, batch_size: int = 1, **kwargs
    ):
        processor = self.load_processor()
        calib_data = []
        for batch in batched_conversations(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            text = processor.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
            audio_inputs = process_audio_info(batch, use_audio_in_video=False)
            inputs = processor(
                text=text,
                audio=audio_inputs if audio_inputs else None,
                padding=True,
                return_tensors="pt",
            )
            calib_data.append(inputs)
        del processor
        return calib_data


__all__ = ["Qwen2AudioGPTQ"]
