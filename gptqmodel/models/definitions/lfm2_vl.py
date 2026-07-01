# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Dict

from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.model import MODALITY, move_to, nested_move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class LFM2VLQModel(BaseQModel):
    loader = AutoModelForImageTextToText

    pre_lm_head_norm_module = "model.language_model.embedding_norm"
    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "operator_norm": ("operator_norm:!",),
            "conv": ("in_proj:0", "out_proj:1"),
            "self_attn": (
                "q_proj:0",
                "q_layernorm:0:!",
                "k_proj:0",
                "k_layernorm:0:!",
                "v_proj:0",
                "out_proj:1",
            ),
            "ffn_norm": ("ffn_norm:!",),
            "feed_forward": ("w1:0", "w3:0", "w2:1"),
        },
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    require_load_processor = True
    require_trust_remote_code = False
    layer_modules_strict = False

    def pre_quantize_generate_hook_start(self):
        language_model = self.model.model.language_model
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(language_model.embedding_norm, self.quantize_config.device)
        self.shell_module_materialize(self.model.model.vision_tower, self.quantize_config.device)
        self.shell_module_materialize(self.model.model.multi_modal_projector, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        language_model = self.model.model.language_model
        if self.quantize_config.offload_to_disk:
            offload_to_disk(
                model=language_model,
                module=language_model.embed_tokens,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=language_model,
                module=language_model.rotary_emb,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=language_model,
                module=language_model.embedding_norm,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=self.model.model,
                module=self.model.model.vision_tower,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=self.model.model,
                module=self.model.model.multi_modal_projector,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        language_model.embed_tokens = move_to(language_model.embed_tokens, device=CPU)
        language_model.rotary_emb = move_to(language_model.rotary_emb, device=CPU)
        language_model.embedding_norm = move_to(language_model.embedding_norm, device=CPU)
        self.model.model.vision_tower = move_to(self.model.model.vision_tower, device=CPU)
        self.model.model.multi_modal_projector = move_to(self.model.model.multi_modal_projector, device=CPU)

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
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            calib_data.append(self.prepare_inputs_for_conversations(processor, batch))
        del processor
        return calib_data

    def move_input_capture_example(self, example, data_device):
        for key, value in example.items():
            example[key] = nested_move_to(value, device=data_device)

        return self.finalize_input_capture_example(example)
    #
    # def run_input_capture(self, example, use_cache: bool, data_device):
    #     return self.model.generate(**example, max_new_tokens=1)


__all__ = ["LFM2VLQModel"]
