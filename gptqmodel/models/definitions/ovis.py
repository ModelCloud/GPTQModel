# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import logging
from typing import Dict

import torch

from ...utils.calibration import batched
from ...utils.image import fetch_image
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..local import LocalModelLoader
from ._vlm_calibration import prepare_vlm_calibration_dataset


class OvisQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.ovis.configuration_ovis.OvisConfig",
        model_class="gptqmodel.models.ports.ovis.modeling_ovis.Ovis",
        processor_class="gptqmodel.models.ports.ovis.processing_ovis.OvisProcessor",
    )
    require_load_processor = True
    require_trust_remote_code = False

    pre_lm_head_norm_module = "llm.model.norm"
    lm_head = "llm.lm_head"
    rotary_embedding = "llm.model.rotary_emb"

    module_tree = [
        "llm",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

    layer_modules_strict = False # the layer modules are in different decode layers

    require_monkeypatch = True

    modality = [MODALITY.IMAGE_TO_TEXT]

    IGNORE_ID = -100

    @classmethod
    def get_base_modules(cls, model):
        return [*super().get_base_modules(model), "visual_tokenizer", "vte"]

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector)
        config = getattr(getattr(model, "config", None), "llm_config", None)
        if getattr(config, "model_type", None) == "gemma2":
            tree[-1]["pre_feedforward_layernorm"] = ("pre_feedforward_layernorm:!",)
            tree[-1]["post_feedforward_layernorm"] = ("post_feedforward_layernorm:!",)
        return tree

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        return []

    def monkey_patch(self):
        self.model.text_tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        self.model.processor = self.processor
        self.model.visual_tokenizer.image_processor = self.processor.image_processor

    def pre_quantize_generate_hook_start(self):
        self.shell_module_materialize(self.model.llm.model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(self.model.llm.model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(self.model.visual_tokenizer, self.quantize_config.device)
        self.shell_module_materialize(self.model.vte, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            offload_to_disk(model=self.model.llm.model,
                            module=self.model.llm.model.embed_tokens,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )
            offload_to_disk(model=self.model.llm.model,
                            module=self.model.llm.model.rotary_emb,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )
            offload_to_disk(model=self.model,
                            module=self.model.visual_tokenizer,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )
            offload_to_disk(model=self.model,
                            module=self.model.vte,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )
            return

        self.model.llm.model.embed_tokens = move_to(self.model.llm.model.embed_tokens, device=CPU)
        self.model.llm.model.rotary_emb = move_to(self.model.llm.model.rotary_emb, device=CPU)
        self.model.visual_tokenizer = move_to(self.model.visual_tokenizer, device=CPU)
        self.model.vte = move_to(self.model.vte, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        text_max_length = 2048
        max_partition = 9
        conversations = copy.deepcopy(sample["conversations"])

        if 'image' not in sample:
            images = []
        else:
            image_paths = sample['image'] if isinstance(sample['image'], list) else [sample['image']]
            images = [fetch_image({'image': path}) for path in image_paths]

        prompt, input_ids, pixel_values, labels = self.model.preprocess_inputs(
            conversations,
            images,
            max_partition=max_partition,
            generation_preface=None,
            return_labels=True,
            propagate_exception=False
        )

        if pixel_values is None:
            pixel_values, _ = self.model.visual_tokenizer.mock_input()

        input_ids = input_ids[:text_max_length]
        labels = labels[:text_max_length]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
        }

    def _prepare_calibration_batches(self, conversations, batch_size):
        calib_data = []
        for batch in batched(conversations, batch_size, self.preprocess_dataset):
            pixel_values, input_ids, labels = tuple([instance[key] for instance in batch]
                                                    for key in ("pixel_values", "input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.model.get_text_tokenizer().pad_token_id)
            attention_mask = torch.ne(input_ids, self.model.get_text_tokenizer().pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=self.IGNORE_ID)

            num_valid_label = torch.not_equal(labels, self.IGNORE_ID).sum().item()
            if num_valid_label == 0:
                logging.warning(
                    f'[DataCollatorForMultimodalDatasetGPTQ] All labels are ignored, may causing training instability\n{input_ids=}\n{attention_mask=}\n{labels=}')
            calib_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values,
            })

        return calib_data

    def prepare_dataset(self,calibration_dataset,batch_size: int = 1, **kwargs):
        return prepare_vlm_calibration_dataset(
            self,
            calibration_dataset,
            batch_size=batch_size,
            options=kwargs,
            prepare_conversations=lambda rows: self._prepare_calibration_batches(rows, batch_size),
        )

    def generate(self, inputs=None, **kwargs):
        """shortcut for model.generate"""
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.generate(inputs, **kwargs)

    def move_input_capture_example(self, example, data_device):
        for key, value in example.items():
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if not torch.is_tensor(item):
                        continue

                    if item.ndim == 1:
                        item = item.unsqueeze(0)

                    value[index] = move_to(
                        item.to(self.model.visual_tokenizer.dtype),
                        device=data_device,
                    )
            elif torch.is_tensor(value):
                if value.ndim == 1:
                    value = value.unsqueeze(0)

                example[key] = move_to(value, device=data_device)

        return self.finalize_input_capture_example(example)

    def run_input_capture(self, example, use_cache: bool, data_device):
        return self.model.generate(inputs=example.pop("input_ids"), **example)
