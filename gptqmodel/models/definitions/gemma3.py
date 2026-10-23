# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import itertools
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Dict

import torch
from transformers import AutoProcessor, ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from . import LlamaQModel


class Gemma3QModel(LlamaQModel):
    shared_input_verified_model_types = frozenset({"gemma3_text"})


class Gemma3ForConditionalGenerationGPTQ(BaseQModel):
    support_batch_quantize = False
    pre_lm_head_norm_module = "model.language_model.norm"

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    require_load_processor = True

    @staticmethod
    def checkpoint_key_for_save(name: str) -> str:
        """Write Transformers 5 shell names using Gemma 3's HF checkpoint names."""
        if name.startswith("model.language_model."):
            suffix = name[len("model.language_model."):]
            if not suffix.startswith("model."):
                suffix = "model." + suffix
            return "language_model." + suffix
        if name.startswith("model.vision_tower."):
            suffix = name[len("model.vision_tower."):]
            if not suffix.startswith("vision_model."):
                suffix = "vision_model." + suffix
            return "vision_tower." + suffix
        if name.startswith("model.multi_modal_projector."):
            return name[len("model."):]
        return name

    @staticmethod
    def checkpoint_key_for_load(name: str) -> str:
        """Map saved Hugging Face Gemma 3 names back to Transformers 5 shell names."""
        if name.startswith("language_model.model."):
            return "model.language_model." + name[len("language_model.model."):]
        if name.startswith("vision_tower.vision_model."):
            return "model.vision_tower." + name[len("vision_tower.vision_model."):]
        if name.startswith("multi_modal_projector."):
            return "model." + name
        return name

    def prepare_quantize_config_for_save(self, quantize_config) -> None:
        # Gemma 3's vision tower and projector remain in their source dtype.
        # Tell vLLM's GPTQ loader not to allocate quantized linear layers for them.
        dynamic = dict(quantize_config.dynamic or {})
        dynamic.setdefault(r"-:vision_tower\..*", {})
        dynamic.setdefault(r"-:multi_modal_projector\..*", {})
        quantize_config.dynamic = dynamic

    def copy_checkpoint_auxiliary_files_for_save(self, save_dir: str) -> None:
        # Keep the standalone processor/tokenizer files required by Transformers 4.x
        # and vLLM releases that predate the consolidated Transformers 5 format.
        import os
        import shutil

        filenames = (
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "added_tokens.json",
            "chat_template.json",
        )
        for filename in filenames:
            source = os.path.join(self.model_local_path, filename)
            if os.path.isfile(source):
                shutil.copy2(source, os.path.join(save_dir, filename))

    HF_CONVERSION_MAP_REVERSED = (
        # Gemma 3 shells expose `model.vision_tower.*`, but checkpoint tensors
        # live under `vision_tower.vision_model.*`, so restore that missing hop first.
        SimpleNamespace(
            source_patterns=[r"^model\.vision_tower\.(?!vision_model\.)(.+)$"],
            target_patterns=[r"^vision_tower.vision_model.\1"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"model\.language_model"],
            target_patterns=[r"^language_model.model"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"lm_head"],
            target_patterns=[r"^language_model.lm_head"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"model\.vision_tower"],
            target_patterns=[r"^vision_tower"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"model\.multi_modal_projector"],
            target_patterns=[r"^multi_modal_projector"],
            operations=[],
        ),
    )

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
        }
    ]

    lm_head_module = "model.lm_head"

    def pre_quantize_generate_hook_start(self):
        language_model = self.model.model.language_model
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(language_model.norm, self.quantize_config.device)
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
                module=language_model.norm,
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
        language_model.norm = move_to(language_model.norm, device=CPU)
        self.model.model.vision_tower = move_to(self.model.model.vision_tower, device=CPU)
        self.model.model.multi_modal_projector = move_to(self.model.model.multi_modal_projector, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=False)

    @staticmethod
    def _is_pretokenized_sample(sample) -> bool:
        if isinstance(sample, Mapping):
            return "input_ids" in sample and sample.keys() <= {"input_ids", "attention_mask"}
        if isinstance(sample, (list, tuple)):
            return bool(sample) and all(isinstance(token, int) for token in sample)
        return torch.is_tensor(sample)

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
        iterator = iter(calibration_dataset)
        try:
            first_sample = next(iterator)
        except StopIteration:
            return super().prepare_dataset([], batch_size=batch_size, **kwargs)

        samples = itertools.chain((first_sample,), iterator)
        if self._is_pretokenized_sample(first_sample):
            samples = (dict(sample) if isinstance(sample, Mapping) else sample for sample in samples)
            return super().prepare_dataset(samples, batch_size=batch_size, **kwargs)

        processor = self.load_processor()
        calib_data = []
        for batch in batched_conversations(samples, batch_size, process_func=self.preprocess_dataset):
            calib_data.append(self.prepare_inputs_for_conversations(processor, batch))
        del processor
        return calib_data
