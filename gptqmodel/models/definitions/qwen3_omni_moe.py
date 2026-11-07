# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
from typing import Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForTextToWaveform, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import (
    MODALITY,
    thinker_multimodal_model,
    thinker_text_model,
    thinker_text_prefix,
)
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


_THINKER_LAYER_MODULES = {
    "input_layernorm": ("input_layernorm:!",),
    "self_attn": ("q_norm:!", "k_norm:!", "q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
    "post_attention_layernorm": ("post_attention_layernorm:!",),
    "mlp:moe": {
        "gate": ("gate:!",),  # router gate is tiny and accuracy-sensitive
        "experts": {
            "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    },
}


class Qwen3OmniMoeGPTQ(BaseQModel):
    fusable = FusableConfig("qwen3_omni_moe").merge(LlamaFusableConfig(), profiles=("qkv",))

    ATTENTION_MASKS_REQUIRED_FOR_INPUT = True
    ATTENTION_MASKS_DTYPE = torch.long

    # The thinker decoder is calibrated on activations that include the vision
    # tower's embeddings, so text-only calibration is not representative.
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    INPUT_EMBEDDING_EXTRA_ARGS = {
        "return_audio": False,
    }

    loader = AutoModelForTextToWaveform

    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "thinker.model.norm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # transformers >= 5.22 nests the thinker decoder under `language_model`.
    module_tree = [
        ["thinker", "model", "layers", "#", _THINKER_LAYER_MODULES],
        ["thinker", "model", "language_model", "layers", "#", _THINKER_LAYER_MODULES],
    ]

    @staticmethod
    def process_vision_info(conversations) -> Optional[list[Image.Image]]:
        image_inputs = []
        for vision_info in extract_vision_info(conversations):
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            else:
                raise ValueError(
                    "Qwen3-Omni calibration supports image content only."
                )
        return image_inputs or None

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path)

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        processor = self.__dict__.get("processor") or self.load_processor()
        calib_data = []
        for batch_index, batch in enumerate(
            batched_conversations(
                calibration_dataset,
                batch_size,
                process_func=self.preprocess_dataset,
            )
        ):
            try:
                text = processor.apply_chat_template(
                    batch, tokenize=False, add_generation_prompt=True
                )
                image_inputs = self.process_vision_info(batch)
                inputs = processor(
                    text=text,
                    images=image_inputs,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                )
            except Exception as exc:
                raise ValueError(
                    f"Qwen3-Omni calibration preparation failed for batch {batch_index}: {exc}"
                ) from exc

            if "input_ids" not in inputs:
                raise ValueError(
                    f"Qwen3-Omni processor output for batch {batch_index} is missing input_ids."
                )
            if image_inputs is not None:
                missing = [
                    name
                    for name in ("pixel_values", "image_grid_thw")
                    if inputs.get(name) is None
                ]
                if missing:
                    raise ValueError(
                        f"Qwen3-Omni processor output for batch {batch_index} "
                        f"is missing required field(s): {', '.join(missing)}."
                    )
            calib_data.append(inputs)

        del processor
        return calib_data

    def pre_quantize_generate_hook_start(self):
        spk_path = os.path.join(self.model_local_path, "spk_dict.pt")
        if os.path.isfile(spk_path):
            self.model.load_speakers(spk_path)

        text_model = thinker_text_model(self.model)
        mm_model = thinker_multimodal_model(self.model)
        self.shell_module_materialize(text_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(mm_model.visual, self.quantize_config.device)
        self.shell_module_materialize(mm_model.audio_tower, self.quantize_config.device)
        self.shell_module_materialize(mm_model.visual.rotary_pos_emb, self.quantize_config.device)
        self.shell_module_materialize(text_model.rotary_emb, self.quantize_config.device)
        if hasattr(self.model, "talker"):
            self.shell_module_materialize(self.model.talker, self.quantize_config.device)
        if hasattr(self.model, "code2wav"):
            self.shell_module_materialize(self.model.code2wav, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        text_model = thinker_text_model(self.model)
        mm_model = thinker_multimodal_model(self.model)
        if self.quantize_config.offload_to_disk:
            offload_to_disk(model=text_model,
                            module=text_model.embed_tokens,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            offload_to_disk(model=mm_model,
                            module=mm_model.visual,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            offload_to_disk(model=mm_model,
                            module=mm_model.audio_tower,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            offload_to_disk(model=mm_model,
                            module=mm_model.visual.rotary_pos_emb,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            offload_to_disk(model=text_model,
                            module=text_model.rotary_emb,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            if hasattr(self.model, "talker"):
                offload_to_disk(model=self.model,
                                module=self.model.talker,
                                disk_path=self.quantize_config.offload_to_disk_path,
                                )
            if hasattr(self.model, "code2wav"):
                offload_to_disk(model=self.model,
                                module=self.model.code2wav,
                                disk_path=self.quantize_config.offload_to_disk_path,
                                )
            return

        text_model.embed_tokens = text_model.embed_tokens.to(CPU)
        mm_model.visual = mm_model.visual.to(CPU)
        mm_model.audio_tower = mm_model.audio_tower.to(CPU)
        if hasattr(self.model, "talker"):
            self.model.talker = self.model.talker.to(CPU)
        if hasattr(self.model, "code2wav"):
            self.model.code2wav = self.model.code2wav.to(CPU)

        mm_model.visual.rotary_pos_emb = mm_model.visual.rotary_pos_emb.to(CPU)
        text_model.rotary_emb = text_model.rotary_emb.to(CPU)

    def after_model_load(self, model, load_quantized_model=False):
        # The thinker decoder lives under `thinker.model` before transformers
        # 5.22 and under `thinker.model.language_model` after it.
        text_prefix = thinker_text_prefix(model)
        if text_prefix:
            self.pre_lm_head_norm_module = f"{text_prefix}.norm"
        # need to load processor for save processor_config and chat_template
        if not load_quantized_model:
            self.processor = AutoProcessor.from_pretrained(self.model_local_path)

        return model
