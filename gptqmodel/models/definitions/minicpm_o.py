# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from copy import deepcopy
from typing import Dict, Optional

from transformers import AutoModel, AutoProcessor, ProcessorMixin

from ...utils.audio import process_audio_info
from ...utils.calibration import batched
from ...utils.image import fetch_image
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class MiniCPMOQModel(BaseQModel):
    loader = AutoModel

    pre_lm_head_norm_module = "llm.model.norm"

    require_trust_remote_code = True

    loader_requires_dtype = False

    module_tree = [
        "llm",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:1", "v_proj:2", "o_proj:3"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj", "up_proj", "down_proj"),
        }
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    require_load_processor = True
    require_pkgs = ["audioread>=3.1.0", "librosa>=0.11.0", "av>=16.0.1"]

    @staticmethod
    def _ensure_tts_sampling_fields(obj):
        if obj is None:
            return

        if not hasattr(obj, "top_p"):
            obj.top_p = 1.0
        if not hasattr(obj, "top_k"):
            obj.top_k = 50
        if not hasattr(obj, "topk"):
            obj.topk = getattr(obj, "top_k", 50)

    def before_model_load(self, model_local_path: str, load_quantized_model: bool):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        tts_config_cls = get_class_from_dynamic_module(
            "configuration_minicpmo.MiniCPMTTSConfig",
            model_local_path,
        )

        print("tts_config_cls", tts_config_cls)

        if not hasattr(tts_config_cls, "top_p"):
            tts_config_cls.top_p = 1.0

        if not hasattr(tts_config_cls, "top_k"):
            tts_config_cls.top_k = 50

        # if not hasattr(obj, "top_p"):
        #     obj.top_p = 1.0
        # if not hasattr(obj, "top_k"):
        #     obj.top_k = 50
        # if not hasattr(obj, "topk"):
        #     obj.topk = getattr(obj, "top_k", 50)


        # if not getattr(tts_config_cls, "_gptqmodel_sampling_compat", False):
        #     original_init = tts_config_cls.__init__
        #     original_getattr = getattr(tts_config_cls, "__getattr__", None)
        #
        #     def __init__(self, *args, **kwargs):
        #         original_init(self, *args, **kwargs)
        #         MiniCPMOQModel._ensure_tts_sampling_fields(self)
        #
        #     def __getattr__(self, name):
        #         if name == "top_p":
        #             return 1.0
        #         if name == "top_k":
        #             return getattr(self, "topk", 50)
        #         if name == "topk":
        #             return getattr(self, "top_k", 50)
        #         if original_getattr is not None:
        #             return original_getattr(self, name)
        #         raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        #
        #     tts_config_cls.__init__ = __init__
        #     tts_config_cls.__getattr__ = __getattr__
        #     tts_config_cls._gptqmodel_sampling_compat = True

    def pre_quantize_generate_hook_start(self):
        self.shell_module_materialize(self.model.llm.model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(self.model.llm.model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(self.model.vpm, self.quantize_config.device)
        self.shell_module_materialize(self.model.resampler, self.quantize_config.device)
        self.shell_module_materialize(self.model.apm, self.quantize_config.device)
        self.shell_module_materialize(self.model.audio_avg_pooler, self.quantize_config.device)
        self.shell_module_materialize(self.model.audio_projection_layer, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            for module in (
                self.model.llm.model.embed_tokens,
                self.model.llm.model.rotary_emb,
            ):
                offload_to_disk(
                    model=self.model.llm.model,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )

            for module in (
                self.model.vpm,
                self.model.resampler,
                self.model.apm,
                self.model.audio_avg_pooler,
                self.model.audio_projection_layer,
            ):
                offload_to_disk(
                    model=self.model,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        self.model.llm.model.embed_tokens = move_to(self.model.llm.model.embed_tokens, device=CPU)
        self.model.llm.model.rotary_emb = move_to(self.model.llm.model.rotary_emb, device=CPU)
        self.model.vpm = move_to(self.model.vpm, device=CPU)
        self.model.resampler = move_to(self.model.resampler, device=CPU)
        self.model.apm = move_to(self.model.apm, device=CPU)
        self.model.audio_avg_pooler = move_to(self.model.audio_avg_pooler, device=CPU)
        self.model.audio_projection_layer = move_to(self.model.audio_projection_layer, device=CPU)
        if hasattr(self.model, "tts"):
            self.model.tts = move_to(self.model.tts, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=True)

    @staticmethod
    def _normalize_conversation(
        conversation: list[dict],
    ) -> tuple[list[dict], list, list[int]]:
        normalized = []
        images = []
        audio_parts = []

        for index, message in enumerate(deepcopy(conversation)):
            content = message.get("content", "")
            if isinstance(content, str):
                normalized.append(message)
                continue

            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue

                item_type = item.get("type")
                if item_type == "image":
                    images.append(fetch_image(item))
                    text_parts.append("<image>./</image>")
                elif item_type == "audio":
                    audio_parts.append(index)
                    text_parts.append("<audio>./</audio>")
                elif item_type == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    raise ValueError(f"Unsupported MiniCPM-o content type: {item_type}")

            message["content"] = "\n".join(part for part in text_parts if part)
            normalized.append(message)

        return normalized, images, audio_parts

    @classmethod
    def prepare_inputs_for_conversations(
        cls,
        processor: ProcessorMixin,
        conversations: list[dict] | list[list[dict]],
    ):
        if conversations and isinstance(conversations[0], dict):
            conversations = [conversations]

        prompts = []
        images = []
        audios = []
        audio_parts = []

        for conversation in conversations:
            normalized, image_inputs, audio_part_inputs = cls._normalize_conversation(conversation)
            audio_inputs = process_audio_info(conversation, use_audio_in_video=False) or []

            prompts.append(
                processor.tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                )
            )
            images.append(image_inputs)
            audios.append(audio_inputs)
            audio_parts.append(audio_part_inputs)

        inputs = processor(
            prompts,
            images,
            audios,
            audio_parts,
            return_tensors="pt",
        )
        return inputs

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        processor = self.processor or self.load_processor()
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            calib_data.append(
                self.prepare_inputs_for_conversations(
                    processor,
                    batch,
                )
            )
        return calib_data
