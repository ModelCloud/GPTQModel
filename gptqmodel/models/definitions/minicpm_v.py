# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from copy import deepcopy
from typing import Dict

from PIL import Image
from transformers import AutoModel, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.image import fetch_image
from ...utils.model import MODALITY, move_to, nested_move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


def _allow_minicpmv_remote_tokenizer() -> None:
    try:
        from transformers.models.auto import tokenization_auto
    except Exception:
        return

    incompatible = getattr(tokenization_auto, "MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS", None)
    if isinstance(incompatible, set):
        incompatible.discard("minicpmv")

class MiniCPMVQModel(BaseQModel):
    loader = AutoModel

    pre_lm_head_norm_module = "llm.model.norm"

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
    require_trust_remote_code = True

    def before_model_load(self, model_local_path: str, load_quantized_model: bool):
        _allow_minicpmv_remote_tokenizer()

    def pre_quantize_generate_hook_start(self):
        self.shell_module_materialize(self.model.llm.model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(self.model.llm.model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(self.model.vpm, self.quantize_config.device)
        self.shell_module_materialize(self.model.resampler, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            offload_to_disk(
                model=self.model.llm.model,
                module=self.model.llm.model.embed_tokens,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=self.model.llm.model,
                module=self.model.llm.model.rotary_emb,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=self.model,
                module=self.model.vpm,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=self.model,
                module=self.model.resampler,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        self.model.llm.model.embed_tokens = move_to(self.model.llm.model.embed_tokens, device=CPU)
        self.model.llm.model.rotary_emb = move_to(self.model.llm.model.rotary_emb, device=CPU)
        self.model.vpm = move_to(self.model.vpm, device=CPU)
        self.model.resampler = move_to(self.model.resampler, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=True)

    @staticmethod
    def _normalize_conversation(
        conversation: list[dict],
    ) -> tuple[list[dict], list[Image.Image]]:
        normalized = []
        images = []

        for message in deepcopy(conversation):
            content = message.get("content", "")
            if isinstance(content, str):
                normalized.append(message)
                continue

            cur_msgs = []
            for item in content:
                if isinstance(item, str):
                    cur_msgs.append(item)
                    continue

                item_type = item.get("type")
                if item_type == "image":
                    images.append(fetch_image(item))
                    cur_msgs.append("(<image>./</image>)")
                elif item_type == "text":
                    cur_msgs.append(item.get("text", ""))
                else:
                    raise ValueError(f"Unsupported MiniCPM-V content type: {item_type}")

            message["content"] = "\n".join(cur_msgs)
            normalized.append(message)

        return normalized, images

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
        for conversation in conversations:
            normalized, image_inputs = cls._normalize_conversation(conversation)
            prompts.append(
                processor.tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            images.append(image_inputs)

        inputs = processor(
            prompts,
            images,
            return_tensors="pt",
        )
        inputs.pop("image_sizes")
        return inputs

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        processor = self.load_processor()
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            calib_data.append(
                self.prepare_inputs_for_conversations(
                    processor,
                    batch,
                )
            )
        del processor
        return calib_data

    def move_input_capture_example(self, example, data_device):
        for key, value in example.items():
            example[key] = nested_move_to(value, device=data_device)

        return self.finalize_input_capture_example(example)

    def run_input_capture(self, example, use_cache: bool, data_device):
        generation_config = {
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 100,
            "repetition_penalty": 1.03,
            "use_cache": use_cache,
        }

        return self.model.generate(
            **example,
            tokenizer=self.model.tokenizer,
            **generation_config,
        )
