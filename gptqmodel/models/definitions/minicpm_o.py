# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from copy import deepcopy
from typing import Dict

from transformers import ProcessorMixin

from ...utils.audio import process_audio_info
from ...utils.calibration import batched_conversations
from ...utils.image import fetch_image
from ...utils.model import MODALITY, move_to, nested_move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..local import LocalModelLoader
from ._vlm_calibration import (
    finalize_conversation_batches,
    pop_conversation_calibration_options,
    split_calibration_dataset,
)


class MiniCPMOQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.minicpm_o.configuration_minicpmo.MiniCPMOConfig",
        model_class="gptqmodel.models.ports.minicpm_o.modeling_minicpmo.MiniCPMO",
        processor_class="gptqmodel.models.ports.minicpm_o.processing_minicpmo.MiniCPMOProcessor",
        tokenizer_class="gptqmodel.models.ports.minicpm_o.tokenization_minicpmo_fast.MiniCPMOTokenizerFast",
    )

    pre_lm_head_norm_module = "llm.model.norm"

    require_trust_remote_code = False
    rotary_embedding = "llm.model.rotary_emb"
    modules_with_direct_meta_tensors = ["resampler"]

    module_tree = [
        "llm",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # Q/K/V are parallel consumers of `input_layernorm`, and `o_proj`
            # consumes the attention output. Sequential stages made the norm
            # fold into q only and then q into k. The MLP stages were also
            # merged, so `down_proj` shared the gate/up GPTQ subset.
            # The per-head q/k norms read the q/k outputs, so they precede
            # their projections here (Qwen3 order): AWQ takes the last entry of
            # the q/k/v stage as the `o_proj` fold source, which must be v_proj.
            "self_attn": (
                "q_norm:!",
                "q_proj:0:in=x",
                "k_norm:!",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
        },
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    require_load_processor = True

    def awq_module_replay_inputs(self, module, x, module_kwargs):
        x, module_kwargs = super().awq_module_replay_inputs(module, x, module_kwargs)
        # Input capture runs the omni `generate` prefill, so the captured decoder
        # kwargs carry its DynamicCache. Layer replay already drops it; the AWQ
        # block replay must too, or every scored scale appends K/V to that cache
        # and the next replay attends over stale keys (mask/key length mismatch).
        cache_keys = ("past_key_values", "past_key_value")
        if any(module_kwargs.get(key) is not None for key in cache_keys):
            module_kwargs = {key: None if key in cache_keys else value for key, value in module_kwargs.items()}
            if "use_cache" in module_kwargs:
                module_kwargs["use_cache"] = False
        return x, module_kwargs

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
        return self.loader.load_processor(
            self.model_local_path,
            config=self.model.config,
            tokenizer=self.tokenizer.tokenizer if self.tokenizer is not None else None,
        )

    @staticmethod
    def _normalize_conversation(
        conversation: list[dict],
        version: str = "4.5",
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
                    text_parts.append("(<image>./</image>)" if str(version) == "2.6" else "<image>./</image>")
                elif item_type == "audio":
                    audio_parts.append(index)
                    text_parts.append("(<audio>./</audio>)" if str(version) == "2.6" else "<audio>./</audio>")
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
            normalized, image_inputs, audio_part_inputs = cls._normalize_conversation(conversation, processor.version)
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
        inputs.pop("image_sizes")
        return inputs

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        label = type(self).__name__
        pretokenized, conversations = split_calibration_dataset(calibration_dataset, label=label)
        if pretokenized is not None:
            return super().prepare_dataset(pretokenized, batch_size=batch_size, **kwargs)

        sort, min_length = pop_conversation_calibration_options(kwargs, label=label)
        processor = self.processor or self.load_processor()
        calib_data = []
        for batch in batched_conversations(conversations, batch_size, process_func=self.preprocess_dataset):
            calib_data.append(
                self.prepare_inputs_for_conversations(
                    processor,
                    batch,
                )
            )
        return finalize_conversation_batches(calib_data, sort=sort, min_length=min_length, label=label)

    def move_input_capture_example(self, example, data_device):
        for key, value in example.items():
            example[key] = nested_move_to(value, device=data_device)

        return self.finalize_input_capture_example(example)

    def run_input_capture(self, example, use_cache: bool, data_device):
        generation_config = self.model.prepare_generation_config(do_sample=False, max_new_tokens=1)
        generation_config["use_cache"] = use_cache

        return self.model.generate(
            **example,
            tokenizer=self.tokenizer.tokenizer,
            **generation_config,
        )
