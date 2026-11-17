# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
from types import SimpleNamespace
from typing import Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForTextToWaveform, AutoProcessor, PreTrainedModel, ProcessorMixin

from ...utils.audio import process_audio_info
from ...utils.calibration import batched_conversations
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY, thinker_multimodal_model, thinker_text_model, thinker_text_prefix
from ...utils.offload import offload_to_disk
from ...utils.structure import LazyTurtle
from .._const import CPU
from ..base import BaseQModel
from ._vlm_calibration import (
    finalize_conversation_batches,
    pop_conversation_calibration_options,
    split_calibration_dataset,
)


_THINKER_LAYER_MODULES = {
    "input_layernorm": ("input_layernorm:!",),
    "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
    "post_attention_layernorm": ("post_attention_layernorm:!",),
    "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
}


def _patch_qwen2_5_omni_talker_prepare_inputs_for_generation(talker_cls):
    original = getattr(talker_cls, "prepare_inputs_for_generation", None)
    if original is None or getattr(original, "_gptqmodel_qwen2_5_omni_compat", False):
        return

    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_text_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        thinker_reply_part=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_audio_features=None,
        audio_feature_attention_mask=None,
        audio_feature_lengths=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super(talker_cls, self).prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            thinker_reply_part=thinker_reply_part,
            input_text_ids=input_text_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_audio_in_video=use_audio_in_video,
            audio_feature_lengths=audio_feature_lengths,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )
        model_inputs["position_ids"] = None
        return model_inputs

    prepare_inputs_for_generation._gptqmodel_qwen2_5_omni_compat = True
    talker_cls.prepare_inputs_for_generation = prepare_inputs_for_generation


class BaseQwen2_5_OmniGPTQ(BaseQModel):
    ATTENTION_MASKS_REQUIRED_FOR_INPUT = True
    ATTENTION_MASKS_DTYPE = torch.long
    require_monkeypatch = True

    # Transformers 5.22 registers the checkpoint -> runtime renames for the
    # Qwen2.5-Omni thinker (``thinker.visual`` -> ``thinker.model.visual``, and
    # likewise for the audio tower and ``language_model``) under the nested
    # ``qwen2_5_omni_thinker`` model type, not the top-level ``qwen2_5_omni``.
    # Resolving the top-level type therefore finds no conversion map, and the
    # lazy disk loader leaves the towers and ``embed_tokens`` on META. Declare
    # the runtime -> checkpoint aliases explicitly so both layouts resolve.
    HF_CONVERSION_MAP_REVERSED = (
        SimpleNamespace(
            source_patterns=[r"^thinker\.model\.visual\.(.+)$"],
            target_patterns=[r"^thinker.visual.\1"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"^thinker\.model\.audio_tower\.(.+)$"],
            target_patterns=[r"^thinker.audio_tower.\1"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"^thinker\.model\.language_model\.(.+)$"],
            target_patterns=[r"^thinker.model.\1"],
            operations=[],
        ),
    )

    @classmethod
    def checkpoint_key_for_save(cls, name: str) -> str:
        """Write checkpoint keys in the source HF layout (issue #4758).

        Transformers 5.22 keeps the thinker towers under ``thinker.model.*``
        and the decoder under ``thinker.model.language_model.*`` in memory, but
        a Qwen2.5-Omni checkpoint stores ``thinker.audio_tower.*``,
        ``thinker.visual.*`` and ``thinker.model.embed_tokens.*``. Saving the
        in-memory names wrote a checkpoint the loader could not map back, so
        the towers and embeddings reloaded as ``torch.empty`` garbage.
        """
        return LazyTurtle.convert_hf_checkpoint_key(name, cls.HF_CONVERSION_MAP_REVERSED)

    @classmethod
    def checkpoint_key_for_load(cls, name: str) -> str:
        """Map a source-layout checkpoint key back onto the runtime module."""
        return LazyTurtle.convert_hf_checkpoint_key(
            name,
            cls.HF_CONVERSION_MAP_REVERSED,
            to_runtime=True,
        )

    loader = AutoModelForTextToWaveform

    lm_head = "thinker.lm_head"
    # transformers 5.22 nests the thinker decoder under `language_model`.
    pre_lm_head_norm_module = [
        "thinker.model.language_model.norm",
        "thinker.model.norm",
    ]

    require_pkgs = ["audioread>=3.1.0", "librosa>=0.11.0", "av>=16.0.1"]

    # transformers >= 5.22 nests the thinker decoder under `language_model`.
    module_tree = [
        ["thinker", "model", "layers", "#", _THINKER_LAYER_MODULES],
        ["thinker", "model", "language_model", "layers", "#", _THINKER_LAYER_MODULES],
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    def before_model_load(self, model_local_path: str, load_quantized_model: bool):
        from transformers import Qwen2_5OmniForConditionalGeneration

        # The quantized loader discovers endpoints before constructing our
        # wrapper. Supply the missing HF getters there as well as on the wrapper.
        model_cls = Qwen2_5OmniForConditionalGeneration
        if model_cls.get_input_embeddings is PreTrainedModel.get_input_embeddings:
            model_cls.get_input_embeddings = lambda model: model.thinker.get_input_embeddings()
        if model_cls.get_output_embeddings is PreTrainedModel.get_output_embeddings:
            model_cls.get_output_embeddings = lambda model: model.thinker.lm_head

    def get_input_embeddings(self):
        return self.model.thinker.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.thinker.lm_head

    def after_model_load(self, model, load_quantized_model=False):
        # The thinker decoder lives under `thinker.model` before transformers
        # 5.22 and under `thinker.model.language_model` after it.
        text_prefix = thinker_text_prefix(model)
        if text_prefix:
            self.pre_lm_head_norm_module = f"{text_prefix}.norm"
        return model

    def run_input_capture(self, example, use_cache, data_device):
        # Calibration only needs the thinker's forward pass. The speech
        # generation wrapper requires a talker even for text-only checkpoints.
        return self.model.thinker(**example, use_cache=use_cache)

    def monkey_patch(self):
        talker = getattr(getattr(self, "model", None), "talker", None)
        if talker is None:
            return

        if not getattr(type(talker), "__module__", "").startswith("transformers.models.qwen2_5_omni."):
            return

        _patch_qwen2_5_omni_talker_prepare_inputs_for_generation(type(talker))

    def pre_quantize_generate_hook_start(self):
        # load speaker
        spk_path = os.path.join(self.model_local_path, "spk_dict.pt")
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
        if hasattr(self.model, "token2wav"):
            self.shell_module_materialize(self.model.token2wav, self.quantize_config.device)
        for layer in text_model.layers:
            if hasattr(layer.self_attn, "rotary_emb"):
                self.shell_module_materialize(layer.self_attn.rotary_emb, self.quantize_config.device)

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
            if hasattr(self.model, "token2wav"):
                offload_to_disk(model=self.model,
                                module=self.model.token2wav,
                                disk_path=self.quantize_config.offload_to_disk_path,
                                )

            for layer in text_model.layers:
                if hasattr(layer.self_attn, "rotary_emb"):
                    offload_to_disk(model=text_model,
                                    module=layer.self_attn.rotary_emb,
                                    disk_path=self.quantize_config.offload_to_disk_path,
                                    )

            return

        text_model.embed_tokens = text_model.embed_tokens.to(CPU)
        mm_model.visual = mm_model.visual.to(CPU)
        mm_model.audio_tower = mm_model.audio_tower.to(CPU)
        if hasattr(self.model, "talker"):
            self.model.talker = self.model.talker.to(CPU)
        if hasattr(self.model, "token2wav"):
            self.model.token2wav = self.model.token2wav.to(CPU)

        mm_model.visual.rotary_pos_emb = mm_model.visual.rotary_pos_emb.to(CPU)
        text_model.rotary_emb = text_model.rotary_emb.to(CPU)

        for layer in text_model.layers:
            if hasattr(layer.self_attn, "rotary_emb"):
                layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(CPU)

    @staticmethod
    def process_vision_info(
            conversations: list[dict] | list[list[dict]],
    ) -> Optional[list[Image.Image]]:
        vision_infos = extract_vision_info(conversations)
        # Read images
        image_inputs = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            else:
                raise ValueError("image, image_url should in content.")
        if len(image_inputs) == 0:
            image_inputs = None
        return image_inputs

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def forward(self, *args, **kwargs):
        """Delegate textual forward passes to the thinker submodule.

        The top-level Hugging Face wrapper leaves ``forward`` unimplemented when
        ``trust_remote_code`` is disabled, so we expose the thinker equivalent to
        keep tooling such as lm-eval operational in quantized environments.
        """

        return self.model.thinker(*args, **kwargs)

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path)

    def prepare_dataset(self, calibration_dataset, calibration_dataset_concat_size=None, batch_size: int = 1, **kwargs):
        label = type(self).__name__
        pretokenized, conversations = split_calibration_dataset(calibration_dataset, label=label)
        if pretokenized is not None:
            return super().prepare_dataset(
                pretokenized,
                calibration_dataset_concat_size=calibration_dataset_concat_size,
                batch_size=batch_size,
                **kwargs,
            )

        kwargs["calibration_dataset_concat_size"] = calibration_dataset_concat_size
        sort, min_length = pop_conversation_calibration_options(kwargs, label=label)
        processor = self.load_processor()
        calib_data = []
        for batch in batched_conversations(conversations, batch_size, process_func=self.preprocess_dataset):
            text = processor.apply_chat_template(
                batch, tokenize=False, add_generation_prompt=True
            )
            image_inputs = self.process_vision_info(batch)
            audio_inputs = process_audio_info(batch, use_audio_in_video=False)
            inputs = processor(
                text=text,
                images=image_inputs,
                audio=audio_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            calib_data.append(inputs)
        del processor
        return finalize_conversation_batches(
            calib_data, sort=sort, min_length=min_length, label=label
        )
