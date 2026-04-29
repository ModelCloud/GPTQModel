# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin
from transformers.masking_utils import create_causal_mask

from ...utils.calibration import batched
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY, get_module, get_module_by_name_prefix, move_to
from ...utils.offload import offload_to_disk
from ...utils.structure import print_module_tree
from .._const import CPU
from ..base import BaseQModel


class Ernie4_5_VLMoeQModel(BaseQModel):
    loader = AutoModelForImageTextToText

    require_load_processor = True
    require_trust_remote_code = False
    layer_modules_strict = False

    pre_lm_head_norm_module = ["model.norm", "model.language_model.norm", "language_model.norm"]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate_proj": ("gate_proj",),
                "up_proj": ("up_proj",),
                "down_proj": ("down_proj",),

                "shared_experts": {
                    "gate_proj": ("gate_proj:0",),
                    "up_proj": ("up_proj:0",),
                    "down_proj": ("down_proj:1",),
                },
                "text_moe": {
                    "gate": ("gate:!",),
                    "experts": {
                        "#": ("gate_proj:0", "upe_proj:0", "down_proj:1"),
                    },
                },
                "vision_moe": {
                    "gate": ("gate:!",),
                    "experts": {
                        "#": ("gate_proj:0", "upe_proj:0", "down_proj:1"),
                    },
                }
            },
        }
    ]

    @classmethod
    def extract_layers_node(cls):
        return ["model.language_model.layers"]

    @classmethod
    def _resolve_multimodal_layout(cls, model):
        for prefix in ("model", ""):
            core_model = get_module(model, prefix) if prefix else model
            if core_model is None:
                continue
            if hasattr(core_model, "language_model") and hasattr(core_model, "vision_tower"):
                return prefix, core_model

        raise AttributeError(
            "Unable to resolve an ERNIE 4.5 VL MoE layout. Expected either "
            "`model.language_model + model.vision_tower`."
        )

    @classmethod
    def get_base_modules(cls, model):
        prefix, core_model = cls._resolve_multimodal_layout(model)
        base_modules = []
        for name, _ in core_model.named_children():
            if name != "language_model":
                base_modules.append(f"{prefix}.{name}" if prefix else name)
        return base_modules

    def _materialize_module(self, parent, attr_name: str):
        module = getattr(parent, attr_name, None)
        if module is None:
            return
        if "_turtle_lock" not in self.__dict__ and "shell_module_materialize" not in self.__dict__:
            setattr(parent, attr_name, move_to(module, device=self.quantize_config.device))
            return
        setattr(parent, attr_name, self.shell_module_materialize(module, self.quantize_config.device))

    def _core_multimodal_model(self):
        _, core_model = self._resolve_multimodal_layout(self.model)
        return core_model

    def pre_quantize_generate_hook_start(self):
        core_model = self._core_multimodal_model()
        language_model = core_model.language_model
        self._materialize_module(core_model, "vision_tower")
        self._materialize_module(language_model, "embed_tokens")
        self._materialize_module(language_model, "rotary_emb")
        self._materialize_module(core_model, "resampler_model")

    def pre_quantize_generate_hook_end(self):
        core_model = self._core_multimodal_model()
        language_model = core_model.language_model
        vision_module = core_model.vision_tower
        resampler_module = core_model.resampler_model

        if self.quantize_config.offload_to_disk:
            if hasattr(language_model, "embed_tokens"):
                offload_to_disk(
                    model=language_model,
                    module=language_model.embed_tokens,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            if hasattr(language_model, "rotary_emb"):
                offload_to_disk(
                    model=language_model,
                    module=language_model.rotary_emb,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            if vision_module is not None:
                offload_to_disk(
                    model=core_model,
                    module=vision_module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            if resampler_module is not None:
                offload_to_disk(
                    model=core_model,
                    module=resampler_module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        if hasattr(language_model, "embed_tokens"):
            language_model.embed_tokens = move_to(language_model.embed_tokens, device=CPU)
        if hasattr(language_model, "rotary_emb"):
            language_model.rotary_emb = move_to(language_model.rotary_emb, device=CPU)
        if vision_module is not None:
            setattr(core_model, vision_module, move_to(vision_module, device=CPU))
        if resampler_module is not None:
            setattr(core_model, resampler_module, move_to(resampler_module, device=CPU))

    def lm_head_pre_quantize_generate_hook(self, inputs):
        norm, _ = get_module_by_name_prefix(self.model, ["model.norm", "model.language_model.norm"])
        if norm is None:
            return inputs

        norm = self.pre_quantize(norm)
        for element in inputs:
            for index in range(len(element)):
                element[index] = norm(element[index])
        self.post_quantize(norm)
        return inputs

    @staticmethod
    def _flatten_multimodal_features(features: Any) -> torch.Tensor:
        if torch.is_tensor(features):
            return features
        if isinstance(features, (list, tuple)):
            return torch.cat(tuple(features), dim=0)
        raise TypeError(f"Unsupported multimodal feature container: {type(features).__name__}")

    @staticmethod
    def _merge_placeholder_features(
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        token_id: int,
        features: torch.Tensor,
        label: str,
    ) -> torch.Tensor:
        token_mask = input_ids == token_id
        token_count = int(token_mask.sum().item())
        expected_numel = token_count * int(inputs_embeds.shape[-1])
        actual_numel = int(features.numel())
        feature_rows = int(features.shape[0]) if features.ndim > 0 else 0
        if expected_numel != actual_numel:
            raise ValueError(
                f"{label} features and {label.lower()} tokens do not match, "
                f"tokens: {token_count}, features: {feature_rows}"
            )

        expanded_mask = token_mask.unsqueeze(-1).expand_as(inputs_embeds).to(device=inputs_embeds.device)
        return inputs_embeds.masked_scatter(
            expanded_mask,
            features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

    @staticmethod
    def _prepare_first_layer_position_inputs(
        *,
        language_model,
        inputs_embeds: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        past_key_values: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        layer_position_ids = position_ids
        if layer_position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            layer_position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            layer_position_ids = layer_position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif layer_position_ids.ndim == 2:
            layer_position_ids = layer_position_ids[None, ...].expand(3, layer_position_ids.shape[0], -1)

        text_position_ids = None
        if layer_position_ids.ndim == 3 and layer_position_ids.shape[0] == 4:
            text_position_ids = layer_position_ids[0]
            layer_position_ids = layer_position_ids[1:]

        position_embeddings = language_model.rotary_emb(inputs_embeds, layer_position_ids)
        return layer_position_ids, text_position_ids, position_embeddings

    @staticmethod
    def _resolve_token_id(model_config, primary: str, secondary: str) -> int:
        if model_config is None:
            raise AttributeError(f"Missing model config while resolving `{primary}`.")

        value = getattr(model_config, primary, None)
        if value is None:
            value = getattr(model_config, secondary, None)
        if value is None:
            raise AttributeError(f"Neither `{primary}` nor `{secondary}` exists in model config.")
        return int(value)

    def run_input_capture(self, example, use_cache: bool, data_device):
        input_ids = example.get("input_ids")
        pixel_values = example.get("pixel_values")
        pixel_values_videos = example.get("pixel_values_videos")
        if input_ids is None or (pixel_values is None and pixel_values_videos is None):
            return super().run_input_capture(example, use_cache=use_cache, data_device=data_device)

        _, core_model = self._resolve_multimodal_layout(self.model)
        language_model = core_model.language_model

        attention_mask = example.get("attention_mask")
        position_ids = example.get("position_ids")
        past_key_values = example.get("past_key_values")
        image_grid_thw = example.get("image_grid_thw")
        video_grid_thw = example.get("video_grid_thw")
        mm_token_type_ids = example.get("mm_token_type_ids")
        moe_mm_token_type_ids = example.get("moe_mm_token_type_ids")

        input_embedder = core_model.get_input_embeddings() if hasattr(core_model, "get_input_embeddings") else None
        if input_embedder is None:
            input_embedder = language_model.embed_tokens
        inputs_embeds = input_embedder(input_ids)

        model_config = getattr(self.model, "config", None)
        print_module_tree(core_model)
        if pixel_values is not None and hasattr(core_model, "get_image_features"):
            image_outputs = core_model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            image_features = self._flatten_multimodal_features(image_outputs.pooler_output)
            image_token_id = self._resolve_token_id(model_config, "image_token_id", "image_token_index")
            inputs_embeds = self._merge_placeholder_features(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                token_id=image_token_id,
                features=image_features,
                label="Image",
            )

        if pixel_values_videos is not None and hasattr(core_model, "get_video_features"):
            video_outputs = core_model.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
            video_features = self._flatten_multimodal_features(video_outputs.pooler_output)
            video_token_id = self._resolve_token_id(model_config, "video_token_id", "video_token_index")
            inputs_embeds = self._merge_placeholder_features(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                token_id=video_token_id,
                features=video_features,
                label="Video",
            )

        if position_ids is None and hasattr(core_model, "compute_3d_position_ids"):
            position_ids = core_model.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        passthrough_kwargs = {}
        consumed = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "mm_token_type_ids",
            "moe_mm_token_type_ids",
            "rope_deltas",
        }
        for key, value in example.items():
            if key not in consumed:
                passthrough_kwargs[key] = value

        layer_position_ids, text_position_ids, position_embeddings = self._prepare_first_layer_position_inputs(
            language_model=language_model,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        return language_model.layers[0](
            inputs_embeds,
            attention_mask=causal_mask,
            position_ids=layer_position_ids,
            moe_mm_token_type_ids=moe_mm_token_type_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **passthrough_kwargs,
        )

    @staticmethod
    def process_vision_info(
            conversations: list[dict] | list[list[dict]],
    ) -> Optional[list[Image.Image]]:
        vision_infos = extract_vision_info(conversations)
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

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=self.trust_remote_code)

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        processor = self.load_processor()
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
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
            calib_data.append(inputs)
        del processor
        return calib_data

    def before_model_load(self, model_local_path: str, load_quantized_model: bool):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        try:
            decoder_cls = get_class_from_dynamic_module(
                "modeling_ernie4_5_vl.Ernie4_5_DecoderLayer",
                model_local_path,
            )
        except Exception:
            return

        if getattr(decoder_cls, "_gptqmodel_vl_moe_patched", False):
            return

        if not hasattr(decoder_cls, "mlp_text"):
            def mlp_text(self):
                return self.mlp

            decoder_cls.mlp_text = mlp_text

        decoder_cls._gptqmodel_vl_moe_patched = True


__all__ = ["Ernie4_5_VLMoeQModel"]
