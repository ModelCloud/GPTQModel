# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import Any, Dict

import torch
from transformers import AutoModel, AutoProcessor

from ...utils.calibration import batched
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class LocateAnythingQModel(BaseQModel):
    """Quantize LocateAnything's Qwen decoder while keeping its vision path dense."""

    loader = AutoModel

    require_trust_remote_code = True
    # Loading is on demand because Transformers 5 serializes this legacy
    # processor into a shape that its own remote constructor cannot reload.
    require_load_processor = False
    support_batch_quantize = False
    require_pkgs = ["peft", "decord", "lmdb"]

    out_of_model_tensors = {
        "files": [
            "preprocessor_config.json",
            "processor_config.json",
            "chat_template.json",
        ],
    }

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    lm_head = "language_model.lm_head"
    pre_lm_head_norm_module = "language_model.model.norm"

    module_tree = [
        "language_model",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    @classmethod
    def get_base_modules(cls, model):
        base_modules = super().get_base_modules(model)
        for name, _module in model.named_children():
            if name != "language_model" and name not in base_modules:
                base_modules.append(name)
        return base_modules

    def load_processor(self):
        return AutoProcessor.from_pretrained(
            self.model_local_path,
            trust_remote_code=True,
        )

    def pre_quantize_generate_hook_start(self):
        text_model = self.model.language_model.model
        text_model.embed_tokens = self.pre_quantize(text_model.embed_tokens)
        self.model.vision_model = self.pre_quantize(self.model.vision_model)
        self.model.mlp1 = self.pre_quantize(self.model.mlp1)

    def pre_quantize_generate_hook_end(self):
        text_model = self.model.language_model.model
        modules = (
            (text_model, text_model.embed_tokens),
            (self.model, self.model.vision_model),
            (self.model, self.model.mlp1),
        )
        if self.quantize_config.offload_to_disk:
            for parent, module in modules:
                offload_to_disk(
                    model=parent,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        text_model.embed_tokens = move_to(text_model.embed_tokens, device=CPU)
        self.model.vision_model = move_to(self.model.vision_model, device=CPU)
        self.model.mlp1 = move_to(self.model.mlp1, device=CPU)

    def prepare_dataset(
        self,
        calibration_dataset,
        batch_size: int = 1,
        **kwargs,
    ) -> list[Dict[str, Any]]:
        del batch_size, kwargs
        processor = self.load_processor()
        calibration_data = []
        for batch in batched(calibration_dataset, 1):
            messages = batch[0]
            text = processor.py_apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            images, videos = processor.process_vision_info(messages)
            calibration_data.append(
                processor(
                    text=[text],
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                )
            )
        del processor
        return calibration_data

    def move_input_capture_example(
        self,
        example: Dict[str, Any],
        data_device: torch.device,
    ) -> Dict[str, Any]:
        example = super().move_input_capture_example(example, data_device)
        pixel_values = example.get("pixel_values")
        if not torch.is_tensor(pixel_values):
            return example

        vision_model = self.model.vision_model
        first_parameter = next(vision_model.parameters(), None)
        vision_device = getattr(first_parameter, "device", pixel_values.device)
        vision_dtype = getattr(vision_model, "dtype", None)
        if not isinstance(vision_dtype, torch.dtype):
            vision_dtype = getattr(first_parameter, "dtype", None)
        if isinstance(vision_dtype, torch.dtype):
            example["pixel_values"] = pixel_values.to(
                device=vision_device,
                dtype=vision_dtype,
            )
        image_grid_hws = example.get("image_grid_hws")
        if image_grid_hws is not None:
            example["image_grid_hws"] = torch.as_tensor(
                image_grid_hws,
                device=vision_device,
                dtype=torch.int32,
            )
        return example

    def run_input_capture(
        self,
        example: Dict[str, Any],
        use_cache: bool,
        data_device: torch.device,
    ):
        del data_device
        input_ids = example["input_ids"]
        vit_embeds = self.model.extract_feature(
            example["pixel_values"],
            example["image_grid_hws"],
        )
        visual_features = None
        if vit_embeds:
            visual_features = self.model.mlp1(torch.cat(vit_embeds, dim=0))

        # The outer remote-code forward replaces image-token embeddings and
        # then calls the Qwen decoder with only inputs_embeds. Its SDPA mask
        # path nevertheless indexes input_ids. The bundled decoder already
        # exposes the equivalent visual_features path, which preserves
        # input_ids and avoids that remote-code inconsistency during capture.
        return self.model.language_model(
            input_ids=input_ids,
            visual_features=visual_features,
            image_token_index=self.model.image_token_index,
            attention_mask=example.get("attention_mask"),
            labels=example.get("labels"),
            use_cache=use_cache,
        )

    @staticmethod
    def _has_visual_inputs(inputs: Any, kwargs: Dict[str, Any]) -> bool:
        if kwargs.get("pixel_values") is not None:
            return True
        return bool(
            hasattr(inputs, "get")
            and inputs.get("pixel_values") is not None
        )

    def forward(self, *args, **kwargs):
        # The checkpoint's multimodal wrapper requires pixel_values even for a
        # text-only call. Language evaluations should exercise the same
        # quantized Qwen decoder directly when no visual input is supplied.
        if not self._has_visual_inputs(None, kwargs):
            return self.model.language_model(*args, **kwargs)
        return self.model(*args, **kwargs)

    def generate(self, inputs=None, **kwargs):
        if self._has_visual_inputs(inputs, kwargs):
            return super().generate(inputs=inputs, **kwargs)

        with torch.inference_mode():
            if kwargs.get("pad_token_id") is None and self.tokenizer is not None:
                kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            if isinstance(inputs, str) or (
                isinstance(inputs, list) and all(isinstance(item, str) for item in inputs)
            ):
                inputs = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                ).to(self.model.language_model.device)

            if hasattr(inputs, "get") and not torch.is_tensor(inputs):
                return self.model.language_model.generate(**inputs, **kwargs)
            return self.model.language_model.generate(inputs=inputs, **kwargs)


__all__ = ["LocateAnythingQModel"]
