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
from .._const import CPU
from ..base import BaseQModel


class OvisQModel(BaseQModel):
    require_trust_remote_code = True
    pre_lm_head_norm_module = "llm.model.norm"

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

    def monkey_patch(self):
        # From config.json, we know that visual_tokenizer.dtype is float32 and text model.confi.dtype is bfloat16.
        # But before transformers<4.49.0, the dtype returned by AutoModel.from_config(config.visual_tokenizer_config)
        # is bfloat16. This should be a bug, but OVIS generate() unexpectedly works properly.
        # This bug was fixed in transformers 4.49.0. So visual_tokenizer needs to be converted to model.config.dtype
        self.model.visual_tokenizer = self.model.visual_tokenizer.to(dtype=self.model.llm.dtype)
        self.model.vte = self.model.vte.to(dtype=self.model.llm.dtype)

    def pre_quantize_generate_hook_start(self):
        visual_tokenizer_meta = any(param.device.type == "meta" for param in self.model.visual_tokenizer.parameters()) or \
            any(buffer.device.type == "meta" for buffer in self.model.visual_tokenizer.buffers())
        if visual_tokenizer_meta:
            try:
                self.shell_module_materialize(self.model.visual_tokenizer, self.quantize_config.device)
            except Exception:
                logging.warning("OVIS visual_tokenizer shell materialization failed; continuing with fallback move.",
                                exc_info=True)

        vte_meta = any(param.device.type == "meta" for param in self.model.vte.parameters()) or \
            any(buffer.device.type == "meta" for buffer in self.model.vte.buffers())
        if vte_meta:
            try:
                self.shell_module_materialize(self.model.vte, self.quantize_config.device)
            except Exception:
                logging.warning("OVIS VTE shell materialization failed; continuing with fallback move.", exc_info=True)

        self.model.visual_tokenizer = move_to(self.model.visual_tokenizer, device=self.quantize_config.device)
        self.model.vte = move_to(self.model.vte, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
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

        target_dtype = self.model.visual_tokenizer.dtype
        if pixel_values is None:
            pixel_values, _ = self.visual_tokenizer.mock_input()
            pixel_values = [pv.to(dtype=target_dtype) for pv in pixel_values]
        elif isinstance(pixel_values, (list, tuple)):
            pixel_values = [pv.to(dtype=target_dtype) for pv in pixel_values]
        else:
            pixel_values = pixel_values.to(dtype=target_dtype)

        input_ids = input_ids[:text_max_length]
        labels = labels[:text_max_length]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
        }

    def prepare_dataset(
            self,
            calibration_dataset,
            calibration_dataset_concat_size,
            batch_size: int = 1,
            tokenizer=None,
            **kwargs):
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, self.preprocess_dataset):
            pixel_values, input_ids, labels = tuple([instance[key] for instance in batch]
                                                    for key in ("pixel_values", "input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.text_tokenizer.pad_token_id)
            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
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

    def generate(self, inputs=None, **kwargs):
        """shortcut for model.generate"""
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            quant_device = getattr(self.quantize_config, "device", None)
            model_device = torch.device(quant_device if quant_device is not None else "cpu")

        llm = getattr(self.model, "llm", None)

        pixel_values = None
        if isinstance(inputs, dict):
            pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values")

        has_real_pixels = False
        if pixel_values is not None:
            if isinstance(pixel_values, (list, tuple)):
                has_real_pixels = any(p is not None for p in pixel_values)
            else:
                has_real_pixels = True

        text_only = (llm is not None) and not has_real_pixels

        if text_only:
            kwargs = dict(kwargs)
            kwargs.pop("pixel_values", None)

            if isinstance(inputs, dict):
                if inputs.get("pixel_values") is not None:
                    inputs = dict(inputs)
                    inputs.pop("pixel_values", None)

            llm_device = next(self.model.llm.parameters()).device

            def ensure_attention_mask(payload):
                if payload.get("attention_mask") is not None or "input_ids" not in payload:
                    return payload
                mask = torch.ones_like(payload["input_ids"], dtype=torch.bool, device=payload["input_ids"].device)
                payload["attention_mask"] = mask
                return payload

            if isinstance(inputs, (str, list)):
                if isinstance(inputs, str) or (isinstance(inputs, list) and all(isinstance(x, str) for x in inputs)):
                    if self.tokenizer is None:
                        raise ValueError(
                            "You passed in text to OvisQModel.generate() but tokenizer is missing."
                        )
                    tokenized = self.tokenizer(
                        inputs,
                        return_tensors="pt",
                        padding=True,
                        padding_side="left"
                    ).to(llm_device)
                    with torch.amp.autocast(device_type=llm_device.type):
                        return self.model.llm.generate(**tokenized, **kwargs)

            if isinstance(inputs, dict):
                payload = {k: v for k, v in inputs.items() if k != "pixel_values"}
                if "input_ids" in payload and isinstance(payload["input_ids"], torch.Tensor):
                    payload["input_ids"] = payload["input_ids"].to(llm_device)
                if "attention_mask" in payload and isinstance(payload["attention_mask"], torch.Tensor):
                    payload["attention_mask"] = payload["attention_mask"].to(llm_device)
                payload = ensure_attention_mask(payload)
                payload.update(kwargs)
                with torch.amp.autocast(device_type=llm_device.type):
                    return self.model.llm.generate(**payload)

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(llm_device)
                attention_mask = kwargs.pop("attention_mask", None)
                if attention_mask is None:
                    attention_mask = torch.ones_like(inputs, dtype=torch.bool, device=inputs.device)
                else:
                    attention_mask = attention_mask.to(inputs.device)

                with torch.amp.autocast(device_type=llm_device.type):
                    return self.model.llm.generate(
                        inputs=inputs,
                        attention_mask=attention_mask,
                        **kwargs
                    )

            if inputs is None:
                payload = ensure_attention_mask({k: v for k, v in kwargs.items() if k in {"input_ids", "attention_mask"}})
                remaining = {k: v for k, v in kwargs.items() if k not in payload}
                payload = {k: (v.to(llm_device) if isinstance(v, torch.Tensor) else v) for k, v in payload.items()}
                with torch.amp.autocast(device_type=llm_device.type):
                    return self.model.llm.generate(**payload, **remaining)

        with torch.amp.autocast(device_type=model_device.type):
            return super().generate(inputs=inputs, **kwargs)

    def forward(self, *args, **kwargs):
        """Allow text-only invocations to bypass vision branch for evaluator compatibility."""
        if args:
            # most callers pass input_ids positionally; forward them as keyword for clarity
            if "input_ids" in kwargs:
                raise TypeError("OvisQModel.forward() received positional and keyword input_ids")
            kwargs = dict(kwargs)  # shallow copy to avoid mutating caller state
            kwargs["input_ids"] = args[0]
            args = args[1:]

        input_ids = kwargs.get("input_ids")
        pixel_values = kwargs.get("pixel_values")

        if input_ids is not None and pixel_values is None and hasattr(self.model, "llm"):
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)

            # Hugging Face text evaluators expect logits only; labels are optional here.
            llm_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in {"pixel_values"}
            }
            llm_kwargs["attention_mask"] = attention_mask

            if llm_kwargs.get("labels") is None:
                llm_kwargs.pop("labels", None)

            return self.model.llm(**llm_kwargs)

        return super().forward(*args, **kwargs)
