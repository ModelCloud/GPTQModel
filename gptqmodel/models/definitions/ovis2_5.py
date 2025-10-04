# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import logging
from typing import Dict, List, Sequence

import torch
from transformers import AutoProcessor

from ...utils.calibration import batched
from ...utils.image import fetch_image
from ...utils.model import MODALITY, move_to, nested_move_to
from .._const import CPU
from ..base import BaseQModel


log = logging.getLogger(__name__)


class Ovis2_5QModel(BaseQModel):
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
        },
    ]

    layer_modules_strict = False

    require_monkeypatch = True
    require_load_processor = True

    modality = [MODALITY.IMAGE_TO_TEXT]

    IGNORE_ID = -100

    def monkey_patch(self):
        # keep the vision tower dtype aligned with the text tower for stable quantization/inference
        dtype = getattr(self.model.llm, "dtype", None)
        if dtype is None or dtype == torch.float32:
            dtype = torch.bfloat16
            try:
                self.model.llm = self.model.llm.to(dtype=dtype)
                if hasattr(self.model.llm, "config"):
                    self.model.llm.config.dtype = dtype
            except Exception:
                log.warning("Failed to cast llm to %s", dtype, exc_info=True)

        try:
            self.model.visual_tokenizer = self.model.visual_tokenizer.to(dtype=dtype)
        except Exception:
            log.warning("Failed to cast visual_tokenizer to %s", dtype, exc_info=True)
        try:
            self.model.vte = self.model.vte.to(dtype=dtype)
        except Exception:
            log.warning("Failed to cast visual embedding to %s", dtype, exc_info=True)

        attn_impl = getattr(self.model.llm.config, "_attn_implementation", None)
        if attn_impl == "flash_attention_2":
            log.info("Ovis2.5 monkey_patch: downgrading attention implementation to eager for compatibility.")
            self.model.llm.config._attn_implementation = "eager"

    @property
    def text_tokenizer(self):
        return getattr(self.model, "text_tokenizer", self.tokenizer)

    @property
    def visual_tokenizer(self):
        return getattr(self.model, "visual_tokenizer", None)

    def load_processor(self):
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=True)

    def get_text_tokenizer(self):
        return getattr(self.model, "text_tokenizer", None)

    def get_visual_tokenizer(self):
        return getattr(self.model, "visual_tokenizer", None)

    def _ensure_image_payload(self, messages: List[Dict]) -> List[Dict]:
        normalized_messages = copy.deepcopy(messages)
        for message in normalized_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue

            updated_content = []
            for item in content:
                if not isinstance(item, dict):
                    updated_content.append(item)
                    continue

                if item.get("type") != "image":
                    updated_content.append(item)
                    continue

                if item.get("image") is None and item.get("image_url") is None:
                    log.warning("Skipping image item without image or image_url: message=%s", message)
                    continue

                if hasattr(item.get("image"), "mode"):
                    updated_content.append(item)
                    continue

                source = {"image": item.get("image")} if item.get("image") is not None else {"image_url": item.get("image_url")}
                try:
                    image = fetch_image(source)
                except Exception as exc:
                    log.warning("Failed to load image for message; skipping image token", exc_info=exc)
                    continue

                new_item = dict(item)
                new_item["image"] = image
                new_item.pop("image_url", None)
                updated_content.append(new_item)

            message["content"] = updated_content

        return normalized_messages

    def _build_messages_from_conversations(
        self,
        conversations: Sequence[Dict],
        sample: Dict,
    ) -> List[Dict]:
        images = sample.get("image")
        if images is None:
            images_sequence: List = []
        elif isinstance(images, (list, tuple)):
            images_sequence = list(images)
        else:
            images_sequence = [images]

        image_objects: List = []
        for image_entry in images_sequence:
            try:
                image_objects.append(fetch_image({"image": image_entry}))
            except Exception as exc:
                log.warning("Failed to load image `%s` referenced by conversations", image_entry, exc_info=exc)

        image_iter = iter(image_objects)
        normalized_messages: List[Dict] = []
        for turn in conversations:
            speaker = turn.get("from") or turn.get("role") or "user"
            value = turn.get("value") or turn.get("content") or ""
            if speaker == "human":
                role = "user"
            elif speaker == "gpt":
                role = "assistant"
            else:
                role = speaker

            if role == "assistant":
                normalized_messages.append({"role": "assistant", "content": value})
                continue

            segments = value.split("<image>")
            content: List[Dict] = []
            for index, segment in enumerate(segments):
                if segment:
                    content.append({"type": "text", "text": segment})
                if index < len(segments) - 1:
                    image_obj = next(image_iter, None)
                    if image_obj is None:
                        log.warning("Conversation refers to more <image> tokens than provided images.")
                        break
                    content.append({"type": "image", "image": image_obj})

            if not content:
                content.append({"type": "text", "text": ""})

            normalized_messages.append({"role": role, "content": content})

        return normalized_messages

    def _normalize_messages(self, sample: Dict) -> List[Dict]:
        if isinstance(sample, list):
            return self._ensure_image_payload(sample)

        if sample.get("messages"):
            return self._ensure_image_payload(sample["messages"])

        conversations = sample.get("conversations")
        if not conversations:
            raise ValueError("Ovis2_5 calibration sample must provide `messages` or `conversations`.")

        return self._ensure_image_payload(self._build_messages_from_conversations(conversations, sample))

    def _coerce_pixel_values(self, pixel_values):
        if pixel_values is None:
            return None

        target_dtype = getattr(getattr(self.visual_tokenizer, "vit", self.visual_tokenizer), "dtype", None)
        target_dtype = target_dtype or getattr(self.visual_tokenizer, "dtype", None)
        if isinstance(pixel_values, torch.Tensor):
            if target_dtype is not None:
                pixel_values = pixel_values.to(dtype=target_dtype)
            return pixel_values

        if isinstance(pixel_values, (list, tuple)):
            coerced = []
            for pv in pixel_values:
                tensor = pv if isinstance(pv, torch.Tensor) else torch.as_tensor(pv)
                if target_dtype is not None:
                    tensor = tensor.to(dtype=target_dtype)
                coerced.append(tensor)
            return coerced

        tensor = torch.as_tensor(pixel_values)
        if target_dtype is not None:
            tensor = tensor.to(dtype=target_dtype)
        return tensor

    def preprocess_dataset(self, sample: Dict) -> Dict:
        messages = self._normalize_messages(sample)
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(messages)

        pixel_values = self._coerce_pixel_values(pixel_values)
        if pixel_values is None:
            pixel_values = []

        if isinstance(grid_thws, torch.Tensor):
            grid_thws = grid_thws.to(dtype=torch.long)
        elif grid_thws is None:
            grid_thws = []

        input_ids = input_ids.squeeze(0)
        pad_token_id = self.text_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.text_tokenizer.eos_token_id

        attention_mask = torch.ne(input_ids, pad_token_id)

        labels = input_ids.clone()
        labels.masked_fill_(labels < 0, self.IGNORE_ID)
        labels.masked_fill_(~attention_mask, self.IGNORE_ID)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
        }

    def prepare_dataset(
        self,
        calibration_dataset,
        calibration_dataset_concat_size=None,
        batch_size: int = 1,
        tokenizer=None,
        **kwargs,
    ):
        pad_token_id = self.text_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.text_tokenizer.eos_token_id

        calib_data = []
        for batch in batched(calibration_dataset, batch_size, self.preprocess_dataset):
            input_ids_list = [instance["input_ids"] for instance in batch]
            attention_masks_list = [instance["attention_mask"].to(torch.bool) for instance in batch]
            labels_list = [instance["labels"] for instance in batch]
            pixel_values_list = [instance["pixel_values"] for instance in batch]
            grid_thws_list = [instance["grid_thws"] for instance in batch]

            def _collate_vision(items):
                if not items:
                    return None
                if len(items) == 1:
                    value = items[0]
                    if value in (None, []):
                        return None
                    return value
                return items

            pixel_values = _collate_vision(pixel_values_list)
            grid_thws = _collate_vision(grid_thws_list)

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=pad_token_id,
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_masks_list,
                batch_first=True,
                padding_value=0,
            ).to(torch.bool)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list,
                batch_first=True,
                padding_value=self.IGNORE_ID,
            )

            calib_data.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "pixel_values": pixel_values,
                    "grid_thws": grid_thws,
                }
            )

        return calib_data

    def pre_quantize_generate_hook_start(self):
        visual_tokenizer = getattr(self.model, "visual_tokenizer", None)
        vision_modules = []
        if visual_tokenizer is not None:
            vision_modules.append(visual_tokenizer)
            vit = getattr(visual_tokenizer, "vit", None)
            if vit is not None:
                vision_modules.append(vit)
        vte = getattr(self.model, "vte", None)
        if vte is not None:
            vision_modules.append(vte)

        for module in vision_modules:
            if module is None:
                continue
            try:
                has_meta_params = any(param.device.type == "meta" for param in module.parameters())
            except Exception:
                has_meta_params = False
            try:
                has_meta_buffers = any(buffer.device.type == "meta" for buffer in module.buffers())
            except Exception:
                has_meta_buffers = False

            if has_meta_params or has_meta_buffers:
                try:
                    self.shell_module_materialize(module, self.quantize_config.device)
                except Exception:  # pragma: no cover - defensive downgrade
                    log.warning("OVIS2.5 module shell materialization failed; continuing with fallback move.", exc_info=True)

        if visual_tokenizer is not None:
            self.model.visual_tokenizer = move_to(visual_tokenizer, device=self.quantize_config.device)
            vit = getattr(self.model.visual_tokenizer, "vit", None)
            if vit is not None:
                self.model.visual_tokenizer.vit = move_to(vit, device=self.quantize_config.device)

        if vte is not None:
            self.model.vte = move_to(vte, device=self.quantize_config.device)

        indicator_buffer = getattr(self.model, "indicator_token_indices", None)
        if isinstance(indicator_buffer, torch.Tensor) and indicator_buffer.device.type == "meta":
            count = indicator_buffer.shape[0]
            vocab_size = getattr(getattr(self.model, "config", None), "visual_vocab_size", count)
            start_index = vocab_size - count
            device = self.quantize_config.device
            if not isinstance(device, torch.device):
                device = torch.device(device)
            materialized = torch.arange(start_index, vocab_size, dtype=torch.long, device=device)
            self.model.register_buffer("indicator_token_indices", materialized, persistent=False)

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            from ...utils.offload import offload_to_disk

            visual_tokenizer = getattr(self.model, "visual_tokenizer", None)
            if visual_tokenizer is not None:
                offload_to_disk(model=self.model, module=visual_tokenizer, disk_path=self.quantize_config.offload_to_disk_path)
                vit = getattr(visual_tokenizer, "vit", None)
                if vit is not None:
                    offload_to_disk(model=visual_tokenizer, module=vit, disk_path=self.quantize_config.offload_to_disk_path)

            vte = getattr(self.model, "vte", None)
            if vte is not None:
                offload_to_disk(model=self.model, module=vte, disk_path=self.quantize_config.offload_to_disk_path)
            return

        visual_tokenizer = getattr(self.model, "visual_tokenizer", None)
        if visual_tokenizer is not None:
            self.model.visual_tokenizer = move_to(visual_tokenizer, device=CPU)
            vit = getattr(self.model.visual_tokenizer, "vit", None)
            if vit is not None:
                self.model.visual_tokenizer.vit = move_to(vit, device=CPU)

        vte = getattr(self.model, "vte", None)
        if vte is not None:
            self.model.vte = move_to(vte, device=CPU)

        indicator_buffer = getattr(self.model, "indicator_token_indices", None)
        if isinstance(indicator_buffer, torch.Tensor):
            self.model.register_buffer("indicator_token_indices", indicator_buffer.to(CPU), persistent=False)

    def generate(self, inputs=None, **kwargs):
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            quant_device = getattr(self.quantize_config, "device", None)
            model_device = torch.device(quant_device if quant_device is not None else "cpu")

        llm = getattr(self.model, "llm", None)

        pixel_values = None
        grid_thws = None
        if isinstance(inputs, dict):
            pixel_values = inputs.get("pixel_values")
            grid_thws = inputs.get("grid_thws")
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values")
        if grid_thws is None:
            grid_thws = kwargs.get("grid_thws")

        def _has_pixels(payload):
            if payload is None:
                return False
            if isinstance(payload, torch.Tensor):
                return payload.numel() > 0
            if isinstance(payload, (str, bytes)):
                return False
            if isinstance(payload, Sequence):
                return any(_has_pixels(item) for item in payload)
            return True

        has_real_pixels = _has_pixels(pixel_values)

        if llm is not None and not has_real_pixels:
            kwargs = dict(kwargs)
            kwargs.pop("pixel_values", None)
            kwargs.pop("grid_thws", None)

            if isinstance(inputs, dict):
                payload = {k: v for k, v in inputs.items() if k not in {"pixel_values", "grid_thws"}}
            else:
                payload = inputs

            def ensure_attention_mask(payload_dict):
                if payload_dict is None:
                    return None
                if payload_dict.get("attention_mask") is not None or "input_ids" not in payload_dict:
                    return payload_dict
                mask = torch.ones_like(payload_dict["input_ids"], dtype=torch.bool, device=payload_dict["input_ids"].device)
                payload_dict["attention_mask"] = mask
                return payload_dict

            llm_device = next(self.model.llm.parameters()).device

            if isinstance(payload, (str, list)):
                if isinstance(payload, str) or (isinstance(payload, list) and all(isinstance(x, str) for x in payload)):
                    if self.tokenizer is None:
                        raise ValueError("You passed in text to Ovis2_5QModel.generate() but tokenizer is missing.")
                    tokenized = self.tokenizer(
                        payload,
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                    ).to(llm_device)
                    with torch.amp.autocast(device_type=llm_device.type):
                        return self.model.llm.generate(**tokenized, **kwargs)

            if isinstance(payload, dict):
                payload = ensure_attention_mask({k: (v.to(llm_device) if isinstance(v, torch.Tensor) else v) for k, v in payload.items()})
                with torch.amp.autocast(device_type=llm_device.type):
                    return self.model.llm.generate(**payload, **kwargs)

            if isinstance(payload, torch.Tensor):
                payload = payload.to(llm_device)
                attention_mask = kwargs.pop("attention_mask", None)
                if attention_mask is None:
                    attention_mask = torch.ones_like(payload, dtype=torch.bool, device=payload.device)
                else:
                    attention_mask = attention_mask.to(payload.device)
                with torch.amp.autocast(device_type=llm_device.type):
                    return self.model.llm.generate(inputs=payload, attention_mask=attention_mask, **kwargs)

            if payload is None:
                payload_inputs = {k: v for k, v in kwargs.items() if k in {"input_ids", "attention_mask"}}
                payload_inputs = ensure_attention_mask(payload_inputs)
                if payload_inputs:
                    payload_inputs = {k: (v.to(llm_device) if isinstance(v, torch.Tensor) else v) for k, v in payload_inputs.items()}
                with torch.amp.autocast(device_type=llm_device.type):
                    return self.model.llm.generate(**payload_inputs, **{k: v for k, v in kwargs.items() if k not in payload_inputs})

        kwargs = dict(kwargs)
        if isinstance(inputs, dict):
            inputs_dict = dict(inputs)
            base_inputs = inputs_dict.pop("input_ids", None)
            if base_inputs is None and "inputs_embeds" in inputs_dict:
                kwargs.setdefault("inputs_embeds", inputs_dict.pop("inputs_embeds"))
            for key in list(inputs_dict.keys()):
                if key in {"pixel_values", "grid_thws"}:
                    inputs_dict.pop(key)
            for key, value in inputs_dict.items():
                kwargs.setdefault(key, value)
            inputs = base_inputs

        if inputs is None and isinstance(kwargs.get("input_ids"), torch.Tensor):
            inputs = kwargs.get("input_ids")

        target_device = next(self.model.llm.parameters()).device
        model_device = target_device
        if isinstance(inputs, torch.Tensor) and inputs.device != target_device:
            inputs = inputs.to(target_device)
        if isinstance(kwargs.get("input_ids"), torch.Tensor) and kwargs["input_ids"].device != target_device:
            kwargs["input_ids"] = kwargs["input_ids"].to(target_device)
        if "input_ids" in kwargs:
            kwargs.pop("input_ids")
        if isinstance(kwargs.get("attention_mask"), torch.Tensor) and kwargs["attention_mask"].device != target_device:
            kwargs["attention_mask"] = kwargs["attention_mask"].to(target_device)

        if pixel_values is not None:
            kwargs["pixel_values"] = nested_move_to(pixel_values, device=model_device)
        if grid_thws is not None:
            kwargs["grid_thws"] = nested_move_to(grid_thws, device=model_device)

        with torch.amp.autocast(device_type=model_device.type):
            return super().generate(inputs=inputs, **kwargs)

    def forward(self, *args, **kwargs):
        if args:
            if "input_ids" in kwargs:
                raise TypeError("Ovis2_5QModel.forward() received positional and keyword input_ids")
            kwargs = dict(kwargs)
            kwargs["input_ids"] = args[0]
            args = args[1:]

        input_ids = kwargs.get("input_ids")
        pixel_values = kwargs.get("pixel_values")

        if input_ids is not None and pixel_values is None and hasattr(self.model, "llm"):
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)

            llm_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in {"pixel_values", "grid_thws"}
            }
            llm_kwargs["attention_mask"] = attention_mask

            if llm_kwargs.get("labels") is None:
                llm_kwargs.pop("labels", None)

            return self.model.llm(**llm_kwargs)

        return super().forward(*args, **kwargs)
