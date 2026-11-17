# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import GenerationConfig

from ...utils.calibration import batched_conversations
from ...utils.image import fetch_image
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks
from ._vlm_calibration import (
    finalize_conversation_batches,
    pop_conversation_calibration_options,
    split_calibration_dataset,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLChatQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.internvl_chat.configuration_internvl_chat.InternVLChatConfig",
        model_class="gptqmodel.models.ports.internvl_chat.modeling_internvl_chat.InternVLChatModel",
    )
    dynamic_expert_index = "num_experts"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()
    defuser_module_paths = ("language_model",)

    lm_head = "language_model.lm_head"
    pre_lm_head_norm_module = "language_model.model.norm"
    rotary_embedding = "language_model.model.rotary_emb"

    module_tree = [
        "language_model",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "k_norm:!",
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    support_batch_quantize = False

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector=detector)
        config = getattr(model, "config", None)
        architectures = getattr(getattr(config, "llm_config", None), "architectures", None)
        architecture = architectures[0] if architectures else None
        blocks = tree[-1]
        if architecture == "InternLM2ForCausalLM":
            tree[-1] = {
                "attention_norm": ("attention_norm:!",),
                "attention": ("wqkv:0", "wo:1"),
                "ffn_norm": ("ffn_norm:!",),
                "feed_forward": ("w1:0", "w3:0", "w2:1"),
            }
        elif architecture == "Phi3ForCausalLM":
            blocks["self_attn"] = ("qkv_proj:0", "o_proj:1")
        elif architecture == "Qwen3MoeForCausalLM":
            blocks.pop("mlp")
            blocks["mlp:moe:?"] = {
                "gate": ("gate:!",),
                "experts": {"#": ("gate_proj:0", "up_proj:0", "down_proj:1")},
            }
        return tree

    @classmethod
    def get_num_experts(cls, model_config):
        return getattr(model_config.llm_config, "num_experts", None)

    @classmethod
    def _expert_count_config(cls, model_config, field_name=None):
        return model_config.llm_config

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Decoder alternatives, Q/K head norms, interleaved QKV and the image
        # bridge have not been jointly qualified for scale search/folding.
        # Empty groups retain ordinary AWQ weight quantization and clipping.
        return []

    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    @classmethod
    def extract_layers_node(cls):
        return ["language_model.model.layers"]

    @classmethod
    def get_base_modules(cls, model):
        base_modules = super().get_base_modules(model)
        for name, _ in model.named_children():
            if name != "language_model":
                base_modules.append(name)
        return base_modules

    def _get_tokenizer(self):
        return getattr(self.tokenizer, "tokenizer", self.tokenizer)

    @staticmethod
    def _build_transform(input_size: int):
        try:
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
        except ModuleNotFoundError as exc:
            if exc.name != "torchvision":
                raise
            raise ImportError(
                "InternVL image preprocessing requires torchvision. Install a "
                "torchvision version compatible with your PyTorch installation."
            ) from exc

        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @staticmethod
    def _find_closest_aspect_ratio(
        aspect_ratio: float,
        target_ratios: set[tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @classmethod
    def _dynamic_preprocess(
        cls,
        image: Image.Image,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = False,
    ) -> list[Image.Image]:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        }
        target_aspect_ratio = cls._find_closest_aspect_ratio(
            aspect_ratio,
            target_ratios,
            orig_width,
            orig_height,
            image_size,
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized = image.resize((target_width, target_height))
        processed_images = []
        blocks_per_row = target_width // image_size
        for index in range(blocks):
            box = (
                (index % blocks_per_row) * image_size,
                (index // blocks_per_row) * image_size,
                ((index % blocks_per_row) + 1) * image_size,
                ((index // blocks_per_row) + 1) * image_size,
            )
            processed_images.append(resized.crop(box))

        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))

        return processed_images

    @staticmethod
    def _normalize_role(message: Dict[str, Any]) -> str:
        role = message.get("role", message.get("from", "user"))
        role = str(role).lower()
        if role in {"human", "user"}:
            return "user"
        if role in {"gpt", "assistant"}:
            return "assistant"
        return role

    @staticmethod
    def _flatten_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        parts = []
        for item in content or []:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, Image.Image):
                parts.append("<image>")
                continue
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            if item_type in {"image", "image_url"} or "image" in item or "image_url" in item:
                parts.append("<image>")
            elif item_type == "text":
                parts.append(item.get("text", ""))

        return "\n".join(part for part in parts if part).strip()

    @classmethod
    def _extract_history_and_question(
        cls,
        conversation: list[Dict[str, Any]],
    ) -> tuple[Optional[list[tuple[str, str]]], str, Optional[str]]:
        """Split ``conversation`` into history, query, and the query's answer.

        A conversation that ends with an assistant message keeps that message as
        the ``answer`` so the calibration text still contains the response the
        model is meant to predict (issue #4227). ``answer`` is ``None`` when the
        conversation ends with a user turn.
        """
        turns: list[tuple[str, str]] = []
        pending_user: Optional[str] = None

        for message in conversation:
            role = cls._normalize_role(message)
            content = cls._flatten_content_to_text(
                message.get("content", message.get("value", ""))
            )

            if role == "assistant":
                if pending_user is not None:
                    turns.append((pending_user, content))
                    pending_user = None
                continue

            if role == "user" and content:
                pending_user = content

        if pending_user is not None:
            return turns or None, pending_user, None

        if not turns:
            raise ValueError("InternVL calibration conversation must contain at least one user prompt.")

        history = turns[:-1] or None
        question, answer = turns[-1]
        return history, question, answer

    @classmethod
    def process_vision_info(
        cls,
        conversation: list[Dict[str, Any]],
        *,
        input_size: int,
        min_num: int,
        max_num: int,
        use_thumbnail: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[list[int]]]:
        transform = cls._build_transform(input_size)
        image_batches: list[torch.Tensor] = []
        num_patches_list: list[int] = []

        for message in conversation:
            content = message.get("content", message.get("value", ""))
            if not isinstance(content, list):
                continue

            for item in content:
                if isinstance(item, Image.Image):
                    image = item
                elif isinstance(item, dict) and (
                    item.get("type") in {"image", "image_url"} or "image" in item or "image_url" in item
                ):
                    image = fetch_image(item)
                else:
                    continue

                processed_images = cls._dynamic_preprocess(
                    image=image,
                    min_num=min_num,
                    max_num=max_num,
                    image_size=input_size,
                    use_thumbnail=use_thumbnail,
                )
                image_batches.extend(transform(processed) for processed in processed_images)
                num_patches_list.append(len(processed_images))

        if not image_batches:
            return None, None

        return torch.stack(image_batches), num_patches_list

    def _build_query(
        self,
        *,
        question: str,
        history: Optional[list[tuple[str, str]]],
        num_patches_list: Optional[list[int]],
        answer: Optional[str] = None,
    ) -> tuple[str, int, int]:
        tokenizer = self._get_tokenizer()
        template = self.model.conv_template.copy() if hasattr(self.model.conv_template, "copy") else deepcopy(self.model.conv_template)
        template.system_message = self.model.system_message

        if history is None and num_patches_list and "<image>" not in question:
            question = "<image>\n" + question

        history = history or []
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], answer)

        query = template.get_prompt()
        for num_patches in num_patches_list or []:
            image_tokens = (
                self.IMG_START_TOKEN
                + self.IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches
                + self.IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        img_context_token_id = tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        return query, eos_token_id, img_context_token_id

    def prepare_inputs_for_conversation(
        self,
        conversation: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        config = self.model.config
        input_size = int(
            getattr(config, "force_image_size", None)
            or getattr(config.vision_config, "image_size", 448)
        )
        min_num = int(getattr(config, "min_dynamic_patch", 1))
        max_num = int(getattr(config, "max_dynamic_patch", 12))
        use_thumbnail = bool(getattr(config, "use_thumbnail", True))

        history, question, answer = self._extract_history_and_question(conversation)
        pixel_values, num_patches_list = self.process_vision_info(
            conversation,
            input_size=input_size,
            min_num=min_num,
            max_num=max_num,
            use_thumbnail=use_thumbnail,
        )

        tokenizer = self._get_tokenizer()
        query, eos_token_id, img_context_token_id = self._build_query(
            question=question,
            history=history,
            num_patches_list=num_patches_list,
            answer=answer,
        )
        tokenized = tokenizer(query, return_tensors="pt")
        tokenized["pixel_values"] = pixel_values
        tokenized["eos_token_id"] = eos_token_id
        tokenized["img_context_token_id"] = img_context_token_id
        return tokenized

    def pre_quantize_generate_hook_start(self):
        self.model.set_input_embeddings(self.pre_quantize(self.model.get_input_embeddings()))
        for path in self._decoder_rotary_paths():
            self.model.set_submodule(path, self.pre_quantize(self.model.get_submodule(path)))
        self.model.vision_model = self.pre_quantize(self.model.vision_model)
        self.model.mlp1 = self.pre_quantize(self.model.mlp1)

    def _decoder_rotary_paths(self):
        core = self.model.language_model.model
        if hasattr(core, "rotary_emb"):
            return ["language_model.model.rotary_emb"]
        return [f"language_model.model.layers.{index}.attention.rotary_emb" for index in range(len(core.layers))]

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            offload_to_disk(
                model=self.model.language_model.model,
                module=self.model.get_input_embeddings(),
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            for path in self._decoder_rotary_paths():
                offload_to_disk(
                    model=self.model.language_model.model,
                    module=self.model.get_submodule(path),
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            offload_to_disk(
                model=self.model,
                module=self.model.vision_model,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=self.model,
                module=self.model.mlp1,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        self.model.set_input_embeddings(move_to(self.model.get_input_embeddings(), device=CPU))
        for path in self._decoder_rotary_paths():
            self.model.set_submodule(path, move_to(self.model.get_submodule(path), device=CPU))
        self.model.vision_model = move_to(self.model.vision_model, device=CPU)
        self.model.mlp1 = move_to(self.model.mlp1, device=CPU)

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        label = type(self).__name__
        pretokenized, conversations = split_calibration_dataset(calibration_dataset, label=label)
        if pretokenized is not None:
            return super().prepare_dataset(pretokenized, batch_size=batch_size, **kwargs)

        sort, min_length = pop_conversation_calibration_options(kwargs, label=label)
        calib_data = []
        # The InternVL chat path prepares one conversation at a time.
        for batch in batched_conversations(conversations, 1):
            calib_data.append(self.prepare_inputs_for_conversation(batch[0]))
        return finalize_conversation_batches(
            calib_data, sort=sort, min_length=min_length, label=label
        )

    def move_input_capture_example(self, example, data_device):
        example = super().move_input_capture_example(example, data_device)
        pixel_values = example.get("pixel_values")
        if torch.is_tensor(pixel_values):
            vision_dtype = getattr(self.model.vision_model, "dtype", None) or getattr(self.model, "dtype", None)
            if isinstance(vision_dtype, torch.dtype):
                example["pixel_values"] = pixel_values.to(dtype=vision_dtype)
        return example

    def run_input_capture(self, example, use_cache: bool, data_device):
        del data_device
        if "img_context_token_id" not in example:
            return self.model(**example, use_cache=use_cache)
        self.model.img_context_token_id = int(example["img_context_token_id"])

        tokenizer = self._get_tokenizer()
        generation_config = deepcopy(getattr(self.model.language_model, "generation_config", None))
        if generation_config is None:
            generation_config = GenerationConfig()
        generation_config.max_new_tokens = 1
        generation_config.do_sample = False
        generation_config.eos_token_id = int(example["eos_token_id"])
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        generation_config.pad_token_id = pad_token_id

        return self.model.generate(
            pixel_values=example.get("pixel_values"),
            input_ids=example["input_ids"],
            attention_mask=example.get("attention_mask"),
            generation_config=generation_config,
        )
