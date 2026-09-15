# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from collections.abc import Mapping, Sized
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY, get_module, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


QWEN_VL_UTILS_INSTALL_HINT = (
    "qwen_vl_utils is required for Qwen3-VL video inputs. "
    "Install it with `pip install qwen-vl-utils`."
)


def _load_fetch_video():
    try:
        from qwen_vl_utils.vision_process import fetch_video
    except ImportError as exc:
        raise ImportError(QWEN_VL_UTILS_INSTALL_HINT) from exc

    return fetch_video


class BaseQwen3VLGPTQ(BaseQModel):
    loader = AutoModelForImageTextToText

    pre_lm_head_norm_module = ["model.norm", "norm"]

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
                "q_norm:!",
                "k_norm:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    # Qwen3.5 requires this field to compute multimodal RoPE, while older
    # Qwen3-VL-compatible processors (notably Mage-VL) do not provide it.
    require_mm_token_type_ids = False

    @classmethod
    def extract_layers_node(cls):
        return ["model.language_model.layers", "language_model.layers"]

    @classmethod
    def get_base_modules(cls, model):
        prefix, core_model = cls._resolve_multimodal_layout(model)
        base_modules = []
        for name, _ in core_model.named_children():
            if name != "language_model":
                base_modules.append(f"{prefix}.{name}" if prefix else name)
        return base_modules

    @classmethod
    def _resolve_multimodal_layout(cls, model):
        for prefix in ("model", ""):
            core_model = get_module(model, prefix) if prefix else model
            if core_model is None:
                continue
            if hasattr(core_model, "language_model") and hasattr(core_model, "visual"):
                return prefix, core_model

        raise AttributeError(
            "Unable to resolve a Qwen VL core model with `language_model` and `visual` modules."
        )

    def _core_multimodal_model(self):
        _, core_model = self._resolve_multimodal_layout(self.model)
        return core_model

    def _materialize_core_module(self, parent, attr_name: str):
        module = getattr(parent, attr_name)
        if (
            "_turtle_lock" not in self.__dict__
            and "shell_module_materialize" not in self.__dict__
        ):
            setattr(
                parent, attr_name, move_to(module, device=self.quantize_config.device)
            )
            return
        setattr(
            parent,
            attr_name,
            self.shell_module_materialize(module, self.quantize_config.device),
        )

    def pre_quantize_generate_hook_start(self):
        core_model = self._core_multimodal_model()
        self._materialize_core_module(core_model.language_model, "embed_tokens")
        self._materialize_core_module(core_model.language_model, "rotary_emb")
        self._materialize_core_module(core_model, "visual")

    def pre_quantize_generate_hook_end(self):
        core_model = self._core_multimodal_model()
        if self.quantize_config.offload_to_disk:
            offload_to_disk(
                model=core_model.language_model,
                module=core_model.language_model.embed_tokens,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=core_model.language_model,
                module=core_model.language_model.rotary_emb,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=core_model,
                module=core_model.visual,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        core_model.language_model.embed_tokens = move_to(
            core_model.language_model.embed_tokens, device=CPU
        )
        core_model.language_model.rotary_emb = move_to(
            core_model.language_model.rotary_emb, device=CPU
        )
        core_model.visual = move_to(core_model.visual, device=CPU)

    @staticmethod
    def process_vision_info(
        conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        return_video_kwargs: bool = False,
        return_video_metadata: bool = False,
        image_patch_size: int = 14,
    ) -> Tuple[
        Optional[List[Image.Image]],
        Optional[List[Union[torch.Tensor, List[Image.Image]]]],
        Optional[Dict[str, Any]],
    ]:
        conversations, vision_context = BaseQwen3VLGPTQ._validate_conversations(
            conversations
        )
        vision_infos = extract_vision_info(conversations)
        ## Read images or videos
        image_inputs = []
        video_inputs = []
        video_sample_fps_list = []
        fetch_video = None
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                sample_index, message_index, content_index = vision_context.get(
                    id(vision_info), ("?", "?", "?")
                )
                try:
                    image_inputs.append(
                        fetch_image(vision_info, image_patch_size=image_patch_size)
                    )
                except Exception as exc:
                    raise ValueError(
                        "Unable to load image for Qwen3-VL calibration "
                        f"sample {sample_index}, message {message_index}, "
                        f"content item {content_index}: {exc}"
                    ) from exc
            elif "video" in vision_info:
                if fetch_video is None:
                    fetch_video = _load_fetch_video()
                sample_index, message_index, content_index = vision_context.get(
                    id(vision_info), ("?", "?", "?")
                )
                try:
                    video_input, video_sample_fps = fetch_video(
                        vision_info,
                        return_video_sample_fps=True,
                        image_patch_size=image_patch_size,
                        return_video_metadata=return_video_metadata,
                    )
                except Exception as exc:
                    raise ValueError(
                        "Unable to load video for Qwen3-VL calibration "
                        f"sample {sample_index}, message {message_index}, "
                        f"content item {content_index}: {exc}"
                    ) from exc
                video_sample_fps_list.append(video_sample_fps)
                video_inputs.append(video_input)
            else:
                sample_index, message_index, content_index = vision_context.get(
                    id(vision_info), ("?", "?", "?")
                )
                raise ValueError(
                    "Qwen3-VL calibration vision item at "
                    f"sample {sample_index}, message {message_index}, "
                    f"content item {content_index} must contain image, image_url, or video."
                )
        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None

        video_kwargs = {"do_sample_frames": False}
        if not return_video_metadata:  # BC for qwen2.5vl
            video_kwargs.update({"fps": video_sample_fps_list})

        if return_video_kwargs:
            return image_inputs, video_inputs, video_kwargs
        if video_inputs is None and not return_video_metadata:
            # Keep the image-only call contract aligned with the earlier VL
            # adapters so processor(images=...) can use the return value directly.
            return image_inputs
        return image_inputs, video_inputs

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        try:
            return AutoProcessor.from_pretrained(self.model_local_path)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the Qwen3-VL processor for model "
                f"{self.model_local_path!r}. Check the processor files and model path."
            ) from exc

    @staticmethod
    def _validate_conversations(conversations):
        """Validate chat structure before a processor can hide its context."""
        if not isinstance(conversations, (list, tuple)) or not conversations:
            raise ValueError(
                "Qwen3-VL calibration conversations must be a non-empty list."
            )

        if isinstance(conversations[0], dict):
            conversations = [conversations]
        elif not all(isinstance(sample, (list, tuple)) for sample in conversations):
            raise ValueError(
                "Invalid Qwen3-VL calibration conversation structure: expected "
                "a list of messages or a batch of message lists."
            )

        normalized = [list(sample) for sample in conversations]
        vision_context = {}
        for sample_index, conversation in enumerate(normalized):
            if not conversation:
                raise ValueError(
                    "Invalid Qwen3-VL calibration conversation at "
                    f"sample {sample_index}: expected at least one message."
                )
            for message_index, message in enumerate(conversation):
                if not isinstance(message, dict):
                    raise ValueError(
                        "Invalid Qwen3-VL calibration conversation at "
                        f"sample {sample_index}, message {message_index}: "
                        "expected a mapping."
                    )
                if not isinstance(message.get("role"), str) or not message["role"]:
                    raise ValueError(
                        "Invalid Qwen3-VL calibration conversation at "
                        f"sample {sample_index}, message {message_index}: "
                        "missing a non-empty string role."
                    )
                if "content" not in message:
                    raise ValueError(
                        "Invalid Qwen3-VL calibration conversation at "
                        f"sample {sample_index}, message {message_index}: missing content."
                    )

                content = message["content"]
                if isinstance(content, str):
                    continue
                if not isinstance(content, (list, tuple)):
                    raise ValueError(
                        "Invalid Qwen3-VL calibration conversation at "
                        f"sample {sample_index}, message {message_index}: content "
                        "must be text or a list of content items."
                    )

                for content_index, item in enumerate(content):
                    if not isinstance(item, dict):
                        raise ValueError(
                            "Invalid Qwen3-VL calibration content at "
                            f"sample {sample_index}, message {message_index}, "
                            f"content item {content_index}: expected a mapping."
                        )

                    item_type = item.get("type")
                    has_image = "image" in item or "image_url" in item
                    has_video = "video" in item
                    if has_image or item_type in ("image", "image_url"):
                        image_key = "image" if "image" in item else "image_url"
                        if image_key not in item or item[image_key] in (None, ""):
                            raise ValueError(
                                "Invalid Qwen3-VL image content at "
                                f"sample {sample_index}, message {message_index}, "
                                f"content item {content_index}: declared image "
                                "must include a non-empty image or image_url value."
                            )
                        vision_context[id(item)] = (
                            sample_index,
                            message_index,
                            content_index,
                        )
                    elif has_video or item_type == "video":
                        if "video" not in item or item["video"] in (None, ""):
                            raise ValueError(
                                "Invalid Qwen3-VL video content at "
                                f"sample {sample_index}, message {message_index}, "
                                f"content item {content_index}: declared video "
                                "must include a non-empty video value."
                            )
                        vision_context[id(item)] = (
                            sample_index,
                            message_index,
                            content_index,
                        )
                    elif item_type == "text":
                        if not isinstance(item.get("text"), str):
                            raise ValueError(
                                "Invalid Qwen3-VL text content at "
                                f"sample {sample_index}, message {message_index}, "
                                f"content item {content_index}: missing text."
                            )
                    else:
                        raise ValueError(
                            "Invalid Qwen3-VL content item at "
                            f"sample {sample_index}, message {message_index}, "
                            f"content item {content_index}: expected text, image, "
                            "image_url, or video content."
                        )

        return normalized, vision_context

    @staticmethod
    def _contains_content_type(conversations, content_type: str) -> bool:
        for conversation in conversations:
            for message in conversation:
                content = message.get("content", [])
                if not isinstance(content, (list, tuple)):
                    continue
                for item in content:
                    if content_type == "image" and (
                        "image" in item
                        or "image_url" in item
                        or item.get("type") in ("image", "image_url")
                    ):
                        return True
                    if content_type == "video" and (
                        "video" in item or item.get("type") == "video"
                    ):
                        return True
        return False

    def _validate_processor_output(
        self, inputs, conversations, batch_index: int
    ) -> None:
        def get_field(name):
            try:
                return inputs[name]
            except (KeyError, IndexError, TypeError, AttributeError):
                return None

        if inputs is None or not hasattr(inputs, "__getitem__"):
            raise ValueError(
                f"Qwen3-VL processor returned an invalid output for batch {batch_index}; "
                "expected a mapping containing input_ids."
            )
        if get_field("input_ids") is None:
            raise ValueError(
                f"Qwen3-VL processor output for batch {batch_index} is missing input_ids."
            )

        required = []
        if self._contains_content_type(conversations, "image"):
            required.extend(("pixel_values", "image_grid_thw"))
        if self._contains_content_type(conversations, "video"):
            required.extend(("pixel_values_videos", "video_grid_thw"))
        if required and getattr(self, "require_mm_token_type_ids", False):
            required.append("mm_token_type_ids")
        missing = [name for name in required if get_field(name) is None]
        if missing:
            raise ValueError(
                f"Qwen3-VL processor output for multimodal batch {batch_index} "
                f"is missing required field(s): {', '.join(missing)}."
            )

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        if isinstance(calibration_dataset, (str, bytes, bytearray, Mapping)):
            raise ValueError(
                "Qwen3-VL calibration dataset must be an iterable of samples, "
                "not a single string or mapping."
            )
        if isinstance(calibration_dataset, Sized) and len(calibration_dataset) == 0:
            raise ValueError("Qwen3-VL calibration dataset is empty.")

        try:
            # Some adapter contract tests construct an uninitialized wrapper
            # with object.__new__. Reading through BaseQModel.__getattr__ in
            # that state recurses because nn.Module internals do not exist yet.
            processor = self.__dict__.get("processor") or self.load_processor()
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "Qwen3-VL processor" in str(exc):
                raise
            raise RuntimeError(
                "Unable to load the Qwen3-VL processor while preparing calibration data."
            ) from exc
        calib_data = []
        try:
            batches = batched_conversations(
                calibration_dataset,
                batch_size,
                process_func=self.preprocess_dataset,
            )
            for batch_index, batch in enumerate(batches):
                try:
                    batch, _vision_context = self._validate_conversations(batch)
                    text = processor.apply_chat_template(
                        batch, tokenize=False, add_generation_prompt=True
                    )
                    vision_inputs = self.process_vision_info(batch)
                    if isinstance(vision_inputs, tuple):
                        image_inputs, video_inputs = vision_inputs
                    else:
                        image_inputs, video_inputs = vision_inputs, None
                    inputs = processor(
                        text=text,
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    self._validate_processor_output(inputs, batch, batch_index)
                    calib_data.append(inputs)
                except Exception as exc:
                    if isinstance(exc, ValueError) and f"batch {batch_index}" in str(
                        exc
                    ):
                        raise
                    raise ValueError(
                        "Qwen3-VL calibration preparation failed for "
                        f"batch {batch_index}: {exc}"
                    ) from exc
        finally:
            del processor
        return calib_data
