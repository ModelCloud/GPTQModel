# Copyright 2026 Agnes AI. All rights reserved.
# Adapted for package-owned Transformers 5 loading by ModelCloud.ai, 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Processor for Agnes 3.0 Flash: tokenizer + image processor + video processor."""

import json

import numpy as np
from transformers.processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from transformers.utils import auto_docstring, logging
from transformers.utils.hub import cached_file

from ....utils.hf import load_hf_tokenizer
from .configuration_agnes import AgnesConfig
from .image_processing_agnes import AgnesImageProcessor
from .video_processing_agnes import AgnesVideoProcessor


logger = logging.get_logger(__name__)


class AgnesProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False, "return_token_type_ids": False, "return_mm_token_type_ids": True},
        "videos_kwargs": {"return_metadata": True},
    }


@auto_docstring
class AgnesProcessor(ProcessorMixin):
    valid_processor_kwargs = AgnesProcessorKwargs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, config=None, tokenizer=None, **kwargs):
        """Load data with package-owned component classes, ignoring remote maps.

        Do not delegate to ProcessorMixin's component auto-discovery: processor
        JSON is untrusted data and must not name executable Python classes.
        Tokenizer normalization remains owned by Tokenicer.
        """
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        model_config = config
        overrides = {name: kwargs.pop(name, None) for name in ("image_processor", "video_processor")}
        chat_template = kwargs.pop("chat_template", None)
        return_unused = kwargs.pop("return_unused_kwargs", False)
        transport = {name: kwargs[name] for name in (
            "cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder",
        ) if name in kwargs}
        processor_path = cached_file(
            pretrained_model_name_or_path, "processor_config.json",
            _raise_exceptions_for_missing_entries=False, **transport,
        )
        data = {}
        if processor_path is not None:
            with open(processor_path, encoding="utf-8") as stream:
                data = json.load(stream)
        components = {}
        for name, component_class in (("image_processor", AgnesImageProcessor), ("video_processor", AgnesVideoProcessor)):
            if overrides[name] is not None:
                components[name] = overrides[name]
            elif isinstance(data.get(name), dict):
                components[name] = component_class.from_dict(data[name], **kwargs)
            else:
                components[name] = component_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if tokenizer is None:
            if model_config is None:
                model_config = AgnesConfig.from_pretrained(pretrained_model_name_or_path, **transport)
            tokenizer = load_hf_tokenizer(
                str(pretrained_model_name_or_path), model_config=model_config, config=model_config,
                trust_remote_code=False, **kwargs,
            )
        if chat_template is None:
            template_path = cached_file(
                pretrained_model_name_or_path, "chat_template.jinja",
                _raise_exceptions_for_missing_entries=False, **transport,
            )
            if template_path is not None:
                with open(template_path, encoding="utf-8") as stream:
                    chat_template = stream.read()
            else:
                chat_template = data.get("chat_template", getattr(tokenizer, "chat_template", None))
        processor = cls(tokenizer=tokenizer, chat_template=chat_template, **components)
        return (processor, {}) if return_unused else processor

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video_pad|>")
        self.image_token_id = getattr(tokenizer, "image_token_id", None) or tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = getattr(tokenizer, "video_token_id", None) or tokenizer.convert_tokens_to_ids(self.video_token)
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.vision_start_token = getattr(tokenizer, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|vision_end|>")
        self.vision_start_token_id = getattr(tokenizer, "vision_start_token_id", None) or tokenizer.convert_tokens_to_ids(
            self.vision_start_token
        )
        self.vision_end_token_id = getattr(tokenizer, "vision_end_token_id", None) or tokenizer.convert_tokens_to_ids(
            self.vision_end_token
        )

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        per_token = self.image_processor.merge_size**2
        n = image_inputs["image_grid_thw"][image_idx].prod() // per_token
        return self.image_token * n

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        per_token = self.video_processor.merge_size**2
        thw = video_inputs["video_grid_thw"][video_idx]
        n_frames = thw[0]
        per_frame = thw[1:].prod() // per_token
        meta = video_inputs["video_metadata"][video_idx]
        if meta.fps is None:
            logger.warning_once(
                "Frame timestamps are needed to build the video prompt but the `fps` of the input video could not be "
                "inferred (no `video_metadata`, pre-sampled frames?). Defaulting to `fps=24`."
            )
        meta.fps = 24 if meta.fps is None else meta.fps
        stamps = self._frame_timestamps(meta.frames_indices, meta.fps, self.video_processor.temporal_patch_size)
        text = ""
        for f in range(n_frames):
            text += f"<{stamps[f]:.1f} seconds>"
            text += self.vision_start_token + self.video_token * per_frame + self.vision_end_token
        return text

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """Placeholder counts for inputs of the given sizes, without running the
        processors on real pixels."""
        data = {}
        if image_sizes is not None:
            ik = dict(AgnesProcessorKwargs._defaults.get("images_kwargs", {}))
            ik.update(kwargs)
            merge = ik.get("merge_size", None) or self.image_processor.merge_size
            patches = [self.image_processor.get_number_of_image_patches(*s, ik) for s in image_sizes]
            data.update({"num_image_tokens": [p // merge**2 for p in patches], "num_image_patches": patches})
        if video_sizes is not None:
            vk = dict(AgnesProcessorKwargs._defaults.get("videos_kwargs", {}))
            vk.update(kwargs)
            merge = vk.get("merge_size", None) or self.video_processor.merge_size
            patches = [self.video_processor.get_number_of_video_patches(*s, vk) for s in video_sizes]
            data["num_video_tokens"] = [p // merge**2 for p in patches]
        return MultiModalData(**data)

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs):
        """Decode generated ids to text."""
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs,
        )

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids"]

    @staticmethod
    def _frame_timestamps(indices: list[int] | np.ndarray, video_fps: float, merge_size: int = 2):
        """One timestamp per temporal patch: the mean of the first and last
        frame time inside the patch."""
        if not isinstance(indices, list):
            indices = indices.tolist()
        else:
            indices = list(indices)
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        times = [i / video_fps for i in indices]
        return [(times[i] + times[i + merge_size - 1]) / 2 for i in range(0, len(times), merge_size)]


__all__ = ["AgnesProcessor"]
