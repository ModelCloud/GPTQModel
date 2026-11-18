# Adapted for package-owned Transformers 5 loading by ModelCloud.ai, 2026.
# Pinned upstream: see NOTICE.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
import importlib
import json
import os

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import auto_docstring, logging
from transformers.utils.hub import cached_file
from transformers.video_utils import VideoInput

from ....utils.hf import load_hf_tokenizer
from ..native.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from ..native.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from .configuration_interns2_preview import InternS2PreviewConfig


logger = logging.get_logger(__name__)


class InternS2PreviewProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
        "time_series_kwargs": {},
    }


@auto_docstring
class InternS2PreviewProcessor(ProcessorMixin):
    valid_processor_kwargs = InternS2PreviewProcessorKwargs

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
        for name, component_class in (("image_processor", Qwen2VLImageProcessor), ("video_processor", Qwen3VLVideoProcessor)):
            if overrides[name] is not None:
                components[name] = overrides[name]
            elif isinstance(data.get(name), dict):
                components[name] = component_class.from_dict(data[name], **kwargs)
            else:
                components[name] = component_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if tokenizer is None:
            if model_config is None:
                model_config = InternS2PreviewConfig.from_pretrained(pretrained_model_name_or_path, **transport)
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
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.vision_start_token = (
            "<|vision_start|>" if not hasattr(tokenizer, "vision_start_token") else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>" if not hasattr(tokenizer, "vision_end_token") else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )
        self.ts_token = "<TS_CONTEXT>" if not hasattr(tokenizer, "ts_token") else tokenizer.ts_token
        self.ts_start_token = "<|ts|>" if not hasattr(tokenizer, "ts_start_token") else tokenizer.ts_start_token
        self.ts_end_token = "<|/ts|>" if not hasattr(tokenizer, "ts_end_token") else tokenizer.ts_end_token
        self.ts_start_token_id = (
            tokenizer.ts_start_token_id
            if getattr(tokenizer, "ts_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.ts_start_token)
        )
        self.ts_end_token_id = (
            tokenizer.ts_end_token_id
            if getattr(tokenizer, "ts_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.ts_end_token)
        )
        self.ts_token_id = (
            tokenizer.ts_token_id
            if getattr(tokenizer, "ts_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.ts_token)
        )

    @auto_docstring
    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        time_series_paths: list[str] = None,
        time_series_sampling_rates: list[int] = None,
        **kwargs: Unpack[InternS2PreviewProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Args:
            time_series_paths (`list[str]`, *optional*):
                Time-series CSV, NumPy or audio paths, one per prompt.
            time_series_sampling_rates (`list[int]`, *optional*):
                Sampling rates for the corresponding time-series paths.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **ts_values** -- List of time series values to be fed to a model. Returned when `time_series_paths` is not `None`.
            - **ts_sr** -- List of time series sampling rates to be fed to a model. Returned when `time_series_sampling_rates` is not `None`.
            - **ts_lens** -- List of time series lengths to be fed to a model. Returned when `time_series_paths` is not `None`.
            - **num_ts_tokens** -- List of number of time series tokens to be fed to a model. Returned when `time_series_paths` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            InternS2PreviewProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            # If user has not requested video metadata, pop it
            if not kwargs.get("return_metadata"):
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        if time_series_paths is not None:
            assert time_series_sampling_rates is not None, (
                "If time_series_signals is provided, time_series_sampling_rates must also be provided."
            )
            assert len(time_series_paths) == len(time_series_sampling_rates), (
                "The number of time series signals must match the number of sampling rates."
            )
            time_series_inputs = self.time_series_processor(
                ts_paths=time_series_paths, sampling_rates=time_series_sampling_rates
            )
            num_ts_tokens = time_series_inputs.pop("num_ts_tokens")
            assert len(num_ts_tokens) == len(text), (
                "The number of time series signals must match the number of text prompts."
            )
            for i in range(len(text)):
                if f"{self.ts_start_token}{self.ts_token}{self.ts_end_token}" in text[i]:
                    ts_placeholder = self.ts_start_token + self.ts_token * num_ts_tokens[i] + self.ts_end_token
                    text[i] = text[i].replace(
                        f"{self.ts_start_token}{self.ts_token}{self.ts_end_token}", ts_placeholder, 1
                    )
                elif self.ts_token in text[i]:
                    text[i] = text[i].replace(self.ts_token, self.ts_token * num_ts_tokens[i])
        else:
            time_series_inputs = {}

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "Qwen3VL requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                            "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                            "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                        )
                        metadata.fps = 24 if metadata.fps is None else metadata.fps

                    # if timestamps are not provided, calculate them
                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        self.video_processor.temporal_patch_size,
                    )

                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
                        )
                    else:
                        # vllm may input video token directly
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video", "ts"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs, **time_series_inputs}, tensor_type=return_tensors
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = dict(InternS2PreviewProcessorKwargs._defaults.get("images_kwargs", {}))
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None:
            videos_kwargs = dict(InternS2PreviewProcessorKwargs._defaults.get("videos_kwargs", {}))
            videos_kwargs.update(kwargs)
            merge_size = videos_kwargs.get("merge_size") or self.video_processor.merge_size
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(*video_size, videos_kwargs)
                for video_size in video_sizes
            ]
            num_video_tokens = [(num_patches // merge_size**2) for num_patches in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _calculate_timestamps(self, indices: list[int] | np.ndarray, video_fps: float, merge_size: int = 2):
        indices = list(indices)
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        # @JJJYmmm frames are merged by self.merge_size, \
        # so we need to average the timestamps between the first/last frame within the temporal patch
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    def time_series_preprocessor(self, conversation):
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            conversations = conversation
        else:
            conversations = [conversation]

        batch_time_series = []
        batch_time_series_metadata = []
        for conversation in conversations:
            for message in conversation:
                if message["role"] != "user":
                    continue
                time_series_fnames = [
                    content["data"]
                    for content in message["content"]
                    if content.get("type") == "time_series" and "data" in content
                ]
                time_series_rates = [
                    content.get("sampling_rate", None)
                    for content in message["content"]
                    if content.get("type") == "time_series"
                ]
                for path, rate in zip(time_series_fnames, time_series_rates):
                    batch_time_series.append(path)
                    batch_time_series_metadata.append(rate)

        return {
            "time_series_paths": batch_time_series or None,
            "time_series_sampling_rates": batch_time_series_metadata or None,
        }

    def time_series_processor(
        self,
        ts_paths: list[str],
        sampling_rates: list[float],
        do_normalize=True,
        do_truncate=True,
    ) -> BatchFeature:
        assert len(ts_paths) == len(sampling_rates), "ts_paths and sampling_rates must have the same length"

        ts_values = []
        ts_sr = []
        ts_lens = []

        for idx, ts_path in enumerate(ts_paths):
            sr = sampling_rates[idx]
            ext = os.path.splitext(ts_path)[-1].lower()
            if ext in [".wav", ".mp3", ".flac"]:
                sf = importlib.import_module("soundfile")
                ts_input, sr = sf.read(ts_path)  # ts_input: np.ndarray, shape [T] or [T, C]
            elif ext == ".csv":
                pd = importlib.import_module("pandas")
                df = pd.read_csv(ts_path, header=None)
                ts_input = df.values  # [T, C]
            elif ext == ".npy":
                ts_input = np.load(ts_path)  # [T, C]
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            if not isinstance(ts_input, np.ndarray):
                ts_input = np.array(ts_input, dtype=np.float32)

            if do_normalize:
                mean = ts_input.mean(axis=0, keepdims=True)
                std = ts_input.std(axis=0, keepdims=True)
                ts_input = (ts_input - mean) / (std + 1e-8)

            if do_truncate and len(ts_input) > 240000:
                ts_input = ts_input[:240000]  # truncate to 240k to avoid oom

            if ts_input.ndim == 1:
                ts_input = ts_input[:, None]  # [T,C]

            ts_len = ts_input.shape[0]

            if sr is None or sr == 0:  # if no sr provided
                sr = ts_len / 4

            ts_values.append(ts_input)
            ts_sr.append(sr)
            ts_lens.append(ts_len)

        ts_lens = np.array(ts_lens)
        ts_sr = np.array(ts_sr)
        num_ts_tokens = self._get_num_ts_tokens(sampling_rates=ts_sr, ts_lens=ts_lens)
        return BatchFeature(
            data={"ts_values": ts_values, "ts_sr": ts_sr, "ts_lens": ts_lens, "num_ts_tokens": num_ts_tokens}
        )

    def _get_num_ts_tokens(self, sampling_rates, ts_lens):
        strides = np.floor(160 / ((1 + np.exp(-sampling_rates / 100)) ** 6))
        patch_sizes = strides * 2
        embed_lengths = (np.ceil((ts_lens - patch_sizes) / strides) + 1).astype(np.int64)
        num_ts_tokens = [(embed_length // 2 + 1) // 2 for embed_length in embed_lengths]
        return num_ts_tokens


__all__ = ["InternS2PreviewProcessor"]
