# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import (
    ImagesKwargs,
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    VideosKwargs,
)


class ZDTaichu5_0_ImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class ZDTaichu5_0_ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: ZDTaichu5_0_ImagesKwargs
    videos_kwargs: VideosKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class ZDTaichu5_0_Processor(ProcessorMixin):
    r"""
    Constructs a ZDTaichu-5.0 processor which wraps an image processor and a tokenizer into a single processor.
    [`ZDTaichu5_0_Processor`] offers all the functionalities of the image processor and tokenizer. See the
    [`~ZDTaichu5_0_Processor.__call__`] and [`~ZDTaichu5_0_Processor.decode`] for more information.
    Args:
        image_processor ([`AutoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`AutoTokenizer`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]

    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("AutoTokenizer")

    @classmethod
    def from_pretrained(cls, model_id_or_path, config=None, tokenizer=None, **kwargs):
        from .configuration import ZDTaichu5_0_Config
        from .image_processing import ZDTaichu5_0_ImageProcessor

        selected = {key: kwargs[key] for key in (
            "cache_dir", "revision", "token", "local_files_only", "subfolder", "force_download",
        ) if key in kwargs}
        config = config or ZDTaichu5_0_Config.from_pretrained(model_id_or_path, **selected)
        if tokenizer is None:
            from ....utils.hf import load_hf_tokenizer
            from ...definitions.zdtaichu5 import ZDTaichu5QModel
            native = ZDTaichu5QModel.loader.load_tokenizer(model_id_or_path, **selected)
            tokenizer = load_hf_tokenizer(native, config=config, trust_remote_code=False)
        image = ZDTaichu5_0_ImageProcessor.from_pretrained(
            model_id_or_path, image_size=config.force_image_size, patch_size=config.vision_config.patch_size,
            downsample_ratio=config.downsample_ratio, **selected,
        )
        result = cls(image, tokenizer)
        result.config = config
        return result

    def save_pretrained(self, save_directory, **kwargs):
        from pathlib import Path

        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.image_processor.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if hasattr(self, "config") and not (path / "config.json").exists():
            self.config.save_pretrained(path)
        return []

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        # Defaults to Qwen3's built-in vision tokens; overridden by tokenizer_config.json attributes.
        self.image_token       = getattr(tokenizer, "image_token",       "<|image_pad|>")
        self.video_token       = getattr(tokenizer, "video_token",       "<|video_pad|>")
        self.image_start_token = getattr(tokenizer, "image_start_token", "<|vision_start|>")
        self.image_end_token   = getattr(tokenizer, "image_end_token",   "<|vision_end|>")
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
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template or tokenizer.chat_template

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        import re

        options = self._merge_kwargs(
            ZDTaichu5_0_ProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs,
        )
        texts = list(text) if isinstance(text, list) else [text]
        if any(not isinstance(value, str) for value in texts):
            raise ValueError("ZDTaichu5 requires text prompts")
        if videos is not None and len(texts) != 1:
            raise ValueError("Video is not supported for batch size > 1")
        image_data = self.image_processor(images=images, **options["images_kwargs"]) if images is not None else None
        frame_data = self.image_processor(
            images=videos, **{**options["images_kwargs"], "max_num_tiles": 1},
        ) if videos is not None else None
        image_index = 0
        pixel_offset = 0
        used_video = False
        ordered_pixels, ordered_counts, ordered_grids = [], [], []
        tokens_per_tile = self.image_processor.num_image_token

        def wrap(count):
            return self.image_start_token + self.image_token * count + self.image_end_token

        def replace(match):
            nonlocal image_index, pixel_offset, used_video
            token = match.group(0)
            if self.video_token in token:
                if frame_data is None or used_video:
                    raise ValueError("Video placeholder count does not match supplied video")
                used_video = True
                metadata = options.get("videos_kwargs", {}).get("video_metadata")
                fps = getattr(metadata, "fps", None)
                result = "This is a video:\n"
                for i, frame in enumerate(frame_data["pixel_values"]):
                    ordered_pixels.append(frame.unsqueeze(0))
                    ordered_counts.append(1)
                    ordered_grids.append([1, 1, 1])
                    prefix = f"Frame {i + 1} sampled at {i / fps:.2f} seconds: " if fps else f"Frame {i + 1}: "
                    result += prefix + wrap(tokens_per_tile) + "\n"
                return result
            if image_data is None or image_index >= len(image_data["num_patches"]):
                raise ValueError("Image placeholder count does not match supplied images")
            count = int(image_data["num_patches"][image_index])
            ordered_pixels.append(image_data["pixel_values"][pixel_offset:pixel_offset + count])
            ordered_counts.append(count)
            ordered_grids.append(image_data["image_grid_thw"][image_index])
            pixel_offset += count
            image_index += 1
            return wrap(tokens_per_tile * count)

        placeholders = "|".join(map(re.escape, (self.image_token, self.video_token)))
        pattern = re.compile(
            "(?:" + re.escape(self.image_start_token) + ")?(?:" + placeholders + ")(?:"
            + re.escape(self.image_end_token) + ")?",
        )
        texts = [pattern.sub(replace, value) for value in texts]
        if image_data is not None and image_index != len(image_data["num_patches"]):
            raise ValueError("Supplied images have no matching prompt placeholders")
        if frame_data is not None and not used_video:
            raise ValueError("Supplied video has no matching prompt placeholder")
        text_options = options["text_kwargs"]
        return_tensors = text_options.pop("return_tensors", None)
        data = dict(self.tokenizer(texts, **text_options))
        ids = data["input_ids"]
        data["mm_token_type_ids"] = [
            [1 if token == self.image_token_id else 2 if token == self.video_token_id else 0 for token in row]
            for row in ids
        ]
        if ordered_pixels:
            data.update(pixel_values=torch.cat(ordered_pixels), num_patches=ordered_counts,
                        image_grid_thw=torch.stack([torch.as_tensor(g, dtype=torch.long) for g in ordered_grids]))
        return BatchFeature(data=data, tensor_type=return_tensors)

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
            images_kwargs = ZDTaichu5_0_ProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})
        return MultiModalData(**vision_data)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

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

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
        # Note: video_grid_thw is NOT emitted under multi-image-as-video —
        # frame data lives in image_grid_thw alongside real images.
        return names_from_processor + ["mm_token_type_ids"]


    def from_messages(
        self,
        messages: list,
        return_tensors: str = "pt",
        add_vision_id: bool = True,
        **kwargs,
    ) -> BatchFeature:
        """
        Prepare model inputs directly from Qwen-style structured messages.

        This is the high-level entry point that handles the full pipeline:
        structured messages → vision loading → chat template → tokenization.

        Supports messages with typed content lists::

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": "photo.jpg"},
                    {"type": "text", "text": "What's in this image?"},
                ]},
            ]

        Video inputs (file path, URL, or list of frame paths)::

            messages = [
                {"role": "user", "content": [
                    {"type": "video", "video": "clip.mp4", "fps": 2.0},
                    {"type": "text", "text": "Describe this video."},
                ]},
            ]

        Multi-image with automatic labelling::

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": "a.jpg"},
                    {"type": "image", "image": "b.jpg"},
                    {"type": "text", "text": "Compare them."},
                ]},
            ]
            # With add_vision_id=True (default), the prompt includes:
            #   Picture 1: <|vision_start|><|image_pad|><|vision_end|>
            #   Picture 2: <|vision_start|><|image_pad|><|vision_end|>
            #   Compare them.

        Args:
            messages: List of message dicts with structured ``content``.
            return_tensors: Framework for returned tensors (default ``"pt"``).
            add_vision_id: If ``True`` (default), the chat template prepends
                ``Picture N:`` / ``Video N:`` labels before each vision token.
                Set to ``False`` to omit labels.
            **kwargs: Forwarded to ``self.__call__``.

        Returns:
            ``BatchFeature`` ready for ``model.generate(**inputs)``.
        """
        from .vision_utils import process_vision_info

        # 1. Load images and videos from the structured messages
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages)

        # 2. Apply chat template — pass structured messages directly so the
        #    template can iterate typed content dicts, count vision elements,
        #    and emit "Picture N:" / "Video N:" labels when add_vision_id=True.
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=add_vision_id,
        )

        # 3. Prepare video inputs and metadata for timestamps
        videos_kwargs = {}
        flat_videos = None
        if video_inputs is not None:
            if len(video_inputs) > 1:
                raise ValueError(
                    "Multiple videos in a single message batch are not yet "
                    "supported. Please use one video per call."
                )
            flat_videos = video_inputs[0]  # List[Image.Image]

            if video_kwargs.get("metadata_list"):
                meta = video_kwargs["metadata_list"][0]
                fps = meta.get("sample_fps") or meta.get("fps")
                if fps:
                    from transformers.video_utils import VideoMetadata
                    videos_kwargs["video_metadata"] = VideoMetadata(
                        fps=fps,
                        total_num_frames=len(flat_videos),
                    )

        return self(
            images=image_inputs,
            text=prompt,
            videos=flat_videos,
            return_tensors=return_tensors,
            videos_kwargs=videos_kwargs,
            **kwargs,
        )

    @property
    def ctx_image_token_id(self) -> int:
        return self.image_token_id


__all__ = ["ZDTaichu5_0_Processor"]
