# Copyright 2026 The HuggingFace Team. All rights reserved.
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
# GPT-QModel adaptation: package-owned loading and Transformers 5 execution.
# Pinned source, modifications and qualification limits: README.md and NOTICE.

import copy
import math
import os
import re

import numpy as np
import torch
from transformers.audio_utils import AudioInput, make_list_of_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import is_pil_image, is_valid_image, make_flat_list_of_images
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, VideosKwargs
from transformers.utils import auto_docstring, is_numpy_array, is_torch_tensor
from transformers.video_utils import is_valid_video, make_batched_metadata, make_batched_videos

from .modeling_lfm3_omni import Lfm3OmniFeatureExtractorKwargs
from .modeling_liquid_codec import LiquidCodecModel


class Lfm3OmniVideosKwargs(VideosKwargs, total=False):
    """
    use_audio_in_video (`bool`, *optional*, defaults to `False`):
        Whether each video comes with its soundtrack: the first audio clip after the video in the text (the one
        `apply_chat_template(load_audio_from_video=True)` puts right behind the video) is interleaved after the
        video's frame groups instead of standing in its own media span.
    """

    use_audio_in_video: bool


class Lfm3OmniProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: Lfm3OmniFeatureExtractorKwargs
    videos_kwargs: Lfm3OmniVideosKwargs
    _defaults = {"videos_kwargs": {"return_metadata": True, "use_audio_in_video": False}}


# The prompt format of the training data: ChatML turns, one placeholder per clip, no thinking block
DEFAULT_CHAT_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
    "{%- if message['content'] is string -%}{{- message['content'] -}}"
    "{%- else -%}{%- for content in message['content'] -%}"
    "{%- if content['type'] == 'audio' -%}{{- '<|audio|>' -}}"
    "{%- elif content['type'] == 'audio_out' -%}{{- '<|audio_out|>' -}}"
    "{%- elif content['type'] == 'text' -%}{{- content['text'] -}}"
    "{%- endif -%}{%- endfor -%}{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
)


def _media_per_prompt(media, is_item) -> list[int] | None:
    # rows name the prompt that owns them while the expansion consumes media in flat order; a row is a list of whole
    # items or one stacked batch, and a list of single frames is one video, not a row
    if not isinstance(media, (list, tuple)):
        return None
    counts = []
    for row in media:
        if isinstance(row, (list, tuple)) and all(is_item(item) for item in row):
            counts.append(len(row))
        elif (is_numpy_array(row) or is_torch_tensor(row)) and row.ndim == 4 and (len(row) == 0 or is_item(row[0])):
            counts.append(len(row))
        else:
            return None
    return counts


def _check_vision_placeholder_counts(processor, prompts, images, videos) -> None:
    for name, token, media, flatten, is_item in (
        (
            "images",
            processor.image_token,
            images,
            make_flat_list_of_images,
            lambda item: is_valid_image(item) and (is_pil_image(item) or item.ndim == 3),
        ),
        (
            "videos",
            processor.video_token,
            videos,
            make_batched_videos,
            lambda item: isinstance(item, str) or (not is_pil_image(item) and is_valid_video(item)),
        ),
    ):
        num_placeholders = sum(prompt.count(token) for prompt in prompts)
        num_media = 0 if media is None else len(flatten(media))
        if num_placeholders != num_media:
            raise ValueError(f"Found {num_placeholders} {token} tokens in the text but received {num_media} {name}")
        # one prompt cannot take another's media
        per_prompt = _media_per_prompt(media, is_item) if len(prompts) > 1 else None
        in_text = [prompt.count(token) for prompt in prompts]
        if per_prompt is not None and per_prompt != in_text:
            raise ValueError(f"Found {in_text} {token} tokens per prompt but received {per_prompt} {name} per prompt")


def _check_video_metadata_count(videos, video_metadata) -> None:
    # the shared sampler pairs clips with metadata by `zip`, which drops the clips past the last entry
    if video_metadata is None:
        return
    videos = make_batched_videos(videos)
    entries = make_batched_metadata(videos, video_metadata) if video_metadata != [] else []
    if len(entries) != len(videos):
        raise ValueError(
            f"Received {len(videos)} videos but {len(entries)} `video_metadata` entries; pass one entry per video"
        )


DEFAULT_OMNI_CHAT_TEMPLATE = DEFAULT_CHAT_TEMPLATE.replace(
    "{%- if content['type'] == 'audio' -%}",
    "{%- if content['type'] == 'image' -%}{{- '<|image|>' -}}"
    "{%- elif content['type'] == 'video' -%}{{- '<|video|>' -}}"
    "{%- elif content['type'] == 'audio' -%}",
)


@auto_docstring
class Lfm3OmniProcessor(ProcessorMixin):
    """The LFM3-Audio processor extended with the LFM3-VL image and video processors: every image, video frame group
    and input clip is expanded in place, wrapped in the media span tokens the Omni SFT data uses."""

    attributes = ["image_processor", "feature_extractor", "tokenizer", "video_processor"]
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "LiquidCodecModel"
    valid_processor_kwargs = Lfm3OmniProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        feature_extractor=None,
        tokenizer=None,
        video_processor=None,
        audio_tokenizer=None,
        chat_template=None,
        audio_token="<|audio|>",
        audio_out_token="<|audio_out|>",
        image_token="<|image|>",
        video_token="<|video|>",
        vision_start_token="<|media_start|>",
        vision_end_token="<|media_end|>",
        vision_markers=True,
        audio_markers=True,
        **kwargs,
    ):
        r"""
        image_processor ([`Lfm3VlImageProcessor`]):
            The image patch processor.
        feature_extractor ([`Lfm3OmniFeatureExtractor`]):
            The log-mel feature extractor.
        tokenizer ([`PreTrainedTokenizerFast`]):
            The LFM3 tokenizer.
        video_processor ([`Lfm3VlVideoProcessor`], *optional*):
            The video frame sampler and patch processor.
        audio_tokenizer ([`LiquidCodecModel`], *optional*):
            The codec for 24 kHz output audio.
        chat_template (`str`, *optional*):
            The conversation template.
        audio_token (`str`, *optional*, defaults to `"<|audio|>"`):
            The input audio placeholder.
        audio_out_token (`str`, *optional*, defaults to `"<|audio_out|>"`):
            The output audio placeholder.
        image_token (`str`, *optional*, defaults to `"<|image|>"`):
            The image placeholder.
        video_token (`str`, *optional*, defaults to `"<|video|>"`):
            The video placeholder.
        vision_start_token (`str`, *optional*, defaults to `"<|media_start|>"`):
            The token opening an image, a video and each of its timestamped frame groups.
        vision_end_token (`str`, *optional*, defaults to `"<|media_end|>"`):
            The token closing them.
        vision_markers (`bool`, *optional*, defaults to `True`):
            Whether images and videos are wrapped in the vision span tokens, as the Omni SFT data renders them.
        audio_markers (`bool`, *optional*, defaults to `True`):
            Whether input audio clips are wrapped in the same span tokens, as the Omni SFT data renders them.
        """
        self.audio_token = audio_token
        self.audio_out_token = audio_out_token
        self.image_token = image_token
        self.video_token = video_token
        self.vision_start_token = vision_start_token
        self.vision_end_token = vision_end_token
        self.vision_markers = vision_markers
        self.audio_markers = audio_markers
        for token in (audio_token, audio_out_token, image_token, video_token):
            if tokenizer.convert_tokens_to_ids(token) in (None, tokenizer.unk_token_id):
                tokenizer.add_tokens([token], special_tokens=True)
        # the span tokens exist in every LFM3 tokenizer under one of two names (`<|vision_*|>` in liquid7.1,
        # `<|media_*|>` in liquid7.1_media) at the same ids; never mint new ones
        for attribute, token, fallback in (
            ("vision_start_token", vision_start_token, "<|vision_start|>"),
            ("vision_end_token", vision_end_token, "<|vision_end|>"),
        ):
            if tokenizer.convert_tokens_to_ids(token) in (None, tokenizer.unk_token_id):
                if tokenizer.convert_tokens_to_ids(fallback) not in (None, tokenizer.unk_token_id):
                    setattr(self, attribute, fallback)
                else:
                    tokenizer.add_tokens([token], special_tokens=True)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        self.audio_out_token_id = tokenizer.convert_tokens_to_ids(audio_out_token)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(video_token)
        super().__init__(
            image_processor,
            feature_extractor,
            tokenizer,
            video_processor,
            chat_template=chat_template if chat_template is not None else DEFAULT_OMNI_CHAT_TEMPLATE,
        )
        # a released transformers only accepts its own codec classes as `audio_tokenizer`
        self.audio_tokenizer = audio_tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, audio_tokenizer=None, **kwargs):
        # Component classes are package metadata. Checkpoint auto_map and class
        # strings never participate in model, codec or processor selection.
        if args:
            raise TypeError("Pass processor options by keyword")
        from ....utils.hf import load_hf_tokenizer
        from .configuration_lfm3_omni import Lfm3OmniConfig
        from .feature_extraction_lfm3_omni import Lfm3OmniFeatureExtractor
        from .image_processing_lfm3_vl import Lfm3VlImageProcessor
        from .video_processing_lfm3_vl import Lfm3VlVideoProcessor

        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        config = kwargs.pop("config", None)
        tokenizer = kwargs.pop("tokenizer", None)
        processor_dict, options = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)
        if tokenizer is None:
            config = config or Lfm3OmniConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            tokenizer = load_hf_tokenizer(
                pretrained_model_name_or_path, config=config, model_config=config, trust_remote_code=False, **kwargs
            )
        components = []
        for name, component_class in (
            ("image_processor", Lfm3VlImageProcessor),
            ("feature_extractor", Lfm3OmniFeatureExtractor),
            ("video_processor", Lfm3VlVideoProcessor),
        ):
            values = processor_dict.get(name)
            if isinstance(values, dict):
                values = {key: value for key, value in values.items() if key not in (
                    "auto_map", "processor_class", f"{name}_type",
                )}
                components.append(component_class(**values))
            else:
                components.append(component_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
        processor = cls.from_args_and_dict(
            [components[0], components[1], tokenizer, components[2]], processor_dict, **options
        )
        if audio_tokenizer is not None:
            processor.audio_tokenizer = audio_tokenizer
        return processor

    def to_dict(self):
        # `ProcessorMixin.to_dict` deep-copies the attributes; the codec weights are saved on their own. A shallow
        # clone drops the codec, so concurrent calls on `self` keep it.
        clone = copy.copy(self)
        codec = clone.__dict__.pop("audio_tokenizer", None)
        output = ProcessorMixin.to_dict(clone)
        if codec is not None:
            output["audio_tokenizer_subfolder"] = "audio_tokenizer"
        return output

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        files = super().save_pretrained(save_directory, push_to_hub=False, **kwargs)
        if getattr(self, "audio_tokenizer", None) is not None:
            self.audio_tokenizer.save_pretrained(os.path.join(save_directory, "audio_tokenizer"))
        if push_to_hub:
            repo_id = kwargs.pop("repo_id", os.path.basename(os.path.abspath(save_directory)))
            self.push_to_hub(repo_id, **kwargs)
        return files

    @classmethod
    def get_processor_dict(cls, pretrained_model_name_or_path, **kwargs):
        codec_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in ("cache_dir", "force_download", "local_files_only", "token", "revision")
        }
        parent = kwargs.get("subfolder") or ""
        processor_dict, kwargs = super().get_processor_dict(pretrained_model_name_or_path, **kwargs)
        subfolder = processor_dict.pop("audio_tokenizer_subfolder", None)
        if subfolder is not None:
            if subfolder != "audio_tokenizer":
                raise ValueError("Expected the bundled codec in the audio_tokenizer subfolder")
            codec_class = LiquidCodecModel
            processor_dict["audio_tokenizer"] = codec_class.from_pretrained(
                pretrained_model_name_or_path, subfolder=os.path.join(parent, subfolder), **codec_kwargs
            )
        return processor_dict, kwargs

    @auto_docstring
    def __call__(
        self,
        text=None,
        audio: AudioInput | None = None,
        audio_out: AudioInput | None = None,
        images=None,
        videos=None,
        **kwargs,
    ):
        r"""
        audio_out (`np.ndarray`, `torch.Tensor` or a list of them, *optional*):
            24 kHz clips, one per `audio_out_token` in the text, encoded with the codec into `audio_out_codes`; the
            placeholder is repeated once per frame plus once for the end-of-speech frame that closes every clip.
        videos (`VideoInput`, *optional*):
            One clip per `video_token`. With `use_audio_in_video=True` each video takes the first audio clip after it
            as its soundtrack, interleaved after each timestamped frame group like the Omni SFT data; otherwise every
            audio clip is its own media span.
        """
        for name, media in (("audio", audio), ("audio_out", audio_out), ("images", images), ("videos", videos)):
            if isinstance(media, (list, tuple)) and len(media) == 0:
                raise ValueError(f"`{name}` must hold at least one item; pass `{name}=None` for prompts without one")
        if text is None:
            # media-only calls return the features alone, like the audio processor
            if audio_out is not None:
                raise ValueError("`audio_out` needs the text holding its placeholders")
            return super().__call__(audio=audio, images=images, videos=videos, **kwargs)
        single = isinstance(text, str)
        caller = [text] if single else list(text)
        stripped = [self._strip_spans(prompt) for prompt in caller]
        prompts, offsets = [prompt for prompt, _ in stripped], [row for _, row in stripped]
        audio_out_codes = None
        if audio_out is not None:
            prompts, audio_out_codes, audio_out_offsets = self._expand_audio_out(prompts, audio_out)
            offsets = self._compose_replacement_offsets(caller, offsets, audio_out_offsets)
            merged = self._merge_kwargs(
                self.valid_processor_kwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
            )
            return_tensors = merged["text_kwargs"].get("return_tensors")
        if videos is not None:
            outputs = self._call_with_videos(prompts, audio, images, videos, single, **kwargs)
        else:
            outputs = super().__call__(
                text=prompts[0] if single else prompts, audio=audio, images=images, videos=videos, **kwargs
            )
        if "text_replacement_offsets" in outputs:
            # the expansion saw the prompts after the span pairs were stripped and the speech placeholders repeated
            outputs["text_replacement_offsets"] = self._compose_replacement_offsets(
                caller, offsets, outputs["text_replacement_offsets"]
            )
        if audio_out_codes is not None:
            self._check_audio_out_placeholders(outputs["input_ids"], audio_out_codes.shape[0])
            outputs["audio_out_codes"] = self._as_return_tensors(audio_out_codes, return_tensors)
        return outputs

    def _check_audio_out_placeholders(self, input_ids, num_frames: int) -> None:
        # truncation may have dropped placeholders the codes still cover
        rows = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
        rows = [rows] if rows and isinstance(rows[0], int) else rows
        num_placeholders = sum(row.count(self.audio_out_token_id) for row in rows)
        if num_placeholders != num_frames:
            raise ValueError(
                f"Found {num_placeholders} {self.audio_out_token} tokens after tokenization for {num_frames} "
                "speech frames; disable truncation or shorten the prompt"
            )

    @staticmethod
    def _as_return_tensors(codes: torch.Tensor, return_tensors):
        return BatchFeature({"codes": codes}, tensor_type=return_tensors)["codes"]

    @staticmethod
    def _rewrite_with_offsets(prompt: str, pattern: str, replace, kind: str) -> tuple[str, list[dict]]:
        """Rewrites `prompt` like `ProcessorMixin.get_text_with_replacements`, with the same offset records."""
        pieces, row, last, offset = [], [], 0, 0
        for match in re.finditer(pattern, prompt):
            start, end = match.span()
            replacement = replace(match)
            pieces.extend((prompt[last:start], replacement))
            row.append(
                {
                    "type": kind,
                    "span": (start, end),
                    "new_span": (start + offset, start + offset + len(replacement)),
                    "text": match.group(),
                    "replacement": replacement,
                }
            )
            offset += len(replacement) - (end - start)
            last = end
        pieces.append(prompt[last:])
        return "".join(pieces), row

    @staticmethod
    def _compose_replacement_offsets(prompts: list[str], first: list[list[dict]], second) -> list[list[dict]]:
        """The records of two successive rewrites (the caller's text A to B, then B to C) as one rewrite of A: an entry
        of the second that covers entries of the first absorbs them, so every entry spans the caller's text."""
        second = second or [[] for _ in prompts]
        composed = []
        for prompt, row_first, row_second in zip(prompts, first, second):

            def to_first(position, row_first=row_first):
                return position - sum(
                    len(r["replacement"]) - (r["span"][1] - r["span"][0])
                    for r in row_first
                    if r["new_span"][1] <= position
                )

            mapped = []
            for entry in row_second:
                start, end = to_first(entry["span"][0]), to_first(entry["span"][1])
                mapped.append({**entry, "span": (start, end), "text": prompt[start:end]})
            kept = [
                r
                for r in row_first
                if not any(m["span"][0] <= r["span"][0] and r["span"][1] <= m["span"][1] for m in mapped)
            ]
            row, offset = [], 0
            for entry in sorted(kept + mapped, key=lambda r: r["span"]):
                start, end = entry["span"]
                row.append({**entry, "new_span": (start + offset, start + offset + len(entry["replacement"]))})
                offset += len(entry["replacement"]) - (end - start)
            composed.append(row)
        return composed

    def _expand_audio_out(
        self, prompts: list[str], audio_out: AudioInput
    ) -> tuple[list[str], torch.Tensor, list[list[dict]]]:
        if getattr(self, "audio_tokenizer", None) is None:
            raise ValueError("`audio_out` requires the processor to hold the codec as `audio_tokenizer`")
        clips = make_list_of_audio(audio_out)
        if sum(prompt.count(self.audio_out_token) for prompt in prompts) != len(clips):
            raise ValueError(f"The number of {self.audio_out_token} tokens in the text must match the number of clips")
        codes = [self.encode_audio(clip) for clip in clips]
        # the model ends speech with a frame of end-of-speech codes before the next text token; the prompt and the
        # training targets carry it the same way `generate(return_audio_eos=True)` does
        eos_frame = torch.full((1, codes[0].shape[1]), self.audio_eos_code, dtype=codes[0].dtype)
        codes = [torch.cat((frames, eos_frame)) for frames in codes]
        frame_counts = iter(code.shape[0] for code in codes)
        expanded, offsets = [], []
        for prompt in prompts:
            prompt, row = self._rewrite_with_offsets(
                prompt,
                re.escape(self.audio_out_token),
                lambda match: self.audio_out_token * next(frame_counts),
                "audio_out",
            )
            expanded.append(prompt)
            offsets.append(row)
        return expanded, torch.cat(codes), offsets

    @property
    def audio_eos_code(self) -> int:
        """The end-of-speech code: the first id after the codec's vocabulary."""
        return math.prod(self.audio_tokenizer.config.fsq_levels)

    @torch.no_grad()
    def encode_audio(self, audio) -> torch.Tensor:
        codec = self.audio_tokenizer
        waveform = torch.as_tensor(audio if torch.is_tensor(audio) else np.asarray(audio), dtype=torch.float32)
        # the quantizer rounds non-finite values to INT64_MIN, which would surface as codes
        if waveform.ndim != 1 or not torch.isfinite(waveform).all():
            raise ValueError("`audio_out` clips must be mono waveforms of finite samples")
        # an offloaded codec keeps some weights on `meta`: send the clip to a device that holds weights, and the hooks
        # move it on from there
        device = next(
            (param.device for param in codec.parameters() if param.device.type != "meta"), torch.device("cpu")
        )
        # the codec's return type follows its config's `return_dict`
        codes = codec.encode(waveform.to(device)[None], return_dict=True).audio_codes[0].T.cpu()
        # a compiled codec skips its latent check, and a non-finite latent rounds to INT64_MIN
        if codes.numel() and not 0 <= int(codes.min()) <= int(codes.max()) < self.audio_eos_code:
            raise ValueError("The audio overflows the codec; expected `audio_out` samples in [-1, 1]")
        return codes

    def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        if text is None:
            return
        prompts = [text] if isinstance(text, str) else text
        num_placeholders = sum(prompt.count(self.audio_token) for prompt in prompts)
        num_clips = 0 if audio is None else len(make_list_of_audio(audio))
        if num_placeholders != num_clips:
            raise ValueError(
                f"Found {num_placeholders} {self.audio_token} tokens in the text but received {num_clips} clips"
            )
        if text is not None:
            _check_vision_placeholder_counts(self, [text] if isinstance(text, str) else text, images, videos)

    def _process_audio(self, audio: AudioInput, **kwargs):
        # the backend only serves `apply_chat_template`'s audio loading
        kwargs.pop("load_audio_backend", None)
        audio_inputs = self.feature_extractor(audio, **kwargs)
        lengths = np.asarray(audio_inputs["feature_attention_mask"]).sum(-1).tolist()
        audio_inputs["num_audio_tokens"] = [
            self.feature_extractor.num_encoder_tokens(int(length)) for length in lengths
        ]
        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(lengths))]
        return audio_inputs, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        return self._wrap(self.audio_token * audio_inputs["num_audio_tokens"][audio_idx], self.audio_markers)

    @property
    def unused_input_names(self) -> list[str]:
        return ["num_audio_tokens"]

    @property
    def model_input_names(self):
        return list(
            dict.fromkeys(
                self.tokenizer.model_input_names
                + self.feature_extractor.model_input_names
                + self.image_processor.model_input_names
                + (self.video_processor.model_input_names if self.video_processor is not None else [])
                + ["audio_out_codes"]
            )
        )

    def _wrap(self, placeholders: str, markers: bool) -> str:
        return self.vision_start_token + placeholders + self.vision_end_token if markers else placeholders

    def _strip_spans(self, prompt: str) -> tuple[str, list[dict]]:
        # a chat template may already wrap a placeholder in the span pair; the expansion adds exactly one pair
        wrapped = {
            self.vision_start_token + token + self.vision_end_token: token
            for token in (self.image_token, self.video_token, self.audio_token)
        }
        return self._rewrite_with_offsets(
            prompt, "|".join(map(re.escape, wrapped)), lambda match: wrapped[match.group()], "span"
        )

    def apply_chat_template(
        self,
        conversation,
        chat_template=None,
        tools=None,
        documents=None,
        add_generation_prompt=False,
        continue_final_message=False,
        return_assistant_tokens_mask=False,
        tokenize=False,
        return_tensors=None,
        return_dict=False,
        load_audio_from_video=False,
        processor_kwargs=None,
        **kwargs,
    ):
        """With `load_audio_from_video=True`, each video's soundtrack becomes the audio part right behind it, which
        `use_audio_in_video` pairs with the video; standalone clips are kept (the shared implementation drops them)."""
        if load_audio_from_video and tokenize:
            load_audio_from_video = False
            conversation = self._with_soundtrack_parts(conversation)
        return super().apply_chat_template(
            conversation,
            chat_template,
            tools=tools,
            documents=documents,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            tokenize=tokenize,
            return_tensors=return_tensors,
            return_dict=return_dict,
            load_audio_from_video=load_audio_from_video,
            processor_kwargs=processor_kwargs,
            **kwargs,
        )

    @staticmethod
    def _with_soundtrack_parts(conversation):
        if not conversation:
            return conversation
        batched = isinstance(conversation[0], (list, tuple))
        rewritten = []
        for messages in conversation if batched else [conversation]:
            rewritten.append([])
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        parts.append(part)
                        source = next((part[key] for key in ("video", "url", "path") if key in part), None)
                        if part.get("type") == "video" and source is not None:
                            parts.append({"type": "audio", "audio": source})
                    message = {**message, "content": parts}
                rewritten[-1].append(message)
        return rewritten if batched else rewritten[0]

    def _attach_soundtracks(self, prompt: str) -> tuple[str, list[dict]]:
        """`use_audio_in_video`: each video takes the first audio placeholder after it as its soundtrack, moved right
        behind the video; returns the prompt and the offset records of the moves."""
        row, taken = [], set()
        audio = [match.start() for match in re.finditer(re.escape(self.audio_token), prompt)]
        for video in re.finditer(re.escape(self.video_token), prompt):
            position = next((p for p in audio if p >= video.end() and p not in taken), None)
            if position is None:
                raise ValueError("`use_audio_in_video` needs an audio clip after each video")
            taken.add(position)
            row.append(
                {"type": "audio", "span": (video.end(), video.end()), "text": "", "replacement": self.audio_token}
            )
            row.append(
                {
                    "type": "audio",
                    "span": (position, position + len(self.audio_token)),
                    "text": self.audio_token,
                    "replacement": "",
                }
            )
        row.sort(key=lambda r: r["span"])
        pieces, last, offset = [], 0, 0
        for entry in row:
            start, end = entry["span"]
            pieces.extend((prompt[last:start], entry["replacement"]))
            entry["new_span"] = (start + offset, start + offset + len(entry["replacement"]))
            offset += len(entry["replacement"]) - (end - start)
            last = end
        pieces.append(prompt[last:])
        return "".join(pieces), row

    def _call_with_videos(self, prompts, audio, images, videos, single, **kwargs):
        """Expands the placeholders itself: a soundtrack's encoder tokens are split across its video's frame groups,
        which the generic one-clip-one-run expansion cannot do."""
        images, prompts, videos, audio = self.prepare_inputs_layout(
            images=images, text=prompts, videos=videos, audio=audio, **kwargs
        )
        self.validate_inputs(images=images, text=prompts, videos=videos, audio=audio, **kwargs)
        merged = self._merge_kwargs(
            self.valid_processor_kwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        return_tensors = merged["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = merged["text_kwargs"].pop("return_mm_token_type_ids", False)
        return_text_replacement_offsets = merged["text_kwargs"].pop("return_text_replacement_offsets", False)
        # Timestamps always need metadata internally, independently of whether the caller wants it returned.
        return_metadata = kwargs.get("return_metadata", kwargs.get("videos_kwargs", {}).get("return_metadata", False))
        merged["videos_kwargs"]["return_metadata"] = True
        use_audio_in_video = merged["videos_kwargs"].pop("use_audio_in_video")
        unattached, attach_offsets = prompts, [[] for _ in prompts]
        if use_audio_in_video:
            attached = [self._attach_soundtracks(prompt) for prompt in prompts]
            prompts, attach_offsets = [prompt for prompt, _ in attached], [row for _, row in attached]
        outputs = {}
        clips = make_list_of_audio(audio) if audio is not None else []
        videos = make_batched_videos(videos)
        self._check_video_clock(videos, merged["videos_kwargs"].get("video_metadata"))
        image_inputs = self.image_processor(images=images, **merged["images_kwargs"]) if images is not None else None
        if image_inputs is not None:
            outputs.update(image_inputs)
        if clips:
            # the decoding backend is a loader option, not a feature-extractor one (the shared path drops it too)
            merged["audio_kwargs"].pop("load_audio_backend", None)
            audio_inputs = self.feature_extractor(clips, **merged["audio_kwargs"])
            lengths = np.asarray(audio_inputs["feature_attention_mask"]).sum(-1).tolist()
            audio_tokens = [self.feature_extractor.num_encoder_tokens(int(length)) for length in lengths]
            outputs.update(audio_inputs)
        video_inputs = self.video_processor(videos=videos, **merged["videos_kwargs"])
        outputs.update(
            {key: value for key, value in video_inputs.items() if key != "video_metadata" or return_metadata}
        )
        video = self.video_token + (self.audio_token if use_audio_in_video else "")
        pattern = re.compile("|".join(map(re.escape, (video, self.image_token, self.audio_token))))
        counters = {"image": 0, "video": 0, "audio": 0}

        def expand(match):
            token = match.group(0)
            if token == self.image_token:
                text = self.replace_image_token(image_inputs, counters["image"])
                counters["image"] += 1
                return text
            if token == self.audio_token:
                text = self._wrap(self.audio_token * audio_tokens[counters["audio"]], self.audio_markers)
                counters["audio"] += 1
                return text
            soundtrack = None
            if token != self.video_token:  # the pair `_attach_soundtracks` made
                soundtrack = audio_tokens[counters["audio"]]
                counters["audio"] += 1
            text = self._video_expansion(video_inputs, counters["video"], soundtrack)
            counters["video"] += 1
            return text

        expanded, batch_offsets = [], []
        for prompt in prompts:
            pieces, offsets = [], []
            last, offset = 0, 0
            for match in pattern.finditer(prompt):
                start, end = match.span()
                replacement = expand(match)
                pieces.extend((prompt[last:start], replacement))
                token = match.group(0)
                offsets.append(
                    {
                        "type": "video"
                        if token.startswith(self.video_token)
                        else "image"
                        if token == self.image_token
                        else "audio",
                        "span": (start, end),
                        "new_span": (start + offset, start + offset + len(replacement)),
                        "text": token,
                        "replacement": replacement,
                    }
                )
                offset += len(replacement) - (end - start)
                last = end
            pieces.append(prompt[last:])
            expanded.append("".join(pieces))
            batch_offsets.append(offsets)
        text_inputs = self.tokenizer(expanded, **merged["text_kwargs"])
        self._check_special_mm_tokens(expanded, text_inputs, modalities=["image", "video", "audio"])
        if return_text_replacement_offsets:
            text_inputs["text_replacement_offsets"] = self._compose_replacement_offsets(
                unattached, attach_offsets, batch_offsets
            )
        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        outputs.update(text_inputs)
        return BatchFeature(outputs, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion)

    def _video_expansion(self, video_inputs: dict, video_idx: int, soundtrack_tokens: int | None = None) -> str:
        grid = video_inputs["video_grid_thw"][video_idx]
        patch_tokens = int(grid[1:].prod() // self.video_processor.merge_size**2)
        metadata = video_inputs["video_metadata"][video_idx]
        if metadata.frames_indices is None or not (
            metadata.fps is not None and math.isfinite(metadata.fps) and metadata.fps > 0
        ):
            raise ValueError(
                "LFM3-Omni prompts carry frame timestamps: pass videos with `video_metadata` holding the sampled frame "
                "indices and a positive fps"
            )
        # a released transformers reads torchcodec's scanned average fps, which carries float noise
        # (30.000000000000004) where the fork reads the container header
        metadata.fps = round(metadata.fps, 6)
        temporal = self.video_processor.temporal_patch_size
        timestamps = self._calculate_timestamps(metadata.frames_indices, metadata.fps, temporal)[: int(grid[0])]
        bounds = None
        if soundtrack_tokens is not None:
            # the soundtrack's encoder tokens are split at each frame group's first frame
            starts = [index / metadata.fps for index in list(metadata.frames_indices)[::temporal]][: int(grid[0])]
            seconds_per_token = (
                self.feature_extractor.hop_length
                * self.feature_extractor.encoder_stride
                / self.feature_extractor.sampling_rate
            )
            bounds = (
                [0]
                + [min(soundtrack_tokens, round(start / seconds_per_token)) for start in starts[1:]]
                + [soundtrack_tokens]
            )
            bounds = [max(bounds[: i + 1]) for i in range(len(bounds))]
        groups = ""
        for index, timestamp in enumerate(timestamps):
            groups += f"<{timestamp:.1f} seconds>" + self._wrap(self.video_token * patch_tokens, self.vision_markers)
            if bounds is not None:
                groups += self.audio_token * (bounds[index + 1] - bounds[index])
        return self._wrap(groups, self.vision_markers)

    def _check_video_clock(self, videos, video_metadata) -> None:
        _check_video_metadata_count(videos, video_metadata)
        if video_metadata is None:
            if any(not isinstance(video, str) for video in make_batched_videos(videos)):
                # decoded frames carry no clock: without metadata the timestamps would silently assume 24 fps
                raise ValueError(
                    "LFM3-Omni prompts carry frame timestamps: pass decoded frames with `video_metadata` holding the "
                    "clip's fps and total_num_frames (a path or URL is decoded with its own metadata)"
                )
        # the shared sampler writes a 24 fps default into metadata lacking a clock before the expansion runs
        elif any(metadata.fps is None for metadata in make_batched_metadata(videos, video_metadata)):
            raise ValueError(
                "LFM3-Omni prompts carry frame timestamps: `video_metadata` must hold the clip's fps "
                "(a path or URL is decoded with its own metadata)"
            )

    def _process_videos(self, videos, **kwargs):
        # only media-only calls reach the shared path (prompts with videos take `_call_with_videos`): the pairing
        # flag is not a video-processor argument and no placeholder text is needed
        kwargs.pop("use_audio_in_video", None)
        self._check_video_clock(videos, kwargs.get("video_metadata"))
        return self.video_processor(videos, **kwargs), []

    def replace_image_token(self, image_inputs, image_idx, **kwargs):
        count = image_inputs["image_grid_thw"][image_idx].prod() // self.image_processor.merge_size**2
        return self._wrap(self.image_token * int(count), self.vision_markers)

    @staticmethod
    def _calculate_timestamps(indices, video_fps: float, temporal_patch_size: int) -> list[float]:
        indices = list(indices)
        if pad := -len(indices) % temporal_patch_size:
            indices.extend([indices[-1]] * pad)
        return [
            (indices[i] + indices[i + temporal_patch_size - 1]) / (2 * video_fps)
            for i in range(0, len(indices), temporal_patch_size)
        ]


__all__ = ["Lfm3OmniProcessor"]
