# SPDX-FileCopyrightText: Microsoft Corporation
# SPDX-License-Identifier: MIT
# Adapted from Microsoft Phi-4-multimodal-instruct; see NOTICE for the source revision.
import json
import math
import re
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput, make_list_of_images, valid_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, TextInput, TruncationStrategy
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN = "<\\|image_\\d+\\|>"
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN = "<\\|audio_\\d+\\|>"
_IMAGE_SPECIAL_TOKEN = "<|endoftext10|>"
_AUDIO_SPECIAL_TOKEN = "<|endoftext11|>"
_IMAGE_SPECIAL_TOKEN_ID = 200010
_AUDIO_SPECIAL_TOKEN_ID = 200011


class InputMode(Enum):
    LANGUAGE = 0
    VISION = 1
    SPEECH = 2
    VISION_SPEECH = 3


class Phi4MMImageProcessor(BaseImageProcessor):
    model_input_names = ["input_image_embeds", "image_sizes", "image_attention_mask"]

    def __init__(self, dynamic_hd, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dynamic_hd = dynamic_hd

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
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

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=384, mask_size=27, use_thumbnail=True):
        orig_width, orig_height = image.size
        w_crop_num = math.ceil(orig_width / float(image_size))
        h_crop_num = math.ceil(orig_height / float(image_size))
        if w_crop_num * h_crop_num > max_num:
            aspect_ratio = orig_width / orig_height
            target_ratios = {
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            }
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            target_aspect_ratio = self.find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size
            )
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
        else:
            target_width = image_size * w_crop_num
            target_height = image_size * h_crop_num
            target_aspect_ratio = (w_crop_num, h_crop_num)
        ratio_width = target_width / orig_width
        ratio_height = target_height / orig_height
        if ratio_width < ratio_height:
            new_size = (target_width, int(orig_height * ratio_width))
            padding_width = 0
            padding_height = target_height - int(orig_height * ratio_width)
        else:
            new_size = (int(orig_width * ratio_height), target_height)
            padding_width = target_width - int(orig_width * ratio_height)
            padding_height = 0
        attention_mask = torch.ones((int(mask_size * target_aspect_ratio[1]), int(mask_size * target_aspect_ratio[0])))
        if padding_width >= 14:
            attention_mask[:, -math.floor(padding_width / 14) :] = 0
        if padding_height >= 14:
            attention_mask[-math.floor(padding_height / 14) :, :] = 0
        assert attention_mask.sum() > 0
        if min(new_size[1], target_height) < 10 or min(new_size[0], target_width) < 10:
            raise ValueError(f"the aspect ratio is very extreme {new_size}")
        image = torchvision.transforms.functional.resize(image, [new_size[1], new_size[0]])
        resized_img = torchvision.transforms.functional.pad(
            image, [0, 0, padding_width, padding_height], fill=[255, 255, 255]
        )
        return (resized_img, attention_mask)

    def pad_to_max_num_crops(self, images, max_crops=5):
        B, _, H, W = images.shape
        if B < max_crops:
            pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
            images = torch.cat([images, pad], dim=0)
        return images

    def pad_mask_to_max_num_crops(self, masks, max_crops=5):
        B, H, W = masks.shape
        if B < max_crops:
            pad = torch.ones(max_crops - B, H, W, dtype=masks.dtype, device=masks.device)
            masks = torch.cat([masks, pad], dim=0)
        return masks

    def preprocess(self, images: ImageInput, return_tensors: Optional[Union[str, TensorType]] = None):
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray."
            )
        img_processor = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        dyhd_base_resolution = 448
        base_resolution = dyhd_base_resolution
        images = [image.convert("RGB") for image in images]
        mask_resolution = base_resolution // 14
        elems, image_attention_masks = ([], [])
        for im in images:
            elem, attention_mask = self.dynamic_preprocess(
                im, max_num=self.dynamic_hd, image_size=base_resolution, mask_size=mask_resolution
            )
            elems.append(elem)
            image_attention_masks.append(attention_mask)
        hd_images = [img_processor(im) for im in elems]
        global_image = [
            torch.nn.functional.interpolate(
                im.unsqueeze(0).float(), size=(base_resolution, base_resolution), mode="bicubic"
            ).to(im.dtype)
            for im in hd_images
        ]
        shapes = [[im.size(1), im.size(2)] for im in hd_images]
        mask_shapes = [[mask.size(0), mask.size(1)] for mask in image_attention_masks]
        global_attention_mask = [torch.ones((1, mask_resolution, mask_resolution)) for _ in hd_images]
        hd_images_reshape = [
            im.reshape(1, 3, h // base_resolution, base_resolution, w // base_resolution, base_resolution)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, 3, base_resolution, base_resolution)
            .contiguous()
            for im, (h, w) in zip(hd_images, shapes)
        ]
        attention_masks_reshape = [
            mask.reshape(1, h // mask_resolution, mask_resolution, w // mask_resolution, mask_resolution)
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, mask_resolution, mask_resolution)
            .contiguous()
            for mask, (h, w) in zip(image_attention_masks, mask_shapes)
        ]
        downsample_attention_masks = [
            mask[:, 0::2, 0::2]
            .reshape(
                1,
                h // mask_resolution,
                w // mask_resolution,
                mask_resolution // 2 + mask_resolution % 2,
                mask_resolution // 2 + mask_resolution % 2,
            )
            .permute(0, 1, 3, 2, 4)
            for mask, (h, w) in zip(attention_masks_reshape, mask_shapes)
        ]
        downsample_attention_masks = [
            mask.reshape(mask.size(1) * mask.size(2), mask.size(3) * mask.size(4))
            for mask in downsample_attention_masks
        ]
        num_img_tokens = [
            256 + 1 + int(mask.sum().item()) + int(mask[:, 0].sum().item()) + 16 for mask in downsample_attention_masks
        ]
        hd_images_reshape = [
            torch.cat([_global_image] + [_im], dim=0) for _global_image, _im in zip(global_image, hd_images_reshape)
        ]
        hd_masks_reshape = [
            torch.cat([_global_mask] + [_mask], dim=0)
            for _global_mask, _mask in zip(global_attention_mask, attention_masks_reshape)
        ]
        max_crops = max([img.size(0) for img in hd_images_reshape])
        image_transformed = [self.pad_to_max_num_crops(im, max_crops) for im in hd_images_reshape]
        image_transformed = torch.stack(image_transformed, dim=0)
        mask_transformed = [self.pad_mask_to_max_num_crops(mask, max_crops) for mask in hd_masks_reshape]
        mask_transformed = torch.stack(mask_transformed, dim=0)
        returned_input_image_embeds = image_transformed
        returned_image_sizes = torch.tensor(shapes, dtype=torch.long)
        returned_image_attention_mask = mask_transformed
        returned_num_img_tokens = num_img_tokens
        data = {
            "input_image_embeds": returned_input_image_embeds,
            "image_sizes": returned_image_sizes,
            "image_attention_mask": returned_image_attention_mask,
            "num_img_tokens": returned_num_img_tokens,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


AudioInput = Tuple[Union[np.ndarray, torch.Tensor], int]
AudioInputs = List[AudioInput]


def speechlib_mel(sample_rate, n_fft, n_mels, fmin=None, fmax=None):
    bank_width = int(n_fft // 2 + 1)
    if fmax is None:
        fmax = sample_rate / 2
    if fmin is None:
        fmin = 0
    assert fmin >= 0, "fmin cannot be negtive"
    assert fmin < fmax <= sample_rate / 2, "fmax must be between (fmin, samplerate / 2]"

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin):
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    def f2bin(f):
        return int(f * n_fft / sample_rate + 0.5)

    klo = f2bin(fmin) + 1
    khi = f2bin(fmax)
    khi = max(khi, klo)
    mlo = mel(fmin)
    mhi = mel(fmax)
    m_centers = np.linspace(mlo, mhi, n_mels + 2)
    ms = (mhi - mlo) / (n_mels + 1)
    matrix = np.zeros((n_mels, bank_width), dtype=np.float32)
    for m in range(0, n_mels):
        left = m_centers[m]
        center = m_centers[m + 1]
        right = m_centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / ms
    return matrix


class Phi4MMAudioFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_audio_embeds", "audio_embed_sizes", "audio_attention_mask"]

    def __init__(self, audio_compression_rate, audio_downsample_rate, audio_feat_stride, **kwargs):
        feature_size = 80
        sampling_rate = 16000
        padding_value = 0.0
        super().__init__(feature_size, sampling_rate, padding_value, **kwargs)
        self.compression_rate = audio_compression_rate
        self.qformer_compression_rate = audio_downsample_rate
        self.feat_stride = audio_feat_stride
        self._eightk_method = "fillzero"
        self._mel = speechlib_mel(16000, 512, 80, fmin=None, fmax=7690).T
        self._hamming400 = np.hamming(400)
        self._hamming200 = np.hamming(200)

    def __call__(self, audios: List[AudioInput], return_tensors: Optional[Union[str, TensorType]] = None):
        returned_input_audio_embeds = []
        returned_audio_embed_sizes = []
        audio_frames_list = []
        for audio_data, sample_rate in audios:
            audio_embeds = self._extract_features(audio_data, sample_rate)
            audio_frames = len(audio_embeds) * self.feat_stride
            audio_embed_size = self._compute_audio_embed_size(audio_frames)
            returned_input_audio_embeds.append(torch.tensor(audio_embeds))
            returned_audio_embed_sizes.append(torch.tensor(audio_embed_size).long())
            audio_frames_list.append(audio_frames)
        returned_input_audio_embeds = pad_sequence(returned_input_audio_embeds, batch_first=True)
        returned_audio_embed_sizes = torch.stack(returned_audio_embed_sizes, dim=0)
        audio_frames = torch.tensor(audio_frames_list)
        returned_audio_attention_mask = (
            torch.arange(0, audio_frames.max()).unsqueeze(0) < audio_frames.unsqueeze(1) if len(audios) > 1 else None
        )
        data = {"input_audio_embeds": returned_input_audio_embeds, "audio_embed_sizes": returned_audio_embed_sizes}
        if returned_audio_attention_mask is not None:
            data["audio_attention_mask"] = returned_audio_attention_mask
        return BatchFeature(data=data, tensor_type=return_tensors)

    def _extract_spectrogram(self, wav, fs):
        if fs not in (8000, 16000) or (fs == 8000 and self._eightk_method == "resample"):
            import scipy.signal
        if wav.ndim > 1:
            wav = np.squeeze(wav)
        if len(wav.shape) == 2:
            wav = wav.mean(1)
        if fs > 16000:
            wav = scipy.signal.resample_poly(wav, 1, fs // 16000)
            fs = 16000
        elif 8000 < fs < 16000:
            wav = scipy.signal.resample_poly(wav, 1, fs // 8000)
            fs = 8000
        elif fs < 8000:
            raise RuntimeError(f"Unsupported sample rate {fs}")
        if fs == 8000:
            if self._eightk_method == "resample":
                wav = scipy.signal.resample_poly(wav, 2, 1)
                fs = 16000
        elif fs != 16000:
            raise RuntimeError(f"Input data using an unsupported sample rate: {fs}")
        preemphasis = 0.97
        if fs == 8000:
            n_fft = 256
            win_length = 200
            hop_length = 80
            fft_window = self._hamming200
        elif fs == 16000:
            n_fft = 512
            win_length = 400
            hop_length = 160
            fft_window = self._hamming400
        n_batch = (wav.shape[0] - win_length) // hop_length + 1
        y_frames = np.array(
            [wav[_stride : _stride + win_length] for _stride in range(0, hop_length * n_batch, hop_length)],
            dtype=np.float32,
        )
        y_frames_prev = np.roll(y_frames, 1, axis=1)
        y_frames_prev[:, 0] = y_frames_prev[:, 1]
        y_frames = (y_frames - preemphasis * y_frames_prev) * 32768
        S = np.fft.rfft(fft_window * y_frames, n=n_fft, axis=1).astype(np.complex64)
        if fs == 8000:
            frames, bins = S.shape
            padarray = np.zeros((frames, bins))
            S = np.concatenate((S[:, 0:-1], padarray), axis=1)
        spec = np.abs(S).astype(np.float32)
        return spec

    def _extract_features(self, wav, fs):
        spec = self._extract_spectrogram(wav, fs)
        spec_power = spec**2
        fbank_power = np.clip(spec_power.dot(self._mel), 1.0, None)
        log_fbank = np.log(fbank_power).astype(np.float32)
        return log_fbank

    def _compute_audio_embed_size(self, audio_frames):
        integer = audio_frames // self.compression_rate
        remainder = audio_frames % self.compression_rate
        result = integer if remainder == 0 else integer + 1
        integer = result // self.qformer_compression_rate
        remainder = result % self.qformer_compression_rate
        result = integer if remainder == 0 else integer + 1
        return result


class Phi4MMProcessor(ProcessorMixin):
    special_image_token = "<|endoftext10|>"
    special_audio_token = "<|endoftext11|>"
    attributes = ["image_processor", "audio_processor", "tokenizer"]
    tokenizer_class = "GPT2TokenizerFast"
    image_processor_class = "AutoImageProcessor"
    audio_processor_class = "AutoFeatureExtractor"

    def __init__(self, image_processor, audio_processor, tokenizer):
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_id_or_path, tokenizer=None, config=None, **kwargs):
        from transformers import TokenizersBackend
        from transformers.utils.hub import cached_file

        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        asset_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder")
            if key in kwargs
        }
        path = cached_file(model_id_or_path, "preprocessor_config.json", **asset_kwargs)
        with open(path, encoding="utf-8") as handle:
            settings = json.load(handle)
        if tokenizer is None:
            tokenizer = TokenizersBackend.from_pretrained(model_id_or_path, **asset_kwargs)
        return cls(
            Phi4MMImageProcessor(dynamic_hd=settings.get("dynamic_hd", 36)),
            Phi4MMAudioFeatureExtractor(
                audio_compression_rate=settings.get("audio_compression_rate", 8),
                audio_downsample_rate=settings.get("audio_downsample_rate", 1),
                audio_feat_stride=settings.get("audio_feat_stride", 1),
            ),
            tokenizer,
        )

    def save_pretrained(self, save_directory, **kwargs):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        settings = {
            "processor_class": "Phi4MMProcessor",
            "dynamic_hd": self.image_processor.dynamic_hd,
            "audio_compression_rate": self.audio_processor.compression_rate,
            "audio_downsample_rate": self.audio_processor.qformer_compression_rate,
            "audio_feat_stride": self.audio_processor.feat_stride,
        }
        (path / "preprocessor_config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        self.tokenizer.save_pretrained(path, **kwargs)
        return [str(path / "preprocessor_config.json")]

    def __call__(
        self,
        text: Union[TextInput, List[TextInput]],
        images: Optional[ImageInput] = None,
        audios: Optional[AudioInputs] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        image_inputs = self.image_processor(images, return_tensors=return_tensors) if images is not None else {}
        audio_inputs = self.audio_processor(audios, return_tensors=return_tensors) if audios is not None else {}
        inputs = self._convert_images_audios_text_to_inputs(
            image_inputs,
            audio_inputs,
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        if len(image_inputs) > 0 and len(audio_inputs) > 0:
            input_mode = InputMode.VISION_SPEECH
        elif len(image_inputs) > 0:
            input_mode = InputMode.VISION
        elif len(audio_inputs) > 0:
            input_mode = InputMode.SPEECH
        else:
            input_mode = InputMode.LANGUAGE
        inputs["input_mode"] = torch.tensor([input_mode.value], dtype=torch.long)
        return inputs

    @property
    def special_image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.special_image_token)

    @property
    def chat_template(self):
        return self.tokenizer.chat_template

    def _convert_images_audios_text_to_inputs(
        self, images, audios, text, padding=False, truncation=None, max_length=None, return_tensors=None
    ):
        if len(images) > 0:
            input_image_embeds = images["input_image_embeds"]
            image_sizes = images["image_sizes"]
            image_attention_mask = images["image_attention_mask"]
            num_img_tokens = images["num_img_tokens"]
        else:
            input_image_embeds = torch.tensor([])
            image_sizes = torch.tensor([])
            image_attention_mask = torch.tensor([])
            num_img_tokens = []
        if len(audios) > 0:
            input_audio_embeds = audios["input_audio_embeds"]
            audio_embed_sizes = audios["audio_embed_sizes"]
            audio_attention_mask = audios.get("audio_attention_mask", None)
        else:
            input_audio_embeds = torch.tensor([])
            audio_embed_sizes = torch.tensor([])
            audio_attention_mask = None
        if isinstance(text, str):
            text = [text]
        assert isinstance(text, list)
        processed_text = [re.sub(_COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN, _IMAGE_SPECIAL_TOKEN, t) for t in text]
        processed_text = [
            re.sub(_COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN, _AUDIO_SPECIAL_TOKEN, t) for t in processed_text
        ]
        input_ids_list = [self.tokenizer(t).input_ids for t in processed_text]
        img_cnt, audio_cnt = (0, 0)
        image_token_count_iter = iter(num_img_tokens)
        audio_embed_size_iter = iter(audio_embed_sizes.tolist())
        new_input_ids_list = []
        for input_ids in input_ids_list:
            i = 0
            while i < len(input_ids):
                token_id = input_ids[i]
                if token_id == _AUDIO_SPECIAL_TOKEN_ID:
                    token_count = next(audio_embed_size_iter)
                    audio_cnt += 1
                elif token_id == _IMAGE_SPECIAL_TOKEN_ID:
                    token_count = next(image_token_count_iter)
                    img_cnt += 1
                else:
                    i += 1
                    continue
                tokens = [token_id] * token_count
                input_ids = input_ids[:i] + tokens + input_ids[i + 1 :]
                i += token_count
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            new_input_ids_list.append(input_ids)
        lengths = torch.tensor([len(input_ids) for input_ids in new_input_ids_list])
        max_len = lengths.max()
        input_ids = input_ids.new_full((len(new_input_ids_list), max_len), self.tokenizer.pad_token_id)
        for i in range(len(new_input_ids_list)):
            input_ids[i, max_len - len(new_input_ids_list[i]) :] = new_input_ids_list[i]
        assert img_cnt == len(num_img_tokens), (
            f"Number of image tokens in prompt_token_ids ({img_cnt}) does not match number of images ({len(num_img_tokens)})"
        )
        assert audio_cnt == len(audio_embed_sizes), (
            f"Number of audio tokens in prompt_token_ids ({audio_cnt}) does not match number of audios ({len(audio_embed_sizes)})"
        )
        seq_range = torch.arange(max_len - 1, -1, -1)
        attention_mask = seq_range.unsqueeze(0) < lengths.unsqueeze(1)
        data = {
            "input_ids": input_ids,
            "input_image_embeds": input_image_embeds,
            "image_sizes": image_sizes,
            "image_attention_mask": image_attention_mask,
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_attention_mask": audio_attention_mask,
            "attention_mask": attention_mask,
        }
        return BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + audio_processor_input_names))
