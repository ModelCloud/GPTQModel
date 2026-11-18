#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2026 The OpenBMB Team. All rights reserved.
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

import copy
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers.audio_utils import spectrogram, window_function
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    infer_channel_dimension_format,
    is_torch_tensor,
    to_numpy_array,
    valid_images,
)
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType, is_torch_device, is_torch_dtype, requires_backends


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)


class MiniCPMOBatchFeature(BatchFeature):
    """Extend from BatchFeature for supporting various image size"""

    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        if tensor_type is None:
            return self

        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)

        def converter(value):
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    return tensor
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        for key, value in self.items():
            self[key] = recursive_converter(converter, value)
        return self

    def to(self, *args, **kwargs) -> "MiniCPMOBatchFeature":
        requires_backends(self, ["torch"])
        import torch

        def cast_tensor(v):
            if not torch.is_tensor(v):
                return v

            if torch.is_floating_point(v):
                return v.to(*args, **kwargs)
            elif device is not None:
                return v.to(device=device)
            else:
                return v

        new_data = {}
        device = kwargs.get("device")
        if device is None and len(args) > 0:
            arg = args[0]
            if is_torch_dtype(arg):
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")

        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            new_data[k] = recursive_converter(cast_tensor, v)
        self.data = new_data
        return self


class MiniCPMVImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(self, max_slice_nums=9, scale_resolution=448, patch_size=14, **kwargs):
        super().__init__(**kwargs)
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
        self.patch_size = patch_size
        self.use_image_id = kwargs.pop("use_image_id", False)
        self.image_feature_size = kwargs.pop("image_feature_size", 64)
        self.im_start_token = kwargs.pop("im_start", "<image>")
        self.im_end_token = kwargs.pop("im_end", "</image>")
        self.slice_start_token = kwargs.pop("slice_start", "<slice>")
        self.slice_end_token = kwargs.pop("slice_end", "</slice>")
        self.unk_token = kwargs.pop("unk", "<unk>")
        self.im_id_start = kwargs.pop("im_id_start", "<image_id>")
        self.im_id_end = kwargs.pop("im_id_end", "</image_id>")
        self.slice_mode = kwargs.pop("slice_mode", True)

        self.mean = np.array(kwargs.pop("norm_mean", [0.5, 0.5, 0.5]))
        self.std = np.array(kwargs.pop("norm_std", [0.5, 0.5, 0.5]))
        self.version = kwargs.pop("version", 2.0)

    @staticmethod
    def ensure_divide(length, patch_size):
        return max(round(length / patch_size) * patch_size, patch_size)

    def find_best_resize(self, original_size, scale_resolution, patch_size, allow_upscale=False):
        width, height = original_size
        if (width * height > scale_resolution * scale_resolution) or allow_upscale:
            r = width / height
            height = int(scale_resolution / math.sqrt(r))
            width = int(height * r)
        best_width = self.ensure_divide(width, patch_size)
        best_height = self.ensure_divide(height, patch_size)
        return best_width, best_height

    def get_refine_size(self, original_size, grid, scale_resolution, patch_size, allow_upscale=False):
        width, height = original_size
        grid_x, grid_y = grid

        refine_width = self.ensure_divide(width, grid_x)
        refine_height = self.ensure_divide(height, grid_y)

        grid_width = refine_width / grid_x
        grid_height = refine_height / grid_y

        best_grid_size = self.find_best_resize(
            (grid_width, grid_height), scale_resolution, patch_size, allow_upscale=allow_upscale
        )
        refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)
        return refine_size

    @staticmethod
    def split_to_patches(image, grid):
        patches = []
        width, height = image.size
        grid_x = int(width / grid[0])
        grid_y = int(height / grid[1])
        for i in range(0, height, grid_y):
            images = []
            for j in range(0, width, grid_x):
                box = (j, i, j + grid_x, i + grid_y)
                patch = image.crop(box)
                images.append(patch)
            patches.append(images)
        return patches

    def slice_image(self, image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False):
        original_size = image.size
        source_image = None
        best_grid = self.get_sliced_grid(original_size, max_slice_nums, never_split)
        patches = []

        if best_grid is None:
            # dont need to slice, upsample
            best_size = self.find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=True)
            source_image = image.resize(best_size, resample=Image.Resampling.BICUBIC)
        else:
            # source image, down-sampling and ensure divided by patch_size
            best_resize = self.find_best_resize(original_size, scale_resolution, patch_size)
            source_image = image.copy().resize(best_resize, resample=Image.Resampling.BICUBIC)
            refine_size = self.get_refine_size(
                original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
            )
            refine_image = image.resize(refine_size, resample=Image.Resampling.BICUBIC)
            patches = self.split_to_patches(refine_image, best_grid)

        return source_image, patches, best_grid

    def get_grid_placeholder(self, grid):
        if grid is None:
            return ""
        slice_image_placeholder = (
            self.slice_start_token + self.unk_token * self.image_feature_size + self.slice_end_token
        )

        cols = grid[0]
        rows = grid[1]
        slices = []
        for i in range(rows):
            lines = []
            for j in range(cols):
                lines.append(slice_image_placeholder)
            slices.append("".join(lines))

        slice_placeholder = "\n".join(slices)
        return slice_placeholder

    def get_image_id_placeholder(self, idx=0):
        return f"{self.im_id_start}{idx}{self.im_id_end}"

    def get_sliced_images(self, image, max_slice_nums=None):
        slice_images = []

        if not self.slice_mode:
            return [image]

        max_slice_nums = self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        assert max_slice_nums > 0
        source_image, patches, sliced_grid = self.slice_image(
            image,
            max_slice_nums,
            self.scale_resolution,
            self.patch_size,  # default: 9  # default: 448  # default: 14
        )

        slice_images.append(source_image)
        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    slice_images.append(patches[i][j])
        return slice_images

    def get_sliced_grid(self, image_size, max_slice_nums, nerver_split=False):
        original_width, original_height = image_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (self.scale_resolution * self.scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1 or nerver_split:
            return None
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        candidate_grids = []
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        return best_grid

    def get_slice_image_placeholder(self, image_size, image_idx=0, max_slice_nums=None, use_image_id=None):
        max_slice_nums = self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        assert max_slice_nums > 0
        grid = self.get_sliced_grid(image_size=image_size, max_slice_nums=max_slice_nums)

        image_placeholder = self.im_start_token + self.unk_token * self.image_feature_size + self.im_end_token
        use_image_id = self.use_image_id if use_image_id is None else bool(use_image_id)
        if use_image_id:
            final_placeholder = self.get_image_id_placeholder(image_idx) + image_placeholder
        else:
            final_placeholder = image_placeholder

        if self.slice_mode:
            final_placeholder = final_placeholder + self.get_grid_placeholder(grid=grid)
        return final_placeholder

    @staticmethod
    def to_pil_image(image, rescale=None) -> Image.Image:
        """Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back
        as the last axis if needed.

        Args:
            image (`Image.Image` or `numpy.ndarray` or `torch.Tensor`):
                The image to convert to the PIL Image format.
            rescale (`bool`, *optional*):
                whether to apply the scaling factor (to make pixel values integers between 0 and 255). Will
                default to `True` if the image type is a floating type, `False` otherwise.
        """
        if isinstance(image, Image.Image):
            return image
        if is_torch_tensor(image):
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if rescale is None:
                # rescale default to the array being of floating type.
                rescale = isinstance(image.flat[0], np.floating)
            # If the channel as been moved to first dim, we put it back at the end.
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if rescale:
                image = image * 255
            image = image.astype(np.uint8)
            return Image.fromarray(image)
        return image

    def reshape_by_patch(self, image):
        image = torch.from_numpy(image)
        patch_size = self.patch_size
        patches = torch.nn.functional.unfold(image, (patch_size, patch_size), stride=(patch_size, patch_size))

        patches = patches.reshape(image.size(0), patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(image.size(0), patch_size, -1)
        return patches.numpy()

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]],
        do_pad: Optional[bool] = True,
        max_slice_nums: int = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> MiniCPMOBatchFeature:
        if isinstance(images, Image.Image):
            images_list = [[images]]
        elif isinstance(images[0], Image.Image):
            images_list = [images]
        else:
            images_list = images

        new_images_list = []
        image_sizes_list = []
        tgt_sizes_list = []

        for _images in images_list:
            if _images is None or len(_images) == 0:
                new_images_list.append([])
                image_sizes_list.append([])
                tgt_sizes_list.append([])
                continue
            if not valid_images(_images):
                raise ValueError(
                    "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                    "torch.Tensor, tf.Tensor or jax.ndarray."
                )

            _images = [self.to_pil_image(image).convert("RGB") for image in _images]
            input_data_format = infer_channel_dimension_format(np.array(_images[0]))

            new_images = []
            image_sizes = [image.size for image in _images]
            tgt_sizes = []
            for image in _images:
                image_patches = self.get_sliced_images(image, max_slice_nums)
                image_patches = [to_numpy_array(image).astype(np.float32) / 255 for image in image_patches]
                image_patches = [
                    self.normalize(image=image, mean=self.mean, std=self.std, input_data_format=input_data_format)
                    for image in image_patches
                ]
                image_patches = [
                    to_channel_dimension_format(image, ChannelDimension.FIRST, input_channel_dim=input_data_format)
                    for image in image_patches
                ]
                for slice_image in image_patches:
                    new_images.append(self.reshape_by_patch(slice_image))
                    tgt_sizes.append(
                        np.array((slice_image.shape[1] // self.patch_size, slice_image.shape[2] // self.patch_size))
                    )

            if tgt_sizes:
                tgt_sizes = np.vstack(tgt_sizes)

            new_images_list.append(new_images)
            image_sizes_list.append(image_sizes)
            tgt_sizes_list.append(tgt_sizes)
        return MiniCPMOBatchFeature(
            data={"pixel_values": new_images_list, "image_sizes": image_sizes_list, "tgt_sizes": tgt_sizes_list},
            tensor_type=return_tensors,
        )


def chunk_audio(audio: np.ndarray, max_duration_seconds: int = 30, sample_rate: int = 16000) -> List[np.ndarray]:
    """split long audio into chunks

    Args:
        audio:
        max_duration_seconds:
        sample_rate:

    Returns:
        chunks
    """
    max_len = int(max_duration_seconds * sample_rate)

    if len(audio) <= max_len:
        return [audio]

    chunks = []
    for i in range(0, len(audio), max_len):
        chunk = audio[i : i + max_len]
        chunks.append(chunk)

    return chunks


def process_audio_batch(
    audios: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]],
    feature_extractor,
    sampling_rate: int = 16000,
    max_duration_seconds: int = 30,
    return_attention_mask: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """extract audio mel features

    Args:
        audios:
        feature_extractor: WhisperFeatureExtractor
        sampling_rate:
        max_duration_seconds:
        return_attention_mask:

    Returns:
        (audio_features, audio_feature_lens)
        audio_features: [batch_size, n_mels, max_frames]
        audio_feature_lens:
    """
    if isinstance(audios, np.ndarray):
        audios_list = [[audios]]
    elif len(audios) > 0 and isinstance(audios[0], np.ndarray):
        audios_list = [audios]
    else:
        audios_list = audios

    audio_features_all = []
    audio_feature_lens_list = []

    for batch_audios in audios_list:
        batch_lens = []

        for audio in batch_audios:
            chunks = chunk_audio(audio, max_duration_seconds, sampling_rate)

            for chunk in chunks:
                audio_input = feature_extractor(
                    chunk,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    padding="max_length",
                    return_attention_mask=return_attention_mask,
                )

                audio_feature = audio_input["input_features"]  # [1, 80, frames]

                if return_attention_mask:
                    actual_len = audio_input["attention_mask"].sum(dim=1)  # Tensor([frames])
                    audio_feature = audio_feature[:, :, : actual_len[0]]
                    batch_lens.append(actual_len[0])
                else:
                    batch_lens.append(torch.tensor(audio_feature.shape[2]))

                audio_features_all.append(audio_feature.squeeze(0))  # [80, frames]

        if len(batch_lens) > 0:
            audio_feature_lens_list.append(torch.hstack(batch_lens))
        else:
            audio_feature_lens_list.append(torch.tensor([]))

    # pad to same length
    if audio_features_all:
        audio_features = torch.nn.utils.rnn.pad_sequence(
            [feat.transpose(0, 1) for feat in audio_features_all], batch_first=True, padding_value=0.0
        ).transpose(1, 2)  # [batch, 80, max_frames]
    else:
        audio_features = torch.tensor([])

    return audio_features, audio_feature_lens_list


def regroup_audio_features(
    audio_features: torch.Tensor, audio_feature_lens: List[torch.Tensor], regroup_seconds: int, fps: int = 100
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """regroup audio features to fixed duration

    Args:
        audio_features: [batch, n_mels, frames]
        audio_feature_lens: each batch's actual length
        regroup_seconds: regroup duration (seconds)
        fps: frames per second

    Returns:
        (regrouped_features, regrouped_lens)
    """
    # flatten to continuous frames sequence
    all_lens = []
    for lens in audio_feature_lens:
        if isinstance(lens, torch.Tensor):
            all_lens.extend(lens.tolist())
        elif isinstance(lens, list):
            all_lens.extend([int(x) for x in lens])

    if len(all_lens) == 0:
        return torch.tensor([]), []

    # concatenate all valid features
    flat_slices = [audio_features[i, :, :L] for i, L in enumerate(all_lens)]  # [n_mels, L]

    if len(flat_slices) == 1:
        full_feat = flat_slices[0]
    else:
        full_feat = torch.cat(flat_slices, dim=1)  # [n_mels, total_frames]

    # split to fixed frames
    frames_per_seg = int(regroup_seconds * fps)
    segments = []

    for start in range(0, full_feat.size(1), frames_per_seg):
        seg = full_feat[:, start : start + frames_per_seg]
        if seg.size(1) > 0:
            segments.append(seg)

    if len(segments) == 0:
        return torch.tensor([]), []

    # pad and convert to batch
    seg_lens = [s.size(1) for s in segments]
    segs_transposed = [s.transpose(0, 1) for s in segments]

    padded = torch.nn.utils.rnn.pad_sequence(
        segs_transposed, batch_first=True, padding_value=0.0
    )  # [N, max_T, n_mels]

    padded = padded.transpose(1, 2)  # [N, n_mels, max_T]
    lens_tensor = torch.tensor(seg_lens, dtype=torch.int32, device=padded.device)

    return padded, [lens_tensor]


class MiniCPMAAudioProcessor(WhisperFeatureExtractor):
    """
    On top of WhisperFeatureExtractor:
    - support dynamic_log_norm (original max-8dB, adjustable dynamic_range_db)
    - or fixed log_floor_db (e.g. -10dB)
        - this is because we need to do streaming scheme, in which we can't do dynamic setting
        - this can be modified in the middle, through set_dynamic_log_norm
    Two paths (torch / numpy) keep consistent clipping and scaling order:
        log10 -> (dynamic/fixed lower limit clipping) -> (+4)/4
    """

    def __init__(
        self,
        *args,
        dynamic_log_norm: bool = True,
        dynamic_range_db: float = 8.0,
        log_floor_db: float = -10.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dynamic_log_norm = bool(dynamic_log_norm)
        self.dynamic_range_db = float(dynamic_range_db)
        self.log_floor_db = float(log_floor_db)

    def set_spac_log_norm(
        self,
        dynamic_range_db: Optional[float] = None,
        log_floor_db: Optional[float] = None,
        *,
        inplace: bool = True,
    ) -> "MiniCPMAAudioProcessor":
        """Hot update dynamic/fixed lower limit strategy.

        Args:
            enabled: True=use dynamic threshold (max - dynamic_range_db), False=use fixed lower limit log_floor_db.
                    None means keep unchanged.
            dynamic_range_db: dynamic range (dB), only effective when enabled=True. None means keep unchanged.
            log_floor_db: fixed log floor (dB, usually <= 0), only effective when enabled=False. None means keep unchanged.
            inplace: True directly modify current instance; False return a shallow copy and modify on it.

        Returns:
            self or new instance (when inplace=False).
        """

        target = self if inplace else copy.copy(self)

        if dynamic_range_db is not None:
            val = float(dynamic_range_db)
            if val < 0:
                raise ValueError("dynamic_range_db must be >= 0.")
            target.dynamic_log_norm = True  # explicitly set the value to dynamic mode
            target.dynamic_range_db = val

        if log_floor_db is not None:
            val = float(log_floor_db)
            # usually log10(mel) maximum is not more than ~0dB, floor should be <= 0; here do loose validation
            if val > 0:
                raise ValueError("log_floor_db should be <= 0 (log10 scale).")
            target.dynamic_log_norm = False  # explicitly set the value to fixed lower limit mode
            target.log_floor_db = val

        return target

    def _np_extract_fbank_features(self, waveform_batch: np.ndarray, device: str) -> np.ndarray:
        """NumPy version consistent with upstream, but replace max-8dB with configurable dynamic/fixed lower limit clipping."""
        if device != "cpu":
            raise ValueError(
                f"Got device `{device}` for feature extraction, but feature extraction on CUDA accelerator "
                "devices requires torch. Set device='cpu' or install torch."
            )

        log_spec_batch: List[np.ndarray] = []
        for waveform in waveform_batch:
            # generate log10 Mel
            log_spec = spectrogram(
                waveform,
                window_function(self.n_fft, "hann"),
                frame_length=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,
                dither=self.dither,
                mel_filters=self.mel_filters,
                log_mel="log10",
            )
            # consistent with upstream: remove the last frame
            log_spec = log_spec[:, :-1]

            # dynamic/fixed clipping
            if self.dynamic_log_norm:
                threshold = log_spec.max() - self.dynamic_range_db
                log_spec = np.maximum(log_spec, threshold)
            else:
                log_spec = np.maximum(log_spec, self.log_floor_db)

            # consistent with Whisper linear scaling
            log_spec = (log_spec + 4.0) / 4.0

            log_spec_batch.append(log_spec)

        return np.array(log_spec_batch)

    def _torch_extract_fbank_features(self, waveform: np.ndarray, device: str = "cpu") -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is not installed, cannot compute STFT on GPU.")

        waveform = torch.from_numpy(waveform).to(device, torch.float32)
        window = torch.hann_window(self.n_fft, device=device)

        if self.dither != 0.0:
            waveform = waveform + self.dither * torch.randn_like(waveform)

        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)  # [n_mels, 1+n_fft//2]
        mel_spec = mel_filters.T @ magnitudes  # [..., n_mels, T]

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()  # <= 0

        if self.dynamic_log_norm:
            if waveform.dim() == 2:
                max_val_t = log_spec.max(dim=2, keepdim=True)[0]  # over T
                max_val_bt = max_val_t.max(dim=1, keepdim=True)[0]  # over mel
                threshold = max_val_bt - self.dynamic_range_db
                log_spec = torch.maximum(log_spec, threshold)
            else:
                threshold = log_spec.max() - self.dynamic_range_db
                log_spec = torch.maximum(log_spec, threshold)
        else:
            floor_tensor = torch.tensor(self.log_floor_db, dtype=log_spec.dtype, device=log_spec.device)
            log_spec = torch.maximum(log_spec, floor_tensor)

        log_spec = (log_spec + 4.0) / 4.0

        if device != "cpu":
            log_spec = log_spec.detach().cpu()
        return log_spec.numpy()

    def process(self, *args, **kwargs):
        """Alias of __call__ for convenience."""
        return self.__call__(*args, **kwargs)


class StreamingMelProcessorExact:
    """Strictly offline equivalent streaming Mel processor.

    - accumulate all historical audio into buffer; use the same feature_extractor to calculate the entire mel after each addition.
    - only output "stable" frames: the frame center does not depend on future (right) context, i.e. center + n_fft//2 <= current buffer length.
    - output the last batch of frames at the end (flush), ensuring complete consistency with offline full-calculation.

    Cost: Each call performs feature extraction on the accumulated buffer (can be optimized to incremental if needed).
    """

    def __init__(
        self,
        feature_extractor: MiniCPMAAudioProcessor,
        chunk_ms: int = 100,
        first_chunk_ms: Optional[int] = None,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        cnn_redundancy_ms: int = 10,  # (given in ms, usually 10ms=1 frame)
        # sliding window parameters
        enable_sliding_window: bool = False,  # whether to enable sliding window
        slide_trigger_seconds: float = 30.0,  # trigger threshold for sliding window in seconds
        slide_stride_seconds: float = 10.0,  # stride for sliding window in seconds
    ):
        self.feature_extractor = feature_extractor
        self.chunk_ms = chunk_ms
        self.first_chunk_ms = first_chunk_ms if first_chunk_ms is not None else chunk_ms
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.chunk_samples = int(round(chunk_ms * sample_rate / 1000))
        self.chunk_frames = self.chunk_samples // hop_length
        # align to hop_length to avoid frame boundary issues
        hop = self.hop_length
        raw_first_samples = int(round(self.first_chunk_ms * sample_rate / 1000))
        aligned_first = max(hop, (raw_first_samples // hop) * hop)
        self.first_chunk_samples = aligned_first
        self.half_window = n_fft // 2  # required right context

        # redundancy frames (in frames), <=1 frame: 10ms → 1 frame
        self.cnn_redundancy_ms = cnn_redundancy_ms
        self.cnn_redundancy_samples = int(cnn_redundancy_ms * sample_rate / 1000)
        self.cnn_redundancy_frames = max(0, self.cnn_redundancy_samples // hop_length)

        # sliding window configuration (Trigger mode)
        self.enable_sliding_window = enable_sliding_window
        self.trigger_seconds = slide_trigger_seconds
        self.slide_seconds = slide_stride_seconds

        # shift/base (global frame coordinates)
        self.left_samples_dropped = 0  # samples dropped from the left
        self.base_T = 0  # index of the "global frame" corresponding to mel_full[:, :, 0]

        self.reset()

    def reset(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self.last_emitted_T = 0
        self.total_samples_processed = 0
        self.chunk_count = 0
        self.is_first = True
        self.left_samples_dropped = 0
        self.base_T = 0

    def get_chunk_size(self) -> int:
        return self.first_chunk_samples if self.is_first else self.chunk_samples

    def get_expected_output_frames(self) -> int:
        raise NotImplementedError("get_expected_output_frames is not implemented")

    def _extract_full(self) -> torch.Tensor:
        # when buffer length is less than n_fft, Whisper's internal STFT will raise an error in center=True and pad mode
        # (pad is greater than input length). At this time, there is no stable frame to output, so return empty features directly.
        if len(self.buffer) < self.n_fft:
            raise ValueError(f"buffer length is shorter than n_fft {len(self.buffer)} < {self.n_fft}")
        # if buffer length is less than 5s, use set_spac_log_norm(log_floor_db=-10) or the last cached result
        if len(self.buffer) < 5 * self.sample_rate:
            # TODO: here the best is to do some experiments to choose the best one, now this is selected through experience, can see MiniCPMAAudioProcessor's main implementation
            self.feature_extractor.set_spac_log_norm(log_floor_db=-10)
        # if buffer length is greater than 5s, use set_spac_log_norm(dynamic_range_db=8)
        else:
            self.feature_extractor.set_spac_log_norm(dynamic_range_db=8)
        feats = self.feature_extractor(
            self.buffer,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        return feats.input_features  # [1, 80, T]

    def _stable_frames_count(self) -> int:
        # number of stable frames = floor((len(buffer) - half_window) / hop) + 1, minimum is 0
        L = int(self.buffer.shape[0])
        if L <= 0:
            return 0
        if L < self.half_window:
            return 0
        return max(0, (L - self.half_window) // self.hop_length + 1)

    def _maybe_slide_buffer(self):
        """Trigger mode sliding window: when the buffer reaches the trigger threshold, slide a fixed length window."""
        if not self.enable_sliding_window:
            return

        sr = self.sample_rate
        hop = self.hop_length
        L = len(self.buffer)

        # convert seconds to samples
        trigger_samples = int(self.trigger_seconds * sr)
        stride_samples = int(self.slide_seconds * sr)

        # check if the trigger threshold is reached
        if L < trigger_samples:
            return

        # calculate the number of samples to drop (fixed sliding stride_samples)
        drop = stride_samples

        # cannot drop the left context that is still needed for subsequent emission
        # in trigger mode, we only need to protect the minimum necessary data
        # i.e. ensure that we do not discard frames that may be needed in the future
        self.last_emitted_T - self.base_T

        # only protect necessary context (e.g. the most recent 1 second data)
        min_keep_seconds = 1.0  # keep at least 1 second of data to ensure continuity
        min_keep_samples = int(min_keep_seconds * sr)

        # guard_samples are the minimum samples we must keep
        guard_samples = min(min_keep_samples, L - drop)

        # limit: do not exceed the safe boundary; and align hop
        max_allowed_drop = max(0, L - guard_samples)
        drop = min(drop, max_allowed_drop)
        drop = (drop // hop) * hop

        if drop <= 0:
            return

        # truly drop & update base
        self.buffer = self.buffer[drop:]
        self.left_samples_dropped += drop
        self.base_T += drop // hop

    def process(self, audio_chunk: np.ndarray, is_last_chunk: bool = False) -> Tuple[torch.Tensor, Dict]:
        self.chunk_count += 1
        # append to buffer
        if len(self.buffer) == 0:
            self.buffer = audio_chunk.astype(np.float32, copy=True)
        else:
            self.buffer = np.concatenate([self.buffer, audio_chunk.astype(np.float32, copy=True)])

        # sliding window processing
        self._maybe_slide_buffer()

        # full extraction (for the current window)
        mel_full = self._extract_full()
        T_full = mel_full.shape[-1]  # local frames in the current window
        stable_T = min(T_full, self._stable_frames_count())  # local stable frames
        stable_T_global = self.base_T + stable_T  # map to global frame coordinates

        # plan the core frames for the current emission (global coordinates)
        core_start_g = self.last_emitted_T
        core_end_g = core_start_g + self.chunk_frames
        required_stable_g = core_end_g + self.cnn_redundancy_frames

        if stable_T_global >= required_stable_g or is_last_chunk:
            emit_start_g = max(0, core_start_g - self.cnn_redundancy_frames)
            emit_end_g = core_end_g + self.cnn_redundancy_frames

            # global -> local index
            emit_start = max(0, emit_start_g - self.base_T)
            emit_end = emit_end_g - self.base_T
            emit_start = max(0, min(emit_start, T_full))
            emit_end = max(emit_start, min(emit_end, T_full))

            mel_output = mel_full[:, :, emit_start:emit_end]
            self.last_emitted_T = core_end_g  # only advance the core frame pointer (global)
        else:
            mel_output = mel_full[:, :, 0:0]

        self.total_samples_processed += len(audio_chunk)
        self.is_first = False

        info = {
            "type": "exact_chunk",
            "chunk_number": self.chunk_count,
            "emitted_frames": mel_output.shape[-1],
            "stable_T": stable_T,
            "T_full": T_full,
            "base_T": self.base_T,
            "stable_T_global": stable_T_global,
            "buffer_len_samples": int(self.buffer.shape[0]),
            "left_samples_dropped": self.left_samples_dropped,
            "core_start": core_start_g,  # if keep the original field name, use the global value here
            "core_end": core_end_g,  # same as above
        }
        return mel_output, info

    def flush(self) -> torch.Tensor:
        """Called when the stream ends, output the remaining unemitted frames, ensuring consistency with offline (calculated by global coordinates)."""
        if len(self.buffer) == 0:
            return torch.zeros(1, 80, 0)

        mel_full = self._extract_full()
        T_local = mel_full.shape[-1]
        T_global = self.base_T + T_local

        if self.last_emitted_T < T_global:
            start_l = max(0, self.last_emitted_T - self.base_T)
            tail = mel_full[:, :, start_l:]
            self.last_emitted_T = T_global
            return tail
        return mel_full[:, :, 0:0]

    def get_config(self) -> Dict:
        return {
            "chunk_ms": self.chunk_ms,
            "first_chunk_ms": self.first_chunk_ms,
            "effective_first_chunk_ms": self.first_chunk_samples / self.sample_rate * 1000.0,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "cnn_redundancy_ms": self.cnn_redundancy_ms,
            "cnn_redundancy_frames": self.cnn_redundancy_frames,
            "enable_sliding_window": self.enable_sliding_window,
            "trigger_seconds": self.trigger_seconds,
            "slide_seconds": self.slide_seconds,
        }

    def get_state(self) -> Dict:
        return {
            "chunk_count": self.chunk_count,
            "last_emitted_T": self.last_emitted_T,
            "total_samples_processed": self.total_samples_processed,
            "buffer_len": int(self.buffer.shape[0]),
            "base_T": self.base_T,
            "left_samples_dropped": self.left_samples_dropped,
        }

    def get_snapshot(self) -> Dict:
        """Get a complete state snapshot (including buffer), used for recovery from a fast start.

        Returns:
            A dictionary containing the complete state, which can be used to restore the snapshot
        """
        buffer_copy = self.buffer.copy()
        snapshot = {
            "chunk_count": self.chunk_count,
            "last_emitted_T": self.last_emitted_T,
            "total_samples_processed": self.total_samples_processed,
            "buffer": buffer_copy,
            "base_T": self.base_T,
            "left_samples_dropped": self.left_samples_dropped,
            "is_first": self.is_first,
            # save the state of the feature_extractor (key: ensure determinism of mel feature extraction)
            "fe_dynamic_log_norm": getattr(self.feature_extractor, "dynamic_log_norm", None),
            "fe_dynamic_range_db": getattr(self.feature_extractor, "dynamic_range_db", None),
            "fe_log_floor_db": getattr(self.feature_extractor, "log_floor_db", None),
        }

        return snapshot

    def restore_snapshot(self, snapshot: Dict) -> None:
        """Restore state from a snapshot

        Args:
            snapshot: the snapshot dictionary returned by get_snapshot
        """
        # record the state before restoration
        {
            "chunk_count": self.chunk_count,
            "last_emitted_T": self.last_emitted_T,
            "buffer_len": len(self.buffer),
        }

        # restore state
        self.chunk_count = snapshot["chunk_count"]
        self.last_emitted_T = snapshot["last_emitted_T"]
        self.total_samples_processed = snapshot["total_samples_processed"]
        self.buffer = snapshot["buffer"].copy()  # copy buffer
        self.base_T = snapshot["base_T"]
        self.left_samples_dropped = snapshot["left_samples_dropped"]
        self.is_first = snapshot["is_first"]

        # restore the state of the feature_extractor (key: ensure determinism of mel feature extraction)
        if snapshot.get("fe_dynamic_log_norm") is not None:
            self.feature_extractor.dynamic_log_norm = snapshot["fe_dynamic_log_norm"]
        if snapshot.get("fe_dynamic_range_db") is not None:
            self.feature_extractor.dynamic_range_db = snapshot["fe_dynamic_range_db"]
        if snapshot.get("fe_log_floor_db") is not None:
            self.feature_extractor.log_floor_db = snapshot["fe_log_floor_db"]


class MiniCPMOProcessor(ProcessorMixin):
    attributes = ["image_processor", "audio_processor", "tokenizer"]
    audio_processor_class = "AutoFeatureExtractor"
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, audio_processor=None, tokenizer=None, **kwargs):
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.chat_template = getattr(tokenizer, "chat_template", None)

        self.version = image_processor.version if image_processor else None
        # audio feature pooling step, needs to be consistent with config.audio_pool_step
        self.pool_step = kwargs.get("audio_pool_step", 5)

        # initialize the streaming audio processor
        self._streaming_mel_processor = None
        if audio_processor is not None:
            self._init_streaming_processor()

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        from transformers.utils import cached_file

        from ....utils.hf import load_hf_tokenizer
        from .configuration_minicpmo import MiniCPMOConfig

        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        config = kwargs.pop("config", None)
        tokenizer = kwargs.pop("tokenizer", None)
        config = config or MiniCPMOConfig.from_pretrained(path, **kwargs)
        options = {
            key: value
            for key, value in kwargs.items()
            if key
            in (
                "cache_dir",
                "force_download",
                "local_files_only",
                "token",
                "revision",
                "subfolder",
            )
        }
        filename = cached_file(path, "preprocessor_config.json", **options)
        import json

        with open(filename, encoding="utf-8") as handle:
            values = json.load(handle)
        blocked = {"auto_map", "processor_class", "image_processor_type", "feature_extractor_type"}
        image_values = values.get("image_processor", values)
        image_values = {key: value for key, value in image_values.items() if key not in blocked}
        audio_values = values.get("audio_processor", {})
        audio_values = {key: value for key, value in audio_values.items() if key not in blocked}
        image = MiniCPMVImageProcessor(**image_values)
        audio = MiniCPMAAudioProcessor(**audio_values)
        if tokenizer is None:
            tokenizer = load_hf_tokenizer(path, config=config, model_config=config, trust_remote_code=False, **options)
        if config.version == "2.6":
            from .processing_minicpmo_26 import MiniCPMOProcessor26

            cls = MiniCPMOProcessor26
        return cls(image, audio, tokenizer, audio_pool_step=config.audio_pool_step)

    def save_pretrained(self, save_directory, **kwargs):
        import json
        from pathlib import Path

        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)

        def plain(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value

        values = {
            "processor_class": type(self).__name__,
            "image_processor": plain(self.image_processor.to_dict()),
            "audio_processor": plain(self.audio_processor.to_dict()),
            "audio_pool_step": self.pool_step,
        }
        filename = directory / "preprocessor_config.json"
        filename.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
        self.tokenizer.save_pretrained(directory)
        return [str(filename)]

    def get_audio_placeholder(
        self,
        audio_lens: int,
        chunk_input: bool = True,
        chunk_length: int = 1,
    ) -> str:
        """
        Public method to get audio placeholder string for vLLM integration.

        Args:
            audio_lens: Length of audio in samples
            chunk_input: Whether to use chunked processing
            chunk_length: Chunk length in seconds

        Returns:
            Audio placeholder string
        """
        pool_step = self.pool_step
        feature_lens = math.ceil(audio_lens / self.audio_processor.hop_length)

        feature_lens = (feature_lens - 1) // 2 + 1
        output_lens = (feature_lens - pool_step) // pool_step + 1

        if chunk_input:
            fbank_feat_in_chunk = int(chunk_length * 100)
            cnn_feat_in_chunk = (fbank_feat_in_chunk - 1) // 2 + 1
            audio_embeds_in_chunk = (cnn_feat_in_chunk - pool_step) // pool_step + 1
            num_audio_chunks = (output_lens + audio_embeds_in_chunk - 1) // audio_embeds_in_chunk

            place_holders = ""
            total_unk_len = 0
            for _ in range(num_audio_chunks):
                unk_len = min(audio_embeds_in_chunk, output_lens - total_unk_len)
                place_holders += self.tokenizer.audio_start + "<unk>" * unk_len + self.tokenizer.audio_end
                total_unk_len += unk_len
            audio_placeholder = place_holders
        else:
            audio_placeholder = self.tokenizer.audio_start + "<unk>" * output_lens + self.tokenizer.audio_end

        return audio_placeholder

    def _init_streaming_processor(
        self,
        chunk_ms: int = 100,
        cnn_redundancy_ms: int = 0,
        *,
        mode: str = "exact",
        first_chunk_ms: Optional[int] = None,
        enable_sliding_window: bool = False,
        slide_trigger_seconds: float = 30.0,
        slide_stride_seconds: float = 10.0,
    ):
        """Initialize the streaming processor

        Args:
            chunk_ms: Chunk size in milliseconds, also the sliding step.
            cnn_redundancy_ms: CNN boundary redundancy in milliseconds (before and after), 0 means standard mode.
            mode: streaming processing mode, currently only supports "exact"
            first_chunk_ms: the size of the first chunk (milliseconds), if not specified, it is the same as chunk_ms
            enable_sliding_window: whether to enable sliding window (trigger mode)
            slide_trigger_seconds: trigger threshold for sliding window in seconds
            slide_stride_seconds: stride for sliding window in seconds
        """
        if mode == "exact":
            self._streaming_mel_processor = StreamingMelProcessorExact(
                feature_extractor=self.audio_processor,
                chunk_ms=chunk_ms,
                first_chunk_ms=first_chunk_ms,
                sample_rate=16000,
                cnn_redundancy_ms=cnn_redundancy_ms,
                enable_sliding_window=enable_sliding_window,
                slide_trigger_seconds=slide_trigger_seconds,
                slide_stride_seconds=slide_stride_seconds,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}, only 'exact' is supported")
        self._streaming_mode = mode if mode in ["exact"] else ("exact")

    def set_streaming_mode(
        self,
        mode: str = "exact",
        chunk_ms: int = 100,
        cnn_redundancy_ms: int = 0,
        *,
        first_chunk_ms: Optional[int] = None,
        enable_sliding_window: bool = False,
        slide_trigger_seconds: float = 30.0,
        slide_stride_seconds: float = 10.0,
    ):
        """Set streaming processing mode

        Args:
            mode: streaming processing mode, currently only supports "exact"
            chunk_ms: chunk size in milliseconds, also the sliding step.
            cnn_redundancy_ms: CNN boundary redundancy in milliseconds (before and after), 0 means standard mode.
            first_chunk_ms: the size of the first chunk (milliseconds), if not specified, it is the same as chunk_ms
            enable_sliding_window: whether to enable sliding window (trigger mode)
            slide_trigger_seconds: trigger threshold for sliding window in seconds
            slide_stride_seconds: stride for sliding window in seconds
        """
        if self.audio_processor is None:
            raise ValueError("audio_processor is not set, cannot initialize the streaming processor")
        self._init_streaming_processor(
            chunk_ms=chunk_ms,
            cnn_redundancy_ms=cnn_redundancy_ms,
            mode=mode,
            first_chunk_ms=first_chunk_ms,
            enable_sliding_window=enable_sliding_window,
            slide_trigger_seconds=slide_trigger_seconds,
            slide_stride_seconds=slide_stride_seconds,
        )

    def process_image(
        self,
        images: Optional[ImageInput] = None,
        do_pad: bool = True,
        max_slice_nums: int = 1,
        return_tensors: str = "pt",
    ) -> MiniCPMOBatchFeature:
        """Process image data

        Args:
            images: input images
            do_pad: whether to pad
            max_slice_nums: maximum number of slices
            return_tensors: return tensor type
        Returns:
            MiniCPMOBatchFeature object
        """
        if images is None:
            return MiniCPMOBatchFeature(data={"pixel_values": [[]], "image_sizes": [[]], "tgt_sizes": [[]]})

        result = self.image_processor(
            images, do_pad=do_pad, max_slice_nums=max_slice_nums, return_tensors=return_tensors
        )

        model_inputs = {
            "pixel_values": result.get("pixel_values", [[]]),
            "image_sizes": result.get("image_sizes", [[]]),
            "tgt_sizes": result.get("tgt_sizes", [[]]),
        }

        return MiniCPMOBatchFeature(data=model_inputs)

    def process_audio(
        self,
        audios: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        sampling_rate: int = 16000,
        regroup_to_seconds: Optional[int] = None,
        fps: int = 100,
    ) -> MiniCPMOBatchFeature:
        """Process audio data in batch

        Args:
            audios: audio data
            sampling_rate: sampling rate
            regroup_to_seconds: regroup duration in seconds
            fps: frames per second
        Returns:
            MiniCPMOBatchFeature object
        """
        if audios is None:
            return MiniCPMOBatchFeature(data={"audio_features": [], "audio_feature_lens": []})

        audio_features, audio_feature_lens = process_audio_batch(
            audios=audios,
            feature_extractor=self.audio_processor,
            sampling_rate=sampling_rate,
            max_duration_seconds=30,
            return_attention_mask=True,
        )

        if regroup_to_seconds is not None and len(audio_features) > 0:
            audio_features, audio_feature_lens = regroup_audio_features(
                audio_features=audio_features,
                audio_feature_lens=audio_feature_lens,
                regroup_seconds=regroup_to_seconds,
                fps=fps,
            )

        model_inputs = {"audio_features": audio_features, "audio_feature_lens": audio_feature_lens}

        return MiniCPMOBatchFeature(data=model_inputs)

    def process_audio_streaming(
        self,
        audio_chunk: np.ndarray,
        reset: bool = False,
        return_batch_feature: bool = False,
        is_last_chunk: bool = False,
    ) -> Union[Tuple[torch.Tensor, dict], MiniCPMOBatchFeature]:
        """Process audio chunk in streaming

        Args:
            audio_chunk: audio data chunk (any audio, e.g. first process 125ms, then process 100ms)
            reset: whether to reset the processor state
            return_batch_feature: whether to return MiniCPMOBatchFeature format (consistent with process_audio)
        Returns:
            If return_batch_feature=False:
                (audio_features, info)
                - audio_features: [1, 80, n_frames] mel features
                - info: processing information dictionary
            If return_batch_feature=True:
                MiniCPMOBatchFeature object, containing:
                - audio_features: [1, 80, n_frames] mel features
                - audio_feature_lens: [tensor([n_frames])]
                - info: processing information (as an extra attribute)
        """
        if self._streaming_mel_processor is None:
            raise ValueError("Streaming processor not initialized, please ensure audio_processor is set")

        if reset:
            self._streaming_mel_processor.reset()

        # process chunk
        mel_features, info = self._streaming_mel_processor.process(audio_chunk, is_last_chunk=is_last_chunk)

        # determine the return format based on the parameters
        if return_batch_feature:
            # return the format consistent with process_audio
            # note: info returns emitted_frames, which represents the actual output frames
            n_frames = info.get("emitted_frames", mel_features.shape[-1])
            model_inputs = {
                "audio_features": mel_features,
                "audio_feature_lens": [torch.tensor([n_frames])],
                "streaming_info": info,  # add streaming processing information
            }
            return MiniCPMOBatchFeature(data=model_inputs)
        else:
            return mel_features, info

    def reset_streaming(self):
        if self._streaming_mel_processor is not None:
            self._streaming_mel_processor.reset()

    def get_streaming_chunk_size(self) -> int:
        if self._streaming_mel_processor is None:
            raise ValueError("Streaming processor not initialized")
        return self._streaming_mel_processor.get_chunk_size()

    def configure_streaming(
        self,
        chunk_ms: int = 100,
        enable_sliding_window: bool = False,
        slide_trigger_seconds: float = 30.0,
        slide_stride_seconds: float = 10.0,
    ):
        """Configure streaming processor parameters

        Args:
            chunk_ms: chunk size in milliseconds
            enable_sliding_window: whether to enable sliding window (trigger mode)
            slide_trigger_seconds: trigger threshold for sliding window in seconds
            slide_stride_seconds: stride for sliding window in seconds
        """
        if self.audio_processor is None:
            raise ValueError("audio_processor is not set")

        self._init_streaming_processor(
            chunk_ms=chunk_ms,
            enable_sliding_window=enable_sliding_window,
            slide_trigger_seconds=slide_trigger_seconds,
            slide_stride_seconds=slide_stride_seconds,
        )

    def get_streaming_config(self) -> dict:
        if self._streaming_mel_processor is None:
            return {}
        return self._streaming_mel_processor.get_config()

    def get_streaming_state(self) -> dict:
        if self._streaming_mel_processor is None:
            return {}
        return self._streaming_mel_processor.get_state()

    def get_streaming_snapshot(self) -> dict:
        if self._streaming_mel_processor is None:
            return {}
        return self._streaming_mel_processor.get_snapshot()

    def restore_streaming_snapshot(self, snapshot: dict) -> None:
        if self._streaming_mel_processor is None:
            return
        if not snapshot:
            return
        self._streaming_mel_processor.restore_snapshot(snapshot)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: ImageInput = None,
        audios: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]] = None,
        audio_parts: Optional[list] = None,
        max_length: Optional[int] = None,
        do_pad: Optional[bool] = True,
        max_slice_nums: int = None,
        use_image_id: bool = True,
        stream_input: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        sampling_rate: Optional[int] = 16000,
        online_streaming: bool = False,
        audio_chunk_idx: int = 0,
        is_last_chunk: bool = False,
        **kwargs,
    ) -> MiniCPMOBatchFeature:
        if images is not None:
            image_inputs = self.process_image(
                images=images, do_pad=do_pad, max_slice_nums=max_slice_nums, return_tensors=return_tensors
            )
        else:
            image_inputs = None

        audio_features, audio_feature_lens, audio_phs = self.audio_feature_extract(
            audios,
            audio_parts,
            stream_input,
            sampling_rate,
            online_streaming=online_streaming,
            is_last_chunk=is_last_chunk,
        )

        model_inputs = self._convert_omni_to_inputs(
            image_inputs,
            audio_phs,
            text,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            max_length=max_length,
            **kwargs,
        )

        model_inputs["audio_features"] = audio_features
        model_inputs["audio_feature_lens"] = audio_feature_lens

        result = MiniCPMOBatchFeature(data={**model_inputs})

        if online_streaming:
            result.use_extra_context = True
            result.prefix_extra_frames = 0 if audio_chunk_idx == 0 else 2
            result.suffix_extra_frames = 2
            result.chunk_idx = audio_chunk_idx

        return result

    def audio_feature_extract(
        self,
        audios: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]], None] = None,
        audio_parts: Optional[list] = None,
        stream_input: Optional[bool] = False,
        sampling_rate: Optional[int] = None,
        chunk_length: Optional[int] = 1,
        online_streaming: bool = False,
        is_last_chunk: bool = False,
        **kwargs,
    ):
        if audios is None:
            return [], [], []

        if isinstance(audios, np.ndarray):
            audios_list = [[audios]]
        elif isinstance(audios[0], np.ndarray):
            audios_list = [audios]
        else:
            audios_list = audios

        if audio_parts is not None:
            assert len(audio_parts) == len(audios_list)
            for parts, audios in zip(audio_parts, audios_list):
                assert len(parts) == len(audios)

        audio_feature_lens_list = []
        audio_ph_list = []
        audio_features_all = []

        # audio placeholder not dependent on audio_parts
        for audios in audios_list:
            if audios:
                audio_ph_list.append(
                    [
                        self.get_audio_placeholder(len(a), chunk_input=stream_input, chunk_length=chunk_length)
                        for a in audios
                    ]
                )
            else:
                audio_ph_list.append([])

        for idx, audios in enumerate(audios_list):
            if audio_parts is not None:
                # same audio part merge
                audio_part = audio_parts[idx]
                merge_audio = []
                cur_audio = []
                for aid, (part, audio) in enumerate(zip(audio_part, audios)):
                    if aid == 0 or audio_part[aid] == audio_part[aid - 1]:
                        cur_audio.append(audio)
                    else:
                        merge_audio.append(np.hstack(cur_audio))
                        cur_audio = [audio]
                if cur_audio:
                    merge_audio.append(np.hstack(cur_audio))
            else:
                merge_audio = audios

            # If the audio exceeds 30 seconds, split it into chunks every 30 seconds.
            final_merge_audio = []
            max_audio_inp_len = 30 * sampling_rate
            for audio in merge_audio:
                if len(audio) <= max_audio_inp_len:
                    final_merge_audio.append(audio)
                else:
                    for i in range(math.ceil(len(audio) / max_audio_inp_len)):
                        final_merge_audio.append(audio[i * max_audio_inp_len : (i + 1) * max_audio_inp_len])

            audio_feature_lens = []

            if audios:
                if online_streaming:
                    # online streaming: only support single audio, directly use process_audio_streaming return format
                    assert len(final_merge_audio) == 1, (
                        f"online streaming mode only supports single audio, currently there are {len(final_merge_audio)}"
                    )
                    audio = final_merge_audio[0]
                    result = self.process_audio_streaming(
                        audio, reset=False, return_batch_feature=True, is_last_chunk=is_last_chunk
                    )
                    audio_features_all.append(
                        result["audio_features"].squeeze(0)
                    )  # [1, 80, T] -> [80, T], keep consistent with batch processing
                    audio_feature_lens_list.append(result["audio_feature_lens"][0])
                else:
                    # batch processing
                    audio_inputs = self.audio_processor(
                        final_merge_audio,
                        sampling_rate=sampling_rate,
                        return_attention_mask=True,
                        padding="max_length",
                        return_tensors="pt",
                        **kwargs,
                    )
                    audio_feature = audio_inputs["input_features"]
                    actual_lens = audio_inputs["attention_mask"].sum(dim=1)

                    for feat, lens in zip(audio_feature, actual_lens):
                        audio_features_all.append(feat[:, :lens])
                        audio_feature_lens.append(lens)

                    audio_feature_lens = torch.hstack(audio_feature_lens)
                    audio_feature_lens_list.append(audio_feature_lens)
            else:
                audio_feature_lens_list.append([])

        if audio_features_all:
            audio_features = [i.permute(1, 0) for i in audio_features_all]
            audio_features = torch.nn.utils.rnn.pad_sequence(
                audio_features, batch_first=True, padding_value=0.0
            ).permute(0, 2, 1)
        else:
            audio_features = []

        return audio_features, audio_feature_lens_list, audio_ph_list

    def _convert(self, input_str, max_inp_length: Optional[int] = None):
        old_input_ids = self.tokenizer.encode(input_str)

        listen_token_id = self.tokenizer.convert_tokens_to_ids("<|listen|>")
        input_ids = []
        for token in old_input_ids:
            if token != listen_token_id:
                input_ids.append(token)

        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        ## image bound
        start_cond = (input_ids == self.tokenizer.im_start_id) | (input_ids == self.tokenizer.slice_start_id)
        end_cond = (input_ids == self.tokenizer.im_end_id) | (input_ids == self.tokenizer.slice_end_id)

        image_start_idx = torch.where(start_cond)[0]
        image_start_idx += 1
        image_end_idx = torch.where(end_cond)[0]

        valid_image_nums = max(len(image_start_idx), len(image_end_idx))

        image_bounds = torch.hstack(
            [
                image_start_idx[:valid_image_nums].unsqueeze(-1),
                image_end_idx[:valid_image_nums].unsqueeze(-1),
            ]
        )

        ##  audio bound
        audio_start_idx = torch.where(input_ids == self.tokenizer.audio_start_id)[0]
        audio_end_idx = torch.where(input_ids == self.tokenizer.audio_end_id)[0]
        assert len(audio_start_idx) == len(audio_end_idx)
        audio_bounds = torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])

        spk_start_idx = torch.where(input_ids == self.tokenizer.spk_start_id)[0]
        spk_end_idx = torch.where(input_ids == self.tokenizer.spk_end_id)[0]
        assert len(spk_start_idx) == len(spk_end_idx)
        spk_bounds = torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])

        return input_ids, image_bounds, audio_bounds, spk_bounds

    def _convert_omni_to_inputs(
        self,
        images,
        audio_phs,
        texts: Union[str, List[str]],
        truncation=None,
        max_length=None,
        max_slice_nums=None,
        use_image_id=None,
        return_tensors=None,
        **kwargs,
    ):
        if images is None and audio_phs is None:
            model_inputs = self.tokenizer(
                texts, return_tensors=return_tensors, truncation=truncation, max_length=max_length, **kwargs
            )
            return MiniCPMOBatchFeature(data={**model_inputs})

        image_pattern = "<image>./</image>"
        audio_pattern = "<audio>./</audio>"
        split_pattern = f"({image_pattern}|{audio_pattern})"

        if isinstance(texts, str):
            texts = [texts]

        bs = len(texts)
        if images is not None:
            images, image_sizes, tgt_sizes = images["pixel_values"], images["image_sizes"], images["tgt_sizes"]
        else:
            images, image_sizes, tgt_sizes = [[]] * bs, [[]] * bs, [[]] * bs

        input_ids_list = []
        image_bounds_list = []
        audio_bounds_list = []
        spk_bounds_list = []

        for index, text in enumerate(texts):
            text_chunks = re.split(split_pattern, text)

            image_tags = re.findall(image_pattern, text)
            audio_tags = re.findall(audio_pattern, text)

            if image_tags:
                assert images is not None
                assert len(image_tags) == len(image_sizes[index])
            if audio_tags:
                assert audio_phs is not None
                assert len(audio_tags) == len(audio_phs[index])

            image_id = 0
            audio_id = 0
            for i, chunk in enumerate(text_chunks):
                if chunk == image_pattern:
                    image_placeholder = self.image_processor.get_slice_image_placeholder(
                        image_sizes[index][image_id], image_id, max_slice_nums, use_image_id
                    )
                    image_id += 1
                    text_chunks[i] = image_placeholder
                elif chunk == audio_pattern:
                    audio_placeholder = audio_phs[index][audio_id]
                    audio_id += 1
                    text_chunks[i] = audio_placeholder

            final_text = "".join(text_chunks)
            input_ids, image_bounds, audio_bounds, spk_bounds = self._convert(final_text, max_length)

            input_ids_list.append(input_ids)
            image_bounds_list.append(image_bounds)
            audio_bounds_list.append(audio_bounds)
            spk_bounds_list.append(spk_bounds)

        padded_input_ids, padding_lengths = self.pad(input_ids_list, padding_side="left")
        attention_mask = torch.ones_like(padded_input_ids, dtype=torch.bool)
        for i, length in enumerate(padding_lengths):
            image_bounds_list[i] = image_bounds_list[i] + length
            audio_bounds_list[i] = audio_bounds_list[i] + length
            spk_bounds_list[i] = spk_bounds_list[i] + length
            attention_mask[i, :length] = False

        data = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "pixel_values": images,
            "image_sizes": image_sizes,
            "image_bound": image_bounds_list,
            "tgt_sizes": tgt_sizes,
            "audio_bounds": audio_bounds_list,
            "spk_bounds": spk_bounds_list,
        }

        return data

    def pad(self, inputs, max_length=None, padding_value=0, padding_side="left"):
        items = []
        if isinstance(inputs[0], list):
            assert isinstance(inputs[0][0], torch.Tensor)
            for it in inputs:
                for tr in it:
                    items.append(tr)
        else:
            assert isinstance(inputs[0], torch.Tensor)
            items = inputs

        batch_size = len(items)
        shape = items[0].shape
        dim = len(shape)
        assert dim <= 2
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(item.shape[-1] for item in items))
        min_length = min(item.shape[-1] for item in items)
        dtype = items[0].dtype

        if dim == 0:
            return torch.stack(list(items), dim=0), [0]
        elif dim == 1:
            if max_length == min_length:
                return torch.stack(list(items), dim=0), [0] * batch_size
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        else:
            tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

        padding_length = []
        for i, item in enumerate(items):
            if dim == 1:
                if padding_side == "left":
                    tensor[i, -len(item) :] = item.clone()
                else:
                    tensor[i, : len(item)] = item.clone()
            elif dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item) :, :] = item.clone()
                else:
                    tensor[i, : len(item), :] = item.clone()
            padding_length.append(tensor.shape[-1] - len(item))

        return tensor, padding_length
